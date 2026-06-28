"""Training datasets and trainer for the Taiji native-v2 stack."""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import time
from contextlib import nullcontext
from typing import Any, Generator, Optional

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from taiji.data.seed_data import get_seed_conversation_data, get_seed_react_data
from taiji.loader import save_model
from taiji.tokenizer import ModelSelfTokenizer

logger = logging.getLogger("ModelSelfTrainer")


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _create_grad_scaler(device: str) -> Any:
    enabled = device.startswith("cuda") and torch.cuda.is_available()
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device="cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _default_observation(action: Optional[str]) -> str:
    if not action:
        return ""
    return f"{action} 执行完成。"


class TextDataset(Dataset):
    """Chunked text dataset for plain language-model pretraining."""

    def __init__(
        self,
        tokenizer: ModelSelfTokenizer,
        texts: list[str],
        max_length: int = 512,
        min_chunk_length: int = 10,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded: list[list[int]] = []

        for text in texts:
            ids = [tokenizer.bos_token_id]
            ids.extend(tokenizer._encode(text))
            ids.append(tokenizer.eos_token_id)

            for start in range(0, len(ids), max_length):
                chunk = ids[start : start + max_length]
                if len(chunk) >= min_chunk_length:
                    self.encoded.append(chunk)

        logger.info("Pretrain dataset created: %s chunks", len(self.encoded))

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.encoded[idx]
        pad_len = self.max_length - len(ids)
        if pad_len > 0:
            ids = ids + [self.tokenizer.pad_token_id] * pad_len
        else:
            ids = ids[: self.max_length]

        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "tool_target": torch.tensor(-100, dtype=torch.long),
        }


class ReActDataset(Dataset):
    """Dataset for tool-using conversation fine-tuning."""

    DEFAULT_TOOL_DESC = """可用工具:
- read_local_file: 读取文件内容。参数 input(str) 为文件路径
- write_file: 创建文件。参数 input(str) 为 "路径 | 内容"
- edit_file: 编辑文件。参数 input(str) 为 "路径 | 旧文本 | 新文本"
- list_directory: 列出目录内容。参数 input(str) 为目录路径
- execute_python: 执行 Python 代码。参数 input(str) 为 Python 代码
- search: 搜索互联网。参数 input(str) 为搜索关键词
- read_webpage: 读取网页内容。参数 input(str) 为 URL
- create_project: 创建项目。参数 input(str) 为 "类型 | 名称"
- analyze_code: 分析代码。参数 input(str) 为文件路径
- install_dependency: 安装依赖。参数 input(str) 为包名
- learn_knowledge: 学习知识。参数 input(str) 为知识内容
- query_knowledge: 查询知识。参数 input(str) 为查询关键词
- run_command: 运行系统命令。参数 input(str) 为命令"""

    def __init__(
        self,
        tokenizer: ModelSelfTokenizer,
        react_data: list[dict[str, Any]],
        conv_data: list[dict[str, Any]],
        max_length: int = 512,
        tool_descriptions: Optional[str] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tool_descriptions = tool_descriptions or self.DEFAULT_TOOL_DESC
        self.samples: list[dict[str, Any]] = []

        for item in react_data:
            self.samples.extend(self._process_react(item))

        for item in conv_data:
            self.samples.extend(self._process_conversation(item))

        logger.info("ReAct dataset created: %s samples", len(self.samples))

    def _ensure_tool_registered(self, tool_name: str) -> int:
        token_id = self.tokenizer.get_tool_id(tool_name)
        if token_id is None:
            token_id = self.tokenizer.register_tool(tool_name)
        return token_id

    def _process_react(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        task = str(item.get("task", "")).strip()
        steps = item.get("steps", []) or []

        modality = item.get("modality")
        media_path = item.get("media_path")
        if modality and media_path:
            task = f"[{str(modality).upper()}: {os.path.basename(str(media_path))}] {task}"

        for index, step in enumerate(steps):
            thought = str(step.get("thought", "")).strip() or "继续推理并完成当前步骤。"
            action = step.get("action")
            action_args = step.get("action_args", {})
            final_answer = str(step.get("final_answer", "")).strip()

            tool_desc = self.tool_descriptions
            if tool_desc.startswith("可用工具:"):
                system_prompt = f"[系统] 你是 Taiji AI 助手。\n{tool_desc}"
            else:
                system_prompt = f"[系统] 你是 Taiji AI 助手。可用工具:\n{tool_desc}"

            input_parts = [
                system_prompt,
                f"[用户] {task}",
                "[助手] ",
            ]

            for previous in steps[:index]:
                prev_action = previous.get("action")
                if not prev_action:
                    continue
                prev_args = _safe_json(previous.get("action_args", {}))
                prev_observation = str(
                    previous.get("observation", _default_observation(str(prev_action)))
                )
                input_parts.append(
                    f"<think>我需要使用 {prev_action} 工具处理这一步。</think>"
                    f"<tool_call>{prev_action} {prev_args}</tool_call>"
                    f"<tool_result>{prev_observation}</tool_result>\n"
                )

            input_text = "".join(input_parts)

            if final_answer:
                target_text = f"<think>{thought}</think><final_answer>{final_answer}"
                tool_target = -100
            elif action:
                tool_token_id = self._ensure_tool_registered(str(action))
                args_text = _safe_json(action_args)
                target_text = (
                    f"<think>{thought}</think>"
                    f"<tool_call>{action} {args_text}</tool_call>"
                )
                tool_target = tool_token_id
            else:
                continue

            samples.append(
                {
                    "input_text": input_text,
                    "target_text": target_text,
                    "tool_target": tool_target,
                }
            )

        return samples

    def _process_conversation(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        messages = item.get("messages", []) or []

        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue

            input_parts: list[str] = []
            for previous in messages[:index]:
                role = previous.get("role")
                content = str(previous.get("content", ""))
                if role == "system":
                    input_parts.append(f"[系统] {content}")
                elif role == "user":
                    input_parts.append(f"[用户] {content}")
                elif role == "assistant":
                    input_parts.append(f"[助手] {content}")

            input_text = "\n".join(input_parts) + "\n[助手] "
            samples.append(
                {
                    "input_text": input_text,
                    "target_text": str(message.get("content", "")),
                    "tool_target": -100,
                }
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        input_ids = [self.tokenizer.bos_token_id]
        input_ids.extend(self.tokenizer._encode(sample["input_text"]))

        target_ids = self.tokenizer._encode(sample["target_text"])
        target_ids.append(self.tokenizer.eos_token_id)

        total_len = len(input_ids) + len(target_ids)
        if total_len > self.max_length:
            max_target_len = min(len(target_ids), max(self.max_length // 2, 32))
            if len(target_ids) > max_target_len:
                target_ids = target_ids[:max_target_len]

            remaining_input = self.max_length - len(target_ids)
            if remaining_input > 0 and len(input_ids) > remaining_input:
                if remaining_input <= 1:
                    input_ids = input_ids[:remaining_input]
                else:
                    prefix_keep = max(remaining_input // 2, 1)
                    suffix_keep = remaining_input - prefix_keep
                    input_ids = input_ids[:prefix_keep] + input_ids[-suffix_keep:]
            elif remaining_input <= 0:
                target_ids = target_ids[: self.max_length]
                input_ids = []

        full_ids = input_ids + target_ids
        labels = [-100] * len(input_ids) + target_ids

        pad_len = self.max_length - len(full_ids)
        if pad_len > 0:
            full_ids.extend([self.tokenizer.pad_token_id] * pad_len)
            labels.extend([-100] * pad_len)
        else:
            full_ids = full_ids[: self.max_length]
            labels = labels[: self.max_length]

        tool_target = int(sample["tool_target"])
        if tool_target >= 0:
            all_tool_ids = sorted(self.tokenizer.get_all_tool_ids().values())
            try:
                tool_index = all_tool_ids.index(tool_target)
            except ValueError:
                tool_index = -100
        else:
            tool_index = -100

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "tool_target": torch.tensor(tool_index, dtype=torch.long),
        }

    def split(self, val_ratio: float = 0.1) -> tuple["ReActDataset", "ReActDataset"]:
        count = len(self.samples)
        split_idx = max(int(count * (1 - val_ratio)), 1)
        if split_idx >= count:
            split_idx = max(count - 1, 1)

        train_ds = copy.copy(self)
        train_ds.samples = self.samples[:split_idx]

        val_ds = copy.copy(self)
        val_ds.samples = self.samples[split_idx:]

        return train_ds, val_ds


class ModelSelfTrainer:
    """Lightweight trainer for Taiji language and tool fine-tuning."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: ModelSelfTokenizer,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler: Optional[LambdaLR] = None

        self.global_step = 0
        self.best_loss = float("inf")
        self.is_paused = False
        self.is_stopped = False
        self.loss_history: list[float] = []

        self.early_stopping_patience = 3
        self.early_stopping_threshold = 1e-3
        self._patience_counter = 0

        self._bottleneck_enabled = True
        self._stagnant_window = 50
        self._stagnant_threshold = 0.01
        self._bottleneck_detected = False

        self._sync_num_tools()

    def _sync_num_tools(self) -> None:
        if hasattr(self.model, "set_num_tools"):
            self.model.set_num_tools(len(self.tokenizer.get_all_tool_ids()))

    def _setup_scheduler(self, total_steps: int) -> None:
        if total_steps <= 1:
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda _: 1.0)
            return

        safe_warmup = min(self.warmup_steps, max(total_steps // 2, 1))

        def lr_lambda(step: int) -> float:
            if safe_warmup > 0 and step < safe_warmup:
                return float(step + 1) / float(safe_warmup)

            progress = (step - safe_warmup) / float(max(total_steps - safe_warmup, 1))
            progress = min(max(progress, 0.0), 1.0)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def pretrain(
        self,
        dataset: TextDataset,
        num_epochs: int = 5,
        batch_size: int = 4,
        save_dir: str = "./taiji_checkpoints/pretrain",
        save_steps: int = 100,
        log_steps: int = 10,
        device: str = "cpu",
    ) -> Generator[tuple[float, str, list[float], dict[str, Any]], None, None]:
        if len(dataset) == 0:
            raise ValueError("Pretrain dataset is empty.")

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        total_updates = max(
            math.ceil(len(dataloader) / self.gradient_accumulation_steps) * num_epochs,
            1,
        )
        self._setup_scheduler(total_updates)
        self.freeze_heads(exclude=["language"])
        self.model.to(device)
        self.model.train()
        os.makedirs(save_dir, exist_ok=True)

        self._patience_counter = 0
        self._bottleneck_detected = False
        loss_history = list(self.loss_history)
        scaler = _create_grad_scaler(device)
        train_start = time.time()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            batch_count = 0
            self.optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(dataloader):
                if self.is_stopped:
                    return

                while self.is_paused and not self.is_stopped:
                    time.sleep(0.1)
                    yield (
                        self.global_step / max(total_updates, 1),
                        "预训练已暂停",
                        loss_history,
                        {"status": "paused", "phase": "pretrain", "step": self.global_step},
                    )

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                with _autocast_context(scaler.is_enabled()):
                    output = self.model(input_ids, targets=labels, tool_head_active=False)
                    loss = output.loss / self.gradient_accumulation_steps

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += float(loss.detach().item()) * self.gradient_accumulation_steps
                batch_count += 1

                should_step = (
                    (batch_idx + 1) % self.gradient_accumulation_steps == 0
                    or batch_idx + 1 == len(dataloader)
                )
                if not should_step:
                    continue

                trainable_params = [
                    parameter
                    for parameter in self.model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                if scaler.is_enabled():
                    scaler.unscale_(self.optimizer)
                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                    if trainable_params
                    else torch.tensor(0.0)
                )

                if scaler.is_enabled():
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                avg_loss = epoch_loss / max(batch_count, 1)
                if self.global_step % max(log_steps, 1) == 0 or batch_idx + 1 == len(dataloader):
                    loss_history.append(avg_loss)
                    self.loss_history = list(loss_history)
                    elapsed = time.time() - train_start
                    eta = (elapsed / self.global_step) * max(total_updates - self.global_step, 0)
                    lr = self.optimizer.param_groups[0]["lr"]
                    yield (
                        self.global_step / max(total_updates, 1),
                        f"预训练 Epoch {epoch + 1}/{num_epochs} | Step {self.global_step} | Loss: {avg_loss:.4f}",
                        loss_history,
                        {
                            "status": "running",
                            "phase": "pretrain",
                            "loss": avg_loss,
                            "lr": lr,
                            "step": self.global_step,
                            "epoch": epoch + 1,
                            "total_epochs": num_epochs,
                            "elapsed": round(elapsed, 1),
                            "eta": round(eta, 1),
                            "grad_norm": round(
                                grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm),
                                4,
                            ),
                        },
                    )

                if save_steps > 0 and self.global_step % save_steps == 0:
                    self._save_checkpoint(save_dir, avg_loss)

            avg_epoch_loss = epoch_loss / max(batch_count, 1)
            if avg_epoch_loss < self.best_loss - self.early_stopping_threshold:
                self.best_loss = avg_epoch_loss
                self._patience_counter = 0
                self._save_checkpoint(save_dir, avg_epoch_loss)
            else:
                self._patience_counter += 1

            if self.early_stopping_patience > 0 and self._patience_counter >= self.early_stopping_patience:
                self._save_final_model(save_dir)
                yield (
                    1.0,
                    f"早停: 连续 {self.early_stopping_patience} 轮 loss 无改善",
                    loss_history,
                    {
                        "status": "early_stopped",
                        "phase": "pretrain",
                        "loss": avg_epoch_loss,
                        "best_loss": self.best_loss,
                    },
                )
                return

        self._save_final_model(save_dir)
        yield (
            1.0,
            f"预训练完成! Loss: {avg_epoch_loss:.4f}",
            loss_history,
            {"status": "completed", "phase": "pretrain", "best_loss": self.best_loss},
        )

    def finetune(
        self,
        dataset: ReActDataset,
        num_epochs: int = 10,
        batch_size: int = 4,
        save_dir: str = "./taiji_checkpoints/finetune",
        save_steps: int = 50,
        log_steps: int = 5,
        device: str = "cpu",
    ) -> Generator[tuple[float, str, list[float], dict[str, Any]], None, None]:
        if len(dataset) == 0:
            raise ValueError("Finetune dataset is empty.")

        self._sync_num_tools()
        if len(dataset) >= 20:
            train_ds, val_ds = dataset.split(val_ratio=0.1)
        else:
            train_ds, val_ds = dataset, None

        dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        total_updates = max(
            math.ceil(len(dataloader) / self.gradient_accumulation_steps) * num_epochs,
            1,
        )

        self._setup_scheduler(total_updates)
        self.unfreeze_all()
        self.model.to(device)
        self.model.train()
        os.makedirs(save_dir, exist_ok=True)

        self._patience_counter = 0
        loss_history = list(self.loss_history)
        scaler = _create_grad_scaler(device)
        train_start = time.time()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            batch_count = 0
            self.optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(dataloader):
                if self.is_stopped:
                    return

                while self.is_paused and not self.is_stopped:
                    time.sleep(0.1)
                    yield (
                        self.global_step / max(total_updates, 1),
                        "微调已暂停",
                        loss_history,
                        {"status": "paused", "phase": "finetune", "step": self.global_step},
                    )

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                tool_targets = batch["tool_target"].to(device)
                has_tool = bool((tool_targets != -100).any().item())

                with _autocast_context(scaler.is_enabled()):
                    output = self.model(
                        input_ids,
                        targets=labels,
                        tool_targets=tool_targets if has_tool else None,
                        tool_head_active=has_tool,
                    )
                    loss = output.loss / self.gradient_accumulation_steps

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += float(loss.detach().item()) * self.gradient_accumulation_steps
                batch_count += 1

                should_step = (
                    (batch_idx + 1) % self.gradient_accumulation_steps == 0
                    or batch_idx + 1 == len(dataloader)
                )
                if not should_step:
                    continue

                trainable_params = [
                    parameter
                    for parameter in self.model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                if scaler.is_enabled():
                    scaler.unscale_(self.optimizer)
                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                    if trainable_params
                    else torch.tensor(0.0)
                )

                if scaler.is_enabled():
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                avg_loss = epoch_loss / max(batch_count, 1)
                if self.global_step % max(log_steps, 1) == 0 or batch_idx + 1 == len(dataloader):
                    loss_history.append(avg_loss)
                    self.loss_history = list(loss_history)
                    elapsed = time.time() - train_start
                    eta = (elapsed / self.global_step) * max(total_updates - self.global_step, 0)
                    lr = self.optimizer.param_groups[0]["lr"]
                    metrics = {
                        "status": "running",
                        "phase": "finetune",
                        "loss": avg_loss,
                        "lr": lr,
                        "step": self.global_step,
                        "epoch": epoch + 1,
                        "total_epochs": num_epochs,
                        "elapsed": round(elapsed, 1),
                        "eta": round(eta, 1),
                        "grad_norm": round(
                            grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm),
                            4,
                        ),
                    }
                    if has_tool:
                        metrics["tool_active"] = True
                    yield (
                        self.global_step / max(total_updates, 1),
                        f"微调 Epoch {epoch + 1}/{num_epochs} | Step {self.global_step} | Loss: {avg_loss:.4f}",
                        loss_history,
                        metrics,
                    )

                    if self._bottleneck_enabled and not self._bottleneck_detected:
                        bottleneck = self._check_bottleneck(loss_history)
                        if bottleneck["bottleneck"]:
                            yield (
                                self.global_step / max(total_updates, 1),
                                f"⚠️ 检测到训练瓶颈: {', '.join(bottleneck['triggers'])}",
                                loss_history,
                                {
                                    "status": "warning",
                                    "phase": "finetune",
                                    "step": self.global_step,
                                    "bottleneck": True,
                                    "bottleneck_triggers": bottleneck["triggers"],
                                    "should_upgrade": True,
                                },
                            )

                if save_steps > 0 and self.global_step % save_steps == 0:
                    self._save_checkpoint(save_dir, avg_loss)

            avg_epoch_loss = epoch_loss / max(batch_count, 1)
            val_loss = None
            if val_ds is not None and len(val_ds) > 0:
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
                val_loss = self._evaluate(val_loader, device)

            monitor_loss = val_loss if val_loss is not None else avg_epoch_loss
            if monitor_loss < self.best_loss - self.early_stopping_threshold:
                self.best_loss = monitor_loss
                self._patience_counter = 0
                self._save_checkpoint(save_dir, monitor_loss)
            else:
                self._patience_counter += 1

            epoch_metrics: dict[str, Any] = {
                "status": "epoch_end",
                "phase": "finetune",
                "loss": avg_epoch_loss,
                "step": self.global_step,
                "epoch": epoch + 1,
                "total_epochs": num_epochs,
            }
            if val_loss is not None:
                epoch_metrics["val_loss"] = round(val_loss, 4)

            yield (
                (epoch + 1) / max(num_epochs, 1),
                f"Epoch {epoch + 1}/{num_epochs} 完成 | Loss: {avg_epoch_loss:.4f}"
                + (f" | Val: {val_loss:.4f}" if val_loss is not None else ""),
                loss_history,
                epoch_metrics,
            )

            if self.early_stopping_patience > 0 and self._patience_counter >= self.early_stopping_patience:
                self._save_final_model(save_dir)
                yield (
                    1.0,
                    f"早停: 连续 {self.early_stopping_patience} 轮 loss 无改善",
                    loss_history,
                    {
                        "status": "early_stopped",
                        "phase": "finetune",
                        "loss": avg_epoch_loss,
                        "val_loss": val_loss,
                        "best_loss": self.best_loss,
                    },
                )
                return

        self._save_final_model(save_dir)
        yield (
            1.0,
            f"微调完成! Loss: {avg_epoch_loss:.4f}",
            loss_history,
            {"status": "completed", "phase": "finetune", "best_loss": self.best_loss},
        )

    def train(
        self,
        dataset: ReActDataset,
        num_epochs: int = 3,
        batch_size: int = 4,
        save_dir: str = "./taiji_checkpoints",
        save_steps: int = 100,
        log_steps: int = 10,
        device: str = "cpu",
    ) -> Generator[tuple[float, str, list[float], dict[str, Any]], None, None]:
        return self.finetune(
            dataset=dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            save_dir=save_dir,
            save_steps=save_steps,
            log_steps=log_steps,
            device=device,
        )

    def _evaluate(self, dataloader: DataLoader, device: str) -> float:
        self.model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                tool_targets = batch["tool_target"].to(device)
                has_tool = bool((tool_targets != -100).any().item())
                output = self.model(
                    input_ids,
                    targets=labels,
                    tool_targets=tool_targets if has_tool else None,
                    tool_head_active=has_tool,
                )
                total_loss += float(output.loss.item())
                count += 1

        self.model.train()
        return total_loss / max(count, 1)

    def _training_state(self) -> dict[str, Any]:
        return {
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "loss_history": self.loss_history[-200:],
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
        }

    def _save_checkpoint(self, save_dir: str, loss: float) -> None:
        checkpoint_dir = os.path.join(save_dir, f"step_{self.global_step}")
        save_model(self.model, self.tokenizer, checkpoint_dir, self._training_state())
        if loss <= self.best_loss:
            self.best_loss = loss
            save_model(self.model, self.tokenizer, os.path.join(save_dir, "best"), self._training_state())

    def _save_final_model(self, save_dir: str) -> None:
        save_model(self.model, self.tokenizer, os.path.join(save_dir, "final"), self._training_state())

    def train_progressive(
        self,
        dataset: ReActDataset,
        save_dir: str,
        num_epochs: int = 5,
        batch_size: int = 4,
        log_steps: int = 5,
        device: str = "cpu",
    ) -> Generator[tuple[float, str, list[float], dict[str, Any]], None, None]:
        yield (
            0.0,
            "🎯 开始渐进式训练",
            list(self.loss_history),
            {"status": "starting", "phase": "progressive"},
        )
        for fraction, desc, history, metrics in self.finetune(
            dataset=dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            save_dir=save_dir,
            save_steps=max(log_steps * 10, 50),
            log_steps=log_steps,
            device=device,
        ):
            yield (fraction, f"[渐进式] {desc}", history, metrics)

    def freeze_heads(self, exclude: Optional[list[str]] = None) -> None:
        exclude_set = set(exclude or [])
        head_map = {
            "tool": "tool_head",
            "perception": "perception_head",
            "memory": "memory_head",
            "plan": "plan_head",
        }

        for head_name, attr_name in head_map.items():
            head = getattr(self.model, attr_name, None)
            if head is None:
                continue
            requires_grad = head_name in exclude_set
            for parameter in head.parameters():
                parameter.requires_grad = requires_grad

    def unfreeze_all(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = True

    def get_head_losses(
        self,
        logits: torch.Tensor,
        targets: Optional[torch.Tensor],
        tool_logits: Optional[torch.Tensor],
        tool_targets: Optional[torch.Tensor],
        tool_head_active: bool,
        perception_logits: Optional[torch.Tensor] = None,
        memory_logits: Optional[torch.Tensor] = None,
        plan_logits: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        del perception_logits, memory_logits, plan_logits
        losses: dict[str, float] = {}

        if targets is not None:
            language_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            losses["language"] = float(language_loss.item())

        if tool_head_active and tool_logits is not None and tool_targets is not None:
            valid_mask = tool_targets != -100
            if valid_mask.any():
                tool_loss = F.cross_entropy(tool_logits[valid_mask], tool_targets[valid_mask])
                losses["tool"] = float(tool_loss.item())

        if losses:
            losses["total"] = float(sum(losses.values()) / len(losses))

        return losses

    def pause(self) -> None:
        self.is_paused = True

    def resume(self) -> None:
        self.is_paused = False

    def stop(self) -> None:
        self.is_stopped = True

    def _check_bottleneck(self, loss_history: list[float]) -> dict[str, Any]:
        if not self._bottleneck_enabled:
            return {"bottleneck": False, "triggers": []}

        triggers: list[str] = []
        window = self._stagnant_window
        if len(loss_history) >= window * 2:
            recent = loss_history[-window:]
            previous = loss_history[-window * 2 : -window]
            recent_range = max(recent) - min(recent)
            if recent_range < self._stagnant_threshold:
                triggers.append("loss_stagnant")

            recent_avg = sum(recent) / len(recent)
            previous_avg = sum(previous) / len(previous)
            if recent_avg >= previous_avg * 0.99:
                triggers.append("loss_not_improving")

        bottleneck = bool(triggers)
        if bottleneck and not self._bottleneck_detected:
            self._bottleneck_detected = True
            logger.warning("Training bottleneck detected: %s", triggers)

        return {
            "bottleneck": bottleneck,
            "triggers": triggers,
            "should_upgrade": bottleneck,
        }


def build_dataset(
    tokenizer: ModelSelfTokenizer,
    extra_react_data: Optional[list[dict[str, Any]]] = None,
    extra_conv_data: Optional[list[dict[str, Any]]] = None,
    max_length: int = 512,
) -> ReActDataset:
    """Build the canonical ReAct fine-tuning dataset."""

    react_data = list(get_seed_react_data())
    conv_data = list(get_seed_conversation_data())

    try:
        from taiji.data.multimodal_data_generator import generate_multimodal_seed_data

        react_data.extend(generate_multimodal_seed_data())
    except Exception:
        logger.debug("Multimodal seed data unavailable; continuing without it.")

    if extra_react_data:
        react_data.extend(extra_react_data)
    if extra_conv_data:
        conv_data.extend(extra_conv_data)

    return ReActDataset(
        tokenizer=tokenizer,
        react_data=react_data,
        conv_data=conv_data,
        max_length=max_length,
    )
