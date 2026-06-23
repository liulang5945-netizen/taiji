"""
态极 Autodl 预训练脚本 v2
==========================
修复数据格式问题，使用态极正确的 [系统]/[用户]/[助手] 格式

用法:
  python taiji/train/autodl_pretrain_v2.py --size 350m --data taiji_data/training_data/pretrain_final.jsonl
"""
import os
import sys
import json
import time
import math
import logging
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("AutodlPretrain")


def get_args():
    parser = argparse.ArgumentParser(description="态极 Autodl 预训练 v2")
    parser.add_argument("--size", type=str, default="350m", choices=["125m", "350m", "1b", "3b", "7b"])
    parser.add_argument("--data", type=str, required=True, help="训练数据路径")
    parser.add_argument("--output_dir", type=str, default="taiji_data/autodl_checkpoints")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--target_loss", type=float, default=1.5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)
    return parser.parse_args()


def format_conversation(messages):
    """
    将 messages 格式转换为态极训练格式

    输入: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
    输出: "[系统] ...\n[用户] ...\n[助手] ..."
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            parts.append(f"[系统] {content}")
        elif role == "user":
            parts.append(f"[用户] {content}")
        elif role == "assistant":
            parts.append(f"[助手] {content}")

    return "\n".join(parts)


def format_react(task, steps):
    """
    将 ReAct 格式转换为态极训练格式

    输入: {"task": "...", "steps": [{"thought": "...", "action": "...", ...}, ...]}
    输出: "[系统] ...\n[用户] {task}\n[助手] <think>{thought}</think><tool_call>{action} {args}\n<tool_result>{result}</tool_result>\n..."
    """
    parts = [
        "[系统] 你是态极AI助手，可以使用工具完成任务。",
        f"[用户] {task}",
        "[助手] "
    ]

    for step in steps:
        thought = step.get("thought", "")
        action = step.get("action")
        action_args = step.get("action_args", {})
        final_answer = step.get("final_answer", "")
        observation = step.get("observation", "")

        if thought:
            parts[2] += f"<think>{thought}</think>"

        if action:
            args_str = json.dumps(action_args, ensure_ascii=False)
            parts[2] += f"<tool_call>{action} {args_str}"
            if observation:
                parts[2] += f"\n<tool_result>{observation}</tool_result>\n"

        if final_answer:
            parts[2] += f"<final_answer>{final_answer}</final_answer>"

    return "\n".join(parts)


class TaijiDataset(torch.utils.data.Dataset):
    """态极训练数据集 - 使用正确的格式"""

    def __init__(self, data_file, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        logger.info(f"加载数据: {data_file}")
        count = 0
        skipped = 0

        with open(data_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)

                    # 处理 messages 格式（对话）
                    if "messages" in item:
                        text = format_conversation(item["messages"])

                    # 处理 ReAct 格式（工具调用）
                    elif "task" in item and "steps" in item:
                        text = format_react(item["task"], item["steps"])

                    # 处理纯文本格式
                    elif "text" in item:
                        text = item["text"]

                    else:
                        skipped += 1
                        continue

                    # 过滤太短的样本
                    if len(text) < 50:
                        skipped += 1
                        continue

                    self.samples.append(text)
                    count += 1

                except json.JSONDecodeError:
                    skipped += 1
                    continue

        logger.info(f"加载完成: {count} 条样本, 跳过 {skipped} 条无效数据")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encoded = self.tokenizer.encode(text)

        # 截断到最大长度
        if len(encoded) > self.max_len:
            encoded = encoded[:self.max_len]

        input_ids = torch.tensor(encoded, dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids}


def collate_fn(batch, pad_id=0, max_len=512):
    """动态padding的collate函数"""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    max_len_batch = min(max(len(ids) for ids in input_ids), max_len)

    padded_input = torch.full((len(batch), max_len_batch), pad_id, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len_batch), -100, dtype=torch.long)

    for i, (ids, labs) in enumerate(zip(input_ids, labels)):
        length = min(len(ids), max_len_batch)
        padded_input[i, :length] = ids[:length]
        padded_labels[i, :length] = labs[:length]

    return {"input_ids": padded_input, "labels": padded_labels}


def setup_model(args):
    """初始化模型"""
    from taiji.config import ModelConfig
    from taiji.architecture import ModelSelf

    preset_map = {
        "125m": ModelConfig.size_125m,
        "350m": ModelConfig.size_350m,
        "1b": ModelConfig.size_1b,
        "3b": ModelConfig.size_3b,
        "7b": ModelConfig.size_7b,
    }
    config = preset_map[args.size]()
    config.max_position_embeddings = args.max_seq_len

    if args.gradient_checkpointing:
        config.use_cache = False

    model = ModelSelf(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数: {total_params/1e6:.1f}M (可训练: {trainable_params/1e6:.1f}M)")

    return model, config


def get_lr(step, max_steps, warmup_steps, max_lr):
    """Cosine warmup 学习率调度"""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train(args):
    """主训练循环"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 初始化 tokenizer
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()

    # 查找 SentencePiece 模型
    sp_candidates = [
        os.path.join("taiji", "tokenizer", "sentencepiece.model"),
        os.path.join("taiji_data", "final", "tokenizer", "sentencepiece.model"),
        os.path.join("model", "tokenizer", "sentencepiece.model"),
    ]

    sp_loaded = False
    for sp_path in sp_candidates:
        if os.path.exists(sp_path):
            sp.Load(sp_path)
            logger.info(f"加载 SentencePiece: {sp_path}")
            sp_loaded = True
            break

    if not sp_loaded:
        logger.error("未找到 SentencePiece 模型文件!")
        return

    class SimpleTokenizer:
        def __init__(self, sp_model):
            self.sp = sp_model
        def encode(self, text):
            return self.sp.EncodeAsIds(text)
        def decode(self, ids):
            return self.sp.DecodeIds(ids)
        @property
        def pad_id(self):
            return self.sp.pad_id() if self.sp.pad_id() >= 0 else 0

    tokenizer = SimpleTokenizer(sp)

    # 加载数据
    if not os.path.exists(args.data):
        logger.error(f"数据文件不存在: {args.data}")
        return

    dataset = TaijiDataset(args.data, tokenizer, args.max_seq_len)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id=0, max_len=args.max_seq_len),
        num_workers=4,
        pin_memory=True,
    )

    # 初始化模型
    model, config = setup_model(args)

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("梯度检查点已启用")

    model = model.to(device)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    # 恢复训练
    start_step = 0
    best_loss = float("inf")
    no_improve = 0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt.get("step", 0)
            best_loss = ckpt.get("best_loss", float("inf"))
        else:
            model.load_state_dict(ckpt)
        logger.info(f"从 checkpoint 恢复: step={start_step}, loss={best_loss:.4f}")

    # 训练循环
    model.train()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = start_step
    epoch = 0
    running_loss = 0.0
    log_loss = 0.0

    logger.info("=" * 60)
    logger.info(f"开始训练 | 模型: {args.size} | 步数: {args.max_steps}")
    logger.info(f"数据: {args.data} | 样本数: {len(dataset)}")
    logger.info(f"Batch: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    logger.info(f"LR: {args.lr} | SeqLen: {args.max_seq_len}")
    logger.info("=" * 60)

    while global_step < args.max_steps:
        epoch += 1
        for batch in dataloader:
            if global_step >= args.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(tokens=input_ids, targets=labels)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]

            loss = loss / args.grad_accum
            loss.backward()

            if (global_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            # 更新学习率
            lr = get_lr(global_step, args.max_steps, args.warmup_steps, args.lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            loss_val = loss.item() * args.grad_accum
            running_loss += loss_val
            log_loss += loss_val
            global_step += 1

            # 日志
            if global_step % args.log_every == 0:
                avg_loss = log_loss / args.log_every
                logger.info(
                    f"Step {global_step}/{args.max_steps} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                    f"Best: {best_loss:.4f}"
                )
                log_loss = 0.0

            # 保存 checkpoint
            if global_step % args.save_every == 0:
                save_path = output_dir / f"checkpoint-{global_step}.pt"
                save_obj = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                    "best_loss": best_loss,
                    "config": config.__dict__,
                }
                torch.save(save_obj, save_path)
                logger.info(f"Checkpoint 已保存: {save_path}")

            # 评估和早停
            if global_step % args.eval_every == 0:
                avg_eval_loss = running_loss / args.eval_every
                running_loss = 0.0

                if avg_eval_loss < best_loss - 0.01:
                    best_loss = avg_eval_loss
                    no_improve = 0
                    best_path = output_dir / "best.pt"
                    save_obj = {
                        "model": model.state_dict(),
                        "config": config.__dict__,
                    }
                    torch.save(save_obj, best_path)
                    logger.info(f"新的最佳模型! Loss: {best_loss:.4f}")
                else:
                    no_improve += 1
                    logger.info(f"未改善 ({no_improve}/{args.patience})")

                if avg_eval_loss <= args.target_loss:
                    logger.info(f"达到目标 loss {args.target_loss}, 训练完成!")
                    break

                if no_improve >= args.patience:
                    logger.info(f"连续 {args.patience} 次未改善, 早停!")
                    break

    # 保存最终模型
    final_path = output_dir / "final.pt"
    save_obj = {
        "model": model.state_dict(),
        "config": config.__dict__,
    }
    torch.save(save_obj, final_path)
    logger.info(f"最终模型已保存: {final_path}")
    logger.info(f"训练完成! 总步数: {global_step}, 最佳Loss: {best_loss:.4f}")


if __name__ == "__main__":
    args = get_args()
    train(args)
