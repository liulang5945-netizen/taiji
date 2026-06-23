"""
态极 1B 微调脚本
================

用指令数据对预训练的 1B 模型进行微调，提升对话和工具调用能力。

用法:
  python taiji/train/finetune_taiji.py

输出:
  - 模型保存到 taiji_data/finetuned/
  - 训练日志保存到 taiji_data/finetuned/train_log.json
"""
import os
import sys
import json
import time
import logging
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("Finetune")


# ======================== 配置 ========================

CONFIG = {
    # 模型
    "model_path": "taiji_data/final",
    "output_dir": "taiji_data/finetuned",

    # 数据（按优先级排序，混合训练）
    "data_files": [
        # 高质量指令数据
        {"path": "taiji_data/training_data/alpaca_zh.jsonl", "weight": 3.0, "name": "alpaca_zh"},
        {"path": "taiji_data/training_data/alpaca_en.jsonl", "weight": 2.0, "name": "alpaca_en"},
        {"path": "taiji_data/training_data/dolly_15k.jsonl", "weight": 2.0, "name": "dolly"},
        # 代码能力
        {"path": "taiji_data/training_data/code_alpaca.jsonl", "weight": 2.0, "name": "code"},
        # ReAct 工具调用
        {"path": "taiji_data/training_data/react_data.jsonl", "weight": 4.0, "name": "react"},
        {"path": "taiji_data/training_data/taiji_react_data.jsonl", "weight": 4.0, "name": "taiji_react"},
        # 对话能力
        {"path": "taiji_data/training_data/taiji_conversation_data.jsonl", "weight": 3.0, "name": "conversation"},
        # 专项能力（错误恢复、记忆使用等）
        {"path": "taiji_data/training_data/gap_error_recovery.jsonl", "weight": 3.0, "name": "error_recovery"},
        {"path": "taiji_data/training_data/gap_memory_usage.jsonl", "weight": 3.0, "name": "memory"},
        {"path": "taiji_data/training_data/gap_complex_tasks.jsonl", "weight": 3.0, "name": "complex"},
    ],

    # 训练参数
    "max_seq_len": 512,
    "batch_size": 2,
    "gradient_accumulation_steps": 8,  # 有效 batch = 2 * 8 = 16
    "learning_rate": 2e-5,              # 微调用小 lr
    "weight_decay": 0.01,
    "warmup_steps": 200,
    "max_steps": 10000,
    "save_every": 500,
    "log_every": 10,
    "eval_ratio": 0.01,                 # 1% 数据用于验证

    # 早停
    "target_loss": 1.0,
    "patience": 10,
}


# ======================== 数据集 ========================

class InstructionDataset(Dataset):
    """指令微调数据集"""

    def __init__(self, samples, tokenizer, max_len=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = self._format(item)

        # 使用 ModelSelfTokenizer 编码
        result = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = result["input_ids"]

        # 去掉 batch 维度
        if input_ids.dim() > 1:
            input_ids = input_ids.squeeze(0)

        # targets = input_ids（因果语言模型）
        targets = input_ids.clone()

        return {
            "tokens": input_ids,
            "targets": targets,
        }

    def _format(self, item):
        """格式化为训练文本"""
        messages = item.get("messages", [])
        if not messages:
            # 兼容 instruction/output 格式
            instruction = item.get("instruction", item.get("input", ""))
            output = item.get("output", item.get("response", ""))
            if instruction and output:
                return f"用户: {instruction}\n助手: {output}"
            return ""

        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"系统: {content}")
            elif role == "user":
                parts.append(f"用户: {content}")
            elif role == "assistant":
                parts.append(f"助手: {content}")
        return "\n".join(parts)


def load_data(data_files, eval_ratio=0.01):
    """加载并混合数据集"""
    all_samples = []

    for entry in data_files:
        path = entry["path"]
        weight = entry.get("weight", 1.0)
        name = entry.get("name", os.path.basename(path))

        if not os.path.exists(path):
            logger.warning(f"跳过不存在的文件: {path}")
            continue

        samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    # 验证数据格式
                    if "messages" in item or "instruction" in item:
                        samples.append(item)
                except json.JSONDecodeError:
                    continue

        # 按权重复制（简单实现：重复添加）
        weighted_count = int(len(samples) * weight)
        sampled = random.choices(samples, k=weighted_count)
        all_samples.extend(sampled)

        logger.info(f"  {name}: {len(samples)} 样本, 权重 {weight}x -> {weighted_count}")

    random.shuffle(all_samples)

    # 分割训练/验证
    eval_count = max(100, int(len(all_samples) * eval_ratio))
    eval_samples = all_samples[:eval_count]
    train_samples = all_samples[eval_count:]

    logger.info(f"总计: {len(train_samples)} 训练 + {len(eval_samples)} 验证")
    return train_samples, eval_samples


# ======================== 训练 ========================

def evaluate(model, dataset, device, max_batches=50):
    """评估模型"""
    model.eval()
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            tokens = batch["tokens"].to(device)
            targets = batch["targets"].to(device)

            output = model(tokens=tokens, targets=targets)
            total_loss += output.loss.item()
            count += 1

    model.train()
    return total_loss / max(count, 1)


def train():
    """主训练函数"""
    logger.info("=" * 60)
    logger.info("态极 1B 微调")
    logger.info("=" * 60)

    # 加载模型
    logger.info(f"加载模型: {CONFIG['model_path']}")
    from taiji.loader import load_model
    model, tokenizer = load_model(CONFIG["model_path"], device="cpu")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.to(device)
        logger.info(f"使用 GPU: {torch.cuda.get_device_name()}")
    else:
        logger.info("使用 CPU（训练会很慢，建议用 GPU）")

    model.train()

    # 加载数据
    logger.info("加载训练数据...")
    train_samples, eval_samples = load_data(CONFIG["data_files"], CONFIG["eval_ratio"])

    train_dataset = InstructionDataset(train_samples, tokenizer, CONFIG["max_seq_len"])
    eval_dataset = InstructionDataset(eval_samples, tokenizer, CONFIG["max_seq_len"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    # 学习率调度
    def lr_lambda(step):
        if step < CONFIG["warmup_steps"]:
            return step / CONFIG["warmup_steps"]
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 输出目录
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 训练日志
    log_file = output_dir / "train_log.json"
    train_log = []

    # 训练循环
    logger.info(f"开始训练: max_steps={CONFIG['max_steps']}, batch={CONFIG['batch_size']}, "
                f"grad_accum={CONFIG['gradient_accumulation_steps']}, lr={CONFIG['learning_rate']}")

    step = 0
    epoch = 0
    best_eval_loss = float("inf")
    patience_counter = 0
    optimizer.zero_grad()

    while step < CONFIG["max_steps"]:
        epoch += 1
        logger.info(f"--- Epoch {epoch} ---")

        for batch in train_loader:
            if step >= CONFIG["max_steps"]:
                break

            tokens = batch["tokens"].to(device)
            targets = batch["targets"].to(device)

            output = model(tokens=tokens, targets=targets)
            loss = output.loss / CONFIG["gradient_accumulation_steps"]
            loss.backward()

            if (step + 1) % CONFIG["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # 日志
            if step % CONFIG["log_every"] == 0:
                lr = optimizer.param_groups[0]["lr"]
                real_loss = loss.item() * CONFIG["gradient_accumulation_steps"]
                logger.info(f"Step {step}/{CONFIG['max_steps']} | loss={real_loss:.4f} | lr={lr:.2e}")
                train_log.append({"step": step, "loss": real_loss, "lr": lr, "time": time.time()})

            # 保存 checkpoint
            if step > 0 and step % CONFIG["save_every"] == 0:
                ckpt_dir = output_dir / f"checkpoint-{step}"
                ckpt_dir.mkdir(exist_ok=True)
                torch.save(model.state_dict(), ckpt_dir / "model.pt")
                tokenizer.save(str(ckpt_dir / "tokenizer"))
                logger.info(f"Saved checkpoint: {ckpt_dir}")

            # 评估
            if step > 0 and step % CONFIG["save_every"] == 0:
                eval_loss = evaluate(model, eval_dataset, device)
                logger.info(f"Eval loss: {eval_loss:.4f}")

                train_log.append({"step": step, "eval_loss": eval_loss, "time": time.time()})

                # 早停检查
                if eval_loss < best_eval_loss - CONFIG["patience"] * 0.005:
                    best_eval_loss = eval_loss
                    patience_counter = 0
                    # 保存最佳模型
                    best_dir = output_dir / "best"
                    best_dir.mkdir(exist_ok=True)
                    torch.save(model.state_dict(), best_dir / "model.pt")
                    tokenizer.save(str(best_dir / "tokenizer"))
                    logger.info(f"New best model: eval_loss={eval_loss:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= CONFIG["patience"]:
                        logger.info(f"Early stopping at step {step}")
                        break

                # 达到目标 loss
                if eval_loss <= CONFIG["target_loss"]:
                    logger.info(f"Reached target loss {CONFIG['target_loss']}")
                    break

            step += 1

        # 保存训练日志
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(train_log, f, ensure_ascii=False, indent=2)

    # 最终保存
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), final_dir / "model.pt")
    tokenizer.save(str(final_dir / "tokenizer"))

    # 复制 config.json
    import shutil
    src_config = Path(CONFIG["model_path"]) / "config.json"
    if src_config.exists():
        shutil.copy(src_config, final_dir / "config.json")

    logger.info("=" * 60)
    logger.info(f"训练完成! 最终模型: {final_dir}")
    logger.info(f"最佳 eval loss: {best_eval_loss:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    train()
