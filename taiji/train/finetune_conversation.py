"""
态极对话 Fine-tuning 脚本
========================

使用对话数据对预训练模型进行 fine-tuning，让态极学会对话。

用法:
  python taiji/train/finetune_conversation.py
"""
import os
import sys
import json
import time
import logging
import torch
import torch.nn.functional as F

# 确保项目根目录在 Python 路径中
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("Finetune")

# ======================== 配置 ========================

CONFIG = {
    "model_path": "taiji/evolution_data/upgraded_models/pretrained/best",
    "data_files": [
        "taiji/training_data/taiji_graduation_conversation.jsonl",
        "taiji/training_data/taiji_graduation_v2_conversation.jsonl",
        "taiji/training_data/taiji_graduation_v3_conversation.jsonl",
        "taiji/training_data/taiji_conversation_data.jsonl",
        "taiji/training_data/taiji_ultimate_conversation.jsonl",
    ],
    "output_dir": "taiji/evolution_data/upgraded_models/finetuned",
    "max_seq_len": 512,
    "batch_size": 2,
    "gradient_accumulation_steps": 16,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "max_steps": 50000,
    "save_every": 100,
    "log_every": 10,
    "eval_every": 500,
    "resume_from": None,

    # 早停配置
    "target_loss": 1.5,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 0.005,
}


# ======================== 数据加载 ========================

class ConversationDataset(torch.utils.data.Dataset):
    """对话数据集"""

    def __init__(self, data_files, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        logger.info("加载对话数据...")
        for data_file in data_files:
            if not os.path.exists(data_file):
                logger.warning(f"文件不存在: {data_file}")
                continue

            with open(data_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        if "messages" in item:
                            # 将对话转换为训练格式
                            text = self._format_conversation(item["messages"])
                            if text and len(text) > 50:
                                self.samples.append(text)
                    except json.JSONDecodeError:
                        continue

        logger.info(f"加载完成: {len(self.samples):,} 条对话")

    def _format_conversation(self, messages):
        """将对话格式化为训练文本"""
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")

        return "\n".join(parts)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        ids = self.tokenizer.encode(text)

        # 截断到 max_len
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]

        # 自回归：input = ids[:-1], target = ids[1:]
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long)
        return input_ids, labels


def collate_fn(batch):
    """动态填充到批次内最长序列"""
    input_ids_list, labels_list = zip(*batch)
    max_len = max(len(ids) for ids in input_ids_list)

    padded_inputs = []
    padded_labels = []
    for ids, lbls in zip(input_ids_list, labels_list):
        pad_len = max_len - len(ids)
        padded_inputs.append(torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)]))
        padded_labels.append(torch.cat([lbls, torch.full((pad_len,), -100, dtype=torch.long)]))

    return torch.stack(padded_inputs), torch.stack(padded_labels)


# ======================== 训练主循环 ========================

def main():
    from taiji.config import ModelConfig
    from taiji.architecture import ModelSelf
    from taiji.tokenizer import ModelSelfTokenizer
    from taiji.loader import save_model, load_model

    cfg = CONFIG

    # 加载预训练模型
    logger.info(f"加载预训练模型: {cfg['model_path']}")
    model, tokenizer = load_model(cfg['model_path'])
    num_params = model.get_num_parameters()
    if isinstance(num_params, dict):
        num_params = num_params.get('total', 0)
    logger.info(f"模型参数: {num_params:,}")

    # 创建数据集
    dataset = ConversationDataset(cfg["data_files"], tokenizer, cfg["max_seq_len"])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["max_steps"],
        eta_min=cfg["learning_rate"] * 0.1,
    )

    # 训练状态
    global_step = 0
    best_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    # 创建输出目录
    os.makedirs(cfg["output_dir"], exist_ok=True)

    logger.info(f"开始训练: max_steps={cfg['max_steps']}, batch={cfg['batch_size']}, lr={cfg['learning_rate']}")
    logger.info(f"数据集: {len(dataset):,} 条对话")

    model.train()

    while global_step < cfg["max_steps"]:
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            if global_step >= cfg["max_steps"]:
                break

            # 前向传播
            output = model(input_ids)
            logits = output.logits

            # 计算 loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

            # 梯度累积
            loss = loss / cfg["gradient_accumulation_steps"]
            loss.backward()

            if (batch_idx + 1) % cfg["gradient_accumulation_steps"] == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # 日志
                if global_step % cfg["log_every"] == 0:
                    elapsed = time.time() - start_time
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"Step {global_step:6d}/{cfg['max_steps']} | "
                        f"Loss: {loss.item() * cfg['gradient_accumulation_steps']:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Elapsed: {elapsed/60:.1f}min"
                    )

                # 保存检查点
                if global_step % cfg["save_every"] == 0:
                    save_path = os.path.join(cfg["output_dir"], f"step_{global_step}")
                    save_model(model, tokenizer, save_path)
                    logger.info(f"Checkpoint saved: {save_path}")

                    # 更新最佳模型
                    current_loss = loss.item() * cfg["gradient_accumulation_steps"]
                    if current_loss < best_loss - cfg["early_stopping_min_delta"]:
                        best_loss = current_loss
                        patience_counter = 0
                        best_path = os.path.join(cfg["output_dir"], "best")
                        save_model(model, tokenizer, best_path)
                        logger.info(f"Best model updated: loss={best_loss:.4f}")
                    else:
                        patience_counter += 1

                    # 早停
                    if patience_counter >= cfg["early_stopping_patience"]:
                        logger.info(f"Early stopping: patience exhausted")
                        break

                    # 达到目标 loss
                    if best_loss <= cfg["target_loss"]:
                        logger.info(f"Target loss reached: {best_loss:.4f} <= {cfg['target_loss']}")
                        break

    # 保存最终模型
    final_path = os.path.join(cfg["output_dir"], "final")
    save_model(model, tokenizer, final_path)
    logger.info(f"Final model saved: {final_path}")

    # 训练统计
    elapsed = time.time() - start_time
    logger.info("=" * 50)
    logger.info(f"训练完成!")
    logger.info(f"  总步数: {global_step}")
    logger.info(f"  最佳 loss: {best_loss:.4f}")
    logger.info(f"  总耗时: {elapsed/3600:.2f} 小时")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
