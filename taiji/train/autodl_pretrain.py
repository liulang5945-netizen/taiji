"""
态极 Autodl 多GPU预训练脚本
============================
在 Autodl 云GPU上进行从零预训练。
支持 1B/3B/7B 模型，自动适配 GPU 数量和显存。

用法:
  # 单机多卡预训练 (推荐4卡A100/A800)
  accelerate launch --multi_gpu --num_processes 4 taiji/train/autodl_pretrain.py --size 1b

  # 单机单卡 (V100/A100)
  python taiji/train/autodl_pretrain.py --size 350m

  # 7B模型 (需要8卡A100 80G)
  accelerate launch --multi_gpu --num_processes 8 taiji/train/autodl_pretrain.py --size 7b --deepspeed configs/deepspeed_zero3_7b.json
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
    parser = argparse.ArgumentParser(description="态极 Autodl 预训练")
    parser.add_argument("--size", type=str, default="1b", choices=["125m", "350m", "1b", "3b", "7b"])
    parser.add_argument("--data", type=str, default=None, help="预训练数据路径 (默认自动选择)")
    parser.add_argument("--output_dir", type=str, default="taiji_data/autodl_checkpoints")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None, help="从checkpoint恢复")
    parser.add_argument("--target_loss", type=float, default=1.5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed配置文件路径")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="启用梯度检查点")
    parser.add_argument("--bf16", action="store_true", default=True, help="使用bf16混合精度")
    parser.add_argument("--use_flash_attn", action="store_true", help="使用Flash Attention")
    parser.add_argument("--local_rank", type=int, default=-1, help="DeepSpeed local_rank")
    return parser.parse_args()


class AutodlPretrainDataset(torch.utils.data.Dataset):
    """预训练数据集 - 支持流式加载大数据"""

    def __init__(self, data_file, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        logger.info(f"加载数据: {data_file}")
        count = 0
        with open(data_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if "messages" in item:
                        # 使用态极正确格式: [系统]/[用户]/[助手]
                        parts = []
                        for m in item["messages"]:
                            role = m.get("role", "")
                            content = m.get("content", "")
                            if role == "system":
                                parts.append(f"[系统] {content}")
                            elif role == "user":
                                parts.append(f"[用户] {content}")
                            elif role == "assistant":
                                parts.append(f"[助手] {content}")
                        text = "\n".join(parts)
                    elif "text" in item:
                        text = item["text"]
                    else:
                        continue
                    if len(text) < 50:
                        continue
                    self.samples.append(text)
                    count += 1
                except json.JSONDecodeError:
                    continue
        logger.info(f"加载完成: {count} 条样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encoded = self.tokenizer.encode(text)
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

    if args.use_flash_attn:
        try:
            from taiji.utils.flash_attention import replace_with_flash_attention
            model = replace_with_flash_attention(model)
            logger.info("Flash Attention 已启用")
        except ImportError:
            logger.warning("Flash Attention 不可用,使用标准注意力")

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
    try:
        from accelerate import Accelerator
        accelerator = Accelerator(
            mixed_precision="bf16" if args.bf16 else "no",
            gradient_accumulation_steps=args.grad_accum,
        )
        use_accelerate = True
        logger.info(f"使用 Accelerate, 设备: {accelerator.device}")
    except ImportError:
        use_accelerate = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用原生 PyTorch, 设备: {device}")

    # 初始化 tokenizer
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()

    # 查找 SentencePiece 模型
    sp_candidates = [
        os.path.join("model", "tokenizer", "sentencepiece.model"),
        os.path.join("taiji", "tokenizer", "sentencepiece.model"),
        os.path.join("taiji_data", "final", "tokenizer", "sentencepiece.model"),
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
    data_file = args.data
    if data_file is None:
        candidates = [
            "taiji_data/training_data/pretrain_final.jsonl",
            "taiji_data/training_data/pretrain_all.jsonl",
        ]
        for c in candidates:
            if os.path.exists(c):
                data_file = c
                break
    if data_file is None:
        logger.error("未找到预训练数据文件!")
        return

    dataset = AutodlPretrainDataset(data_file, tokenizer, args.max_seq_len)
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

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    # 使用 accelerate 准备
    if use_accelerate:
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    else:
        model = model.to(device)

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
            start_step = 0
            best_loss = float("inf")
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
    logger.info(f"数据: {data_file} | 样本数: {len(dataset)}")
    logger.info(f"Batch: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    logger.info(f"LR: {args.lr} | SeqLen: {args.max_seq_len}")
    logger.info("=" * 60)

    while global_step < args.max_steps:
        epoch += 1
        for batch in dataloader:
            if global_step >= args.max_steps:
                break

            if use_accelerate:
                with accelerator.accumulate(model):
                    outputs = model(
                        tokens=batch["input_ids"],
                        targets=batch["labels"],
                    )
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(tokens=input_ids, targets=labels)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            # 更新学习率
            lr = get_lr(global_step, args.max_steps, args.warmup_steps, args.lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            loss_val = loss.item()
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
                    "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                    "best_loss": best_loss,
                    "config": config.__dict__ if hasattr(config, "__dict__") else {},
                }
                torch.save(save_obj, save_path)
                logger.info(f"Checkpoint 已保存: {save_path}")

            # 评估和早停
            if global_step % args.eval_every == 0:
                avg_eval_loss = running_loss / args.eval_every
                running_loss = 0.0

                if avg_eval_loss < best_loss - args.patience * 0.001:
                    best_loss = avg_eval_loss
                    no_improve = 0
                    best_path = output_dir / "best.pt"
                    save_obj = {
                        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                        "config": config.__dict__ if hasattr(config, "__dict__") else {},
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
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "config": config.__dict__ if hasattr(config, "__dict__") else {},
    }
    torch.save(save_obj, final_path)
    logger.info(f"最终模型已保存: {final_path}")
    logger.info(f"训练完成! 总步数: {global_step}, 最佳Loss: {best_loss:.4f}")


if __name__ == "__main__":
    args = get_args()
    train(args)
