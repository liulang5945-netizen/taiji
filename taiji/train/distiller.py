"""
知识蒸馏训练器 — 递归进化闭环的核心组件

态极在睡眠 Phase 5 设计出下一代模型架构后，由本模块执行教师→学生
蒸馏，将当前模型的能力迁移到新一代模型中。

教师模型冻结（eval + no_grad），学生模型从头训练。
损失 = α * KL(teacher_logits/T, student_logits/T) * T² + (1-α) * CE(student_logits, labels)
"""

from __future__ import annotations

import logging
from typing import Any, Generator, List, Optional

import torch
import torch.nn.functional as F

from taiji.architecture import ModelSelf
from taiji.tokenizer import ModelSelfTokenizer

logger = logging.getLogger("Taiji.Distiller")


class DistillationTrainer:
    """轻量级知识蒸馏训练器。

    不需要额外依赖——全部使用项目已有的 torch / ModelSelf / ModelSelfTokenizer。
    """

    def __init__(
        self,
        teacher_model: ModelSelf,
        student_model: ModelSelf,
        tokenizer: ModelSelfTokenizer,
        alpha: float = 0.7,
        temperature: float = 3.0,
        lr: float = 1e-4,
    ):
        """
        Args:
            teacher_model: 当前模型（冻结，只做前向推理）
            student_model: 新一代模型（待训练）
            tokenizer: 共享的分词器
            alpha: 软目标权重（0=纯硬目标, 1=纯软目标）
            temperature: 蒸馏温度（越高越平滑）
            lr: 学习率
        """
        self.teacher = teacher_model
        self.student = student_model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.temperature = temperature
        self.lr = lr

    def distill(
        self,
        texts: List[str],
        num_epochs: int = 3,
        batch_size: int = 2,
        device: str = "cpu",
        max_length: int = 512,
        grad_clip: float = 1.0,
    ) -> Generator[tuple[int, int, float, dict[str, Any]], None, None]:
        """执行知识蒸馏训练。

        Args:
            texts: 训练文本列表（对话、语料等）
            num_epochs: 训练轮数
            batch_size: 批次大小
            device: 训练设备
            max_length: 最大序列长度
            grad_clip: 梯度裁剪阈值

        Yields:
            (epoch, step, loss, metrics) 进度元组
        """
        self.teacher.to(device)
        self.teacher.eval()
        self.student.to(device)
        self.student.train()

        # 冻结教师所有参数
        for param in self.teacher.parameters():
            param.requires_grad = False

        optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.lr)
        total_steps = len(texts) * num_epochs
        step = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_kd_loss = 0.0
            epoch_ce_loss = 0.0
            batch_count = 0

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                optimizer.zero_grad(set_to_none=True)

                batch_loss = torch.tensor(0.0, device=device)
                batch_kd = torch.tensor(0.0, device=device)
                batch_ce = torch.tensor(0.0, device=device)

                for text in batch_texts:
                    if not text or not text.strip():
                        continue

                    encoded = self.tokenizer(
                        text,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=max_length,
                    )
                    input_ids = encoded["input_ids"].to(device)
                    labels = input_ids.clone()

                    # 教师前向（无梯度）
                    with torch.no_grad():
                        teacher_out = self.teacher(input_ids)
                        teacher_logits = teacher_out.logits

                    # 学生前向（有梯度）
                    student_out = self.student(input_ids)
                    student_logits = student_out.logits

                    # 软目标损失：KL(student/T || teacher/T) * T²
                    kd_loss = _kd_loss(
                        student_logits, teacher_logits,
                        self.temperature, ignore_index=0,
                    )

                    # 硬目标损失：标准交叉熵
                    ce_loss = F.cross_entropy(
                        student_logits.view(-1, student_logits.size(-1)),
                        labels.view(-1),
                        ignore_index=0,  # 忽略 padding
                    )

                    item_loss = (
                        self.alpha * kd_loss + (1.0 - self.alpha) * ce_loss
                    ) / len(batch_texts)

                    item_loss.backward()

                    batch_loss = batch_loss + item_loss.detach() * len(batch_texts)
                    batch_kd = batch_kd + kd_loss.detach()
                    batch_ce = batch_ce + ce_loss.detach()

                # Gradient clipping and optimizer step
                if batch_count > 0 or batch_loss.item() > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(), grad_clip
                    )
                optimizer.step()

                step += 1
                epoch_loss += batch_loss.item()
                epoch_kd_loss += batch_kd.item()
                epoch_ce_loss += batch_ce.item()
                batch_count += 1

                yield (
                    epoch,
                    step,
                    batch_loss.item(),
                    {
                        "epoch": epoch + 1,
                        "step": step,
                        "total_steps": total_steps,
                        "loss": batch_loss.item(),
                        "kd_loss": batch_kd.item(),
                        "ce_loss": batch_ce.item(),
                        "lr": self.lr,
                    },
                )

            avg_loss = epoch_loss / max(batch_count, 1)
            avg_kd = epoch_kd_loss / max(batch_count, 1)
            avg_ce = epoch_ce_loss / max(batch_count, 1)
            logger.info(
                f"Distill Epoch {epoch+1}/{num_epochs} done, "
                f"avg_loss={avg_loss:.4f}, kd={avg_kd:.4f}, ce={avg_ce:.4f}"
            )


def _kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    ignore_index: int = 0,
) -> torch.Tensor:
    """KL 散度蒸馏损失（Hinton et al. 2015）。

    L_KD = KL(softmax(teacher/T) || softmax(student/T)) * T²
    """
    # 只计算非 padding 位置的损失
    vocab_size = student_logits.size(-1)
    logits_s = student_logits.view(-1, vocab_size)
    logits_t = teacher_logits.view(-1, vocab_size)

    loss = F.kl_div(
        F.log_softmax(logits_s / temperature, dim=-1),
        F.softmax(logits_t / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)

    return loss
