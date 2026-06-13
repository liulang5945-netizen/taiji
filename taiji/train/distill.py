"""
知识蒸馏系统
============
让小模型（Teacher）的知识传递给大模型（Student）

用法:
  python -m taiji.train.distill

蒸馏流程:
  1. Teacher (1B) 生成高质量回答
  2. Student (3B) 学习 Teacher 的回答
  3. Student 继承 Teacher 的知识，同时拥有更大的容量
"""
import os
import sys
import json
import time
import logging
import torch
import torch.nn.functional as F

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("Distill")

# ======================== 配置 ========================

CONFIG = {
    # Teacher 模型（已训练好的小模型）
    "teacher_model_size": "1b",
    "teacher_checkpoint": "taiji_data/evolution_data/upgraded_models/pretrained_1b/best",

    # Student 模型（要训练的大模型）
    "student_model_size": "3b",

    # 蒸馏数据
    "distill_data": "taiji_data/training_data/pretrain_final.jsonl",
    "output_dir": "taiji_data/evolution_data/upgraded_models/distilled_3b",

    # 蒸馏参数
    "temperature": 2.0,          # 温度参数，软化概率分布
    "alpha": 0.5,                # 蒸馏 loss 权重 (1-alpha 为硬标签 loss)
    "max_seq_len": 512,
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 1e-4,       # 蒸馏用较小学习率
    "warmup_steps": 500,
    "max_steps": 30000,

    # 保存
    "save_every": 500,
    "log_every": 10,
}


# ======================== 蒸馏数据集 ========================

class DistillDataset(torch.utils.data.Dataset):
    """蒸馏数据集：同时生成 Teacher 和 Student 的输入"""

    def __init__(self, data_file, teacher_tokenizer, student_tokenizer, max_len=512):
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.max_len = max_len
        self.samples = []

        logger.info(f"加载蒸馏数据: {data_file}")
        with open(data_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    messages = item.get("messages", [])
                    if len(messages) >= 2:
                        # 提取用户输入和助手回答
                        user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")
                        assistant_msg = next((m["content"] for m in messages if m.get("role") == "assistant"), "")

                        if user_msg and assistant_msg and len(user_msg) > 10:
                            self.samples.append({
                                "input": user_msg,
                                "target": assistant_msg,
                            })
                except json.JSONDecodeError:
                    continue

        logger.info(f"加载完成: {len(self.samples):,} 条蒸馏样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Teacher 编码
        teacher_input = self.teacher_tokenizer.encode(sample["input"])
        teacher_target = self.teacher_tokenizer.encode(sample["target"])

        # Student 编码
        student_input = self.student_tokenizer.encode(sample["input"])
        student_target = self.student_tokenizer.encode(sample["target"])

        # 截断
        teacher_input = teacher_input[:self.max_len]
        teacher_target = teacher_target[:self.max_len]
        student_input = student_input[:self.max_len]
        student_target = student_target[:self.max_len]

        return {
            "teacher_input": torch.tensor(teacher_input, dtype=torch.long),
            "teacher_target": torch.tensor(teacher_target, dtype=torch.long),
            "student_input": torch.tensor(student_input, dtype=torch.long),
            "student_target": torch.tensor(student_target, dtype=torch.long),
        }


def collate_fn(batch):
    """动态填充"""
    # 分别处理 teacher 和 student
    teacher_inputs = [b["teacher_input"] for b in batch]
    teacher_targets = [b["teacher_target"] for b in batch]
    student_inputs = [b["student_input"] for b in batch]
    student_targets = [b["student_target"] for b in batch]

    def pad_and_stack(seqs):
        max_len = max(len(s) for s in seqs)
        padded = [torch.cat([s, torch.zeros(max_len - len(s), dtype=torch.long)]) for s in seqs]
        return torch.stack(padded)

    return {
        "teacher_inputs": pad_and_stack(teacher_inputs),
        "teacher_targets": pad_and_stack(teacher_targets),
        "student_inputs": pad_and_stack(student_inputs),
        "student_targets": pad_and_stack(student_targets),
    }


# ======================== 蒸馏损失 ========================

def distillation_loss(student_logits, teacher_logits, targets, temperature, alpha):
    """
    蒸馏损失 = alpha * KL(soft) + (1-alpha) * CE(hard)

    Args:
        student_logits: Student 模型的输出 logits
        teacher_logits: Teacher 模型的输出 logits
        targets: 真实标签
        temperature: 温度参数
        alpha: 蒸馏 loss 权重
    """
    # 软标签损失 (KL 散度)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)

    # 硬标签损失 (交叉熵)
    hard_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), targets.view(-1), ignore_index=-100)

    return alpha * soft_loss + (1 - alpha) * hard_loss


# ======================== 训练主循环 ========================

def main():
    from taiji.config import ModelConfig
    from taiji.architecture import ModelSelf
    from taiji.tokenizer import ModelSelfTokenizer
    from taiji.loader import save_model, load_model

    cfg = CONFIG

    # ======================== 加载 Teacher ========================
    logger.info("=" * 50)
    logger.info("加载 Teacher 模型")
    logger.info("=" * 50)

    teacher_config = ModelConfig.size_1b()
    teacher = ModelSelf(teacher_config)

    # 加载 Teacher 权重
    teacher_ckpt_path = os.path.join(cfg["teacher_checkpoint"], "model.pt")
    if os.path.exists(teacher_ckpt_path):
        ckpt = torch.load(teacher_ckpt_path, map_location="cpu", weights_only=False)
        teacher.load_state_dict({k: v for k, v in ckpt.items() if not k.startswith("_")}, strict=False)
        logger.info(f"Teacher 加载完成: {cfg['teacher_checkpoint']}")
    else:
        logger.error(f"Teacher checkpoint 不存在: {teacher_ckpt_path}")
        return

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False  # Teacher 不更新

    # ======================== 创建 Student ========================
    logger.info("=" * 50)
    logger.info("创建 Student 模型 (3B)")
    logger.info("=" * 50)

    student_config = ModelConfig.size_3b()  # 3B 模型
    student = ModelSelf(student_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = teacher.to(device)
    student = student.to(device)

    student_params = sum(p.numel() for p in student.parameters())
    logger.info(f"Student 参数量: {student_params:,} ({student_params/1e9:.2f}B)")

    # ======================== 加载 Tokenizer ========================
    sp_path = os.path.join("taiji", "tokenizer", "sentencepiece.model")
    if os.path.exists(sp_path):
        teacher_tokenizer = ModelSelfTokenizer(sp_model_path=sp_path)
        student_tokenizer = ModelSelfTokenizer(sp_model_path=sp_path)
    else:
        teacher_tokenizer = ModelSelfTokenizer()
        student_tokenizer = ModelSelfTokenizer()

    # ======================== 加载数据 ========================
    dataset = DistillDataset(
        cfg["distill_data"],
        teacher_tokenizer,
        student_tokenizer,
        max_len=cfg["max_seq_len"]
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True,
    )

    # ======================== 优化器 ========================
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    # 学习率调度
    def lr_lambda(step):
        if step < cfg["warmup_steps"]:
            return step / cfg["warmup_steps"]
        progress = (step - cfg["warmup_steps"]) / (cfg["max_steps"] - cfg["warmup_steps"])
        return max(0.1, 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ======================== 训练 ========================
    os.makedirs(cfg["output_dir"], exist_ok=True)

    global_step = 0
    total_loss = 0.0
    start_time = time.time()

    logger.info("=" * 50)
    logger.info("开始蒸馏训练")
    logger.info("=" * 50)
    logger.info(f"  Teacher: {cfg['teacher_model_size']}")
    logger.info(f"  Student: {cfg['student_model_size']}")
    logger.info(f"  Temperature: {cfg['temperature']}")
    logger.info(f"  Alpha: {cfg['alpha']}")
    logger.info(f"  Max steps: {cfg['max_steps']}")
    logger.info("")

    student.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(100):  # 足够多的 epoch
        for batch_idx, batch in enumerate(dataloader):
            if global_step >= cfg["max_steps"]:
                break

            # 移到 GPU
            teacher_inputs = batch["teacher_inputs"].to(device)
            teacher_targets = batch["teacher_targets"].to(device)
            student_inputs = batch["student_inputs"].to(device)
            student_targets = batch["student_targets"].to(device)

            # Teacher 前向（不计算梯度）
            with torch.no_grad():
                teacher_output = teacher(teacher_inputs, targets=teacher_targets)
                teacher_logits = teacher_output.logits if hasattr(teacher_output, 'logits') else None

            # Student 前向
            student_output = student(student_inputs, targets=student_targets)
            student_logits = student_output.logits if hasattr(student_output, 'logits') else None

            # 计算蒸馏损失
            if teacher_logits is not None and student_logits is not None:
                loss = distillation_loss(
                    student_logits,
                    teacher_logits,
                    student_targets,
                    cfg["temperature"],
                    cfg["alpha"]
                )
            else:
                # 如果没有 logits，使用普通交叉熵
                loss = student_output.loss

            loss = loss / cfg["gradient_accumulation_steps"]
            loss.backward()
            total_loss += loss.item()

            # 梯度累积
            if (batch_idx + 1) % cfg["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # 日志
                if global_step % cfg["log_every"] == 0:
                    avg_loss = total_loss / cfg["log_every"] * cfg["gradient_accumulation_steps"]
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Step {global_step:>6}/{cfg['max_steps']} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                        f"Elapsed: {elapsed/60:.1f}min"
                    )
                    total_loss = 0.0

                # 保存
                if global_step % cfg["save_every"] == 0:
                    ckpt_dir = os.path.join(cfg["output_dir"], f"step_{global_step}")
                    save_model(student, student_tokenizer, ckpt_dir)
                    logger.info(f"Checkpoint 保存: {ckpt_dir}")

        if global_step >= cfg["max_steps"]:
            break

    # 保存最终模型
    final_dir = os.path.join(cfg["output_dir"], "final")
    save_model(student, student_tokenizer, final_dir)
    logger.info(f"最终模型保存: {final_dir}")

    logger.info("=" * 50)
    logger.info("蒸馏完成！")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
