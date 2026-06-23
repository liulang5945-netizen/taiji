# Autodl 平台训练指南

## 📋 概述

本指南帮助你在 Autodl 云 GPU 平台上训练态极模型，使用清洗后的 SFT 对话数据。

## 🚀 快速开始

### 1. 准备数据

数据已经清洗完成，位于：
```
taiji_data/training_data/sft_merged_clean.jsonl  # 106 条高质量对话
```

### 2. 上传项目到 Autodl

```bash
# 在本地打包项目
tar -czf taiji_project.tar.gz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' .

# 上传到 Autodl 实例
scp taiji_project.tar.gz root@<autodl-ip>:/root/

# 在 Autodl 上解压
ssh root@<autodl-ip>
tar -xzf taiji_project.tar.gz
cd taiji
```

### 3. 安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install sentencepiece accelerate
pip install -e .
```

## 🎯 训练命令

### 单卡训练 (适合 V100/A100 40G)

```bash
python taiji/train/autodl_pretrain.py \
    --size 350m \
    --data taiji_data/training_data/sft_merged_clean.jsonl \
    --max_seq_len 512 \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 3e-4 \
    --max_steps 10000 \
    --save_every 500 \
    --eval_every 200 \
    --output_dir taiji_data/autodl_checkpoints/sft_350m
```

### 多卡训练 (推荐 4 卡 A100 80G)

```bash
accelerate launch --multi_gpu --num_processes 4 \
    taiji/train/autodl_pretrain.py \
    --size 1b \
    --data taiji_data/training_data/sft_merged_clean.jsonl \
    --max_seq_len 512 \
    --batch_size 8 \
    --grad_accum 4 \
    --lr 1e-4 \
    --max_steps 50000 \
    --save_every 1000 \
    --eval_every 500 \
    --gradient_checkpointing \
    --bf16 \
    --output_dir taiji_data/autodl_checkpoints/sft_1b
```

### 从 Checkpoint 恢复训练

```bash
python taiji/train/autodl_pretrain.py \
    --size 350m \
    --data taiji_data/training_data/sft_merged_clean.jsonl \
    --resume taiji_data/autodl_checkpoints/sft_350m/checkpoint-5000.pt \
    --max_steps 20000 \
    --output_dir taiji_data/autodl_checkpoints/sft_350m_continued
```

## ⚙️ 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--size` | 1b | 模型大小: 125m/350m/1b/3b/7b |
| `--data` | 自动 | 训练数据路径 |
| `--max_seq_len` | 512 | 最大序列长度 |
| `--batch_size` | 2 | 每 GPU batch size |
| `--grad_accum` | 16 | 梯度累积步数 |
| `--lr` | 3e-4 | 学习率 |
| `--max_steps` | 50000 | 最大训练步数 |
| `--warmup_steps` | 200 | 预热步数 |
| `--target_loss` | 1.5 | 目标 loss |
| `--patience` | 20 | 早停耐心值 |
| `--gradient_checkpointing` | False | 启用梯度检查点 (省显存) |
| `--bf16` | True | 使用 bf16 混合精度 |
| `--use_flash_attn` | False | 使用 Flash Attention |

## 📊 显存估算

| 模型大小 | batch_size=2 | batch_size=4 | batch_size=8 |
|----------|--------------|--------------|--------------|
| 350m | ~4 GB | ~8 GB | ~16 GB |
| 1b | ~8 GB | ~16 GB | ~32 GB |
| 3b | ~16 GB | ~32 GB | ~64 GB |
| 7b | ~32 GB | ~64 GB | OOM |

> 💡 使用 `--gradient_checkpointing` 可减少约 30% 显存

## 🔧 Autodl 实例推荐

### 性价比方案
- **GPU**: RTX 3090 (24G) 或 RTX 4090 (24G)
- **模型**: 350m
- **配置**: 单卡, batch_size=4

### 性能方案
- **GPU**: A100 40G x 4
- **模型**: 1b
- **配置**: 多卡, batch_size=8

### 大模型方案
- **GPU**: A100 80G x 8
- **模型**: 3b 或 7b
- **配置**: 多卡 + DeepSpeed ZeRO-3

## 📈 训练监控

### 查看训练日志
```bash
# 实时查看日志
tail -f taiji_data/autodl_checkpoints/sft_350m/train.log

# 查看 GPU 使用
watch -n 1 nvidia-smi
```

### 关键指标
- **Loss**: 应持续下降，目标 < 1.5
- **Learning Rate**: 预热后应稳定在 lr 附近
- **GPU Memory**: 应保持在 90% 以下

## 🔄 训练后操作

### 1. 下载模型
```bash
# 在 Autodl 上打包
tar -czf model_checkpoint.tar.gz taiji_data/autodl_checkpoints/sft_350m/

# 下载到本地
scp root@<autodl-ip>:/root/taiji/model_checkpoint.tar.gz .
```

### 2. 合并到本地项目
```bash
# 解压到本地
tar -xzf model_checkpoint.tar.gz -C taiji_data/

# 测试模型
python -c "
from taiji.loader import load_model
model = load_model('taiji_data/autodl_checkpoints/sft_350m/best.pt')
print('模型加载成功!')
"
```

## ⚠️ 注意事项

1. **数据量**: 当前仅 106 条高质量对话，建议补充更多数据
2. **过拟合**: 数据量少时注意监控验证集 loss
3. **学习率**: 数据量少时建议降低学习率 (1e-4 ~ 5e-4)
4. **保存频率**: 数据量少时增加保存频率，方便回退

## 📝 数据补充建议

当前数据统计：
- 高质量对话: 106 条
- 身份数据: 48 条
- 技术问答: 约 50 条

建议补充：
1. 更多技术领域对话 (AI、编程、数学等)
2. 日常闲聊对话
3. 多轮上下文对话
4. 工具调用相关对话

## 🆘 常见问题

### Q: 训练 loss 不下降
A: 检查学习率是否过大，尝试降低到 1e-4

### Q: GPU 内存不足
A: 启用 `--gradient_checkpointing` 或减小 `--batch_size`

### Q: 训练速度慢
A: 确认使用了 `--bf16`，考虑使用多卡训练

### Q: 如何评估模型质量
A: 生成一些测试对话，检查回答的流畅度和准确性

---

*最后更新: 2026-06-23*
