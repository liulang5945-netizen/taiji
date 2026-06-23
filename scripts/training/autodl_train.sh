#!/bin/bash
# ============================================================
# 态极 Autodl 预训练启动脚本
# ============================================================
# 使用方法:
#   1. 在 Autodl 上创建实例 (推荐: A100 40G x4 或 A100 80G x8)
#   2. 上传项目到 /root/taiji
#   3. cd /root/taiji
#   4. bash scripts/autodl_train.sh [模型大小]
#
# 推荐配置:
#   350m: 1x A100 40G (单卡)
#   1b:   4x A100 40G (多卡)
#   3b:   4x A100 80G (多卡)
#   7b:   8x A100 80G (多卡, 需要 zero3)
# ============================================================

set -e

MODEL_SIZE=${1:-"1b"}
PROJECT_DIR="/root/taiji"
cd "$PROJECT_DIR"

echo "============================================================"
echo "态极预训练 - 模型大小: $MODEL_SIZE"
echo "时间: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "GPU数量: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "============================================================"

# 安装依赖
pip install -r requirements.txt -q
pip install deepspeed accelerate -q

# 可选: 安装 Flash Attention (A100/H100 推荐)
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "A100\|H100"; then
    echo "检测到 A100/H100, 安装 Flash Attention..."
    pip install flash-attn --no-build-isolation -q || echo "Flash Attention 安装失败, 使用标准注意力"
fi

# 下载补充数据 (如果需要)
if [ ! -d "taiji_data/training_data/supplementary" ]; then
    echo "下载补充训练数据..."
    python scripts/download_supplementary_data.py --all --sample 50000
fi

# 根据模型大小选择配置
case $MODEL_SIZE in
    125m|350m)
        echo "单卡训练 $MODEL_SIZE"
        python taiji/train/autodl_pretrain.py \
            --size $MODEL_SIZE \
            --batch_size 4 \
            --grad_accum 8 \
            --max_seq_len 512 \
            --lr 3e-4 \
            --max_steps 30000 \
            --save_every 500 \
            --eval_every 500 \
            --gradient_checkpointing \
            --bf16
        ;;
    1b)
        NUM_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        echo "多卡训练 $MODEL_SIZE, GPU数量: $NUM_GPU"
        accelerate launch \
            --multi_gpu \
            --num_processes $NUM_GPU \
            --mixed_precision bf16 \
            taiji/train/autodl_pretrain.py \
            --size 1b \
            --batch_size 2 \
            --grad_accum 16 \
            --max_seq_len 512 \
            --lr 3e-4 \
            --max_steps 50000 \
            --save_every 500 \
            --eval_every 1000 \
            --gradient_checkpointing \
            --use_flash_attn
        ;;
    3b)
        NUM_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        echo "多卡训练 $MODEL_SIZE (DeepSpeed ZeRO-2), GPU数量: $NUM_GPU"
        accelerate launch \
            --multi_gpu \
            --num_processes $NUM_GPU \
            --mixed_precision bf16 \
            taiji/train/autodl_pretrain.py \
            --size 3b \
            --batch_size 2 \
            --grad_accum 16 \
            --max_seq_len 512 \
            --lr 2e-4 \
            --max_steps 50000 \
            --save_every 500 \
            --eval_every 1000 \
            --deepspeed configs/deepspeed_zero2.json \
            --gradient_checkpointing \
            --use_flash_attn
        ;;
    7b)
        NUM_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        echo "多卡训练 $MODEL_SIZE (DeepSpeed ZeRO-3), GPU数量: $NUM_GPU"
        accelerate launch \
            --multi_gpu \
            --num_processes $NUM_GPU \
            --mixed_precision bf16 \
            taiji/train/autodl_pretrain.py \
            --size 7b \
            --batch_size 1 \
            --grad_accum 32 \
            --max_seq_len 512 \
            --lr 1e-4 \
            --max_steps 50000 \
            --save_every 500 \
            --eval_every 1000 \
            --deepspeed configs/deepspeed_zero3_7b.json \
            --gradient_checkpointing \
            --use_flash_attn
        ;;
    *)
        echo "未知模型大小: $MODEL_SIZE"
        echo "支持: 125m, 350m, 1b, 3b, 7b"
        exit 1
        ;;
esac

echo "============================================================"
echo "训练完成! 时间: $(date)"
echo "模型保存在: taiji_data/autodl_checkpoints/"
echo "============================================================"
