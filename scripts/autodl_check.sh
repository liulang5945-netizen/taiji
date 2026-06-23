#!/bin/bash
# Autodl 平台环境检查脚本
# 在 JupyterLab 终端中运行这些命令

echo "=========================================="
echo "1. 系统信息"
echo "=========================================="
uname -a
echo ""

echo "=========================================="
echo "2. GPU 信息"
echo "=========================================="
nvidia-smi
echo ""

echo "=========================================="
echo "3. 磁盘空间"
echo "=========================================="
df -h
echo ""

echo "=========================================="
echo "4. 当前目录结构"
echo "=========================================="
pwd
ls -la
echo ""

echo "=========================================="
echo "5. Python 环境"
echo "=========================================="
python --version
pip list | grep -E "torch|transformers|accelerate|datasets|sentencepiece"
echo ""

echo "=========================================="
echo "6. 检查是否有 taiji 项目"
echo "=========================================="
if [ -d "taiji" ]; then
    echo "找到 taiji 目录"
    ls -la taiji/
    echo ""
    echo "taiji_data 目录:"
    if [ -d "taiji/taiji_data" ]; then
        ls -la taiji/taiji_data/
    else
        echo "未找到 taiji_data"
    fi
else
    echo "未找到 taiji 目录"
    echo "检查其他位置..."
    find /root -name "taiji" -type d 2>/dev/null | head -5
fi
echo ""

echo "=========================================="
echo "7. 检查训练数据"
echo "=========================================="
if [ -d "taiji/taiji_data/training_data" ]; then
    echo "训练数据目录:"
    ls -lh taiji/taiji_data/training_data/
    echo ""
    echo "SFT 数据:"
    ls -lh taiji/taiji_data/training_data/sft/*.jsonl 2>/dev/null | head -20
    echo ""
    echo "预训练数据:"
    ls -lh taiji/taiji_data/training_data/pretrain_final.jsonl 2>/dev/null
else
    echo "未找到训练数据目录"
fi
echo ""

echo "=========================================="
echo "8. 内存信息"
echo "=========================================="
free -h
echo ""

echo "=========================================="
echo "检查完成!"
echo "=========================================="
