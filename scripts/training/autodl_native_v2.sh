#!/bin/bash
# ============================================================
# 态极 1B Native-V2 AUTODL 一键训练脚本
# ============================================================
#
# 适用配置:
#   - GPU: RTX 4090D 24G (单卡) 或 A100 80G (单卡)
#   - 系统盘: 30G | 数据盘: 50G
#   - 镜像: PyTorch 2.3+ / Python 3.12 / CUDA 12.1
#
# 使用方法:
#   1. 在 AUTODL 上创建实例 (RTX 4090D 或 A100 80G 单卡)
#   2. 在 JupyterLab 终端中执行:
#      cd /root/autodl-tmp
#      git clone https://github.com/liulang5945-netizen/taiji.git
#      cd taiji
#      bash scripts/training/autodl_native_v2.sh [阶段]
#
# 阶段说明:
#   setup     - 仅安装依赖
#   data      - 仅下载预训练数据
#   tokenizer - 仅重建 tokenizer
#   smoke     - 仅跑 smoke run (200 步验证)
#   train     - 仅跑正式 1B stage1 预训练
#   all       - 执行全部步骤 (默认)
#
# ============================================================

set -e

# ============================================================
# 配置区 (可根据需要修改)
# ============================================================
STAGE=${1:-"all"}

# 项目根目录 (自动检测)
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"

# 数据目录
DATA_DIR="taiji_data/training_data/pretrain_mix_v1"
TOKENIZER_DIR="taiji/tokenizer_native_v2"
CORPUS_FILE="taiji_data/tokenizer/native_v2_corpus.txt"
SMOKE_OUTPUT="taiji_data/taiji_pretrained_1b_smoke"
TRAIN_OUTPUT="taiji_data/taiji_pretrained_1b_stage1"

# 训练参数 (适配 4090D 24G)
BATCH_SIZE=1
GRAD_ACCUM_SMOKE=8
GRAD_ACCUM_TRAIN=32
MAX_LENGTH_SMOKE=1024
MAX_LENGTH_TRAIN=2048
LEARNING_RATE=3e-4
MIN_LEARNING_RATE=3e-5
WARMUP_STEPS=1000
MAX_STEPS_SMOKE=200
MAX_STEPS_TRAIN=50000
LOG_EVERY_SMOKE=10
LOG_EVERY_TRAIN=50
SAVE_EVERY_SMOKE=100
SAVE_EVERY_TRAIN=10000
NUM_WORKERS=2
MAX_RECORDS_PER_SOURCE=100000

# ============================================================
# 辅助函数
# ============================================================
print_banner() {
    echo ""
    echo "============================================================"
    echo "  态极 1B Native-V2 训练"
    echo "  阶段: $1"
    echo "  时间: $(date)"
    echo "  目录: $PROJECT_DIR"
    echo "============================================================"
    echo ""
}

check_gpu() {
    echo "--- GPU 信息 ---"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$FREE_MEM" -lt 20000 ]; then
        echo "⚠️  警告: GPU 可用显存 ${FREE_MEM}MB，可能不足以运行 max_length=2048"
        echo "   建议先用 smoke run 测试，如果 OOM 则降低 max_length"
    else
        echo "✅ GPU 可用显存: ${FREE_MEM}MB，足够运行 1B 训练"
    fi
    echo ""
}

check_disk() {
    echo "--- 磁盘空间 ---"
    df -h /root/autodl-tmp 2>/dev/null || df -h .
    echo ""
}

# ============================================================
# Step 1: 安装依赖
# ============================================================
setup_deps() {
    print_banner "安装依赖"

    echo "安装训练核心依赖..."
    pip install sentencepiece huggingface_hub pyarrow tqdm numpy tensorboard -q

    # 安装项目本身 (不安装依赖，避免装 PyQt6 等桌面依赖)
    pip install --no-deps -e . -q

    echo "✅ 依赖安装完成"
    echo ""
    pip list 2>/dev/null | grep -E "sentencepiece|huggingface|torch|numpy|pyarrow"
    echo ""
}

# ============================================================
# Step 2: 下载预训练数据
# ============================================================
download_data() {
    print_banner "下载预训练数据"

    if [ -d "$DATA_DIR" ] && [ "$(ls -A $DATA_DIR/*.jsonl 2>/dev/null | wc -l)" -ge 4 ]; then
        echo "数据已存在，跳过下载。如需重新下载，请删除 $DATA_DIR"
        echo ""
        ls -lh $DATA_DIR/*.jsonl
        echo ""
        return 0
    fi

    echo "从 HuggingFace 下载预训练数据 (4 个数据源，每源 $MAX_RECORDS_PER_SOURCE 条)..."
    echo "数据源: fineweb_edu, skypile_zh, openwebmath, codeparrot_code"
    echo ""

    python scripts/data_prep/download_pretrain_mix_v1.py \
        --max-records-per-source "$MAX_RECORDS_PER_SOURCE"

    echo ""
    echo "✅ 数据下载完成"
    echo ""
    ls -lh $DATA_DIR/*.jsonl
    echo ""
}

# ============================================================
# Step 3: 重建 Tokenizer
# ============================================================
rebuild_tokenizer() {
    print_banner "重建 Native-V2 Tokenizer"

    if [ -f "$TOKENIZER_DIR/sentencepiece.model" ] && [ -f "$TOKENIZER_DIR/tokenizer_contract.json" ]; then
        echo "Tokenizer 已存在，跳过重建。如需重建，请删除 $TOKENIZER_DIR"
        echo ""
        return 0
    fi

    # 3.1 构建词表训练语料
    echo "Step 3.1: 构建词表训练语料..."
    python scripts/build_native_vocab_corpus.py \
        --data-dir "$DATA_DIR" \
        --data-dir taiji \
        --output "$CORPUS_FILE"
    echo ""

    # 3.2 训练 SentencePiece
    echo "Step 3.2: 训练 SentencePiece 词表..."
    python scripts/train_native_text_sp.py \
        --corpus "$CORPUS_FILE" \
        --output-dir "$TOKENIZER_DIR" \
        --contract taiji/tokenizer_contract.json
    echo ""

    # 3.3 验证
    echo "Step 3.3: 验证 Tokenizer..."
    python scripts/verify_native_tokenizer.py \
        --tokenizer-dir "$TOKENIZER_DIR" \
        --contract taiji/tokenizer_contract.json
    echo ""

    echo "✅ Tokenizer 重建完成"
    echo ""
}

# ============================================================
# Step 4: Smoke Run (200 步验证)
# ============================================================
smoke_run() {
    print_banner "Smoke Run (200 步)"

    check_gpu
    check_disk

    echo "训练参数:"
    echo "  max_steps=$MAX_STEPS_SMOKE"
    echo "  batch_size=$BATCH_SIZE"
    echo "  gradient_accumulation_steps=$GRAD_ACCUM_SMOKE"
    echo "  max_length=$MAX_LENGTH_SMOKE"
    echo "  output=$SMOKE_OUTPUT"
    echo ""
    echo "观察重点:"
    echo "  1. loss 是否稳定下降"
    echo "  2. 是否出现 shape / vocab / offset 错误"
    echo "  3. checkpoint 是否正常保存"
    echo "  4. GPU 显存使用情况"
    echo ""

    python scripts/native_v2/pretrain.py \
        --data_dir "$DATA_DIR" \
        --tokenizer-dir "$TOKENIZER_DIR" \
        --output "$SMOKE_OUTPUT" \
        --max_steps "$MAX_STEPS_SMOKE" \
        --batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUM_SMOKE" \
        --max_length "$MAX_LENGTH_SMOKE" \
        --learning_rate "$LEARNING_RATE" \
        --log_every "$LOG_EVERY_SMOKE" \
        --save_every "$SAVE_EVERY_SMOKE" \
        --num_workers "$NUM_WORKERS"

    echo ""
    echo "✅ Smoke Run 完成"
    echo ""
    echo "请检查:"
    echo "  1. loss 是否从 ~11 下降到 ~8-10 (200 步)"
    echo "  2. checkpoint 是否保存在 $SMOKE_OUTPUT/"
    echo "  3. 如有问题，检查日志输出"
    echo ""
    echo "如果 smoke run 正常，执行正式训练:"
    echo "  bash scripts/training/autodl_native_v2.sh train"
    echo ""
}

# ============================================================
# Step 5: 正式 1B Stage1 预训练
# ============================================================
full_train() {
    print_banner "正式 1B Stage1 预训练"

    check_gpu
    check_disk

    echo "训练参数:"
    echo "  max_steps=$MAX_STEPS_TRAIN"
    echo "  batch_size=$BATCH_SIZE"
    echo "  gradient_accumulation_steps=$GRAD_ACCUM_TRAIN"
    echo "  max_length=$MAX_LENGTH_TRAIN"
    echo "  learning_rate=$LEARNING_RATE"
    echo "  min_learning_rate=$MIN_LEARNING_RATE"
    echo "  warmup_steps=$WARMUP_STEPS"
    echo "  save_every=$SAVE_EVERY_TRAIN"
    echo "  output=$TRAIN_OUTPUT"
    echo ""
    echo "预计时间: 3-7 天 (4090D 单卡)"
    echo "预计显存: ~16-20GB (max_length=2048)"
    echo ""
    echo "⚠️  如果 OOM，请降低 max_length:"
    echo "   python scripts/native_v2/pretrain.py --max_length 1024 ..."
    echo ""

    # 使用 nohup 确保断开终端后继续训练
    echo "启动训练 (使用 nohup 后台运行)..."
    echo "查看日志: tail -f taiji_data/train_stage1.log"
    echo ""

    nohup python scripts/native_v2/pretrain.py \
        --data_dir "$DATA_DIR" \
        --tokenizer-dir "$TOKENIZER_DIR" \
        --output "$TRAIN_OUTPUT" \
        --max_steps "$MAX_STEPS_TRAIN" \
        --batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUM_TRAIN" \
        --max_length "$MAX_LENGTH_TRAIN" \
        --learning_rate "$LEARNING_RATE" \
        --min_learning_rate "$MIN_LEARNING_RATE" \
        --warmup_steps "$WARMUP_STEPS" \
        --log_every "$LOG_EVERY_TRAIN" \
        --save_every "$SAVE_EVERY_TRAIN" \
        --num_workers "$NUM_WORKERS" \
        > taiji_data/train_stage1.log 2>&1 &

    TRAIN_PID=$!
    echo "训练进程 PID: $TRAIN_PID"
    echo "日志文件: taiji_data/train_stage1.log"
    echo ""
    echo "常用命令:"
    echo "  查看日志:   tail -f taiji_data/train_stage1.log"
    echo "  查看 GPU:   watch -n 1 nvidia-smi"
    echo "  停止训练:   kill $TRAIN_PID"
    echo ""
    echo "✅ 正式训练已在后台启动"
    echo ""
}

# ============================================================
# 主流程
# ============================================================
case $STAGE in
    setup)
        setup_deps
        ;;
    data)
        download_data
        ;;
    tokenizer)
        rebuild_tokenizer
        ;;
    smoke)
        smoke_run
        ;;
    train)
        full_train
        ;;
    all)
        print_banner "全部步骤"
        echo "将依次执行: 安装依赖 → 下载数据 → 重建 Tokenizer → Smoke Run"
        echo "正式训练需要手动启动: bash scripts/training/autodl_native_v2.sh train"
        echo ""
        echo "按 Ctrl+C 可在任意步骤中断"
        echo ""

        setup_deps
        download_data
        rebuild_tokenizer
        smoke_run

        print_banner "全部验证步骤完成"
        echo "下一步: 检查 smoke run 结果，如果正常则启动正式训练:"
        echo ""
        echo "  bash scripts/training/autodl_native_v2.sh train"
        echo ""
        ;;
    *)
        echo "用法: bash scripts/training/autodl_native_v2.sh [阶段]"
        echo ""
        echo "阶段:"
        echo "  setup     - 仅安装依赖"
        echo "  data      - 仅下载预训练数据"
        echo "  tokenizer - 仅重建 tokenizer"
        echo "  smoke     - 仅跑 smoke run (200 步验证)"
        echo "  train     - 仅跑正式 1B stage1 预训练"
        echo "  all       - 执行全部步骤 (默认)"
        echo ""
        exit 1
        ;;
esac