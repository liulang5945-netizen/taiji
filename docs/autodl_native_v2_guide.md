# 态极 1B Native-V2 AUTODL 训练指南

> 本文档适用于 **native-v2 训练路线**，替代旧的 `autodl_training_guide.md`。
> 推荐配合 `scripts/training/autodl_native_v2.sh` 一键脚本使用。

## 📋 环境要求

| 项目 | 推荐配置 |
|------|----------|
| GPU | RTX 4090D 24G（性价比）或 A100 80G（性能）|
| 内存 | ≥ 32GB |
| 系统盘 | 30GB+ |
| 数据盘 | 50GB+（AUTODL 数据盘通常挂载在 `/root/autodl-tmp`）|
| 镜像 | PyTorch 2.3+ / Python 3.12 / CUDA 12.1 |

## 🚀 快速开始

### 1. 创建 AUTODL 实例

在 [AUTODL 控制台](https://www.autodl.com/) 创建实例：

- **GPU**：选择 RTX 4090D（或 A100 80G）
- **镜像**：选择 PyTorch 2.3+ / CUDA 12.1
- **数据盘**：建议 50GB+

### 2. 克隆项目

```bash
# 在 JupyterLab 终端中执行
cd /root/autodl-tmp
git clone https://github.com/liulang5945-netizen/taiji.git
cd taiji
```

### 3. 一键执行全部步骤

```bash
bash scripts/training/autodl_native_v2.sh all
```

这会依次执行：
1. 安装依赖
2. 下载预训练数据
3. 重建 Native-V2 Tokenizer
4. Smoke Run（200 步验证）

### 4. 启动正式训练

```bash
# 确认 smoke run 正常后
bash scripts/training/autodl_native_v2.sh train
```

---

## 📖 分步详解

如果想逐步执行（推荐新手），可以分阶段运行：

```bash
# Step 1: 安装依赖
bash scripts/training/autodl_native_v2.sh setup

# Step 2: 下载预训练数据
bash scripts/training/autodl_native_v2.sh data

# Step 3: 重建 Tokenizer
bash scripts/training/autodl_native_v2.sh tokenizer

# Step 4: Smoke Run
bash scripts/training/autodl_native_v2.sh smoke

# Step 5: 正式训练
bash scripts/training/autodl_native_v2.sh train
```

---

## 🔧 手动执行（高级用户）

如果你想手动控制每个步骤，不使用一键脚本：

### 安装依赖

```bash
pip install sentencepiece huggingface_hub pyarrow tqdm numpy tensorboard
pip install --no-deps -e .
```

> ⚠️ 使用 `--no-deps` 避免安装 PyQt6 等桌面依赖。

### 下载预训练数据

```bash
python scripts/data_prep/download_pretrain_mix_v1.py \
    --max-records-per-source 100000
```

数据源：
- `fineweb_edu` — 英文高质量教育网页
- `skypile_zh` — 中文网页语料
- `openwebmath` — 数学语料
- `codeparrot_code` — 代码语料

下载后数据保存在 `taiji_data/training_data/pretrain_mix_v1/`。

### 重建 Tokenizer

```bash
# 1. 构建词表训练语料
python scripts/build_native_vocab_corpus.py \
    --data-dir taiji_data/training_data/pretrain_mix_v1 \
    --data-dir taiji \
    --output taiji_data/tokenizer/native_v2_corpus.txt

# 2. 训练 SentencePiece
python scripts/train_native_text_sp.py \
    --corpus taiji_data/tokenizer/native_v2_corpus.txt \
    --output-dir taiji/tokenizer_native_v2 \
    --contract taiji/tokenizer_contract.json

# 3. 验证
python scripts/verify_native_tokenizer.py \
    --tokenizer-dir taiji/tokenizer_native_v2 \
    --contract taiji/tokenizer_contract.json
```

验证标准：
- `sp.GetPieceSize()` ≤ `text_vocab_size` (242612)
- `text_offset` = 13388
- `total_vocab_size` = 256000

### Smoke Run（200 步验证）

```bash
python scripts/native_v2/pretrain.py \
    --data_dir taiji_data/training_data/pretrain_mix_v1 \
    --tokenizer-dir taiji/tokenizer_native_v2 \
    --output taiji_data/taiji_pretrained_1b_smoke \
    --max_steps 200 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_length 1024 \
    --log_every 10 \
    --save_every 100
```

### 正式 1B Stage1 预训练

```bash
python scripts/native_v2/pretrain.py \
    --data_dir taiji_data/training_data/pretrain_mix_v1 \
    --tokenizer-dir taiji/tokenizer_native_v2 \
    --output taiji_data/taiji_pretrained_1b_stage1 \
    --max_steps 50000 \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_length 2048 \
    --learning_rate 3e-4 \
    --min_learning_rate 3e-5 \
    --warmup_steps 1000 \
    --log_every 50 \
    --save_every 10000
```

---

## 📊 训练参数说明

| 参数 | Smoke Run | 正式训练 | 说明 |
|------|-----------|----------|------|
| `max_steps` | 200 | 50000 | 最大训练步数 |
| `batch_size` | 1 | 1 | 每步样本数 |
| `gradient_accumulation_steps` | 8 | 32 | 梯度累积，等效 batch=8/32 |
| `max_length` | 1024 | 2048 | 最大序列长度 |
| `learning_rate` | 3e-4 | 3e-4 | 初始学习率 |
| `min_learning_rate` | - | 3e-5 | 余弦退火最低学习率 |
| `warmup_steps` | - | 1000 | 学习率预热步数 |
| `save_every` | 100 | 10000 | checkpoint 保存间隔 |

### 显存估算（RTX 4090D 24G）

| 配置 | 预估显存 |
|------|----------|
| batch=1, max_length=1024 | ~8-10 GB |
| batch=1, max_length=2048 | ~14-18 GB |
| batch=1, max_length=2048 + AMP | ~16-20 GB |

> 4090D 24G 可以跑 `max_length=2048`。如果 OOM，降低 `max_length` 到 1024。

---

## 📈 训练监控

### 查看训练日志

```bash
# 如果使用一键脚本的 nohup 后台模式
tail -f taiji_data/train_stage1.log

# 如果是前台运行，直接看终端输出
```

### 查看 GPU 使用

```bash
watch -n 1 nvidia-smi
```

### 关键指标

- **Loss**：应持续下降。200 步后约 ~8-10，50000 步后目标 < 3.0
- **Learning Rate**：预热阶段线性增长，之后余弦退火
- **GPU Memory**：应保持在 22GB 以下（4090D 24G）

---

## ⚠️ 已知问题与解决方案

### 1. OOM (Out of Memory)

**症状**：`RuntimeError: CUDA out of memory`

**解决**：
```bash
# 降低 max_length
python scripts/native_v2/pretrain.py --max_length 1024 ...
```

### 2. 数据下载失败

**症状**：HuggingFace 下载超时或被墙

**解决**：
```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 然后重新下载
python scripts/data_prep/download_pretrain_mix_v1.py --max-records-per-source 100000
```

### 3. 磁盘空间不足

**症状**：训练中途因磁盘满而失败

**解决**：
```bash
# 增大 save_every，减少 checkpoint 数量
python scripts/native_v2/pretrain.py --save_every 20000 ...

# 或手动清理旧 checkpoint
rm -rf taiji_data/taiji_pretrained_1b_stage1/checkpoint-5000
```

### 4. AUTODL 实例断开

**症状**：长时间训练时实例被回收

**解决**：
- 使用一键脚本的 `train` 阶段，已内置 `nohup` 后台运行
- 重新连接后检查进程：`ps aux | grep pretrain`
- 如进程已终止，从最新 checkpoint 恢复（当前脚本暂不支持自动 resume）

### 5. HuggingFace 登录

部分数据源可能需要 HuggingFace 登录：

```bash
pip install huggingface_hub
huggingface-cli login
```

---

## 🔄 训练后操作

### 查看训练产物

```bash
ls -lh taiji_data/taiji_pretrained_1b_stage1/
# checkpoint-10000/
# checkpoint-20000/
# checkpoint-30000/
# checkpoint-40000/
# checkpoint-50000/
# final/
```

每个 checkpoint 包含：
- `model.pt` — 模型权重
- `config.json` — 模型配置
- `tokenizer_contract.json` — tokenizer 合约
- `sentencepiece.model` — tokenizer 模型

### 下载模型到本地

```bash
# 在 AUTODL 上打包
cd /root/autodl-tmp
tar -czf taiji_1b_stage1_final.tar.gz \
    taiji/taiji_data/taiji_pretrained_1b_stage1/final/

# 下载（使用 AUTODL 的文件传输功能或 scp）
```

### 后续训练路线

1. **pretrain_stage1** → 当前阶段
2. **pretrain_stage2** → 扩充更多 shard 继续预训练
3. **conversation_sft** → 对话 SFT 微调
4. **agent_tool_sft** → Agent/工具调用微调

---

## 💰 费用估算

| GPU | 单价（参考）| Smoke Run | 正式训练 50000 步 |
|-----|------------|-----------|-------------------|
| RTX 4090D 24G | ~¥8-12/小时 | ~¥2-5 | ~¥600-2000 |
| A100 80G | ~¥30-40/小时 | ~¥5-15 | ~¥2000-6000 |

> 4090D 单卡训练 1B 模型 50000 步，预计 3-7 天。

---

## 📝 与旧教程的区别

| 项目 | 旧教程 (`autodl_training_guide.md`) | 本教程 |
|------|--------------------------------------|--------|
| 训练入口 | `taiji/train/autodl_pretrain.py` | `scripts/native_v2/pretrain.py` |
| Tokenizer | 使用旧词表 | 重新训练 native-v2 词表 |
| 数据 | 106 条 SFT 对话 | 4 源预训练语料（10万+条）|
| 词表大小 | - | 256000 (text_offset=13388) |
| 训练目标 | SFT 微调 | 从零预训练 1B |
| 上下文长度 | 512 | 2048 |

---

*最后更新: 2026-06-24*