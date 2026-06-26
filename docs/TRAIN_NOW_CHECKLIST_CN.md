# 态极 1B 立即开训清单

如果你现在着急开训，不要在平台上临时想命令，直接按下面顺序执行。

如果目标是完整 `12B tokens` 训练规划，而不是马上跑 smoke，请先读：

- [TAIJI_1B_12B_TOKEN_TRAINING_PLAN_CN.md](/E:/taiji/docs/TAIJI_1B_12B_TOKEN_TRAINING_PLAN_CN.md)

那份文档已经把 `12B tokens` 的数据配比、词表重练、预训练步数、SFT、多模态和给其他 AI 的执行提示词写清楚。

## 1. 克隆和安装

```bash
cd /root/autodl-tmp
git clone https://github.com/liulang5945-netizen/taiji.git
cd taiji

export HF_ENDPOINT=https://hf-mirror.com

pip install sentencepiece huggingface_hub pyarrow tqdm numpy tensorboard
pip install --no-deps -e .
```

## 2. 先下最小可用集

```bash
export TAIJI_DATA_PRESET=stage0_smoke
export TAIJI_DATA_MAX_RECORDS=100000
export TAIJI_DATA_SHARDS_PER_SOURCE=1

bash scripts/training/autodl_native_v2.sh data
bash scripts/training/autodl_native_v2.sh audit
```

## 3. 再补镜像友好英文

```bash
export TAIJI_DATA_PRESET=english_boost_mirror
export TAIJI_DATA_SHARDS_PER_SOURCE=1

bash scripts/training/autodl_native_v2.sh data
bash scripts/training/autodl_native_v2.sh audit
```

## 4. 重练 tokenizer

```bash
bash scripts/training/autodl_native_v2.sh tokenizer
```

## 5. smoke run

```bash
bash scripts/training/autodl_native_v2.sh smoke
```

## 6. smoke 正常后再正式训练

```bash
bash scripts/training/autodl_native_v2.sh train
```

## 7. 你现在最该用的 preset

- 最先跑：`stage0_smoke`
- 英文补强：`english_boost_mirror`
- 如果后面还有空间：`chinese_boost`
- 如果你确认平台镜像很稳再扩：`base_mirror_safe`

## 8. 不要现在做的事

- 不要一上来下载太多 shard
- 不要先补多模态
- 不要跳过 smoke run
- 不要一开始就把 `max_length` 顶到 `2048`
