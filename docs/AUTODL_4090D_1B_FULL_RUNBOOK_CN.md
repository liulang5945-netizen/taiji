# 态极 1B 在 AUTODL 4090D 单卡环境的完整执行手册

本文档面向这套目标环境：

- GPU: `RTX 4090D 24G x1`
- PyTorch: `2.3.0`
- Python: `3.12`
- CUDA: `12.1`
- System Disk: `30G`
- Data Disk: `50G`

目标不是“理论可行”，而是让你按这份文档直接上机，把态极的 `1B` 基座训练链路完整跑起来。

---

## 1. 先说结论

这套环境适合：

1. 扩充首批 `1B` 基座语料
2. 重练 `native-v2` 文本词表
3. 跑 `smoke run`
4. 跑第一轮单卡 `stage1` 预训练

这套环境不适合：

1. 一次性下载过多大 shard
2. 高密度 checkpoint
3. 激进的 `2048` 长上下文长时间硬顶
4. 同时做大规模多模态训练

所以正确策略是：

1. 先把文本基座做稳
2. 先以 `1024` 上下文跑通全链路
3. 英文优先补料
4. 多模态单独放到第二阶段

---

## 2. 目录与磁盘规划

你这套环境真正容易出问题的不是 CUDA，而是磁盘。

建议你在 AUTODL 上统一放到数据盘，例如 `/root/autodl-tmp`：

```bash
cd /root/autodl-tmp
git clone https://github.com/liulang5945-netizen/taiji.git
cd taiji
```

建议约束：

- 项目仓库：`5G` 以内
- raw 下载缓存：`10G - 20G`
- 标准化 JSONL：`5G - 15G`
- tokenizer 语料与中间件：`2G - 5G`
- checkpoint 与最终模型：`10G - 20G`

结论：

- `50G` 数据盘够你跑第一轮
- 但必须分阶段下载、分阶段清理
- 不要一上来把所有英文中文 shard 全灌进去

---

## 3. 环境初始化

### 3.1 安装依赖

```bash
cd /root/autodl-tmp/taiji
pip install sentencepiece huggingface_hub pyarrow tqdm numpy tensorboard
pip install --no-deps -e .
```

说明：

- `--no-deps` 是为了避免把桌面相关依赖也装进训练环境

### 3.2 可选：HuggingFace 镜像

如果你所在网络对 HF 不稳定，可以先设镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

如果有需要登录的源，再执行：

```bash
huggingface-cli login
```

---

## 4. 先检查机器状态

```bash
nvidia-smi
df -h
python -V
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

你要确认四件事：

1. GPU 被 PyTorch 正常识别
2. CUDA 版本正常
3. 数据盘剩余空间足够
4. 当前没有别的训练任务占显存

---

## 5. 数据下载总策略

这台机器不要一次性贪大。

推荐顺序：

1. `stage0_smoke`
2. `english_boost`
3. 审计
4. `chinese_boost`
5. 再审计
6. 重练 tokenizer
7. smoke run
8. 正式训练

当前项目已经支持这些预设：

- `stage0_smoke`
- `english_boost`
- `chinese_boost`
- `stage1_base_default`
- `base_full_text`

---

## 6. Stage0：先下载能支持 tokenizer 和 smoke 的最小集

```bash
cd /root/autodl-tmp/taiji
python scripts/data_prep/download_pretrain_mix_v1.py \
  --preset stage0_smoke \
  --max-records-per-source 100000
```

这一步会优先处理：

- `fineweb_edu`
- `skypile_zh`
- `openwebmath`
- `codeparrot_code`

执行后检查：

```bash
ls -lh taiji_data/training_data/pretrain_mix_v1/
cat taiji_data/training_data/pretrain_mix_v1/manifest.json
```

---

## 7. 第二步先补英文

英文现在是态极 `1B` 基座里最明显的结构性短板之一，所以优先补英文。

```bash
python scripts/data_prep/download_pretrain_mix_v1.py \
  --preset english_boost \
  --shards-per-source 1
```

说明：

- 这一步会优先处理 `fineweb_edu + fineweb2_en`
- `fineweb2_en` 是更现代的英文网页主料
- 如果下载压力明显升高，可以先只做 dry run

如果你的平台更依赖镜像，可以改用这个预设：

```bash
python scripts/data_prep/download_pretrain_mix_v1.py \
  --preset english_boost_mirror \
  --shards-per-source 1
```

它会把 `falcon_refinedweb_en` 一起纳入英文补强。

```bash
python scripts/data_prep/download_pretrain_mix_v1.py \
  --preset english_boost \
  --shards-per-source 1 \
  --dry-run
```

---

## 8. 跑一次审计，确认缺口缩了多少

```bash
python scripts/data_prep/audit_1b_training_assets.py
```

输出文件：

- [audit_1b_training_assets.json](/E:/taiji/taiji_data/training_data/reports/audit_1b_training_assets.json)
- [1B_DATA_GAP_REPORT_CN.md](/E:/taiji/docs/1B_DATA_GAP_REPORT_CN.md)

你要重点看：

1. `total_estimated_tokens`
2. `english_general`
3. `base_token_gap_min`
4. `english_token_gap_recommended`

---

## 9. 如磁盘和网络允许，再补中文第二批

```bash
python scripts/data_prep/download_pretrain_mix_v1.py \
  --preset chinese_boost \
  --shards-per-source 1
```

说明：

- 这一步是 `skypile_zh + fineweb2_zh`
- `fineweb2_zh` shard 可能更大
- 如果磁盘紧张，优先保英文，再慢慢扩中文

补完后再审计一次：

```bash
python scripts/data_prep/audit_1b_training_assets.py
```

---

## 10. 重练 native-v2 文本词表

### 10.1 构建词表语料

```bash
python scripts/build_native_vocab_corpus.py \
  --data-dir taiji_data/training_data/pretrain_mix_v1 \
  --data-dir taiji \
  --output taiji_data/tokenizer/native_v2_corpus.txt
```

### 10.2 训练 SentencePiece

```bash
python scripts/train_native_text_sp.py \
  --corpus taiji_data/tokenizer/native_v2_corpus.txt \
  --output-dir taiji/tokenizer_native_v2 \
  --contract taiji/tokenizer_contract.json
```

### 10.3 验证 tokenizer

```bash
python scripts/verify_native_tokenizer.py \
  --tokenizer-dir taiji/tokenizer_native_v2 \
  --contract taiji/tokenizer_contract.json
```

通过标准：

1. `text_offset=13388`
2. `total_vocab_size=256000`
3. `SentencePiece` 大小不超过文本词表空间

---

## 11. smoke run：先把链路跑通

不要一上来就正式长训，先做 `200 step` 验证。

```bash
python scripts/native_v2/pretrain.py \
  --data_dir taiji_data/training_data/pretrain_mix_v1 \
  --tokenizer-dir taiji/tokenizer_native_v2 \
  --output taiji_data/taiji_pretrained_1b_smoke \
  --max_steps 200 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --learning_rate 3e-4 \
  --log_every 10 \
  --save_every 100 \
  --num_workers 2
```

你要观察：

1. loss 是否下降
2. 是否出现 `vocab / offset / shape` 报错
3. checkpoint 是否正常写出
4. 显存是否稳定

如果这一阶段都不稳，不要进入正式训练。

---

## 12. 正式 1B stage1：先按保守单卡参数跑

对 `4090D 24G x1`，我建议第一轮先用更保守、可复现的参数：

```bash
python scripts/native_v2/pretrain.py \
  --data_dir taiji_data/training_data/pretrain_mix_v1 \
  --tokenizer-dir taiji/tokenizer_native_v2 \
  --output taiji_data/taiji_pretrained_1b_stage1 \
  --max_steps 50000 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_length 1024 \
  --learning_rate 3e-4 \
  --min_learning_rate 3e-5 \
  --warmup_steps 1000 \
  --log_every 50 \
  --save_every 10000 \
  --num_workers 2
```

为什么先不主推 `2048`：

1. `24G` 单卡空间紧
2. 你现在最重要的是把一版 1B 稳定训出来
3. `1024` 更适合当前阶段验证数据混合、词表和训练链路

如果 smoke 与短程正式训练都稳定，再试：

```bash
--max_length 2048
--gradient_accumulation_steps 8
```

但不要一开始就默认它一定稳。

---

## 13. 后台运行方式

如果你不想终端断开后任务停掉，可以这样：

```bash
nohup python scripts/native_v2/pretrain.py \
  --data_dir taiji_data/training_data/pretrain_mix_v1 \
  --tokenizer-dir taiji/tokenizer_native_v2 \
  --output taiji_data/taiji_pretrained_1b_stage1 \
  --max_steps 50000 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_length 1024 \
  --learning_rate 3e-4 \
  --min_learning_rate 3e-5 \
  --warmup_steps 1000 \
  --log_every 50 \
  --save_every 10000 \
  --num_workers 2 \
  > taiji_data/train_stage1.log 2>&1 &
```

查看日志：

```bash
tail -f taiji_data/train_stage1.log
```

查看 GPU：

```bash
watch -n 1 nvidia-smi
```

---

## 14. 磁盘清理策略

这部分很重要。

### 可以清的

1. 已经完成标准化且暂时不再需要的 raw shard
2. 过密的旧 checkpoint
3. 临时构建文件

### 不要乱清的

1. 当前正在训练所依赖的 JSONL
2. 最新 checkpoint
3. 当前 tokenizer 目录
4. 当前训练日志

### 推荐做法

在确认标准化后的 `jsonl` 已经完整可用后，再决定是否删除 raw 目录。

例如只查看，不直接删：

```bash
du -sh taiji_data/training_data/raw_pretrain_mix_v1
du -sh taiji_data/training_data/pretrain_mix_v1
du -sh taiji_data/taiji_pretrained_1b_stage1
```

---

## 15. 多模态现在怎么处理

现在不要把多模态混进这轮 base pretrain。

原因很明确：

- 本地多模态标注量仍然太少
- 当前主矛盾是 1B 文本基座
- 现在混训，只会同时伤到文本和多模态效果

正确顺序：

1. `base_pretrain`
2. `multimodal_alignment`
3. `conversation_sft`
4. `agent_tool_sft`

---

## 16. 训练中遇到问题时怎么判断

### OOM

先降：

1. `max_length`
2. `gradient_accumulation_steps`
3. `num_workers`

首选处理：

```bash
--max_length 1024
```

### 下载太慢或失败

先：

1. 开 `--dry-run`
2. 先跑 `stage0_smoke`
3. 再跑 `english_boost`
4. 最后视空间决定是否跑 `chinese_boost`

### 磁盘快满

先检查：

```bash
df -h
du -sh taiji_data/*
```

优先处理：

1. raw shard
2. 旧 checkpoint
3. 非当前阶段的中间文件

---

## 17. 我建议你的实际上机顺序

直接照这个顺序做：

```bash
cd /root/autodl-tmp
git clone https://github.com/liulang5945-netizen/taiji.git
cd taiji

pip install sentencepiece huggingface_hub pyarrow tqdm numpy tensorboard
pip install --no-deps -e .

python scripts/data_prep/download_pretrain_mix_v1.py --preset stage0_smoke --max-records-per-source 100000
python scripts/data_prep/download_pretrain_mix_v1.py --preset english_boost --shards-per-source 1
python scripts/data_prep/audit_1b_training_assets.py

python scripts/build_native_vocab_corpus.py \
  --data-dir taiji_data/training_data/pretrain_mix_v1 \
  --data-dir taiji \
  --output taiji_data/tokenizer/native_v2_corpus.txt

python scripts/train_native_text_sp.py \
  --corpus taiji_data/tokenizer/native_v2_corpus.txt \
  --output-dir taiji/tokenizer_native_v2 \
  --contract taiji/tokenizer_contract.json

python scripts/verify_native_tokenizer.py \
  --tokenizer-dir taiji/tokenizer_native_v2 \
  --contract taiji/tokenizer_contract.json

python scripts/native_v2/pretrain.py \
  --data_dir taiji_data/training_data/pretrain_mix_v1 \
  --tokenizer-dir taiji/tokenizer_native_v2 \
  --output taiji_data/taiji_pretrained_1b_smoke \
  --max_steps 200 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --learning_rate 3e-4 \
  --log_every 10 \
  --save_every 100 \
  --num_workers 2
```

如果 smoke run 正常，再跑正式训练：

```bash
python scripts/native_v2/pretrain.py \
  --data_dir taiji_data/training_data/pretrain_mix_v1 \
  --tokenizer-dir taiji/tokenizer_native_v2 \
  --output taiji_data/taiji_pretrained_1b_stage1 \
  --max_steps 50000 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_length 1024 \
  --learning_rate 3e-4 \
  --min_learning_rate 3e-5 \
  --warmup_steps 1000 \
  --log_every 50 \
  --save_every 10000 \
  --num_workers 2
```

---

## 18. 最后判断

如果让我来做，我会把这台机器的任务定义成：

**不是“直接把最完美的态极 1B 一口气训完”，而是“把一版真正可用、结构正确、英文不至于太弱的 1B 基座稳定做出来”。**

这才是这台 `4090D 24G x1` 机器最合理、性价比最高的使用方式。
