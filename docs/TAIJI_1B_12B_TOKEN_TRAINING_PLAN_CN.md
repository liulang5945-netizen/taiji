# 态极 1B：12B tokens 训练规划（词表、预训练、微调）

这份文档给人和其他 AI 执行用。不要自由解释这里的口径：

- `12B` 指 **约 120 亿个 tokenizer 后的训练 token**，不是 12B 条样本，也不是 12GB 数据。
- 目标模型是态极 `1B native-v2` 路线。
- 词表只重练 **文本 SentencePiece**，不推翻 `taiji/tokenizer_contract.json` 的全局 ID 空间。
- 预训练结束后才能做 SFT/Agent/多模态对齐；不要把 SFT 当成 base pretrain 的主料。
- 当前仓库已有单卡入口能支持 smoke 和分阶段训练，但完整 12B 长跑前，最好补齐 streaming dataset、resume 和验证集评估。

---

## 0. 当前项目事实

相关文件：

- 全局词表契约：`taiji/tokenizer_contract.json`
- 词表训练：`scripts/train_native_text_sp.py`
- 词表语料构建：`scripts/build_native_vocab_corpus.py`
- 词表验证：`scripts/verify_native_tokenizer.py`
- 当前单卡预训练入口：`scripts/native_v2/pretrain.py`
- 当前数据下载入口：`scripts/data_prep/download_pretrain_mix_v1.py`

当前 `native-v2` 词表契约：

```text
total_vocab_size = 256000
text_offset      = 13388
text_vocab_size  = 242612
text range       = [13388, 255999]
image range      = [1000, 9191]
audio range      = [9192, 13287]
mm control range = [13288, 13387]
```

结论：

1. 可以重新训练 `sentencepiece.model`。
2. 不要改 `<pad>/<unk>/<s></s>`、态极特殊 token、多模态 token 的固定 ID。
3. 一旦开始预训练，tokenizer 必须冻结，后面不能再换词表继续训练。

---

## 1. 12B 数据总体配比

12B tokens 建议按“文本基座优先，兼顾英文、中文、代码、数学”的结构做。

| 类别 | token 目标 | 占比 | 主要用途 | 当前优先数据源 |
| --- | ---: | ---: | --- | --- |
| 英文通用/教育网页 | 3.0B | 25% | 英文基础表达、世界知识、技术文档阅读 | `fineweb_edu`, `fineweb2_en`, `falcon_refinedweb_en` |
| 中文通用网页 | 3.0B | 25% | 中文表达、中文常识、国内语境 | `skypile_zh`, `fineweb2_zh` |
| 代码 | 1.2B | 10% | Python/JS/工程语料、代码补全基础 | `codeparrot_code`，后续可加 The Stack 类数据但要先看 license |
| 数学/科学 | 1.0B | 8.3% | 数学语言、公式、推理文本 | `openwebmath` |
| 高质量百科/文档/书面知识 | 1.2B | 10% | 降低网页噪声，提高知识密度 | 可从公开 wiki/docs 数据补充 |
| 双语技术/开发者文档 | 0.8B | 6.7% | 让态极更懂软件、API、英文技术材料 | 官方文档、开源文档、技术文章 |
| 态极原生/工具/记忆/规划文本 | 0.3B | 2.5% | 让特殊 token 和态极行为格式不陌生 | 项目内 docs、工具调用样例、合成 trace |
| 多语言少量补充 | 0.5B | 4.2% | 提升 tokenizer 和泛化，不追求多语强能力 | FineWeb2 其他高质量语言切片 |
| 高质量 replay/验证后补料 | 1.0B | 8.3% | 最后阶段强化干净样本，修正偏科 | 从上面各类中筛选 loss/质量更好的样本 |

执行要求：

- 统计时以 `sentencepiece.model` 编码后的 token 数为准。
- 允许每类误差 `±10%`，但英文和中文都不要低于 `2.5B tokens`。
- 代码数据不要超过 `15%`，否则小模型容易聊天变硬、自然语言退化。
- 态极原生格式不要超过 `5%`，否则会过早学会“标签腔”。
- 多模态不要混进 12B 文本基座主训练，先保留 token 空间和少量文本占位，真正图文/音文放到后续 alignment。

---

## 2. 数据落盘与审计目录

在 AutoDL 这类 50G 数据盘环境，不要试图让 12B 全部常驻本地。正确做法是“下载一批、清洗一批、训练/打包一批、删 raw 缓存”。

推荐目录：

```text
/root/autodl-tmp/taiji/
  taiji_data/
    training_data/
      pretrain_12b/
        raw/              # 临时 raw shard，可删
        normalized/       # 标准 jsonl，可按批保留
        tokenizer_sample/ # 训练词表用的平衡采样
        holdout/          # 固定验证集，不能参与训练
        reports/          # token 审计、质量报告
```

标准 JSONL 格式：

```json
{"text": "...", "source": "fineweb2_en", "category": "english_web", "language": "en"}
```

不要把下载缓存、parquet 元数据、manifest、checkpoint 混进训练目录。训练入口只应该看到 `.jsonl/.json/.txt` 训练文件。

---

## 3. 数据下载顺序

先跑小闭环，再扩大，不要一上来拉满 12B。

### 3.1 Stage0：链路验证数据

目标：只够 tokenizer 初版、smoke run、格式验证。

```bash
cd /root/autodl-tmp/taiji
export HF_ENDPOINT=https://hf-mirror.com

python scripts/data_prep/download_pretrain_mix_v1.py \
  --preset stage0_smoke \
  --shards-per-source 1 \
  --max-records-per-source 100000 \
  --output-dir taiji_data/training_data/pretrain_12b/normalized/stage0
```

通过标准：

- 每个源都有 JSONL 输出。
- 没有空文件。
- 任意抽样 100 条，`text` 字段可读。

### 3.2 Stage1：补足英文和中文主料

英文优先，因为态极不能只会中文。

```bash
python scripts/data_prep/download_pretrain_mix_v1.py \
  --preset english_boost_mirror \
  --shards-per-source 1 \
  --max-records-per-source 500000 \
  --output-dir taiji_data/training_data/pretrain_12b/normalized/en_batch_001
```

中文第二批：

```bash
python scripts/data_prep/download_pretrain_mix_v1.py \
  --preset chinese_boost \
  --shards-per-source 1 \
  --max-records-per-source 500000 \
  --output-dir taiji_data/training_data/pretrain_12b/normalized/zh_batch_001
```

如果网络卡在 HuggingFace 文件列表：

1. 保留已下载 shard。
2. 降低 `--shards-per-source` 到 `1`。
3. 单独下载更容易成功的源，比如 `falcon_refinedweb_en` 或 `skypile_zh`。
4. 不要让一个卡住的源阻塞全部训练。

### 3.3 Stage2：按 token 缺口循环补料

每轮补料后都做 token 审计。不要靠“文件大小”和“样本条数”估算最终规模。

执行循环：

```text
下载 1-3 个 shard
-> 标准化为 JSONL
-> 清洗/去重
-> 用当前 tokenizer 估算 token
-> 更新 reports/token_budget.json
-> raw 空间紧张时删除 raw，保留 normalized 或转移到对象存储
```

每轮补料目标：

- 小盘环境：每批 `100M-300M tokens`
- 稳定大盘环境：每批 `500M-1B tokens`
- 累计到 `12B tokens` 后冻结数据 manifest

---

## 4. 清洗规则

最低清洗规则：

- 删除长度 `<64 chars` 的短文本。
- 删除重复行和完全重复文档。
- 删除乱码、模板页、导航页、cookie/banner、SEO 关键词堆叠页。
- 删除大量重复 n-gram 的文本。
- 删除明显 PII、密钥、token、邮箱/手机号堆叠样本。
- 保留代码时要保留换行和缩进。
- 数学语料保留 LaTeX、公式符号，不要粗暴清洗成纯文本。

建议阈值：

```text
min_chars = 64
max_chars = 20000
max_repeated_line_ratio = 0.3
max_same_char_run = 80
dedup_key = normalized_text_sha1
```

训练集和验证集必须分开去重，holdout 不能泄漏到训练集。

---

## 5. 词表训练规划

### 5.1 是否需要重练词表

需要。但只重练 `sentencepiece.model`，不改 `tokenizer_contract.json`。

原因：

- 态极当前有固定的特殊 token、多模态 token 和文本 offset。
- 12B 预训练前，文本词表应该覆盖中文、英文、代码、数学。
- 如果预训练开始后再换词表，已有 checkpoint 基本不能直接继承。

### 5.2 词表语料规模

词表不需要吃完整 12B tokens。它需要的是“平衡、多样、足够大”的采样。

推荐：

```text
最低可用：500M chars
推荐规模：1B-3B chars
上限建议：5B chars，再大收益变小且训练慢
```

词表采样配比：

| 类别 | 字符采样占比 |
| --- | ---: |
| 中文通用 | 30% |
| 英文通用 | 30% |
| 代码 | 15% |
| 数学/科学 | 10% |
| 技术文档 | 10% |
| 态极特殊格式/工具调用/多模态占位文本 | 5% |

### 5.3 构建词表语料

优先使用强制配比脚本。不要让辅助 AI 自己“理解并采样”，它很容易被中文大文件或英文大文件吞掉配额。

推荐命令：

```bash
python scripts/data_prep/generate_taiji_special_vocab_corpus.py \
  --output taiji_data/training_data/pretrain_12b/tokenizer_sample/taiji_special_vocab.jsonl \
  --records 20000

python scripts/build_balanced_native_vocab_corpus.py \
  --normalized-dir taiji_data/training_data/pretrain_12b/normalized \
  --project-dir . \
  --output taiji_data/training_data/pretrain_12b/tokenizer_sample/native_v2_corpus_balanced.txt \
  --report taiji_data/training_data/pretrain_12b/reports/tokenizer_corpus_balanced_report.json \
  --max-chars 2000000000 \
  --strict
```

这个脚本会强制按下面配额构建：

```text
zh=30%, en=30%, code=15%, math=10%, tech=10%, taiji_special=5%
```

如果 `code/math/tech/taiji_special` 不足，脚本会在 report 里明确写出缺口；加了 `--strict` 时，任意类别低于 90% 配额会直接失败。辅助 AI 不能忽略这个失败。

正式训练 SentencePiece 时，优先使用 balanced corpus：

```bash
python scripts/train_native_text_sp.py \
  --corpus taiji_data/training_data/pretrain_12b/tokenizer_sample/native_v2_corpus_balanced.txt \
  --output-dir taiji/tokenizer_native_v2 \
  --contract taiji/tokenizer_contract.json \
  --model-type bpe
```

旧脚本仍可用于快速链路验证：

```bash
python scripts/build_native_vocab_corpus.py \
  --data-dir taiji_data/training_data/pretrain_12b/normalized \
  --data-dir taiji \
  --output taiji_data/training_data/pretrain_12b/tokenizer_sample/native_v2_corpus.txt \
  --max-chars 2000000000
```

如果磁盘不足，把 `--max-chars` 降到 `800000000`，但仍然优先用 balanced 脚本。

### 5.4 训练 SentencePiece

```bash
python scripts/train_native_text_sp.py \
  --corpus taiji_data/training_data/pretrain_12b/tokenizer_sample/native_v2_corpus.txt \
  --output-dir taiji/tokenizer_native_v2 \
  --contract taiji/tokenizer_contract.json \
  --model-type bpe
```

除非有强理由，不要随便改：

```text
vocab_size      = contract["text_vocab_size"] = 242612
byte_fallback   = true
character_coverage = 0.9999
normalization_rule_name = identity
```

### 5.5 验证词表

```bash
python scripts/verify_native_tokenizer.py \
  --tokenizer-dir taiji/tokenizer_native_v2 \
  --contract taiji/tokenizer_contract.json
```

额外验收：

- 所有特殊 token 都是单 token。
- `<image>`, `<audio>`, `<tool_call>`, `<final_answer>` 等 ID 与 contract 一致。
- 中文、英文、代码、数学 holdout 都能 encode/decode。
- 新词表在 holdout 上不能比旧词表 token 数恶化超过 `5%`。
- tokenizer 训练完成后立刻备份：

```bash
cp -r taiji/tokenizer_native_v2 taiji/tokenizer_native_v2_12b_frozen
```

---

## 6. 预训练 token 与步数换算

公式：

```text
每个 optimizer step 消耗 token 数 =
batch_size * gradient_accumulation_steps * max_length
```

当前 4090D 24G 单卡保守配置：

```text
batch_size = 1
gradient_accumulation_steps = 32
max_length = 2048
tokens_per_step = 65536
```

12B tokens 对应步数：

```text
12,000,000,000 / 65,536 ≈ 183,106 optimizer steps
```

如果先用 `1024` 上下文跑 1B tokens warmup：

```text
Stage A: 1B tokens, seq=1024, accum=32 -> 约 30,518 steps
Stage B: 8B tokens, seq=2048, accum=32 -> 约 122,071 steps
Stage C: 3B tokens, seq=2048, accum=32 -> 约 45,777 steps
合计约 198,366 optimizer steps
```

---

## 7. 预训练阶段设计

### 7.1 Smoke run

目的：只验证链路，不追求效果。

```bash
python scripts/native_v2/pretrain.py \
  --data_dir taiji_data/training_data/pretrain_12b/normalized/stage0 \
  --tokenizer-dir taiji/tokenizer_native_v2 \
  --output taiji_data/checkpoints/taiji_1b_smoke \
  --max_steps 200 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --learning_rate 3e-4 \
  --log_every 10 \
  --save_every 100 \
  --num_workers 2
```

通过标准：

- 不 OOM。
- loss 有正常数值，不是 `nan/inf`。
- checkpoint 能保存。
- `sentencepiece.model` 和 `tokenizer_contract.json` 被写入 checkpoint。

### 7.2 Stage A：1B tokens warmup

目的：让模型先学稳定的中英文、标点、短文结构。

数据：

- 高质量英文 `40%`
- 高质量中文 `40%`
- 数学/代码/文档 `20%`

命令模板：

```bash
python scripts/native_v2/pretrain.py \
  --data_dir taiji_data/training_data/pretrain_12b/normalized/stage_a \
  --tokenizer-dir taiji/tokenizer_native_v2 \
  --output taiji_data/checkpoints/taiji_1b_stage_a \
  --max_steps 30518 \
  --batch_size 1 \
  --gradient_accumulation_steps 32 \
  --max_length 1024 \
  --learning_rate 3e-4 \
  --min_learning_rate 2e-4 \
  --warmup_steps 1000 \
  --log_every 50 \
  --save_every 3000 \
  --num_workers 2
```

### 7.3 Stage B：8B tokens main pretrain

目的：主力学习语言、知识、代码、数学。

数据按第 1 节 12B 配比滚动喂入。

命令模板：

```bash
python scripts/native_v2/pretrain.py \
  --data_dir taiji_data/training_data/pretrain_12b/normalized/stage_b \
  --tokenizer-dir taiji/tokenizer_native_v2 \
  --output taiji_data/checkpoints/taiji_1b_stage_b \
  --max_steps 122071 \
  --batch_size 1 \
  --gradient_accumulation_steps 32 \
  --max_length 2048 \
  --learning_rate 3e-4 \
  --min_learning_rate 5e-5 \
  --warmup_steps 2000 \
  --log_every 50 \
  --save_every 5000 \
  --num_workers 2
```

### 7.4 Stage C：3B tokens quality replay

目的：最后用更干净的数据修正噪声和偏科。

数据：

- 高质量中文/英文各 `30%`
- 数学/代码/技术文档合计 `30%`
- 态极格式、工具、规划、记忆文本 `10%`

命令模板：

```bash
python scripts/native_v2/pretrain.py \
  --data_dir taiji_data/training_data/pretrain_12b/normalized/stage_c_quality \
  --tokenizer-dir taiji/tokenizer_native_v2 \
  --output taiji_data/checkpoints/taiji_1b_stage_c \
  --max_steps 45777 \
  --batch_size 1 \
  --gradient_accumulation_steps 32 \
  --max_length 2048 \
  --learning_rate 1e-4 \
  --min_learning_rate 3e-5 \
  --warmup_steps 500 \
  --log_every 50 \
  --save_every 5000 \
  --num_workers 2
```

注意：当前 `scripts/native_v2/pretrain.py` 每次都是新建模型训练。完整分阶段续训前，需要让脚本支持 `--resume_from_checkpoint`，否则 Stage B/C 不能继承 Stage A。没有 resume 时，不要假装已经完成 12B 连续预训练。

---

## 8. 12B 长跑前必须补的工程能力

当前仓库可以 smoke，但 12B 长跑需要更稳。

必须补：

1. `--resume_from_checkpoint`
   - 加载 `model.pt`
   - 加载 optimizer/scheduler/scaler 状态
   - 恢复 step

2. streaming dataset
   - 不要把所有 JSONL 行一次性读进内存。
   - 支持按 shard 迭代、随机 buffer shuffle、断点恢复。

3. 固定 holdout eval
   - `holdout_zh.jsonl`
   - `holdout_en.jsonl`
   - `holdout_code.jsonl`
   - `holdout_math.jsonl`

4. checkpoint 清理
   - 50G 盘只保留 `last-3 + best + final`。
   - 每个 checkpoint 保存前先检查剩余空间。

5. token 进度统计
   - 日志必须打印 `tokens_seen`。
   - 不要只打印 step。

没有这 5 个能力，12B 训练很容易中途断了无法恢复，或者训练了多少 token 都说不清。

---

## 9. 预训练验收标准

每 `100M-300M tokens` 做一次检查：

- loss 平滑下降，没有持续 `nan/inf`。
- 中文/英文生成都不乱码。
- 不出现长段重复 token。
- 特殊 token 不乱吐，除非 prompt 要求工具/规划格式。
- holdout perplexity 至少在大趋势下降。
- 代码样例能保持缩进、括号、字符串结构。
- 数学文本能保持基本公式和推导格式。

最小人工验收 prompt：

```text
中文：请解释什么是梯度下降，并举一个生活例子。
英文：Explain why attention mechanisms help language models.
代码：写一个 Python 函数，读取 JSONL 并统计 text 字段平均长度。
数学：解释贝叶斯公式，并给出一个简单计算。
态极：你是谁？你如何使用工具和记忆？
```

---

## 10. 微调规划

SFT 不是越大越好。对 1B 模型，建议先做 `50M-200M SFT tokens`，比堆脏数据更重要的是质量。

### 10.1 SFT 数据配比

| 阶段 | token 目标 | 内容 |
| --- | ---: | --- |
| SFT-A 通用助手 | 60M | 中英问答、写作、总结、翻译、解释 |
| SFT-B 代码/数学 | 30M | 代码生成、代码解释、基础数学推理 |
| SFT-C 态极人格/本地 AI/生命系统 | 10M-20M | 身份、记忆、吃饭睡觉玩耍、用户关系 |
| SFT-D 工具/ReAct/Agent | 20M-40M | `<tool_call>`, `<tool_result>`, 规划、错误恢复 |
| SFT-E 安全/拒答/边界 | 10M | 安全边界、隐私、本地数据保护 |
| SFT-F 多模态文本指令占位 | 5M-10M | `<image>`, `<audio>`, `<screen>` 的文本级格式熟悉 |

总量先控制在 `100M tokens` 左右。效果不够再扩到 `200M`，不要第一轮就上 1B SFT tokens。

### 10.2 SFT 格式

统一使用 messages：

```json
{"messages":[{"role":"system","content":"你是态极，一个本地运行的 AI。"},{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"category":"general_zh"}
```

工具调用格式：

```text
<tool_call>{"name":"search","args":{"query":"..."}}</tool_call>
<tool_result>...</tool_result>
<final_answer>...</final_answer>
```

不要让模型在普通问答里频繁输出 `<think>`。内部思考标签只能少量训练，否则用户体验会变差。

### 10.3 SFT 训练参数

全量微调建议：

```text
max_seq_len = 1024 或 2048
learning_rate = 1e-5 到 2e-5
batch_size = 1
gradient_accumulation_steps = 16 到 32
warmup_steps = 200 到 1000
epochs = 1 到 3
```

如果显存紧张，优先用 LoRA/QLoRA；如果当前项目还没有 LoRA 入口，先用短程全量 SFT 验证。

SFT 验收：

- 中文自然，不像模板。
- 英文可用，不只会中文。
- 工具调用 JSON 合法率高。
- 不该调用工具时不会硬调用。
- 态极身份稳定，但不过度自我介绍。
- 多轮对话不快速遗忘上下文。

---

## 11. 多模态路线

态极已经在词表契约里预留图像、音频和多模态控制 token，但这不等于模型已经具备多模态能力。

正确顺序：

1. 文本基座 12B pretrain。
2. 文本 SFT，让模型会正常对话和工具使用。
3. 图文/音文 alignment，训练 projector 或适配层。
4. 多模态指令 SFT。

最小多模态数据量：

```text
图文 caption/问答：至少 100k 对，推荐 500k-1M 对
屏幕理解/软件 UI：至少 20k-50k 对
音频/语音：先放后，不要和第一版 1B 混在一起
```

第一版 1B 可以先做到：

- 看得懂 `<image>...</image>` 这类文本占位格式。
- 不要声称自己已经能真正看图/听音频。
- 真正多模态能力留到 alignment 阶段。

---

## 12. 给其他 AI 的强制执行规则

如果你是接手执行的 AI，请严格遵守：

1. 不要把 raw bytes、样本条数、文件大小当成 token 数。
2. 不要改 `tokenizer_contract.json` 的 ID 分配。
3. 不要在预训练开始后重练 tokenizer。
4. 不要把少量 SFT 数据混成 base pretrain 主料。
5. 不要声称已经完成 12B，除非 token 审计报告显示 `tokens_seen >= 12,000,000,000`。
6. 不要跳过 smoke run。
7. 不要在没有 resume 的情况下进行长跑。
8. 不要忽略英文；英文目标至少 `2.5B tokens`。
9. 不要把“多模态 token 预留”误解成“已经完成多模态训练”。
10. 每个阶段结束必须产出 manifest、token report、checkpoint 和人工样例输出。

---

## 13. 最终交付物清单

完整 12B 训练完成后，目录里至少应有：

```text
taiji/tokenizer_native_v2_12b_frozen/
  sentencepiece.model
  sentencepiece.vocab
  tokenizer_contract.json

taiji_data/training_data/pretrain_12b/reports/
  manifest_12b.json
  token_budget_12b.json
  quality_report.json
  holdout_report.json

taiji_data/checkpoints/taiji_1b_pretrain_12b/
  final/
    model.pt
    config.json
    sentencepiece.model
    tokenizer_contract.json

taiji_data/checkpoints/taiji_1b_sft/
  final/
```

最终判断：

- tokenizer 验证通过。
- 预训练 token 审计达到 12B。
- 预训练 final checkpoint 可加载。
- SFT 后中英文、代码、数学、工具调用、态极身份都通过人工样例。
- 多模态只声明“预留/文本占位/待 alignment”，除非已经完成独立多模态训练。

---

## 14. 可以直接复制给其他 AI 的执行提示词

把下面这段完整复制给接手训练的 AI。不要让它自己重新解释目标。

```text
你正在协助执行“态极 1B native-v2，12B tokens 训练计划”。

强制口径：
1. 12B 指 12,000,000,000 个 tokenizer 后的训练 token，不是样本数、文件大小或下载量。
2. 必须先读 docs/TAIJI_1B_12B_TOKEN_TRAINING_PLAN_CN.md。
3. 不允许修改 taiji/tokenizer_contract.json 的 total_vocab_size、text_offset、特殊 token、多模态 token 范围。
4. 只允许重练 taiji/tokenizer_native_v2/sentencepiece.model。
5. tokenizer 一旦用于正式预训练就冻结，后续不得更换。
6. 不允许把 SFT/对话/工具调用数据当成 base pretrain 主料。
7. 不允许声称完成 12B，除非 token 审计显示 tokens_seen >= 12,000,000,000。
8. 当前 scripts/native_v2/pretrain.py 是单卡入口；长跑前必须确认支持 resume、streaming dataset、holdout eval。
9. 英文不能忽略，英文通用数据目标至少 2.5B tokens，推荐 3B tokens。
10. 多模态 token 预留不等于多模态训练完成。第一版 1B 先做文本基座，图文/音文 alignment 后置。
11. 构建正式词表语料时必须使用 scripts/build_balanced_native_vocab_corpus.py，并检查 tokenizer_corpus_balanced_report.json；不得直接用顺序拼接的 native_v2_corpus.txt 作为最终词表语料。

你每次行动前必须输出：
- 当前阶段：tokenizer / data / smoke / pretrain_stage_a / pretrain_stage_b / pretrain_stage_c / sft / eval
- 目标 token 数
- 已确认 token 数
- 本次会修改或运行的命令
- 预期输出文件

你每次行动后必须输出：
- 实际新增 token 数或处理样本数
- 当前累计 tokens_seen
- checkpoint 路径
- 是否可 resume
- 下一步阻塞项

如果缺少 token 审计脚本、resume、streaming dataset 或 holdout eval，不要假装训练计划完整；先明确指出缺口并补工程能力。
```

---

## 15. 对当前 4090D 24G / 50G 数据盘的现实判断

这台机器可以做：

- 下载和验证小批数据。
- 训练/验证 tokenizer。
- 跑 `200 steps` smoke。
- 跑短程 stage，例如 `100M-500M tokens` 的试训。
- 验证 loss、checkpoint、推理样例。

这台机器不适合直接一次性完成：

- 12B tokens 全量数据常驻。
- 长时间无断点的 180k+ optimizer steps。
- 大量 checkpoint 常驻。
- 文本和多模态同时大规模训练。

可行策略：

```text
用这台机器完成 tokenizer + smoke + 若干短程训练验证
-> 补齐 resume/streaming/eval
-> 分批滚动训练
-> raw shard 处理完就删
-> checkpoint 只保留 last/best/final
```

如果预算允许，更稳的正式 12B 训练环境：

```text
GPU: A100 80G 或 2-4 张 4090/4090D
数据盘: 至少 500G，推荐 1T+
训练框架: DDP/FSDP/DeepSpeed 之一
必须能力: resume、streaming、eval、checkpoint rotation
```

---

## 16. 最短可执行路线

如果现在目标是尽快把态极 1B 跑起来，不要等待 12B 全量齐备才动手。

最短路线：

1. 先用当前数据训练/验证 tokenizer。
2. 跑 `200 steps` smoke。
3. 补英文和中文各一批。
4. 做 token 审计，确认累计量。
5. 补 `resume_from_checkpoint`。
6. 跑 `100M tokens` 短程预训练。
7. 人工测试中英、代码、数学、态极身份。
8. 确认链路稳定后，再滚动扩到 `1B -> 3B -> 6B -> 12B tokens`。

这样做的好处是每一步都能产出可验证资产，不会把平台钱花在“不知道是否能恢复”的长跑上。
