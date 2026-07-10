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
<final_answer>{??????????}</final_answer>
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

## 14. 本地优先原则

本计划默认采用 `本地优先`，不是 `平台优先`。

原因不是因为本地更快，而是因为当前用户已经明确接受：

- 本地训练可以更慢
- 时间成本高于平台切换和上传下载折腾，不构成否决条件
- 当前目标是先把 `1B` 这条线稳定打通，而不是追求最短 wall-clock 时间
- 数据、词表、审计、短程预训练链路优先保证可控、可复现、可反复迭代

因此，后续所有执行判断必须遵守下面这条硬约束：

```text
只要本地仍然可行，就默认继续本地执行。
```

这里的“可行”指：

- 可以完成 tokenizer 重练
- 可以完成数据清洗、补料、配比审计和平衡采样
- 可以完成 smoke run
- 可以完成短程预训练验证
- 可以分阶段滚动完成 1B 文本基座训练，即使耗时更长

下面这些理由，单独出现时都不能构成“必须上传平台”的依据：

1. 本地更慢
2. 平台更标准
3. 平台更容易复制命令
4. 平台更适合长跑
5. 远端 GPU 更强

只有在出现下面这些情况时，才允许把平台列为 `必要路线` 而不是 `可选加速器`：

1. 本地显存客观无法完成当前阶段，即使已经缩小 batch、缩短序列长度、采用梯度累计后仍然不可运行
2. 本地磁盘无法容纳当前必要数据与 checkpoint，而且通过分批滚动、删 raw shard、保留 last/best/final 后仍然无解
3. 本地训练脚本客观依赖远端环境中的特定能力，本地无法提供等价替代
4. 用户再次明确要求改回平台优先

换句话说：

```text
本地更慢 != 本地不可行
```

当前态极 1B 这条线的默认路线应当理解为：

```text
本地 tokenizer
-> 本地数据补齐与审计
-> 本地 smoke run
-> 本地短程预训练
-> 本地滚动扩展
-> 只有在客观资源卡死时，才切到平台
```

---

## 15. 可以直接复制给其他 AI 的执行提示词

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
12. 默认本地优先。只要本地仍然可行，就不允许把“上传平台”当成默认建议。
13. “本地更慢”不是“本地不可行”的同义词，禁止混淆这两件事。
14. 当前用户已经明确接受更长的本地训练时间，因此不能以“平台更快”为唯一理由改变路线。
15. 除非出现显存、磁盘、恢复能力等客观不可行条件，否则平台只能作为可选加速器，而不是强制路线。

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

## 16. 对当前 4090D 24G / 50G 数据盘的现实判断

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

但这段现实判断不能被误读成：

```text
“既然平台更适合长跑，所以现在就必须上传平台。”
```

正确理解是：

- 平台环境更适合未来的大规模长跑
- 但当前阶段仍然允许、甚至优先采用本地路线
- 当前用户已接受本地耗时增加，所以“时间更长”本身不是路线切换理由

---

## 17. 最短可执行路线

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

---

## 18. 本地重练 tokenizer 的直接命令

如果现在决定先在本地把 `native-v2` 词表重练好，再考虑后续长程预训练，直接使用下面这条命令：

```bash
python scripts/training/run_local_native_vocab.py --strict
```

这条命令会按顺序自动执行：

1. 生成本地技术文档补充语料  
   输出到：`taiji_data/tokenizer/local_vocab_sources/tech_docs_vocab.jsonl`
2. 生成态极特殊格式补充语料  
   输出到：`taiji_data/tokenizer/local_vocab_sources/taiji_special_vocab.jsonl`
3. 基于本地 `taiji_data/training_data/pretrain_mix_v1` 构建平衡词表语料
4. 自动探测当前数据能支撑的最大平衡字符量，避免 `tech` 或 `taiji_special` 配额不足时反复手调
5. 训练新的 SentencePiece
6. 验证 tokenizer contract

默认输出：

```text
balanced corpus:
  taiji_data/tokenizer/native_v2_corpus_balanced_local.txt

report:
  taiji_data/training_data/reports/tokenizer_corpus_balanced_local_report.json

tokenizer:
  taiji/tokenizer_native_v2/
```

默认参数说明：

- `--max-chars 400000000`：期望最多构建 4 亿字符的平衡词表语料
- `--min-effective-chars 20000000`：如果当前本地数据连 2000 万平衡字符都凑不齐，直接失败并提示补料
- `--strict`：任一类别低于 `0.9` 配额时失败

常用变体：

```bash
python scripts/training/run_local_native_vocab.py --strict --max-chars 200000000

python scripts/training/run_local_native_vocab.py --skip-train --skip-verify
```

第二条命令只做语料生成和平衡报告，非常适合先看当前本地数据是否足够支撑新的词表训练。

---

## 19. 技术文档补料附录

这一节专门回答一个问题：

```text
如果后续 AI 说“技术文档不够”，那应该去补哪些具体数据源？
```

答案不是笼统的“去找一些官方文档”，而是优先按下面这张表执行。

### 19.1 技术文档数据源清单

| 优先级 | 类别 | 具体来源 | 用途 | 落地方式 |
| --- | --- | --- | --- | --- |
| P0 | 本地项目技术文本 | `docs/`、`scripts/`、`taiji/`、`api/`、`frontend/src/`、`tests/` | 让态极熟悉自身工程结构、训练脚本、接口和 UI 语境 | 已由 `scripts/data_prep/generate_tech_docs_vocab_corpus.py` 自动扫描 |
| P0 | 英文高质量网页技术文本 | `fineweb_edu` | 英文技术阅读、说明文、教程风格 | 已在 `download_pretrain_mix_v1.py` 中支持 |
| P0 | 英文通用网页补强 | `fineweb2_en`、`falcon_refinedweb_en` | 扩英文基础表达和技术英语覆盖 | 已在 `download_pretrain_mix_v1.py` 中支持 |
| P1 | 大规模技术/文档型多模态文本 | `HuggingFaceM4/Docmatix` | 补充说明文、教程文档、图文并存页面中的技术语言 | 先抽文本字段，不急着上图像 |
| P1 | 大规模网页图文文档语料 | `HuggingFaceM4/OBELICS` | 补充网页说明文、长段结构化文本、图文网页语气 | 第一阶段优先抽文本，不急着用图像 |
| P1 | 官方 Python 文档 | `docs.python.org` | Python 语义、标准库、工程表达 | 抓取正文，转成 JSONL |
| P1 | 官方 PyTorch 文档 | `pytorch.org/docs` | 张量、训练、优化器、分布式、AMP 等技术语言 | 抓取正文，转成 JSONL |
| P1 | 官方 FastAPI 文档 | `fastapi.tiangolo.com` | API、Schema、异步服务、文档风格 | 抓取正文，转成 JSONL |
| P1 | 官方 Vue 文档 | `vuejs.org/guide` | 前端组件、状态、组合式 API、工程表达 | 抓取正文，转成 JSONL |
| P1 | 官方 MDN 文档 | `developer.mozilla.org` | HTML/CSS/JS/Web API 通用技术语料 | 抓取正文，转成 JSONL |
| P2 | 高质量开源仓库文档 | `python/cpython`、`pytorch/pytorch`、`fastapi/fastapi`、`vuejs/core` 等仓库中的 README/docs/examples | 补充真实工程文档、注释和开发者表达 | 只抓 `README/docs/examples`，不要整仓库生吞 |

### 19.2 技术文档补料原则

1. 第一优先永远是 `本地项目真实技术文本`。  
   因为态极不是在做一个泛技术模型，而是在做“更懂自己工程”的模型。

2. 第二优先是 `英文技术文档`。  
   因为当前缺口里，英文主料比中文更紧。

3. `Docmatix / OBELICS` 这类大源，第一阶段先抽文本，不要一开始就把图像链路也扛进来。

4. 对外部开源仓库，优先抓：
   - `README`
   - `docs/`
   - `examples/`
   - `tutorials/`

   不要直接把整个源码树当“技术文档”。

5. 技术文档在 base pretrain 中建议控制在：

```text
0.8B tokens 左右为主目标
最低不要低于 0.4B
第一阶段先做到 100M-300M 的高质量技术文档增量也完全有价值
```

### 19.3 如果现在就要执行，推荐补料顺序

```text
P0: 本地项目扫描语料（已完成）
-> P0: fineweb_edu / fineweb2_en / falcon_refinedweb_en
-> P1: Docmatix 文本抽取
-> P1: OBELICS 文本抽取
-> P1: Python / PyTorch / FastAPI / Vue / MDN 官方文档正文
-> P2: 高质量开源仓库 docs/examples
```

---

## 20. 多模态补料附录

这一节专门回答另一个问题：

```text
如果后续 AI 说“多模态缺数据”，那应该优先补哪些具体数据源？
```

注意：第一版 1B 文本基座训练时，多模态数据不是主料；它们主要用于后续 `alignment / multimodal SFT`。

### 20.1 多模态数据源清单

| 优先级 | 模态 | 具体来源 | 用途 | 第一阶段做法 |
| --- | --- | --- | --- | --- |
| P0 | 图文 | `google-research-datasets/wit` | 多语言图文配对，含中文和英文切片 | 先抽中文/英文 caption 与描述文本，图像链路后置 |
| P0 | 图文网页 | `HuggingFaceM4/OBELICS` | 网页图文对、长说明、网页多模态语境 | 先做文本和元数据审计，再决定图像是否落地 |
| P0 | 文档图文 | `HuggingFaceM4/Docmatix` | 文档、教程、图文说明、技术图文场景 | 先抽文本；适合作为技术图文过渡源 |
| P0 | 语音文本 | `Mozilla Common Voice` | 多语言 ASR/语音文本对齐，英语覆盖强 | 先作为后续音文对齐候选，不混入 base pretrain |
| P0 | 中文语音文本 | `OpenSLR AISHELL-1` | 中文语音识别基础语料 | 用于中文音文 alignment 起步 |
| P1 | 屏幕/网页 | `Mind2Web` | 网页操作、界面理解、动作轨迹、屏幕语境 | 主要服务 UI/Agent/屏幕理解路线 |
| P1 | 自建态极 UI 截图-描述对 | 态极前端页面、设置页、训练页、工作区页面 | 让模型熟悉态极自己的界面元素和术语 | 本地自动截图 + 人工/半自动标注 |
| P1 | 自建工具调用过程截图-说明对 | JupyterLab、终端、训练日志、报错界面 | 让模型学会“看训练界面并解释状态” | 只做后续专项对齐，不混主预训练 |

### 20.2 多模态补料原则

1. 第一版不要把多模态数据混进当前 `1B text base pretrain` 主料。  
   当前主线仍然是把文本基座做稳。

2. 如果只是想让 tokenizer 和文本模型先熟悉多模态格式：

```text
只保留多模态占位 token
+ 少量文本级 caption / screen / audio 标签格式样本
```

3. 真正的图文/音文能力要放到：

```text
文本 base 完成
-> 文本 SFT 完成
-> projector / adapter alignment
-> multimodal instruction SFT
```

4. 态极自己的 UI、训练终端、日志页面，最好自己构建小而精的数据集。  
   通用图文数据集能给基础视觉语言能力，但不能替代“懂态极自己的界面”。

### 20.3 多模态最小目标

建议把“最小能开工”和“真正够用”分开：

```text
图文 caption / QA：
  最小开工：100k 对
  更稳妥：500k - 1M 对

中文音文：
  最小开工：50k - 100k 对
  更稳妥：200k+ 对

态极自建 UI / screen 数据：
  最小开工：10k 对
  更稳妥：20k - 50k 对
```

### 20.4 现在最值得执行的多模态顺序

```text
WIT 中文/英文切片
-> Common Voice / AISHELL-1
-> Mind2Web
-> 态极自建 UI 截图-描述对
-> 态极训练终端/日志截图-解释对
```

---

## 21. 外部数据源官方入口

下面这些入口已经确认可访问，后续补料时优先用它们作为基准来源：

- FineWeb EDU: [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- FineWeb2: [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
- SkyPile-150B: [Skywork/SkyPile-150B](https://huggingface.co/datasets/Skywork/SkyPile-150B)
- OpenWebMath: [open-web-math/open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math)
- CodeParrot Clean: [codeparrot/codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean)
- OBELICS: [HuggingFaceM4/OBELICS](https://huggingface.co/datasets/HuggingFaceM4/OBELICS)
- Docmatix: [HuggingFaceM4/Docmatix](https://huggingface.co/datasets/HuggingFaceM4/Docmatix)
- Common Voice: [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets)
- AISHELL-1: [OpenSLR 33](https://www.openslr.org/33/)
- Mind2Web: [OSU-NLP-Group/Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web)
- WIT: [google-research-datasets/wit](https://github.com/google-research-datasets/wit)

## 22. Checkpoint-400000 ?????????

> ???? 2026-07-08 ? `checkpoint-400000` ????????
> ?????? 26.21B tokens????? 12B???????????

### 22.1 ????

| ?? | ? | ?? |
| --- | --- | --- |
| Step | 400,000 | ? |
| Loss (??) | 2.96 | PPL ? 19.3??? |
| Tokens Seen | 26.21B | ?? 12B ?? 2 ? |
| PPL ??? (27 tok) | 40.36 | ?? |
| PPL ??? (32 tok) | 25.87 | ???? loss |
| ???? | ???????/??? | **??????** |
| ???? | ????? | **??????** |
| ???? | ?????????? | ????? SFT |
| ???? | ????? | ???? |

**????**?26B tokens ??????????????????/?????????

1. ?????? prompt ?????"XX ??""XX ??""??????"?????
2. ?????????Python ????????????
3. ??????????????????

### 22.2 ????

```text
checkpoint-400000 (26B, loss=2.96, ????)
  |
  ?? Phase A: ??????? (3-5B tokens, ~1-2 ?)
  ?   ??: ???????????/??/??/????
  ?   LR: 1e-4 ? 3e-5 (cosine)
  ?
  ?? checkpoint-445000 (loss ~2.4-2.7 ??)
  |
  ?? Phase B: ????? (1-2B tokens, ~3-5 ?)
  ?   ??: ??????????? polish
  ?   LR: 5e-5 ? 1e-5 (cosine)
  ?
  ?? checkpoint-460000 (base pretrain ??)
  |
  ?? Phase C: SFT ???? (100M-200M tokens, ~2-4 ?)
      ??: ????????????????
      LR: 2e-5 (?? epoch)
      |
      ?? taiji_1b_sft_final
```

### 22.3 Phase A?????????3-5B tokens?

**?????????**???? 26B tokens ???????????-domain prior??????????????? prior??? SFT ???"??????????"??????????

#### 22.3.0 ?????????????

**??? Phase A ???????????**????????????/???????????????

```bash
# ???????????????????
python -c "
import os, glob
data_dir = 'taiji_data/training_data/pretrain_mix_v1'
for f in sorted(glob.glob(f'{data_dir}/**/*.jsonl', recursive=True)):
    count = sum(1 for _ in open(f, encoding='utf-8', errors='ignore'))
    print(f'{count:>12,}  {os.path.basename(f)}')
"
```

**??????**??????????????

| ?? | ???? | ???? | ?? |
| --- | ---: | ---: | --- |
| code | 500,000 | 340,428 | ? ?? |
| tech_docs | 200,000 | 42,975 | ? ???? |
| math | 500,000 | 551,335 | ? |
| english | 3,000,000 | 5,425,095 | ? |
| chinese | 3,000,000 | 10,817,451 | ??? parquet ??? |
| taiji | 200,000 | 319,846 | ? |

**?? code ? tech_docs ?????????????? Phase A????????**

#### 22.3.1 ?????Phase A?

**???????????? >= ???????**????????"????????? = ?? / ??????"???????????????

| ?? | ?? | ?? token | ????? | ?????? |
| --- | ---: | ---: | --- | ---: |
| **????** | **30%** | 0.9B-1.5B | FineWeb-Edu, FineWeb2-en, Falcon-RefinedWeb | ? 3M |
| **????** | **20%** | 0.6B-1.0B | SkyPile ??????? parquet ? jsonl? | ? 3M |
| **??** | **20%** | 0.6B-1.0B | CodeParrot ?? shards, The Stack Python/JS | ? 500k |
| **??/??** | **15%** | 0.45B-0.75B | OpenWebMath ?? shards | ? 500k |
| **????** | **10%** | 0.3B-0.5B | Docmatix ??, Python/PyTorch docs | ? 200k |
| **????** | **5%** | 0.15B-0.25B | ?????????/???? | ? 200k |

??????? Round 1??
- ??? 1.9% ?? **20%**??????????? 0????????
- ??? 31% ??????????
- ??? 61.8% ?? 20%?????????????????????
- ?????? 10%?????/????????
- **??????????????????**

#### 22.3.2 ?????Phase A?

```bash
cd E:\taiji
set HF_ENDPOINT=https://hf-mirror.com

# P0: ???? ? CodeParrot
python scripts/data_prep/download_pretrain_mix_v1.py \
  --preset code_boost \
  --source codeparrot_code \
  --shards-per-source 3 \
  --max-records-per-source 1000000 \
  --output-dir taiji_data/training_data/phase_a/code

# P0: ???? ? FineWeb-Edu + FineWeb2
python scripts/data_prep/download_pretrain_mix_v1.py \
  --preset english_diverse \
  --shards-per-source 2 \
  --max-records-per-source 1000000 \
  --output-dir taiji_data/training_data/phase_a/english

# P0: ???? ? OpenWebMath
python scripts/data_prep/download_pretrain_mix_v1.py \
  --source openwebmath \
  --shards-per-source 2 \
  --max-records-per-source 500000 \
  --output-dir taiji_data/training_data/phase_a/math

# P1: ??????????? slice?
python scripts/data_prep/download_pretrain_mix_v1.py \
  --preset chinese_diverse \
  --source skypile_zh \
  --shards-per-source 1 \
  --max-records-per-source 1000000 \
  --output-dir taiji_data/training_data/phase_a/chinese
```

#### 22.3.3 ??

```bash
python scripts/data_prep/clean_pretrain_data.py \
  --input-dir taiji_data/training_data/phase_a \
  --output-dir taiji_data/training_data/phase_a_clean \
  --min-chars 64 \
  --max-chars 20000 \
  --dedup
```

#### 22.3.4 ?????Phase A?

```bash
# Phase A: ? checkpoint-400000 ????????????
python scripts/native_v2/pretrain.py   --data_dir taiji_data/training_data/phase_a_clean   --tokenizer_path taiji/tokenizer_native_v2/sentencepiece.model   --contract taiji/tokenizer_native_v2/tokenizer_contract.json   --output taiji_checkpoints/phase_a   --resume_from_checkpoint E:/taiji/checkpoint-400000   --replay_data_dir taiji_data/training_data/pretrain_mix_v1   --replay_ratio 0.30   --warn_oversample 5   --max_steps 45000   --batch_size 1   --gradient_accumulation_steps 64   --max_length 2048   --learning_rate 1e-4   --min_learning_rate 3e-5   --warmup_steps 200   --save_every 3000   --log_every 50   --use_streaming   --streaming_buffer 10000   --holdout_dir taiji_data/training_data/pretrain_12b/holdout   --num_workers 2
```

**????????**?

| ?? | ? | ?? |
| --- | --- | --- |
| `--replay_data_dir` | `pretrain_mix_v1` | ?? Round 1 ???????????? |
| `--replay_ratio` | `0.30` | 30% ?????????70% ????? |
| `--warn_oversample` | `5` | ????????? 5? ??????? |

**????**?
- `resume_from_checkpoint` ? ??? checkpoint-400000 ??????? checkpoint ?? optimizer state?
- `learning_rate: 1e-4` ? ??? 3e-4 ???? optimizer state ??????? warmup
- `max_steps: 45000` ? 45000 ? 64 ? 2048 ? 5.9B tokens
- `replay_ratio: 0.30` ? **??**?30% ???? Round 1 ????????????????????
- `warn_oversample: 5` ? **??**??????????? 5 ??????



#### 22.3.5 ???????

?? Round 2 / Phase A ?????????????????????

| ?? | ???? | ???? |
| --- | --- | --- |
| ??????? | `--warn_oversample 5` + ?????????? | 22.3.0 ?? + pretrain.py |
| ????? | `--replay_data_dir` + `--replay_ratio 0.30` | pretrain.py StreamingNativeDataset |
| ??????? | ?? warning + ???? warmup | pretrain.py load_checkpoint |
| LR schedule ?? | warmup ?? 200??? opt ?????? | pretrain.py train_config |
| ?????? | ????????????????? | 22.3.1 ??? |

**??????**?`replay_ratio=0.30` ???? 10 ? batch ?? 3 ??? Round 1 ? `pretrain_mix_v1`?7 ??? Phase A ??????????????????????????????????????????????????
### 22.4 Phase B???????1-2B tokens?

Phase A ????????????????? polish?

#### 22.4.1 ?????Phase B?

| ?? | ?? | ?? |
| --- | ---: | --- |
| ??????FineWeb-Edu top? | 30% | ????? |
| ??????????????? | 25% | ????? |
| ????? Python/JS? | 20% | ???? |
| ??/?? | 10% | ???? |
| ?????Python/PyTorch ??? | 10% | ???? |
| ?????? | 5% | ???? |

#### 22.4.2 ?????Phase B?

```bash
python scripts/native_v2/pretrain.py \
  --data_dir taiji_data/training_data/phase_b_clean \
  --tokenizer_path taiji/tokenizer_native_v2/sentencepiece.model \
  --contract taiji/tokenizer_native_v2/tokenizer_contract.json \
  --output taiji_checkpoints/phase_b \
  --resume_from_checkpoint taiji_checkpoints/phase_a/final \
  --max_steps 15000 \
  --batch_size 1 \
  --gradient_accumulation_steps 64 \
  --max_length 2048 \
  --learning_rate 5e-5 \
  --min_learning_rate 1e-5 \
  --warmup_steps 200 \
  --save_every 2000 \
  --log_every 50 \
  --use_streaming \
  --holdout_dir taiji_data/training_data/pretrain_12b/holdout \
  --num_workers 2
```

### 22.5 Phase C?SFT ?????100M-200M tokens?

Phase B ??????????????????????? SFT ????????????

#### 22.5.1 SFT ????

| ?? | token ?? | ?? | ??? |
| --- | ---: | --- | --- |
| SFT-1 ?????? | 30M | ????????????????? | P0 |
| SFT-2 ???? | 25M | Python ????????debug???? | P0 |
| SFT-3 ???? | 20M | ???????????? | P1 |
| SFT-4 ????/?? AI | 15M | ?????????????? | P1 |
| SFT-5 ????/ReAct | 10M | tool_call/tool_result ???? | P2 |

????? **100M tokens** ???

#### 22.5.2 SFT ????

```json
{"messages": [
  {"role": "system", "content": "?????????????????"},
  {"role": "user", "content": "?? Python ???????"},
  {"role": "assistant", "content": "def quicksort(arr):\n    ..."}
]}
```

#### 22.5.3 SFT ?? prompt

```text
?????  ?????????????
?????  ?? Python ??????????????????????
?????  Explain what machine learning is in simple terms.
?????  ????????????????????
?????  ????"?? AI"??????
?????  ????????????
```

### 22.6 ?????

????????????

```text
Phase A ?????
  [ ] holdout loss ????????????
  [ ] ????????????
  [ ] ?? prompt ?????????
  [ ] ??????????

Phase B ?????
  [ ] holdout ppl ?????
  [ ] ????????? Phase A
  [ ] ?????????????

Phase C ?????
  [ ] ????????????
  [ ] ???????????
  [ ] ???? JSON ????
  [ ] ??????????????
  [ ] ??????????
```

### 22.7 ??????? 4090D 24G?tokens_per_step=64?2048=131072?

| ?? | tokens | ?? steps | ???? |
| --- | ---: | ---: | --- |
| Phase A | 3-5B | ~23k-38k | 5-8 ? |
| Phase B | 1-2B | ~8k-15k | 2-3 ? |
| Phase C (SFT) | 100M | ~800-1500 | 1-2 ? |
| **??** | **~5-7B** | **~30k-55k** | **8-13 ?** |

### 22.8 ?????????

#### checkpoint-400000 ?? optimizer state

?? `checkpoint-400000` ? `taiji/loader.py` ? `save_model` ???????????config?tokenizer ? training_state.json?**??? optimizer.pt**????? resume ??

- ? ????????
- ? training step / tokens_seen ????
- ? Adam ?????m???????v?**??????**
- ?? ??? loss ???????? 0.5-1.0 ? spike????????

**??**?pretrain.py ????????????? opt ????? warmup?????? LR ???

**??**??????? `pretrain.py` ? `save_checkpoint`?????? optimizer.pt???????????????????

### 22.9 ?????

?????????????

| ?? | ?? | Key ?? | ?? |
| --- | --- | --- | --- |
| `scripts/native_v2/pretrain.py` | `TaijiBackbone` | ???`embed.weight`, `layers.N.w1.weight` | **??/??** |
| `taiji/architecture.py` | `ModelSelf` | ???`backbone.embedding.weight`, `backbone.layers.N.feed_forward.w1.weight` | **??/??** |

- **??? `pretrain.py`** ? checkpoint-400000 ? key ?????????? `--resume_from_checkpoint`
- **??? `taiji.loader.load_model`** ? ??? `_remap_legacy_keys` ???????????

?????????????????
