# 态极 1B Native-v2 训练教程

本文档面向当前仓库的真实状态，目标是把态极按 `native-v2` 路线完成一轮完整训练：

1. 重建文本词表
2. 从零预训练 1B
3. 再做多模态对齐 / SFT / Agent 微调

这份教程默认你已经接受一个关键前提：

- 保留 [taiji/tokenizer_contract.json](/E:/taiji/taiji/tokenizer_contract.json) 的全局词表空间设计
- 重新训练 `text-only SentencePiece`
- 不再使用旧的 HF/Qwen 兼容预训练路径做正式 native-v2 训练

## 1. 当前推荐路线

当前项目里同时存在两条训练思路：

- 旧路线：`taiji/train/autodl_pretrain.py`
- 新路线：`scripts/native_v2/pretrain.py`

正式训练 `1B` 时，推荐使用 **native-v2 路线**，原因是：

- native-v2 明确使用 `text_offset`
- native-v2 会校验 `tokenizer_contract.json`
- native-v2 的默认 `vocab_size=256000`
- native-v2 默认上下文长度已经是 `2048`

相关文件：

- [docs/NATIVE_V2_TOKENIZER.md](/E:/taiji/docs/NATIVE_V2_TOKENIZER.md)
- [scripts/native_v2/pretrain.py](/E:/taiji/scripts/native_v2/pretrain.py)
- [taiji/tokenizer_contract.json](/E:/taiji/taiji/tokenizer_contract.json)

## 2. 先说结论：词表要不要重练

要。

但不是把整个 contract 推倒重来，而是：

- 保留 `tokenizer_contract.json`
- 只重练文本词表 `sentencepiece.model`

原因：

1. 现在项目已经固定了 multimodal/control/text 的全局 ID 空间
2. 当前 text vocab 明显不是按这次大规模预训练数据重新拟合的
3. 如果继续沿用旧词表，中文、代码、数学的切分质量都会拖后腿

所以正确顺序是：

1. 准备预训练语料
2. 用预训练语料重练 text SP
3. 验证 tokenizer
4. 从零预训练 1B
5. 再做后续微调

## 3. 当前已经准备好的首批预训练语料

本地已经生成了第一批混合语料：

- [taiji_data/training_data/pretrain_mix_v1/fineweb_edu.jsonl](/E:/taiji/taiji_data/training_data/pretrain_mix_v1/fineweb_edu.jsonl)
- [taiji_data/training_data/pretrain_mix_v1/skypile_zh.jsonl](/E:/taiji/taiji_data/training_data/pretrain_mix_v1/skypile_zh.jsonl)
- [taiji_data/training_data/pretrain_mix_v1/openwebmath.jsonl](/E:/taiji/taiji_data/training_data/pretrain_mix_v1/openwebmath.jsonl)
- [taiji_data/training_data/pretrain_mix_v1/codeparrot_code.jsonl](/E:/taiji/taiji_data/training_data/pretrain_mix_v1/codeparrot_code.jsonl)

语料说明见：

- [taiji_data/training_data/pretrain_mix_v1/manifest.json](/E:/taiji/taiji_data/training_data/pretrain_mix_v1/manifest.json)

当前这批数据是一个 **starter tranche**，适合：

- 重练 tokenizer
- 验证 1B 训练链路
- 启动第一轮预训练

它还不等于最终完整版 1B 语料池，后续还需要继续扩 shard。

另外要特别说明：

- 当前仓库里的英文 base pretrain 规划已经切到 `fineweb2_en`
- 但你本地这次审计到的 `pretrain_mix_v1` 还没有把 `fineweb2_en.jsonl` 真正下载落地
- 所以现在的英文 token 统计，仍然主要来自 `fineweb_edu`

这也是为什么当前 1B 缺口里，英文仍然是最需要优先补的部分之一。

## 4. 如何继续下载更多预训练数据

使用新脚本：

- [scripts/data_prep/download_pretrain_mix_v1.py](/E:/taiji/scripts/data_prep/download_pretrain_mix_v1.py)

先看 manifest：

```bash
python scripts/data_prep/download_pretrain_mix_v1.py --dry-run
```

下载并标准化首批数据：

```bash
python scripts/data_prep/download_pretrain_mix_v1.py --max-records-per-source 100000
```

只继续下载某一个源：

```bash
python scripts/data_prep/download_pretrain_mix_v1.py --sources fineweb_edu --download-only
python scripts/data_prep/download_pretrain_mix_v1.py --sources fineweb_edu --normalize-only
```

当前内置源：

- `fineweb_edu`
- `fineweb2_en`
- `fineweb2_zh`
- `skypile_zh`
- `openwebmath`
- `codeparrot_code`

更建议的下载节奏不是“一次全梭”，而是按阶段补：

1. `stage0 / tokenizer + smoke`
   - `fineweb_edu`
   - `skypile_zh`
   - `openwebmath`
   - `codeparrot_code`
2. `stage1 / 英文补强`
   - `fineweb2_en`
3. `stage1 / 中文补强`
   - `fineweb2_zh`
   - `skypile_zh` 更多 shard

示例：

```bash
python scripts/data_prep/download_pretrain_mix_v1.py --sources fineweb2_en --dry-run
python scripts/data_prep/download_pretrain_mix_v1.py --sources fineweb2_en --shards-per-source 1
```

## 5. 重建 tokenizer

先构建语料：

```bash
python scripts/build_native_vocab_corpus.py ^
  --data-dir taiji_data/training_data/pretrain_mix_v1 ^
  --data-dir taiji ^
  --output taiji_data/tokenizer/native_v2_corpus.txt
```

训练新的 text-only SentencePiece：

```bash
python scripts/train_native_text_sp.py ^
  --corpus taiji_data/tokenizer/native_v2_corpus.txt ^
  --output-dir taiji/tokenizer_native_v2 ^
  --contract taiji/tokenizer_contract.json
```

验证：

```bash
python scripts/verify_native_tokenizer.py ^
  --tokenizer-dir taiji/tokenizer_native_v2 ^
  --contract taiji/tokenizer_contract.json
```

通过标准：

1. `sp.GetPieceSize()` 不大于 `text_vocab_size`
2. `text_offset=13388`
3. `total_vocab_size=256000`

## 6. 组织训练数据目录

建议单独建一个只放训练 JSONL 的目录，例如：

```text
taiji_data/training_data/pretrain_stage_1/
  fineweb_edu.jsonl
  skypile_zh.jsonl
  openwebmath.jsonl
  codeparrot_code.jsonl
```

不要把 raw shard、下载缓存、无关 JSON 元数据混在这里。

虽然当前 [scripts/native_v2/pretrain.py](/E:/taiji/scripts/native_v2/pretrain.py) 已经加了跳过 `.cache` 和 `manifest.json` 的保护，但训练目录保持干净仍然是最稳的做法。

## 7. 开始 1B 预训练

### 单卡验证版

先跑一个短程 sanity check：

```bash
python scripts/native_v2/pretrain.py ^
  --data_dir taiji_data/training_data/pretrain_mix_v1 ^
  --tokenizer-dir taiji/tokenizer_native_v2 ^
  --output taiji_data/taiji_pretrained_1b_smoke ^
  --max_steps 200 ^
  --batch_size 1 ^
  --gradient_accumulation_steps 8 ^
  --max_length 1024 ^
  --log_every 10 ^
  --save_every 100
```

观察重点：

- loss 是否稳定下降
- 是否出现 shape / vocab / offset 错误
- checkpoint 是否能正常保存

### 正式 1B 预训练

如果 smoke run 正常，再拉长：

```bash
python scripts/native_v2/pretrain.py ^
  --data_dir taiji_data/training_data/pretrain_mix_v1 ^
  --tokenizer-dir taiji/tokenizer_native_v2 ^
  --output taiji_data/taiji_pretrained_1b_stage1 ^
  --max_steps 50000 ^
  --batch_size 1 ^
  --gradient_accumulation_steps 32 ^
  --max_length 2048 ^
  --learning_rate 3e-4 ^
  --min_learning_rate 3e-5 ^
  --warmup_steps 1000 ^
  --log_every 50 ^
  --save_every 5000
```

### 资源预期

当前 native-v2 训练脚本是 **单机单进程** 路线，更适合：

- A100 80G 单卡
- 或先小步验证，再迁到多卡版脚本

如果你是多卡 AutoDL，当前仓库还缺一个干净的 native-v2 多卡训练入口，这点见后面的“已知问题”。

## 8. 多模态应该怎么接

你之前提醒得对，态极不能只盯中文文本。

但多模态也不应该直接混进当前这批 1B base pretrain 主流程。更稳的方式是分两段：

1. `1B base pretrain`
   - 中文
   - 英文
   - 代码
   - 数学
2. `1B multimodal alignment`
   - 图文对齐
   - 音文对齐
   - 多模态指令微调

原因很简单：

- 你现在本地多模态标注只有 `100` 对
- 这远远不够把多模态训练混入基座阶段
- 强行混训，只会让文本基座和多模态都学不好

所以当前最合理的策略是：

1. 先把文本基座做强
2. 多模态单独做第二阶段
3. 最后再做统一 assistant / agent 行为微调

## 9. 预训练结束后做什么

顺序建议：

1. 先做基础续训检查
2. 再做多模态对齐
3. 再做 SFT
4. 最后做 Agent / 工具调用微调

不要一上来就把预训练模型直接拿去做复杂 agent 微调。

更稳的节奏是：

1. `pretrain_stage1`
2. `pretrain_stage2`（扩更多 shard）
3. `multimodal_alignment`
4. `conversation_sft`
5. `agent_tool_sft`

## 10. 当前已知问题

态极项目 **不是只卡在模型这一步**。

模型是主卡点，但还有几类隐藏问题仍然在：

### 10.1 训练入口分裂

- 旧入口 [taiji/train/autodl_pretrain.py](/E:/taiji/taiji/train/autodl_pretrain.py)
- 新入口 [scripts/native_v2/pretrain.py](/E:/taiji/scripts/native_v2/pretrain.py)

这两个入口的 tokenizer 假设不同。  
如果混着用，最容易出问题的是 token ID 空间和训练数据格式。

### 10.2 旧文档仍然在指向 106 条 SFT 对话

[docs/autodl_training_guide.md](/E:/taiji/docs/autodl_training_guide.md) 现在还是旧思路，示例数据是 `sft_merged_clean.jsonl`，不适合作为正式 1B 预训练教程。

### 10.3 native-v2 多卡训练链路还不完整

当前 [scripts/native_v2/pretrain.py](/E:/taiji/scripts/native_v2/pretrain.py) 是单卡脚本。  
如果你要高效跑 1B 长程训练，后面还需要补：

- `accelerate` 多卡版本
- resume/断点续训管理
- 验证集 / perplexity 监控

### 10.4 数据量仍然只是第一批

现在这批 `pretrain_mix_v1` 够启动、够练 tokenizer、够 smoke run，  
但还不够你放心说“1B 已经吃饱了”。

### 10.5 评估链路还不够产品化

当前更像“能训练”，还不是“训练完能稳定知道它变好了多少”。

还缺：

- 固定验证集
- 中文/代码/数学分开评测
- native-v2 专项回归评测

## 11. 我对当前状态的判断

一句话版：

**不是“项目已经没有隐藏问题，只差训练一下模型”**。  
更准确地说，是：

**项目已经进入“模型与训练基础设施是主矛盾”的阶段。**

也就是：

- 前端/接口/安全/基础组织问题已经比之前好很多
- 但真正决定态极上限的，已经变成 tokenizer、数据、预训练入口、评测闭环

所以现在最该做的不是再铺更多功能，而是把这一条链路彻底跑顺：

1. 数据
2. 词表
3. 1B 预训练
4. 评测
5. 再微调

## 12. 推荐下一步

最推荐的执行顺序：

1. 扩 `pretrain_mix_v1` 第二批英文与中文 shard
2. 重练 native-v2 text tokenizer
3. 跑 200 step smoke run
4. 跑正式 `1B stage1`
5. 开始准备 `multimodal_alignment` 数据池
6. 补 native-v2 多卡训练入口

如果你继续让我往前做，最合适的下一步就是：

- 我直接帮你写一个 **native-v2 多卡 1B 训练脚本**
- 再顺手补一个 **stage1/stage2 训练配置模板**
