# 态极 1B 基座与多模态训练计划

## 1. 目标先定清楚

态极当前最合理的路线，不是一次把“中文、英文、代码、数学、多模态、助手行为”全部揉进一个训练阶段。

更稳的路线是两大阶段：

1. `1B base pretrain`
2. `1B multimodal alignment + SFT`

也就是先把文本基座做扎实，再让它学图像、音频和助手行为。

## 2. 为什么要这么拆

当前本地审计结果已经很明确：

- 文本主料总量约 `769,938,441` tokens
- 距离最低可接受 `5B` 还差 `4,230,061,559`
- 英文主料仅约 `109,367,552` tokens
- 多模态标注仅 `100` 对

这意味着：

- 文本基座还没吃饱
- 英文更没吃饱
- 多模态数据量还远远不够混入 base pretrain

所以如果现在强行一锅炖，结果通常是三输：

- 中文基座不够稳
- 英文能力偏弱
- 多模态对齐也学不明白

## 3. 当前建议的数据目标

### 基座文本

- 最低目标：`5B tokens`
- 推荐目标：`10B tokens`

建议内部占比可以先按这个数量级理解：

- 中文通用：`1.5B - 3B`
- 英文通用：`2B - 2.5B`
- 代码：`0.8B - 2B`
- 数学：`0.3B - 1B`

这不是必须绝对精确的配方，但它能避免态极最后变成：

- 中文看着还行
- 英文明显偏弱
- 代码和数学带一点点
- 多模态只是功能挂件

### 多模态对齐

当前先定一个务实门槛：

- 图文/音文标注最少 `100k` 对

更舒服的启动规模：

- `300k - 1M` 对

## 4. 推荐阶段划分

### Stage 0：链路验证

用途：

- 下载首批混合数据
- 重练 tokenizer
- 跑 smoke run

当前本地这批 `pretrain_mix_v1` 就够做这个阶段。

### Stage 1：1B 文本基座正式预训练

训练内容：

- 中文
- 英文
- 代码
- 数学

这一阶段不要混入多模态标注，不要混入 SFT 主料，也不要把合成对话当主 pretrain 语料。

### Stage 2：多模态对齐

训练内容：

- 图像理解
- 音频理解
- 图文问答
- 音文问答

目标不是重练语言基座，而是让基座学会接住多模态输入。

### Stage 3：SFT

训练内容：

- 中文助手
- 英文助手
- 工具调用
- 结构化输出

### Stage 4：Agent / 产品行为微调

训练内容：

- 工具规划
- 长任务执行
- 产品内工作流

## 5. 当前最优先补什么

优先级建议：

1. 英文通用主料
2. 中文第二批主料
3. 代码继续扩量
4. 数学适度扩量
5. 多模态标注池

理由很直接：

- 你已经明确不想把 1B 做成只偏中文的小模型
- 英文是当前最大的结构性短板之一
- 多模态很重要，但它不该抢走基座阶段的预算

## 6. 对“词表要不要完全重头重练”的判断

我的建议还是：

- 不重做全局 `tokenizer_contract.json`
- 只重练文本词表 `sentencepiece.model`

这样能保住：

- control token 空间
- multimodal token 空间
- native-v2 的固定词表布局

同时又能让文本切分真正适配新的中文、英文、代码、数学主料。

## 7. 当前仓库里哪些数据不适合当 1B 基座主料

不适合作为 base pretrain 主料的，主要是这类：

- SFT 对话混合
- 强合成助手数据
- 明显 instruction-heavy 的行为数据

例如：

- `pretrain_final.jsonl` 这类名字看起来像“大混合终版”的数据，必须先看成分，不能默认当基座主料
- `OpenHermes` 这一类更适合放在 SFT 阶段

## 8. 现在就能执行的动作

### 立刻可做

1. 用现有 `pretrain_mix_v1` 重练 tokenizer
2. 跑 `200 step smoke run`
3. 用 `--preset english_boost` 补 `fineweb2_en`
4. 用 `--preset chinese_boost` 扩中文第二批 shard
5. 继续做 1B stage1 数据审计

示例：

```bash
python scripts/data_prep/download_pretrain_mix_v1.py --preset stage0_smoke --max-records-per-source 100000
python scripts/data_prep/download_pretrain_mix_v1.py --preset english_boost --shards-per-source 1
python scripts/data_prep/audit_1b_training_assets.py
```

### 暂时不要急着做

1. 把多模态混进 base pretrain
2. 把 SFT 语料当基座主料
3. 过早开始 3B 规划

## 9. 一句话判断

态极现在已经不是“没有方向”，而是方向很清楚了：

**先把 1B 文本基座做好，明确补足英文，再单独把多模态做成第二阶段。**

这条线跑顺了，后面的 `3B`、更强 agent、真正产品级多模态，才有地基。
