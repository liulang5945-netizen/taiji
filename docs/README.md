# Taiji Docs Index

`docs/` 只保留当前仍然有效的文档。

## 阅读顺序

1. **`RESONANCE_FIELD_ARCHITECTURE.md`** — 态极的未来架构
   - 共振场 + 小模型神经元协同
   - 动态激活（场自己决定用多少神经元）
   - 三层自限机制（防失控）
   - 递归自进化（场自然淘汰不共振的神经元）
   - 硬件需求分析（vs 传统单体模型）

2. **`TAIJI_1B_12B_TOKEN_TRAINING_PLAN_CN.md`** — 训练计划与试验记录
   - 词表、数据、预训练、SFT 的完整计划
   - 第 22 章：checkpoint-400000 到 SFT 的三阶段试验记录
   - Phase A/B/C 的实际结果与根因分析

3. **`NATIVE_V2_TOKENIZER.md`** — 词表规范
   - 256K vocab tokenizer contract
   - 多模态 token 预留

4. **`ENTRYPOINTS.md`** / `ENTRYPOINTS_EN.md` — 运行入口清单

5. **`INSTALL.md`** / `INSTALL_EN.md` — 安装指南

---

## 已移除的旧文档

以下文档已被共振场架构取代，不再保留：

- ~~`ARCHITECTURE.md`~~ → 旧的高层结构说明，已被 `RESONANCE_FIELD_ARCHITECTURE.md` 取代
- ~~`ARCHITECTURE_EN.md`~~ → 同上
- ~~`1B_DATA_GAP_REPORT_CN.md`~~ → 旧的数据缺口报告，训练已推进到 31B tokens，不再有参考价值
