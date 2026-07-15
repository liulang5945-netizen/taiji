# Taiji Docs Index

`docs/` 只保留当前仍然有效的文档。

## 阅读顺序

1. **`TAIJI_1B_12B_TOKEN_TRAINING_PLAN_CN.md`** — 训练计划与试验记录
   - 词表、数据、预训练、SFT 的完整计划
   - 第 22 章：checkpoint-400000 到 SFT 的三阶段试验记录
   - Phase A/B/C 的实际结果与根因分析

2. **`NATIVE_V2_TOKENIZER.md`** — 词表规范
   - 256K vocab tokenizer contract
   - 多模态 token 预留

3. **`ENTRYPOINTS.md`** / `ENTRYPOINTS_EN.md` — 运行入口清单

4. **`INSTALL.md`** / `INSTALL_EN.md` — 安装指南

---

## 已移除的文档

- ~~`ARCHITECTURE.md`~~ → 旧的高层结构说明
- ~~`ARCHITECTURE_EN.md`~~ → 同上
- ~~`1B_DATA_GAP_REPORT_CN.md`~~ → 旧的数据缺口报告
- ~~共振场架构相关文档~~ → 已迁移至下一代项目 `taiji-neuron`
