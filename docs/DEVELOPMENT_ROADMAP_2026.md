# 态极 v2.0 开发路线图

> 基于项目全面审计 + 2025-2026 AI Agent 最新研究成果

---

## 一、现状审计总结

### ✅ 已实现且扎实的部分

| 模块 | 代码量 | 质量 |
|------|--------|------|
| 自研推理引擎 (NativeInferenceEngine) | 501行 | 生产级，KV cache + top-p + 重复检测 |
| 混合引擎 (HybridEngine) | 449行 | 外部模型借用 + 训练数据收集 |
| ReAct 引擎 | 847行 | 多策略工具调用解析，含 JSON 修复 |
| 生命系统 (6个引擎) | 3600+行 | 需求驱动调度，4阶段睡眠周期 |
| RAG 系统 | 1100行 | Dense + BM25 + 交叉编码器重排 |
| 沙箱执行器 | 665行 | 进程隔离 + 模块白名单 |
| MCP 市场 | 905行 | 安装/卸载/启停/市场浏览 |
| 安全系统 | 1100行 | Constitutional AI + 威胁检测 |
| 前端 (Vue 3) | 7视图 + 13组件 | 完整的 IDE + 聊天 + 训练界面 |
| API 层 | 18个路由模块 | REST + WebSocket + JWT + 限流 |

### ⚠️ 存在隐患的部分

1. **自修改引擎** — 28行 stub，返回 `{"available": False}`
2. **插件系统** — 基础设施完成，但 0 个插件
3. **多模态模块** — audio/video/screen 仅 120-240 行，功能单薄
4. **body/ vs life/ 重复** — limbs.py、metabolism.py、senses.py 存在两份
5. **desktop/main.py vs api/run_app.py** — 功能重叠，注释说要合并但没做
6. **ModelSelf 125M 参数** — 太小，实际对话能力有限
7. **无端到端验证** — 没有证明"喂养→睡眠→成长"全链路真的 work

### ❌ 理论可行但从未实际验证的功能

- 睡眠训练（sleep_engine 调用 trainer，但从未跑通完整周期）
- 自主探索（explore_engine 能搜网页，但不知道搜完后学到什么）
- 科学发现（science_engine 存在，但无评估指标）
- 进化系统（evolution_engine 追踪阶段，但成长阈值未校准）
- 多智能体协作（multi_agent.py 有角色系统，但无实际部署案例）

---

## 二、2025-2026 AI Agent 最新趋势对标

### 主流方向 vs 态极现状

| 趋势 | 代表项目 | 态极现状 | 差距 |
|------|---------|---------|------|
| **MCP 协议** | Cursor, Claude Code | ✅ 已有 mcp_manager | 小 |
| **多智能体** | AutoGen, CrewAI, Camel | ✅ 有 multi_agent.py | 中：缺实际验证 |
| **ReAct 推理** | LangChain, ToolLLM | ✅ 有 react_engine.py | 小 |
| **自我改进** | Self-Refine, Reflexion | ❌ stub | 大 |
| **代码执行** | OpenDevin, SWE-Agent | ✅ 有 sandbox_executor | 小 |
| **长期记忆** | MemGPT, Letta | ✅ 有完整记忆系统 | 小 |
| **RAG** | LlamaIndex, RAGFlow | ✅ 混合检索 + 重排 | 小 |
| **工具学习** | ToolBench, Gorilla | ✅ 有 tool_registry | 中：缺自动发现 |
| **DPO/RLHF** | TRL, OpenRLHF | ✅ 有 dpo_trainer.py | 中：缺实际训练验证 |
| **小模型蒸馏** | Phi, Gemma, Qwen | ⚠️ 有 distill.py | 大：125M 太小 |
| **Agent Benchmark** | AgentBench, GAIA | ❌ 无评估体系 | 大 |

### 关键论文启示

1. **"More Agents Is All You Need" (2024)** — 简单的采样投票就能 scaling，态极的 multi_agent 可以直接用
2. **"Self-Refine" (2023)** — LLM 自我反馈改进，态极的 Constitutional AI 已有基础
3. **"Reflexion" (2023)** — 语言反馈的强化学习，态极的 reflector.py 可以对接
4. **"Voyager" (2023)** — 终身学习 Agent，态极的 life system 理念一致
5. **"Generative Agents" (2023)** — 斯坦福小镇，态极的生命系统更深入

---

## 三、开发路线图（分 4 个阶段）

### Phase 1: 验证与修复（2-3周）

**目标：证明现有功能真的 work**

1. **端到端集成测试**
   - 写一个完整的测试：启动后端 → 加载模型 → 对话 → 喂养 → 睡眠 → 验证模型更新
   - 目标：证明"生命系统"不是空转

2. **修复 body/ vs life/ 重复**
   - 合并 `taiji/life/limbs.py` 和 `taiji/body/limbs.py`
   - 统一为单一实现，消除维护隐患

3. **合并 desktop/main.py 和 api/run_app.py**
   - 提取公共逻辑到 `api/server_core.py`
   - desktop 和 api 入口都调用同一个核心

4. **ModelSelf 训练验证**
   - 用 taiji_data/training_data/ 里的数据跑一轮训练
   - 验证 loss 下降、模型能生成有意义的回复
   - 如果 125M 确实太小，升级到 350M 配置

5. **前端连接状态修复**
   - 统一 runtimeStore 和 useWebSocket 的状态源
   - 确保后端未运行时正确显示"未连接"

### Phase 2: 核心能力补全（4-6周）

**目标：让自修改和自我改进从 stub 变成真实功能**

1. **自修改引擎 (self_modification.py)**
   - 参考 Self-Refine / Reflexion 论文
   - 实现：对话后自我评估 → 识别不足 → 生成改进建议 → 应用到下次对话
   - 关键：不改模型权重，改 prompt / context / memory

2. **Agent Benchmark 评估体系**
   - 参考 AgentBench / GAIA
   - 实现：任务库 → 执行器 → 评分器 → 报告生成
   - 指标：任务成功率、工具调用准确率、推理步数、用户满意度

3. **睡眠训练全流程验证**
   - 确保 sleep_engine → trainer → model_save 全链路 work
   - 添加训练日志和可视化
   - 设置安全阈值防止灾难性遗忘

4. **工具自动发现**
   - Agent 在对话中遇到未知 API 时，自动搜索 MCP 市场
   - 参考 ToolBench 的工具学习方法

5. **多模态增强**
   - 至少让 vision_encoder 能处理图片理解（接入 CLIP 或 LLaVA）
   - voice_interface 接入 Whisper / TTS

### Phase 3: 规模化与生态（6-8周）

**目标：让态极从"一个项目"变成"一个平台"**

1. **插件系统激活**
   - 写 3-5 个示例插件（天气查询、日历管理、代码格式化）
   - 完善插件 API 文档
   - 添加插件沙箱隔离

2. **多智能体实际部署**
   - 参考 "More Agents Is All You Need"
   - 实现：任务分解 → 并行 Agent 执行 → 投票聚合
   - 场景：代码审查、文档生成、数据分析

3. **知识蒸馏管道**
   - 用大模型（GPT-4/Claude）生成高质量训练数据
   - 蒸馏到 ModelSelf 350M/1B
   - 参考 Phi-3 的数据策略

4. **RAG 系统增强**
   - 支持更多文档格式（PPT、Excel、图片 OCR）
   - 添加知识图谱（实体关系抽取）
   - 实现增量索引更新

5. **前端体验优化**
   - 对话支持 Markdown 渲染优化（代码高亮、数学公式）
   - 添加对话导出功能（PDF/Markdown）
   - 支持多窗口/分屏

### Phase 4: 产品化（持续迭代）

**目标：让态极可以被普通用户使用**

1. **一键安装包**
   - Windows installer（已有 NSIS 脚本，需要完善）
   - macOS .dmg
   - Linux AppImage

2. **模型分发**
   - 预训练好的 ModelSelf 模型上传到 HuggingFace
   - 提供 125M/350M/1B 三个版本
   - 首次启动自动下载

3. **用户引导**
   - 首次使用向导（选择模型、配置硬件、了解功能）
   - 内置教程和示例对话
   - 生命系统状态可视化优化

4. **社区建设**
   - 插件市场 Web 界面
   - 用户分享对话和插件
   - GitHub Discussions / Discord

---

## 四、技术债务清单

| 优先级 | 问题 | 影响 | 建议 |
|--------|------|------|------|
| P0 | self_modification.py 是 stub | 核心卖点缺失 | Phase 2 实现 |
| P0 | 无端到端测试 | 不知道系统是否真的 work | Phase 1 补充 |
| P1 | body/ vs life/ 重复代码 | 维护成本高 | Phase 1 合并 |
| P1 | desktop 与 api 入口重叠 | 部署混乱 | Phase 1 合并 |
| P1 | ModelSelf 125M 太小 | 实际能力有限 | Phase 3 蒸馏升级 |
| P2 | 多模态模块单薄 | 功能不完整 | Phase 2 增强 |
| P2 | 无评估体系 | 无法量化改进 | Phase 2 建立 |
| P3 | 插件系统空转 | 生态缺失 | Phase 3 激活 |

---

## 五、立即可做的事（本周）

1. ✅ 前端配色重构（已完成）
2. ✅ IDE 布局修复（已完成）
3. ✅ 错误处理统一化（已完成）
4. ✅ 项目清理（已完成）
5. 🔲 写一个端到端集成测试，验证生命系统全链路
6. 🔲 合并 body/ 和 life/ 的重复代码
7. 🔲 验证 ModelSelf 能否实际训练并生成有意义的输出

---

## 六、参考资源

- [Self-Refine](https://arxiv.org/abs/2303.17651) — 自我反馈改进
- [Reflexion](https://arxiv.org/abs/2303.11366) — 语言反馈强化学习
- [Voyager](https://arxiv.org/abs/2305.16291) — 终身学习 Agent
- [Generative Agents](https://arxiv.org/abs/2304.03442) — 斯坦福小镇
- [More Agents Is All You Need](https://arxiv.org/abs/2402.05120) — 多 Agent scaling
- [AgentBench](https://arxiv.org/abs/2308.03688) — Agent 评估基准
- [ToolBench](https://arxiv.org/abs/2307.16789) — 工具学习
- [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219) — 小模型蒸馏最佳实践
