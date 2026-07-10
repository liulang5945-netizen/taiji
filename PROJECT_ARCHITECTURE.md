# 态极 (TaijiCore) 全项目总流程图

## 图例
- ⚡ = 启动入口
- 🧠 = 推理核心
- 🔗 = 联网 / 搜索
- ❤️ = 生命系统
- 🛠️ = 工具 / Agent
- 📡 = 前端 API
- 🔐 = 安全
- ✅ = 闭环已确认

---

## 一、启动总览 (Startup)

```
⚡ api/app.py (FastAPI lifespan)
│
├─► load_model_on_startup()
│   └─► model_loader.py
│       ├─ create ModelSelf
│       ├─ create ModelSelfTokenizer
│       └─ app_state.update_model(model, tokenizer, trainer, model_name)
│
├─► scheduler = get_life_scheduler()
│   └─► scheduler.start()
│       └─► 60s 心跳循环           ✅
│
└─► 注册所有 API 路由 (routes_chat / routes_agent / routes_life / ...)
```

---

## 二、对话推理总链路 (Chat / Agent Pipeline)

```
📡 用户输入
│
├─ POST /api/agent/react          → agent_service.run_react_task()
├─ POST /api/agent/react/stream   → agent_service.run_react_stream()
├─ POST /api/agent/collaborate    → agent_service.collaborate()
│
▼
🧠 NativeAgentEngine.run(task)
│
├─ PerceptionSystem.encode_workspace()     # 感知工作区
├─ MemorySystem 加载情景/语义记忆
│
├─ _build_prompt(task, step_num)
│   ├─ system_prompt
│   ├─ <web_search>   ← _auto_search_context       ✅  联网搜索
│   ├─ <rag_knowledge>← app_state.rag_kb.search()   ✅  知识库检索
│   └─ 历史 steps (think / tool_call / observation)
│
├─ PlannerSystem.plan(task) → 决策下一步做什么
│
├─ ModelSelf.generate(prompt) → 推理输出
│   └─ ConstitutionalAI.critique(response)           ✅  安全审查
│       └─ 违规则修正输出
│
├─ 解析工具调用 → registry.execute(tool, args)
│   │
│   ├── search / search_deep / search_local     🔗 联网搜索
│   ├── read_webpage / browse_web              🔗 网页阅读
│   ├── smart_crawl                             🔗 智能爬取
│   ├── crawl_site                              🔗 批量建索引
│   ├── generate_image / text_to_speech         🎨 多模态输出
│   ├── file_parser / code_executor / ...       🛠️ 工具
│   └── MCP tools (mcp_manager → 桥接注册)      ✅
│
├─ 观察结果 → AgentStep.observation
│
├─ ReflectorSystem.reflect() → 反思是否完成任务
│   └─ self_modification.evaluate_response()    ✅
│
├─ _is_final_answer() → 判断是否终止
│
└─ 每 3 步 check_rollback()                     ✅
   每 5 步 batch_apply_improvements()           ✅
```

---

## 三、联网搜索闭环 (Search Engine) 🔗

```
Agent 调用 search / search_deep / smart_crawl
│
▼
Pipeline (taiji/tools/search/pipeline.py)
│
├─ search_deep(query)
│   ├─ ① 先查本地索引 (Index)
│   │   └─ Tokenizer (词级分词) → BM25 倒排索引 → SearchHit[]
│   │       └─ 命中 → 直接返回                    ✅
│   │
│   └─ ② 未命中 → Discovery 层
│       ├─ JSON API 引擎 (Wikipedia/HN/SO/GitHub/arXiv)  ✅  <1s
│       └─ HTML 爬取引擎 (Baidu/Bing)            ✅  <1s
│           └─ SearchResult[] 返回 URL 列表
│
├─ ③ Fetcher 层: 并行 HTTP 抓取 (DualFetcher)
│   └─ HTTP 优先 → 浏览器回退 (Playwright)     ✅
│
├─ ④ Extractor 层: Readability 正文提取
│   └─ HTML → Markdown → PageContent           ✅
│
├─ ⑤ 入索引: Index.add_page()                   ✅  闭环!
│   └─ 下次搜索直接本地命中
│
└─ ⑥ 返回结构化摘要

🧠 SmartCrawler (智能爬虫)
│
├─ crawl_from_search(query)
│   └─ discovery.search() → 找种子 URL
│       └─ 对每个站点 crawl_topic()
│           ├─ LinkScorer: 7 维度评分链接      ✅
│           ├─ ContentQuality: 评估页面质量    ✅
│           └─ 自适应深度: 高质量深入 / 低质停  ✅
│
└─ crawl_topic(seed_url, topic)
    └─ 主题聚焦: 只爬和 topic 相关的链接       ✅
```

---

## 四、生命系统 (Life System) ❤️

```
LifeScheduler (60s 心跳) ❤️
│
├─ _decide_action()     # 根据需求决定行动
│   ├─ hunger >= 70    → "feed"
│   ├─ fatigue >= 80   → "sleep"
│   ├─ curiosity >= 70  → "explore"
│   ├─ curiosity >= 85  → "research"          ✅ B1 修复
│   └─ boredom >= 60   → "play"
│
├─ FeedEngine.execute()
│   └─ 数据摄取 → 待消化队列
│       └─ SleepEngine Phase 2 消费            ✅
│
├─ SleepEngine.sleep()       # 六阶段睡眠
│   │
│   ├─ Phase 1: 记忆整理
│   │   └─ ContextManager.consolidate_for_sleep()
│   │
│   ├─ Phase 2: 模型训练                        ✅
│   │   ├─ DataCollector.load_as_training_data()
│   │   ├─ 加载 weakness_training_data/         ✅ B5 修复
│   │   ├─ FeedEngine.get_pending_samples()
│   │   ├─ _run_sleep_training() → 在线微调
│   │   └─ app_state.update_model(model)        ✅ B6 修复
│   │
│   ├─ Phase 3: 知识整合
│   │   └─ KnowledgeStore → 语义记忆           ✅
│   │
│   ├─ Phase 3.5: 知识蒸馏
│   │   └─ KnowledgeToIntelligence.start_intelligence_boost()
│   │       └─ → training_data/*.jsonl          ✅
│   │
│   ├─ Phase 4: 自我评估
│   │   ├─ SelfEvaluator.get_stats()            ✅
│   │   └─ health_report.json
│   │
│   └─ Phase 5: 递归改进                        ✅
│       ├─ RecursiveImprover.analyze_and_improve()
│       │   └─ 全局单例 (历史不丢)             ✅ B4 修复
│       ├─ EvolutionEngine.check_evolution_ready()
│       ├─ design_next_generation() → next_gen_design.json
│       └─ _generate_weakness_training_data()
│           └─ → weakness_training_data/        ✅ (→ Phase 2 消费)
│
├─ ExploreEngine.explore()
│   ├─ tool_registry.search()                   ✅ (联网搜索)
│   ├─ _store_knowledge() → 本地知识文件        ✅
│   └─ _record_to_evolution() → evolution_engine ✅
│
├─ PlayEngine.play()
│   └─ 随机创作/代码生成/写诗...
│       └─ 存入 play_data/
│
└─ ScienceEngine (curiosity>85 触发)            ✅ B1 修复
    ├─ propose_hypothesis()
    ├─ run_experiment()
    └─ draw_conclusion()
```

---

## 五、知识学习闭环

```
对话中的网页内容                                      知识蒸馏
─────────────────────────────────────────────────   ─────────────────
NativeAgent._auto_search_context()                 SleepEngine Phase 3.5
│                                                   │
├─ search_deep → 抓取网页                            ├─ KnowledgeToIntelligence
├─ 提取正文                                          │   .start_intelligence_boost()
│                                                   │
└─ _background_learn()                              ├─ knowledge_distilled_*.jsonl
    └─ KnowledgeLearner                               │
        └─ → KnowledgeStore (结构化知识)              └─→ SleepEngine Phase 2
                                                         训练数据消费       ✅
```

---

## 六、前端 API 对接

```
📡 FastAPI 路由
│
├─ routes_chat.py        → POST /api/chat          (SSE 流式)
│                           └─ WebSocket (终端)
│
├─ routes_agent.py       → POST /api/agent/react   (ReAct 任务)
│                           POST /api/agent/collaborate (Multi-Agent)
│
├─ routes_life.py        → GET  /api/life/status   (生命状态 REST)
│                           └─ WebSocket (实时推送) ✅
│
├─ routes_training.py?   → SSE 训练进度流           ✅
│   (api/training/stream.py)
│
├─ training_service.py   → trigger/pause/resume/stop ✅ B7 修复
│
├─ runtime_service.py    → GET /api/runtime/status  (运行状态)
│
├─ settings_service.py   → GET/POST /api/settings   (配置)
│
└─ tool_service.py       → GET /api/tools           (工具列表)
```

---

## 七、安全系统

```
用户输入 / Agent 输出
│
├─ PiiSanitizer: 脱敏个人隐私信息
│
├─ SandboxSecurity: 工具调用前的路径/命令白名单
│   ├─ sandbox_executor (代码执行)
│   └─ limbs.py (文件操作)
│
├─ ConstitutionalAI.critique()                     ✅
│   ├─ 8 条安全原则检查
│   ├─ 违规 → revised_response 替换原始输出
│   └─ 记录 critique_log
│
└─ SecurityGuard: 速率限制 / IP 黑名单
```

---

## 八、进化系统 (自我超越)

```
SelfModificationEngine (策略级)        EvolutionEngine (模型级)
─────────────────────────────────      ─────────────────────────
NativeAgent.run()                      SleepEngine Phase 5
│                                      │
├─ evaluate_response() 每步            ├─ check_evolution_ready()
│   └─ 积累 _recent_evaluations        │   └─ 判断是否该进化
│                                      │
├─ check_rollback()    每 3 步        ├─ design_next_generation()
│   └─ 评估骤降? → 回滚策略            │   └─ → next_gen_design.json
│                                      │
└─ batch_apply_improvements() 每 5 步  ├─ execute_generation_transition()
    └─ 分析评估 → 优化 prompt/temp      │   ├─ DistillationTrainer.distill()
                                       │   ├─ save_model(student)
                                       │   └─ app_state.update_model(student) ✅
                                       │
                                       └─ auto_generation_transition=False
                                           (安全锁，需手动开启)
```

---

## 九、关键闭环确认清单

| # | 环路 | 状态 |
|---|---|---|
| 1 | 搜索 → 抓取 → 入索引 → 下次本地命中 | ✅ |
| 2 | 智能爬虫 → 链接评分 → 主题聚焦 → 自适应 | ✅ |
| 3 | 心跳 → feed → sleep Phase 2 训练 → update_model | ✅ B6 |
| 4 | Phase 4 评估 → Phase 5 改进 → 弱点数据 → Phase 2 消费 | ✅ B5 |
| 5 | 评估 → RecursiveImprover (全局单例) | ✅ B3/B4 |
| 6 | curiosity>85 → ScienceEngine (假设→实验→发现) | ✅ B1 |
| 7 | 文档上传 → RAG 向量化 → 检索 → 注入对话 | ✅ |
| 8 | 对话网页内容 → 后台知识提取 → 蒸馏 → 训练 | ✅ |
| 9 | 推理 → ConstitutionalAI 审查 → 违规修正 | ✅ |
| 10 | MCP 工具 → 桥接 → 注册 → Agent 可用 | ✅ |
| 11 | 多模态图片/语音生成 → Agent 工具注册 | ✅ |
| 12 | 训练 API: trigger/pause/resume/stop | ✅ B7 |
| 13 | 进化: design → execute → update_model (安全锁保护) | ✅ |
| 14 | ReactEngine: 空 content 不无限循环 | ✅ |
| 15 | WorkflowEngine: agent 节点执行工具 | ✅ |
