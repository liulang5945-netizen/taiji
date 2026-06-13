# 态极 · 核心模块边界

> 每个包的职责、允许的依赖方向、禁止的反模式。

## 依赖方向

```
api (routes)  →  taiji.services  →  taiji.{core,agent,agent_ext,life,body,tools}
                                     ↑
                                     不反向依赖 services
```

**规则：上层可以依赖下层，下层不能依赖上层。**

---

## `taiji.core` — 基础设施

**职责：** 配置、应用状态、推理引擎、安全、模型加载。

| 文件 | 职责 |
|------|------|
| `app_state.py` | 全局应用状态单例 |
| `config.py` | 路径、环境变量、训练配置 |
| `inference.py` | 原生推理引擎（PyTorch） |
| `cuda_inference.py` | CUDA 推理引擎 |
| `hybrid_engine.py` | 混合推理引擎 |
| `model_loader.py` | 模型加载、下载进度 |
| `security.py` | AuthManager、JWT、审计日志 |
| `memory_watchdog.py` | 系统内存监控 |
| `hardware.py` | 硬件检测 |
| `quantization.py` | 模型量化 |
| `tokenizer_compat.py` | Tokenizer 兼容层 |
| `utils.py` | 路径工具函数 |
| `plugin_manager.py` | 插件管理 |
| `websocket_server.py` | WebSocket 服务器 |

**禁止：**
- 不导入 `taiji.services`
- 不导入 `api.*`
- `app_state.py` 不在 import 时创建目录、启动线程、安装依赖

---

## `taiji.services` — 服务层

**职责：** 聚合底层状态，为 API 路由提供薄代理。

| 文件 | 职责 |
|------|------|
| `runtime_service.py` | 聚合 health/memory/auth/life/tools/training |
| `auth_service.py` | 认证操作（login/enable/disable） |
| `model_service.py` | 模型生命周期状态 |
| `tool_service.py` | 工具列表（registry + MCP + plugins） |
| `training_service.py` | 训练锁和状态 |
| `life_service.py` | 生命调度器状态 |

**规则：**
- 每个 service 是纯函数聚合，不在 import 时启动任何东西
- service 可以导入 `taiji.core`、`taiji.agent`、`taiji.life` 等
- service 不导入 `api.*`

---

## `taiji.agent` — 稳定 Agent 接口

**职责：** 记忆系统、感知、规划、反思 — 稳定的核心 Agent 能力。

| 文件 | 职责 |
|------|------|
| `working_memory.py` | 工作记忆 |
| `memory.py` | 记忆条目 |
| `semantic_memory.py` | 语义记忆 |
| `perception.py` | 感知系统 |
| `planner.py` | 规划系统 |
| `reflector.py` | 反思系统 |
| `context_manager.py` | 上下文管理器 |

**规则：**
- 不导入 `taiji.agent_ext`（稳定层不依赖实验层）
- 不导入 `taiji.services` 或 `api.*`

---

## `taiji.agent_ext` — 实验能力

**职责：** MCP、多 Agent、ReAct、工具注册、自修改 — 实验性扩展。

| 文件 | 职责 |
|------|------|
| `react_engine.py` | ReAct 推理引擎 |
| `tool_registry.py` | 工具注册表 |
| `mcp_manager.py` | MCP 服务器管理 |
| `mcp_client.py` | MCP 客户端 |
| `multi_agent.py` | 多 Agent 协作 |
| `self_modification.py` | 自修改引擎 |
| `workflow_engine.py` | 工作流引擎 |
| `data_collector.py` | 数据收集器 |
| `knowledge_learner.py` | 知识学习器 |
| `token_optimizer.py` | Token 优化器 |
| `memory_manager.py` | 记忆管理器 |
| `sandbox_executor.py` | 沙箱执行器 |

**规则：**
- 可以导入 `taiji.agent`（实验层可以依赖稳定层）
- 不导入 `taiji.services` 或 `api.*`

---

## `taiji.life` — 生命系统

**职责：** 生命调度、需求系统、feed/sleep/play/evolution。

| 文件 | 职责 |
|------|------|
| `life_scheduler.py` | 生命调度器（心跳循环） |
| `feed_engine.py` | 进食引擎 |
| `sleep_engine.py` | 睡眠引擎 |
| `play_engine.py` | 玩耍引擎 |
| `explore_engine.py` | 探索引擎 |
| `evolution_engine.py` | 进化引擎 |
| `life_interface.py` | 生命接口 |
| `senses.py` | 感官系统 |
| `limbs.py` | 肢体系统 |
| `metabolism.py` | 代谢系统 |
| `body.py` | 身体核心 |

**规则：**
- 可以导入 `taiji.core`、`taiji.agent`
- 不导入 `taiji.services` 或 `api.*`

---

## `taiji.body` — 感知行动抽象

**职责：** 身体核心、感知、行动 — 不保存生命状态。

| 文件 | 职责 |
|------|------|
| `core.py` | 身体核心 |
| `senses.py` | 感官抽象 |
| `limbs.py` | 肢体抽象 |
| `metabolism.py` | 代谢抽象 |

**规则：**
- 纯抽象层，不持有状态
- 不导入 `taiji.life`（身体不反向依赖生命）

---

## `taiji.tools` — 可调用工具

**职责：** 具体工具实现（浏览器、文件解析、RAG、搜索等）。

| 文件 | 职责 |
|------|------|
| `file_parser.py` | 文件解析器 |
| `rag.py` | RAG 知识库 |
| `web.py` | Web 搜索 |
| `browser.py` | 浏览器工具 |
| `desktop.py` | 桌面工具 |
| `searxng.py` | SearXNG 搜索 |

**规则：**
- 纯工具实现，不持有全局状态
- 可以导入 `taiji.core`（用于配置/路径）

---

## 禁止的反模式

1. **循环依赖**：A 导入 B，B 导入 A
2. **反向依赖**：下层导入上层（如 `core` 导入 `services`）
3. **import 副作用**：在 `__init__.py` 或模块顶层启动线程、创建目录、安装依赖
4. **路由直接操作 app_state**：路由应通过 `taiji.services` 访问状态
5. **service 内嵌业务逻辑**：service 只聚合，不执行推理/训练/工具调用
