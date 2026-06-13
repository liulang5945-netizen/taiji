# 态极 (Taiji) 架构

## 架构概述

```
Vue3 前端 ← HTTP/SSE → FastAPI 后端 → 态极核心 (taiji/)
```

### 核心设计

1. **REST API** — FastAPI 提供 HTTP 接口，SSE 流式聊天
2. **生命系统** — 态极有自主生命活动（吃饭/睡觉/玩耍/进化）
3. **本地运行** — 数据不出本机，保护隐私
4. **模块化** — 大脑、身体、生命系统独立可配置
5. **自训练** — 支持从零预训练和微调

## 文件结构

```
taiji/
├── taiji/                      # 核心 AI 生命系统
│   ├── __init__.py             # TaijiCore 统一入口
│   ├── config.py               # 模型配置 (125M/350M/1B/3B/7B)
│   ├── architecture.py         # 神经网络架构 (LLaMA 3 风格)
│   ├── layers.py               # 基础层 (RMSNorm, RoPE, GQA, SwiGLU)
│   ├── loader.py               # 模型加载/保存
│   ├── tokenizer.py            # SentencePiece 分词器
│   ├── brain/                  # 大脑系统 (cortex)
│   ├── body/                   # 身体系统 (资源管理/工具执行)
│   ├── life/                   # 生命系统 (吃饭/睡觉/玩耍/进化)
│   ├── core/                   # 核心运行时 (推理/Agent/WebSocket)
│   ├── agent/                  # Agent 子系统 (记忆/感知/规划/反思)
│   ├── train/                  # 训练系统 (预训练/微调/DPO)
│   ├── multimodal/             # 多模态 (视觉/语音/视频)
│   ├── safety/                 # 安全系统 (输入过滤/输出消毒)
│   ├── infra/                  # 基础设施 (事件总线/自动升级)
│   ├── core/                   # 从 omnicore 移植的核心模块
│   │   ├── config.py           # 训练配置
│   │   ├── app_state.py        # 全局状态管理
│   │   ├── utils.py            # 工具函数
│   │   ├── security.py         # JWT 认证/安全存储
│   │   ├── model_loader.py     # 模型加载器
│   │   ├── hardware.py         # 硬件检测
│   │   └── memory_watchdog.py  # 内存监控
│   ├── agent_ext/              # 从 omnicore 移植的 Agent 模块
│   ├── model_ext/              # 从 omnicore 移植的模型管理模块
│   └── tools/                  # 工具 (RAG/文件解析)
│
├── api/                        # FastAPI 后端
│   ├── app.py                  # FastAPI 应用入口
│   ├── routes_chat.py          # SSE 流式聊天
│   ├── routes_models.py        # 模型管理
│   ├── routes_taiji.py         # 态极专属 API (喂养/睡眠/玩耍)
│   ├── routes_agent.py         # Agent 模式
│   ├── routes_rag.py           # RAG 知识库
│   ├── routes_training.py      # 训练控制
│   ├── routes_settings.py      # 系统设置
│   └── training/               # 训练 API 路由
│
├── frontend/                   # Vue3 Web 前端
│   ├── src/
│   │   ├── App.vue             # 根组件
│   │   ├── components/         # 通用组件 (ChatView, AppSidebar)
│   │   ├── views/              # 页面视图
│   │   ├── composables/        # Vue 组合式函数
│   │   ├── stores/             # Pinia 状态管理
│   │   └── assets/styles/      # CSS (毛玻璃暗色主题)
│   └── package.json            # Vue3 + Vite8 + Element Plus
│
├── taiji_data/                 # 训练数据和模型文件
│   ├── training_data/          # 训练数据 (JSONL)
│   ├── evolution_data/         # 进化数据和模型检查点
│   └── tokenizer_corpus.txt    # 分词器训练语料
│
├── tests/                      # 测试套件
├── docs/                       # 文档
├── app_settings.json           # 运行时配置
├── dev.bat                     # 开发启动脚本
└── requirements.txt            # Python 依赖
```

## 使用方法

### 开发模式

```bash
cd e:/taiji
dev.bat
# 后端: http://localhost:8000
# 前端: http://localhost:5173
```

### 单独启动

```bash
# 后端
uvicorn api.app:app --host 127.0.0.1 --port 8000

# 前端
cd frontend && npm run dev
```

### 训练

```bash
# 训练分词器
python -m taiji.train.train_tokenizer

# 预训练
python -m taiji.train.pretrain_from_scratch

# 微调
python -m taiji.train.finetune_conversation
```

## 技术栈

### 后端 (Python 3.11)
- **深度学习**: PyTorch 2.x, HuggingFace Transformers, PEFT
- **API**: FastAPI + Uvicorn + Pydantic
- **RAG**: sentence-transformers, numpy, jieba
- **语音**: edge-tts, pyttsx3, SpeechRecognition

### 前端 (Vue 3)
- **框架**: Vue 3.5 + Pinia + Vue Router
- **UI**: Element Plus 2.14
- **构建**: Vite 8
- **编辑器**: Monaco Editor 0.55
- **终端**: xterm.js 6

## 生命系统

态极有 5 个内在需求驱动自主行为：

| 需求 | 触发行为 | 引擎 |
|------|---------|------|
| 饥饿 | 进食 (数据摄取) | FeedEngine |
| 疲劳 | 睡眠 (训练/整合) | SleepEngine |
| 无聊 | 玩耍 (探索/创造) | PlayEngine |
| 压力 | 反思 (自我评估) | SelfEvaluator |
| 好奇 | 进化 (能力提升) | EvolutionEngine |

LifeScheduler 以 60 秒心跳循环监控需求，自动触发生命活动。
