# 态极架构

这是当前仓库的高层结构说明，只描述仍然有效的主线。

## 总体链路

```text
frontend (Vue 3)
    -> HTTP / SSE / WebSocket
api (FastAPI)
    -> taiji.services
    -> taiji.core / taiji.agent / taiji.life / taiji.tools
```

## 关键目录

- `frontend/`
  - Web UI
  - Vite dev server and production bundle
- `api/`
  - FastAPI app and route modules
  - runtime, chat, training, settings, taiji, multimodal APIs
- `taiji/core/`
  - runtime core
  - model loading, inference, hardware checks, app state, websocket server
- `taiji/services/`
  - service layer used by API routes
  - thin aggregation over runtime state and subsystems
- `taiji/agent/`
  - stable agent-facing memory, planning, perception primitives
- `taiji/agent_ext/`
  - experimental agent features such as ReAct, MCP, sandbox, workflows
- `taiji/life/`
  - feed, sleep, play, explore, evolution scheduling
- `taiji/body/`
  - body abstractions such as senses, limbs, metabolism
- `taiji/train/`
  - fine-tuning and training-side Python modules
- `taiji/multimodal/`
  - multimodal engine and tokenizer-side multimodal adapters
- `scripts/`
  - data prep, tokenizer build, native-v2 pretrain entrypoints
- `taiji_data/`
  - local training data, tokenizer artifacts, checkpoints

## Current Training Architecture

The current canonical pretraining route is:

```text
data prep
-> native-v2 tokenizer rebuild
-> scripts/native_v2/pretrain.py
-> taiji/train/finetune_taiji.py
-> later multimodal alignment
```

Important:

- native-v2 is the only active tokenizer/pretrain route
- legacy HF/Qwen-compatible training routes are not canonical
- multimodal alignment happens after the text base model is stable

## Runtime Notes

- `api/app.py` is the canonical FastAPI app definition
- `desktop/main.py` is the desktop development entry
- `api/run_app.py` is the packaged desktop entry
- `start_taiji.py` starts the standalone WebSocket server used by the desktop path

## Doc Boundaries

- startup details: `docs/ENTRYPOINTS.md`
- tokenizer contract: `docs/NATIVE_V2_TOKENIZER.md`
- install and local run: `docs/INSTALL.md`
- training plan: `docs/TAIJI_1B_12B_TOKEN_TRAINING_PLAN_CN.md`
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
