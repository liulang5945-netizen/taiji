# Taiji Entry Points

Taiji has five canonical entry points for different use cases:

## 1. FastAPI Backend (Primary)

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

The main application server. Provides REST API + WebSocket. The lifespan manager auto-loads the model on startup.

- API docs: `http://127.0.0.1:8000/docs`
- WebSocket: `ws://127.0.0.1:8000/ws`

## 2. Desktop App

```bash
python desktop/main.py
```

PyQt6 desktop application with QWebEngineView embedding the Vue frontend. Spawns the uvicorn backend (port 8000) and WebSocket server (port 8765) as subprocesses.

## 3. Standalone WebSocket Server

```bash
python start_taiji.py
```

Lightweight WebSocket server on port 8765. Handles chat, feed, train, sleep, play, voice, and status endpoints.

## 4. Frontend Dev Server

```bash
cd frontend
npm install
npm run dev
```

Vue 3 development server at `http://127.0.0.1:5173`. Proxies `/api` to port 8000 and `/ws` to the WebSocket server.

## 5. Training

```bash
python scripts/native_v2/pretrain.py --config scripts/native_v2/pretrain_config_phase_a.json
```

Runs the pretraining pipeline. Supports streaming datasets, checkpoint resume, data replay, and oversample protection.

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TAIJI_HOST` | `127.0.0.1` | Server bind address |
| `TAIJI_PORT` | `8000` | Server port |
| `TAIJI_DATA_DIR` | `./taiji_data` | Writable data directory |
| `TAIJI_LOG_LEVEL` | `INFO` | Logging level |
| `TAIJI_API_KEYS` | (empty) | Comma-separated API keys; empty = no auth |
| `TAIJI_ALLOWED_ORIGINS` | (empty) | CORS allowed origins |
| `HF_ENDPOINT` | `https://huggingface.co` | HuggingFace endpoint |

Copy `.env.example` to `.env` to configure.

## Startup Sequence

1. FastAPI lifespan starts → background thread calls `load_model_on_startup()`
2. Model loader checks settings → loads ModelSelf or starts empty
3. If training resume is configured → loads optimizer/scheduler state
4. RAG index is built from available documents
5. Life system scheduler starts in background (60s heartbeat)
