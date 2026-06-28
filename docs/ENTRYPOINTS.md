# 态极入口清单

只保留当前仍然有效的运行入口。

## Canonical Runtime Entrypoints

| file | role |
| --- | --- |
| `api/app.py` | canonical FastAPI app definition |
| `desktop/main.py` | desktop development entry |
| `api/run_app.py` | packaged desktop entry |
| `start_taiji.py` | standalone WebSocket server |
| `frontend/vite.config.js` | frontend dev/build config |

## Startup Paths

### Backend only

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

### Frontend dev

```bash
cd frontend
npm run dev
```

### Desktop dev

```bash
python desktop/main.py
```

This path starts:

- an HTTP API process for `api.app:app`
- a standalone WebSocket server via `start_taiji.py`
- the embedded desktop web view

### Packaged desktop

`api/run_app.py` is the packaged desktop entry used by the PyInstaller path.

## Rules

1. `api/app.py` is the only canonical FastAPI app definition.
2. New startup code must reuse existing app definitions instead of creating parallel apps.
3. Runtime entrypoints must stay thin and avoid business logic.
4. Legacy or duplicate startup files should be removed instead of documented as alternatives.
