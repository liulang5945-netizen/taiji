# 态极安装与本地运行

这份文档只覆盖当前仓库仍然有效的本地安装与启动方式。

## Environment

- Python `3.10+`
- Node.js for `frontend/`
- optional NVIDIA CUDA environment for GPU inference or training

## Install

```bash
git clone <repository-url>
cd taiji
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install Python package dependencies:

```bash
pip install -e .
```

Install frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

## Run

### Backend only

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

### Frontend dev

```bash
cd frontend
npm run dev
```

Frontend dev default:

- `http://127.0.0.1:5173`

Backend default:

- `http://127.0.0.1:8000`

### Desktop dev

```bash
python desktop/main.py
```

### Standalone WebSocket server

```bash
python start_taiji.py
```

Use this only when a workflow actually needs the separate WebSocket server.

## Verify

```bash
python -m pytest tests/ -v
```

Health endpoint:

```bash
curl http://127.0.0.1:8000/api/health
```

## Notes

- `api/app.py` is the canonical FastAPI app definition
- `desktop/main.py` is the development desktop entry
- `api/run_app.py` is for the packaged desktop path
- training documentation lives in `docs/TAIJI_1B_12B_TOKEN_TRAINING_PLAN_CN.md`
- tokenizer documentation lives in `docs/NATIVE_V2_TOKENIZER.md`

## Common Problems

- `ModuleNotFoundError`
  - activate the virtual environment and rerun `pip install -e .`
- frontend cannot reach backend
  - ensure backend is running on `127.0.0.1:8000`
- CUDA not available
  - verify the installed PyTorch build matches your CUDA environment
