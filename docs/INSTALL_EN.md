# Taiji Installation Guide

## Requirements

- Python >= 3.10
- Node.js >= 20.19 (for frontend)
- CUDA-capable GPU optional (CPU fallback works for inference)
- ~2 GB disk space for dependencies

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/taiji-community/taiji.git
cd taiji

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# 3. Install Python dependencies
pip install -e .

# Optional: install GPU dependencies
pip install -e ".[gpu]"

# 4. Install frontend dependencies
cd frontend
npm install
cd ..

# 5. Start the development servers
# Terminal 1: Backend
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000

# Terminal 2: Frontend
cd frontend && npm run dev
```

Open `http://127.0.0.1:5173` in your browser.

## Running Tests

```bash
python -m pytest tests/ -x -q
```

## Docker

```bash
docker compose up -d
```

## Building the Desktop App

```bash
pip install -e ".[desktop]"
python desktop/build.py
```

The packaged executable will be in `dist/Taiji/`.

## Troubleshooting

**ImportError: torch not found**
Install PyTorch manually: `pip install torch>=2.0.0`

**Model not loaded error**
Taiji starts without a model by default. Train one first:
```bash
python scripts/native_v2/pretrain.py --config scripts/native_v2/smoke_125m.json
```

**CUDA out of memory**
Use CPU mode or reduce batch size in your config.
