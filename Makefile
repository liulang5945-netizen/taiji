.PHONY: install install-dev test lint clean dev dev-backend dev-frontend docker-up docker-down build-frontend build-desktop

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-gpu:
	pip install -e ".[gpu]"

test:
	python -m pytest tests/ -x -q --timeout=120

lint:
	ruff check taiji/ api/ scripts/

format:
	ruff format taiji/ api/ scripts/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

dev-backend:
	python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload

dev-frontend:
	cd frontend && npm run dev

dev:
	@echo "Start backend and frontend in separate terminals:"
	@echo "  make dev-backend"
	@echo "  make dev-frontend"

docker-up:
	docker compose up -d

docker-down:
	docker compose down

build-frontend:
	cd frontend && npm install && npm run build

build-desktop: build-frontend
	python desktop/build.py

smoke-train:
	python scripts/native_v2/pretrain.py --config scripts/native_v2/smoke_125m.json
