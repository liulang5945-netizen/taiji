"""Test fixtures shared by the suite."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TAIJI_BASE_DIR", os.path.abspath("."))


@pytest.fixture
def client():
    """Create a FastAPI test client without startup side effects."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch is not installed; skipping API client tests")

    from fastapi.testclient import TestClient
    from api.app import create_app

    return TestClient(create_app(startup_tasks=False))


@pytest.fixture
def sample_jsonl(tmp_path):
    """Create a temporary JSONL dataset."""
    path = tmp_path / "test.jsonl"
    data = [{"instruction": f"question{i}", "output": f"answer{i}"} for i in range(20)]
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in data), encoding="utf-8")
    return str(path)


@pytest.fixture
def sample_alpaca(tmp_path):
    """Create a temporary Alpaca-style dataset."""
    path = tmp_path / "alpaca.json"
    data = [
        {
            "instruction": "Explain machine learning",
            "input": "",
            "output": "Machine learning is a branch of AI.",
        },
        {
            "instruction": "Write a sorting function",
            "input": "Python",
            "output": "def sort(arr): ...",
        },
    ] * 10
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)


@pytest.fixture
def tmp_persist_dir(tmp_path):
    """Create a temporary persistence directory."""
    persist_dir = tmp_path / "persist"
    persist_dir.mkdir()
    return str(persist_dir)
