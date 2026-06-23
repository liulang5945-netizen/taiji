"""Centralized helpers for reading and writing app settings."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from typing import Any, Mapping

from taiji.core.utils import get_external_path

_SETTINGS_LOCK = threading.RLock()
_DEFAULT_SETTINGS: dict[str, Any] = {
    "device": "auto",
    "model_name": "",
    "model_type": "huggingface",
    "gguf_path": "",
    "load_in_4bit": False,
    "load_in_8bit": False,
    "n_gpu_layers": -1,
    "n_ctx": 2048,
    "workspace_path": "",
    "auth_enabled": False,
    "auth_username": "admin",
    "auth_password_hash": "",
    "terminal_allow_unauthenticated": False,
    "search_engine": "智能多核",
    "search_key": "",
    "cloud_api_key": "",
    "cloud_api_base": "",
    "rag_enable_hybrid": True,
    "rag_enable_reranker": True,
    "rag_enable_query_rewrite": False,
    "rag_dense_weight": 0.6,
    "rag_bm25_weight": 0.4,
    "rag_candidate_k": 20,
    "rag_reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}


def _to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_str(value: Any, default: str) -> str:
    if value is None:
        return default
    return str(value).strip()


def normalize_settings(settings: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize persisted settings with light schema defaults."""
    merged = dict(_DEFAULT_SETTINGS)
    if isinstance(settings, Mapping):
        merged.update(dict(settings))

    merged["device"] = _to_str(merged.get("device"), _DEFAULT_SETTINGS["device"])
    merged["model_name"] = _to_str(merged.get("model_name"), "")
    merged["model_type"] = _to_str(merged.get("model_type"), _DEFAULT_SETTINGS["model_type"]).lower() or "huggingface"
    merged["gguf_path"] = _to_str(merged.get("gguf_path"), "")
    merged["workspace_path"] = _to_str(merged.get("workspace_path"), "")
    merged["auth_username"] = _to_str(merged.get("auth_username"), _DEFAULT_SETTINGS["auth_username"])
    merged["auth_password_hash"] = _to_str(merged.get("auth_password_hash"), "")
    merged["search_engine"] = _to_str(merged.get("search_engine"), _DEFAULT_SETTINGS["search_engine"])
    merged["search_key"] = _to_str(merged.get("search_key"), "")
    merged["cloud_api_key"] = _to_str(merged.get("cloud_api_key"), "")
    merged["cloud_api_base"] = _to_str(merged.get("cloud_api_base"), "")
    merged["rag_reranker_model"] = _to_str(
        merged.get("rag_reranker_model"),
        _DEFAULT_SETTINGS["rag_reranker_model"],
    )

    merged["load_in_4bit"] = _to_bool(merged.get("load_in_4bit"), False)
    merged["load_in_8bit"] = _to_bool(merged.get("load_in_8bit"), False)
    if merged["load_in_4bit"] and merged["load_in_8bit"]:
        merged["load_in_8bit"] = False

    merged["auth_enabled"] = _to_bool(merged.get("auth_enabled"), False)
    merged["terminal_allow_unauthenticated"] = _to_bool(
        merged.get("terminal_allow_unauthenticated"),
        False,
    )
    merged["rag_enable_hybrid"] = _to_bool(merged.get("rag_enable_hybrid"), True)
    merged["rag_enable_reranker"] = _to_bool(merged.get("rag_enable_reranker"), True)
    merged["rag_enable_query_rewrite"] = _to_bool(merged.get("rag_enable_query_rewrite"), False)

    merged["n_gpu_layers"] = _to_int(merged.get("n_gpu_layers"), -1)
    merged["n_ctx"] = max(1, _to_int(merged.get("n_ctx"), 2048))
    merged["rag_candidate_k"] = max(1, _to_int(merged.get("rag_candidate_k"), 20))

    merged["rag_dense_weight"] = _to_float(merged.get("rag_dense_weight"), 0.6)
    merged["rag_bm25_weight"] = _to_float(merged.get("rag_bm25_weight"), 0.4)

    if merged["gguf_path"] and merged["model_type"] == "huggingface":
        merged["model_type"] = "gguf"

    return merged


def get_settings_path() -> str:
    """Return the canonical app settings path."""
    return get_external_path("app_settings.json")


def load_settings() -> dict[str, Any]:
    """Load settings from disk, returning an empty mapping on failure."""
    settings_path = get_settings_path()
    with _SETTINGS_LOCK:
        if not os.path.exists(settings_path):
            return normalize_settings({})
        try:
            with open(settings_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return normalize_settings(data if isinstance(data, dict) else {})
        except Exception:
            return normalize_settings({})


def save_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Persist the full settings payload atomically."""
    settings_path = get_settings_path()
    payload = normalize_settings(settings)
    with _SETTINGS_LOCK:
        os.makedirs(os.path.dirname(settings_path), exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=os.path.dirname(settings_path),
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            temp_path = handle.name
        os.replace(temp_path, settings_path)
    return payload


def update_settings(updates: Mapping[str, Any]) -> dict[str, Any]:
    """Merge updates into the current settings and persist them."""
    with _SETTINGS_LOCK:
        merged = load_settings()
        merged.update(dict(updates))
        return save_settings(merged)


def get_setting(key: str, default: Any = None) -> Any:
    """Read a single setting value."""
    return load_settings().get(key, default)
