"""Canonical model loading pipeline for startup, hot reload, and auto-reload."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import threading
import time
from typing import Any

from taiji.core.app_state import app_state
from taiji.core.config import TrainingConfig
from taiji.core.utils import get_external_path
from taiji.services.settings_service import load_settings, save_settings

logger = logging.getLogger("ModelLoader")

startup_download_progress: dict[str, Any] = {
    "active": False,
    "percent": 0.0,
    "message": "",
    "total_mb": 0.0,
    "downloaded_mb": 0.0,
}

_auto_reload_thread: threading.Thread | None = None
_auto_reload_running = False
_auto_reload_inflight = False


def load_model_on_startup() -> None:
    """Load the configured model stack during API startup."""
    asyncio.run(_async_load_model())


async def _async_load_model() -> None:
    app_state.mark_starting()
    try:
        config = TrainingConfig()
        config.cache_dir = get_external_path("model_cache")
        config.resume_from_checkpoint = get_external_path("final_checkpoint.pt")

        settings = load_settings()
        _apply_settings_to_config(config, settings)

        if _should_auto_download(settings):
            _perform_auto_download(config, settings)

        _detect_and_load_model(config)
        _build_rag_index()
        app_state.mark_started()
    except Exception as exc:
        import traceback

        message = f"Model loading failed: {exc}\n{traceback.format_exc()}"
        logger.error(message)
        app_state.mark_startup_failed(message)


def _apply_settings_to_config(config: TrainingConfig, settings: dict[str, Any]) -> None:
    config.device = str(settings.get("device", "auto") or "auto")
    config.model_name = str(settings.get("model_name", "") or "").strip()
    config.load_in_4bit = bool(settings.get("load_in_4bit", False))
    config.load_in_8bit = bool(settings.get("load_in_8bit", False))
    config.model_type = str(settings.get("model_type", config.model_type) or config.model_type).lower()
    config.gguf_path = str(settings.get("gguf_path", "") or "")
    config.n_gpu_layers = int(settings.get("n_gpu_layers", -1) or -1)
    config.n_ctx = int(settings.get("n_ctx", 2048) or 2048)


def _should_auto_download(settings: dict[str, Any]) -> bool:
    # 原生态极：无需自动下载外部模型
    return False


def _perform_auto_download(config: TrainingConfig, settings: dict[str, Any]) -> None:
    # 原生态极：无需自动下载外部模型
    pass


def _detect_and_load_model(config: TrainingConfig) -> None:
    if config.cache_dir:
        os.makedirs(config.cache_dir, exist_ok=True)

    model_name = getattr(config, "model_name", "") or ""
    taiji_dir = _check_taiji_dir(model_name)

    if taiji_dir:
        config.model_type = "self"
        _auto_fix_settings("self", "", model_name=model_name)
    else:
        # 原生态极：仅支持原生模型
        config.model_type = "self"

    _load_self_model(config)


def _check_taiji_dir(model_path: str) -> bool:
    if not model_path or not os.path.isdir(model_path):
        return False
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return False
    return "base_vocab_size" in payload or "num_special_tokens" in payload


def _auto_fix_settings(model_type: str, gguf_path: str, model_name: str | None = None) -> None:
    try:
        saved = load_settings()
        saved["model_type"] = model_type
        if model_name is not None:
            saved["model_name"] = model_name
        if gguf_path:
            saved["gguf_path"] = gguf_path
        else:
            saved.pop("gguf_path", None)
        save_settings(saved)
    except Exception as exc:
        logger.debug("Failed to persist auto-detected settings: %s", exc)


def _load_self_model(config: TrainingConfig) -> None:
    try:
        from taiji import NativeInferenceEngine, TaijiMultimodalEngine, load_model
        from taiji.agent_ext.tool_registry import registry
        from taiji.core.taiji_bridge import get_taiji_bridge
    except ImportError as exc:
        logger.warning("Native self-model stack unavailable: %s", exc)
        app_state.mark_started()
        return

    model_path = getattr(config, "model_name", "") or ""
    if not model_path or not os.path.isdir(model_path):
        logger.info("No valid self-model directory detected; starting without a loaded model")
        app_state.mark_started()
        return

    device = config.resolve_device()
    model, tokenizer = load_model(model_path, device=device)

    tool_names = [tool.name for tool in registry.list_tools(enabled_only=True)]
    if hasattr(tokenizer, "register_tool"):
        for name in tool_names:
            tokenizer.register_tool(name)
        if hasattr(tokenizer, "_tool_name_to_id"):
            model.set_num_tools(len(tokenizer._tool_name_to_id))
    elif hasattr(model, "set_num_tools"):
        model.set_num_tools(len(tool_names))

    taiji = None
    try:
        taiji = TaijiMultimodalEngine(
            model,
            tokenizer,
            device=device,
            workspace_path=get_external_path("agent_workspace"),
            memory_save_path=get_external_path(os.path.join("taiji", "user_data", "memory.json")),
        )
        taiji.register_tools(tool_names)
    except Exception as exc:
        logger.warning("Failed to initialize Taiji multimodal engine: %s", exc)

    trainer = NativeInferenceEngine(model, tokenizer, device)
    app_state.update_model(model, tokenizer, trainer, model_path)
    app_state.set_taiji_engine(taiji)

    try:
        bridge = get_taiji_bridge()
        bridge.initialize(model=model, tokenizer=tokenizer, device=str(device))
        bridge.start_life()
        app_state.set_taiji_bridge(bridge)
    except Exception as exc:
        logger.warning("Failed to initialize Taiji bridge: %s", exc)


def refresh_taiji_tools() -> None:
    try:
        from taiji.agent_ext.tool_registry import registry
        from taiji.core.taiji_bridge import get_taiji_bridge

        bridge = get_taiji_bridge()
        if not bridge.is_initialized:
            return

        taiji = bridge.taiji
        model = taiji.body.model
        tokenizer = taiji.body.tokenizer
        if model is None or tokenizer is None or not hasattr(tokenizer, "register_tool"):
            return

        current_tools = set(getattr(tokenizer, "_tool_name_to_id", {}).keys())
        new_tools = [tool.name for tool in registry.list_tools(enabled_only=True) if tool.name not in current_tools]
        if not new_tools:
            return

        for name in new_tools:
            tokenizer.register_tool(name)
        if hasattr(tokenizer, "_tool_name_to_id"):
            model.set_num_tools(len(tokenizer._tool_name_to_id))
        logger.info("Refreshed Taiji tools: added %s", ", ".join(new_tools[:5]))
    except Exception as exc:
        logger.debug("Skipping Taiji tool refresh: %s", exc)


def _build_rag_index() -> None:
    doc_dir = get_external_path("docs")
    if not os.path.isdir(doc_dir) or app_state.rag_kb is None:
        return

    loaded_any = False
    for filename in os.listdir(doc_dir):
        file_path = os.path.join(doc_dir, filename)
        if not os.path.isfile(file_path):
            continue
        try:
            app_state.rag_kb.add_file(file_path)
            loaded_any = True
        except Exception as exc:
            logger.warning("Failed to add RAG document %s: %s", filename, exc)

    if loaded_any:
        try:
            app_state.rag_kb.rebuild_index()
        except Exception as exc:
            logger.warning("Failed to rebuild RAG index: %s", exc)


def start_auto_reload(check_interval: int = 60) -> None:
    global _auto_reload_thread, _auto_reload_running

    if _auto_reload_running:
        return

    _auto_reload_running = True

    def _loop() -> None:
        global _auto_reload_inflight

        while _auto_reload_running:
            time.sleep(check_interval)
            if app_state.model is not None or _auto_reload_inflight:
                continue
            try:
                from taiji.core.memory_watchdog import MemoryWatchdog

                status = MemoryWatchdog().status
                if status.avail_pct < 0.30:
                    continue
                _auto_reload_inflight = True
                try:
                    asyncio.run(_async_load_model())
                finally:
                    _auto_reload_inflight = False
            except Exception as exc:
                _auto_reload_inflight = False
                logger.debug("Auto reload check failed: %s", exc)

    _auto_reload_thread = threading.Thread(target=_loop, name="ModelAutoReload", daemon=True)
    _auto_reload_thread.start()
    logger.info("Model auto reload started (interval=%ss)", check_interval)


def stop_auto_reload() -> None:
    global _auto_reload_running
    _auto_reload_running = False
