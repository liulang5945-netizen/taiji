"""Runtime status service.

This service is the shared read model for the client shell. It aggregates
existing state without starting heavyweight work from polling requests.
"""
import os
import time
from typing import Any, Callable

from taiji.core.app_state import app_state
from taiji.core.memory_watchdog import get_memory_status_dict
from taiji.core.utils import get_external_path


def _safe(name: str, fn: Callable[[], Any], fallback: Any) -> Any:
    try:
        return fn()
    except Exception as exc:
        if isinstance(fallback, dict):
            return {
                **fallback,
                "status": "error",
                "source": name,
                "message": str(exc),
            }
        return {
            "status": "error",
            "source": name,
            "message": str(exc),
        }


def get_runtime_health() -> dict:
    model_info = app_state.get_model_info()
    switch = app_state.get_switch_status()
    download = _get_startup_download()

    if not app_state.startup_complete and download.get("active"):
        return {
            "state": "downloading",
            "message": download.get("message") or "Model is downloading",
            "model_loaded": False,
            "model_name": model_info.get("model_name") or "",
            "is_taiji": app_state.is_taiji(),
            "startup_complete": app_state.startup_complete,
            "startup_error": app_state.startup_error or "",
            "switch": switch,
            "download": download,
            "checked_at": int(time.time() * 1000),
        }

    if app_state.startup_error:
        state = "error"
        message = app_state.startup_error
    elif switch.get("status") == "switching":
        state = "loading"
        message = switch.get("message") or "Model is switching"
    elif app_state.startup_complete:
        state = "connected"
        message = "Local runtime is connected"
    else:
        state = "loading"
        message = "Local runtime is starting"

    return {
        "state": state,
        "message": message,
        "model_loaded": bool(model_info.get("loaded")),
        "model_name": model_info.get("model_name") or "",
        "is_taiji": app_state.is_taiji(),
        "startup_complete": app_state.startup_complete,
        "startup_error": app_state.startup_error or "",
        "switch": switch,
        "download": download,
        "checked_at": int(time.time() * 1000),
    }


def _get_startup_download() -> dict:
    try:
        from taiji.core.model_loader import startup_download_progress

        return dict(startup_download_progress)
    except Exception:
        return {
            "active": False,
            "percent": 0,
            "message": "",
            "total_mb": 0,
            "downloaded_mb": 0,
        }


def get_auth_status(authorization_header: str = "") -> dict:
    from taiji.core.security import AuthManager

    auth = AuthManager()
    status = auth.get_status()
    token = ""
    if authorization_header.startswith("Bearer "):
        token = authorization_header[7:]

    authenticated = not status.get("enabled", False)
    token_valid = False
    if token:
        token_valid = bool(auth.verify_token(token))
        authenticated = token_valid

    return {
        "enabled": bool(status.get("enabled", False)),
        "authenticated": authenticated,
        "token_valid": token_valid,
        "username": status.get("username") or "",
        "has_password": bool(status.get("has_password", False)),
    }


def get_life_status() -> dict:
    from taiji.life.life_scheduler import get_life_scheduler

    scheduler = get_life_scheduler()
    status = scheduler.get_status()
    needs = scheduler.needs.to_dict() if hasattr(scheduler, "needs") else {}
    return {
        "status": "ok",
        "is_running": status.get("is_running", False),
        "needs": needs,
        "total_interactions": status.get("total_interactions", 0),
        "uptime_seconds": status.get("uptime_seconds", 0),
    }


def get_tool_status() -> dict:
    from taiji.agent_ext.tool_registry import registry

    tools = []
    seen = set()
    for tool in registry.list_tools(enabled_only=True):
        seen.add(tool.name)
        tools.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "source": tool.source,
            "source_id": tool.source_id,
            "category": tool.category,
            "enabled": tool.enabled,
        })

    mcp_error = ""
    try:
        # NOTE: get_all_mcp_tools() may trigger MCP server connections
        # if they haven't been initialized yet. This is a known side effect
        # that should be addressed by making MCP initialization lazy.
        from taiji.agent_ext.mcp_manager import mcp_manager

        for tool in mcp_manager.get_all_mcp_tools():
            name = tool.get("name") or tool.get("function", {}).get("name")
            if not name or name in seen:
                continue
            seen.add(name)
            tools.append({
                "name": name,
                "description": tool.get("description") or tool.get("function", {}).get("description", ""),
                "parameters": tool.get("parameters") or tool.get("function", {}).get("parameters", {}),
                "source": "mcp",
                "source_id": tool.get("server_id") or tool.get("source_id", ""),
                "category": "MCP",
                "enabled": True,
            })
    except Exception as exc:
        mcp_error = f"MCP tools are unavailable: {exc}"

    plugins_dir = get_external_path("plugins")
    if os.path.exists(plugins_dir):
        for filename in os.listdir(plugins_dir):
            if not filename.endswith(".py") or filename.startswith("_"):
                continue
            name = filename[:-3]
            if name in seen:
                continue
            seen.add(name)
            tools.append({
                "name": name,
                "description": "Dynamic hot-loaded plugin",
                "parameters": {},
                "source": "plugin",
                "source_id": name,
                "category": "plugin",
                "enabled": True,
            })

    return {
        "status": "partial" if mcp_error else "ok",
        "tools": tools,
        "count": len(tools),
        "error": mcp_error,
    }


def get_training_status() -> dict:
    return {
        "is_training": bool(app_state.is_training),
        "publishing": bool(app_state.publishing),
        "pause_requested": bool(app_state.pause_training_requested),
        "stop_requested": bool(app_state.stop_training_requested),
    }


def get_runtime_status(authorization_header: str = "") -> dict:
    return {
        "status": "ok",
        "timestamp": int(time.time() * 1000),
        "health": _safe("health", get_runtime_health, {"state": "error", "message": ""}),
        "memory": _safe("memory", get_memory_status_dict, {"status": "error"}),
        "auth": _safe(
            "auth",
            lambda: get_auth_status(authorization_header),
            {"enabled": False, "authenticated": True},
        ),
        "life": _safe("life", get_life_status, {"status": "error", "is_running": False, "needs": {}}),
        "tools": _safe("tools", get_tool_status, {"status": "error", "tools": [], "count": 0, "error": ""}),
        "training": _safe("training", get_training_status, {}),
    }


def get_bootstrap_status() -> dict:
    """Public endpoint — minimal info, no auth required.

    Returns just enough for the client shell to decide whether to
    show a login screen or proceed to load the full runtime status.
    No side effects — does not trigger MCP connections, model loading, etc.
    """
    from taiji.core.security import AuthManager

    auth = AuthManager()
    auth_status = auth.get_status()
    auth_enabled = bool(auth_status.get("enabled", False))

    return {
        "alive": True,
        "auth_enabled": auth_enabled,
        "need_login": auth_enabled,  # client should show login if auth is on
        "startup_complete": bool(app_state.startup_complete),
        "startup_error": app_state.startup_error or "",
    }
