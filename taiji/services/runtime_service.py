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
    """Delegate to model_service — model lifecycle status."""
    from taiji.services.model_service import get_health_state
    return get_health_state()


def get_auth_status(authorization_header: str = "") -> dict:
    """Delegate to auth_service — unified auth interface."""
    from taiji.services.auth_service import get_authenticated_status
    return get_authenticated_status(authorization_header)


def get_life_status() -> dict:
    """Delegate to life_service — life scheduler status."""
    from taiji.services.life_service import get_life_status as _get
    return _get()


def get_tool_status() -> dict:
    """Delegate to tool_service — single source of truth for tool listing."""
    from taiji.services.tool_service import list_tools
    return list_tools()


def get_training_status() -> dict:
    """Delegate to training_service — training lock and status."""
    from taiji.services.training_service import get_training_status as _get
    return _get()


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
    from taiji.services.auth_service import get_status

    auth_status = get_status()
    auth_enabled = bool(auth_status.get("enabled", False))

    return {
        "alive": True,
        "auth_enabled": auth_enabled,
        "need_login": auth_enabled,  # client should show login if auth is on
        "startup_complete": bool(app_state.startup_complete),
        "startup_error": app_state.startup_error or "",
    }
