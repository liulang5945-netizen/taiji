"""Model service — model lifecycle status.

Provides read-only access to model state without triggering side effects.
Used by runtime_service.py for health reporting.
"""
from taiji.core.app_state import app_state


def get_model_info() -> dict:
    """Return model metadata (name, loaded, type)."""
    return app_state.get_model_info()


def get_switch_status() -> dict:
    """Return model switching status."""
    return app_state.get_switch_status()


def get_startup_download() -> dict:
    """Return startup download progress (safe, no side effects)."""
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


def get_health_state() -> dict:
    """Return health state for runtime status.

    Returns:
        {
            "state": "connected" | "loading" | "downloading" | "error",
            "message": str,
            "model_loaded": bool,
            "model_name": str,
            "is_taiji": bool,
            "startup_complete": bool,
            "startup_error": str,
            "switch": dict,
            "download": dict,
        }
    """
    import time

    model_info = get_model_info()
    switch = get_switch_status()
    download = get_startup_download()

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
