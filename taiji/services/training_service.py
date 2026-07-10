"""Training service — training lock and status.

Provides read-only access to training state without triggering side effects.
Used by runtime_service.py for status reporting.

B7 修复：新增主动触发训练和睡眠的接口。
"""
from taiji.core.app_state import app_state


def get_training_status() -> dict:
    """Return current training flags."""
    return {
        "is_training": bool(app_state.is_training),
        "publishing": bool(app_state.publishing),
        "pause_requested": bool(app_state.pause_training_requested),
        "stop_requested": bool(app_state.stop_training_requested),
    }


def trigger_sleep_training() -> dict:
    """B7 修复：主动触发一次睡眠训练。

    在生命调度器未启动时也能手动触发训练。
    """
    try:
        from taiji.life.sleep_engine import get_sleep_engine
        engine = get_sleep_engine()
        if engine.is_sleeping():
            return {"success": False, "message": "Already sleeping"}
        engine.sleep(duration_minutes=5)
        return {"success": True, "message": "Sleep training triggered"}
    except Exception as e:
        return {"success": False, "message": str(e)}


def pause_training() -> dict:
    """暂停当前训练"""
    try:
        app_state.pause_training_requested = True
        return {"success": True}
    except Exception as e:
        return {"success": False, "message": str(e)}


def resume_training() -> dict:
    """恢复训练"""
    try:
        app_state.pause_training_requested = False
        return {"success": True}
    except Exception as e:
        return {"success": False, "message": str(e)}


def stop_training() -> dict:
    """停止当前训练"""
    try:
        app_state.stop_training_requested = True
        return {"success": True}
    except Exception as e:
        return {"success": False, "message": str(e)}
