"""Life service — life scheduler status and manual actions.

Provides read-only access to life state without triggering side effects.
Used by runtime_service.py for status reporting.
"""
from typing import Any


def get_life_status() -> dict:
    """Return life scheduler status.

    Returns:
        {
            "status": "ok" | "error",
            "is_running": bool,
            "needs": dict,
            "total_interactions": int,
            "uptime_seconds": int,
        }
    """
    from taiji.life.life_scheduler import get_life_scheduler

    scheduler = get_life_scheduler()
    status = scheduler.get_status()
    needs = scheduler.needs.to_dict() if hasattr(scheduler, "needs") else {}
    return {
        "status": "ok",
        "is_running": status.get("is_running", False),
        "life_state": status.get("life_state", "idle"),
        "needs": needs,
        "dominant_need": status.get("dominant_need", ""),
        "last_heartbeat": status.get("last_heartbeat"),
        "last_activity": status.get("last_activity"),
        "total_interactions": status.get("total_interactions", 0),
        "uptime_seconds": status.get("uptime_seconds", 0),
    }
