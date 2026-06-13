"""Training service — training lock and status.

Provides read-only access to training state without triggering side effects.
Used by runtime_service.py for status reporting.
"""
from taiji.core.app_state import app_state


def get_training_status() -> dict:
    """Return current training flags.

    Returns:
        {
            "is_training": bool,
            "publishing": bool,
            "pause_requested": bool,
            "stop_requested": bool,
        }
    """
    return {
        "is_training": bool(app_state.is_training),
        "publishing": bool(app_state.publishing),
        "pause_requested": bool(app_state.pause_training_requested),
        "stop_requested": bool(app_state.stop_training_requested),
    }
