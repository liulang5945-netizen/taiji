"""
Minimal tests for the runtime service.

These tests verify that the runtime status payload can be constructed
without crashing, even when the model is not loaded and torch is not
available. They do NOT require a running server.
"""
import pytest
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_get_bootstrap_status_returns_required_fields():
    """bootstrap payload must contain alive, auth_enabled, need_login."""
    from taiji.services.runtime_service import get_bootstrap_status

    result = get_bootstrap_status()
    assert "alive" in result
    assert "auth_enabled" in result
    assert "need_login" in result
    assert "startup_complete" in result
    assert "startup_error" in result


def test_get_bootstrap_status_alive_is_true():
    """If we can call the function, the runtime is alive."""
    from taiji.services.runtime_service import get_bootstrap_status

    result = get_bootstrap_status()
    assert result["alive"] is True


def test_get_runtime_status_returns_required_sections():
    """Full status must contain health, memory, auth, life, tools, training."""
    from taiji.services.runtime_service import get_runtime_status

    result = get_runtime_status()
    assert result["status"] == "ok"
    assert "timestamp" in result
    assert "health" in result
    assert "memory" in result
    assert "auth" in result
    assert "life" in result
    assert "tools" in result
    assert "training" in result


def test_get_runtime_status_health_has_state():
    """health.state must be one of the known values."""
    from taiji.services.runtime_service import get_runtime_status

    result = get_runtime_status()
    health = result["health"]
    assert "state" in health
    assert health["state"] in ("connected", "loading", "downloading", "error")


def test_get_runtime_status_tools_has_count():
    """tools.count must be a non-negative integer."""
    from taiji.services.runtime_service import get_runtime_status

    result = get_runtime_status()
    tools = result["tools"]
    assert "count" in tools
    assert isinstance(tools["count"], int)
    assert tools["count"] >= 0


def test_get_runtime_status_training_has_flags():
    """training must have is_training and publishing flags."""
    from taiji.services.runtime_service import get_runtime_status

    result = get_runtime_status()
    training = result["training"]
    assert "is_training" in training
    assert "publishing" in training


def test_get_runtime_status_no_side_effect_on_torch_import():
    """Importing runtime_service should not crash even if torch is missing."""
    # This test just verifies the import doesn't throw
    from taiji.services import runtime_service
    assert hasattr(runtime_service, "get_runtime_status")
    assert hasattr(runtime_service, "get_bootstrap_status")
