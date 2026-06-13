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
    from taiji.services import runtime_service
    assert hasattr(runtime_service, "get_runtime_status")
    assert hasattr(runtime_service, "get_bootstrap_status")


# ===== Domain service tests =====

def test_tool_service_list_tools():
    """tool_service.list_tools() returns tools dict with count."""
    from taiji.services.tool_service import list_tools
    result = list_tools()
    assert "tools" in result
    assert "count" in result
    assert isinstance(result["tools"], list)
    assert result["count"] == len(result["tools"])


def test_auth_service_get_status():
    """auth_service.get_status() returns auth status dict."""
    from taiji.services.auth_service import get_status
    result = get_status()
    assert "enabled" in result
    assert isinstance(result["enabled"], bool)


def test_auth_service_get_authenticated_status():
    """auth_service.get_authenticated_status() returns full auth info."""
    from taiji.services.auth_service import get_authenticated_status
    result = get_authenticated_status()
    assert "enabled" in result
    assert "authenticated" in result
    assert "token_valid" in result
    assert "username" in result
    assert "has_password" in result


def test_model_service_get_health_state():
    """model_service.get_health_state() returns health dict."""
    from taiji.services.model_service import get_health_state
    result = get_health_state()
    assert "state" in result
    assert result["state"] in ("connected", "loading", "downloading", "error")


def test_training_service_get_training_status():
    """training_service.get_training_status() returns training flags."""
    from taiji.services.training_service import get_training_status
    result = get_training_status()
    assert "is_training" in result
    assert "publishing" in result
    assert isinstance(result["is_training"], bool)


def test_life_service_get_life_status():
    """life_service.get_life_status() returns life status dict."""
    from taiji.services.life_service import get_life_status
    result = get_life_status()
    assert "status" in result
    assert "is_running" in result
    assert "needs" in result


def test_agent_service_import():
    """agent_service can be imported without side effects."""
    from taiji.services import agent_service
    assert hasattr(agent_service, "run_react_task")
    assert hasattr(agent_service, "cancel_active_task")
    assert hasattr(agent_service, "list_roles")


def test_taiji_model_service_import():
    """taiji_model_service can be imported without side effects."""
    from taiji.services import taiji_model_service
    assert hasattr(taiji_model_service, "is_available")
    assert hasattr(taiji_model_service, "get_status")


def test_pydantic_schema_validation():
    """RuntimeStatusPayload validates correctly."""
    from api.models_runtime import RuntimeStatusPayload
    payload = RuntimeStatusPayload()
    assert payload.status == "ok"
    assert payload.health.state == "loading"
    assert payload.auth.enabled is False
    assert payload.training.is_training is False


def test_pydantic_bootstrap_validation():
    """BootstrapPayload validates correctly."""
    from api.models_runtime import BootstrapPayload
    payload = BootstrapPayload()
    assert payload.alive is True
    assert payload.auth_enabled is False
    assert payload.need_login is False
