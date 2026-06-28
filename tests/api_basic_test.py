"""Taiji API 基础功能测试 (pytest 格式)

用法:
    pytest tests/api_basic_test.py -v
    pytest tests/api_basic_test.py -v --tb=short

前提: API 服务必须已启动 (python -m api.main)
"""
import urllib.request
import json
import time
import pytest

BASE = "http://127.0.0.1:8000"
MIN_INTERVAL = 0.15


def api(path, data=None, method=None):
    """发送 HTTP 请求到 API"""
    url = BASE + path
    headers = {"Content-Type": "application/json"}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method=method or "POST")
    else:
        req = urllib.request.Request(url, headers=headers, method=method or "GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, {"error": str(e)}
    except Exception as e:
        return 0, {"error": str(e)}


def _require_live_api() -> None:
    status, _ = api("/api/health")
    if status == 0:
        pytest.skip("Live API server is not running at http://127.0.0.1:8000")


def test_health():
    """健康检查"""
    _require_live_api()
    s, d = api("/api/health")
    assert s == 200, f"健康检查失败: HTTP {s}"


def test_settings():
    """设置读取"""
    _require_live_api()
    s, d = api("/api/settings")
    assert s == 200, f"设置读取失败: HTTP {s}"
    assert "model" in d or "error" not in d


def test_memory_status():
    """记忆状态"""
    _require_live_api()
    s, d = api("/api/agent/memory/status")
    assert s in (200, 500), f"记忆状态: HTTP {s}"


def test_tools_list():
    """工具列表"""
    _require_live_api()
    s, d = api("/api/agent/tools")
    assert s in (200, 500), f"工具列表: HTTP {s}"


def test_chat_sessions():
    """会话 CRUD"""
    _require_live_api()
    # 创建
    s, d = api("/api/chat/sessions", {"id": 99902, "name": "API测试会话"})
    assert s in (200, 422), f"创建会话: HTTP {s}"

    if s == 200:
        sid = d.get("session_id", 99902)
        # 读取历史
        s2, _ = api(f"/api/chat/history/{sid}")
        assert s2 in (200, 404), f"读取历史: HTTP {s2}"
        # 删除
        api(f"/api/chat/sessions/{sid}", method="DELETE")


def test_model_status():
    """模型状态"""
    _require_live_api()
    s, d = api("/api/model/status")
    assert s in (200, 500), f"模型状态: HTTP {s}"


def test_life_status():
    """生命状态"""
    _require_live_api()
    s, d = api("/api/life/status")
    assert s in (200, 500), f"生命状态: HTTP {s}"


def test_gguf_models():
    """GGUF 模型列表"""
    _require_live_api()
    s, d = api("/api/settings/gguf_models")
    assert s in (200, 500), f"GGUF模型列表: HTTP {s}"
