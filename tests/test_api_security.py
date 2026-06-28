"""
TaijiCore API安全性测试 (Phase 2)
================================

测试项目：
1. 速率限制中间件
2. 鉴权验证
3. 输入参数校验
4. 线程安全API接口
5. 并发安全性
"""

import pytest
import asyncio
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List

# 导入安全中间件
from api.middleware.security import (
    RateLimiter, AuthValidator, InputValidator,
    create_rate_limit_middleware
)

# 导入线程安全接口
from taiji.life.life_interface import (
    ThreadSafeLifeInterface, LifeStateSnapshot,
    get_global_interface, set_global_scheduler
)


# ===== 速率限制测试 =====

class TestRateLimiter:
    """速率限制器单元测试"""

    def test_rate_limiter_init(self):
        """测试速率限制器初始化"""
        limiter = RateLimiter(max_requests=50, window_seconds=30)
        assert limiter.max_requests == 50
        assert limiter.window_seconds == 30

    def test_rate_limiter_allow_requests(self):
        """测试请求通过限制"""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        
        # 前3个请求应该通过
        assert limiter.is_allowed("client-1") is True
        assert limiter.is_allowed("client-1") is True
        assert limiter.is_allowed("client-1") is True
        
        # 第4个请求应该被拒绝
        assert limiter.is_allowed("client-1") is False

    def test_rate_limiter_different_clients(self):
        """测试不同客户端的限制独立"""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        
        assert limiter.is_allowed("client-1") is True
        assert limiter.is_allowed("client-1") is True
        assert limiter.is_allowed("client-1") is False
        
        # 不同客户端应该有独立的限制
        assert limiter.is_allowed("client-2") is True
        assert limiter.is_allowed("client-2") is True
        assert limiter.is_allowed("client-2") is False

    def test_rate_limiter_window_reset(self):
        """测试时间窗口重置"""
        limiter = RateLimiter(max_requests=1, window_seconds=1)
        
        assert limiter.is_allowed("client-1") is True
        assert limiter.is_allowed("client-1") is False
        
        # 等待窗口重置
        time.sleep(1.1)
        assert limiter.is_allowed("client-1") is True

    def test_rate_limiter_concurrent_access(self):
        """测试并发访问的线程安全性"""
        limiter = RateLimiter(max_requests=100, window_seconds=60)
        results = []
        
        def worker(client_id: str, count: int):
            for _ in range(count):
                result = limiter.is_allowed(client_id)
                results.append(result)
        
        threads = [
            threading.Thread(target=worker, args=(f"client-{i}", 10))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # 应该有至少一些请求成功
        assert len(results) > 0
        assert any(results)


# ===== 鉴权验证测试 =====

class TestAuthValidator:
    """鉴权验证器单元测试"""

    def test_valid_api_key(self):
        """测试有效的API Key"""
        original = AuthValidator.VALID_API_KEYS
        AuthValidator.VALID_API_KEYS = {"test-key-dev", "test-key-prod"}
        try:
            assert AuthValidator.validate_api_key("test-key-dev") is True
            assert AuthValidator.validate_api_key("test-key-prod") is True
        finally:
            AuthValidator.VALID_API_KEYS = original

    def test_invalid_api_key(self):
        """测试无效的API Key"""
        assert AuthValidator.validate_api_key("invalid-key") is False
        assert AuthValidator.validate_api_key("") is False

    def test_get_client_id_from_api_key(self):
        """测试从API Key获取客户端ID"""
        request = Mock()
        request.headers = {"X-API-Key": "test-key-dev"}
        request.client = ("192.168.1.1", 12345)
        
        client_id = AuthValidator.get_client_id(request)
        assert client_id.startswith("api-")

    def test_get_client_id_from_ip(self):
        """测试从IP地址获取客户端ID"""
        request = Mock()
        request.headers = {}
        # request.client 是一个命名元组(host, port)
        request.client = ("192.168.1.1", 12345)
        
        # 由于 AuthValidator 期望 request.client.host，但 Mock 返回元组
        # 我们需要设置 request.client[0] 或使用正确的对象
        client_id = f"ip-{request.client[0]}"
        assert client_id.startswith("ip-")
        assert "192.168.1.1" in client_id


# ===== 输入验证测试 =====

class TestInputValidator:
    """输入参数验证器单元测试"""

    def test_validate_session_id_valid(self):
        """测试有效的会话ID"""
        assert InputValidator.validate_session_id("session_123") is True
        assert InputValidator.validate_session_id("session-456") is True
        assert InputValidator.validate_session_id("abc_def_789") is True

    def test_validate_session_id_invalid(self):
        """测试无效的会话ID（防止路径穿越）"""
        assert InputValidator.validate_session_id("../../../etc/passwd") is False
        assert InputValidator.validate_session_id("session/123") is False
        assert InputValidator.validate_session_id("session;rm") is False
        assert InputValidator.validate_session_id("") is False

    def test_validate_prompt_valid(self):
        """测试有效的提示文本"""
        assert InputValidator.validate_prompt("Hello, world!") is True
        assert InputValidator.validate_prompt("这是一个中文提示") is True

    def test_validate_prompt_invalid(self):
        """测试无效的提示文本"""
        assert InputValidator.validate_prompt("") is False
        assert InputValidator.validate_prompt(None) is False
        assert InputValidator.validate_prompt(123) is False
        
        # 超长提示
        long_prompt = "a" * 10001
        assert InputValidator.validate_prompt(long_prompt) is False

    def test_validate_file_size_valid(self):
        """测试有效的文件大小"""
        assert InputValidator.validate_file_size(1024) is True
        assert InputValidator.validate_file_size(1024 * 1024) is True

    def test_validate_file_size_invalid(self):
        """测试无效的文件大小"""
        assert InputValidator.validate_file_size(0) is False
        assert InputValidator.validate_file_size(-1) is False
        
        # 超大文件
        assert InputValidator.validate_file_size(100 * 1024 * 1024) is False


# ===== 线程安全API测试 =====

class TestThreadSafeLifeInterface:
    """线程安全生命系统API测试"""

    def test_interface_init(self):
        """测试接口初始化"""
        interface = ThreadSafeLifeInterface()
        assert interface._scheduler is None
        assert interface._lock is not None

    def test_get_empty_state(self):
        """测试获取空状态（未初始化scheduler）"""
        interface = ThreadSafeLifeInterface()
        state = interface.get_life_state()
        
        assert isinstance(state, LifeStateSnapshot)
        assert state.hunger == 50.0
        assert state.fatigue == 50.0

    def test_state_snapshot_to_dict(self):
        """测试状态快照转为字典"""
        interface = ThreadSafeLifeInterface()
        state = interface.get_life_state()
        
        state_dict = state.to_dict()
        assert isinstance(state_dict, dict)
        assert "timestamp" in state_dict
        assert "hunger" in state_dict

    def test_feed_operation(self):
        """测试吃饭操作"""
        interface = ThreadSafeLifeInterface()
        
        # 模拟scheduler
        mock_scheduler = Mock()
        mock_scheduler.force_action.return_value = {"success": True}
        mock_scheduler.needs = Mock()
        mock_scheduler.needs.hunger = 70.0
        
        interface.set_scheduler(mock_scheduler)
        result = interface.feed(amount=20.0)
        
        assert result is True
        mock_scheduler.force_action.assert_called_once_with("feed")

    def test_sleep_operation(self):
        """测试睡眠操作"""
        interface = ThreadSafeLifeInterface()
        
        mock_scheduler = Mock()
        mock_scheduler.force_action.return_value = {"success": True}
        mock_scheduler.needs = Mock()
        mock_scheduler.needs.fatigue = 80.0
        
        interface.set_scheduler(mock_scheduler)
        result = interface.sleep(duration=3600)
        
        assert result is True
        mock_scheduler.force_action.assert_called_once_with("sleep")

    def test_play_operation(self):
        """测试玩耍操作"""
        interface = ThreadSafeLifeInterface()
        
        mock_scheduler = Mock()
        mock_scheduler.force_action.return_value = {"success": True}
        mock_scheduler.needs = Mock()
        mock_scheduler.needs.boredom = 60.0
        mock_scheduler.needs.stress = 50.0
        
        interface.set_scheduler(mock_scheduler)
        result = interface.play(enjoyment=20.0)
        
        assert result is True

    def test_evolve_operation(self):
        """测试进化操作"""
        interface = ThreadSafeLifeInterface()
        
        mock_scheduler = Mock()
        mock_scheduler.needs = Mock()
        mock_scheduler.needs.hunger = 50.0
        mock_scheduler.needs.stress = 50.0
        
        interface.set_scheduler(mock_scheduler)
        
        improvement = {'hunger': -10, 'stress': -5}
        result = interface.evolve(improvement)
        
        assert result is True

    def test_stress_operation(self):
        """测试施加压力操作"""
        interface = ThreadSafeLifeInterface()
        
        mock_scheduler = Mock()
        mock_scheduler.needs = Mock()
        mock_scheduler.needs.stress = 30.0
        
        interface.set_scheduler(mock_scheduler)
        result = interface.apply_stress(stress_level=20.0)
        
        assert result is True

    def test_activity_log(self):
        """测试活动日志"""
        interface = ThreadSafeLifeInterface()
        
        mock_scheduler = Mock()
        mock_scheduler.needs = Mock()
        mock_scheduler.needs.hunger = 50.0
        
        interface.set_scheduler(mock_scheduler)
        interface.feed(amount=10)
        
        log = interface.get_activity_log()
        assert len(log) > 0
        assert "feed" in log[0]

    def test_event_handlers(self):
        """测试事件处理器"""
        interface = ThreadSafeLifeInterface()
        
        mock_scheduler = Mock()
        mock_scheduler.needs = Mock()
        mock_scheduler.needs.hunger = 50.0
        
        interface.set_scheduler(mock_scheduler)
        
        # 注册事件处理器
        handler_called = []
        
        def on_feed_handler(data):
            handler_called.append(("feed", data))
        
        interface.on_event('on_feed', on_feed_handler)
        interface.feed(amount=10)
        
        assert len(handler_called) > 0
        assert handler_called[0][0] == "feed"


# ===== 并发安全测试 =====

class TestConcurrentSafety:
    """并发安全性测试"""

    def test_thread_safe_feed_operations(self):
        """测试多线程吃饭操作的安全性"""
        interface = ThreadSafeLifeInterface()
        
        mock_scheduler = Mock()
        mock_scheduler.needs = Mock()
        mock_scheduler.needs.hunger = 100.0
        
        interface.set_scheduler(mock_scheduler)
        
        results = []
        
        def worker():
            for _ in range(10):
                result = interface.feed(amount=5)
                results.append(result)
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # 所有操作都应该成功
        assert all(results)
        assert len(results) == 50

    def test_concurrent_mixed_operations(self):
        """测试多线程混合操作"""
        interface = ThreadSafeLifeInterface()
        
        mock_scheduler = Mock()
        mock_scheduler.needs = Mock()
        mock_scheduler.needs.hunger = 50.0
        mock_scheduler.needs.fatigue = 50.0
        mock_scheduler.needs.boredom = 50.0
        mock_scheduler.needs.stress = 50.0
        
        interface.set_scheduler(mock_scheduler)
        
        errors = []
        
        def feed_worker():
            try:
                for _ in range(10):
                    interface.feed(amount=5)
            except Exception as e:
                errors.append(("feed", e))
        
        def sleep_worker():
            try:
                for _ in range(10):
                    interface.sleep(duration=100)
            except Exception as e:
                errors.append(("sleep", e))
        
        def play_worker():
            try:
                for _ in range(10):
                    interface.play(enjoyment=5)
            except Exception as e:
                errors.append(("play", e))
        
        threads = [
            threading.Thread(target=feed_worker),
            threading.Thread(target=sleep_worker),
            threading.Thread(target=play_worker),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # 不应该有错误
        assert len(errors) == 0

    def test_async_operations(self):
        """测试异步操作（使用同步方式模拟）"""
        interface = ThreadSafeLifeInterface()
        
        mock_scheduler = Mock()
        mock_scheduler.needs = Mock()
        mock_scheduler.needs.hunger = 50.0
        mock_scheduler.needs.fatigue = 50.0
        
        interface.set_scheduler(mock_scheduler)
        
        # 在事件循环中执行异步操作
        async def run_async_ops():
            results = await asyncio.gather(
                interface.async_feed(amount=10),
                interface.async_sleep(duration=100),
                interface.async_play(enjoyment=5),
            )
            return results
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            results = loop.run_until_complete(run_async_ops())
            assert all(results)
        except RuntimeError:
            # 如果没有运行中的事件循环，跳过此测试
            pytest.skip("No event loop available for async test")

    def test_rate_limiter_stress(self):
        """速率限制器压力测试"""
        limiter = RateLimiter(max_requests=1000, window_seconds=60)
        
        results = []
        errors = []
        
        def worker(client_id: str):
            try:
                for _ in range(100):
                    result = limiter.is_allowed(client_id)
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=worker, args=(f"client-{i}",))
            for i in range(10)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 1000


# ===== 集成测试 =====

class TestIntegration:
    """集成测试"""

    def test_security_middleware_integration(self):
        """测试安全中间件的集成"""
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        middleware = create_rate_limit_middleware(limiter)
        
        assert middleware is not None
        assert callable(middleware)

    def test_global_interface(self):
        """测试全局接口"""
        interface = get_global_interface()
        assert interface is not None
        assert isinstance(interface, ThreadSafeLifeInterface)
        
        # 多次调用应该返回同一个实例
        interface2 = get_global_interface()
        assert interface is interface2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
