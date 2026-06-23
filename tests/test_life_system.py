"""
态极生命系统单元测试
====================

测试 LifeScheduler、NeedsState、FeedEngine、ScienceEngine、
SelfModificationEngine、ToolCallParser、Orchestrator 的核心逻辑。
"""
import pytest
import threading
import time


# ═══════════════════════════════════════════
# LifeScheduler / NeedsState 测试
# ═══════════════════════════════════════════

class TestNeedsState:
    """测试 NeedsState 边界值和核心逻辑"""

    def test_clamp_upper_bound(self):
        from taiji.life.life_scheduler import NeedsState
        needs = NeedsState()
        needs.hunger = 150
        needs.fatigue = 200
        needs.boredom = 999
        needs.stress = -50
        needs.curiosity = -10
        needs.clamp_all()
        assert needs.hunger == 100
        assert needs.fatigue == 100
        assert needs.boredom == 100
        assert needs.stress == 0
        assert needs.curiosity == 0

    def test_clamp_preserves_valid(self):
        from taiji.life.life_scheduler import NeedsState
        needs = NeedsState(hunger=50, fatigue=30, boredom=70, stress=20, curiosity=80)
        needs.clamp_all()
        assert needs.hunger == 50
        assert needs.fatigue == 30
        assert needs.boredom == 70
        assert needs.stress == 20
        assert needs.curiosity == 80

    def test_to_dict(self):
        from taiji.life.life_scheduler import NeedsState
        needs = NeedsState(hunger=30.456, fatigue=10.789)
        d = needs.to_dict()
        assert d["hunger"] == 30.5
        assert d["fatigue"] == 10.8

    def test_dominant_need(self):
        from taiji.life.life_scheduler import NeedsState
        needs = NeedsState(hunger=30, fatigue=10, boredom=90, stress=20, curiosity=50)
        assert needs.dominant_need() == "boredom"

    def test_dominant_need_tie(self):
        from taiji.life.life_scheduler import NeedsState
        needs = NeedsState(hunger=80, fatigue=80, boredom=20, stress=10, curiosity=10)
        result = needs.dominant_need()
        assert result in ("hunger", "fatigue")


class TestLifeScheduler:
    """测试 LifeScheduler 公开 API 和决策逻辑"""

    def test_initial_state(self):
        from taiji.life.life_scheduler import LifeScheduler
        scheduler = LifeScheduler()
        status = scheduler.get_status()
        assert status["life_state"] == "idle"
        assert status["is_running"] == False

    def test_add_fatigue(self):
        from taiji.life.life_scheduler import LifeScheduler
        scheduler = LifeScheduler()
        initial = scheduler.needs.fatigue
        scheduler.add_fatigue(10.0)
        assert scheduler.needs.fatigue == initial + 10.0

    def test_add_fatigue_clamped(self):
        from taiji.life.life_scheduler import LifeScheduler
        scheduler = LifeScheduler()
        scheduler.add_fatigue(200.0)
        assert scheduler.needs.fatigue == 100.0

    def test_add_hunger(self):
        from taiji.life.life_scheduler import LifeScheduler
        scheduler = LifeScheduler()
        scheduler.add_hunger(5.0)
        snapshot = scheduler.get_needs_snapshot()
        assert snapshot["hunger"] >= 35.0  # initial 30 + 5

    def test_decide_action_idle(self):
        from taiji.life.life_scheduler import LifeScheduler
        scheduler = LifeScheduler()
        # 默认状态不应触发任何行动
        assert scheduler._decide_action() is None

    def test_decide_action_feed(self):
        from taiji.life.life_scheduler import LifeScheduler
        scheduler = LifeScheduler()
        scheduler.needs.hunger = 80  # > HUNGER_THRESHOLD (70)
        assert scheduler._decide_action() == "feed"

    def test_decide_action_sleep(self):
        from taiji.life.life_scheduler import LifeScheduler
        scheduler = LifeScheduler()
        scheduler.needs.fatigue = 85  # > FATIGUE_THRESHOLD (80)
        assert scheduler._decide_action() == "sleep"

    def test_decide_action_explore(self):
        from taiji.life.life_scheduler import LifeScheduler
        scheduler = LifeScheduler()
        scheduler.needs.curiosity = 75  # > CURIOSITY_THRESHOLD (70)
        assert scheduler._decide_action() == "explore"

    def test_decide_action_play(self):
        from taiji.life.life_scheduler import LifeScheduler
        scheduler = LifeScheduler()
        scheduler.needs.boredom = 65  # > BOREDOM_THRESHOLD (60)
        assert scheduler._decide_action() == "play"

    def test_decide_action_priority(self):
        """饥饿 > 疲劳 > 好奇 > 无聊"""
        from taiji.life.life_scheduler import LifeScheduler
        scheduler = LifeScheduler()
        scheduler.needs.hunger = 80
        scheduler.needs.fatigue = 85
        scheduler.needs.curiosity = 75
        scheduler.needs.boredom = 65
        assert scheduler._decide_action() == "feed"

    def test_decide_action_while_busy(self):
        from taiji.life.life_scheduler import LifeScheduler
        scheduler = LifeScheduler()
        scheduler._life_state = "sleeping"
        scheduler.needs.hunger = 90
        assert scheduler._decide_action() is None

    def test_concurrent_add_fatigue(self):
        """并发安全性测试"""
        from taiji.life.life_scheduler import LifeScheduler
        scheduler = LifeScheduler()
        errors = []

        def worker():
            for _ in range(100):
                try:
                    scheduler.add_fatigue(0.1)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"并发错误: {errors}"

    def test_record_interaction_success(self):
        from taiji.life.life_scheduler import LifeScheduler
        scheduler = LifeScheduler()
        initial_hunger = scheduler.needs.hunger
        initial_stress = scheduler.needs.stress
        scheduler.record_interaction(success=True, reasoning_steps=2, used_tools=True)
        # 成功交互应降低饥饿和压力
        assert scheduler.needs.hunger < initial_hunger

    def test_record_interaction_failure(self):
        from taiji.life.life_scheduler import LifeScheduler
        scheduler = LifeScheduler()
        initial_stress = scheduler.needs.stress
        scheduler.record_interaction(success=False, reasoning_steps=3)
        # 失败交互应增加压力
        assert scheduler.needs.stress > initial_stress


# ═══════════════════════════════════════════
# SelfModificationEngine 测试
# ═══════════════════════════════════════════

class TestSelfModification:
    """测试自修改引擎"""

    def test_evaluate_response(self):
        from taiji.agent_ext.self_modification import SelfModificationEngine
        engine = SelfModificationEngine()
        result = engine.evaluate_response(
            response="这是一个详细的回答，包含了一些有用的信息。",
            query="你好"
        )
        assert "overall" in result
        assert "completeness" in result
        assert "defects" in result
        assert 0 <= result["overall"] <= 1

    def test_evaluate_short_response(self):
        from taiji.agent_ext.self_modification import SelfModificationEngine
        engine = SelfModificationEngine()
        result = engine.evaluate_response(response="好", query="请详细解释Python")
        assert "too_short" in result["defects"]

    def test_evaluate_overconfident(self):
        from taiji.agent_ext.self_modification import SelfModificationEngine
        engine = SelfModificationEngine()
        result = engine.evaluate_response(
            response="这绝对肯定一定毫无疑问是正确的答案。",
            query="什么是1+1"
        )
        assert "overconfident" in result["defects"]

    def test_propose_improvement_needs_data(self):
        """少于3条评估结果时不应生成提案"""
        from taiji.agent_ext.self_modification import SelfModificationEngine
        engine = SelfModificationEngine()
        proposals = engine.propose_improvement()
        assert proposals == []

    def test_get_status(self):
        from taiji.agent_ext.self_modification import SelfModificationEngine
        engine = SelfModificationEngine()
        status = engine.get_status()
        assert status["available"] == True
        assert "applied_count" in status

    def test_temperature_recommendation(self):
        from taiji.agent_ext.self_modification import SelfModificationEngine
        engine = SelfModificationEngine()
        temp = engine.get_temperature_for_task("code")
        # 代码任务应返回较低温度
        if temp is not None:
            assert temp < 0.5


# ═══════════════════════════════════════════
# ToolCallParser 测试
# ═══════════════════════════════════════════

class TestToolCallParser:
    """测试多策略工具调用解析"""

    def test_parse_json_block(self):
        from taiji.agent_ext.react_engine import ToolCallParser
        parser = ToolCallParser()
        result = parser.parse('```json\n{"tool": "search", "args": {"q": "test"}}\n```')
        assert len(result) == 1
        assert result[0]["name"] == "search"

    def test_parse_xml_tag(self):
        from taiji.agent_ext.react_engine import ToolCallParser
        parser = ToolCallParser()
        result = parser.parse('<tool_call>\n{"tool": "search", "args": {"q": "test"}}\n</tool_call>')
        assert len(result) == 1
        assert result[0]["name"] == "search"

    def test_parse_inline_json(self):
        from taiji.agent_ext.react_engine import ToolCallParser
        parser = ToolCallParser()
        result = parser.parse('{"tool": "search", "args": {"q": "test"}}')
        assert len(result) == 1

    def test_parse_function_call(self):
        from taiji.agent_ext.react_engine import ToolCallParser
        parser = ToolCallParser()
        result = parser.parse('use_tool("search", {"q": "test"})')
        assert len(result) == 1
        assert result[0]["name"] == "search"

    def test_parse_action_format(self):
        from taiji.agent_ext.react_engine import ToolCallParser
        parser = ToolCallParser()
        result = parser.parse('Action: search({"q": "test"})')
        assert len(result) == 1
        assert result[0]["name"] == "search"

    def test_parse_thought_action(self):
        from taiji.agent_ext.react_engine import ToolCallParser
        parser = ToolCallParser()
        result = parser.parse('Thought: I need to search\nAction: search\nAction Input: {"q": "test"}')
        assert len(result) == 1
        assert result[0]["name"] == "search"

    def test_parse_empty(self):
        from taiji.agent_ext.react_engine import ToolCallParser
        parser = ToolCallParser()
        result = parser.parse("")
        assert result == []

    def test_parse_no_tool(self):
        from taiji.agent_ext.react_engine import ToolCallParser
        parser = ToolCallParser()
        result = parser.parse("This is just regular text with no tool calls.")
        assert result == []

    def test_normalize_tool_name(self):
        from taiji.agent_ext.react_engine import ToolCallParser
        parser = ToolCallParser()
        result = parser.parse('{"name": "search", "args": {"q": "test"}}')
        assert len(result) == 1
        assert result[0]["name"] == "search"

    def test_validate_available_tools(self):
        from taiji.agent_ext.react_engine import ToolCallParser
        parser = ToolCallParser()
        result = parser.parse(
            '{"tool": "search", "args": {"q": "test"}}',
            available_tools=["search", "read_file"]
        )
        assert len(result) == 1

    def test_validate_case_insensitive(self):
        from taiji.agent_ext.react_engine import ToolCallParser
        parser = ToolCallParser()
        result = parser.parse(
            '{"tool": "Search", "args": {"q": "test"}}',
            available_tools=["search"]
        )
        assert len(result) == 1
        assert result[0]["name"] == "search"


# ═══════════════════════════════════════════
# ScienceEngine 测试
# ═══════════════════════════════════════════

class TestScienceEngine:
    """测试科学发现引擎"""

    def test_auto_discover_prime(self):
        from taiji.life.science_engine import ScienceEngine
        engine = ScienceEngine()
        result = engine.auto_discover("素数分布的密度")
        assert result["experiment_success"] == True
        assert "证实" in result["conclusion"] or "结论" in result["conclusion"]

    def test_auto_discover_fibonacci(self):
        from taiji.life.science_engine import ScienceEngine
        engine = ScienceEngine()
        result = engine.auto_discover("斐波那契数列的增长规律")
        assert result["experiment_success"] == True

    def test_auto_discover_sort(self):
        from taiji.life.science_engine import ScienceEngine
        engine = ScienceEngine()
        result = engine.auto_discover("排序算法的性能比较")
        assert result["experiment_success"] == True

    def test_propose_hypothesis(self):
        from taiji.life.science_engine import ScienceEngine
        engine = ScienceEngine()
        h = engine.propose_hypothesis("优化搜索复杂度")
        assert h.domain == "algorithm"  # "优化" in algorithm_keywords
        assert h.hypothesis is not None
        assert h.status == "proposed"

    def test_detect_domain(self):
        from taiji.life.science_engine import ScienceEngine
        engine = ScienceEngine()
        assert engine._detect_domain("数学公式") == "math"
        # "算法" appears in both code_keywords and algorithm_keywords; code checked first
        assert engine._detect_domain("排序算法") == "code"
        assert engine._detect_domain("数据分析") == "data"
        assert engine._detect_domain("Python代码") == "code"
        assert engine._detect_domain("优化搜索复杂度") == "algorithm"

    def test_run_experiment_and_conclusion(self):
        from taiji.life.science_engine import ScienceEngine
        engine = ScienceEngine()
        h = engine.propose_hypothesis("随机数生成器的均匀性")
        exp = engine.run_experiment(h.id)
        assert exp is not None
        d = engine.draw_conclusion(h.id)
        assert d is not None
        assert d.confidence >= 0


# ═══════════════════════════════════════════
# MultiAgent Orchestrator 测试
# ═══════════════════════════════════════════

class TestOrchestrator:
    """测试多 Agent 编排器"""

    def test_decompose_code_task(self):
        from taiji.agent_ext.multi_agent import Orchestrator
        orch = Orchestrator()
        task = orch.decompose_task("创建一个Python脚本实现数据搜索功能")
        assert len(task.subtasks) == 4  # researcher + planner + coder + reviewer

    def test_decompose_general_task(self):
        from taiji.agent_ext.multi_agent import Orchestrator
        orch = Orchestrator()
        task = orch.decompose_task("解释量子力学的基本原理")
        assert len(task.subtasks) >= 2

    def test_default_roles(self):
        from taiji.agent_ext.multi_agent import Orchestrator
        orch = Orchestrator()
        assert "coder" in orch.roles
        assert "researcher" in orch.roles
        assert "planner" in orch.roles
        assert "reviewer" in orch.roles

    def test_message_bus(self):
        from taiji.agent_ext.multi_agent import Orchestrator
        orch = Orchestrator()
        orch.message_bus.publish("test", "sender", "content")
        msgs = orch.message_bus.get_messages("test")
        assert len(msgs) == 1

    def test_list_tasks(self):
        from taiji.agent_ext.multi_agent import Orchestrator
        orch = Orchestrator()
        orch.decompose_task("测试任务")
        tasks = orch.list_tasks()
        assert len(tasks) >= 1


# ═══════════════════════════════════════════
# FeedEngine 测试
# ═══════════════════════════════════════════

class TestFeedEngine:
    """测试喂养引擎"""

    def test_assess_quality(self):
        from taiji.life.feed_engine import FeedEngine, FeedConfig
        engine = FeedEngine()
        # 短文本应评分低
        q1 = engine._assess_quality("hi", "knowledge")
        # 长文本应评分高
        q2 = engine._assess_quality("这是一段较长的文本内容，包含了足够的信息量。" * 5, "knowledge")
        assert q2 > q1

    def test_feed_text(self):
        from taiji.life.feed_engine import FeedEngine
        engine = FeedEngine()
        item = engine.feed_text("测试内容，这是一段有意义的文字", category="knowledge")
        # 可能返回 None（质量不达标）或 FeedItem
        if item:
            assert item.status in ("digested", "rejected")


# ═══════════════════════════════════════════
# Integration: 跨模块集成测试
# ═══════════════════════════════════════════

class TestIntegration:
    """跨模块集成测试"""

    def test_import_all_modules(self):
        """验证所有核心模块可以正常导入"""
        from taiji.life.life_scheduler import LifeScheduler
        from taiji.life.feed_engine import FeedEngine
        from taiji.life.sleep_engine import SleepEngine
        from taiji.life.play_engine import PlayEngine
        from taiji.life.evolution_engine import EvolutionEngine
        from taiji.life.science_engine import ScienceEngine
        from taiji.life.recursive_improver import RecursiveImprover
        from taiji.life.explore_engine import ExploreEngine
        from taiji.agent_ext.self_modification import SelfModificationEngine
        from taiji.agent_ext.react_engine import ToolCallParser, ReActEngine
        from taiji.agent_ext.multi_agent import Orchestrator
        from taiji.brain.cortex import Cortex
        from taiji.train.trainer import ModelSelfTrainer

    def test_engine_initialization(self):
        """验证所有引擎可以初始化"""
        from taiji.life.life_scheduler import LifeScheduler
        from taiji.life.feed_engine import FeedEngine
        from taiji.life.evolution_engine import EvolutionEngine
        from taiji.life.science_engine import ScienceEngine
        from taiji.agent_ext.self_modification import SelfModificationEngine

        ls = LifeScheduler()
        fe = FeedEngine()
        ee = EvolutionEngine()
        se = ScienceEngine()
        sm = SelfModificationEngine()

        assert ls.get_status()["life_state"] == "idle"
        assert fe.get_status()["total_feeds"] >= 0
        assert ee.get_status()["phase"] == "infant"
        assert se.get_status()["hypotheses"] >= 0
        assert sm.get_status()["available"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])