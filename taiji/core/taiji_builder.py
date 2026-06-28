"""
TaijiBuilder — 态极器官组装器

将所有生命子系统（骨骼、大脑、生命引擎、记忆系统等）的
创建逻辑从 TaijiCore.__init__ 中提取出来，独立为可测试的构建器。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from taiji.core.taiji_context import TaijiContext

# 模块级导入（而非构造函数内延迟导入）
from taiji.body.core import BodyCore
from taiji.infra.events import EventBus
from taiji.safety.safety import SafetyGuard
from taiji.brain.cortex import Cortex
from taiji.agent.context_manager import ContextManager
from taiji.agent.working_memory import get_working_memory
from taiji.life.feed_engine import FeedEngine
from taiji.life.sleep_engine import SleepEngine
from taiji.life.play_engine import PlayEngine
from taiji.life.evolution_engine import EvolutionEngine
from taiji.life.explore_engine import ExploreEngine
from taiji.life.science_engine import ScienceEngine
from taiji.life.recursive_improver import RecursiveImprover
from taiji.life.life_scheduler import LifeScheduler

logger = logging.getLogger("TaijiBuilder")


class TaijiBuilder:
    """态极器官的构建与组装器。

    使用方式:
        ctx = TaijiBuilder(model=model, tokenizer=tokenizer).assemble()
        taiji = TaijiCore(ctx)
    """

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        device: str = "cpu",
        action_provider: Any = None,
        data_collector: Any = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.action_provider = action_provider
        self.data_collector = data_collector

    def assemble(self) -> TaijiContext:
        """按顺序创建并连接所有器官，返回完整上下文。"""
        ctx = TaijiContext()
        ctx.model = self.model
        ctx.tokenizer = self.tokenizer

        self._build_skeleton(ctx)
        self._build_circulation(ctx)
        self._build_immunity(ctx)
        self._build_cortex(ctx)
        self._build_memory(ctx)
        self._build_life_engines(ctx)
        self._build_life_scheduler(ctx)
        self._wire_nerves(ctx)

        logger.info("TaijiContext assembled — 所有器官已连接")
        return ctx

    # ── 骨骼 ──

    def _build_skeleton(self, ctx: TaijiContext):
        ctx.body = BodyCore()
        ctx.body.set_model(ctx.model)
        ctx.body.set_tokenizer(ctx.tokenizer)
        ctx.body.set_device(self.device)
        if self.action_provider:
            ctx.body.set_action_provider(self.action_provider)
        if self.data_collector:
            ctx.body.set_data_collector(self.data_collector)

    # ── 循环系统 ──

    def _build_circulation(self, ctx: TaijiContext):
        ctx.events = EventBus()

    # ── 免疫系统 ──

    def _build_immunity(self, ctx: TaijiContext):
        ctx.safety = SafetyGuard()

    # ── 大脑皮层 ──

    def _build_cortex(self, ctx: TaijiContext):
        ctx.cortex = Cortex(body=ctx.body)

    # ── 记忆系统 ──

    def _build_memory(self, ctx: TaijiContext):
        ctx.context_manager = ContextManager(max_context_tokens=2048)

        # 工作记忆
        wm = get_working_memory()
        ctx.context_manager.set_working_memory(wm)

        # 长期记忆（可选）
        try:
            from taiji.agent.memory import MemorySystem
            ms = MemorySystem()
            ms.load(os.path.join("taiji_data", "memory"))
            ctx.context_manager.set_memory_system(ms)
        except Exception as e:
            logger.warning("MemorySystem 加载失败（非致命）: %s", e)

        ctx.context_manager.set_persistent_path(
            os.path.join("taiji_data", "memory", "persistent_memory.json")
        )

        # 语义记忆（可选）
        try:
            from taiji.agent.semantic_memory import get_semantic_memory
            sm = get_semantic_memory()
            ctx.context_manager.set_semantic_memory(sm)
        except Exception as e:
            logger.warning("语义记忆加载失败（非致命）: %s", e)

    # ── 生命引擎 ──

    def _build_life_engines(self, ctx: TaijiContext):
        ctx.feed = FeedEngine()
        ctx.sleep = SleepEngine()
        ctx.play = PlayEngine()
        ctx.evolution = EvolutionEngine()
        ctx.explore = ExploreEngine()
        ctx.science = ScienceEngine()
        ctx.improver = RecursiveImprover()

    # ── 生命调度器 ──

    def _build_life_scheduler(self, ctx: TaijiContext):
        ctx.life_scheduler = LifeScheduler(event_bus=ctx.events)
        # 注入引擎引用
        ctx.life_scheduler._feed_engine = ctx.feed
        ctx.life_scheduler._sleep_engine = ctx.sleep
        ctx.life_scheduler._play_engine = ctx.play
        ctx.life_scheduler._explore_engine = ctx.explore

    # ── 神经系统 ──

    def _wire_nerves(self, ctx: TaijiContext):
        """连接引擎间的事件订阅。"""

        def _on_interaction(event):
            success = event.event_type == "interaction_success"
            topic = event.data.get("topic", "")
            ctx.life_scheduler.record_interaction(success=success, topic=topic)

        ctx.events.subscribe("interaction_success", _on_interaction)
        ctx.events.subscribe("interaction_failure", _on_interaction)
