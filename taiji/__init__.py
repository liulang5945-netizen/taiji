"""
Taiji Core (态极) — Unified Entry Point

Assembles all organs into a complete living being.
One line to start Taiji:
    from taiji import TaijiCore
    taiji = TaijiCore(model, tokenizer)
    taiji.start_life()

态极递归蒸馏进化 (Taiji Recursive Distillation):
- 基底: Qwen2.5 系列（提供语言能力）
- 生命系统: 饥饿、好奇、疲劳、进化（提供态极性）
- 进化路线: 小态极火种 → 生成进化语料 → 蒸馏 → 更大态极 → 循环
- 核心哲学: 小态极把自己的结构投射到更大的计算载体中
"""
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("TaijiCore")


def __getattr__(name: str):
    """Lazy compatibility exports for model-heavy package attributes."""
    if name in {"load_model", "save_model", "create_model"}:
        from taiji import loader

        return getattr(loader, name)
    if name == "ModelSelf":
        from taiji.architecture import ModelSelf

        return ModelSelf
    if name == "NativeInferenceEngine":
        from taiji.core.inference import NativeInferenceEngine

        return NativeInferenceEngine
    if name == "TaijiMultimodalEngine":
        try:
            from taiji.multimodal.multimodal_engine import TaijiMultimodalEngine
        except ImportError:
            return None
        return TaijiMultimodalEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class TaijiCore:
    """
    态极核心 - 完整的生命体

    将大脑皮层、骨骼、循环系统、消化系统、睡眠系统、
    玩耍系统、进化系统、记忆系统、免疫系统组装成一个完整的生命。

    设计原则：
    - 所有器官由 TaijiCore 统一创建和管理
    - 器官间通过 EventBus 通信（循环系统）
    - 思考通过 Cortex（意识中枢），消耗能量、产生记忆
    - 需求变化通过事件驱动，保持一致性
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        device: str = "cpu",
        action_provider=None,
        data_collector=None,
    ):
        """
        创建态极生命体。

        Args:
            model: 态极的大脑（ModelSelf）
            tokenizer: 语言中枢（分词器）
            device: 计算设备（cpu/cuda）
            action_provider: 手脚（ActionProvider，可选）
            data_collector: 数据收集器（可选）
        """
        from taiji.body.core import BodyCore
        from taiji.infra.events import EventBus
        from taiji.safety.safety import SafetyGuard
        from taiji.brain.cortex import Cortex
        from taiji.life.feed_engine import FeedEngine
        from taiji.life.sleep_engine import SleepEngine
        from taiji.life.play_engine import PlayEngine
        from taiji.life.evolution_engine import EvolutionEngine
        from taiji.life.life_scheduler import LifeScheduler

        # ── 骨骼：资源管理 ──
        self.body = BodyCore()
        self.body.set_model(model)
        self.body.set_tokenizer(tokenizer)
        self.body.set_device(device)
        if action_provider:
            self.body.set_action_provider(action_provider)
        if data_collector:
            self.body.set_data_collector(data_collector)

        # ── 循环系统：事件总线 ──
        self.events = EventBus()

        # ── 免疫系统 ──
        self.safety = SafetyGuard()

        # ── 大脑皮层：意识中枢 ──
        self.cortex = Cortex(body=self.body)

        # ── 上下文管理器：统一记忆系统 ──
        from taiji.agent.context_manager import ContextManager
        self.context = ContextManager(max_context_tokens=2048)

        # 注入记忆系统到上下文管理器
        from taiji.agent.working_memory import get_working_memory
        wm = get_working_memory()
        self.context.set_working_memory(wm)

        # 注入 MemorySystem（如果可用）
        try:
            from taiji.agent.memory import MemorySystem
            ms = MemorySystem()
            ms.load(os.path.join("taiji_data", "memory"))
            self.context.set_memory_system(ms)
        except Exception:
            pass

        # 设置持久化路径
        self.context.set_persistent_path(
            os.path.join("taiji_data", "memory", "persistent_memory.json")
        )

        # 注入语义记忆（向量检索）
        try:
            from taiji.agent.semantic_memory import get_semantic_memory
            sm = get_semantic_memory()
            self.context.set_semantic_memory(sm)
        except Exception:
            pass

        # ── 生命系统（统一创建，非全局单例）──
        self._feed = FeedEngine()
        self._sleep = SleepEngine()
        self._play = PlayEngine()
        self._evolution = EvolutionEngine()

        from taiji.life.explore_engine import ExploreEngine
        self._explore = ExploreEngine()

        from taiji.life.science_engine import ScienceEngine
        self._science = ScienceEngine()

        # ── 递归改进系统（Gödel Agent 思想）──
        from taiji.life.recursive_improver import RecursiveImprover
        self._improver = RecursiveImprover()

        # ── 生命调度器（本能系统）──
        self._life = LifeScheduler(event_bus=self.events)

        # 注入引擎引用到 LifeScheduler（避免全局单例查找）
        self._life._feed_engine = self._feed
        self._life._sleep_engine = self._sleep
        self._life._play_engine = self._play
        self._life._explore_engine = self._explore

        # ── 推理引擎缓存 ──
        self._inference_engine = None

        # ── 建立事件订阅（神经系统）──
        self._wire_events()

        logger.info("TaijiCore created — 生命体组装完成")

    # ── 事件订阅（神经系统）──

    def _wire_events(self):
        """
        连接引擎间的事件订阅 — 态极的"神经系统"。

        需求变化由 LifeScheduler 统一管理（单一数据源），
        事件系统负责广播到前端和通知其他子系统。
        """
        # 用户交互 → 通知生命调度器
        self.events.subscribe("interaction_success", self._on_interaction)
        self.events.subscribe("interaction_failure", self._on_interaction)

    def _on_interaction(self, event):
        """用户交互 → 通知生命调度器"""
        success = event.event_type == "interaction_success"
        topic = event.data.get("topic", "")
        self._life.record_interaction(success=success, topic=topic)

    # ── 生命系统属性（向后兼容）──

    @property
    def life(self):
        """生命调度器（心跳）"""
        return self._life

    @property
    def feed(self):
        """喂养引擎（吃饭）"""
        return self._feed

    @property
    def sleep(self):
        """睡眠引擎（睡觉）"""
        return self._sleep

    @property
    def play(self):
        """玩耍引擎（娱乐）"""
        return self._play

    @property
    def evolution(self):
        """进化引擎"""
        return self._evolution

    @property
    def improver(self):
        """递归改进系统"""
        return self._improver

    @property
    def explore(self):
        """探索引擎"""
        return self._explore

    @property
    def science(self):
        """科学发现引擎"""
        return self._science

    # ── 生命控制 ──

    def start_life(self):
        """启动态极的生命（启动心跳循环）"""
        self._life.start()
        self.events.publish("life_started", source="taikicore")
        logger.info("Taiji life started")

    def stop_life(self):
        """暂停态极的生命"""
        self._life.stop()
        self.events.publish("life_stopped", source="taikicore")
        logger.info("Taiji life stopped")

    def record_interaction(self, success: bool = True, topic: str = ""):
        """记录一次用户交互（影响需求状态）"""
        self._life.record_interaction(success=success, topic=topic)
        self.events.publish(
            "interaction_success" if success else "interaction_failure",
            {"topic": topic},
            source="interaction",
        )

    # ── 感知和行动 ──

    def think(self, prompt: str, **kwargs) -> str:
        """
        思考 — 通过意识中枢（Cortex）的完整意识流。

        设计原则：
        - 模型未加载时不消耗能量、不记录交互
        - 只有真正完成思考后才产生生命活动
        - 态极会在回复中表达自身感受
        """
        try:
            # 记录用户消息到上下文
            self.context.add_message("user", prompt)

            # 通过 Cortex 的完整意识流
            result = self.cortex.think_sync(prompt, **kwargs)

            # 如果模型未加载，不记录交互、不消耗资源
            if result.startswith("[模型未加载"):
                return result

            # 记录助手回复到上下文
            self.context.add_message("assistant", result)

            # 记住这次交互（只有成功的交互才记住）
            self.context.remember(
                f"interaction_{len(self.context._conversation_history)}",
                f"Q: {prompt[:100]} A: {result[:100]}",
                source="interaction",
                importance=0.3,
            )

            return self.safety.validate_output(result)
        except Exception as e:
            logger.error(f"Think failed: {e}")
            return f"[思考失败: {e}]"

    def see(self, image_path: str) -> str:
        """看（调用视觉编码器理解图像）"""
        try:
            from taiji.multimodal.vision_encoder import TaijiVisionEncoder
            hidden_size = self.body.model.hidden_size if self.body.model else 768
            encoder = TaijiVisionEncoder(hidden_size=hidden_size)
            return encoder.describe_image_simple(image_path)
        except Exception as e:
            logger.error(f"See failed: {e}")
            return f"[视觉处理失败: {e}]"

    # ── 记忆 ──

    def remember(self, key: str, value: str, persistent: bool = False):
        """记住信息（通过统一上下文管理器）"""
        try:
            self.context.remember(key, value, source="user_input", persistent=persistent)
            logger.debug(f"Remembered: {key}")
        except Exception as e:
            logger.error(f"Remember failed: {e}")

    def recall(self, key: str) -> Optional[str]:
        """回忆信息（通过统一上下文管理器）"""
        try:
            return self.context.recall(key)
        except Exception as e:
            logger.error(f"Recall failed: {e}")
            return None

    # ── 手动触发生命活动 ──

    def do_feed(self) -> dict:
        """手动触发吃饭"""
        if not self.safety.rate_limit("feed"):
            return {"success": False, "reason": "频率限制"}
        report = self._feed.feed(reason="manual")
        self.events.publish("feed_complete", {"samples": report.samples_generated}, source="feed")
        return {
            "items_fed": report.items_fed,
            "samples_generated": report.samples_generated,
            "avg_quality": report.avg_quality,
        }

    def do_sleep(self) -> dict:
        """手动触发睡觉"""
        if not self.safety.rate_limit("sleep"):
            return {"success": False, "reason": "频率限制"}
        report = self._sleep.sleep(reason="manual")
        self.events.publish("sleep_complete", {"loss": report.training_loss}, source="sleep")
        return {
            "phases": report.phases_completed,
            "training_loss": report.training_loss,
            "health": report.health_status,
        }

    def do_play(self) -> dict:
        """手动触发玩耍"""
        if not self.safety.rate_limit("play"):
            return {"success": False, "reason": "频率限制"}
        report = self._play.play(reason="manual")
        self.events.publish("play_complete", {"mood": report.mood}, source="play")
        return {
            "activities": len(report.activities),
            "mood": report.mood,
            "traits": report.personality_traits_discovered,
        }

    # ── 状态查询 ──

    def get_status(self) -> dict:
        """获取态极完整状态"""
        return {
            "body": self.body.get_status(),
            "life": self._life.get_status(),
            "needs": self._life.needs.to_dict(),
            "cortex": self.cortex.get_status(),
            "context": self.context.get_status(),
            "safety": self.safety.get_status(),
            "events": {
                "total": len(self.events.get_history(1000)),
                "subscribers": self.events.get_subscriber_count(),
            },
            "feed": self._feed.get_status(),
            "sleep": self._sleep.get_status(),
            "play": self._play.get_status(),
            "evolution": {
                "phase": self._evolution.metrics.current_phase,
                "tasks_completed": self._evolution.metrics.tasks_completed,
            },
        }

    def get_summary(self) -> str:
        """获取人类可读的状态摘要"""
        status = self.get_status()
        needs = status["needs"]

        lines = [
            "态极生命状态",
            "============",
            f"大脑: {'就绪' if status['body']['has_model'] else '未加载'}",
            f"心跳: {'运行中' if status['life']['is_running'] else '暂停'}",
            f"免疫: {status['safety']['threat_level']}",
            f"意识: 已思考 {status['cortex']['total_thoughts']} 次, "
            f"消耗能量 {status['cortex']['total_energy_spent']}",
            "",
            "内在需求:",
            f"  饥饿: {needs['hunger']:.0f}/100",
            f"  疲劳: {needs['fatigue']:.0f}/100",
            f"  无聊: {needs['boredom']:.0f}/100",
            f"  压力: {needs['stress']:.0f}/100",
            f"  好奇: {needs['curiosity']:.0f}/100",
            "",
            f"吃饭次数: {status['feed']['total_feeds']}",
            f"睡觉次数: {status['sleep']['total_sleeps']}",
            f"玩耍次数: {status['play']['total_plays']}",
            f"进化阶段: {status['evolution']['phase']}",
        ]

        return "\n".join(lines)

    # ── 导出（生殖系统）──

    def export(self, path: str):
        """将态极导出为独立包"""
        if self.body.model and self.body.tokenizer:
            from taiji.loader import save_model

            save_model(self.body.model, self.body.tokenizer, path)
            logger.info(f"Taiji exported to {path}")
        else:
            logger.warning("No model/tokenizer to export")

    # ── 加载（类方法）──

    @classmethod
    def load(cls, model_path: str, device: str = "cpu") -> "TaijiCore":
        """从磁盘加载态极"""
        from taiji.loader import load_model

        model, tokenizer = load_model(model_path, device=device)
        taiji = cls(model=model, tokenizer=tokenizer, device=device)
        logger.info(f"Taiji loaded from {model_path}")
        return taiji
