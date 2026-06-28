"""
Taiji Core (态极) — Unified Entry Point

Assembles all organs into a complete living being.
One line to start Taiji:
    from taiji import TaijiCore
    taiji = TaijiCore(model, tokenizer)
    taiji.start_life()

态极递归蒸馏进化 (Taiji Recursive Distillation):
- 基底: 态极 1B~12B 原生模型（ModelSelf）
- 生命系统: 饥饿、好奇、疲劳、进化（提供态极性）
- 进化路线: 小态极火种 → 生成进化语料 → 蒸馏 → 更大态极 → 循环
- 核心哲学: 小态极把自己的结构投射到更大的计算载体中
"""
import logging
from typing import Optional, Dict, Any

from taiji.core.taiji_context import TaijiContext
from taiji.core.taiji_builder import TaijiBuilder

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
        from taiji.multimodal.multimodal_engine import TaijiMultimodalEngine
        return TaijiMultimodalEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class TaijiCore:
    """
    态极核心 - 完整的生命体

    设计原则：
    - 器官由 TaijiBuilder 统一创建，注入 TaijiContext
    - 器官间通过 EventBus 通信（循环系统）
    - 思考通过 Cortex（意识中枢），消耗能量、产生记忆
    - 需求变化通过事件驱动，保持一致性
    """

    def __init__(
        self,
        ctx: Optional[TaijiContext] = None,
        model=None,
        tokenizer=None,
        device: str = "cpu",
        action_provider=None,
        data_collector=None,
    ):
        """
        创建态极生命体。

        两种方式：
        1. 传入预构建的 TaijiContext:  TaijiCore(ctx=builder.assemble())
        2. 快速创建（向后兼容）:     TaijiCore(model=model, tokenizer=tokenizer)
        """
        if ctx is not None:
            self._ctx = ctx
        else:
            builder = TaijiBuilder(
                model=model, tokenizer=tokenizer, device=device,
                action_provider=action_provider, data_collector=data_collector,
            )
            self._ctx = builder.assemble()

        logger.info("TaijiCore created — 生命体组装完成")

    # ── 生命系统属性 ──

    @property
    def body(self): return self._ctx.body
    @property
    def events(self): return self._ctx.events
    @property
    def safety(self): return self._ctx.safety
    @property
    def cortex(self): return self._ctx.cortex
    @property
    def context(self): return self._ctx.context_manager
    @property
    def life(self): return self._ctx.life_scheduler
    @property
    def feed(self): return self._ctx.feed
    @property
    def sleep(self): return self._ctx.sleep
    @property
    def play(self): return self._ctx.play
    @property
    def evolution(self): return self._ctx.evolution
    @property
    def improver(self): return self._ctx.improver
    @property
    def explore(self): return self._ctx.explore
    @property
    def science(self): return self._ctx.science

    # ── 生命控制 ──

    def start_life(self):
        self._ctx.life_scheduler.start()
        self._ctx.events.publish("life_started", source="taikicore")
        logger.info("Taiji life started")

    def stop_life(self):
        self._ctx.life_scheduler.stop()
        self._ctx.events.publish("life_stopped", source="taikicore")
        logger.info("Taiji life stopped")

    def record_interaction(self, success: bool = True, topic: str = ""):
        self._ctx.life_scheduler.record_interaction(success=success, topic=topic)
        self._ctx.events.publish(
            "interaction_success" if success else "interaction_failure",
            {"topic": topic}, source="interaction",
        )

    # ── 感知和行动 ──

    def think(self, prompt: str, **kwargs) -> str:
        """思考 — 通过意识中枢的完整意识流。"""
        try:
            self._ctx.context_manager.add_message("user", prompt)
            result = self._ctx.cortex.think_sync(prompt, **kwargs)
            if result.startswith("[模型未加载"):
                return result
            self._ctx.context_manager.add_message("assistant", result)
            self._ctx.context_manager.remember(
                f"interaction_{len(self._ctx.context_manager._conversation_history)}",
                f"Q: {prompt[:100]} A: {result[:100]}",
                source="interaction", importance=0.3,
            )
            return self._ctx.safety.validate_output(result)
        except Exception as e:
            logger.error(f"Think failed: {e}")
            return f"[思考失败: {e}]"

    def see(self, image_path: str) -> str:
        """看（调用视觉编码器理解图像）"""
        try:
            from taiji.multimodal.vision_encoder import TaijiVisionEncoder
            hidden_size = self._ctx.body.model.hidden_size if self._ctx.body.model else 768
            encoder = TaijiVisionEncoder(hidden_size=hidden_size)
            return encoder.describe_image_simple(image_path)
        except Exception as e:
            logger.error(f"See failed: {e}")
            return f"[视觉处理失败: {e}]"

    # ── 记忆 ──

    def remember(self, key: str, value: str, persistent: bool = False):
        try:
            self._ctx.context_manager.remember(key, value, source="user_input", persistent=persistent)
            logger.debug(f"Remembered: {key}")
        except Exception as e:
            logger.error(f"Remember failed: {e}")

    def recall(self, key: str) -> Optional[str]:
        try:
            return self._ctx.context_manager.recall(key)
        except Exception as e:
            logger.error(f"Recall failed: {e}")
            return None

    # ── 手动触发生命活动 ──

    def do_feed(self) -> dict:
        if not self._ctx.safety.rate_limit("feed"):
            return {"success": False, "reason": "频率限制"}
        report = self._ctx.feed.feed(reason="manual")
        self._ctx.events.publish("feed_complete", {"samples": report.samples_generated}, source="feed")
        return {"items_fed": report.items_fed, "samples_generated": report.samples_generated,
                "avg_quality": report.avg_quality}

    def do_sleep(self) -> dict:
        if not self._ctx.safety.rate_limit("sleep"):
            return {"success": False, "reason": "频率限制"}
        report = self._ctx.sleep.sleep(reason="manual")
        self._ctx.events.publish("sleep_complete", {"loss": report.training_loss}, source="sleep")
        return {"phases": report.phases_completed, "training_loss": report.training_loss,
                "health": report.health_status}

    def do_play(self) -> dict:
        if not self._ctx.safety.rate_limit("play"):
            return {"success": False, "reason": "频率限制"}
        report = self._ctx.play.play(reason="manual")
        self._ctx.events.publish("play_complete", {"mood": report.mood}, source="play")
        return {"activities": len(report.activities), "mood": report.mood,
                "traits": report.personality_traits_discovered}

    # ── 状态查询 ──

    def get_status(self) -> dict:
        ctx = self._ctx
        return {
            "body": ctx.body.get_status(),
            "life": ctx.life_scheduler.get_status(),
            "needs": ctx.life_scheduler.needs.to_dict(),
            "cortex": ctx.cortex.get_status(),
            "context": ctx.context_manager.get_status(),
            "safety": ctx.safety.get_status(),
            "events": {"total": len(ctx.events.get_history(1000)),
                       "subscribers": ctx.events.get_subscriber_count()},
            "feed": ctx.feed.get_status(),
            "sleep": ctx.sleep.get_status(),
            "play": ctx.play.get_status(),
            "evolution": {"phase": ctx.evolution.metrics.current_phase,
                          "tasks_completed": ctx.evolution.metrics.tasks_completed},
        }

    def get_summary(self) -> str:
        status = self.get_status()
        needs = status["needs"]
        return "\n".join([
            "态极生命状态",
            "============",
            f"大脑: {'就绪' if status['body']['has_model'] else '未加载'}",
            f"心跳: {'运行中' if status['life']['is_running'] else '暂停'}",
            f"免疫: {status['safety']['threat_level']}",
            f"意识: 已思考 {status['cortex']['total_thoughts']} 次, 消耗能量 {status['cortex']['total_energy_spent']}",
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
        ])

    # ── 导出 ──

    def export(self, path: str):
        if self._ctx.body.model and self._ctx.body.tokenizer:
            from taiji.loader import save_model
            save_model(self._ctx.body.model, self._ctx.body.tokenizer, path)
            logger.info(f"Taiji exported to {path}")
        else:
            logger.warning("No model/tokenizer to export")

    # ── 加载（类方法）──

    @classmethod
    def load(cls, model_path: str, device: str = "cpu") -> "TaijiCore":
        from taiji.loader import load_model
        model, tokenizer = load_model(model_path, device=device)
        taiji = cls(model=model, tokenizer=tokenizer, device=device)
        logger.info(f"Taiji loaded from {model_path}")
        return taiji
