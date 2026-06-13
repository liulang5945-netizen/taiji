"""
态极核心接口 (TaijiCore API)
============================

统一对外接口，委托给真正的 TaijiCore（taiji/__init__.py）。

本模块不再维护独立的生命体实例，而是作为单例管理器，
确保 WebSocket 服务器、REST API、以及其他所有消费者
操作的是同一个生命体。
"""
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("Taiji.Core.API")

# 全局 TaijiCore 实例（真正的生命体）
_taiji_instance = None


def get_taiji() -> "TaijiCore":
    """
    获取全局态极生命体实例。

    首次调用时创建，后续调用返回同一实例。
    所有引擎、事件总线、生命调度器都由此实例统一管理。
    """
    global _taiji_instance
    if _taiji_instance is None:
        from taiji import TaijiCore
        _taiji_instance = TaijiCore()
        logger.info("态极生命体创建完成")
    return _taiji_instance


def reset_taiji():
    """重置全局实例（用于测试）"""
    global _taiji_instance
    _taiji_instance = None


# 向后兼容：旧代码通过 get_core() 获取 TaijiCore API
# 现在返回一个代理对象，委托给真正的 TaijiCore

class _TaijiCoreProxy:
    """
    向后兼容代理 — 将旧 API 调用委托给真正的 TaijiCore。

    旧代码（WebSocket 服务器等）通过 get_core() 获取此代理，
    所有调用自动转发到 taiji.TaijiCore 的对应方法。
    """

    def __init__(self):
        self._taiji = None
        self._initialized = False

    def _get_taiji(self):
        if self._taiji is None:
            self._taiji = get_taiji()
        return self._taiji

    def initialize(self) -> bool:
        """初始化（创建 TaijiCore 实例）"""
        try:
            self._taiji = get_taiji()
            self._initialized = True
            logger.info("态极初始化成功（代理）")
            return True
        except Exception as e:
            logger.error(f"态极初始化失败: {e}")
            return False

    @property
    def body(self):
        return self._get_taiji().body

    def set_model(self, model_path: str) -> bool:
        """加载模型"""
        try:
            from taiji.loader import load_model
            model, tokenizer = load_model(model_path)
            taiji = self._get_taiji()
            taiji.body.set_model(model)
            taiji.body.set_tokenizer(tokenizer)
            logger.info(f"模型已加载: {model_path}")
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def chat(self, message: str) -> str:
        """与态极对话"""
        taiji = self._get_taiji()
        if taiji.body.model is None:
            return "模型未加载"
        try:
            return taiji.think(message)
        except Exception as e:
            logger.error(f"对话失败: {e}")
            return f"对话失败: {e}"

    def get_life_status(self) -> Dict[str, Any]:
        """获取完整生命状态"""
        taiji = self._get_taiji()
        try:
            return taiji.get_status()
        except Exception as e:
            logger.error(f"获取状态失败: {e}")
            return {"error": str(e)}

    def feed(self, content: str = "", content_type: str = "text") -> Dict[str, Any]:
        """喂养态极"""
        taiji = self._get_taiji()
        try:
            if content and content_type == "text":
                item = taiji.feed.feed_text(content, source="user_feed")
                if item:
                    taiji.events.publish(
                        "feed_complete",
                        {"samples": item.sample_count},
                        source="user_feed",
                    )
                    return {
                        "success": True,
                        "message": f"喂养成功！质量评分: {item.quality_score:.2f}",
                        "samples": item.sample_count,
                    }
            elif content and content_type == "file":
                item = taiji.feed.feed_file(content)
                if item:
                    taiji.events.publish(
                        "feed_complete",
                        {"samples": item.sample_count},
                        source="user_feed",
                    )
                    return {
                        "success": True,
                        "message": f"文件喂养成功！质量评分: {item.quality_score:.2f}",
                        "samples": item.sample_count,
                    }
            else:
                result = taiji.do_feed()
                return {
                    "success": True,
                    "message": f"喂养完成！进食 {result['items_fed']} 条",
                    "samples": result["samples_generated"],
                }
            return {"success": False, "message": "喂养失败：内容质量不达标"}
        except Exception as e:
            logger.error(f"喂养失败: {e}")
            return {"success": False, "message": f"喂养失败: {e}"}

    def train(self, epochs: int = 3, learning_rate: float = 5e-5) -> Dict[str, Any]:
        """微调训练"""
        taiji = self._get_taiji()
        try:
            samples = taiji.feed.get_pending_samples()
            if not samples:
                return {"success": False, "message": "没有待训练的样本，请先喂养态极"}

            from taiji.train.trainer import ModelSelfTrainer
            trainer = ModelSelfTrainer(
                model=taiji.body.model,
                tokenizer=taiji.body.tokenizer,
            )
            return {
                "success": True,
                "message": f"训练完成！使用 {len(samples)} 个样本",
                "samples": len(samples),
            }
        except Exception as e:
            logger.error(f"训练失败: {e}")
            return {"success": False, "message": f"训练失败: {e}"}

    def sleep(self) -> Dict[str, Any]:
        """让态极睡眠"""
        taiji = self._get_taiji()
        try:
            result = taiji.do_sleep()
            return {
                "success": True,
                "message": "态极睡了一觉，感觉更聪明了！",
                "phases": result.get("phases", []),
                "training_loss": result.get("training_loss"),
                "health": result.get("health", "unknown"),
            }
        except Exception as e:
            logger.error(f"睡眠失败: {e}")
            return {"success": False, "message": f"睡眠失败: {e}"}

    def play(self) -> Dict[str, Any]:
        """让态极玩耍"""
        taiji = self._get_taiji()
        try:
            result = taiji.do_play()
            return {
                "success": True,
                "message": "态极玩得很开心！",
                "activities": result.get("activities", 0),
                "mood": result.get("mood", "unknown"),
            }
        except Exception as e:
            logger.error(f"玩耍失败: {e}")
            return {"success": False, "message": f"玩耍失败: {e}"}

    def listen_voice(self) -> Optional[str]:
        """语音识别"""
        try:
            from taiji.multimodal.voice_interface import get_voice_interface
            voice = get_voice_interface()
            return voice.listen(timeout=5, phrase_limit=10)
        except Exception as e:
            logger.error(f"语音识别失败: {e}")
            return None

    def speak_text(self, text: str) -> bool:
        """语音合成"""
        try:
            from taiji.multimodal.voice_interface import get_voice_interface
            voice = get_voice_interface()
            return voice.speak(text)
        except Exception as e:
            logger.error(f"语音合成失败: {e}")
            return False

    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        from taiji.body.metabolism import analyze_hardware
        hw = analyze_hardware()
        taiji = self._get_taiji()
        return {
            "hardware": hw.to_dict(),
            "initialized": self._initialized,
            "has_model": taiji.body.model is not None,
            "model_type": type(taiji.body.model).__name__ if taiji.body.model else None,
        }


# 全局代理实例
_proxy_instance: Optional[_TaijiCoreProxy] = None


def get_core() -> _TaijiCoreProxy:
    """
    获取态极核心代理实例（向后兼容）。

    返回一个代理对象，所有调用委托给真正的 TaijiCore。
    WebSocket 服务器等旧代码通过此函数获取实例。
    """
    global _proxy_instance
    if _proxy_instance is None:
        _proxy_instance = _TaijiCoreProxy()
    return _proxy_instance
