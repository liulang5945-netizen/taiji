"""
态极语音生成模块（已合并到 SpeechEngine）

此文件保留向后兼容别名。新代码请直接用 taiji.multimodal.speech_engine.SpeechEngine。
"""

from taiji.multimodal.speech_engine import SpeechEngine as TaijiVoiceGenerator

__all__ = ["TaijiVoiceGenerator"]
