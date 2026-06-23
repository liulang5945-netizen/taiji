"""
态极语音引擎 (Speech Engine)
============================
让态极能"说话"。

支持：
- Text-to-Speech (TTS): 文字转语音
- 多种中文语音
- 语速、音调调节

使用 edge-tts（微软免费 TTS，中文效果好，本地运行）
"""
import os
import asyncio
import logging
from typing import Optional

logger = logging.getLogger("SpeechEngine")

# 默认语音配置
DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"  # 晓晓（女声，温柔）
VOICES = {
    "xiaoxiao": "zh-CN-XiaoxiaoNeural",   # 晓晓 - 温柔女声
    "yunxi": "zh-CN-YunxiNeural",         # 云希 - 阳光男声
    "yunyang": "zh-CN-YunyangNeural",     # 云扬 - 专业男声
    "xiaoyi": "zh-CN-XiaoyiNeural",       # 晓艺 - 活泼女声
    "yunjian": "zh-CN-YunjianNeural",     # 云健 - 沉稳男声
}


class SpeechEngine:
    """
    态极语音引擎

    将态极的文字回复转换为语音，让用户能"听到"态极说话。
    """

    def __init__(self, output_dir: str = None, voice: str = None):
        """
        Args:
            output_dir: 音频文件输出目录
            voice: 语音名称（xiaoxiao/yunxi/yunyang/xiaoyi/yunjian）
        """
        if output_dir is None:
            from taiji.config import get_taiji_data_path
            output_dir = get_taiji_data_path("speech_output")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.voice = VOICES.get(voice, voice or DEFAULT_VOICE)
        self._initialized = False

        logger.info(f"SpeechEngine initialized: voice={self.voice}")

    async def _generate_speech(self, text: str, output_path: str,
                                rate: str = "+0%", volume: str = "+0%") -> str:
        """
        生成语音文件（异步）

        Args:
            text: 要转换的文字
            output_path: 输出文件路径
            rate: 语速（如 "+20%" 加快，"-10%" 减慢）
            volume: 音量（如 "+50%" 加大）

        Returns:
            输出文件路径
        """
        try:
            import edge_tts
        except ImportError:
            raise ImportError("需要安装 edge-tts: pip install edge-tts")

        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=rate,
            volume=volume,
        )
        await communicate.save(output_path)
        return output_path

    def speak(self, text: str, filename: str = None,
              rate: str = "+0%", volume: str = "+0%") -> str:
        """
        让态极"说话"（同步接口）

        Args:
            text: 要说的文字
            filename: 输出文件名（默认自动生成）
            rate: 语速
            volume: 音量

        Returns:
            音频文件路径
        """
        if not filename:
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
            filename = f"taiji_speech_{text_hash}.mp3"

        output_path = os.path.join(self.output_dir, filename)

        # 运行异步生成
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                self._generate_speech(text, output_path, rate, volume)
            )
        finally:
            loop.close()

        logger.info(f"Speech generated: {output_path}")
        return output_path

    def speak_stream(self, text: str, callback=None):
        """
        流式语音生成（边生成边播放）

        Args:
            text: 要说的文字
            callback: 每生成一段音频时的回调函数
        """
        try:
            import edge_tts
        except ImportError:
            raise ImportError("需要安装 edge-tts: pip install edge-tts")

        async def _stream():
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
            )
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    if callback:
                        callback(chunk["data"])

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_stream())
        finally:
            loop.close()

    def set_voice(self, voice_name: str):
        """切换语音"""
        if voice_name in VOICES:
            self.voice = VOICES[voice_name]
            logger.info(f"Voice changed to: {self.voice}")
        else:
            logger.warning(f"Unknown voice: {voice_name}")

    def list_voices(self) -> dict:
        """列出可用语音"""
        return VOICES.copy()

    def get_status(self) -> dict:
        """获取状态"""
        return {
            "voice": self.voice,
            "output_dir": self.output_dir,
            "available_voices": list(VOICES.keys()),
        }
