"""
态极多模态输出引擎 (Multimodal Output Engine)
==============================================
统一管理态极的多模态输出能力。

输出类型：
- 文字：语言模型直接生成
- 语音：TTS 引擎生成
- 图片：图像引擎生成
"""
import os
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger("MultimodalOutput")


class MultimodalOutputEngine:
    """
    态极多模态输出引擎

    统一管理态极的所有输出能力：
    - 文字输出（已有）
    - 语音输出（TTS）
    - 图片输出（文生图）
    """

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            from taiji.config import get_taiji_data_path
            output_dir = get_taiji_data_path("multimodal_output")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 懒加载引擎
        self._speech_engine = None
        self._image_engine = None

        logger.info("MultimodalOutputEngine initialized")

    @property
    def speech(self):
        """语音引擎（懒加载）"""
        if self._speech_engine is None:
            try:
                from taiji.multimodal.speech_engine import SpeechEngine
                self._speech_engine = SpeechEngine(
                    output_dir=os.path.join(self.output_dir, "speech")
                )
            except ImportError as e:
                logger.warning(f"Speech engine not available: {e}")
                return None
        return self._speech_engine

    @property
    def image(self):
        """图像引擎（懒加载）"""
        if self._image_engine is None:
            try:
                from taiji.multimodal.image_engine import ImageEngine
                self._image_engine = ImageEngine(
                    output_dir=os.path.join(self.output_dir, "images")
                )
            except ImportError as e:
                logger.warning(f"Image engine not available: {e}")
                return None
        return self._image_engine

    def text_to_speech(self, text: str, voice: str = None) -> Optional[str]:
        """
        文字转语音

        Args:
            text: 要转换的文字
            voice: 语音名称

        Returns:
            音频文件路径，失败返回 None
        """
        if not self.speech:
            logger.warning("Speech engine not available")
            return None

        try:
            if voice:
                self.speech.set_voice(voice)
            return self.speech.speak(text)
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return None

    def generate_image(self, prompt: str, **kwargs) -> Optional[str]:
        """
        文生图

        Args:
            prompt: 图片描述
            **kwargs: 其他参数（width, height, steps 等）

        Returns:
            图片文件路径，失败返回 None
        """
        if not self.image:
            logger.warning("Image engine not available")
            return None

        try:
            return self.image.generate(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return None

    def describe_image(self, image_path: str) -> Optional[str]:
        """
        描述图片

        Args:
            image_path: 图片路径

        Returns:
            图片描述，失败返回 None
        """
        if not self.image:
            logger.warning("Image engine not available")
            return None

        try:
            return self.image.describe_image(image_path)
        except Exception as e:
            logger.error(f"Image description failed: {e}")
            return None

    def get_available_outputs(self) -> List[str]:
        """获取可用的输出类型"""
        outputs = ["text"]  # 文字总是可用
        if self.speech:
            outputs.append("speech")
        if self.image:
            outputs.append("image")
        return outputs

    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            "available_outputs": self.get_available_outputs(),
            "speech": self.speech.get_status() if self.speech else None,
            "image": self.image.get_status() if self.image else None,
        }
