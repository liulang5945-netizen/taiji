"""
态极图像引擎 (Image Engine)
============================
让态极能"画图"。

支持：
- 文生图（Text-to-Image）
- 图像描述（Image Captioning）

使用本地模型或 API：
- 本地：Stable Diffusion（需要 GPU）
- API：Stability AI、DALL-E（需要 API key）
"""
import os
import logging
import hashlib
from typing import Optional

logger = logging.getLogger("ImageEngine")


class ImageEngine:
    """
    态极图像引擎

    让态极能根据文字描述生成图片。
    """

    def __init__(self, output_dir: str = None, backend: str = "auto"):
        """
        Args:
            output_dir: 图片输出目录
            backend: 后端（"auto"/"local"/"api"）
        """
        if output_dir is None:
            from taiji.config import get_taiji_data_path
            output_dir = get_taiji_data_path("image_output")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.backend = backend
        self._pipeline = None
        self._api_key = None

        logger.info(f"ImageEngine initialized: backend={backend}")

    def _load_local_pipeline(self):
        """加载本地 Stable Diffusion 模型"""
        try:
            import torch
            from diffusers import StableDiffusionPipeline

            model_id = "stabilityai/stable-diffusion-2-1"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            logger.info(f"Loading Stable Diffusion on {device}...")
            self._pipeline = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=dtype
            )
            self._pipeline = self._pipeline.to(device)
            logger.info("Stable Diffusion loaded")
            return True
        except Exception as e:
            logger.warning(f"Failed to load local model: {e}")
            return False

    def generate(self, prompt: str, negative_prompt: str = None,
                 width: int = 512, height: int = 512,
                 steps: int = 30, guidance_scale: float = 7.5,
                 filename: str = None) -> str:
        """
        根据文字描述生成图片

        Args:
            prompt: 图片描述
            negative_prompt: 不想要的内容
            width: 图片宽度
            height: 图片高度
            steps: 推理步数
            guidance_scale: 引导强度
            filename: 输出文件名

        Returns:
            图片文件路径
        """
        if not filename:
            text_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
            filename = f"taiji_image_{text_hash}.png"

        output_path = os.path.join(self.output_dir, filename)

        # 尝试本地生成
        if self.backend in ("auto", "local"):
            if self._pipeline is None:
                self._load_local_pipeline()
            if self._pipeline:
                return self._generate_local(prompt, negative_prompt,
                                            width, height, steps,
                                            guidance_scale, output_path)

        # 尝试 API 生成
        if self.backend in ("auto", "api"):
            return self._generate_api(prompt, output_path, width, height)

        raise RuntimeError("没有可用的图像生成后端")

    def _generate_local(self, prompt, negative_prompt, width, height,
                        steps, guidance_scale, output_path) -> str:
        """使用本地模型生成图片"""
        try:
            image = self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
            ).images[0]
            image.save(output_path)
            logger.info(f"Image generated (local): {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Local generation failed: {e}")
            raise

    def _generate_api(self, prompt, output_path, width, height) -> str:
        """使用 API 生成图片（纯 stdlib）"""
        try:
            import urllib.request
            import base64 as _base64

            api_key = self._get_api_key()
            if not api_key:
                raise RuntimeError("未配置 Stability AI API key")

            payload = json.dumps({
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 7, "width": width, "height": height, "steps": 30, "samples": 1,
            }).encode('utf-8')
            req = urllib.request.Request(
                "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode('utf-8'))

            image_data = _base64.b64decode(data["artifacts"][0]["base64"])
            with open(output_path, "wb") as f:
                f.write(image_data)

            logger.info(f"Image generated (API): {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"API generation failed: {e}")
            raise

    def _get_api_key(self) -> Optional[str]:
        """获取 API key"""
        if self._api_key:
            return self._api_key
        return os.environ.get("STABILITY_API_KEY")

    def set_api_key(self, key: str):
        """设置 API key"""
        self._api_key = key

    def describe_image(self, image_path: str) -> str:
        """
        描述图片内容

        Args:
            image_path: 图片路径

        Returns:
            图片描述文字
        """
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            from PIL import Image

            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

            image = Image.open(image_path).convert("RGB")
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=50)
            description = processor.decode(out[0], skip_special_tokens=True)

            logger.info(f"Image described: {description}")
            return description
        except Exception as e:
            logger.error(f"Image description failed: {e}")
            return f"[无法描述图片: {e}]"

    def get_status(self) -> dict:
        """获取状态"""
        return {
            "backend": self.backend,
            "local_model_loaded": self._pipeline is not None,
            "output_dir": self.output_dir,
        }
