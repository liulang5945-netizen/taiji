"""
态极 (Taiji) 知识→智力 转化引擎
核心: 让学到的知识真正变成模型的智能，而不是单纯的存储

三条路径:
1. 多模态知识提取 (视频/音频/图片 → 结构化知识)
2. 知识蒸馏 (知识库 → 微调训练数据)
3. 增量微调 (训练数据 → 模型参数更新)
"""
import os
import json
import logging
import tempfile
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger("Taiji.KnowledgeToIntelligence")


# ======================== Phase 1: 多模态知识提取 ========================

class MultimodalExtractor:
    """
    多模态知识提取器
    从视频/音频/图片中提取结构化知识
    """

    def __init__(self, vision_encoder=None, audio_encoder=None, video_engine=None):
        self.vision = vision_encoder
        self.audio = audio_encoder
        self.video = video_engine

    def extract_from_url(self, url: str) -> Optional[Dict]:
        """
        根据URL自动识别类型并提取知识

        Returns:
            {"source": url, "type": str, "content": str, "title": str} 或 None
        """
        from urllib.parse import urlparse
        path = urlparse(url).path.lower()

        # 视频网站
        video_domains = ["youtube.com", "youtu.be", "bilibili.com", "v.youku.com"]
        if any(d in url for d in video_domains):
            return self.extract_from_video_url(url)

        # 直接视频文件
        if any(path.endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]):
            return self.extract_from_video_file(url)

        # 音频
        if any(path.endswith(ext) for ext in [".mp3", ".wav", ".flac", ".ogg", ".m4a"]):
            return self.extract_from_audio(url)

        # 图片
        if any(path.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"]):
            return self.extract_from_image(url)

        # 默认作为网页处理
        return None

    def extract_from_video_url(self, url: str) -> Optional[Dict]:
        """
        从视频网站URL提取知识
        策略: 获取字幕/描述 → 转为文字 → 提取知识
        """
        try:
            content_parts = []

            # 尝试获取字幕 (youtube_transcript_api)
            try:
                from youtube_transcript_api import YouTubeTranscriptApi
                video_id = None
                if "youtube.com/watch?v=" in url:
                    video_id = url.split("v=")[1].split("&")[0]
                elif "youtu.be/" in url:
                    video_id = url.split("youtu.be/")[1].split("?")[0]

                if video_id:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['zh-Hans', 'zh-CN', 'en'])
                    text = " ".join([item['text'] for item in transcript])
                    content_parts.append(f"[视频字幕]\n{text[:5000]}")
            except Exception:
                pass

            # 尝试获取b站字幕
            try:
                from tools.bilibili_subtitle import get_bilibili_subtitle
                subtitle = get_bilibili_subtitle(url)
                if subtitle:
                    content_parts.append(f"[B站字幕]\n{subtitle[:3000]}")
            except Exception:
                pass

            if content_parts:
                return {
                    "source": url,
                    "type": "video_transcript",
                    "content": "\n\n".join(content_parts),
                    "title": f"视频: {url[:60]}",
                }

            # 降级: 用网页读取获取视频页面描述
            try:
                import requests
                from bs4 import BeautifulSoup
                resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                soup = BeautifulSoup(resp.text, "html.parser")
                # 提取标题和描述
                title = ""
                desc = ""
                for meta in soup.find_all("meta"):
                    if meta.get("property") == "og:title":
                        title = meta.get("content", "")
                    if meta.get("name") == "description" or meta.get("property") == "og:description":
                        desc = meta.get("content", "")
                if title or desc:
                    text = f"标题: {title}\n描述: {desc}"
                    return {"source": url, "type": "video_meta", "content": text[:3000], "title": title}
            except Exception:
                pass

            return {"source": url, "type": "video", "content": "无法获取视频字幕内容", "title": f"视频: {url[:50]}"}

        except Exception as e:
            logger.error(f"视频知识提取失败: {e}")
            return None

    def extract_from_video_file(self, path: str) -> Optional[Dict]:
        """从本地视频文件提取知识"""
        try:
            content_parts = []
            filename = os.path.basename(path)

            # 提取帧描述
            if self.video:
                frames = self.video.extract_frames(path, fps=0.2, max_frames=5)
                if frames and self.vision:
                    for frame in frames:
                        desc = self.vision.describe_image_simple(frame)
                        content_parts.append(f"[帧画面] {desc}")

            # 提取音频并转文字
            if self.audio:
                try:
                    audio_path = None
                    try:
                        from moviepy import VideoFileClip
                        clip = VideoFileClip(path)
                        audio_path = path + "_audio_temp.wav"
                        clip.audio.write_audiofile(audio_path, logger=None)
                        clip.close()
                    except Exception:
                        pass

                    if audio_path and os.path.exists(audio_path):
                        transcription = self.audio.transcribe(audio_path)
                        if transcription and "[语音识别" not in transcription:
                            content_parts.append(f"[语音内容]\n{transcription}")
                        try:
                            os.remove(audio_path)
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"音频提取失败: {e}")

            content = "\n\n".join(content_parts) if content_parts else f"视频文件: {filename}"
            return {"source": path, "type": "video_file", "content": content[:5000], "title": filename}

        except Exception as e:
            logger.error(f"视频文件提取失败: {e}")
            return None

    def extract_from_audio(self, path_or_url: str) -> Optional[Dict]:
        """从音频提取知识（语音转文字）"""
        if not self.audio:
            return None

        try:
            # 下载远程音频
            file_path = path_or_url
            if path_or_url.startswith("http"):
                try:
                    import requests
                    resp = requests.get(path_or_url, timeout=30)
                    _, file_path = tempfile.mkstemp(suffix=".mp3", prefix="audio_temp_")
                    with open(file_path, "wb") as f:
                        f.write(resp.content)
                except Exception as e:
                    return {"source": path_or_url, "type": "audio",
                            "content": f"下载失败: {e}", "title": "音频文件"}

            transcription = self.audio.transcribe(file_path)
            title = os.path.basename(path_or_url)

            if file_path != path_or_url:
                try:
                    os.remove(file_path)
                except Exception:
                    pass

            return {"source": path_or_url, "type": "audio_transcription",
                    "content": transcription[:5000], "title": title}

        except Exception as e:
            return {"source": path_or_url, "type": "audio",
                    "content": f"识别失败: {e}", "title": os.path.basename(path_or_url)}

    def extract_from_image(self, path_or_url: str) -> Optional[Dict]:
        """从图片提取知识"""
        if not self.vision:
            return None

        try:
            file_path = path_or_url
            if path_or_url.startswith("http"):
                try:
                    import requests
                    resp = requests.get(path_or_url, timeout=30)
                    _, file_path = tempfile.mkstemp(suffix=".jpg", prefix="img_temp_")
                    with open(file_path, "wb") as f:
                        f.write(resp.content)
                except Exception as e:
                    return {"source": path_or_url, "type": "image",
                            "content": f"下载失败: {e}", "title": "图片"}

            desc = self.vision.describe_image_simple(file_path)
            title = os.path.basename(path_or_url)

            if file_path != path_or_url:
                try:
                    os.remove(file_path)
                except Exception:
                    pass

            return {"source": path_or_url, "type": "image_description",
                    "content": desc[:2000], "title": title}

        except Exception as e:
            return {"source": path_or_url, "type": "image",
                    "content": f"识别失败: {e}", "title": os.path.basename(path_or_url)}


# ======================== Phase 2: 知识→训练数据 ========================

class KnowledgeDistiller:
    """
    知识蒸馏器
    将知识库内容转化为模型微调用的训练数据
    """

    def __init__(self, knowledge_store=None):
        self.store = knowledge_store

    def distill_to_training_data(self, domains: List[str] = None,
                                  max_samples: int = 500) -> List[Dict]:
        """
        将知识库中的条目蒸馏为训练样本

        转化格式:
        知识条目 → {"instruction": "什么是XXX?", "output": "XXX是..."}
        """
        if not self.store:
            logger.warning("知识库未连接，无法蒸馏")
            return []

        samples = []
        domains_to_process = domains or [d["name"] for d in self.store.list_domains()]

        for domain in domains_to_process:
            entries = self.store.get_entries_by_domain(domain)
            for entry in entries:
                if not entry.concept or not entry.content:
                    continue

                # 生成 Question-Answer 对
                question = f"解释一下{entry.concept}"
                answer = entry.content[:1000]

                samples.append({
                    "instruction": question,
                    "input": "",
                    "output": answer,
                    "domain": domain,
                    "source": entry.source,
                    "confidence": entry.confidence,
                })

                # 对长知识生成多个样本
                if len(entry.content) > 500:
                    samples.append({
                        "instruction": f"详细讲解{entry.concept}的原理",
                        "input": "",
                        "output": entry.content[:1500],
                        "domain": domain,
                        "source": entry.source,
                        "confidence": entry.confidence * 0.9,
                    })

                if len(samples) >= max_samples:
                    break
            if len(samples) >= max_samples:
                break

        logger.info(f"知识蒸馏完成: {len(samples)} 条训练样本，{len(domains_to_process)} 个领域")
        return samples[:max_samples]

    def save_training_data(self, output_dir: str = "./training_data",
                           domains: List[str] = None) -> str:
        """保存蒸馏后的训练数据为 JSONL 文件"""
        samples = self.distill_to_training_data(domains=domains)
        if not samples:
            return ""

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"knowledge_distilled_{timestamp}.jsonl")

        with open(filepath, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info(f"训练数据已保存: {filepath} ({len(samples)} 条)")
        return filepath


# ======================== Phase 3: 增量微调 ========================

class KnowledgeToIntelligence:
    """
    知识→智力 转化主引擎
    整合提取→蒸馏→微调 完整流程
    """

    def __init__(self, knowledge_learner=None, model=None, tokenizer=None):
        self.knowledge_learner = knowledge_learner
        self.model = model
        self.tokenizer = tokenizer
        self.extractor = MultimodalExtractor()
        self.distiller = KnowledgeDistiller()

    def set_taiji(self, model, tokenizer):
        """设置态极模型引用"""
        self.model = model
        self.tokenizer = tokenizer

    def learn_from_url(self, url: str, domain: str = "通用") -> Dict:
        """
        从任意URL学习知识（自动识别视频/音频/图片/网页）

        Returns:
            {"success": bool, "source_type": str, "content_summary": str}
        """
        from agent.knowledge_learner import KnowledgeEntry
        import uuid, datetime

        result = self.extractor.extract_from_url(url)
        if not result or not result.get("content"):
            return {"success": False, "source_type": "unknown",
                    "message": "无法从该URL提取知识"}

        if self.knowledge_learner and hasattr(self.knowledge_learner, 'store'):
            store = self.knowledge_learner.store
            entry = KnowledgeEntry(
                id=str(uuid.uuid4())[:12],
                domain=domain,
                concept=result.get("title", "知识")[:100],
                content=result["content"][:5000],
                source=url,
                source_type=result.get("type", "web"),
                tags=[result.get("type", "web"), domain],
                confidence=0.7,
                created_at=datetime.datetime.now().isoformat(),
            )
            store.add_entry(entry)

        return {
            "success": True,
            "source_type": result.get("type", "web"),
            "content_summary": result["content"][:200],
            "title": result.get("title", ""),
            "domain": domain,
        }

    def start_intelligence_boost(self, domains: List[str] = None) -> Dict:
        """
        启动智力增强流程:
        1. 知识蒸馏 (知识库 → 训练数据)
        2. 增量微调 (训练数据 → 模型参数)

        Returns:
            {"status": str, "samples_generated": int, "trained": bool}
        """
        if self.model is None:
            return {"status": "error", "message": "模型未加载"}

        # Step 1: 知识蒸馏
        if self.knowledge_learner:
            self.distiller.store = self.knowledge_learner.store
            samples = self.distiller.distill_to_training_data(domains=domains)
        else:
            samples = []

        if not samples:
            return {"status": "error", "samples_generated": 0, "message": "没有可蒸馏的知识"}

        # Step 2: 保存训练数据
        data_path = self.distiller.save_training_data()
        logger.info(f"智力增强: 生成 {len(samples)} 条训练样本 -> {data_path}")

        # Step 3: 增量微调
        trained = self._incremental_train(data_path)

        return {
            "status": "completed" if trained else "data_only",
            "samples_generated": len(samples),
            "data_path": data_path,
            "trained": trained,
            "domains": domains or ["全部"],
        }

    def _incremental_train(self, data_path: str, epochs: int = 1) -> bool:
        """执行增量微调"""
        try:
            from taiji.train.trainer import ModelSelfTrainer, build_dataset
            import torch

            if self.model is None or self.tokenizer is None:
                logger.warning("模型未加载，跳过微调")
                return False

            # 加载蒸馏数据
            trajs = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    trajs.append(json.loads(line))

            if not trajs:
                return False

            # 使用 ReActDataset 包装
            dataset = build_dataset(self.tokenizer, extra_react_data=[], extra_conv_data=[])
            # 自定义: 将蒸馏样本添加到数据集
            from taiji.train.trainer import ReActDataset
            class KnowledgeDataset(ReActDataset):
                def __init__(self, tokenizer, knowledge_samples, max_length=512):
                    super().__init__(tokenizer, [], [], max_length)
                    self.samples = []
                    for s in knowledge_samples:
                        inp = s.get("instruction", "")
                        out = s.get("output", "")
                        if inp and out:
                            self.samples.append({
                                "input_text": f"[用户] {inp}\n[助手] ",
                                "target_text": out,
                                "tool_target": -100,
                            })
                    logger.info(f"知识微调数据集: {len(self.samples)} 条")

            dataset = KnowledgeDataset(self.tokenizer, trajs)

            # 创建训练器
            trainer = ModelSelfTrainer(
                self.model, self.tokenizer,
                learning_rate=2e-5,  # 低学习率避免灾难性遗忘
                warmup_steps=10,
                max_grad_norm=0.5,
            )

            device = next(self.model.parameters()).device
            logger.info(f"开始增量微调: {len(dataset)} 条样本, {epochs} epoch")

            for progress, desc, losses, metrics in trainer.train(
                dataset, num_epochs=epochs, batch_size=2,
                save_dir="./checkpoints/knowledge_growth",
                save_steps=50, log_steps=5, device=str(device),
            ):
                logger.info(f"  微调进度: {progress:.0%} - {desc}")

            logger.info("✅ 增量微调完成，智力已提升")
            return True

        except Exception as e:
            logger.error(f"增量微调失败: {e}")
            import traceback
            traceback.print_exc()
            return False