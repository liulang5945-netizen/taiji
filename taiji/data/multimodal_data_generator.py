"""
态极多模态训练数据生成器

扫描本地图片/音频/视频文件，自动生成 ReAct 格式的多模态训练数据。
模型学到：看到多模态任务 → 调用对应工具 → 根据工具返回结果生成回答。

用法：
    from taiji.data.multimodal_data_generator import MultimodalDataGenerator
    gen = MultimodalDataGenerator()
    samples = gen.generate_from_directory("/path/to/media")
    # 保存为 JSONL
    gen.save_jsonl(samples, "multimodal_train.jsonl")
"""
import os
import json
import random
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("Taiji.MultimodalDataGen")

# 支持的媒体格式
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff", ".tif"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".aac"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}


class MultimodalDataGenerator:
    """生成多模态 ReAct 训练数据"""

    # ── 任务模板 ──

    IMAGE_TASKS = [
        {"task": "描述这张图片的内容", "tool": "describe_image", "thought": "用户想了解图片内容，我需要使用视觉工具查看图片"},
        {"task": "这张图片里有什么", "tool": "describe_image", "thought": "需要使用 describe_image 工具来分析图片"},
        {"task": "识别图片中的文字", "tool": "describe_image", "thought": "用户需要 OCR 识别，我先用视觉工具查看图片"},
        {"task": "分析这张截图展示了什么", "tool": "describe_image", "thought": "这是一张截图，我需要使用视觉工具分析界面内容"},
        {"task": "这张照片是在哪里拍的", "tool": "describe_image", "thought": "需要分析照片的场景信息，使用视觉工具"},
        {"task": "图片中有几个人", "tool": "describe_image", "thought": "用户想统计人数，我先用视觉工具查看图片"},
        {"task": "解释这张图表的含义", "tool": "describe_image", "thought": "这是一张图表，我需要使用视觉工具分析数据"},
        {"task": "这张图片的色调和氛围是什么", "tool": "describe_image", "thought": "用户想了解图片的视觉风格，使用视觉工具"},
    ]

    AUDIO_TASKS = [
        {"task": "转录这段音频的内容", "tool": "transcribe_audio", "thought": "用户想转录音频，我需要使用语音识别工具"},
        {"task": "这段语音说了什么", "tool": "transcribe_audio", "thought": "需要使用 transcribe_audio 工具来识别语音内容"},
        {"task": "把这段音频转成文字", "tool": "transcribe_audio", "thought": "用户需要音频转文字，使用语音识别工具"},
        {"task": "音频里的人在说什么", "tool": "transcribe_audio", "thought": "需要使用语音识别工具来获取音频内容"},
    ]

    VIDEO_TASKS = [
        {"task": "描述这个视频的内容", "tool": "understand_video", "thought": "用户想了解视频内容，我需要使用视频理解工具"},
        {"task": "视频里发生了什么", "tool": "understand_video", "thought": "需要使用 understand_video 工具来分析视频"},
        {"task": "总结这个视频的要点", "tool": "understand_video", "thought": "用户需要视频摘要，使用视频理解工具"},
        {"task": "视频中有哪些关键画面", "tool": "understand_video", "thought": "需要使用视频工具来提取关键帧信息"},
    ]

    def generate_from_directory(
        self,
        media_dir: str,
        max_per_modality: int = 100,
        include_subdirs: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        扫描目录中的媒体文件，生成多模态训练数据。

        Args:
            media_dir: 媒体文件目录
            max_per_modality: 每种模态最多生成多少条
            include_subdirs: 是否递归扫描子目录

        Returns:
            ReAct 格式的训练数据列表
        """
        samples = []
        scan = Path(media_dir).rglob("*") if include_subdirs else Path(media_dir).glob("*")

        images, audios, videos = [], [], []
        for f in scan:
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext in IMAGE_EXTS:
                images.append(str(f))
            elif ext in AUDIO_EXTS:
                audios.append(str(f))
            elif ext in VIDEO_EXTS:
                videos.append(str(f))

        logger.info(f"扫描到: {len(images)} 图片, {len(audios)} 音频, {len(videos)} 视频")

        # 图片样本
        for path in images[:max_per_modality]:
            samples.append(self._make_sample(path, "image"))

        # 音频样本
        for path in audios[:max_per_modality]:
            samples.append(self._make_sample(path, "audio"))

        # 视频样本
        for path in videos[:max_per_modality]:
            samples.append(self._make_sample(path, "video"))

        random.shuffle(samples)
        logger.info(f"生成了 {len(samples)} 条多模态训练样本")
        return samples

    def generate_from_paths(
        self,
        image_paths: List[str] = None,
        audio_paths: List[str] = None,
        video_paths: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """从指定路径列表生成训练数据"""
        samples = []
        for p in (image_paths or []):
            samples.append(self._make_sample(p, "image"))
        for p in (audio_paths or []):
            samples.append(self._make_sample(p, "audio"))
        for p in (video_paths or []):
            samples.append(self._make_sample(p, "video"))
        return samples

    def generate_caption_pairs(
        self,
        image_dir: str,
        captions_file: Optional[str] = None,
        max_count: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        生成图片-描述对训练数据（用于视觉-语言对齐）。

        如果提供 captions_file（JSON/JSONL 格式，含 image-caption 映射），
        使用真实描述；否则生成占位描述让模型通过工具学习。

        captions_file 格式：
            {"image.jpg": "一只橘猫坐在垫子上", ...}
            或 JSONL: {"image": "image.jpg", "caption": "一只橘猫..."}
        """
        samples = []
        image_dir = Path(image_dir)

        # 加载描述
        captions = {}
        if captions_file and os.path.exists(captions_file):
            captions = self._load_captions(captions_file)

        # 扫描图片
        images = []
        for ext in IMAGE_EXTS:
            images.extend(image_dir.rglob(f"*{ext}"))

        for img_path in images[:max_count]:
            img_name = img_path.name
            caption = captions.get(img_name) or captions.get(str(img_path))

            if caption:
                # 有真实描述：生成对话格式训练数据
                samples.append({
                    "messages": [
                        {"role": "user", "content": "请描述这张图片"},
                        {"role": "assistant", "content": caption},
                    ]
                })
            else:
                # 无描述：生成 ReAct 格式（模型学调用工具）
                samples.append(self._make_sample(str(img_path), "image"))

        logger.info(f"生成了 {len(samples)} 条图片-描述对")
        return samples

    def _make_sample(self, media_path: str, modality: str) -> Dict[str, Any]:
        """生成一条 ReAct 格式的多模态训练样本"""
        if modality == "image":
            template = random.choice(self.IMAGE_TASKS)
        elif modality == "audio":
            template = random.choice(self.AUDIO_TASKS)
        elif modality == "video":
            template = random.choice(self.VIDEO_TASKS)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        tool_name = template["tool"]
        file_name = os.path.basename(media_path)

        return {
            "task": template["task"],
            "modality": modality,
            "media_path": media_path,
            "steps": [
                {
                    "thought": f"{template['thought']}。文件: {file_name}",
                    "action": tool_name,
                    "action_args": {"input": media_path},
                    "observation": f"[{modality.upper()} 工具返回结果]",
                },
                {
                    "thought": "已经获取了媒体内容，现在给出回答",
                    "final_answer": f"[基于{modality}内容的回答]",
                },
            ],
        }

    def _load_captions(self, path: str) -> dict:
        """加载图片描述文件"""
        captions = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content.startswith("{"):
                # JSON dict 或 JSONL
                if "\n" in content:
                    for line in content.split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            img = item.get("image") or item.get("file") or item.get("path")
                            cap = item.get("caption") or item.get("description") or item.get("text")
                            if img and cap:
                                captions[os.path.basename(img)] = cap
                        except json.JSONDecodeError:
                            pass
                else:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        captions = {os.path.basename(k): v for k, v in data.items()}
        except Exception as e:
            logger.warning(f"加载描述文件失败: {e}")
        return captions

    @staticmethod
    def save_jsonl(samples: List[Dict], output_path: str):
        """保存训练数据为 JSONL 格式"""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        logger.info(f"已保存 {len(samples)} 条样本到 {output_path}")


def generate_multimodal_seed_data() -> List[Dict[str, Any]]:
    """
    生成内置的多模态种子训练数据（无需外部文件）。
    模型通过这些样本学会：遇到多模态任务时调用正确的工具。
    """
    return [
        # ── 图像理解 ──
        {
            "task": "描述用户上传的图片",
            "modality": "image",
            "steps": [
                {"thought": "用户上传了图片想了解内容，我需要使用 describe_image 工具",
                 "action": "describe_image",
                 "action_args": {"input": "uploaded_image.png"}},
                {"thought": "图片分析完成，给出描述",
                 "final_answer": "这张图片展示了..."},
            ],
        },
        {
            "task": "识别截图中的错误信息",
            "modality": "image",
            "steps": [
                {"thought": "用户上传了错误截图，我需要先看图片内容",
                 "action": "describe_image",
                 "action_args": {"input": "error_screenshot.png"}},
                {"thought": "看到了错误信息，分析原因",
                 "final_answer": "截图显示了一个错误..."},
            ],
        },
        {
            "task": "分析这张数据图表",
            "modality": "image",
            "steps": [
                {"thought": "用户上传了图表，我需要使用视觉工具分析",
                 "action": "describe_image",
                 "action_args": {"input": "chart.png"}},
                {"thought": "图表数据已获取，进行分析",
                 "final_answer": "从图表可以看出..."},
            ],
        },
        # ── 音频理解 ──
        {
            "task": "转录这段语音消息",
            "modality": "audio",
            "steps": [
                {"thought": "用户发了一段语音，我需要转录它",
                 "action": "transcribe_audio",
                 "action_args": {"input": "voice_message.wav"}},
                {"thought": "语音已转录为文字",
                 "final_answer": "语音内容是：..."},
            ],
        },
        {
            "task": "把这段会议录音整理成文字",
            "modality": "audio",
            "steps": [
                {"thought": "用户需要转录会议录音，使用语音识别工具",
                 "action": "transcribe_audio",
                 "action_args": {"input": "meeting.mp3"}},
                {"thought": "录音已转录，整理格式",
                 "final_answer": "会议录音内容如下：..."},
            ],
        },
        # ── 视频理解 ──
        {
            "task": "总结这个教程视频的要点",
            "modality": "video",
            "steps": [
                {"thought": "用户上传了视频，我需要使用视频理解工具",
                 "action": "understand_video",
                 "action_args": {"input": "tutorial.mp4"}},
                {"thought": "视频内容已分析，总结要点",
                 "final_answer": "这个教程视频主要讲了..."},
            ],
        },
        {
            "task": "描述这个视频里发生了什么",
            "modality": "video",
            "steps": [
                {"thought": "用户想了解视频内容，使用视频理解工具",
                 "action": "understand_video",
                 "action_args": {"input": "video.mp4"}},
                {"thought": "视频分析完成",
                 "final_answer": "视频中展示了..."},
            ],
        },
        # ── 多模态组合 ──
        {
            "task": "对比这张图片和这段音频的内容",
            "modality": "image",
            "steps": [
                {"thought": "用户想对比图片和音频，我先看图片",
                 "action": "describe_image",
                 "action_args": {"input": "photo.jpg"}},
                {"thought": "图片已分析，再听音频",
                 "action": "transcribe_audio",
                 "action_args": {"input": "audio.wav"}},
                {"thought": "两个模态的内容都获取了，进行对比",
                 "final_answer": "图片显示的是...而音频说的是...两者的关系是..."},
            ],
        },
    ]
