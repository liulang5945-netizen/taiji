"""Backward-compatible multimodal metadata shim.

This module used to define an older standalone multimodal token space and
mutate ``taiji.config.SPECIAL_TOKENS`` at import time. Native-v2 now uses
``taiji/tokenizer_contract.json`` as the single source of truth, so this file
only keeps lightweight compatibility metadata and must not redefine token IDs.
"""

from taiji.config import MM_CONTROL_TOKENS


class Modality:
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


MULTIMODAL_TOKENS = {
    "image_start": MM_CONTROL_TOKENS["image_start"],
    "image_end": MM_CONTROL_TOKENS["image_end"],
    "audio_start": MM_CONTROL_TOKENS["audio_start"],
    "audio_end": MM_CONTROL_TOKENS["audio_end"],
    "video_start": MM_CONTROL_TOKENS.get("video_start"),
    "video_end": MM_CONTROL_TOKENS.get("video_end"),
    "img_row": MM_CONTROL_TOKENS["img_row"],
    "gen_image": MM_CONTROL_TOKENS["gen_image"],
    "gen_audio": MM_CONTROL_TOKENS["gen_audio"],
    "gen_video": MM_CONTROL_TOKENS.get("gen_video"),
}


MULTIMODAL_TOOLS = {
    "generate_image": {
        "description": "根据文字描述生成图像",
        "input_format": '{"prompt": "描述", "size": "512x512"}',
        "modality": Modality.IMAGE,
        "output": "image_path",
    },
    "edit_image": {
        "description": "编辑现有图像",
        "input_format": '{"image_path": "图片路径", "instruction": "指令"}',
        "modality": Modality.IMAGE,
        "output": "image_path",
    },
    "describe_image": {
        "description": "描述图像内容",
        "input_format": '{"image_path": "图片路径"}',
        "modality": Modality.TEXT,
        "output": "text",
    },
    "transcribe_audio": {
        "description": "将音频转为文字（语音识别）",
        "input_format": '{"audio_path": "音频路径"}',
        "modality": Modality.TEXT,
        "output": "text",
    },
    "text_to_speech": {
        "description": "将文字转为语音（语音合成）",
        "input_format": '{"text": "文本", "voice": "声音"}',
        "modality": Modality.AUDIO,
        "output": "audio_path",
    },
    "understand_video": {
        "description": "理解视频内容并描述",
        "input_format": '{"video_path": "视频路径"}',
        "modality": Modality.TEXT,
        "output": "text",
    },
    "generate_video": {
        "description": "根据文字描述生成视频",
        "input_format": '{"prompt": "描述", "duration": "时长"}',
        "modality": Modality.VIDEO,
        "output": "video_path",
    },
    "video_to_gif": {
        "description": "将视频转换为GIF动图",
        "input_format": '{"video_path": "视频路径", "fps": "帧率"}',
        "modality": Modality.IMAGE,
        "output": "image_path",
    },
}


TAIJI_VERSION = "4.0.0"
TAIJI_NAME = "态极"
TAIJI_NAME_EN = "Taiji"
TAIJI_DESCRIPTION = "Taiji 原生多模态 Agent 模型 — 天生完整的智能体"
