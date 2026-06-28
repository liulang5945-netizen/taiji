"""Unified multimodal engine built on top of the native agent loop."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from taiji.core.native_agent import AgentResult, NativeAgentEngine
from taiji.multimodal.audio_encoder import TaijiAudioEncoder
from taiji.multimodal.image_generator import TaijiImageGenerator
from taiji.multimodal.video_engine import TaijiVideoEngine
from taiji.multimodal.vision_encoder import TaijiVisionEncoder
from taiji.multimodal.voice_generator import TaijiVoiceGenerator

logger = logging.getLogger("Taiji.Multimodal")


class _MultimodalToolRegistryAdapter:
    """Adapter that merges multimodal built-ins with the external tool registry."""

    def __init__(self, engine: "TaijiMultimodalEngine", external_registry=None):
        self._engine = engine
        self._external_registry = external_registry

    def get_tool_descriptions(self) -> str:
        lines = ["可用工具:"]
        for name, metadata in self._engine.MULTIMODAL_TOOLS.items():
            lines.append(f"- {name}: {metadata.get('description', '')}")
        if self._external_registry:
            try:
                extra = self._external_registry.get_tool_descriptions()
                if extra:
                    lines.append(extra)
            except Exception:
                try:
                    for tool in self._external_registry.list_tools(enabled_only=True):
                        lines.append(f"- {tool.name}: {tool.description}")
                except Exception:
                    pass
        return "\n".join(lines)

    def execute(self, tool_name: str, args: dict):
        if tool_name in self._engine.MULTIMODAL_TOOLS:
            return self._engine._execute_multimodal_tool(tool_name, args)
        if self._external_registry is None:
            raise RuntimeError(f"Tool not available: {tool_name}")
        return self._external_registry.execute(tool_name, args)

    def list_tools(self, enabled_only: bool = True):
        if self._external_registry is None:
            return []
        return self._external_registry.list_tools(enabled_only=enabled_only)


class TaijiMultimodalEngine:
    """Canonical Taiji engine for self-model mode."""

    MULTIMODAL_TOOLS = {
        "describe_image": {
            "description": "Describe an image file",
            "input_format": "image_path",
            "modality": "image",
            "output": "text",
        },
        "generate_image": {
            "description": "Generate an image from a text prompt",
            "input_format": "prompt | size",
            "modality": "image",
            "output": "image_path",
        },
        "transcribe_audio": {
            "description": "Transcribe an audio file to text",
            "input_format": "audio_path",
            "modality": "audio",
            "output": "text",
        },
        "text_to_speech": {
            "description": "Synthesize speech from text",
            "input_format": "text | voice",
            "modality": "audio",
            "output": "audio_path",
        },
        "understand_video": {
            "description": "Describe a video file",
            "input_format": "video_path",
            "modality": "video",
            "output": "text",
        },
        "generate_video": {
            "description": "Generate a video from a text prompt",
            "input_format": "prompt | duration",
            "modality": "video",
            "output": "video_path",
        },
        "video_to_gif": {
            "description": "Convert a video file to GIF",
            "input_format": "video_path | fps",
            "modality": "video",
            "output": "image_path",
        },
    }

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cpu",
        workspace_path: Optional[str] = None,
        memory_save_path: Optional[str] = None,
        stream_callback=None,
        max_steps: int = 15,
    ) -> None:
        self.text_engine = NativeAgentEngine(
            model,
            tokenizer,
            device=device,
            max_steps=max_steps,
            stream_callback=stream_callback,
            workspace_path=workspace_path,
            memory_save_path=memory_save_path,
        )

        self.vision = TaijiVisionEncoder(hidden_size=model.config.hidden_size, device=device)
        self.audio = TaijiAudioEncoder(hidden_size=model.config.hidden_size, device=device)
        self.image_gen = TaijiImageGenerator()
        self.voice_gen = TaijiVoiceGenerator()
        self.video = TaijiVideoEngine()

        self.max_steps = max_steps
        self.stream_callback = stream_callback
        self.workspace_path = workspace_path
        self.memory_save_path = memory_save_path

    def generate(self, prompt, tokenizer=None, max_new_tokens=256, temperature=0.7, top_p=0.9):
        return self.text_engine.inference.generate(prompt, tokenizer, max_new_tokens, temperature, top_p)

    def generate_stream(
        self,
        prompt,
        tokenizer=None,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        stop_event=None,
    ):
        return self.text_engine.inference.generate_stream(
            prompt,
            tokenizer,
            max_new_tokens,
            temperature,
            top_p,
            stop_event,
        )

    def run(self, task: str, tool_registry=None) -> AgentResult:
        merged_registry = _MultimodalToolRegistryAdapter(self, tool_registry)
        return self.text_engine.run(task, merged_registry)

    def _execute_multimodal_tool(self, tool_name: str, args: dict) -> str:
        try:
            input_value = args.get("input", "")

            def extract(key: str, default_index: int = 0, default_value: str = "") -> str:
                if key in args:
                    return args[key]
                parts = str(input_value).split("|")
                if len(parts) > default_index:
                    return parts[default_index].strip()
                return default_value

            if tool_name == "describe_image":
                return self.vision.describe_image_simple(extract("image_path", 0, str(input_value)))
            if tool_name == "generate_image":
                result = self.image_gen.generate(extract("prompt", 0, str(input_value)), extract("size", 1, "512x512"))
                if result.get("success"):
                    self.text_engine.memory.auto_write(f"<img_result>{result.get('path')}</img_result>", importance=0.8)
                return f"[Success] Image: {result.get('path')}" if result.get("success") else f"[Error] {result.get('error')}"
            if tool_name == "transcribe_audio":
                return self.audio.transcribe(extract("audio_path", 0, str(input_value)))
            if tool_name == "text_to_speech":
                result = self.voice_gen.synthesize(
                    extract("text", 0, str(input_value)),
                    extract("voice", 1, "zh-CN-XiaoxiaoNeural"),
                )
                if result.get("success"):
                    self.text_engine.memory.auto_write(f"<tts_result>{result.get('path')}</tts_result>", importance=0.8)
                return f"[Success] Audio: {result.get('path')}" if result.get("success") else f"[Error] {result.get('error')}"
            if tool_name == "understand_video":
                return self.video.understand_video(extract("video_path", 0, str(input_value)))
            if tool_name == "generate_video":
                result = self.video.generate_video(
                    extract("prompt", 0, str(input_value)),
                    int(extract("duration", 1, "4")),
                )
                if result.get("success"):
                    self.text_engine.memory.auto_write(f"<video>{result.get('path')}</video>", importance=0.8)
                return f"[Success] Video: {result.get('path')}" if result.get("success") else f"[Error] {result.get('error')}"
            if tool_name == "video_to_gif":
                result = self.video.video_to_gif(
                    extract("video_path", 0, str(input_value)),
                    fps=float(extract("fps", 1, "4.0")),
                )
                if result.get("success"):
                    self.text_engine.memory.auto_write(f"<img_result>{result.get('path')}</img_result>", importance=0.8)
                return f"[Success] GIF: {result.get('path')}" if result.get("success") else f"[Error] {result.get('error')}"
            return f"Unknown tool: {tool_name}"
        except Exception as exc:
            return f"Tool error: {exc}"

    def register_tools(self, tool_names: list) -> None:
        self.text_engine.register_tools(tool_names)

    def get_status(self) -> dict:
        return {
            "type": "multimodal_agent",
            "name": "Taiji",
            "version": "4.0.0",
            "memory": self.text_engine.memory.get_stats(),
            "plan": self.text_engine.planner.get_status(),
            "reflection": self.text_engine.reflector.get_stats(),
            "registered_tools": len(self.text_engine.inference.tokenizer._tool_name_to_id),
            "multimodal_tools": list(self.MULTIMODAL_TOOLS.keys()),
        }

    def cancel(self) -> None:
        self.text_engine.cancel()
