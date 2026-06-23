#!/usr/bin/env python3
"""Build a clean corpus for Taiji native tokenizer training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


TEXT_KEYS = (
    "text",
    "content",
    "output",
    "instruction",
    "input",
    "question",
    "answer",
    "task",
    "thought",
    "final_answer",
    "description",
)


TAIJI_SEED_LINES = [
    "态极是一个本地运行的 AI 生命体。",
    "态极通过感知、工具调用、记忆、规划和反思来完成任务。",
    "<think>我需要分析目标、选择工具、验证结果。</think>",
    '<tool_call>{"name":"search","args":{"query":"人工智能新闻"}}</tool_call>',
    "<tool_result>工具返回了可验证的信息。</tool_result>",
    "<reflect><cause>失败原因是参数不完整</cause><correct>下次先检查 schema</correct></reflect>",
    "<plan><goal>完成用户目标</goal><step>理解请求</step><step>执行操作</step></plan>",
    "Python 函数、类、模块、异常处理、异步编程、装饰器、类型标注。",
    "Transformer 使用注意力机制、RMSNorm、RoPE、GQA 和 SwiGLU。",
    "The native Taiji tokenizer separates control tokens, multimodal tokens, and text tokens.",
]


def iter_text_from_json(obj) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
        return
    if isinstance(obj, dict):
        for key in TEXT_KEYS:
            value = obj.get(key)
            if isinstance(value, str):
                yield value
        messages = obj.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    yield msg["content"]
        steps = obj.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if isinstance(step, dict):
                    for key in ("thought", "action", "final_answer"):
                        value = step.get(key)
                        if isinstance(value, str):
                            yield value


def iter_source_lines(path: Path) -> Iterable[str]:
    suffix = path.suffix.lower()
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if suffix in {".jsonl", ".json"}:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        yield line
                    else:
                        yield from iter_text_from_json(obj)
                else:
                    yield line
    except OSError:
        return


def should_skip(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    if {"node_modules", ".git", "__pycache__", "model_cache"} & parts:
        return True
    name = path.name.lower()
    return name.endswith((".pt", ".pth", ".safetensors", ".model", ".png", ".jpg", ".jpeg", ".webp"))


def build_corpus(data_dirs: list[Path], output: Path, max_chars: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    total_chars = 0
    files = 0
    suffixes = {".jsonl", ".json", ".txt", ".md", ".py", ".js", ".ts", ".vue", ".css", ".html", ".yml", ".yaml", ".toml"}

    with output.open("w", encoding="utf-8", newline="\n") as out:
        for line in TAIJI_SEED_LINES:
            out.write(line + "\n")
            total_chars += len(line)

        for data_dir in data_dirs:
            if not data_dir.exists():
                continue
            for path in data_dir.rglob("*"):
                if total_chars >= max_chars:
                    break
                if not path.is_file() or should_skip(path) or path.suffix.lower() not in suffixes:
                    continue
                wrote = False
                for text in iter_source_lines(path):
                    text = " ".join(text.split())
                    if len(text) < 2:
                        continue
                    out.write(text + "\n")
                    total_chars += len(text)
                    wrote = True
                    if total_chars >= max_chars:
                        break
                if wrote:
                    files += 1

    print(f"corpus={output}")
    print(f"files={files}")
    print(f"chars={total_chars}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Taiji native tokenizer corpus")
    parser.add_argument("--data-dir", action="append", required=True, help="Data directory; can be repeated")
    parser.add_argument("--output", default="taiji_data/tokenizer/native_v2_corpus.txt")
    parser.add_argument("--max-chars", type=int, default=1_000_000_000)
    args = parser.parse_args()

    build_corpus([Path(p) for p in args.data_dir], Path(args.output), args.max_chars)


if __name__ == "__main__":
    main()
