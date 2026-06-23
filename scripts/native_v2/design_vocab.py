#!/usr/bin/env python3
"""Train Taiji native v2 text SentencePiece for the fixed 256K ID space."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Iterable


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

SUPPORTED_SUFFIXES = {".jsonl", ".json", ".txt", ".md", ".py", ".csv", ".tsv", ".yaml", ".yml"}

SEED_LINES = [
    "态极是一个本地运行的 AI 生命体。",
    "态极拥有感知、工具调用、记忆、规划和反思能力。",
    "<think>我需要分析目标并选择可靠的行动。</think>",
    '<tool_call>{"name":"search","args":{"query":"AI 新闻"}}</tool_call>',
    "<tool_result>工具返回了结构化结果。</tool_result>",
    "<final_answer>我会给出简洁且可验证的回答。</final_answer>",
    "Transformer RMSNorm RoPE GQA SwiGLU causal attention.",
    "Python JavaScript TypeScript shell SQL Docker Kubernetes Git.",
]


def extract_text(obj: Any) -> Iterable[str]:
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


def normalize_text(text: str, max_text_chars: int) -> str:
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()
    if max_text_chars > 0 and len(text) > max_text_chars:
        text = text[:max_text_chars]
    return text


def iter_file_texts(path: Path, max_text_chars: int, max_lines_per_file: int) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw in enumerate(f, start=1):
            if max_lines_per_file > 0 and line_no > max_lines_per_file:
                break
            raw = normalize_text(raw, max_text_chars)
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                yield raw
                continue
            for text in extract_text(obj):
                text = normalize_text(text, max_text_chars)
                if text:
                    yield text


def build_corpus(
    data_dir: Path,
    output: Path,
    max_chars: int,
    max_file_mb: int,
    max_text_chars: int,
    max_lines_per_file: int,
    progress_every_files: int,
    progress_every_chars: int,
) -> int:
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    files_seen = 0
    files_used = 0
    last_report_chars = 0
    start = time.time()

    with output.open("w", encoding="utf-8", newline="\n") as out:
        for line in SEED_LINES:
            out.write(line + "\n")
            total += len(line)
        for path in data_dir.rglob("*"):
            if total >= max_chars:
                break
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            files_seen += 1
            try:
                size_mb = path.stat().st_size / (1024 * 1024)
                if max_file_mb > 0 and size_mb > max_file_mb:
                    print(f"[skip] {path} size={size_mb:.1f}MB > {max_file_mb}MB", flush=True)
                    continue

                wrote_this_file = False
                for text in iter_file_texts(path, max_text_chars, max_lines_per_file):
                    if len(text) < 2:
                        continue
                    out.write(text + "\n")
                    total += len(text)
                    wrote_this_file = True
                    if total >= max_chars:
                        break
                if wrote_this_file:
                    files_used += 1
            except OSError as exc:
                print(f"[skip] {path} error={exc}", flush=True)
                continue

            should_report_file = progress_every_files > 0 and files_seen % progress_every_files == 0
            should_report_chars = progress_every_chars > 0 and total - last_report_chars >= progress_every_chars
            if should_report_file or should_report_chars:
                elapsed = max(1e-6, time.time() - start)
                print(
                    f"[corpus] files_seen={files_seen} files_used={files_used} "
                    f"chars={total:,} speed={total / elapsed / 1e6:.2f}M chars/s",
                    flush=True,
                )
                last_report_chars = total

    print(f"corpus={output} chars={total:,} files_seen={files_seen} files_used={files_used}", flush=True)
    return total


def train_sp(corpus: Path, output_dir: Path, contract: dict[str, Any], vocab_size: int, num_threads: int) -> None:
    import sentencepiece as spm

    text_vocab_size = int(contract["text_vocab_size"])
    if vocab_size > text_vocab_size:
        raise ValueError(f"Text vocab {vocab_size} exceeds contract text range {text_vocab_size}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = str(output_dir / "sentencepiece")
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9999,
        byte_fallback=True,
        normalization_rule_name="identity",
        add_dummy_prefix=True,
        remove_extra_whitespaces=False,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        split_digits=True,
        split_by_whitespace=True,
        split_by_unicode_script=True,
        split_by_number=True,
        max_sentence_length=16384,
        num_threads=num_threads,
        input_sentence_size=0,
        shuffle_input_sentence=True,
        hard_vocab_limit=False,
    )
    (output_dir / "tokenizer_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Taiji native v2 text SentencePiece")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", default="taiji/tokenizer_native_v2")
    parser.add_argument("--contract", default="tokenizer_contract.json")
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--max_chars", type=int, default=1_000_000_000)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--max_file_mb", type=int, default=0)
    parser.add_argument("--max_text_chars", type=int, default=200_000)
    parser.add_argument("--max_lines_per_file", type=int, default=0)
    parser.add_argument("--progress_every_files", type=int, default=50)
    parser.add_argument("--progress_every_chars", type=int, default=50_000_000)
    parser.add_argument("--num_threads", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = Path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    shutil.copy2(contract_path, output_dir / "tokenizer_contract.json")

    vocab_size = args.vocab_size or int(contract["text_vocab_size"])
    corpus = Path(args.corpus) if args.corpus else output_dir / "native_vocab_corpus.txt"
    chars = build_corpus(
        data_dir=Path(args.data_dir),
        output=corpus,
        max_chars=args.max_chars,
        max_file_mb=args.max_file_mb,
        max_text_chars=args.max_text_chars,
        max_lines_per_file=args.max_lines_per_file,
        progress_every_files=args.progress_every_files,
        progress_every_chars=args.progress_every_chars,
    )
    if chars < 10_000:
        print(f"[warn] corpus is small ({chars:,} chars); tokenizer quality will be weak.", flush=True)
    print(f"[train] sentencepiece vocab_size={vocab_size} total_vocab={contract['total_vocab_size']}", flush=True)
    train_sp(corpus, output_dir, contract, vocab_size, args.num_threads)
    print(f"trained={output_dir / 'sentencepiece.model'}")
    print(f"sp_vocab={vocab_size} total_vocab={contract['total_vocab_size']}")


if __name__ == "__main__":
    main()
