#!/usr/bin/env python3
"""Build a quota-balanced corpus for Taiji native tokenizer training."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


TEXT_KEYS = (
    "text",
    "content",
    "code",
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

DEFAULT_RATIOS = {
    "zh": 0.30,
    "en": 0.30,
    "code": 0.15,
    "math": 0.10,
    "tech": 0.10,
    "taiji_special": 0.05,
}

NORMALIZED_CATEGORY_HINTS = {
    "zh": ("skypile", "fineweb2_zh", "cmn", "chinese", "_zh"),
    "en": ("fineweb_edu", "fineweb2_en", "falcon_refinedweb", "english", "_en"),
    "code": ("codeparrot", "code", "stack"),
    "math": ("openwebmath", "math", "arxiv"),
}

TECH_SUFFIXES = {
    ".jsonl",
    ".md",
    ".rst",
    ".txt",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".vue",
    ".css",
    ".html",
    ".yml",
    ".yaml",
    ".toml",
}

SKIP_DIRS = {".git", "__pycache__", "node_modules", "model_cache", ".cache"}
SKIP_SUFFIXES = {
    ".pt",
    ".pth",
    ".safetensors",
    ".model",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".parquet",
    ".gz",
    ".zip",
}

TAIJI_SPECIAL_LINES = [
    "?????????? AI ????????????????????",
    "??????????????????????????",
    "<think>????????????????????????</think>",
    "<inner_voice>????????????</inner_voice>",
    "<reflect><cause>??????????</cause><correct>????? schema ?????</correct></reflect>",
    "<plan><goal>??????</goal><step>?????</step><step>????</step><step>????</step></plan>",
    '<tool_call>{"name":"search","args":{"query":"?? ?? ?? ??"}}</tool_call>',
    "<tool_result>?????????????????????</tool_result>",
    "<final_answer>?????????????????</final_answer>",
    "<observe>????????????????????</observe>",
    "<screen>???? JupyterLab ??? AutoDL ?????</screen>",
    "<mem_read>?????????????????</mem_read>",
    "<mem_write>????????? native-v2 contract?</mem_write>",
    "<image>???????????</image>",
    "<audio>?????????????</audio>",
    "The native Taiji tokenizer separates control tokens, multimodal tokens, and text tokens.",
]


@dataclass
class CategoryReport:
    name: str
    quota_chars: int
    written_chars: int = 0
    records: int = 0
    files: list[str] = field(default_factory=list)

    @property
    def shortfall_chars(self) -> int:
        return max(0, self.quota_chars - self.written_chars)

    @property
    def fill_ratio(self) -> float:
        if self.quota_chars <= 0:
            return 1.0
        return self.written_chars / self.quota_chars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced Taiji tokenizer corpus")
    parser.add_argument(
        "--normalized-dir",
        default="taiji_data/training_data/pretrain_12b/normalized",
        help="Directory containing normalized pretraining JSONL files.",
    )
    parser.add_argument(
        "--project-dir",
        default=".",
        help="Project root used for technical docs and Taiji special-format text.",
    )
    parser.add_argument(
        "--tech-dir",
        action="append",
        default=None,
        help="Extra technical/documentation directory. Can be repeated.",
    )
    parser.add_argument(
        "--taiji-dir",
        action="append",
        default=None,
        help="Extra Taiji-specific data directory. Can be repeated.",
    )
    parser.add_argument(
        "--output",
        default="taiji_data/training_data/pretrain_12b/tokenizer_sample/native_v2_corpus_balanced.txt",
    )
    parser.add_argument(
        "--report",
        default="taiji_data/training_data/pretrain_12b/reports/tokenizer_corpus_balanced_report.json",
    )
    parser.add_argument("--max-chars", type=int, default=2_000_000_000)
    parser.add_argument("--min-text-chars", type=int, default=32)
    parser.add_argument("--max-record-chars", type=int, default=20_000)
    parser.add_argument(
        "--ratios",
        default=",".join(f"{name}:{ratio}" for name, ratio in DEFAULT_RATIOS.items()),
        help="Comma-separated category ratios, e.g. zh:0.3,en:0.3,code:0.15.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any category fills less than --strict-threshold of its quota.",
    )
    parser.add_argument(
        "--strict-threshold",
        type=float,
        default=0.9,
        help="Minimum fill ratio required when --strict is set.",
    )
    return parser.parse_args()


def parse_ratios(raw: str) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, value = item.partition(":")
        if not name or not value:
            raise ValueError(f"Invalid ratio item: {item}")
        ratios[name.strip()] = float(value)
    missing = [name for name in DEFAULT_RATIOS if name not in ratios]
    if missing:
        raise ValueError(f"Missing required categories: {', '.join(missing)}")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("Ratio total must be positive")
    return {name: value / total for name, value in ratios.items()}


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def should_skip_path(path: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    if lowered_parts & SKIP_DIRS:
        return True
    return path.suffix.lower() in SKIP_SUFFIXES


def iter_text_from_json(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
        return
    if not isinstance(obj, dict):
        return
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
            if not isinstance(step, dict):
                continue
            for key in ("thought", "action", "observation", "final_answer"):
                value = step.get(key)
                if isinstance(value, str):
                    yield value


def iter_jsonl_texts(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield line
            else:
                yield from iter_text_from_json(obj)


def iter_plain_texts(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        if path.suffix.lower() in {".json", ".jsonl"}:
            yield from iter_jsonl_texts(path)
            return
        chunks: list[str] = []
        chars = 0
        for line in handle:
            if not line.strip():
                if chunks:
                    yield "".join(chunks)
                    chunks = []
                    chars = 0
                continue
            chunks.append(line)
            chars += len(line)
            if chars >= 8_000:
                yield "".join(chunks)
                chunks = []
                chars = 0
        if chunks:
            yield "".join(chunks)


def clean_text(text: str, *, min_chars: int, max_chars: int) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    if len(text) < min_chars:
        return ""
    if text.count("\ufffd") / max(1, len(text)) > 0.02:
        return ""
    if has_pathological_repetition(text):
        return ""
    return text


def has_pathological_repetition(text: str) -> bool:
    if len(text) < 200:
        return False
    for char in set(text[:200]):
        if char and text.count(char * 40):
            return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 10:
        return False
    unique_ratio = len(set(lines)) / len(lines)
    return unique_ratio < 0.35


def normalized_sources_by_category(normalized_dir: Path) -> dict[str, list[Path]]:
    sources: dict[str, list[Path]] = {name: [] for name in DEFAULT_RATIOS}
    if not normalized_dir.exists():
        return sources
    # 递归查找所有 JSONL 文件
    for path in sorted(normalized_dir.rglob("*.jsonl")):
        lower_name = path.name.lower()
        # 也检查父目录名称
        parent_name = path.parent.name.lower()
        for category, hints in NORMALIZED_CATEGORY_HINTS.items():
            if any(hint in lower_name or hint in parent_name for hint in hints):
                sources[category].append(path)
                break
    return sources


def project_files(paths: list[Path], suffixes: set[str]) -> list[Path]:
    files: list[Path] = []
    for root in paths:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix.lower() in suffixes and not should_skip_path(root):
                files.append(root)
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes and not should_skip_path(path):
                files.append(path)
    return sorted(files)


def default_tech_dirs(project_dir: Path) -> list[Path]:
    return [project_dir / "docs", project_dir / "scripts", project_dir / "taiji"]


def default_taiji_dirs(project_dir: Path) -> list[Path]:
    return [
        project_dir / "taiji" / "data",
        project_dir / "taiji_data" / "training_data" / "pretrain_12b" / "tokenizer_sample",
    ]


def write_text_record(handle, text: str) -> int:
    text = "\n".join(part.rstrip() for part in text.splitlines()).strip()
    if not text:
        return 0
    handle.write(text.replace("\n", "\\n") + "\n")
    return len(text)


def fill_category_from_files(
    *,
    handle,
    report: CategoryReport,
    paths: list[Path],
    seen: set[str],
    min_chars: int,
    max_record_chars: int,
    unlimited: bool = False,
) -> None:
    for path in paths:
        if not unlimited and report.written_chars >= report.quota_chars:
            break
        if should_skip_path(path):
            continue
        wrote_file = False
        try:
            iterator = iter_jsonl_texts(path) if path.suffix.lower() == ".jsonl" else iter_plain_texts(path)
            for raw_text in iterator:
                if not unlimited and report.written_chars >= report.quota_chars:
                    break
                text = clean_text(raw_text, min_chars=min_chars, max_chars=max_record_chars)
                if not text:
                    continue
                digest = stable_hash(text[:4096])
                if digest in seen:
                    continue
                seen.add(digest)
                written = write_text_record(handle, text)
                if written <= 0:
                    continue
                report.written_chars += written
                report.records += 1
                wrote_file = True
        except OSError:
            continue
        if wrote_file:
            report.files.append(str(path))


def fill_taiji_seed_lines(
    *,
    handle,
    report: CategoryReport,
    seen: set[str],
    min_chars: int,
    max_record_chars: int,
) -> None:
    variants = []
    for line in TAIJI_SPECIAL_LINES:
        variants.append(line)
        variants.append(f"<context>{line}</context>")
        variants.append(f"<assistant>{line}</assistant>")

    for raw_text in variants:
        if report.written_chars >= report.quota_chars:
            break
        text = clean_text(raw_text, min_chars=min_chars, max_chars=max_record_chars)
        if not text:
            continue
        digest = stable_hash(text)
        if digest in seen:
            continue
        seen.add(digest)
        written = write_text_record(handle, text)
        if written <= 0:
            continue
        report.written_chars += written
        report.records += 1


def build_balanced_corpus(args: argparse.Namespace) -> dict[str, Any]:
    ratios = parse_ratios(args.ratios)
    normalized_dir = Path(args.normalized_dir)
    project_dir = Path(args.project_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    reports = {
        name: CategoryReport(name=name, quota_chars=int(args.max_chars * ratio))
        for name, ratio in ratios.items()
    }

    normalized_sources = normalized_sources_by_category(normalized_dir)
    tech_dirs = [Path(p) for p in args.tech_dir] if args.tech_dir else default_tech_dirs(project_dir)
    taiji_dirs = [Path(p) for p in args.taiji_dir] if args.taiji_dir else default_taiji_dirs(project_dir)
    tech_files = project_files(tech_dirs, TECH_SUFFIXES)
    taiji_files = project_files(taiji_dirs, TECH_SUFFIXES)

    seen: set[str] = set()
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        # 中文、英文、代码、数学：使用所有可用数据（不限制配额）
        for category in ("zh", "en", "code", "math"):
            fill_category_from_files(
                handle=handle,
                report=reports[category],
                paths=normalized_sources.get(category, []),
                seen=seen,
                min_chars=args.min_text_chars,
                max_record_chars=args.max_record_chars,
                unlimited=True,
            )

        # 技术文档：使用所有可用数据（不限制配额）
        fill_category_from_files(
            handle=handle,
            report=reports["tech"],
            paths=tech_files,
            seen=seen,
            min_chars=args.min_text_chars,
            max_record_chars=args.max_record_chars,
            unlimited=True,
        )

        # 态极特殊：使用所有可用数据（不限制配额）
        fill_taiji_seed_lines(
            handle=handle,
            report=reports["taiji_special"],
            seen=seen,
            min_chars=args.min_text_chars,
            max_record_chars=args.max_record_chars,
        )
        fill_category_from_files(
            handle=handle,
            report=reports["taiji_special"],
            paths=taiji_files,
            seen=seen,
            min_chars=args.min_text_chars,
            max_record_chars=args.max_record_chars,
            unlimited=True,
        )

    result = {
        "output": str(output),
        "output_size_bytes": output.stat().st_size if output.exists() else 0,
        "target_total_chars": args.max_chars,
        "actual_total_chars": sum(item.written_chars for item in reports.values()),
        "ratios": ratios,
        "categories": {
            name: {
                "quota_chars": item.quota_chars,
                "written_chars": item.written_chars,
                "records": item.records,
                "fill_ratio": round(item.fill_ratio, 4),
                "shortfall_chars": item.shortfall_chars,
                "files": item.files[:100],
                "file_count": len(item.files),
            }
            for name, item in reports.items()
        },
        "advice": build_advice(reports),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["report"] = str(report_path)
    return result


def build_advice(reports: dict[str, CategoryReport]) -> list[str]:
    advice: list[str] = []
    for name, report in reports.items():
        if report.fill_ratio >= 0.9:
            continue
        if name == "code":
            advice.append("code quota is low: add more codeparrot/the-stack style normalized code JSONL.")
        elif name == "math":
            advice.append("math quota is low: add more openwebmath/arxiv/math-heavy normalized JSONL.")
        elif name == "tech":
            advice.append("tech quota is low: add official docs, API docs, README/manual datasets via --tech-dir.")
        elif name == "taiji_special":
            advice.append("taiji_special quota is low: generate more Taiji tool/memory/planning traces via --taiji-dir.")
        elif name == "en":
            advice.append("English quota is low: add fineweb/falcon-refinedweb/fineweb2_en shards.")
        elif name == "zh":
            advice.append("Chinese quota is low: add skypile/fineweb2_zh shards.")
    return advice


def main() -> None:
    args = parse_args()
    result = build_balanced_corpus(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.strict:
        bad = [
            name
            for name, item in result["categories"].items()
            if item["fill_ratio"] < args.strict_threshold
        ]
        if bad:
            raise SystemExit(f"Underfilled categories: {', '.join(bad)}")


if __name__ == "__main__":
    main()

