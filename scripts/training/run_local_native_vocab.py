#!/usr/bin/env python3
"""Run the local tokenizer rebuild workflow for Taiji native-v2."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NORMALIZED_DIR = PROJECT_ROOT / "taiji_data" / "training_data" / "pretrain_mix_v1"
DEFAULT_LOCAL_SOURCE_DIR = PROJECT_ROOT / "taiji_data" / "tokenizer" / "local_vocab_sources"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "taiji_data" / "training_data" / "reports" / "tokenizer_corpus_balanced_local_report.json"
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "taiji_data" / "tokenizer" / "native_v2_corpus_balanced_local.txt"
DEFAULT_TOKENIZER_DIR = PROJECT_ROOT / "taiji" / "tokenizer_native_v2"
DEFAULT_CONTRACT = PROJECT_ROOT / "taiji" / "tokenizer_contract.json"
DEFAULT_TECH_DIR = DEFAULT_LOCAL_SOURCE_DIR / "tech"
DEFAULT_TAIJI_DIR = DEFAULT_LOCAL_SOURCE_DIR / "taiji_special"
DEFAULT_TECH_FILE = DEFAULT_TECH_DIR / "tech_docs_vocab.jsonl"
DEFAULT_TAIJI_FILE = DEFAULT_TAIJI_DIR / "taiji_special_vocab.jsonl"

RATIO_KEYS = ("zh", "en", "code", "math", "tech", "taiji_special")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Taiji tokenizer rebuild workflow")
    parser.add_argument("--normalized-dir", default=str(DEFAULT_NORMALIZED_DIR))
    parser.add_argument("--project-dir", default=str(PROJECT_ROOT))
    parser.add_argument("--local-source-dir", default=str(DEFAULT_LOCAL_SOURCE_DIR))
    parser.add_argument("--tech-output", default=str(DEFAULT_TECH_FILE))
    parser.add_argument("--taiji-output", default=str(DEFAULT_TAIJI_FILE))
    parser.add_argument("--corpus-output", default=str(DEFAULT_CORPUS_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER_DIR))
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--max-chars", type=int, default=400_000_000)
    parser.add_argument("--min-effective-chars", type=int, default=20_000_000)
    parser.add_argument("--strict-threshold", type=float, default=0.9)
    parser.add_argument("--tech-max-records", type=int, default=50_000)
    parser.add_argument("--taiji-records", type=int, default=20_000)
    parser.add_argument("--model-type", default="bpe", choices=["bpe", "unigram"])
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail if any category is under --strict-threshold.")
    return parser.parse_args()


def run_python(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(script), *args]
    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="ignore",
    )


def load_report(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_effective_max_chars(report: dict[str, object], requested_max_chars: int) -> int:
    categories = report["categories"]
    supported_chars = min(int(categories[key]["written_chars"] / report["ratios"][key]) for key in RATIO_KEYS)
    return min(requested_max_chars, supported_chars)


def summarize_fill_ratios(report: dict[str, object]) -> dict[str, float]:
    categories = report["categories"]
    return {key: float(categories[key]["fill_ratio"]) for key in RATIO_KEYS}


def main() -> None:
    args = parse_args()

    normalized_dir = Path(args.normalized_dir)
    local_source_dir = Path(args.local_source_dir)
    tech_output = Path(args.tech_output)
    taiji_output = Path(args.taiji_output)
    corpus_output = Path(args.corpus_output)
    report_path = Path(args.report)
    tokenizer_dir = Path(args.tokenizer_dir)

    local_source_dir.mkdir(parents=True, exist_ok=True)
    tech_output.parent.mkdir(parents=True, exist_ok=True)
    taiji_output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    tech_result = run_python(
        PROJECT_ROOT / "scripts" / "data_prep" / "generate_tech_docs_vocab_corpus.py",
        "--project-dir",
        str(args.project_dir),
        "--output",
        str(tech_output),
        "--max-records",
        str(args.tech_max_records),
    )
    taiji_result = run_python(
        PROJECT_ROOT / "scripts" / "data_prep" / "generate_taiji_special_vocab_corpus.py",
        "--output",
        str(taiji_output),
        "--records",
        str(args.taiji_records),
    )

    probe_report = report_path.with_name(report_path.stem + "_probe.json")
    probe_output = corpus_output.with_name(corpus_output.stem + "_probe.txt")
    probe_result = run_python(
        PROJECT_ROOT / "scripts" / "build_balanced_native_vocab_corpus.py",
        "--normalized-dir",
        str(normalized_dir),
        "--project-dir",
        str(args.project_dir),
        "--tech-dir",
        str(tech_output.parent),
        "--taiji-dir",
        str(taiji_output.parent),
        "--output",
        str(probe_output),
        "--report",
        str(probe_report),
        "--max-chars",
        str(args.max_chars),
    )
    probe_data = load_report(probe_report)
    effective_max_chars = compute_effective_max_chars(probe_data, args.max_chars)
    if effective_max_chars < args.min_effective_chars:
        raise SystemExit(
            f"Effective balanced max chars too low: {effective_max_chars} < min_effective_chars={args.min_effective_chars}"
        )

    build_args = [
        "--normalized-dir",
        str(normalized_dir),
        "--project-dir",
        str(args.project_dir),
        "--tech-dir",
        str(tech_output.parent),
        "--taiji-dir",
        str(taiji_output.parent),
        "--output",
        str(corpus_output),
        "--report",
        str(report_path),
        "--max-chars",
        str(effective_max_chars),
        "--strict-threshold",
        str(args.strict_threshold),
    ]
    if args.strict:
        build_args.append("--strict")
    build_result = run_python(
        PROJECT_ROOT / "scripts" / "build_balanced_native_vocab_corpus.py",
        *build_args,
    )

    train_result = None
    verify_result = None
    if not args.skip_train:
        train_result = run_python(
            PROJECT_ROOT / "scripts" / "train_native_text_sp.py",
            "--corpus",
            str(corpus_output),
            "--output-dir",
            str(tokenizer_dir),
            "--contract",
            str(args.contract),
            "--model-type",
            args.model_type,
        )

    if not args.skip_verify:
        verify_result = run_python(
            PROJECT_ROOT / "scripts" / "verify_native_tokenizer.py",
            "--tokenizer-dir",
            str(tokenizer_dir),
            "--contract",
            str(args.contract),
        )

    final_report = load_report(report_path)
    summary = {
        "normalized_dir": str(normalized_dir),
        "tech_output": str(tech_output),
        "taiji_output": str(taiji_output),
        "corpus_output": str(corpus_output),
        "report": str(report_path),
        "tokenizer_dir": str(tokenizer_dir),
        "requested_max_chars": args.max_chars,
        "effective_max_chars": effective_max_chars,
        "fill_ratios": summarize_fill_ratios(final_report),
        "tech_generator": json.loads(tech_result.stdout),
        "taiji_generator": json.loads(taiji_result.stdout),
        "probe_stdout": probe_result.stdout.strip().splitlines()[-1] if probe_result.stdout.strip() else "",
        "build_stdout": build_result.stdout.strip().splitlines()[-1] if build_result.stdout.strip() else "",
        "train_stdout": train_result.stdout.strip().splitlines() if train_result else [],
        "verify_stdout": verify_result.stdout.strip().splitlines() if verify_result else [],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
