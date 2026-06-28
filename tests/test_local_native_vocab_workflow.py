from __future__ import annotations

import json
import shutil
from pathlib import Path

from scripts.data_prep.audit_1b_training_assets import tokenizer_candidates
from scripts.data_prep.generate_taiji_special_vocab_corpus import write_records as write_taiji_records
from scripts.data_prep.generate_tech_docs_vocab_corpus import write_records as write_tech_records
from scripts.training.run_local_native_vocab import compute_effective_max_chars, summarize_fill_ratios


def _make_test_dir(name: str) -> Path:
    path = Path("test_artifacts") / name
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_generate_tech_docs_vocab_records() -> None:
    test_dir = _make_test_dir("tech_docs_vocab")
    try:
        docs_dir = test_dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "guide.md").write_text("# API\n\nUse stable schemas and retries.\n", encoding="utf-8")
        (docs_dir / "service.py").write_text("def run_job(task_id: str) -> None:\n    return None\n", encoding="utf-8")
        output = test_dir / "tech_docs_vocab.jsonl"
        result = write_tech_records(
            output,
            project_dir=test_dir,
            input_dirs=[docs_dir],
            max_records=24,
            min_chars=16,
            max_record_chars=256,
        )

        lines = output.read_text(encoding="utf-8").splitlines()
        assert result["records"] >= 3
        assert len(lines) == result["records"]
        first = json.loads(lines[0])
        assert first["source"] == "tech_docs_vocab"
        assert "text" in first and first["text"]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_generate_taiji_special_records() -> None:
    test_dir = _make_test_dir("taiji_special_vocab")
    try:
        output = test_dir / "taiji_special_vocab.jsonl"
        result = write_taiji_records(output, records=16)

        lines = output.read_text(encoding="utf-8").splitlines()
        assert result["records"] == 16
        assert len(lines) == 16
        first = json.loads(lines[0])
        assert first["source"] == "taiji_special_vocab"
        assert "text" in first and first["text"]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_compute_effective_max_chars_uses_most_constrained_bucket() -> None:
    report = {
        "ratios": {
            "zh": 0.3,
            "en": 0.3,
            "code": 0.15,
            "math": 0.1,
            "tech": 0.1,
            "taiji_special": 0.05,
        },
        "categories": {
            "zh": {"written_chars": 30_000_000, "fill_ratio": 1.0},
            "en": {"written_chars": 30_000_000, "fill_ratio": 1.0},
            "code": {"written_chars": 15_000_000, "fill_ratio": 1.0},
            "math": {"written_chars": 10_000_000, "fill_ratio": 1.0},
            "tech": {"written_chars": 2_000_000, "fill_ratio": 0.2},
            "taiji_special": {"written_chars": 5_000_000, "fill_ratio": 1.0},
        },
    }

    assert compute_effective_max_chars(report, requested_max_chars=100_000_000) == 20_000_000


def test_summarize_fill_ratios_returns_all_expected_keys() -> None:
    report = {
        "categories": {
            "zh": {"fill_ratio": 1.0},
            "en": {"fill_ratio": 0.95},
            "code": {"fill_ratio": 0.91},
            "math": {"fill_ratio": 0.87},
            "tech": {"fill_ratio": 0.76},
            "taiji_special": {"fill_ratio": 0.99},
        }
    }

    ratios = summarize_fill_ratios(report)
    assert ratios["zh"] == 1.0
    assert ratios["tech"] == 0.76
    assert set(ratios) == {"zh", "en", "code", "math", "tech", "taiji_special"}


def test_audit_prefers_package_tokenizer_path_first() -> None:
    candidates = tokenizer_candidates()
    assert candidates[0].as_posix().endswith("taiji/tokenizer_native_v2/sentencepiece.model")
    assert candidates[1].as_posix().endswith("tokenizer_native_v2/sentencepiece.model")
