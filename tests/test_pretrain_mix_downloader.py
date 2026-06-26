from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from uuid import uuid4

from scripts.data_prep.download_pretrain_mix_v1 import (
    SOURCES,
    extract_text,
    iter_json_gz_lines,
    iter_json_lines,
    normalize_text,
    resolve_sources,
    resolve_sources_from_args,
)


def _make_test_dir() -> Path:
    path = Path("test_artifacts") / f"pretrain_mix_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_resolve_sources_returns_expected_specs() -> None:
    sources = resolve_sources("fineweb_edu,skypile_zh")
    assert [source.name for source in sources] == ["fineweb_edu", "skypile_zh"]
    assert sources[0] == SOURCES["fineweb_edu"]


def test_resolve_sources_supports_new_english_sources() -> None:
    sources = resolve_sources("fineweb_edu,fineweb2_en")
    assert [source.name for source in sources] == ["fineweb_edu", "fineweb2_en"]
    assert sources[1] == SOURCES["fineweb2_en"]


def test_resolve_sources_supports_falcon_refinedweb() -> None:
    sources = resolve_sources("fineweb_edu,falcon_refinedweb_en")
    assert [source.name for source in sources] == ["fineweb_edu", "falcon_refinedweb_en"]
    assert sources[1] == SOURCES["falcon_refinedweb_en"]


def test_resolve_sources_from_args_uses_stage_preset() -> None:
    sources = resolve_sources_from_args(preset="english_boost", raw_sources="skypile_zh")
    assert [source.name for source in sources] == ["fineweb_edu", "fineweb2_en"]


def test_resolve_sources_from_args_supports_mirror_preset() -> None:
    sources = resolve_sources_from_args(preset="english_boost_mirror", raw_sources="skypile_zh")
    assert [source.name for source in sources] == [
        "fineweb_edu",
        "fineweb2_en",
        "falcon_refinedweb_en",
    ]


def test_normalize_text_preserves_newlines_but_cleans_control_chars() -> None:
    raw = " hello\r\nworld\x00 \r\n"
    assert normalize_text(raw) == "hello\nworld"


def test_extract_text_uses_configured_field() -> None:
    row = {"content": "print('taiji')", "text": "ignored"}
    assert extract_text(row, "content") == "print('taiji')"


def test_iter_json_lines_falls_back_to_text_wrapper() -> None:
    test_dir = _make_test_dir()
    try:
        path = test_dir / "sample.jsonl"
        path.write_text('{"text":"alpha"}\nplain text line\n', encoding="utf-8")

        rows = list(iter_json_lines(path))

        assert rows == [{"text": "alpha"}, {"text": "plain text line"}]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_iter_json_gz_lines_reads_gzip_jsonl() -> None:
    test_dir = _make_test_dir()
    try:
        path = test_dir / "sample.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps({"content": "beta"}) + "\n")

        rows = list(iter_json_gz_lines(path))

        assert rows == [{"content": "beta"}]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
