from __future__ import annotations

from scripts.native_v2.pretrain import _should_skip_data_path


def test_should_skip_cache_and_manifest_paths() -> None:
    assert _should_skip_data_path("taiji_data/training_data/pretrain_mix_v1/manifest.json") is True
    assert _should_skip_data_path("taiji_data/training_data/raw_pretrain_mix_v1/.cache/huggingface/file.lock") is True


def test_should_keep_regular_training_jsonl() -> None:
    assert _should_skip_data_path("taiji_data/training_data/pretrain_mix_v1/fineweb_edu.jsonl") is False
