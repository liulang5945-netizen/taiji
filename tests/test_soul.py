"""Focused soul-system integration tests.

These tests replace the old script-style file that executed long workflows at
import time. Keep them small, assertive, and pytest-native.
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from taiji.core.inference import NativeInferenceEngine

# These tests create real models and run training; only run with --runslow.
pytestmark = pytest.mark.slow
from taiji.data.data_generator import (
    generate_bulk_conversation_data,
    generate_bulk_react_data,
)
from taiji.data.knowledge_distiller import KnowledgeDistiller
from taiji.infra.user_profile import UserProfile
from taiji.life.evolution_engine import EvolutionEngine
from taiji.loader import create_model
from taiji.train.trainer import ModelSelfTrainer, ReActDataset, build_dataset


TOOL_NAMES = [
    "read_local_file",
    "write_file",
    "execute_python",
    "search",
]
TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".taiji_test_tmp"


@pytest.fixture()
def workspace_tmp_path(request: pytest.FixtureRequest) -> Iterator[Path]:
    """Create a repo-local temp directory to avoid host temp permission issues."""

    TEST_TMP_ROOT.mkdir(exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in request.node.name).strip("_")
    unique_name = f"{safe_name or 'test'}-{uuid.uuid4().hex[:8]}"
    path = TEST_TMP_ROOT / unique_name
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def model_and_tokenizer():
    model, tokenizer = create_model("125m", "cpu")
    for name in TOOL_NAMES:
        tokenizer.register_tool(name)
    model.set_num_tools(len(TOOL_NAMES))
    return model, tokenizer


def test_core_components_initialize(workspace_tmp_path: Path, model_and_tokenizer) -> None:
    model, tokenizer = model_and_tokenizer

    params_m = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    assert params_m > 0

    evolution_dir = workspace_tmp_path / "evolution"
    profile_dir = workspace_tmp_path / "profile"

    evolution = EvolutionEngine(data_dir=str(evolution_dir))
    profile = UserProfile(data_dir=str(profile_dir))
    inference = NativeInferenceEngine(model, tokenizer, "cpu")

    assert evolution.metrics.current_phase == "infant"
    assert profile.preferences.total_interactions >= 0
    assert inference is not None


def test_knowledge_distillation_and_generators_produce_samples() -> None:
    distiller = KnowledgeDistiller()
    react, conv = distiller.distill_all()
    generated_react = generate_bulk_react_data(5)
    generated_conv = generate_bulk_conversation_data(3)

    assert len(react) > 0
    assert len(conv) > 0
    assert len(generated_react) == 5
    assert len(generated_conv) == 3


def test_build_dataset_and_prompt_alignment(model_and_tokenizer) -> None:
    _, tokenizer = model_and_tokenizer

    distiller = KnowledgeDistiller()
    react, conv = distiller.distill_all()
    dataset = build_dataset(
        tokenizer,
        extra_react_data=react[:4] + generate_bulk_react_data(2),
        extra_conv_data=conv[:2] + generate_bulk_conversation_data(2),
        max_length=256,
    )

    assert len(dataset) > 0

    sample = dataset[0]
    decoded = tokenizer.decode(sample["input_ids"].tolist(), skip_special_tokens=False)
    assert "可用工具" in decoded
    assert sample["tool_target"].item() >= -100


def test_react_dataset_keeps_system_prompt_after_truncation(model_and_tokenizer) -> None:
    _, tokenizer = model_and_tokenizer

    dataset = ReActDataset(
        tokenizer,
        react_data=generate_bulk_react_data(1),
        conv_data=[],
        max_length=256,
    )

    sample = dataset[0]
    decoded = tokenizer.decode(sample["input_ids"].tolist(), skip_special_tokens=False)

    assert "[系统]" in decoded
    assert "可用工具" in decoded
    assert "[用户]" in decoded


@pytest.mark.slow
def test_single_epoch_finetune_completes(
    model_and_tokenizer, workspace_tmp_path: Path
) -> None:
    model, tokenizer = model_and_tokenizer

    distiller = KnowledgeDistiller()
    react, conv = distiller.distill_all()
    dataset = build_dataset(
        tokenizer,
        extra_react_data=react[:2] + generate_bulk_react_data(2),
        extra_conv_data=conv[:2] + generate_bulk_conversation_data(2),
        max_length=192,
    )

    trainer = ModelSelfTrainer(model, tokenizer, learning_rate=1e-4, warmup_steps=2)
    final_metrics = None

    for _fraction, _desc, _history, metrics in trainer.finetune(
        dataset,
        num_epochs=1,
        batch_size=4,
        log_steps=999,
        save_dir=str(workspace_tmp_path / "finetune"),
        device="cpu",
    ):
        final_metrics = metrics

    assert final_metrics is not None
    assert final_metrics.get("status") in {"completed", "early_stopped"}
