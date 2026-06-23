"""Smoke tests for installer-related model download wiring."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from taiji.model_ext.model_downloader import (
    DEFAULT_MODEL_DIR,
    MIRROR_URLS,
    ModelDownloader,
    diagnose_network,
    list_downloaded_models,
)
from taiji.model_ext.model_registry import get_all_models, get_model_download_info


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_root_on_sys_path() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    assert str(PROJECT_ROOT) in sys.path


def test_external_and_internal_paths_in_dev_mode() -> None:
    from taiji.core.utils import get_external_path, get_internal_path

    assert os.path.normpath(get_external_path("gguf_models")) == os.path.normpath(
        os.path.join(os.getcwd(), "gguf_models")
    )
    assert os.path.normpath(get_internal_path("model")) == os.path.normpath(
        os.path.join(os.getcwd(), "model")
    )


def test_model_registry_has_models_and_variants() -> None:
    models = get_all_models()

    assert models
    assert any(getattr(model, "variants", []) for model in models)


def test_model_download_info_for_known_models() -> None:
    for model_name in [
        "SmolLM2-360M-Instruct",
        "DeepSeek-R1-Distill-Qwen-1.5B",
        "Qwen2.5-7B-Instruct",
    ]:
        info = get_model_download_info(model_name, "Q4_K_M")
        assert info is not None
        assert "/" in info["repo"]
        assert info["filename"].endswith(".gguf")


def test_model_downloader_initialization(tmp_path: Path) -> None:
    downloader = ModelDownloader(save_dir=str(tmp_path), mirror=True, verify_ssl=True)

    assert downloader.save_dir == str(tmp_path)
    assert downloader.prefer_mirror is True
    assert downloader.verify_ssl is True
    assert downloader.progress.status == "idle"
    assert MIRROR_URLS[0][0] == "hf-mirror.com"


def test_model_downloader_default_directory() -> None:
    downloader = ModelDownloader()
    assert downloader.save_dir == DEFAULT_MODEL_DIR


def test_list_downloaded_models_returns_list(tmp_path: Path) -> None:
    models = list_downloaded_models(save_dir=str(tmp_path))
    assert isinstance(models, list)


def test_api_model_routes_registered() -> None:
    from api.app import app

    routes = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/api/models/download" in routes
    assert "/api/models/download_progress" in routes
    assert "/api/models/downloaded" in routes


def test_diagnose_network_shape() -> None:
    result = diagnose_network()

    assert "overall_status" in result
    assert "mirrors" in result
    assert isinstance(result["mirrors"], list)
