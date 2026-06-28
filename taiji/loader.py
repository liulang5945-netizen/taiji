"""Taiji model loader and saver for the native-v2 tokenizer contract."""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Callable, Optional

import torch

from .architecture import ModelSelf
from .config import ModelConfig, NATIVE_V2_TOKENIZER_CONTRACT
from .tokenizer import ModelSelfTokenizer

logger = logging.getLogger("Taiji")

_SIZE_BUILDERS: dict[str, Callable[[], ModelConfig]] = {
    "125m": ModelConfig.size_125m,
    "350m": ModelConfig.size_350m,
    "1b": ModelConfig.size_1b,
    "3b": ModelConfig.size_3b,
    "7b": ModelConfig.size_7b,
}


def create_model(
    size: str = "125m",
    device: str = "cpu",
    active_heads: list[str] | None = None,
    sp_model_path: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[ModelSelf, ModelSelfTokenizer]:
    """Create a fresh Taiji model and tokenizer pair."""
    normalized_size = str(size).strip().lower()
    builder = _SIZE_BUILDERS.get(normalized_size)
    if builder is None:
        raise ValueError(f"Unsupported model size: {size}")

    config = builder()
    if active_heads is not None:
        config.active_heads = list(active_heads)

    tokenizer = ModelSelfTokenizer(sp_model_path=sp_model_path)
    config.vocab_size = tokenizer.vocab_size
    config.base_vocab_size = int(NATIVE_V2_TOKENIZER_CONTRACT["text_vocab_size"])
    config.num_special_tokens = config.vocab_size - config.base_vocab_size
    model = ModelSelf(config).to(device=device, dtype=dtype)
    return model, tokenizer


def load_model(
    model_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[ModelSelf, ModelSelfTokenizer]:
    """Load a saved Taiji model directory."""
    model_path = os.path.abspath(model_path)
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        config_dict = json.load(handle)
    config = _dict_to_config(config_dict)

    model = ModelSelf(config)

    weights_path = os.path.join(model_path, "model.pt")
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded weights from %s", weights_path)
    else:
        logger.warning("No weights found at %s; using random initialization", weights_path)

    sp_path = _find_sentencepiece(model_path)
    if sp_path is None:
        raise FileNotFoundError(f"SentencePiece model not found in {model_path}")

    tokenizer = ModelSelfTokenizer.load(model_path, sp_model_path=sp_path)
    if hasattr(tokenizer, "_tool_name_to_id"):
        model.set_num_tools(len(tokenizer._tool_name_to_id))

    model = model.to(device=device, dtype=dtype)
    logger.info("Loaded model: %s", config.describe())
    logger.info("Loaded tokenizer: vocab=%s tools=%s", tokenizer.vocab_size, len(tokenizer.get_all_tool_ids()))
    return model, tokenizer


def _find_sentencepiece(model_path: str) -> Optional[str]:
    candidates = [
        os.path.join(model_path, "sentencepiece.model"),
        os.path.join(model_path, "tokenizer_native_v2", "sentencepiece.model"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer_native_v2", "sentencepiece.model"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def save_model(
    model: ModelSelf,
    tokenizer: ModelSelfTokenizer,
    save_path: str,
    training_state: Optional[dict] = None,
) -> None:
    """Save a Taiji model, tokenizer, and optional training state."""
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(_config_to_dict(model.config), handle, indent=2, ensure_ascii=False)

    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))

    if hasattr(tokenizer, "save"):
        tokenizer.save(save_path)

    sp_model_path = getattr(tokenizer, "sp_model_path", None)
    if sp_model_path and os.path.exists(sp_model_path):
        shutil.copyfile(sp_model_path, os.path.join(save_path, "sentencepiece.model"))

    if training_state is not None:
        torch.save(training_state, os.path.join(save_path, "training_state.pt"))

    logger.info("Model saved to %s", save_path)


def _config_to_dict(config: ModelConfig) -> dict:
    """Serialise ModelConfig using dataclass introspection — never drifts."""
    return config.to_dict()


def _dict_to_config(data: dict) -> ModelConfig:
    """Deserialise ModelConfig from dict."""
    return ModelConfig.from_dict(data)
