"""Unified tokenizer entrypoint for Taiji native-v2."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from .tokenizer_native_v2 import TaijiNativeTokenizerV2


def _default_native_v2_sp_model() -> str:
    candidates = [
        Path(__file__).with_name("tokenizer_native_v2") / "sentencepiece.model",
        Path("taiji") / "tokenizer_native_v2" / "sentencepiece.model",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("No SentencePiece model found for the Taiji native-v2 tokenizer.")


class ModelSelfTokenizer(TaijiNativeTokenizerV2):
    """Project-level tokenizer wrapper with sensible defaults and legacy loading."""

    def __init__(
        self,
        sp_model_path: Optional[str] = None,
        contract_path: Optional[str] = None,
    ) -> None:
        resolved_sp_path = sp_model_path if sp_model_path and os.path.exists(sp_model_path) else _default_native_v2_sp_model()
        super().__init__(sp_model_path=resolved_sp_path, contract_path=contract_path)

    @classmethod
    def load(
        cls,
        path: str,
        sp_model_path: Optional[str] = None,
    ) -> "ModelSelfTokenizer":
        base_path = os.path.abspath(path)
        resolved_sp_path = sp_model_path or os.path.join(base_path, "sentencepiece.model")
        if not os.path.exists(resolved_sp_path):
            resolved_sp_path = _default_native_v2_sp_model()

        contract_path = os.path.join(base_path, "tokenizer_contract.json")
        tokenizer = cls(
            sp_model_path=resolved_sp_path,
            contract_path=contract_path if os.path.exists(contract_path) else None,
        )

        native_state_path = os.path.join(base_path, "tokenizer_native_v2.json")
        legacy_state_path = os.path.join(base_path, "tokenizer_config.json")

        if os.path.exists(native_state_path):
            with open(native_state_path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
            for tool_name, token_id in state.get("tool_mappings", {}).items():
                token_id = int(token_id)
                tokenizer._tool_name_to_id[tool_name] = token_id
                tokenizer._id_to_tool_name[token_id] = tool_name
                tokenizer.special_text_to_id[tool_name] = token_id
            tokenizer._next_tool_id = int(state.get("next_tool_id", tokenizer._next_tool_id))
            tokenizer._refresh_special_index()
            return tokenizer

        if os.path.exists(legacy_state_path):
            with open(legacy_state_path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
            for tool_name, token_id in state.get("tool_mappings", {}).items():
                token_id = int(token_id)
                tokenizer._tool_name_to_id[tool_name] = token_id
                tokenizer._id_to_tool_name[token_id] = tool_name
                tokenizer.special_text_to_id[tool_name] = token_id
            tokenizer._next_tool_id = int(state.get("next_tool_id", tokenizer._next_tool_id))
            tokenizer._refresh_special_index()

        return tokenizer
