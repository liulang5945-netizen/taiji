"""Native Taiji tokenizer v2.

This tokenizer is driven by ``tokenizer_contract.json``. SentencePiece is only
responsible for the text range; Taiji control, tool, and multimodal IDs are
fixed by the contract.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch


class TaijiNativeTokenizerV2:
    """Tokenizer for the native Taiji 256K ID space."""

    def __init__(
        self,
        sp_model_path: str,
        contract_path: Optional[str] = None,
    ) -> None:
        import sentencepiece as spm

        self.sp_model_path = str(sp_model_path)
        if contract_path is None:
            contract_path = str(Path(__file__).with_name("tokenizer_contract.json"))
        self.contract_path = str(contract_path)

        with open(self.contract_path, "r", encoding="utf-8") as f:
            self.contract = json.load(f)

        self.text_offset = int(self.contract["text_offset"])
        self.text_vocab_size = int(self.contract["text_vocab_size"])
        self.total_vocab_size = int(self.contract["total_vocab_size"])

        self.special_text_to_id: Dict[str, int] = {
            k: int(v) for k, v in self.contract["special_tokens"].items()
        }
        self.special_text_to_id.update(
            {k: int(v) for k, v in self.contract["multimodal"]["control_tokens"].items()}
        )
        for spec in self.contract.get("generated_special_ranges", {}).values():
            pattern = spec["pattern"]
            start = int(spec["start"])
            count = int(spec["count"])
            for i in range(count):
                self.special_text_to_id[pattern.format(i=i)] = start + i

        self.special_id_to_text = {v: k for k, v in self.special_text_to_id.items()}
        self._special_tokens_by_length = sorted(
            self.special_text_to_id.keys(), key=len, reverse=True
        )

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.sp_model_path)
        if self.sp.GetPieceSize() > self.text_vocab_size:
            raise ValueError(
                f"SentencePiece vocab {self.sp.GetPieceSize()} exceeds text range "
                f"{self.text_vocab_size}"
            )

    @property
    def vocab_size(self) -> int:
        return self.total_vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.special_text_to_id["<pad>"]

    @property
    def unk_token_id(self) -> int:
        return self.special_text_to_id["<unk>"]

    @property
    def bos_token_id(self) -> int:
        return self.special_text_to_id["<s>"]

    @property
    def eos_token_id(self) -> int:
        return self.special_text_to_id["</s>"]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_token_id)

        pos = 0
        while pos < len(text):
            matched = None
            for token in self._special_tokens_by_length:
                if text.startswith(token, pos):
                    matched = token
                    break
            if matched is not None:
                ids.append(self.special_text_to_id[matched])
                pos += len(matched)
                continue

            next_pos = len(text)
            for token in self._special_tokens_by_length:
                found = text.find(token, pos + 1)
                if found != -1:
                    next_pos = min(next_pos, found)
            ids.extend(self._encode_text(text[pos:next_pos]))
            pos = next_pos

        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def _encode_text(self, text: str) -> List[int]:
        if not text:
            return []
        return [self.text_offset + i for i in self.sp.EncodeAsIds(text)]

    def decode(self, ids: Iterable[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        parts: List[str] = []
        text_ids: List[int] = []

        def flush_text() -> None:
            nonlocal text_ids
            if text_ids:
                parts.append(self.sp.DecodeIds(text_ids))
                text_ids = []

        for raw_id in ids:
            token_id = int(raw_id)
            if token_id == self.pad_token_id and skip_special_tokens:
                continue
            if token_id in self.special_id_to_text:
                flush_text()
                if not skip_special_tokens:
                    parts.append(self.special_id_to_text[token_id])
                continue
            if self.text_offset <= token_id < self.text_offset + self.text_vocab_size:
                text_ids.append(token_id - self.text_offset)
                continue
            flush_text()
            if not skip_special_tokens:
                parts.append(f"<reserved_{token_id}>")

        flush_text()
        return "".join(parts)

    def __call__(
        self,
        text: str | List[str],
        return_tensors: Optional[str] = None,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor | List[List[int]]]:
        texts = text if isinstance(text, list) else [text]
        encoded = [self.encode(t) for t in texts]
        if truncation and max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]
        if padding:
            target = max_length or max(len(ids) for ids in encoded)
            encoded = [ids + [self.pad_token_id] * max(0, target - len(ids)) for ids in encoded]

        attention = [[0 if tok == self.pad_token_id else 1 for tok in ids] for ids in encoded]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(encoded, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
            }
        return {"input_ids": encoded, "attention_mask": attention}

    def encode_image(self, raw_ids: torch.Tensor) -> torch.Tensor:
        image = self.contract["multimodal"]["image"]
        return raw_ids.long() + int(image["base"])

    def decode_image(self, ids: torch.Tensor) -> torch.Tensor:
        image = self.contract["multimodal"]["image"]
        return ids.long() - int(image["base"])

    def encode_audio(self, raw_ids: torch.Tensor) -> torch.Tensor:
        audio = self.contract["multimodal"]["audio"]
        return raw_ids.long() + int(audio["base"])

    def decode_audio(self, ids: torch.Tensor) -> torch.Tensor:
        audio = self.contract["multimodal"]["audio"]
        return ids.long() - int(audio["base"])

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer_contract.json"), "w", encoding="utf-8") as f:
            json.dump(self.contract, f, indent=2, ensure_ascii=False)
        with open(os.path.join(path, "tokenizer_native_v2.json"), "w", encoding="utf-8") as f:
            json.dump({"sp_model": "sentencepiece.model"}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TaijiNativeTokenizerV2":
        return cls(
            sp_model_path=os.path.join(path, "sentencepiece.model"),
            contract_path=os.path.join(path, "tokenizer_contract.json"),
        )

