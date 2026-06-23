#!/usr/bin/env python3
"""Verify AutoDL native tokenizer files."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def load_contract(tok_dir: Path, contract_arg: str | None) -> dict[str, Any]:
    local_contract = tok_dir / "tokenizer_contract.json"
    if local_contract.exists():
        return json.loads(local_contract.read_text(encoding="utf-8"))
    if contract_arg:
        contract_path = Path(contract_arg)
        if contract_path.exists():
            tok_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(contract_path, local_contract)
            return json.loads(contract_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"Missing {local_contract}. Pass --contract /path/to/tokenizer_contract.json "
        "or rerun design_vocab_native_v2.py."
    )


def assert_range(name: str, actual: list[int], expected_start: int, expected_end: int) -> None:
    assert actual == [expected_start, expected_end], f"{name}: {actual} != {[expected_start, expected_end]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Taiji native v2 tokenizer")
    parser.add_argument("--tokenizer-dir", default="taiji/tokenizer_native_v2")
    parser.add_argument("--contract", default=None)
    args = parser.parse_args()

    import sentencepiece as spm

    tok_dir = Path(args.tokenizer_dir)
    sp_path = tok_dir / "sentencepiece.model"
    if not sp_path.exists():
        raise FileNotFoundError(
            f"Missing {sp_path}. Tokenizer training did not finish; rerun design_vocab_native_v2.py."
        )

    contract = load_contract(tok_dir, args.contract)
    sp = spm.SentencePieceProcessor()
    sp.Load(str(sp_path))

    assert contract["total_vocab_size"] == 256000
    assert contract["text_offset"] == 13388
    assert contract["text_vocab_size"] == 242612
    assert sp.GetPieceSize() <= contract["text_vocab_size"]

    assert_range("control", contract["ranges"]["control"], 0, 3)
    assert_range("taiji_special", contract["ranges"]["taiji_special"], 4, 999)
    assert_range("image", contract["ranges"]["image"], 1000, 9191)
    assert_range("audio", contract["ranges"]["audio"], 9192, 13287)
    assert_range("multimodal_control", contract["ranges"]["multimodal_control"], 13288, 13387)
    assert_range("text", contract["ranges"]["text"], 13388, 255999)

    special = contract["special_tokens"]
    assert special["<pad>"] == 0
    assert special["<unk>"] == 1
    assert special["<s>"] == 2
    assert special["</s>"] == 3
    assert special["<think>"] == 10
    assert special["<tool_call>"] == 20
    assert special["<final_answer>"] == 30

    multimodal = contract["multimodal"]
    assert multimodal["image"]["base"] == 1000
    assert multimodal["image"]["codebook_size"] == 8192
    assert multimodal["audio"]["base"] == 9192
    assert multimodal["audio"]["codebook_size"] == 4096
    assert multimodal["control_tokens"]["<mm_image>"] == 13288

    print("native tokenizer contract OK")
    print(f"tokenizer_dir={tok_dir}")
    print(f"sp_vocab={sp.GetPieceSize()}")
    print(f"text_offset={contract['text_offset']}")
    print(f"total_vocab={contract['total_vocab_size']}")


if __name__ == "__main__":
    main()
