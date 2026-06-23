#!/usr/bin/env python3
"""Verify the Taiji native tokenizer v2 contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taiji.tokenizer_native_v2 import TaijiNativeTokenizerV2


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Taiji native tokenizer v2")
    parser.add_argument("--tokenizer-dir", default="taiji/tokenizer_native_v2")
    parser.add_argument("--contract", default="taiji/tokenizer_contract.json")
    args = parser.parse_args()

    tokenizer_dir = Path(args.tokenizer_dir)
    sp_path = tokenizer_dir / "sentencepiece.model"
    contract_path = Path(args.contract)
    if not sp_path.exists():
        raise FileNotFoundError(sp_path)

    tok = TaijiNativeTokenizerV2(str(sp_path), str(contract_path))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert tok.vocab_size == contract["total_vocab_size"]
    assert tok.pad_token_id == 0
    assert tok.unk_token_id == 1
    assert tok.bos_token_id == 2
    assert tok.eos_token_id == 3
    assert tok.sp.GetPieceSize() <= contract["text_vocab_size"]

    for token, expected in contract["special_tokens"].items():
        ids = tok.encode(token)
        assert ids == [expected], f"{token}: {ids} != {[expected]}"

    text = "<think>态极需要调用工具</think><tool_call>{\"name\":\"search\"}</tool_call>你好，world"
    ids = tok.encode(text)
    assert all(0 <= i < tok.vocab_size for i in ids)
    decoded = tok.decode(ids, skip_special_tokens=False)
    assert "<think>" in decoded and "<tool_call>" in decoded and "你好" in decoded

    image_raw = torch.tensor([[0, 8191]])
    image_ids = tok.encode_image(image_raw)
    assert image_ids.tolist() == [[1000, 9191]]
    assert tok.decode_image(image_ids).tolist() == image_raw.tolist()

    audio_raw = torch.tensor([[0, 4095]])
    audio_ids = tok.encode_audio(audio_raw)
    assert audio_ids.tolist() == [[9192, 13287]]
    assert tok.decode_audio(audio_ids).tolist() == audio_raw.tolist()

    print("native tokenizer v2 OK")
    print(f"sp_vocab={tok.sp.GetPieceSize()}")
    print(f"total_vocab={tok.vocab_size}")


if __name__ == "__main__":
    main()
