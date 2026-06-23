#!/usr/bin/env python3
"""Train the text-only SentencePiece model for Taiji native tokenizer v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Taiji native text SentencePiece")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output-dir", default="taiji/tokenizer_native_v2")
    parser.add_argument("--contract", default="taiji/tokenizer_contract.json")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--model-type", default="bpe", choices=["bpe", "unigram"])
    args = parser.parse_args()

    import sentencepiece as spm

    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    text_vocab_size = int(contract["text_vocab_size"])
    vocab_size = args.vocab_size or text_vocab_size
    if vocab_size > text_vocab_size:
        raise ValueError(f"vocab_size {vocab_size} exceeds text range {text_vocab_size}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = str(output_dir / "sentencepiece")

    spm.SentencePieceTrainer.train(
        input=args.corpus,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=args.model_type,
        character_coverage=0.9999,
        byte_fallback=True,
        normalization_rule_name="identity",
        add_dummy_prefix=True,
        remove_extra_whitespaces=False,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        split_digits=True,
        split_by_whitespace=True,
        split_by_unicode_script=True,
        split_by_number=True,
        max_sentence_length=16384,
        num_threads=8,
        input_sentence_size=0,
        shuffle_input_sentence=True,
        hard_vocab_limit=False,
    )

    (output_dir / "tokenizer_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"model={output_dir / 'sentencepiece.model'}")
    print(f"text_vocab_size={vocab_size}")
    print(f"total_vocab_size={contract['total_vocab_size']}")


if __name__ == "__main__":
    main()
