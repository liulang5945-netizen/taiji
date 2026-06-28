# Taiji Native-v2 Tokenizer

This is the only canonical tokenizer route for current Taiji pretraining.
Legacy HF/Qwen-compatible routes are not part of the current training path.

## Contract

`taiji/tokenizer_contract.json` owns the global 256K ID space:

- `0-3`: fixed control tokens
- `4-999`: Taiji structural tokens
- `1000-9191`: image codebook IDs
- `9192-13287`: audio codebook IDs
- `13288-13387`: multimodal control tokens
- `13388-255999`: text tokens

SentencePiece is text-only. Raw SentencePiece IDs are never fed to the model.

```text
native_text_id = text_offset + sentencepiece_id
text_offset = 13388
```

## Canonical Files

- `taiji/tokenizer_contract.json`
- `taiji/tokenizer_native_v2.py`
- `scripts/build_native_vocab_corpus.py`
- `scripts/train_native_text_sp.py`
- `scripts/verify_native_tokenizer.py`

## Local Workflow

```bash
python scripts/build_native_vocab_corpus.py ^
  --data-dir taiji_data/training_data ^
  --data-dir taiji ^
  --output taiji_data/tokenizer/native_v2_corpus.txt

python scripts/train_native_text_sp.py ^
  --corpus taiji_data/tokenizer/native_v2_corpus.txt ^
  --output-dir taiji/tokenizer_native_v2 ^
  --contract taiji/tokenizer_contract.json

python scripts/verify_native_tokenizer.py ^
  --tokenizer-dir taiji/tokenizer_native_v2 ^
  --contract taiji/tokenizer_contract.json
```

## AutoDL / Linux Workflow

```bash
python scripts/build_native_vocab_corpus.py \
  --data-dir taiji_data/training_data \
  --data-dir taiji \
  --output taiji_data/tokenizer/native_v2_corpus.txt

python scripts/train_native_text_sp.py \
  --corpus taiji_data/tokenizer/native_v2_corpus.txt \
  --output-dir taiji/tokenizer_native_v2 \
  --contract taiji/tokenizer_contract.json

python scripts/verify_native_tokenizer.py \
  --tokenizer-dir taiji/tokenizer_native_v2 \
  --contract taiji/tokenizer_contract.json
```

For staged AutoDL runs, use `scripts/training/autodl_native_v2.sh`.

## Hard Rules

- do not put image or audio codebook IDs into SentencePiece
- do not use HF/Qwen tokenizers in native-v2 pretraining
- do not feed raw SentencePiece IDs to the model
- `model.vocab_size` must equal `contract.total_vocab_size`
- `sp.GetPieceSize()` must be less than or equal to `contract.text_vocab_size`
