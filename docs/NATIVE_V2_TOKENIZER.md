# Taiji Native v2 Tokenizer Contract

This is the canonical path for native Taiji pretraining. The old HF/Qwen
conversion path is legacy and must not be mixed into native-v2 runs.

## Global ID Space

`taiji/tokenizer_contract.json` owns the 256K vocabulary layout:

- `0-3`: fixed control tokens: `<pad>`, `<unk>`, `<s>`, `</s>`
- `4-999`: Taiji structural tokens
- `1000-9191`: image codebook IDs
- `9192-13287`: audio codebook IDs
- `13288-13387`: multimodal control tokens
- `13388-255999`: text tokens

SentencePiece is text-only. Its raw IDs are never fed to the model directly.
Native input IDs are computed as:

```text
native_text_id = 13388 + sentencepiece_id
```

## Main Files

- `taiji/tokenizer_contract.json`: fixed 256K layout
- `taiji/tokenizer_native_v2.py`: native wrapper enforcing the layout
- `scripts/build_native_vocab_corpus.py`: clean tokenizer corpus builder
- `scripts/train_native_text_sp.py`: text-only SentencePiece trainer
- `scripts/verify_native_tokenizer.py`: contract verifier

## Local Flow

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

## AutoDL Flow

Use `upload_to_autodl/setup_native_v2.sh`.

Do not use the old `setup_and_train.sh` for native-v2 training.

## Hard Rules

- Do not put image/audio codebook tokens into SentencePiece.
- Do not use Qwen/HF tokenizers in native-v2 pretraining.
- Do not feed raw SentencePiece IDs to the model.
- `model.vocab_size` must equal `contract.total_vocab_size`.
- `sp.GetPieceSize()` must be less than or equal to `contract.text_vocab_size`.
