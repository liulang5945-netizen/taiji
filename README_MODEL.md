---
language:
- zh
- en
license: gpl-3.0
tags:
- taiji
- llm
- self-evolving
- pytorch
- custom-transformer
- llama3-architecture
- 1b-parameters
- chinese
- gqa
- swiglu
- rope
datasets:
- custom-pretrain-mix
pipeline_tag: text-generation
---

# Taiji-1B-Phase-A

A 1.1B-parameter self-evolving language model from the [Taiji (态极)](https://github.com/liulang5945-netizen/taiji) project. Trained for 445,000 steps on a mixed Chinese-English corpus with category-weighted sampling and data replay.

## Model Description

Taiji-1B-Phase-A is the first generation model in the Taiji recursive evolution pipeline. Unlike static models that freeze after training, Taiji models are designed to continue learning through real-world use via strategy improvement, sleep fine-tuning, and recursive distillation.

## Architecture

| Parameter | Value |
|-----------|-------|
| Hidden Size | 2048 |
| Layers | 22 |
| Attention Heads | 32 |
| KV Heads | 4 (GQA) |
| Intermediate Size | 5504 |
| Max Position Embeddings | 4096 |
| Vocab Size | 256,000 |
| RMS Norm Epsilon | 1e-6 |
| RoPE Theta | 500,000 |

- **Backbone**: Custom transformer (LLaMA 3-style) with RMSNorm, RoPE, GQA, SwiGLU
- **Heads**: Language modeling + ToolHead (750 tools) + MemoryHead + PlanHead + PerceptionHead
- **Tokenizer**: Native-v2, 256K ID space with SentencePiece text range and dedicated control/tool/multimodal token spaces
- **Weight Tying**: Embedding and lm_head share weights

## Training

| Config | Value |
|--------|-------|
| Steps | 445,000 |
| Batch Size | 1 (effective 64 with gradient accumulation) |
| Max Sequence Length | 2048 |
| Learning Rate | Cosine schedule with warmup |
| Mixed Precision | AMP (bfloat16) |
| Data Replay | 30% from previous training mix |

## Quick Start

```python
from taiji.loader import load_model

model, tokenizer = load_model("liulang5945/taiji-1b-phase-a")

# Generate text
output = model.generate(tokenizer("你好，请介绍一下自己", return_tensors="pt")["input_ids"], max_new_tokens=128)
print(tokenizer.decode(output[0]))
```

Or download manually:
```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('liulang5945/taiji-1b-phase-a', local_dir='./checkpoint-445000')"
```

## Intended Use

- Research on self-evolving language models
- Chinese-English bilingual text generation
- Fine-tuning on domain-specific data
- Study of recursive distillation and model evolution

## Limitations

- 1B scale — not suitable for complex reasoning tasks that require larger models
- Trained on a custom data mix — may not cover all domains equally
- Tool-use capabilities require the full Taiji agent infrastructure
- No RLHF alignment — may produce unexpected outputs

## License

GNU General Public License v3.0

## Citation

```bibtex
@software{taiji2026,
  author = {Taiji Community},
  title = {Taiji: A Self-Evolving LLM Framework},
  year = {2026},
  url = {https://github.com/liulang5945-netizen/taiji}
}
```
