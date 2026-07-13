# Taiji Architecture

## Overview

Taiji is a self-evolving agent-native AI system with four layers:

```
┌─────────────────────────────────────┐
│           Life System               │
│  Scheduler → Feed → Sleep → Play    │
├─────────────────────────────────────┤
│         Agent Layer                 │
│  Perception → Memory → Plan → React │
├─────────────────────────────────────┤
│        Model Layer                  │
│  ModelSelf + Heads + Tokenizer      │
├─────────────────────────────────────┤
│      Infrastructure                 │
│  Inference → Training → Safety      │
└─────────────────────────────────────┘
```

## Model Architecture

### ModelSelf

Custom transformer based on LLaMA 3 architecture:

- **Backbone**: Embedding + N × TransformerBlock + RMSNorm
- **Attention**: Grouped-Query Attention (GQA) with RoPE and FlashAttention fallback
- **FFN**: SwiGLU activation
- **Weight Tying**: `lm_head.weight = embedding.weight`

### Specialized Heads

| Head | Purpose | Output Size |
|------|---------|-------------|
| ToolHead | Tool classification + argument prediction | 750 tools, 4 arg slots |
| PerceptionHead | Environment state encoding | 200 env tokens |
| MemoryHead | Differentiable read/write memory | 30 slots × 64-dim |
| PlanHead | Task planning, difficulty estimation | 8 actions |

### Model Sizes

| Size | Hidden | Layers | Heads | KV Heads |
|------|--------|--------|-------|----------|
| 125M | 768 | 12 | 12 | 12 |
| 350M | 1024 | 24 | 16 | 16 |
| 1B | 2048 | 22 | 32 | 4 |
| 3B | 3072 | 28 | 32 | 4 |
| 7B | 4096 | 32 | 32 | 4 |

## Tokenizer

Native-v2 tokenizer with 256K ID space:

- **Text range**: SentencePiece model (shifted by `text_offset`)
- **Special tokens**: 30+ control tokens (think, tool_call, plan, reflect, etc.)
- **Tool tokens**: Dynamically allocated range (default 190-940)
- **Multimodal tokens**: Image, audio, video token ranges

See [NATIVE_V2_TOKENIZER.md](NATIVE_V2_TOKENIZER.md) for the full contract.

## Life System

The life system uses a need-driven scheduler:

- **5 needs**: hunger, fatigue, boredom, stress, curiosity (each 0-100)
- **Heartbeat**: every 60 seconds, needs grow with random jitter
- **Actions**: feed (knowledge ingestion), sleep (training/consolidation), play (exploration), explore (curiosity-driven)
- **Evolution**: growth metrics → `check_evolution_ready()` → `design_next_generation()`

## Training Pipeline

| Phase | Trainer | What it does |
|-------|---------|-------------|
| Pretrain | `ModelSelfTrainer.pretrain()` | Language modeling on text datasets |
| Finetune | `ModelSelfTrainer.finetune()` | ReAct tool-use training |
| DPO | `DPOTrainer` | Preference alignment with preference pairs |
| Distillation | `DistillationTrainer` | Teacher→student knowledge transfer for generation transition |
| Multimodal | `MultimodalTrainer` | Vision/audio projection layer training |
