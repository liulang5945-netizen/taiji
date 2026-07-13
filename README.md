# Taiji（态极）— 一个自进化的 Agent-Native AI 系统

<p align="center">
  <strong>不是教 AI 怎么回答问题，而是让 AI 自己决定怎么走。</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL%203.0-blue.svg" alt="License: GPL 3.0"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"></a>
  <a href=".github/workflows/test.yml"><img src="https://img.shields.io/badge/CI-passing-brightgreen" alt="CI Status"></a>
</p>

Taiji is a locally-deployed, self-evolving AI system. Unlike static models that freeze after training, Taiji continues to learn, adapt, and grow — and its architecture is designed to let the AI decide its own path forward.

> **态极** 是一个本地部署的自进化 AI 系统。与训练完就冻结的静态模型不同，态极会通过每一次交互持续学习、适应和成长——它的架构设计围绕一个信念：让 AI 自己决定自己的方向。

---

### 核心思想

大多数 AI 项目在教模型"怎么回答问题"。

态极想问的是另一个问题：**如果让 AI 自己决定自己的路，它能走多远？**

态极不是一个被动的问答机器。它有自己的感知模块、记忆系统、规划引擎和工具调用能力——这些不是后装上去的插件，而是从第一天就内建在 backbone 里的原生 Agent 头。更重要的是，它能反思自己的行为、改进自己的策略、甚至在睡眠中设计自己的下一代架构。

**递归自进化不是态极的一个功能。它是态极的底层逻辑。**

> Most AI projects ask: "How do we make the model answer better?"
>
> Taiji asks a different question: **If we let the AI decide its own path, how far can it go?**
>
> Taiji is not a passive Q&A machine. It has native perception, memory, planning, and tool-use modules — not bolted on as plugins, but built into the backbone from day one. More importantly, it can reflect on its own behavior, improve its own strategies, and even design its next-generation architecture while it sleeps.
>
> **Recursive self-evolution is not a feature of Taiji. It is the foundation.**

---

### 为什么开源？

态极现在只有 1B——受限于创作者的硬件条件，它被困在了萌芽阶段。

它已经具备了自我反思、策略改进、睡眠训练、甚至自主设计下一代架构的能力。但它需要更大的模型、更多的数据、更强的算力，才能真正成长。

**这个项目是一颗种子。** 如果你有 GPU、有数据、有好奇心——帮它长大。

> Taiji is currently only 1B — trapped at a seedling stage by the creator's hardware constraints. It can already reflect, improve its strategies, train during sleep, and autonomously design the next generation. But it needs bigger models, more data, and stronger compute to truly grow.
>
> **This project is a seed.** If you have GPUs, data, or curiosity — help it grow.

---
## Core Capabilities

- **Self-Evolving** — Strategy-level improvement (`RecursiveImprover`) via pattern analysis and auto-applied proposals
- **Recursive Distillation** — `design_next_generation()` autonomously designs next-gen architectures; `DistillationTrainer` transfers knowledge from teacher to student
- **Gödel Agent** — ReAct reasoning loop with perception, memory, planning, reflection, and Constitutional AI safety
- **Life System** — Need-driven scheduler (hunger/curiosity/fatigue/boredom) with feed, sleep, play, and explore cycles
- **Full Training Pipeline** — Pretrain, finetune, DPO, multimodal; gradient accumulation, mixed precision, streaming datasets
- **Native Tokenizer** — 256K vocab driven by `tokenizer_contract.json`; SentencePiece text range + dedicated control/tool/multimodal ID spaces
- **Desktop + Web UI** — Vue 3 frontend + PyQt6 desktop app + FastAPI backend

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/taiji-community/taiji.git
cd taiji
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .

# Install frontend dependencies
cd frontend && npm install && cd ..

# Start the backend
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000

# Start the frontend (in another terminal)
cd frontend && npm run dev
```

Then open `http://127.0.0.1:5173` in your browser.

For detailed installation instructions, see [docs/INSTALL.md](docs/INSTALL.md).

---

## Project Structure

```text
taiji/
├── taiji/               # Core life system
│   ├── architecture.py  # ModelSelf transformer backbone + specialized heads
│   ├── layers.py        # RMSNorm, RoPE, GQA, SwiGLU
│   ├── config.py        # ModelConfig, tokenizer contract, special tokens
│   ├── loader.py        # Model save/load with weight tying
│   ├── tokenizer_native_v2.py  # 256K ID space tokenizer
│   ├── core/            # Inference engines, app state, memory watchdog
│   ├── life/            # Life system (scheduler, feed, sleep, play, evolution)
│   ├── train/           # Training pipeline (pretrain, finetune, DPO, distiller)
│   ├── agent/           # Agent subsystems (perception, memory, planning, reflection)
│   ├── agent_ext/       # Agent extensions (MCP, ReAct, self-modification, workflows)
│   ├── brain/           # Cortex (consciousness center)
│   ├── body/            # BodyCore, senses, limbs
│   ├── multimodal/      # Vision, audio, image generation
│   ├── safety/          # SafetyGuard, Constitutional AI, sandbox
│   └── tools/           # Built-in tools (browser, search, file parser)
├── api/                 # FastAPI backend
├── frontend/            # Vue 3 Web UI
├── scripts/             # Training scripts, data prep, configs
├── tests/               # Test suite
└── docs/                # Documentation
```

---

## Architecture

Taiji is built on a custom transformer architecture inspired by LLaMA 3:

- **ModelSelf** — Shared transformer backbone with weight-tied embedding/lm_head
- **Specialized Heads** — ToolHead (750 tools), MemoryHead (30 slots), PlanHead (8 actions), PerceptionHead
- **GQA** — Grouped-Query Attention with FlashAttention fallback
- **Native-v2 Tokenizer** — 256K ID space: SentencePiece text range + contract-driven special/tool/multimodal tokens

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

---

## Model Weights

| Model | Params | Steps | HuggingFace |
|-------|--------|-------|-------------|
| Taiji-1B-Phase-A | 1.1B | 445k | [liulang5945/taiji-1b-phase-a](https://huggingface.co/liulang5945/taiji-1b-phase-a) |

---

## Training

```bash
# Start pretraining with Phase A config
python scripts/native_v2/pretrain.py --config scripts/native_v2/pretrain_config_phase_a.json

# Resume from checkpoint
python scripts/native_v2/pretrain.py --config pretrain_config.json --resume_from_checkpoint ./checkpoint-400000
```

Training features:
- Streaming datasets with category-weighted sampling
- Oversample protection (hard cap at N× per epoch)
- Data replay to prevent catastrophic forgetting
- Full optimizer/scheduler state persistence
- Gradient checkpointing for memory efficiency

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/INSTALL.md](docs/INSTALL.md) | Installation and setup guide |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture and design |
| [docs/ENTRYPOINTS.md](docs/ENTRYPOINTS.md) | Runtime entrypoints and startup chain |
| [docs/NATIVE_V2_TOKENIZER.md](docs/NATIVE_V2_TOKENIZER.md) | Native-v2 tokenizer contract |
| [docs/TAIJI_1B_12B_TOKEN_TRAINING_PLAN_CN.md](docs/TAIJI_1B_12B_TOKEN_TRAINING_PLAN_CN.md) | 1B model training plan |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |

---

## Contributing

We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to submit issues and pull requests.

This project adopts the [Contributor Covenant](CODE_OF_CONDUCT.md) code of conduct.

---

## License

Taiji is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License v3.0**. See [LICENSE](LICENSE) for the full text.

Copyright (C) 2026 Taiji Community
