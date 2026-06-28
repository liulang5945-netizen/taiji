# Taiji Docs Index

`docs/` only keeps the current, canonical documents.

Read them in this order:

1. `TAIJI_1B_12B_TOKEN_TRAINING_PLAN_CN.md`
   - canonical 1B / 12B-token training plan
   - tokenizer, pretrain, SFT, multimodal staging
2. `1B_DATA_GAP_REPORT_CN.md`
   - current local data audit
   - token gaps and multimodal gap
3. `NATIVE_V2_TOKENIZER.md`
   - canonical native-v2 tokenizer contract
4. `ENTRYPOINTS.md`
   - runtime entrypoints and startup chain
5. `ARCHITECTURE.md`
   - high-level system structure
6. `INSTALL.md`
   - local install and run guide

Rules:

- deleted docs are not canonical and must not be revived
- do not mix legacy HF/Qwen pretrain routes into native-v2
- do not invent alternate training plans outside the canonical plan
- historical review, roadmap, and duplicate runbook docs were intentionally removed
