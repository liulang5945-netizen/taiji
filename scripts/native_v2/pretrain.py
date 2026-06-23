#!/usr/bin/env python3
"""Single-GPU native Taiji pretraining with tokenizer contract v2."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_MODEL_CONFIG = {
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "intermediate_size": 2816,
    "vocab_size": 256000,
    "max_position_embeddings": 2048,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
}

DEFAULT_TRAINING_CONFIG = {
    "data_dir": "taiji_data/training_data",
    "tokenizer_path": "taiji/tokenizer_native_v2/sentencepiece.model",
    "contract_path": "taiji/tokenizer_native_v2/tokenizer_contract.json",
    "output_dir": "taiji_data/taiji_pretrained",
    "max_steps": 50_000,
    "batch_size": 1,
    "gradient_accumulation_steps": 32,
    "learning_rate": 3e-4,
    "min_learning_rate": 3e-5,
    "warmup_steps": 1000,
    "max_length": 2048,
    "save_every": 5000,
    "log_every": 50,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "use_amp": True,
    "num_workers": 4,
    "seed": 42,
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 1_000_000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs, persistent=False)
        self._cache = {}

    def forward(self, seq_len: int, device, dtype):
        key = (seq_len, device, dtype)
        if key not in self._cache:
            pos = torch.arange(seq_len, device=device, dtype=torch.float32)
            angles = torch.outer(pos, self.freqs.to(device))
            self._cache[key] = (torch.sin(angles).to(dtype), torch.cos(angles).to(dtype))
        return self._cache[key]


def apply_rope(q: torch.Tensor, k: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
    q_r, q_i = q[..., ::2], q[..., 1::2]
    k_r, k_i = k[..., ::2], k[..., 1::2]
    sin = sin.unsqueeze(0).unsqueeze(2)
    cos = cos.unsqueeze(0).unsqueeze(2)
    q_out = torch.stack((q_r * cos - q_i * sin, q_r * sin + q_i * cos), dim=-1).flatten(-2)
    k_out = torch.stack((k_r * cos - k_i * sin, k_r * sin + k_i * cos), dim=-1).flatten(-2)
    return q_out.type_as(q), k_out.type_as(k)


class Attention(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, theta: float):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = hidden // heads
        self.repeat = heads // kv_heads
        self.wq = nn.Linear(hidden, heads * self.head_dim, bias=False)
        self.wk = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(heads * self.head_dim, hidden, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.wq(x).view(bsz, seqlen, self.heads, self.head_dim)
        k = self.wk(x).view(bsz, seqlen, self.kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seqlen, self.kv_heads, self.head_dim)
        sin, cos = self.rope(seqlen, x.device, x.dtype)
        q, k = apply_rope(q, k, sin, cos)
        if self.repeat > 1:
            k = k.repeat_interleave(self.repeat, dim=2)
            v = v.repeat_interleave(self.repeat, dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).contiguous().view(bsz, seqlen, -1))


class Block(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.attn_norm = RMSNorm(cfg["hidden_size"], cfg.get("rms_norm_eps", 1e-6))
        self.attn = Attention(
            cfg["hidden_size"],
            cfg["num_attention_heads"],
            cfg["num_key_value_heads"],
            cfg.get("rope_theta", 1_000_000.0),
        )
        self.ffn_norm = RMSNorm(cfg["hidden_size"], cfg.get("rms_norm_eps", 1e-6))
        self.w1 = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], bias=False)
        self.wg = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], bias=False)
        self.w2 = nn.Linear(cfg["intermediate_size"], cfg["hidden_size"], bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        h = self.ffn_norm(x)
        x = x + self.w2(F.silu(self.wg(h)) * self.w1(h))
        return x


class TaijiBackbone(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg["num_hidden_layers"])])
        self.norm = RMSNorm(cfg["hidden_size"], cfg.get("rms_norm_eps", 1e-6))
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        self.lm_head.weight = self.embed.weight
        self.apply(self._init)

    def _init(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        logits = self.lm_head(self.norm(h))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}


class NativeDataset(Dataset):
    def __init__(self, data_dir: str, sp, contract: dict, max_length: int):
        self.sp = sp
        self.text_offset = int(contract["text_offset"])
        self.pad_id = int(contract["special_tokens"]["<pad>"])
        self.max_length = max_length
        self.lines = []
        for pattern in ("**/*.jsonl", "**/*.json", "**/*.txt"):
            for path in glob.glob(os.path.join(data_dir, pattern), recursive=True):
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            line = line.strip()
                            if len(line) > 40:
                                self.lines.append(line)
                except OSError:
                    pass
        random.shuffle(self.lines)
        print(f"dataset_lines={len(self.lines)}")

    def __len__(self):
        return len(self.lines)

    def _extract(self, line: str) -> str:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return line
        if isinstance(obj, dict):
            texts = []
            for key in ("text", "content", "output", "instruction", "input", "question", "answer"):
                if isinstance(obj.get(key), str):
                    texts.append(obj[key])
            if isinstance(obj.get("messages"), list):
                texts.extend(m.get("content", "") for m in obj["messages"] if isinstance(m, dict))
            return " ".join(t for t in texts if t)
        return line

    def __getitem__(self, idx: int):
        text = self._extract(self.lines[idx])
        ids = [self.text_offset + i for i in self.sp.EncodeAsIds(text)]
        ids = ids[: self.max_length]
        ids += [self.pad_id] * (self.max_length - len(ids))
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.pad_id] = -100
        return {"input_ids": input_ids, "labels": labels}


def lr_at(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * max(1, step) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def save_checkpoint(model, cfg, contract, tokenizer_path, path, step, loss):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    (path / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (path / "tokenizer_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    import shutil

    shutil.copy2(tokenizer_path, path / "sentencepiece.model")
    print(f"saved={path} step={step} loss={loss:.4f}")


def train_config(cfg_all: dict[str, Any]) -> None:
    model_cfg = cfg_all["model"]
    train_cfg = cfg_all["training"]
    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    contract = json.loads(Path(train_cfg["contract_path"]).read_text(encoding="utf-8"))
    assert model_cfg["vocab_size"] == contract["total_vocab_size"]

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(train_cfg["tokenizer_path"])
    assert sp.GetPieceSize() <= contract["text_vocab_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TaijiBackbone(model_cfg).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"device={device} params={params:.1f}M sp_vocab={sp.GetPieceSize()}")

    dataset = NativeDataset(train_cfg["data_dir"], sp, contract, train_cfg["max_length"])
    if len(dataset) == 0:
        raise RuntimeError(f"No usable training lines found in data_dir={train_cfg['data_dir']}")
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 2),
        pin_memory=True,
        drop_last=True,
    )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.1),
        betas=(0.9, 0.95),
    )
    use_amp = torch.cuda.is_available() and train_cfg.get("use_amp", True)
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda") if use_amp and amp_dtype == torch.float16 else None

    max_steps = train_cfg["max_steps"]
    accum = train_cfg["gradient_accumulation_steps"]
    output = Path(train_cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    step = 0
    micro = 0
    running = 0.0
    last_loss = 0.0
    start = time.time()
    model.train()
    opt.zero_grad(set_to_none=True)

    while step < max_steps:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    loss = model(input_ids, labels)["loss"] / accum
            else:
                loss = model(input_ids, labels)["loss"] / accum
            last_loss = loss.item() * accum
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += loss.item() * accum
            micro += 1
            if micro % accum != 0:
                continue

            lr = lr_at(step, train_cfg["warmup_steps"], max_steps, train_cfg["learning_rate"], train_cfg["min_learning_rate"])
            for group in opt.param_groups:
                group["lr"] = lr
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("grad_clip", 1.0))
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % train_cfg["log_every"] == 0:
                elapsed = max(1e-6, time.time() - start)
                print(f"step={step}/{max_steps} loss={running / train_cfg['log_every']:.4f} lr={lr:.2e} hours={elapsed/3600:.2f}")
                running = 0.0
            if step % train_cfg["save_every"] == 0:
                save_checkpoint(model, model_cfg, contract, train_cfg["tokenizer_path"], output / f"checkpoint-{step}", step, last_loss)
            if step >= max_steps:
                break
    save_checkpoint(model, model_cfg, contract, train_cfg["tokenizer_path"], output / "final", step, last_loss)


def train(config_path: str) -> None:
    cfg_all = json.loads(Path(config_path).read_text(encoding="utf-8"))
    train_config(cfg_all)


def build_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.tokenizer_dir:
        tokenizer_path = str(Path(args.tokenizer_dir) / "sentencepiece.model")
        contract_path = str(Path(args.tokenizer_dir) / "tokenizer_contract.json")
    else:
        tokenizer_path = DEFAULT_TRAINING_CONFIG["tokenizer_path"]
        contract_path = DEFAULT_TRAINING_CONFIG["contract_path"]

    if args.tokenizer_path:
        tokenizer_path = args.tokenizer_path
    if args.contract:
        contract_path = args.contract

    model_cfg = dict(DEFAULT_MODEL_CONFIG)
    train_cfg = dict(DEFAULT_TRAINING_CONFIG)
    train_cfg.update(
        {
            "data_dir": args.data_dir or train_cfg["data_dir"],
            "tokenizer_path": tokenizer_path,
            "contract_path": contract_path,
            "output_dir": args.output or train_cfg["output_dir"],
        }
    )

    for key in (
        "max_steps",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "min_learning_rate",
        "warmup_steps",
        "max_length",
        "save_every",
        "log_every",
        "num_workers",
        "seed",
    ):
        value = getattr(args, key)
        if value is not None:
            train_cfg[key] = value

    return {"model": model_cfg, "training": train_cfg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--tokenizer-dir", dest="tokenizer_dir", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--contract", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--min_learning_rate", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.config:
        train(args.config)
    else:
        train_config(build_config_from_args(args))
