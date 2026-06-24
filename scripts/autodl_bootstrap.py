#!/usr/bin/env python3
"""
态极 1B Native-V2 AUTODL 一键部署脚本

在 AUTODL 终端中执行:
    python3 /root/autodl-tmp/autodl_bootstrap.py

此脚本会自动:
1. 创建项目目录结构和所有训练脚本
2. 安装依赖
3. 下载预训练数据
4. 重建 Native-V2 Tokenizer
5. 运行 Smoke Run (200 步)
6. (可选) 启动正式 1B Stage1 训练
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ============================================================
# 配置
# ============================================================
PROJECT_DIR = Path("/root/autodl-tmp/taiji")
DATA_DIR = PROJECT_DIR / "taiji_data" / "training_data" / "pretrain_mix_v1"
TOKENIZER_DIR = PROJECT_DIR / "taiji" / "tokenizer_native_v2"
CORPUS_FILE = PROJECT_DIR / "taiji_data" / "tokenizer" / "native_v2_corpus.txt"
SMOKE_OUTPUT = PROJECT_DIR / "taiji_data" / "taiji_pretrained_1b_smoke"
TRAIN_OUTPUT = PROJECT_DIR / "taiji_data" / "taiji_pretrained_1b_stage1"
CONTRACT_FILE = PROJECT_DIR / "taiji" / "tokenizer_contract.json"


# ============================================================
# tokenizer_contract.json 内容
# ============================================================
TOKENIZER_CONTRACT = {
    "name": "taiji-native-v2",
    "version": 2,
    "total_vocab_size": 256000,
    "text_offset": 13388,
    "text_vocab_size": 242612,
    "ranges": {
        "control": [0, 3],
        "taiji_special": [4, 999],
        "image": [1000, 9191],
        "audio": [9192, 13287],
        "multimodal_control": [13288, 13387],
        "text": [13388, 255999]
    },
    "special_tokens": {
        "<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3,
        "<think>": 10, "</think>": 11,
        "<inner_voice>": 12, "</inner_voice>": 13,
        "<reflect>": 14, "</reflect>": 15,
        "<tool_call>": 20, "</tool_call>": 21,
        "<tool_result>": 22, "</tool_result>": 23,
        "<tool_name>": 26, "</tool_name>": 27,
        "<tool_arg>": 28, "</tool_arg>": 29,
        "<final_answer>": 30,
        "<system>": 180, "</system>": 181,
        "<user>": 182, "</user>": 183,
        "<assistant>": 184, "</assistant>": 185,
    },
    "multimodal": {
        "image": {"base": 1000, "codebook_size": 8192},
        "audio": {"base": 9192, "codebook_size": 4096},
        "control_tokens": {
            "<mm_image>": 13288, "</mm_image>": 13289,
            "<mm_audio>": 13290, "</mm_audio>": 13291,
            "<mm_video>": 13292, "</mm_video>": 13293,
        }
    }
}


# ============================================================
# 预训练数据源定义
# ============================================================
SOURCES = {
    "fineweb_edu": {
        "repo_id": "HuggingFaceFW/fineweb-edu",
        "file_prefix": "data/",
        "suffix": ".parquet",
        "text_field": "text",
        "file_format": "parquet",
        "category": "general_web",
    },
    "skypile_zh": {
        "repo_id": "Skywork/SkyPile-150B",
        "file_prefix": "data/",
        "suffix": ".jsonl",
        "text_field": "text",
        "file_format": "jsonl",
        "category": "chinese_web",
    },
    "openwebmath": {
        "repo_id": "open-web-math/open-web-math",
        "file_prefix": "data/",
        "suffix": ".parquet",
        "text_field": "text",
        "file_format": "parquet",
        "category": "math",
    },
    "codeparrot_code": {
        "repo_id": "codeparrot/codeparrot-clean",
        "file_prefix": "file-",
        "suffix": ".json.gz",
        "text_field": "content",
        "file_format": "json.gz",
        "category": "code",
    },
}

DEFAULT_SOURCES = ["fineweb_edu", "skypile_zh", "openwebmath", "codeparrot_code"]


# ============================================================
# 工具函数
# ============================================================
def banner(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")


def run_cmd(cmd: list[str], cwd: str | None = None, check: bool = True) -> int:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or str(PROJECT_DIR))
    if check and result.returncode != 0:
        print(f"  ❌ 命令失败 (exit={result.returncode})")
        sys.exit(1)
    return result.returncode


def install_deps() -> None:
    banner("Step 1: 安装依赖")
    run_cmd([sys.executable, "-m", "pip", "install", "-q",
             "sentencepiece", "huggingface_hub", "pyarrow", "tqdm", "numpy", "tensorboard"])
    print("✅ 依赖安装完成\n")


def download_data(max_records: int = 100_000) -> None:
    banner("Step 2: 下载预训练数据")

    if DATA_DIR.exists():
        jsonl_files = list(DATA_DIR.glob("*.jsonl"))
        if len(jsonl_files) >= 4:
            print(f"数据已存在 ({len(jsonl_files)} 个 JSONL 文件)，跳过下载。")
            for f in jsonl_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.name}: {size_mb:.1f} MB")
            return

    # 在 Python 内部设置 HuggingFace 镜像（确保生效）
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"  HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '(默认)')}")

    # 快速测试 HuggingFace 连通性
    print("  测试 HuggingFace 连通性...")
    import urllib.request
    hf_reachable = False
    for endpoint in ["https://hf-mirror.com", "https://huggingface.co"]:
        try:
            req = urllib.request.Request(endpoint, method="HEAD")
            req.add_header("User-Agent", "taiji-bootstrap/1.0")
            resp = urllib.request.urlopen(req, timeout=10)
            print(f"  ✅ {endpoint} 可达 (status={resp.status})")
            os.environ["HF_ENDPOINT"] = endpoint
            hf_reachable = True
            break
        except Exception as e:
            print(f"  ❌ {endpoint} 不可达: {type(e).__name__}")

    if not hf_reachable:
        print("  ⚠️  HuggingFace 完全不可达，直接使用示例数据")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for source_name in DEFAULT_SOURCES:
            output_path = DATA_DIR / f"{source_name}.jsonl"
            if not output_path.exists():
                _create_sample_data(source_name, output_path, max_records)
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"  {source_name}: {size_mb:.1f} MB (示例数据)")
        return

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("❌ huggingface_hub 未安装")
        sys.exit(1)

    import gzip
    try:
        import pyarrow.parquet as pq
    except ImportError:
        pq = None

    raw_dir = PROJECT_DIR / "taiji_data" / "training_data" / "raw_pretrain_mix_v1"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()

    for source_name in DEFAULT_SOURCES:
        source = SOURCES[source_name]
        output_path = DATA_DIR / f"{source_name}.jsonl"

        if output_path.exists():
            print(f"  {source_name}: 已存在，跳过")
            continue

        print(f"  下载 {source_name} ...")

        try:
            files = api.list_repo_files(source["repo_id"], repo_type="dataset")
            matched = [f for f in files
                       if f.startswith(source["file_prefix"]) and f.endswith(source["suffix"])]
            matched.sort()
            if not matched:
                print(f"  ⚠️  {source_name}: 没有匹配的文件")
                continue

            target_dir = raw_dir / source_name
            target_dir.mkdir(parents=True, exist_ok=True)

            downloaded = []
            for remote_file in matched[:1]:  # 只下载第一个 shard
                try:
                    local_path = hf_hub_download(
                        repo_id=source["repo_id"],
                        filename=remote_file,
                        repo_type="dataset",
                        local_dir=str(target_dir),
                    )
                    downloaded.append(Path(local_path))
                except Exception as e:
                    print(f"  ⚠️  下载失败 {remote_file}: {e}")

            if not downloaded:
                print(f"  ⚠️  {source_name}: 下载失败")
                continue

            # 标准化
            written = 0
            with output_path.open("w", encoding="utf-8", newline="\n") as out:
                for dl_path in downloaded:
                    try:
                        if source["file_format"] == "parquet" and pq:
                            pf = pq.ParquetFile(dl_path)
                            for batch in pf.iter_batches():
                                for row in batch.to_pylist():
                                    if isinstance(row, dict) and isinstance(row.get(source["text_field"]), str):
                                        text = row[source["text_field"]].strip()
                                        if len(text) >= 32:
                                            record = {"text": text, "source": source_name, "category": source["category"]}
                                            out.write(json.dumps(record, ensure_ascii=False) + "\n")
                                            written += 1
                                            if written >= max_records:
                                                break
                                if written >= max_records:
                                    break
                        elif source["file_format"] == "jsonl":
                            with dl_path.open("r", encoding="utf-8", errors="ignore") as f:
                                for line in f:
                                    try:
                                        obj = json.loads(line.strip())
                                        if isinstance(obj, dict) and isinstance(obj.get(source["text_field"]), str):
                                            text = obj[source["text_field"]].strip()
                                            if len(text) >= 32:
                                                record = {"text": text, "source": source_name, "category": source["category"]}
                                                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                                                written += 1
                                                if written >= max_records:
                                                    break
                                    except json.JSONDecodeError:
                                        pass
                        elif source["file_format"] == "json.gz":
                            import gzip as gz
                            with gz.open(dl_path, "rt", encoding="utf-8", errors="ignore") as f:
                                for line in f:
                                    try:
                                        obj = json.loads(line.strip())
                                        if isinstance(obj, dict) and isinstance(obj.get(source["text_field"]), str):
                                            text = obj[source["text_field"]].strip()
                                            if len(text) >= 32:
                                                record = {"text": text, "source": source_name, "category": source["category"]}
                                                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                                                written += 1
                                                if written >= max_records:
                                                    break
                                    except json.JSONDecodeError:
                                        pass
                    except Exception as e:
                        print(f"  ⚠️  处理失败 {dl_path}: {e}")

            print(f"  {source_name}: {written} 条记录")

        except Exception as e:
            print(f"  ❌ {source_name} 失败: {e}")
            # 创建示例数据作为 fallback
            print(f"  创建示例数据 {source_name} ...")
            _create_sample_data(source_name, output_path, max_records)


def _create_sample_data(source_name: str, output_path: Path, max_records: int) -> None:
    """创建示例数据作为 fallback（当 HuggingFace 下载失败时）"""
    samples = {
        "fineweb_edu": [
            "The transformer architecture has revolutionized natural language processing by introducing the self-attention mechanism.",
            "Machine learning models require large datasets for training to achieve good generalization performance.",
            "Deep learning has enabled significant advances in computer vision, natural language processing, and speech recognition.",
            "The attention mechanism allows models to focus on relevant parts of the input sequence when generating output.",
            "Neural networks with more parameters can represent more complex functions but require more data to train effectively.",
        ],
        "skypile_zh": [
            "人工智能技术正在改变我们的生活方式，从智能语音助手到自动驾驶汽车。",
            "深度学习是机器学习的一个分支，它试图模拟人脑的神经网络结构来处理复杂的数据。",
            "自然语言处理是人工智能领域的一个重要研究方向，旨在让计算机理解和生成人类语言。",
            "大数据技术的发展为人工智能的训练提供了丰富的数据资源，推动了AI技术的快速发展。",
            "机器学习算法可以从大量数据中自动学习规律，并用于预测和决策。",
        ],
        "openwebmath": [
            "The fundamental theorem of calculus states that differentiation and integration are inverse operations.",
            "Linear algebra provides the mathematical foundation for many machine learning algorithms including neural networks.",
            "The gradient descent optimization algorithm iteratively adjusts parameters to minimize a loss function.",
            "Probability theory and statistics are essential tools for understanding uncertainty in data and models.",
            "The chain rule of calculus is fundamental to backpropagation in neural networks.",
        ],
        "codeparrot_code": [
            "def train_model(model, dataloader, optimizer, num_epochs):\n    for epoch in range(num_epochs):\n        for batch in dataloader:\n            loss = model(batch)\n            loss.backward()\n            optimizer.step()\n            optimizer.zero_grad()",
            "import torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n    def __init__(self, hidden_size, num_heads):\n        super().__init__()\n        self.attention = nn.MultiheadAttention(hidden_size, num_heads)\n        self.norm = nn.LayerNorm(hidden_size)\n",
            "class DataLoader:\n    def __init__(self, dataset, batch_size, shuffle=True):\n        self.dataset = dataset\n        self.batch_size = batch_size\n        self.shuffle = shuffle\n\n    def __iter__(self):\n        indices = list(range(len(self.dataset)))\n        if self.shuffle:\n            random.shuffle(indices)\n",
            "async def fetch_data(url: str) -> dict:\n    import aiohttp\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            return await response.json()",
            "from typing import List, Optional\n\ndef tokenize(text: str, vocab: dict, max_length: int = 512) -> List[int]:\n    tokens = []\n    for char in text:\n        if char in vocab:\n            tokens.append(vocab[char])\n    return tokens[:max_length]",
        ],
    }

    lines = samples.get(source_name, samples["fineweb_edu"])
    with output_path.open("w", encoding="utf-8", newline="\n") as out:
        for _ in range(max_records // len(lines) + 1):
            for line in lines:
                record = {"text": line, "source": source_name, "category": "sample"}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  {source_name}: 已创建示例数据")


def build_tokenizer() -> None:
    banner("Step 3: 重建 Native-V2 Tokenizer")

    if (TOKENIZER_DIR / "sentencepiece.model").exists():
        print("Tokenizer 已存在，跳过重建。")
        return

    CONTRACT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_FILE.write_text(json.dumps(TOKENIZER_CONTRACT, indent=2, ensure_ascii=False), encoding="utf-8")

    # 3.1 构建语料
    print("Step 3.1: 构建词表训练语料...")
    _build_corpus()

    # 3.2 训练 SentencePiece
    print("\nStep 3.2: 训练 SentencePiece 词表...")
    _train_sentencepiece()

    # 3.3 验证
    print("\nStep 3.3: 验证 Tokenizer...")
    _verify_tokenizer()


def _build_corpus() -> None:
    """构建词表训练语料"""
    import sentencepiece as spm

    CORPUS_FILE.parent.mkdir(parents=True, exist_ok=True)

    TAIJI_SEED_LINES = [
        "态极是一个本地运行的 AI 生命体。",
        "态极通过感知、工具调用、记忆、规划和反思来完成任务。",
        "<think>我需要分析目标、选择工具、验证结果。</think>",
        '<tool_call>{"name":"search","args":{"query":"人工智能新闻"}}</tool_call>',
        "<tool_result>工具返回了可验证的信息。</tool_result>",
        "Python 函数、类、模块、异常处理、异步编程、装饰器、类型标注。",
        "Transformer 使用注意力机制、RMSNorm、RoPE、GQA 和 SwiGLU。",
        "The native Taiji tokenizer separates control tokens, multimodal tokens, and text tokens.",
    ]

    TEXT_KEYS = ("text", "content", "output", "instruction", "input", "question", "answer")
    total_chars = 0
    max_chars = 1_000_000_000  # 1B chars max

    with CORPUS_FILE.open("w", encoding="utf-8", newline="\n") as out:
        for line in TAIJI_SEED_LINES:
            out.write(line + "\n")
            total_chars += len(line)

        # 从预训练数据中提取文本
        for jsonl_path in sorted(DATA_DIR.glob("*.jsonl")):
            if total_chars >= max_chars:
                break
            try:
                with jsonl_path.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if total_chars >= max_chars:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                for key in TEXT_KEYS:
                                    val = obj.get(key)
                                    if isinstance(val, str) and len(val) > 2:
                                        text = " ".join(val.split())
                                        out.write(text + "\n")
                                        total_chars += len(text)
                        except json.JSONDecodeError:
                            text = " ".join(line.split())
                            if len(text) > 2:
                                out.write(text + "\n")
                                total_chars += len(text)
            except OSError:
                pass

    print(f"  语料文件: {CORPUS_FILE}")
    print(f"  总字符数: {total_chars:,}")


def _train_sentencepiece() -> None:
    """训练 SentencePiece 词表"""
    import sentencepiece as spm

    contract = TOKENIZER_CONTRACT
    text_vocab_size = int(contract["text_vocab_size"])

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    model_prefix = str(TOKENIZER_DIR / "sentencepiece")

    spm.SentencePieceTrainer.train(
        input=str(CORPUS_FILE),
        model_prefix=model_prefix,
        vocab_size=text_vocab_size,
        model_type="bpe",
        character_coverage=0.9999,
        byte_fallback=True,
        normalization_rule_name="identity",
        add_dummy_prefix=True,
        remove_extra_whitespaces=False,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
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

    # 复制 contract 到 tokenizer 目录
    (TOKENIZER_DIR / "tokenizer_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  模型: {TOKENIZER_DIR / 'sentencepiece.model'}")


def _verify_tokenizer() -> None:
    """验证 Tokenizer"""
    import sentencepiece as spm

    contract = TOKENIZER_CONTRACT
    sp = spm.SentencePieceProcessor()
    sp.Load(str(TOKENIZER_DIR / "sentencepiece.model"))

    piece_size = sp.GetPieceSize()
    text_vocab_size = contract["text_vocab_size"]
    text_offset = contract["text_offset"]
    total_vocab_size = contract["total_vocab_size"]

    print(f"  sp.GetPieceSize() = {piece_size}")
    print(f"  text_vocab_size   = {text_vocab_size}")
    print(f"  text_offset       = {text_offset}")
    print(f"  total_vocab_size  = {total_vocab_size}")

    ok = True
    if piece_size > text_vocab_size:
        print(f"  ❌ piece_size ({piece_size}) > text_vocab_size ({text_vocab_size})")
        ok = False
    else:
        print(f"  ✅ piece_size <= text_vocab_size")

    if ok:
        print("  ✅ Tokenizer 验证通过!")
    else:
        print("  ❌ Tokenizer 验证失败!")
        sys.exit(1)


def run_smoke() -> None:
    banner("Step 4: Smoke Run (200 步)")

    # GPU 检查
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print("  ⚠️  未检测到 GPU，将使用 CPU 训练（会很慢）")
    except Exception:
        pass

    _run_pretrain(
        output_dir=str(SMOKE_OUTPUT),
        max_steps=200,
        batch_size=1,
        gradient_accumulation_steps=8,
        max_length=1024,
        learning_rate=3e-4,
        min_learning_rate=1e-4,
        warmup_steps=20,
        log_every=10,
        save_every=100,
    )

    print("\n✅ Smoke Run 完成!")
    print("请检查:")
    print("  1. loss 是否从 ~11 下降到 ~8-10")
    print("  2. checkpoint 是否正常保存")
    print("  3. 如正常，执行正式训练:")
    print(f"     python3 {__file__} --stage train")


def run_train() -> None:
    banner("Step 5: 正式 1B Stage1 预训练")

    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        max_length = 2048 if gpu_mem >= 20 else 1024
    else:
        max_length = 512

    print(f"  max_length: {max_length}")
    print(f"  预计时间: 3-7 天")
    print(f"  日志: taiji_data/train_stage1.log")
    print()

    log_file = PROJECT_DIR / "taiji_data" / "train_stage1.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    _run_pretrain(
        output_dir=str(TRAIN_OUTPUT),
        max_steps=50000,
        batch_size=1,
        gradient_accumulation_steps=32,
        max_length=max_length,
        learning_rate=3e-4,
        min_learning_rate=3e-5,
        warmup_steps=1000,
        log_every=50,
        save_every=10000,
        log_file=str(log_file),
    )


def _run_pretrain(
    output_dir: str,
    max_steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_length: int,
    learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    log_every: int,
    save_every: int,
    log_file: str | None = None,
) -> None:
    """运行预训练（内联实现，不依赖外部脚本文件）"""

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    # ---- 模型定义 ----
    MODEL_CONFIG = {
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

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            return self.weight * (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps))

    class RotaryEmbedding(nn.Module):
        def __init__(self, dim: int, theta: float = 1_000_000.0):
            super().__init__()
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("freqs", freqs, persistent=False)
            self._cache = {}
        def forward(self, seq_len, device, dtype):
            key = (seq_len, device, dtype)
            if key not in self._cache:
                pos = torch.arange(seq_len, device=device, dtype=torch.float32)
                angles = torch.outer(pos, self.freqs.to(device))
                self._cache[key] = (torch.sin(angles).to(dtype), torch.cos(angles).to(dtype))
            return self._cache[key]

    def apply_rope(q, k, sin, cos):
        q_r, q_i = q[..., ::2], q[..., 1::2]
        k_r, k_i = k[..., ::2], k[..., 1::2]
        sin = sin.unsqueeze(0).unsqueeze(2)
        cos = cos.unsqueeze(0).unsqueeze(2)
        q_out = torch.stack((q_r * cos - q_i * sin, q_r * sin + q_i * cos), dim=-1).flatten(-2)
        k_out = torch.stack((k_r * cos - k_i * sin, k_r * sin + k_i * cos), dim=-1).flatten(-2)
        return q_out.type_as(q), k_out.type_as(k)

    class Attention(nn.Module):
        def __init__(self, hidden, heads, kv_heads, theta):
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
        def forward(self, x):
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
        def __init__(self, cfg):
            super().__init__()
            self.attn_norm = RMSNorm(cfg["hidden_size"], cfg.get("rms_norm_eps", 1e-6))
            self.attn = Attention(cfg["hidden_size"], cfg["num_attention_heads"],
                                  cfg["num_key_value_heads"], cfg.get("rope_theta", 1_000_000.0))
            self.ffn_norm = RMSNorm(cfg["hidden_size"], cfg.get("rms_norm_eps", 1e-6))
            self.w1 = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], bias=False)
            self.wg = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], bias=False)
            self.w2 = nn.Linear(cfg["intermediate_size"], cfg["hidden_size"], bias=False)
        def forward(self, x):
            x = x + self.attn(self.attn_norm(x))
            h = self.ffn_norm(x)
            x = x + self.w2(F.silu(self.wg(h)) * self.w1(h))
            return x

    class TaijiBackbone(nn.Module):
        def __init__(self, cfg):
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
        def forward(self, input_ids, labels=None):
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

    # ---- 数据集 ----
    _SKIP_DIR_NAMES = {".cache", "__pycache__", ".git", "node_modules"}
    _SKIP_FILE_NAMES = {"manifest.json"}
    _SKIP_FILE_SUFFIXES = {".lock", ".metadata"}

    def _should_skip(path: str) -> bool:
        p = Path(path)
        if any(part in _SKIP_DIR_NAMES for part in p.parts):
            return True
        if p.name in _SKIP_FILE_NAMES:
            return True
        return any(p.name.endswith(s) for s in _SKIP_FILE_SUFFIXES)

    class NativeDataset(Dataset):
        def __init__(self, data_dir, sp, contract, max_length):
            self.sp = sp
            self.text_offset = int(contract["text_offset"])
            self.pad_id = int(contract["special_tokens"]["<pad>"])
            self.max_length = max_length
            self.lines = []
            for pattern in ("**/*.jsonl", "**/*.json", "**/*.txt"):
                for path in glob.glob(os.path.join(data_dir, pattern), recursive=True):
                    if _should_skip(path):
                        continue
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            for line in f:
                                line = line.strip()
                                if len(line) > 40:
                                    self.lines.append(line)
                    except OSError:
                        pass
            random.shuffle(self.lines)
            print(f"  dataset_lines={len(self.lines)}")

        def __len__(self):
            return len(self.lines)

        def _extract(self, line):
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

        def __getitem__(self, idx):
            text = self._extract(self.lines[idx])
            ids = [self.text_offset + i for i in self.sp.EncodeAsIds(text)]
            ids = ids[:self.max_length]
            ids += [self.pad_id] * (self.max_length - len(ids))
            input_ids = torch.tensor(ids, dtype=torch.long)
            labels = input_ids.clone()
            labels[labels == self.pad_id] = -100
            return {"input_ids": input_ids, "labels": labels}

    # ---- 学习率调度 ----
    def lr_at(step, warmup, total_steps, max_lr, min_lr):
        if step < warmup:
            return max_lr * max(1, step) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

    # ---- 保存 checkpoint ----
    def save_checkpoint(model, cfg, contract, tok_path, path, step, loss):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path / "model.pt")
        (path / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        (path / "tokenizer_contract.json").write_text(
            json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
        shutil.copy2(tok_path, path / "sentencepiece.model")
        print(f"  saved={path} step={step} loss={loss:.4f}")

    # ---- 训练主循环 ----
    import sentencepiece as spm

    contract = TOKENIZER_CONTRACT
    sp = spm.SentencePieceProcessor()
    sp.Load(str(TOKENIZER_DIR / "sentencepiece.model"))

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TaijiBackbone(MODEL_CONFIG).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  device={device} params={params:.1f}M sp_vocab={sp.GetPieceSize()}")

    dataset = NativeDataset(str(DATA_DIR), sp, contract, max_length)
    if len(dataset) == 0:
        print("❌ 没有找到训练数据!")
        sys.exit(1)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1, betas=(0.9, 0.95))
    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda") if use_amp and amp_dtype == torch.float16 else None

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    step = 0
    micro = 0
    running = 0.0
    last_loss = 0.0
    start = time.time()
    model.train()
    opt.zero_grad(set_to_none=True)

    # 重定向日志
    if log_file:
        import io
        log_fh = open(log_file, "a", encoding="utf-8")
        original_stdout = sys.stdout

        class TeeWriter:
            def write(self, text):
                original_stdout.write(text)
                log_fh.write(text)
                log_fh.flush()
            def flush(self):
                original_stdout.flush()
                log_fh.flush()

        sys.stdout = TeeWriter()

    print(f"  开始训练: max_steps={max_steps} batch_size={batch_size} accum={gradient_accumulation_steps}")
    print(f"  max_length={max_length} lr={learning_rate} device={device}")

    while step < max_steps:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    loss = model(input_ids, labels)["loss"] / gradient_accumulation_steps
            else:
                loss = model(input_ids, labels)["loss"] / gradient_accumulation_steps
            last_loss = loss.item() * gradient_accumulation_steps
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += loss.item() * gradient_accumulation_steps
            micro += 1
            if micro % gradient_accumulation_steps != 0:
                continue

            lr = lr_at(step, warmup_steps, max_steps, learning_rate, min_learning_rate)
            for group in opt.param_groups:
                group["lr"] = lr
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % log_every == 0:
                elapsed = max(1e-6, time.time() - start)
                print(f"step={step}/{max_steps} loss={running / log_every:.4f} lr={lr:.2e} hours={elapsed/3600:.2f}")
                running = 0.0
            if step % save_every == 0:
                save_checkpoint(model, MODEL_CONFIG, contract,
                               str(TOKENIZER_DIR / "sentencepiece.model"),
                               output / f"checkpoint-{step}", step, last_loss)
            if step >= max_steps:
                break

    save_checkpoint(model, MODEL_CONFIG, contract,
                   str(TOKENIZER_DIR / "sentencepiece.model"),
                   output / "final", step, last_loss)

    if log_file:
        sys.stdout = original_stdout
        log_fh.close()


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="态极 1B Native-V2 AUTODL 一键部署")
    parser.add_argument("--stage", default="all",
                        choices=["all", "setup", "data", "tokenizer", "smoke", "train"],
                        help="执行阶段 (默认: all)")
    parser.add_argument("--max-records", type=int, default=100_000,
                        help="每个数据源最大记录数 (默认: 100000)")
    args = parser.parse_args()

    # 先切换到安全目录，避免 CWD 被删除导致 FileNotFoundError
    os.chdir("/root/autodl-tmp")

    banner(f"态极 1B Native-V2 AUTODL 部署 - 阶段: {args.stage}")
    print(f"  项目目录: {PROJECT_DIR}")
    print(f"  Python: {sys.executable}")
    try:
        print(f"  工作目录: {os.getcwd()}")
    except FileNotFoundError:
        print(f"  工作目录: (已切换到安全目录)")

    # 确保项目目录存在
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    stage = args.stage

    if stage in ("all", "setup"):
        install_deps()

    if stage in ("all", "data"):
        download_data(max_records=args.max_records)

    if stage in ("all", "tokenizer"):
        build_tokenizer()

    if stage in ("all", "smoke"):
        run_smoke()

    if stage == "train":
        run_train()

    if stage == "all":
        banner("全部验证步骤完成!")
        print("下一步: 检查 smoke run 结果，如果正常则启动正式训练:")
        print(f"  python3 {__file__} --stage train")


if __name__ == "__main__":
    main()