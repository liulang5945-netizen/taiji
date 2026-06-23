"""
Taiji Model Loader (native-v2)

统一加载器：只走 native-v2 路线。
模型目录结构:
    config.json              — 模型配置
    model.pt                 — 模型权重
    sentencepiece.model      — SentencePiece 词表
    tokenizer_contract.json  — 词表 ID 空间契约 (可选，优先使用)
"""
import json
import logging
import os
import shutil
import torch
from typing import Callable, Optional, Tuple

from .config import ModelConfig
from .architecture import ModelSelf
from .tokenizer import ModelSelfTokenizer

logger = logging.getLogger("Taiji")

_SIZE_BUILDERS: dict[str, Callable[[], ModelConfig]] = {
    "125m": ModelConfig.size_125m,
    "350m": ModelConfig.size_350m,
    "1b": ModelConfig.size_1b,
    "3b": ModelConfig.size_3b,
    "7b": ModelConfig.size_7b,
}


def create_model(
    size: str = "125m",
    device: str = "cpu",
    active_heads: list[str] | None = None,
    sp_model_path: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[ModelSelf, ModelSelfTokenizer]:
    """Create a fresh ModelSelf model and tokenizer pair."""
    normalized_size = str(size).strip().lower()
    builder = _SIZE_BUILDERS.get(normalized_size)
    if builder is None:
        raise ValueError(f"Unsupported model size: {size}")

    config = builder()
    if active_heads is not None:
        config.active_heads = list(active_heads)

    tokenizer = ModelSelfTokenizer(sp_model_path=sp_model_path)
    config.vocab_size = tokenizer.vocab_size
    config.base_vocab_size = min(config.base_vocab_size, tokenizer.vocab_size)
    config.num_special_tokens = max(0, tokenizer.vocab_size - config.base_vocab_size)
    model = ModelSelf(config).to(device=device, dtype=dtype)
    return model, tokenizer


def load_model(
    model_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[ModelSelf, "SentencePieceTokenizer"]:
    """
    加载态极模型 (native-v2)。

    Args:
        model_path: 模型目录 (含 config.json, model.pt, sentencepiece.model)
        device: 设备
        dtype: 数据类型

    Returns:
        (model, tokenizer) 元组
    """
    import sentencepiece as spm

    model_path = os.path.abspath(model_path)

    # 1. 加载配置
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    config = _dict_to_config(config_dict)

    # 2. 创建模型
    model = ModelSelf(config)

    # 3. 加载权重
    weights_path = os.path.join(model_path, "model.pt")
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded weights: {weights_path}")
    else:
        logger.warning(f"No weights at {weights_path}, random init")

    # 4. 加载 SentencePiece tokenizer
    sp_path = _find_sentencepiece(model_path)
    if sp_path is None:
        raise FileNotFoundError(f"SentencePiece model not found in {model_path}")

    sp = spm.SentencePieceProcessor()
    sp.Load(sp_path)
    logger.info(f"Loaded tokenizer: {sp_path} (vocab={sp.GetPieceSize()})")

    # 5. 加载 tokenizer contract (如果有)
    contract_path = os.path.join(model_path, "tokenizer_contract.json")
    contract = None
    if os.path.exists(contract_path):
        with open(contract_path, "r", encoding="utf-8") as f:
            contract = json.load(f)
        logger.info(f"Loaded contract: vocab={contract.get('total_vocab_size')}, "
                     f"text_offset={contract.get('text_offset')}")

    # 6. 创建统一 tokenizer
    tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        tokenizer = ModelSelfTokenizer.load(model_path, sp_model_path=sp_path)
    else:
        tokenizer = SentencePieceTokenizer(sp, contract)

    # 7. 移动模型到设备
    model = model.to(device=device, dtype=dtype)

    logger.info(f"Model loaded: {config.describe()}")
    return model, tokenizer


def _find_sentencepiece(model_path: str) -> Optional[str]:
    """查找 SentencePiece 模型文件"""
    # 优先: 模型目录下的 sentencepiece.model
    sp = os.path.join(model_path, "sentencepiece.model")
    if os.path.exists(sp):
        return sp

    # 回退: tokenizer/ 子目录
    sp = os.path.join(model_path, "tokenizer", "sentencepiece.model")
    if os.path.exists(sp):
        return sp

    # 回退: 包内默认
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    sp = os.path.join(pkg_dir, "tokenizer", "sentencepiece.model")
    if os.path.exists(sp):
        return sp

    return None


class SentencePieceTokenizer:
    """
    统一 tokenizer 封装。

    兼容 native-v2 的 contract ID 空间：
    - 特殊 token 有固定 ID (contract 中定义)
    - 文本 token 从 text_offset 开始
    - SentencePiece 内部 ID + text_offset = 全局 ID
    """

    def __init__(self, sp, contract: dict = None):
        self._sp = sp
        self._contract = contract
        self._text_offset = 0

        if contract:
            self._text_offset = int(contract.get("text_offset", 0))
            self._special_tokens = contract.get("special_tokens", {})
        else:
            self._special_tokens = {}

    @property
    def vocab_size(self) -> int:
        if self._contract:
            return int(self._contract.get("total_vocab_size", self._sp.GetPieceSize()))
        return self._sp.GetPieceSize()

    @property
    def eos_token_id(self) -> int:
        return self._special_tokens.get("</s>", 3)

    @property
    def pad_token_id(self) -> int:
        return self._special_tokens.get("<pad>", 0)

    def __call__(self, text: str, return_tensors: str = "pt",
                 max_length: int = 2048, truncation: bool = True,
                 padding: str = "max_length", **kwargs):
        """编码文本，返回 tensor"""
        ids = self.encode(text)
        if truncation:
            ids = ids[:max_length]

        if return_tensors == "pt":
            import torch
            if padding == "max_length":
                pad_len = max_length - len(ids)
                ids = ids + [self.pad_token_id] * pad_len
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def encode(self, text: str) -> list:
        """文本 → 全局 token ID 列表"""
        sp_ids = self._sp.EncodeAsIds(text)
        # 偏移到全局 ID 空间
        return [tid + self._text_offset for tid in sp_ids]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        """全局 token ID → 文本"""
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]

        # 还原 SentencePiece 内部 ID
        sp_ids = []
        for tid in ids:
            if tid >= self._text_offset:
                sp_ids.append(tid - self._text_offset)
            elif not skip_special_tokens:
                # 特殊 token，用 contract 查内容
                content = self._id_to_special.get(tid, "")
                if content:
                    sp_ids.append(self._sp.PieceToId(content))

        if not sp_ids:
            return ""
        return self._sp.DecodeIds(sp_ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        """token 字符串 → 全局 ID"""
        # 先查特殊 token
        if token in self._special_tokens:
            return self._special_tokens[token]
        # 再查 SentencePiece
        sp_id = self._sp.PieceToId(token)
        if sp_id != self._sp.unk_id():
            return sp_id + self._text_offset
        return self._sp.unk_id()

    @property
    def _id_to_special(self) -> dict:
        """反向映射: ID → 特殊 token 内容"""
        if not hasattr(self, '_id_to_special_cache'):
            self._id_to_special_cache = {v: k for k, v in self._special_tokens.items()}
        return self._id_to_special_cache


def save_model(model, tokenizer, save_path: str):
    """保存态极模型"""
    os.makedirs(save_path, exist_ok=True)

    # 保存 config
    config_dict = {
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": model.config.num_key_value_heads,
        "max_position_embeddings": model.config.max_position_embeddings,
        "rms_norm_eps": model.config.rms_norm_eps,
        "rope_theta": model.config.rope_theta,
        "base_vocab_size": model.config.base_vocab_size,
        "num_special_tokens": model.config.num_special_tokens,
        "num_short_term_slots": model.config.num_short_term_slots,
        "num_long_term_slots": model.config.num_long_term_slots,
        "memory_dim": model.config.memory_dim,
    }
    with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # 保存权重
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))

    if hasattr(tokenizer, "save"):
        tokenizer.save(save_path)

    sp_model_path = getattr(tokenizer, "sp_model_path", None)
    if sp_model_path and os.path.exists(sp_model_path):
        shutil.copyfile(sp_model_path, os.path.join(save_path, "sentencepiece.model"))

    logger.info(f"Model saved to {save_path}")


def _dict_to_config(d: dict) -> ModelConfig:
    """字典 → ModelConfig"""
    return ModelConfig(
        vocab_size=d.get("vocab_size", 256000),
        hidden_size=d.get("hidden_size", 1024),
        intermediate_size=d.get("intermediate_size", 2816),
        num_hidden_layers=d.get("num_hidden_layers", 24),
        num_attention_heads=d.get("num_attention_heads", 16),
        num_key_value_heads=d.get("num_key_value_heads", 16),
        max_position_embeddings=d.get("max_position_embeddings", 2048),
        rms_norm_eps=d.get("rms_norm_eps", 1e-6),
        rope_theta=d.get("rope_theta", 1000000.0),
        attention_bias=d.get("attention_bias", False),
        base_vocab_size=d.get("base_vocab_size", 256000),
        num_special_tokens=d.get("num_special_tokens", 0),
        num_short_term_slots=d.get("num_short_term_slots", 20),
        num_long_term_slots=d.get("num_long_term_slots", 10),
        memory_dim=d.get("memory_dim", 64),
        active_heads=d.get("active_heads", None),
    )
