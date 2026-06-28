"""
态极多模态离散 Tokenizer
========================

将图像和音频编码为离散 token，与文本 token 共享同一个词表。模型可以用统一的 Transformer 同时生成文本、图像、音频。

架构:
  图像 -> VQ-VAE Encoder -> 离散 token (codebook 8192)
  音频 -> EnCodec-style Encoder -> 离散 token (codebook 4096)
  文本 -> SentencePiece/BPE -> 离散 token (已有)

  所有 token 拼接成统一序列 -> Transformer -> 输出 token -> 各模态 decoder

参考:
  - DALL-E (图像 token): VQ-VAE + Transformer
  - MusicLM (音频 token): SoundStream/EnCodec + Transformer
  - Chameleon (统一 token): Meta 的多模态原生模型
"""
import os
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

logger = logging.getLogger("Taiji.Tokenizers")


# ============================================================================
# 向量量化层 (VQ)
# ============================================================================
class VectorQuantizer(nn.Module):
    """
    向量量化层 — 将连续向量映射到离散 codebook

    输入: [batch, dim, h, w] 或 [batch, dim, seq]
    输出: 离散 token IDs [batch, h, w] 或 [batch, seq]
         量化后的连续向量 (用于 decoder)
    """

    def __init__(self, codebook_size: int, dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment_cost = commitment_cost

        # codebook: 可学习的嵌入表
        self.embedding = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [batch, dim, ...] 连续向量

        Returns:
            z_q: 量化后的向量 (same shape as z)
            indices: 离散 token IDs [batch, ...]
            vq_loss: 向量量化损失
        """
        # 将 dim 维度放到最后
        z_perm = z.permute(0, 2, 3, 1) if z.dim() == 4 else z.permute(0, 2, 1)  # [batch, ..., dim]
        flat = z_perm.reshape(-1, self.dim)  # [N, dim]

        # 计算到 codebook 的距离
        dist = (flat.unsqueeze(1) - self.embedding.weight.unsqueeze(0)) ** 2  # [N, K, dim]
        dist = dist.sum(dim=-1)  # [N, K]

        # 找到最近的 codebook entry
        indices = dist.argmin(dim=1).long()  # [N]
        z_q = self.embedding(indices)  # [N, dim]

        # reshape 回原尺寸
        z_q = z_q.reshape(z_perm.shape)
        indices = indices.reshape(z_perm.shape[:-1])

        # 恢复维度顺序
        if z.dim() == 4:
            z_q = z_q.permute(0, 3, 1, 2)
        else:
            z_q = z_q.permute(0, 2, 1)

        # VQ 损失: commitment loss + codebook loss
        commitment_loss = F.mse_loss(z, z_q.detach())
        codebook_loss = F.mse_loss(z.detach(), z_q)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: 前向用 z_q，反向梯度传给 z
        z_q = z + (z_q - z).detach()

        return z_q, indices, vq_loss


# ============================================================================
# 图像 VQ-VAE Tokenizer
# ============================================================================
class ImageVQEncoder(nn.Module):
    """
    图像 VQ-VAE 编码器
    输入: [batch, 3, 256, 256] 图像
    输出: [batch, 16, 16] 离散 token IDs (256 个 token)

    架构: 4 层下采样卷积 -> VQ
    """

    def __init__(self, codebook_size: int = 8192, hidden_dim: int = 256):
        super().__init__()
        self.codebook_size = codebook_size

        # 编码器: 256x256 -> 16x16 (4 层下采样, 每层 2x)
        self.encoder = nn.Sequential(
            # 256->128
            nn.Conv2d(3, hidden_dim // 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim // 4),
            nn.SiLU(),
            # 128->64
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.SiLU(),
            # 64->32
            nn.Conv2d(hidden_dim // 2, hidden_dim, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            # 32->16
            nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            # 投影到 codebook 维度
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        )

        # 向量量化
        self.vq = VectorQuantizer(codebook_size, hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, 3, 256, 256] 图像

        Returns:
            z_q: 量化后的特征 [batch, hidden_dim, 16, 16]
            indices: 离散 token IDs [batch, 16, 16]
            vq_loss: VQ 损失
        """
        z = self.encoder(x)
        z_q, indices, vq_loss = self.vq(z)
        return z_q, indices, vq_loss


class ImageVQDecoder(nn.Module):
    """
    图像 VQ-VAE 解码器
    输入: [batch, hidden_dim, 16, 16] 量化特征 (或 codebook IDs)
    输出: [batch, 3, 256, 256] 重建图像
    """

    def __init__(self, codebook_size: int = 8192, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, hidden_dim)

        self.decoder = nn.Sequential(
            # 16->32
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            # 32->64
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.SiLU(),
            # 64->128
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim // 4),
            nn.SiLU(),
            # 128->256
            nn.ConvTranspose2d(hidden_dim // 4, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: [batch, 16, 16] 离散 token IDs

        Returns:
            image: [batch, 3, 256, 256] 重建图像 (pixel range [-1, 1])
        """
        # 从 codebook 查找连续向量
        z_q = self.embedding(indices.long())  # [batch, 16, 16, hidden_dim]
        z_q = z_q.permute(0, 3, 1, 2)  # [batch, hidden_dim, 16, 16]
        return self.decoder(z_q)


class ImageTokenizer:
    """
    图像离散 Tokenizer (推理用)

    将图像编码为离散 token ID 序列，用于 Transformer 输入。
    将离散 token ID 解码回图像，用于 Transformer 输出。

    用法:
        tokenizer = ImageTokenizer("path/to/image_vq.pt")
        tokens = tokenizer.encode(image)        # [batch, 256] (16x16 展平)
        image = tokenizer.decode(tokens)         # [batch, 3, 256, 256]
    """

    def __init__(self, model_path: str = None, device: str = "cpu",
                 codebook_size: int = 8192, hidden_dim: int = 256):
        self.device = device
        self.codebook_size = codebook_size
        self.grid_size = 16  # 16x16 = 256 tokens per image

        self.encoder = ImageVQEncoder(codebook_size, hidden_dim).to(device)
        self.decoder = ImageVQDecoder(codebook_size, hidden_dim).to(device)

        if model_path and os.path.exists(model_path):
            self.load(model_path)

        self.encoder.eval()
        self.decoder.eval()

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """
        图像 -> 离散 token IDs

        Args:
            image: [batch, 3, 256, 256] 像素范围 [0, 1] 或 [-1, 1]

        Returns:
            indices: [batch, 256] 离散 token IDs (0 ~ codebook_size-1)
        """
        # 归一化到 [-1, 1]
        if image.max() > 1.0:
            image = image / 127.5 - 1.0

        _, indices, _ = self.encoder(image.to(self.device))
        # 展平 16x16 -> 256
        return indices.reshape(image.shape[0], -1)

    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        离散 token IDs -> 图像

        Args:
            indices: [batch, 256] 或 [batch, 16, 16] 离散 token IDs

        Returns:
            image: [batch, 3, 256, 256] 像素范围 [-1, 1]
        """
        if indices.dim() == 2:
            batch = indices.shape[0]
            indices = indices.reshape(batch, self.grid_size, self.grid_size)

        return self.decoder(indices.to(self.device))

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'codebook_size': self.codebook_size,
        }, path)
        logger.info(f"ImageTokenizer 保存到: {path}")

    def load(self, path: str):
        """加载模型"""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.decoder.load_state_dict(ckpt['decoder'])
        logger.info(f"ImageTokenizer 加载: {path}")


# ============================================================================
# 音频 Tokenizer (EnCodec-style)
# ============================================================================
class AudioEncoder(nn.Module):
    """
    音频编码器 (1D 卷积)

    输入: [batch, 1, samples] 波形 (16kHz, 4秒 = 64000 samples)
    输出: [batch, 128, frames] 特征序列 (约 50 frames)
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        # 下采样: 64000 -> 50 (约 1280x 压缩)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 4, 7, stride=2, padding=3),
            nn.GroupNorm(8, hidden_dim // 4),
            nn.SiLU(),
            nn.Conv1d(hidden_dim // 4, hidden_dim // 2, 5, stride=2, padding=2),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, 5, stride=2, padding=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, stride=2, padding=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, stride=2, padding=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, stride=2, padding=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, stride=2, padding=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, stride=2, padding=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, stride=2, padding=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class AudioDecoder(nn.Module):
    """
    音频解码器 (1D 转置卷积)

    输入: [batch, hidden_dim, frames] 量化特征
    输出: [batch, 1, samples] 波形
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.SiLU(),
            nn.ConvTranspose1d(hidden_dim // 2, hidden_dim // 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim // 4),
            nn.SiLU(),
            nn.ConvTranspose1d(hidden_dim // 4, 1, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class AudioTokenizer:
    """
    音频离散 Tokenizer (推理用)

    将音频波形编码为离散 token ID 序列。
    将离散 token ID 解码回音频波形。

    用法:
        tokenizer = AudioTokenizer("path/to/audio_vq.pt")
        tokens = tokenizer.encode(waveform)     # [batch, max_frames]
        audio = tokenizer.decode(tokens)         # [batch, 1, samples]
    """

    def __init__(self, model_path: str = None, device: str = "cpu",
                 codebook_size: int = 4096, hidden_dim: int = 128,
                 max_frames: int = 64):
        self.device = device
        self.codebook_size = codebook_size
        self.max_frames = max_frames

        self.encoder = AudioEncoder(hidden_dim).to(device)
        self.decoder = AudioDecoder(hidden_dim).to(device)
        self.vq = VectorQuantizer(codebook_size, hidden_dim).to(device)

        if model_path and os.path.exists(model_path):
            self.load(model_path)

        self.encoder.eval()
        self.decoder.eval()

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """
        音频波形 -> 离散 token IDs

        Args:
            audio: [batch, 1, samples] 波形 (16kHz)

        Returns:
            indices: [batch, max_frames] 离散 token IDs
        """
        z = self.encoder(audio.to(self.device))
        _, indices, _ = self.vq(z)

        # 截断或填充到 max_frames
        if indices.shape[1] > self.max_frames:
            indices = indices[:, :self.max_frames]
        elif indices.shape[1] < self.max_frames:
            pad = torch.zeros(indices.shape[0], self.max_frames - indices.shape[1],
                              dtype=indices.dtype, device=indices.device)
            indices = torch.cat([indices, pad], dim=1)

        return indices.squeeze(1) if indices.dim() == 3 else indices

    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        离散 token IDs -> 音频波形

        Args:
            indices: [batch, max_frames] 离散 token IDs

        Returns:
            audio: [batch, 1, samples] 波形
        """
        z_q = self.vq.embedding(indices.to(self.device))
        z_q = z_q.permute(0, 2, 1)  # [batch, hidden_dim, frames]
        return self.decoder(z_q)

    def save(self, path: str):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'vq': self.vq.state_dict(),
            'codebook_size': self.codebook_size,
            'max_frames': self.max_frames,
        }, path)
        logger.info(f"AudioTokenizer 保存到: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.decoder.load_state_dict(ckpt['decoder'])
        self.vq.load_state_dict(ckpt['vq'])
        logger.info(f"AudioTokenizer 加载: {path}")


# ============================================================================
# 统一多模态 Token 管理器
# ============================================================================
@dataclass
class MultimodalInput:
    """多模态统一输入"""
    text_ids: torch.Tensor = None         # [batch, text_len] 文本 token IDs
    image_ids: torch.Tensor = None        # [batch, 256] 图像 token IDs
    audio_ids: torch.Tensor = None        # [batch, frames] 音频 token IDs
    attention_mask: torch.Tensor = None   # [batch, total_len] 注意力掩码

class MultimodalTokenManager:
    """
    多模态 Token 管理器

    职责:
    1. 管理各模态 token 的 ID 范围
    2. 将多模态输入拼接成统一序列
    3. 从统一输出中分离各模态 token
    4. 将图像/音频 token 偏移到词表对应区域

    用法:
        manager = MultimodalTokenManager()

        # 编码: 图像 -> token IDs (偏移到词表中的图像区域)
        image_tokens = manager.encode_image(raw_indices)  # 0~8191 -> 1000~9191

        # 拼接: 文本 + 图像 + 音频 -> 统一序列
        unified = manager.merge(text_ids, image_tokens, audio_tokens)

        # 分离: 从模型输出中提取各模态
        text_out, image_out, audio_out = manager.split(logits)
    """

    def __init__(self):
        from taiji.config import (
            MM_CONTROL_TOKENS,
            MULTIMODAL_TOKENS,
            MULTIMODAL_VOCAB_SIZE,
        )

        self.image_base = MULTIMODAL_TOKENS["image_token_base"]
        self.audio_base = MULTIMODAL_TOKENS["audio_token_base"]
        self.image_codebook = MULTIMODAL_TOKENS["image_codebook_size"]
        self.audio_codebook = MULTIMODAL_TOKENS["audio_codebook_size"]

        # 控制 token
        self.IMAGE_START = MM_CONTROL_TOKENS["image_start"]
        self.IMAGE_END = MM_CONTROL_TOKENS["image_end"]
        self.AUDIO_START = MM_CONTROL_TOKENS["audio_start"]
        self.AUDIO_END = MM_CONTROL_TOKENS["audio_end"]

        # Keep both the full model vocab and the end of the reserved multimodal span.
        self.reserved_vocab_end = (
            self.audio_base
            + self.audio_codebook
            + MULTIMODAL_TOKENS["mm_control_size"]
        )
        self.total_vocab = MULTIMODAL_VOCAB_SIZE

    def encode_image(self, raw_indices: torch.Tensor) -> torch.Tensor:
        """
        将 VQ-VAE 输出的原始 token IDs (0~8191) 偏移到词表中的图像区域

        Args:
            raw_indices: [batch, 256] 值域 0~8191

        Returns:
            offset_indices: [batch, 256] 值域 1000~9191
        """
        return raw_indices + self.image_base

    def decode_image(self, offset_indices: torch.Tensor) -> torch.Tensor:
        """将词表中的图像 token IDs 还原为 VQ-VAE 原始 IDs"""
        return offset_indices - self.image_base

    def encode_audio(self, raw_indices: torch.Tensor) -> torch.Tensor:
        """将 EnCodec 输出的原始 token IDs (0~4095) 偏移到词表中的音频区域"""
        return raw_indices + self.audio_base

    def decode_audio(self, offset_indices: torch.Tensor) -> torch.Tensor:
        """将词表中的音频 token IDs 还原为 EnCodec 原始 IDs"""
        return offset_indices - self.audio_base

    def wrap_image_tokens(self, image_ids: torch.Tensor) -> torch.Tensor:
        """在图像 token 序列前后加上控制 token: <image> ... </image>"""
        batch = image_ids.shape[0]
        start = torch.full((batch, 1), self.IMAGE_START, dtype=image_ids.dtype, device=image_ids.device)
        end = torch.full((batch, 1), self.IMAGE_END, dtype=image_ids.dtype, device=image_ids.device)
        return torch.cat([start, image_ids, end], dim=1)

    def wrap_audio_tokens(self, audio_ids: torch.Tensor) -> torch.Tensor:
        """在音频 token 序列前后加上控制 token: <audio> ... </audio>"""
        batch = audio_ids.shape[0]
        start = torch.full((batch, 1), self.AUDIO_START, dtype=audio_ids.dtype, device=audio_ids.device)
        end = torch.full((batch, 1), self.AUDIO_END, dtype=audio_ids.dtype, device=audio_ids.device)
        return torch.cat([start, audio_ids, end], dim=1)

    def is_image_token(self, token_id: int) -> bool:
        """判断 token ID 是否在图像区域"""
        return self.image_base <= token_id < self.image_base + self.image_codebook

    def is_audio_token(self, token_id: int) -> bool:
        """判断 token ID 是否在音频区域"""
        return self.audio_base <= token_id < self.audio_base + self.audio_codebook

    def get_image_logit_range(self) -> Tuple[int, int]:
        """获取 lm_head 中图像 token 的 logit 切片范围"""
        return self.image_base, self.image_base + self.image_codebook

    def get_audio_logit_range(self) -> Tuple[int, int]:
        """获取 lm_head 中音频 token 的 logit 切片范围"""
        return self.audio_base, self.audio_base + self.audio_codebook
