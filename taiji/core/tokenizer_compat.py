"""
Tokenizer 兼容层
统一处理不同 tokenizer 实现的差异，避免直接调用私有 API

支持的 tokenizer：
- transformers.PreTrainedTokenizer（通用）
- sentencepiece.SentencePieceProcessor
- tiktoken.Encoding
"""
from typing import List, Union, Optional
import logging

logger = logging.getLogger("Taiji.Tokenizer.Compat")


class TokenizerCompat:
    """Tokenizer 兼容适配器"""
    
    @staticmethod
    def encode(tokenizer, text: str) -> List[int]:
        """
        统一编码接口
        
        Args:
            tokenizer: 任何支持的 tokenizer 实例
            text: 要编码的文本
            
        Returns:
            token IDs 列表
        """
        if tokenizer is None:
            raise ValueError("Tokenizer cannot be None")
        
        # 方法优先级：公开 API → 私有 API → 备选
        
        # 1. transformers PreTrainedTokenizer（推荐）
        if hasattr(tokenizer, 'encode') and callable(tokenizer.encode):
            try:
                return tokenizer.encode(text, add_special_tokens=True)
            except Exception as e:
                logger.warning(f"encode() failed: {e}, trying alternatives")
        
        # 2. sentencepiece 风格
        if hasattr(tokenizer, 'EncodeAsIds') and callable(tokenizer.EncodeAsIds):
            try:
                return tokenizer.EncodeAsIds(text)
            except Exception as e:
                logger.warning(f"EncodeAsIds() failed: {e}, trying alternatives")
        
        # 3. tiktoken 风格
        if hasattr(tokenizer, 'encode_ordinary') and callable(tokenizer.encode_ordinary):
            try:
                return tokenizer.encode_ordinary(text)
            except Exception as e:
                logger.warning(f"encode_ordinary() failed: {e}, trying alternatives")
        
        # 4. 私有 API（最后尝试）
        if hasattr(tokenizer, '_encode') and callable(tokenizer._encode):
            logger.info("Using private _encode() API - consider updating tokenizer")
            try:
                return tokenizer._encode(text)
            except Exception as e:
                logger.error(f"_encode() failed: {e}")
        
        # 都不支持
        raise ValueError(
            f"Tokenizer {type(tokenizer).__name__} does not support any known encode method. "
            f"Supported: encode(), EncodeAsIds(), encode_ordinary(), _encode()"
        )
    
    @staticmethod
    def decode(tokenizer, token_ids: List[int]) -> str:
        """
        统一解码接口
        
        Args:
            tokenizer: 任何支持的 tokenizer 实例
            token_ids: token IDs 列表
            
        Returns:
            解码后的文本
        """
        if tokenizer is None:
            raise ValueError("Tokenizer cannot be None")
        
        # 1. 标准 decode
        if hasattr(tokenizer, 'decode') and callable(tokenizer.decode):
            try:
                return tokenizer.decode(token_ids, skip_special_tokens=True)
            except Exception as e:
                logger.warning(f"decode() failed: {e}, trying alternatives")
        
        # 2. sentencepiece 风格
        if hasattr(tokenizer, 'DecodeIds') and callable(tokenizer.DecodeIds):
            try:
                return tokenizer.DecodeIds(token_ids)
            except Exception as e:
                logger.warning(f"DecodeIds() failed: {e}")
        
        raise ValueError(
            f"Tokenizer {type(tokenizer).__name__} does not support any known decode method"
        )
    
    @staticmethod
    def get_pad_token_id(tokenizer) -> Optional[int]:
        """
        获取 pad token ID，兼容不同实现
        
        Args:
            tokenizer: 任何支持的 tokenizer 实例
            
        Returns:
            pad token ID 或 None
        """
        if tokenizer is None:
            return None
        
        # 1. 公开属性
        if hasattr(tokenizer, 'pad_token_id'):
            return tokenizer.pad_token_id
        
        # 2. 私有属性（备选）
        if hasattr(tokenizer, '_pad_token_id'):
            logger.info("Using private _pad_token_id attribute")
            return tokenizer._pad_token_id
        
        # 3. 尝试 eos_token_id（备选）
        if hasattr(tokenizer, 'eos_token_id'):
            logger.info("Using eos_token_id as fallback for pad_token_id")
            return tokenizer.eos_token_id
        
        # 4. 常见的 pad token ID（最后尝试）
        logger.warning("Could not find pad_token_id, using default 0")
        return 0
    
    @staticmethod
    def get_vocab_size(tokenizer) -> Optional[int]:
        """获取词表大小"""
        if tokenizer is None:
            return None
        
        # 优先级：vocab_size 属性 → get_vocab() 方法 → len(get_vocab())
        if hasattr(tokenizer, 'vocab_size'):
            return tokenizer.vocab_size
        
        if hasattr(tokenizer, 'get_vocab') and callable(tokenizer.get_vocab):
            try:
                return len(tokenizer.get_vocab())
            except Exception:
                pass
        
        if hasattr(tokenizer, '__len__'):
            return len(tokenizer)
        
        logger.warning("Could not determine vocab_size")
        return None
    
    @staticmethod
    def validate(tokenizer) -> bool:
        """
        验证 tokenizer 是否可用
        
        Args:
            tokenizer: tokenizer 实例
            
        Returns:
            True 如果 tokenizer 有效且可用
        """
        if tokenizer is None:
            return False
        
        # 检查是否至少有一个编码方法
        has_encode = (
            hasattr(tokenizer, 'encode') or
            hasattr(tokenizer, 'EncodeAsIds') or
            hasattr(tokenizer, 'encode_ordinary') or
            hasattr(tokenizer, '_encode')
        )
        
        # 检查是否至少有一个解码方法
        has_decode = (
            hasattr(tokenizer, 'decode') or
            hasattr(tokenizer, 'DecodeIds')
        )
        
        return has_encode and has_decode


# 向后兼容别名
def safe_tokenizer_encode(tokenizer, text: str) -> List[int]:
    """安全的 tokenizer 编码（已弃用，使用 TokenizerCompat.encode）"""
    return TokenizerCompat.encode(tokenizer, text)

def safe_tokenizer_decode(tokenizer, token_ids: List[int]) -> str:
    """安全的 tokenizer 解码（已弃用，使用 TokenizerCompat.decode）"""
    return TokenizerCompat.decode(tokenizer, token_ids)