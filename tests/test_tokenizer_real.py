"""Test the real SentencePiece tokenizer"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taiji.tokenizer import ModelSelfTokenizer


@pytest.fixture
def tokenizer():
    """Create a tokenizer instance for testing."""
    return ModelSelfTokenizer(sp_model_path='taiji/tokenizer_native_v2/sentencepiece.model')


class TestTokenizerBasic:
    """Test basic tokenizer functionality."""

    def test_encode_decode_english(self, tokenizer):
        """Test encoding and decoding English text."""
        text = 'Hello, how are you?'
        result = tokenizer(text)
        assert 'input_ids' in result
        assert 'attention_mask' in result

        ids = result['input_ids'][0]
        assert len(ids) > 0

        decoded = tokenizer.decode(ids)
        assert 'Hello' in decoded
        assert 'how are you' in decoded

    def test_encode_decode_chinese(self, tokenizer):
        """Test encoding and decoding Chinese text."""
        text = '你好世界'
        result = tokenizer(text)
        ids = result['input_ids'][0]
        assert len(ids) > 0

        decoded = tokenizer.decode(ids)
        assert '你好' in decoded or '世界' in decoded

    def test_encode_decode_mixed(self, tokenizer):
        """Test encoding and decoding mixed language text."""
        text = 'Taiji AI 是一个本地运行的 AI'
        result = tokenizer(text)
        ids = result['input_ids'][0]
        assert len(ids) > 0

        decoded = tokenizer.decode(ids)
        assert 'Taiji' in decoded

    def test_batch_encoding(self, tokenizer):
        """Test batch encoding."""
        texts = ['Hello', 'World', 'Test']
        result = tokenizer(texts, padding=True)
        assert 'input_ids' in result
        assert len(result['input_ids']) == 3

    def test_truncation(self, tokenizer):
        """Test truncation."""
        text = 'This is a long text ' * 100
        result = tokenizer(text, truncation=True, max_length=10)
        ids = result['input_ids'][0]
        assert len(ids) <= 10


class TestTokenizerSpecialTokens:
    """Test special token handling."""

    def test_special_token_encoding(self, tokenizer):
        """Test encoding of special tokens."""
        special = '<think>test</think><tool_call>search'
        result = tokenizer(special)
        ids = result['input_ids'][0]

        think_start = tokenizer.convert_tokens_to_ids('<think>')
        think_end = tokenizer.convert_tokens_to_ids('</think>')
        tool_call = tokenizer.convert_tokens_to_ids('<tool_call>')

        assert think_start in ids
        assert think_end in ids
        assert tool_call in ids

    def test_tool_registration(self, tokenizer):
        """Test tool token registration."""
        tokenizer.register_tool('search')
        tokenizer.register_tool('read_file')

        tool_id = tokenizer.get_tool_id('search')
        assert tool_id is not None
        assert tool_id >= 190  # Tool tokens start at 190

    def test_tool_encoding(self, tokenizer):
        """Test tool token in encoding."""
        tokenizer.register_tool('search')
        tool_id = tokenizer.get_tool_id('search')

        text = f'<tool_call>search'
        result = tokenizer(text)
        ids = result['input_ids'][0]

        assert tool_id in ids

    def test_pad_token(self, tokenizer):
        """Test pad token handling."""
        assert tokenizer.pad_token_id == 0

    def test_bos_eos_tokens(self, tokenizer):
        """Test BOS and EOS tokens."""
        assert tokenizer.bos_token_id == 2
        assert tokenizer.eos_token_id == 3


class TestTokenizerContract:
    """Test tokenizer contract compliance."""

    def test_vocab_size(self, tokenizer):
        """Test vocab size matches contract."""
        assert tokenizer.total_vocab_size == 256000
        assert tokenizer.text_vocab_size == 242612
        assert tokenizer.text_offset == 13388

    def test_special_token_ids(self, tokenizer):
        """Test special token IDs match contract."""
        assert tokenizer.special_text_to_id['<pad>'] == 0
        assert tokenizer.special_text_to_id['<unk>'] == 1
        assert tokenizer.special_text_to_id['<s>'] == 2
        assert tokenizer.special_text_to_id['</s>'] == 3

    def test_multimodal_token_ranges(self, tokenizer):
        """Test multimodal token ranges."""
        image_base = tokenizer.contract['multimodal']['image']['base']
        audio_base = tokenizer.contract['multimodal']['audio']['base']

        assert image_base == 1000
        assert audio_base == 9192


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
