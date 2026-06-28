from __future__ import annotations

import json
from pathlib import Path

from taiji.config import MM_CONTROL_TOKENS, MULTIMODAL_TOKENS, MULTIMODAL_VOCAB_SIZE
from taiji.multimodal.tokenizers import MultimodalTokenManager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "taiji" / "tokenizer_contract.json"
IMAGE_INFO_PATH = PROJECT_ROOT / "taiji_data" / "multimodal" / "image_tokenizer" / "codebook_info.json"
AUDIO_INFO_PATH = PROJECT_ROOT / "taiji_data" / "multimodal" / "audio_tokenizer" / "codebook_info.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_multimodal_config_matches_native_v2_contract() -> None:
    contract = _load_json(CONTRACT_PATH)
    mm = contract["multimodal"]
    mm_control = mm["control_tokens"]

    assert MULTIMODAL_TOKENS["image_token_base"] == mm["image"]["base"]
    assert MULTIMODAL_TOKENS["image_codebook_size"] == mm["image"]["codebook_size"]
    assert MULTIMODAL_TOKENS["audio_token_base"] == mm["audio"]["base"]
    assert MULTIMODAL_TOKENS["audio_codebook_size"] == mm["audio"]["codebook_size"]
    assert MM_CONTROL_TOKENS["image_start"] == mm_control["<mm_image>"]
    assert MM_CONTROL_TOKENS["image_end"] == mm_control["</mm_image>"]
    assert MM_CONTROL_TOKENS["audio_start"] == mm_control["<mm_audio>"]
    assert MM_CONTROL_TOKENS["audio_end"] == mm_control["</mm_audio>"]
    assert MULTIMODAL_VOCAB_SIZE == contract["total_vocab_size"]


def test_codebook_info_files_match_native_v2_contract() -> None:
    contract = _load_json(CONTRACT_PATH)
    image_info = _load_json(IMAGE_INFO_PATH)
    audio_info = _load_json(AUDIO_INFO_PATH)

    image_base = contract["multimodal"]["image"]["base"]
    image_size = contract["multimodal"]["image"]["codebook_size"]
    audio_base = contract["multimodal"]["audio"]["base"]
    audio_size = contract["multimodal"]["audio"]["codebook_size"]

    assert image_info["image_token_base"] == image_base
    assert image_info["vocab_range"] == [image_base, image_base + image_size - 1]
    assert audio_info["audio_token_base"] == audio_base
    assert audio_info["vocab_range"] == [audio_base, audio_base + audio_size - 1]


def test_multimodal_token_manager_uses_native_v2_offsets() -> None:
    manager = MultimodalTokenManager()

    assert manager.image_base == 1000
    assert manager.audio_base == 9192
    assert manager.reserved_vocab_end == 13388
    assert manager.total_vocab == 256000
    assert manager.get_image_logit_range() == (1000, 9192)
    assert manager.get_audio_logit_range() == (9192, 13288)
