# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time token contract: prompt ids, time-marker digits and the decode table.

The prompt is rendered once at build time with the checkpoint's own chat
template and tokenizer, exactly as the reference processor renders and splits
it around the single audio placeholder. The native runtime therefore never
encodes text; it only joins the stored ids with the expanded audio span and
decodes generated ids through the table produced here.
"""

from __future__ import annotations

import struct
from functools import lru_cache
from pathlib import Path

from .config import MossConfig

# The transcription instruction of the model card and of the official
# inference utilities (timestamped, speaker-labelled transcript).
DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)
AUDIO_PAD = "<|audio_pad|>"

TOKEN_TABLE_MAGIC = b"MTDTOKV1"
TOKEN_SKIPPED_ON_DECODE = 1


@lru_cache(maxsize=1)
def _byte_decoder() -> dict[str, int]:
    """Inverse of the GPT-2 byte-to-unicode map used by byte-level BPE."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codes = printable[:]
    shift = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codes.append(256 + shift)
            shift += 1
    return {chr(code): byte for byte, code in zip(printable, codes)}


def token_bytes(token: str) -> bytes:
    """Bytes one token contributes, following the tokenizers ByteLevel decoder."""
    table = _byte_decoder()
    if all(char in table for char in token):
        return bytes(table[char] for char in token)
    return token.encode("utf-8")


def _load_tokenizer(model_dir: Path):
    """The checkpoint's tokenizer class, loaded without resolving the model config.

    Resolving the config would execute the checkpoint's remote code; the
    tokenizer itself is a standard Qwen2 byte-level BPE.
    """
    import json

    from transformers import Qwen2Tokenizer

    declared = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    if declared.get("tokenizer_class") != "Qwen2Tokenizer":
        raise ValueError("moss_transcribe_diarize expects the checkpoint's Qwen2Tokenizer")
    return Qwen2Tokenizer.from_pretrained(str(model_dir))


def prompt_contract(model_dir: str | Path, config: MossConfig) -> dict[str, object]:
    """Prompt ids around the audio span and the digit ids of the time markers."""
    tokenizer = _load_tokenizer(Path(model_dir))
    if tokenizer.convert_tokens_to_ids(AUDIO_PAD) != config.audio_token_id:
        raise ValueError(f"tokenizer id of {AUDIO_PAD} differs from config.audio_token_id")
    messages = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": "input"},
            {"type": "text", "text": DEFAULT_PROMPT},
        ],
    }]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if rendered.count(AUDIO_PAD) != 1:
        raise ValueError("the chat template must place exactly one audio placeholder")
    before, after = rendered.split(AUDIO_PAD, 1)
    digits = []
    for digit in "0123456789":
        ids = tokenizer.encode(digit, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"time-marker digit {digit!r} is not a single token")
        digits.append(int(ids[0]))
    return {
        "prompt_prefix_ids": [int(i) for i in tokenizer.encode(before, add_special_tokens=False)],
        "prompt_suffix_ids": [int(i) for i in tokenizer.encode(after, add_special_tokens=False)],
        "digit_token_ids": digits,
    }


def decode_table(model_dir: str | Path, vocab_size: int) -> bytes:
    """Per-id decode bytes and skip flags, checked against the checkpoint tokenizer.

    Layout (little endian): magic, u32 count, u32 offsets[count + 1],
    u8 flags[count], then the concatenated bytes. Ids the tokenizer does not
    define contribute nothing, as the tokenizer's decoder drops them.
    """
    tokenizer = _load_tokenizer(Path(model_dir))
    backend = tokenizer.backend_tokenizer
    special = {int(i) for i in tokenizer.all_special_ids}
    for token in backend.get_added_tokens_decoder().values():
        if token.special:
            special.add(int(backend.token_to_id(token.content)))
    pieces: list[bytes] = []
    flags = bytearray()
    for token_id in range(vocab_size):
        token = backend.id_to_token(token_id)
        pieces.append(b"" if token is None else token_bytes(token))
        flags.append(TOKEN_SKIPPED_ON_DECODE if token_id in special else 0)
    for token_id in range(vocab_size):
        if flags[token_id] or not pieces[token_id]:
            continue
        expected = tokenizer.decode([token_id], skip_special_tokens=True)
        if pieces[token_id].decode("utf-8", errors="replace") != expected:
            raise ValueError(f"decode table disagrees with the tokenizer for id {token_id}")
    offsets = [0]
    for piece in pieces:
        offsets.append(offsets[-1] + len(piece))
    return b"".join([
        TOKEN_TABLE_MAGIC,
        struct.pack("<I", vocab_size),
        struct.pack(f"<{vocab_size + 1}I", *offsets),
        bytes(flags),
        *pieces,
    ])


def mel_filters(model_dir: str | Path) -> bytes:
    """The checkpoint feature extractor's mel filter bank as FP32 [freq_bins, mel_bins]."""
    import numpy as np
    from transformers import WhisperFeatureExtractor

    extractor = WhisperFeatureExtractor.from_pretrained(str(model_dir))
    filters = np.ascontiguousarray(extractor.mel_filters, dtype=np.float32)
    return struct.pack("<II", *filters.shape) + filters.tobytes()
