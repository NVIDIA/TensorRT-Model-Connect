# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time tokenization for the Qwen3-ASR runtime.

The transcription prompt is fixed apart from the number of audio placeholders and the optional
forced language, so the native runtime needs no BPE encoder: the prompt pieces and every
forced-language suffix are tokenized here, and the runtime only turns generated ids back into text
through a byte table.

Prompt: ``Qwen3ASRProcessor.apply_transcription_request`` renders the checkpoint's chat template
for one audio turn with an empty system prompt and an assistant turn to continue:

    <|im_start|>system\\n<|im_end|>\\n<|im_start|>user\\n<|audio_start|><|audio_pad|><|audio_end|>
    <|im_end|>\\n<|im_start|>assistant\\n[language <Name><asr_text>]

and replaces the single ``<|audio_pad|>`` with one placeholder per audio token.
"""

from __future__ import annotations

import struct
from pathlib import Path

from .frontend import LANGUAGES

_PROMPT = (
    "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"
    "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n"
)
ASR_TEXT_MARKER = "<asr_text>"

VOCAB_MAGIC = b"Q3AV"
VOCAB_VERSION = 1
FLAG_SPECIAL = 1
FLAG_UNKNOWN = 2


def load_tokenizer(model_dir: Path):
    from tokenizers import Tokenizer

    path = model_dir / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"Qwen3-ASR checkpoint is missing {path}")
    return Tokenizer.from_file(str(path))


def _encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False).ids)


def prompt_pieces(tokenizer, audio_token_id: int) -> tuple[list[int], list[int]]:
    """Prompt ids before and after the audio placeholders."""
    ids = _encode(tokenizer, _PROMPT)
    if ids.count(audio_token_id) != 1:
        raise ValueError("the Qwen3-ASR prompt must contain exactly one audio placeholder")
    split = ids.index(audio_token_id)
    return ids[:split], ids[split + 1 :]


def language_suffixes(tokenizer) -> list[dict]:
    """Assistant prefill ``language <Name><asr_text>`` for every language the processor accepts."""
    marker = tokenizer.token_to_id(ASR_TEXT_MARKER)
    if marker is None:
        raise ValueError(f"the Qwen3-ASR tokenizer has no {ASR_TEXT_MARKER} token")
    entries = []
    for code, name in LANGUAGES:
        ids = _encode(tokenizer, f"language {name}{ASR_TEXT_MARKER}")
        if not ids or ids[-1] != marker:
            raise ValueError(f"forced-language prefill for {name} does not end with {ASR_TEXT_MARKER}")
        entries.append({"code": code, "name": name, "ids": ids})
    return entries


def _byte_decoder() -> dict[str, int]:
    """Inverse of the GPT-2 byte-to-unicode map used by byte-level BPE."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    characters = printable[:]
    extra = 0
    for value in range(256):
        if value not in printable:
            printable.append(value)
            characters.append(256 + extra)
            extra += 1
    return {chr(character): value for value, character in zip(printable, characters)}


def vocabulary_table(tokenizer, vocab_size: int) -> bytes:
    """Bytes of every token id for native detokenization.

    Layout (little endian): magic ``Q3AV``, uint32 version, uint32 count, then per id a uint8 flag
    (``1`` special: dropped like ``skip_special_tokens=True``; ``2`` unknown id), a uint16 length
    and the token's bytes. Added tokens contribute their literal text; ordinary tokens are
    byte-level BPE strings mapped back to bytes.
    """
    decoder = _byte_decoder()
    added = tokenizer.get_added_tokens_decoder()
    out = bytearray(VOCAB_MAGIC)
    out += struct.pack("<II", VOCAB_VERSION, vocab_size)
    for token_id in range(vocab_size):
        if token_id in added:
            token = added[token_id]
            data = token.content.encode("utf-8")
            flag = FLAG_SPECIAL if token.special else 0
        else:
            text = tokenizer.id_to_token(token_id)
            if text is None:
                data, flag = b"", FLAG_UNKNOWN
            else:
                try:
                    data = bytes(decoder[character] for character in text)
                except KeyError as error:
                    raise ValueError(f"token {token_id} is not a byte-level BPE string") from error
                flag = 0
        if len(data) > 0xFFFF:
            raise ValueError(f"token {token_id} is too long for the vocabulary table")
        out += struct.pack("<BH", flag, len(data))
        out += data
    return bytes(out)


def decode_with_table(table: bytes, token_ids: list[int]) -> str:
    """Python mirror of the runtime's detokenization, for tests."""
    if table[:4] != VOCAB_MAGIC:
        raise ValueError("not a Qwen3-ASR vocabulary table")
    version, count = struct.unpack_from("<II", table, 4)
    if version != VOCAB_VERSION:
        raise ValueError("unsupported vocabulary table version")
    offset = 12
    entries = []
    for _ in range(count):
        flag, length = struct.unpack_from("<BH", table, offset)
        offset += 3
        entries.append((flag, table[offset : offset + length]))
        offset += length
    data = b"".join(entries[token][1] for token in token_ids if entries[token][0] == 0)
    return data.decode("utf-8", errors="replace")
