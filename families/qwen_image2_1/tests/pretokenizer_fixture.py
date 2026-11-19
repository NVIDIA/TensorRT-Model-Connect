# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tokenizer fixture that makes pre-token boundaries visible in token ids.

``data/boundary_tokenizer.json`` is a byte-level BPE tokenizer with the normalizer, pre-tokenizer
and decoder of the Qwen-Image 2.1 checkpoint's tokenizer, a vocabulary of the 256 byte symbols, no
merges and an end-of-word suffix. Byte ``b`` encodes as id ``b`` inside a pre-token and as id
``256 + b`` at its end, so the ids returned by ``encode`` spell out where the text was split. The
Hugging Face library and the family's native tokenizer both load this file; the cases in
``data/pretokenizer_cases.json`` hold the pieces the library produces.
"""

from __future__ import annotations

from pathlib import Path

from families.qwen_image2_1.bundle_config import TOKENIZER_NORMALIZER, TOKENIZER_PRE_TOKENIZER

DATA_ROOT = Path(__file__).resolve().parent / "data"
TOKENIZER_PATH = DATA_ROOT / "boundary_tokenizer.json"
CASES_PATH = DATA_ROOT / "pretokenizer_cases.json"

END_OF_WORD = "</w>"
ADDED_TOKENS = ("<|im_start|>", "<|im_end|>")
_FIRST_ADDED_ID = 512


def byte_symbols() -> list[str]:
    """The byte-level alphabet: printable Latin-1 bytes stand for themselves, the rest follow U+0100."""
    symbols = {
        byte: chr(byte)
        for byte in (*range(ord("!"), ord("~") + 1), *range(0xA1, 0xAD), *range(0xAE, 0x100))
    }
    shifted = 0
    for byte in range(256):
        if byte not in symbols:
            symbols[byte] = chr(256 + shifted)
            shifted += 1
    return [symbols[byte] for byte in range(256)]


def boundary_tokenizer() -> dict:
    """The content of ``data/boundary_tokenizer.json``."""
    symbols = byte_symbols()
    vocab = {symbol: index for index, symbol in enumerate(symbols)}
    vocab.update({symbol + END_OF_WORD: 256 + index for index, symbol in enumerate(symbols)})
    byte_level = dict(TOKENIZER_PRE_TOKENIZER["pretokenizers"][1])
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": _FIRST_ADDED_ID + index,
                "content": content,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for index, content in enumerate(ADDED_TOKENS)
        ],
        "normalizer": TOKENIZER_NORMALIZER,
        "pre_tokenizer": TOKENIZER_PRE_TOKENIZER,
        "post_processor": byte_level,
        "decoder": byte_level,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": END_OF_WORD,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": [],
        },
    }


def pieces(ids: list[int]) -> list[str]:
    """The pre-tokens and added tokens that ids of the boundary tokenizer spell out."""
    result: list[str] = []
    current = bytearray()
    for token in ids:
        if token >= _FIRST_ADDED_ID:
            if current:
                raise ValueError("an added token inside a pre-token")
            result.append(ADDED_TOKENS[token - _FIRST_ADDED_ID])
            continue
        current.append(token % 256)
        if token >= 256:
            result.append(current.decode("utf-8"))
            current = bytearray()
    if current:
        raise ValueError("the last pre-token has no end marker")
    return result
