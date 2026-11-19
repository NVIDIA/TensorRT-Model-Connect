# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer fixtures that the Hugging Face library and the native tokenizer both load.

``data/boundary_tokenizer.json`` makes pre-token boundaries visible in token ids. It is a
byte-level BPE tokenizer with the normalizer, pre-tokenizer and decoder of the Qwen-Image 2.1
checkpoint's tokenizer, a vocabulary of the 256 byte symbols, no merges and an end-of-word suffix.
Byte ``b`` encodes as id ``b`` inside a pre-token and as id ``256 + b`` at its end, so the ids
returned by ``encode`` spell out where the text was split; the cases in
``data/pretokenizer_cases.json`` hold the pieces the library produces.

``data/merge_tokenizer.json`` pins the order in which merges apply. It has the same frame, no
end-of-word suffix and the merge table ``MERGE_RULES``; ``data/merge_cases.json`` holds the ids
the library produces.
"""

from __future__ import annotations

from pathlib import Path

from families.qwen_image2_1.bundle_config import TOKENIZER_NORMALIZER, TOKENIZER_PRE_TOKENIZER

DATA_ROOT = Path(__file__).resolve().parent / "data"
TOKENIZER_PATH = DATA_ROOT / "boundary_tokenizer.json"
CASES_PATH = DATA_ROOT / "pretokenizer_cases.json"

MERGE_TOKENIZER_PATH = DATA_ROOT / "merge_tokenizer.json"
MERGE_CASES_PATH = DATA_ROOT / "merge_cases.json"

END_OF_WORD = "</w>"
ADDED_TOKENS = ("<|im_start|>", "<|im_end|>")
_FIRST_ADDED_ID = 512

# Merge rules in rank order ("\u0120" is the byte-level symbol of a space). The reference applies
# one merge at a time, the lowest rank first and the leftmost pair among equal ranks, and each
# merge makes the pairs around the new token candidates. Three properties of this table give
# other ids when every occurrence of the best pair is merged per pass instead: "ab" + "a" ranks
# ahead of "a" + "b", which produces "ab"; "c" + "d" appears twice, so its later rank counts and
# "cd" + "c" ranks ahead of it; "e" + "fg" and "ef" + "g" have the same result.
MERGE_RULES = (
    ("ab", "a"),
    ("a", "b"),
    ("c", "d"),
    ("cd", "c"),
    ("e", "fg"),
    ("f", "g"),
    ("e", "f"),
    ("ef", "g"),
    ("c", "d"),
    ("\u0120", "a"),
    ("\u0120a", "b"),
)


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


def _tokenizer(vocab: dict[str, int], merges: list[list[str]], end_of_word_suffix: str) -> dict:
    """A tokenizer.json with the checkpoint's frame around the given BPE model."""
    byte_level = dict(TOKENIZER_PRE_TOKENIZER["pretokenizers"][1])
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": len(vocab) + index,
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
            "end_of_word_suffix": end_of_word_suffix,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": merges,
        },
    }


def boundary_tokenizer() -> dict:
    """The content of ``data/boundary_tokenizer.json``."""
    symbols = byte_symbols()
    vocab = {symbol: index for index, symbol in enumerate(symbols)}
    vocab.update({symbol + END_OF_WORD: 256 + index for index, symbol in enumerate(symbols)})
    assert len(vocab) == _FIRST_ADDED_ID
    return _tokenizer(vocab, [], END_OF_WORD)


def merge_tokenizer() -> dict:
    """The content of ``data/merge_tokenizer.json``."""
    vocab = {symbol: index for index, symbol in enumerate(byte_symbols())}
    for first, second in MERGE_RULES:
        vocab.setdefault(first + second, len(vocab))
    return _tokenizer(vocab, [list(rule) for rule in MERGE_RULES], "")


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
