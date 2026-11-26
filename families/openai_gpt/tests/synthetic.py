# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A small tokenizer.json of the checkpoint's shape, and the variants both tokenizer contracts
refuse.

``TOKENIZER`` is the tokenizer of ``tests/cpp/test_openai_gpt_text_runtime.cpp``;
``mutations()`` mirrors the refusals that test checks natively, so the builder
(``tokenizer_contract.py``) and the runtime refuse the same files.
"""

from __future__ import annotations

import copy

_CHARS = ["h", "e", "l", "o", "w", "r", "d", "i", "n", "g", "a", "b", "c", "t", "s"]
_MERGES = [
    ("a", "a"), ("a", "a</w>"), ("aa", "a</w>"), ("h", "e"), ("l", "l"), ("he", "ll"),
    ("hell", "o</w>"), ("w", "o"), ("r", "l"), ("wo", "rl"), ("worl", "d</w>"), ("i", "n"),
    ("in", "g</w>"), ("t", "h"), ("th", "e</w>"),
]


def _vocab() -> dict[str, int]:
    tokens = ["<unk>", *_CHARS, *(char + "</w>" for char in [*_CHARS, "!", ",", "."])]
    for left, right in _MERGES:
        if left + right not in tokens:
            tokens.append(left + right)
    return {token: index for index, token in enumerate(tokens)}


TOKENIZER = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        {
            "id": 0,
            "special": True,
            "content": "<unk>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
        }
    ],
    "normalizer": {
        "type": "BertNormalizer",
        "clean_text": True,
        "handle_chinese_chars": True,
        "strip_accents": None,
        "lowercase": True,
    },
    "pre_tokenizer": {"type": "BertPreTokenizer"},
    "post_processor": None,
    "decoder": {"type": "BPEDecoder", "suffix": "</w>"},
    "model": {
        "dropout": None,
        "unk_token": "<unk>",
        "continuing_subword_prefix": None,
        "end_of_word_suffix": "</w>",
        "fuse_unk": False,
        "vocab": _vocab(),
        "merges": [f"{left} {right}" for left, right in _MERGES],
    },
}


def _replace_token(tokenizer: dict, token: str) -> None:
    """Swap the last vocabulary token (and the merge that builds it) for ``token``."""
    del tokenizer["model"]["vocab"]["the</w>"]
    tokenizer["model"]["vocab"][token] = 48
    tokenizer["model"]["merges"].remove("th e</w>")


def mutations() -> dict[str, dict]:
    """Variants of ``TOKENIZER`` that the native tokenizer refuses, by name."""
    changes = {
        "truncation": lambda t: t.update(truncation={"max_length": 8}),
        "cased normalizer": lambda t: t["normalizer"].update(lowercase=False),
        "accents kept": lambda t: t["normalizer"].update(strip_accents=False),
        "other normalizer": lambda t: t.update(normalizer={"type": "NFC"}),
        "other pre-tokenizer": lambda t: t.update(pre_tokenizer={"type": "Whitespace"}),
        "post-processor": lambda t: t.update(post_processor={"type": "TemplateProcessing"}),
        "decoder suffix": lambda t: t["decoder"].update(suffix="</x>"),
        "WordPiece model": lambda t: t["model"].update(type="WordPiece"),
        "fused unknown": lambda t: t["model"].update(fuse_unk=True),
        "byte fallback": lambda t: t["model"].update(byte_fallback=True),
        "dropout": lambda t: t["model"].update(dropout=0.1),
        "subword prefix": lambda t: t["model"].update(continuing_subword_prefix="##"),
        "empty suffix": lambda t: (
            t["model"].update(end_of_word_suffix=""),
            t["decoder"].update(suffix=""),
        ),
        "unknown token": lambda t: t["model"].update(unk_token="<missing>"),
        "id gap": lambda t: t["model"]["vocab"].update({"the</w>": 49}),
        "repeated id": lambda t: t["model"]["vocab"].update({"the</w>": 47}),
        "fractional id": lambda t: t["model"]["vocab"].update({"the</w>": 48.5}),
        "empty token": lambda t: _replace_token(t, ""),
        # An escaped lone surrogate, which no UTF-8 text holds.
        "lone surrogate token": lambda t: _replace_token(t, "x\ud800"),
        # U+1D165 is a combining mark that survives normalization and can be reordered.
        "reorderable mark": lambda t: _replace_token(t, "x\U0001d165"),
        "merge outside the vocabulary": lambda t: t["model"]["merges"].append("x y"),
        "merge result outside the vocabulary": lambda t: t["model"]["merges"].append("b c"),
        "repeated merge": lambda t: t["model"]["merges"].append("a a"),
        "merge without a pair": lambda t: t["model"]["merges"].append("aa"),
        "merge with two spaces": lambda t: t["model"]["merges"].append("a  a"),
        "added token outside the vocabulary": lambda t: t["added_tokens"][0].update(
            content="<pad>"
        ),
        "added token id": lambda t: t["added_tokens"][0].update(id=1),
        "normalized added token": lambda t: t["added_tokens"][0].update(normalized=True),
        "stripped added token": lambda t: t["added_tokens"][0].update(lstrip=True),
        "non-boolean option": lambda t: t["model"].update(fuse_unk=0),
        "missing vocabulary": lambda t: t["model"].pop("vocab"),
        # The reference refuses these files: the normaliser and added-token flags are required.
        "normalizer without clean_text": lambda t: t["normalizer"].pop("clean_text"),
        "normalizer without handle_chinese_chars": lambda t: t["normalizer"].pop(
            "handle_chinese_chars"
        ),
        "normalizer without lowercase": lambda t: t["normalizer"].pop("lowercase"),
        "added token without special": lambda t: t["added_tokens"][0].pop("special"),
        "added token without normalized": lambda t: t["added_tokens"][0].pop("normalized"),
        "added token without single_word": lambda t: t["added_tokens"][0].pop("single_word"),
        "added token without lstrip": lambda t: t["added_tokens"][0].pop("lstrip"),
        "added token without rstrip": lambda t: t["added_tokens"][0].pop("rstrip"),
        "repeated added token": lambda t: t["added_tokens"].append(
            {**t["added_tokens"][0], "special": False}
        ),
    }
    result = {}
    for name, change in changes.items():
        tokenizer = copy.deepcopy(TOKENIZER)
        change(tokenizer)
        result[name] = tokenizer
    return result
