# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``tokenizer.json`` shape the native tokenizer reproduces, checked before a build.

The native tokenizer (``runtime/tokenizer.cpp``) implements one pipeline: the uncased
``BertNormalizer``, ``BertPreTokenizer``, a BPE model whose words end with an
end-of-word suffix (``</w>``) and map each unknown character to the unknown token,
no post-processor, and the ``BPEDecoder`` of the same suffix. It refuses any other
file when the bundle loads; this check refuses the same files when the bundle is
built, so every bundle that builds also loads.
"""

from __future__ import annotations

from .config import parse_json_object
from .reorderable_marks import RANGES as _REORDERABLE


def _flag(section: dict, key: str, default: bool) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"OpenAI GPT tokenizer field {key!r} must be a boolean")
    return value


def _required_flag(section: dict, key: str) -> bool:
    """A boolean the reference requires to be present."""
    if key not in section:
        raise ValueError(f"OpenAI GPT tokenizer field {key!r} is missing")
    return _flag(section, key, False)


def _holds_reorderable_mark(token: str) -> bool:
    return any(lo <= ord(char) <= hi for char in token for lo, hi in _REORDERABLE)


def load_tokenizer(data: bytes) -> dict:
    """Parse ``tokenizer.json``; a repeated key (for example a vocabulary entry) is refused."""
    return parse_json_object(data, "tokenizer.json")


def _check_normalizer(tokenizer: dict) -> None:
    normalizer = tokenizer.get("normalizer")
    # strip_accents null follows lowercase, which must be true.
    strip = normalizer.get("strip_accents") if isinstance(normalizer, dict) else None
    if (
        not isinstance(normalizer, dict)
        or normalizer.get("type") != "BertNormalizer"
        or not _required_flag(normalizer, "clean_text")
        or not _required_flag(normalizer, "handle_chinese_chars")
        or not _required_flag(normalizer, "lowercase")
        or (strip is not None and strip is not True)
    ):
        raise ValueError(
            "OpenAI GPT tokenizer.json must use the uncased BertNormalizer (clean text, separate "
            "CJK, strip accents, lowercase)"
        )
    pre_tokenizer = tokenizer.get("pre_tokenizer")
    if not isinstance(pre_tokenizer, dict) or pre_tokenizer.get("type") != "BertPreTokenizer":
        raise ValueError("OpenAI GPT tokenizer.json must use BertPreTokenizer")
    if tokenizer.get("post_processor") is not None:
        raise ValueError("OpenAI GPT tokenizer.json must not add special tokens (no post-processor)")


def _merge_pair(entry: object) -> tuple[str, str]:
    if isinstance(entry, str):
        parts = entry.split(" ")
    elif isinstance(entry, list):
        parts = entry
    else:
        parts = []
    if len(parts) != 2 or not all(isinstance(part, str) and part for part in parts):
        raise ValueError(f"OpenAI GPT tokenizer merge {entry!r} is not one pair of tokens")
    return parts[0], parts[1]


def _check_model(model: object) -> tuple[dict, str]:
    if not isinstance(model, dict) or model.get("type", "BPE") != "BPE" or "merges" not in model:
        raise ValueError("OpenAI GPT tokenizer.json must hold a BPE model")
    if model.get("dropout") is not None:
        raise ValueError("OpenAI GPT tokenizer BPE dropout is not supported")
    if model.get("continuing_subword_prefix") is not None:
        raise ValueError("OpenAI GPT tokenizer BPE continuing_subword_prefix is not supported")
    if _flag(model, "fuse_unk", False) or _flag(model, "byte_fallback", False):
        raise ValueError("OpenAI GPT tokenizer must map each unknown character to its own token")
    if _flag(model, "ignore_merges", False):
        raise ValueError("OpenAI GPT tokenizer BPE ignore_merges is not supported")
    suffix = model.get("end_of_word_suffix")
    if not isinstance(suffix, str) or not suffix:
        raise ValueError("OpenAI GPT tokenizer BPE must declare an end_of_word_suffix")
    vocab = model.get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("OpenAI GPT tokenizer BPE vocabulary must be a non-empty object")
    ids = list(vocab.values())
    if any(isinstance(value, bool) or not isinstance(value, int) for value in ids):
        raise ValueError("OpenAI GPT tokenizer vocabulary ids must be integers")
    if sorted(ids) != list(range(len(ids))):
        raise ValueError("OpenAI GPT tokenizer vocabulary ids must be 0..n-1, each once")
    for token in vocab:
        if not token:
            raise ValueError("OpenAI GPT tokenizer vocabulary holds an empty token")
        if any(0xD800 <= ord(char) <= 0xDFFF for char in token):
            raise ValueError(f"OpenAI GPT tokenizer vocabulary token {token!r} is not valid Unicode")
        if _holds_reorderable_mark(token):
            raise ValueError(
                f"OpenAI GPT tokenizer vocabulary token {token!r} holds a combining mark the "
                "native tokenizer cannot order"
            )
    unk = model.get("unk_token")
    if not isinstance(unk, str) or unk not in vocab:
        raise ValueError("OpenAI GPT tokenizer unk_token must be a vocabulary token")
    merges = model["merges"]
    if not isinstance(merges, list):
        raise ValueError("OpenAI GPT tokenizer merges must be a list")
    seen = set()
    for entry in merges:
        left, right = _merge_pair(entry)
        if left not in vocab or right not in vocab or left + right not in vocab:
            raise ValueError(f"OpenAI GPT tokenizer merge {entry!r} leaves the vocabulary")
        if (left, right) in seen:
            raise ValueError(f"OpenAI GPT tokenizer merge {entry!r} is listed twice")
        seen.add((left, right))
    return vocab, suffix


def _check_added_tokens(tokenizer: dict, vocab: dict) -> None:
    added = tokenizer.get("added_tokens", [])
    if not isinstance(added, list):
        raise ValueError("OpenAI GPT tokenizer added_tokens must be a list")
    seen: set[int] = set()
    for entry in added:
        if not isinstance(entry, dict):
            raise ValueError("OpenAI GPT tokenizer added token must be an object")
        content, token_id = entry.get("content"), entry.get("id")
        if (
            not isinstance(content, str)
            or not content
            or isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or vocab.get(content) != token_id
        ):
            raise ValueError(
                f"OpenAI GPT tokenizer added token {content!r} must be a vocabulary token with "
                "its vocabulary id"
            )
        # A second entry for one token could change whether decoding skips it.
        if token_id in seen:
            raise ValueError(f"OpenAI GPT tokenizer added token {content!r} is listed twice")
        seen.add(token_id)
        _required_flag(entry, "special")
        if any(
            _required_flag(entry, key) for key in ("normalized", "single_word", "lstrip", "rstrip")
        ):
            raise ValueError(
                f"OpenAI GPT tokenizer added token {content!r} uses matching options the native "
                "tokenizer does not implement"
            )


def check_tokenizer(tokenizer: object, vocab_size: int) -> None:
    """Raise unless ``tokenizer`` (parsed ``tokenizer.json``) is what the runtime accepts."""
    if not isinstance(tokenizer, dict):
        raise ValueError("OpenAI GPT tokenizer.json must contain one JSON object")
    if tokenizer.get("truncation") is not None or tokenizer.get("padding") is not None:
        raise ValueError("OpenAI GPT tokenizer.json must neither truncate nor pad")
    _check_normalizer(tokenizer)
    vocab, suffix = _check_model(tokenizer.get("model"))
    decoder = tokenizer.get("decoder")
    if (
        not isinstance(decoder, dict)
        or decoder.get("type") != "BPEDecoder"
        or decoder.get("suffix") != suffix
    ):
        raise ValueError("OpenAI GPT tokenizer.json must decode with BPEDecoder of the model suffix")
    _check_added_tokens(tokenizer, vocab)
    # Every id the model can produce must have a token, and every token an embedding row.
    if len(vocab) != vocab_size:
        raise ValueError(
            f"OpenAI GPT tokenizer has {len(vocab)} tokens for {vocab_size} embedding rows"
        )
