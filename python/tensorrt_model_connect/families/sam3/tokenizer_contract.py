# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-data validation for SAM3's native CLIP BPE tokenizer asset."""

from __future__ import annotations

import json
from typing import Any


class Sam3TokenizerContractError(ValueError):
    """Raised when tokenizer.json cannot drive the native SAM3 tokenizer."""


def _parse_merge(entry: Any) -> tuple[str, str] | None:
    if isinstance(entry, str):
        tokens = entry.split(" ")
        return (tokens[0], tokens[1]) if len(tokens) == 2 and all(tokens) else None
    if (
        isinstance(entry, list)
        and len(entry) == 2
        and all(isinstance(token, str) and token for token in entry)
    ):
        return entry[0], entry[1]
    return None


def validate_sam3_tokenizer_json(
    payload: bytes | str,
    *,
    expected_vocab_size: int | None = None,
) -> tuple[int, int]:
    """Validate and return ``(vocab_size, merge_count)`` for native BPE use."""

    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        document = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Sam3TokenizerContractError("tokenizer.json is not valid UTF-8 JSON") from error

    model = document.get("model") if isinstance(document, dict) else None
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise Sam3TokenizerContractError("tokenizer.json must contain a BPE model")

    vocab = model.get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise Sam3TokenizerContractError("tokenizer.json must contain a non-empty BPE vocab")
    if not all(isinstance(token, str) and token for token in vocab):
        raise Sam3TokenizerContractError("tokenizer.json BPE vocab tokens must be strings")
    ids = list(vocab.values())
    if not all(isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in ids):
        raise Sam3TokenizerContractError("tokenizer.json BPE vocab IDs must be integers")
    if len(set(ids)) != len(ids) or set(ids) != set(range(len(ids))):
        raise Sam3TokenizerContractError(
            "tokenizer.json BPE vocab IDs must be unique and dense from zero"
        )
    if expected_vocab_size is not None and len(vocab) != expected_vocab_size:
        raise Sam3TokenizerContractError(
            "tokenizer.json BPE vocab size does not match the SAM3 text encoder: "
            f"expected {expected_vocab_size}, found {len(vocab)}"
        )

    merges = model.get("merges")
    if not isinstance(merges, list) or not merges:
        raise Sam3TokenizerContractError("tokenizer.json must contain non-empty BPE merges")
    for index, entry in enumerate(merges):
        pair = _parse_merge(entry)
        if pair is None:
            raise Sam3TokenizerContractError(
                f"tokenizer.json contains an invalid BPE merge at index {index}"
            )
        left, right = pair
        if left not in vocab or right not in vocab or left + right not in vocab:
            raise Sam3TokenizerContractError(
                f"tokenizer.json BPE merge operands and result must exist in vocab: index {index}"
            )

    added_tokens = document.get("added_tokens", [])
    if not isinstance(added_tokens, list):
        raise Sam3TokenizerContractError("tokenizer.json added_tokens must be a list")
    for index, token in enumerate(added_tokens):
        token_id = token.get("id") if isinstance(token, dict) else None
        content = token.get("content") if isinstance(token, dict) else None
        if (
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or not isinstance(content, str)
            or not content
            or vocab.get(content) != token_id
        ):
            raise Sam3TokenizerContractError(
                "tokenizer.json added token must map to the same in-vocab integer ID: "
                f"index {index}"
            )
        if "special" in token and not isinstance(token["special"], bool):
            raise Sam3TokenizerContractError(
                f"tokenizer.json added token special flag must be a boolean: index {index}"
            )
    return len(vocab), len(merges)


__all__ = ["Sam3TokenizerContractError", "validate_sam3_tokenizer_json"]
