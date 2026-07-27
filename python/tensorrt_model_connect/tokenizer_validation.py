# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural validation for tokenizer artifacts consumed by native runtimes."""

from __future__ import annotations

import json
from pathlib import Path


def native_tokenizer_json_error(tokenizer_path: Path) -> str | None:
    """Return why ``tokenizer.json`` is incompatible with native tokenizers.

    This validates the minimum model structure consumed by the native BPE,
    WordPiece, and Unigram implementations. Optional normalization,
    pre-tokenization, decoding, and special-token sections remain optional.
    """
    try:
        document = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    except OSError as exc:
        return f"cannot read tokenizer.json: {exc}"
    except json.JSONDecodeError as exc:
        return f"invalid tokenizer.json: {exc}"

    if not isinstance(document, dict):
        return "tokenizer.json root must be an object"
    model = document.get("model")
    if not isinstance(model, dict):
        return "tokenizer.json must contain an object-valued model"

    model_type = model.get("type")
    if model_type is not None and (
        not isinstance(model_type, str)
        or model_type not in {"BPE", "WordPiece", "Unigram"}
    ):
        return f"unsupported tokenizer model.type: {model_type!r}"

    vocab = model.get("vocab")
    if model_type is None:
        if isinstance(vocab, dict) and isinstance(model.get("merges"), list):
            model_type = "BPE"
        elif (
            isinstance(vocab, dict)
            and "merges" not in model
            and "continuing_subword_prefix" in model
        ):
            model_type = "WordPiece"
        elif (
            isinstance(vocab, list)
            and "merges" not in model
            and "continuing_subword_prefix" not in model
        ):
            model_type = "Unigram"
        else:
            return "cannot identify a native BPE, WordPiece, or Unigram model"

    if model_type in {"BPE", "WordPiece"}:
        if not isinstance(vocab, dict) or not vocab:
            return f"{model_type} model.vocab must be a non-empty object"
        token_ids = list(vocab.values())
        if any(type(token_id) is not int for token_id in token_ids):
            return f"{model_type} model.vocab IDs must be integers"
        if len(set(token_ids)) != len(token_ids):
            return f"{model_type} model.vocab IDs must be unique"
        if set(token_ids) != set(range(len(token_ids))):
            return f"{model_type} model.vocab IDs must cover 0..{len(token_ids) - 1}"

    if model_type == "BPE":
        merges = model.get("merges")
        if not isinstance(merges, list):
            return "BPE model.merges must be an array"
        for merge in merges:
            if isinstance(merge, str):
                continue
            if (
                isinstance(merge, list)
                and len(merge) == 2
                and all(isinstance(token, str) for token in merge)
            ):
                continue
            return "BPE model.merges entries must be strings or two-string arrays"
        return None

    if model_type == "WordPiece":
        return None

    if not isinstance(vocab, list) or not vocab:
        return "Unigram model.vocab must be a non-empty array"
    for entry in vocab:
        if (
            not isinstance(entry, list)
            or len(entry) < 2
            or not isinstance(entry[0], str)
            or isinstance(entry[1], bool)
            or not isinstance(entry[1], (int, float))
        ):
            return "Unigram model.vocab entries must contain a string token and numeric score"
    unk_id = model.get("unk_id", 0)
    if type(unk_id) is not int or not 0 <= unk_id < len(vocab):
        return "Unigram model.unk_id must index model.vocab"
    return None
