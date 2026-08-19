# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""StarCoder2-owned validation refinements."""

from __future__ import annotations

import json
from typing import Any, Mapping


def transform_bundle_tokenizer(
    source_payload: bytes,
    tokenizer_config: Mapping[str, Any],
) -> bytes:
    """Apply StarCoder2's declared GPT-2 wrapper to its backend tokenizer.

    The published tokenizer uses ``Sequence[Digits, ByteLevel]`` while the
    declared ``GPT2Tokenizer`` wrapper replaces that sequence with ByteLevel.
    Only the exact reviewed shape is transformed; every other shape is left
    byte-for-byte unchanged.
    """

    tokenizer_class = str(tokenizer_config.get("tokenizer_class", "") or "")
    if tokenizer_class.removesuffix("Fast") != "GPT2Tokenizer":
        return source_payload

    tokenizer = json.loads(source_payload.decode("utf-8"))
    if not isinstance(tokenizer, dict):
        raise ValueError("tokenizer.json must contain an object")
    model = tokenizer.get("model")
    pre_tokenizer = tokenizer.get("pre_tokenizer")
    if (
        not isinstance(model, dict)
        or model.get("type") != "BPE"
        or not isinstance(pre_tokenizer, dict)
        or pre_tokenizer.get("type") != "Sequence"
    ):
        return source_payload

    sequence = pre_tokenizer.get("pretokenizers")
    if not isinstance(sequence, list) or len(sequence) != 2:
        return source_payload
    digits = [
        item
        for item in sequence
        if isinstance(item, dict) and item.get("type") == "Digits"
    ]
    byte_levels = [
        item
        for item in sequence
        if isinstance(item, dict) and item.get("type") == "ByteLevel"
    ]
    if (
        len(digits) != 1
        or digits[0].get("individual_digits") is not True
        or len(byte_levels) != 1
    ):
        return source_payload

    byte_level = dict(byte_levels[0])
    configured_prefix = tokenizer_config.get("add_prefix_space")
    if (
        configured_prefix is not None
        and (
            not isinstance(configured_prefix, bool)
            or byte_level.get("add_prefix_space") is not configured_prefix
        )
    ):
        return source_payload
    tokenizer["pre_tokenizer"] = byte_level
    return json.dumps(tokenizer, separators=(",", ":")).encode("utf-8")
