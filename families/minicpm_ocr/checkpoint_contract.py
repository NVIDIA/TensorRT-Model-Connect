# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free checks of what the builder accepts from a checkpoint.

The builder refuses quantized checkpoints and a tokenizer.json that declares another pipeline than the native
tokenizer implements, instead of failing late or shipping a bundle whose prompts would tokenize differently from
the official processor.
"""

from __future__ import annotations

import json
from pathlib import Path

# runtime/bpe_tokenizer.cpp: NFC, then this Split pattern (every match and every gap between matches is a piece),
# then byte-level BPE with merges.
SPLIT_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                 r"|\s*[\r\n]+|\s+(?!\S)|\s+")


def tokenizer_bytes(model_dir: Path) -> bytes:
    """Return tokenizer.json after checking that it declares the pipeline the native tokenizer implements."""
    data = (model_dir / "tokenizer.json").read_bytes()
    tokenizer = json.loads(data)
    model = tokenizer.get("model") or {}
    pre_tokenizer = tokenizer.get("pre_tokenizer") or {}
    steps = pre_tokenizer.get("pretokenizers") or []
    split, byte_level = steps if len(steps) == 2 else ({}, {})
    supported = (
        model.get("type") == "BPE" and not model.get("byte_fallback") and not model.get("ignore_merges")
        and not model.get("continuing_subword_prefix") and not model.get("end_of_word_suffix")
        and tokenizer.get("normalizer") == {"type": "NFC"}
        and pre_tokenizer.get("type") == "Sequence"
        and split.get("type") == "Split" and split.get("pattern") == {"Regex": SPLIT_PATTERN}
        and split.get("behavior") == "Isolated" and split.get("invert") is False
        and byte_level.get("type") == "ByteLevel" and byte_level.get("use_regex") is False
        and byte_level.get("add_prefix_space") is False
        and (tokenizer.get("decoder") or {}).get("type") == "ByteLevel")
    if not supported:
        raise ValueError("minicpm_ocr: tokenizer.json must declare the NFC normalizer, the Qwen Split pattern and "
                         "byte-level BPE, the pipeline the native tokenizer implements")
    return data


def require_unquantized(config: dict) -> None:
    """Refuse quantized derivatives (AWQ, GPTQ, ...): same identity, different tensor layout."""
    sections = (config, config.get("text_config") or {}, config.get("vision_config") or {})
    if any("quantization_config" in section for section in sections):
        raise ValueError("minicpm_ocr: quantized checkpoints are not supported; build from the BF16 release")
