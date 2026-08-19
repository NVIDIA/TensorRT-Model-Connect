# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preserve the effective Hugging Face StarCoder2 tokenizer contract."""

from __future__ import annotations

import json
from pathlib import Path


def tokenizer_json_bundle_override(model_dir: str | Path) -> bytes:
    """Return the effective tokenizer backend used by Transformers.

    The StarCoder2 checkpoint publishes a tokenizer.json with a leading
    ``Digits`` pre-tokenizer. Transformers configures its GPT-2 tokenizer with
    ByteLevel alone, so embedding the checkpoint file directly changes input
    token IDs for prompts containing indented numeric literals.
    """

    from transformers import AutoTokenizer

    model_dir_path = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir_path),
        local_files_only=True,
        use_fast=True,
    )
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    if backend_tokenizer is None or not hasattr(backend_tokenizer, "to_str"):
        raise RuntimeError(
            "StarCoder2 requires a fast Hugging Face tokenizer backend"
        )

    payload = str(backend_tokenizer.to_str()).encode("utf-8")
    json.loads(payload)
    return payload
