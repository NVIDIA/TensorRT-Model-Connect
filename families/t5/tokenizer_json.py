# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T5-owned tokenizer.json conversion."""

from __future__ import annotations

from pathlib import Path

from .tokenizer_conversion import ensure_unigram_tokenizer_json


def ensure_tokenizer_json(
    model_dir: str | Path,
    *,
    previous_error: str | None = None,
) -> bool:
    return ensure_unigram_tokenizer_json(
        model_dir,
        sentencepiece_candidates=("spiece.model", "tokenizer.model", "*.model"),
        previous_error=previous_error,
    )
