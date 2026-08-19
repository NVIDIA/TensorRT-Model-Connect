# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M2M-100/NLLB-owned tokenizer.json conversion fallback."""

from __future__ import annotations

from pathlib import Path

from ...tokenizer_conversion import ensure_unigram_tokenizer_json


def ensure_tokenizer_json(
    model_dir: str | Path,
    *,
    previous_error: str | None = None,
) -> bool:
    return ensure_unigram_tokenizer_json(
        model_dir,
        sentencepiece_candidates=("source.spm", "spiece.model", "*.spm", "*.model"),
        vocab_json_name="vocab.json",
        previous_error=previous_error,
    )
