# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InternLM-owned fast-tokenizer serialization fallback."""

from __future__ import annotations

from pathlib import Path


def ensure_tokenizer_json(
    model_dir: str | Path,
    *,
    previous_error: str | None = None,
    trust_remote_code: bool = False,
) -> bool:
    if type(trust_remote_code) is not bool:
        raise TypeError(
            "trust_remote_code must be a bool, got "
            f"{type(trust_remote_code).__name__}"
        )
    del previous_error
    path = Path(model_dir)
    if (path / "tokenizer.json").exists():
        return True

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(path),
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        if not getattr(tokenizer, "is_fast", False):
            return False
        tokenizer.save_pretrained(str(path))
    except Exception:
        return False

    return (path / "tokenizer.json").exists()
