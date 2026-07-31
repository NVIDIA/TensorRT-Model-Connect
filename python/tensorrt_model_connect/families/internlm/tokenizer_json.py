# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InternLM-owned fast-tokenizer serialization fallback."""

from __future__ import annotations

from pathlib import Path


def ensure_tokenizer_json(
    model_dir: str | Path,
    *,
    previous_error: str | None = None,
) -> bool:
    del previous_error
    path = Path(model_dir)
    if (path / "tokenizer.json").exists():
        return True

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(path),
            trust_remote_code=True,
            use_fast=True,
            from_slow=True,
        )
        if not getattr(tokenizer, "is_fast", False):
            return False
        tokenizer.save_pretrained(str(path))
    except Exception:
        return False

    return (path / "tokenizer.json").exists()
