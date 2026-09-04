# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve InternLM's family-owned native tokenizer artifact."""

from __future__ import annotations

from pathlib import Path


TOKENIZER_REPO = "internlm/internlm2-step-prover"
TOKENIZER_REVISION = "6c727046190546168bf3aba9a1d78d5fb325ff14"


def ensure_tokenizer_json(
    model_dir: str | Path,
    *,
    previous_error: str | None = None,
) -> Path:
    """Return the checkpoint JSON or the family's exact official dependency."""

    path = Path(model_dir) / "tokenizer.json"
    if path.is_file():
        return path
    if not (Path(model_dir) / "tokenizer.model").is_file():
        detail = f": {previous_error}" if previous_error else ""
        raise FileNotFoundError(f"InternLM checkpoint is missing tokenizer.model{detail}")

    try:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=TOKENIZER_REPO,
                filename="tokenizer.json",
                revision=TOKENIZER_REVISION,
            )
        )
    except Exception as error:
        detail = f": {previous_error}" if previous_error else ""
        raise FileNotFoundError(
            f"InternLM tokenizer dependency is unavailable: "
            f"{TOKENIZER_REPO}@{TOKENIZER_REVISION}{detail}"
        ) from error
