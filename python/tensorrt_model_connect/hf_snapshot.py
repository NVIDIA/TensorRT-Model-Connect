# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Hugging Face snapshot selection contract.

Model Connect intentionally caches only files used by its builders and
references.  Every local snapshot probe must use this same positive allowlist;
an unfiltered ``snapshot_download`` would require unrelated upstream files and
reject an otherwise complete Model Connect cache.
"""

from __future__ import annotations

from pathlib import Path

from .families import family_hf_allow_patterns


GENERIC_HF_ALLOW_PATTERNS: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors-*.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "*.model",
    "*.spm",
    "*.py",
)


def hf_snapshot_allow_patterns() -> list[str]:
    """Return the complete builder/reference snapshot allowlist."""
    return [
        *GENERIC_HF_ALLOW_PATTERNS,
        *family_hf_allow_patterns(),
        "*.nemo",
    ]


def hf_cache_snapshot_identity(
    path: str | Path,
) -> tuple[str, str] | None:
    """Return ``(org/repo, revision)`` for one canonical HF cache snapshot.

    Hugging Face cache snapshots use
    ``models--<org>--<repo>/snapshots/<revision>``.  Resolve symlinks before
    inspecting the path so callers receive the identity of the actual snapshot
    directory rather than an arbitrary local alias.
    """

    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_dir() or resolved.parent.name != "snapshots":
        return None
    cache_name = resolved.parent.parent.name
    if not cache_name.startswith("models--"):
        return None
    identity_parts = cache_name.removeprefix("models--").split("--", 1)
    if len(identity_parts) != 2 or not all(identity_parts):
        return None
    return "/".join(identity_parts), resolved.name
