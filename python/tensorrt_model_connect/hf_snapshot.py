# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Hugging Face snapshot selection contract.

Model Connect intentionally caches only files used by its builders and
references.  Every local snapshot probe must use this same positive allowlist;
an unfiltered ``snapshot_download`` would require unrelated upstream files and
reject an otherwise complete Model Connect cache.

The shared cache may contain repository Python files needed by explicit
reference workflows. Caching a file does not authorize importing it: native
tokenizer loads pass the public ``trust_remote_code`` value explicitly and
default it to false.
"""

from __future__ import annotations

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
