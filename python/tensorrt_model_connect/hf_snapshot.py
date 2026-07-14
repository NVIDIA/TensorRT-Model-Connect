# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Hugging Face snapshot selection contract."""

from __future__ import annotations

from .families import family_hf_allow_patterns


GENERIC_HF_ALLOW_PATTERNS = (
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
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "*.model",
    "*.spm",
    "*.py",
)


def hf_snapshot_allow_patterns() -> list[str]:
    """Return every model file pattern accepted by the shared builder."""
    return [*GENERIC_HF_ALLOW_PATTERNS, *family_hf_allow_patterns(), "*.nemo"]
