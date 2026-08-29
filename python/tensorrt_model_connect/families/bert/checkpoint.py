# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-semantic helpers for BERT-family runtime selection."""

from __future__ import annotations

import json
from pathlib import Path


ENCODER_RUNTIME_STRATEGY = "bert_encoder_only"
EMBEDDING_RUNTIME_STRATEGY = "bert_embedding"


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def runtime_strategy_for_model(model_path: Path) -> str:
    """Select embedding only for explicit mean-pool + normalize semantics."""
    modules = _read_json(model_path / "modules.json")
    if not isinstance(modules, list):
        return ENCODER_RUNTIME_STRATEGY

    pooling_path: str | None = None
    semantic_modules: list[str] = []
    for module in modules:
        if not isinstance(module, dict):
            return ENCODER_RUNTIME_STRATEGY
        module_type = str(module.get("type", ""))
        semantic_modules.append(module_type.rsplit(".", 1)[-1])
        if module_type.endswith(".Pooling"):
            pooling_path = str(module.get("path") or "1_Pooling")

    if semantic_modules != ["Transformer", "Pooling", "Normalize"] or pooling_path is None:
        return ENCODER_RUNTIME_STRATEGY
    pooling_relative = Path(pooling_path)
    if pooling_relative.is_absolute() or ".." in pooling_relative.parts:
        return ENCODER_RUNTIME_STRATEGY

    pooling = _read_json(model_path / pooling_relative / "config.json")
    if not isinstance(pooling, dict):
        return ENCODER_RUNTIME_STRATEGY
    enabled_modes = {
        str(key)
        for key, value in pooling.items()
        if str(key).startswith("pooling_mode_") and value is True
    }
    mean_only = enabled_modes == {"pooling_mode_mean_tokens"}
    return EMBEDDING_RUNTIME_STRATEGY if mean_only else ENCODER_RUNTIME_STRATEGY
