# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for timm CrossViT builds."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file


try:
    import tensorrt  # noqa: F401
except ModuleNotFoundError:
    sys.modules["tensorrt"] = SimpleNamespace()

from families.timm_crossvit import model  # noqa: E402
from families.timm_crossvit.checkpoint import Checkpoint  # noqa: E402
from families.timm_crossvit.support import describe  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


def _random(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _checkpoint(tmp_path: Path, depths: tuple[tuple[int, int], ...] = ((1, 2),)) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, np.ndarray] = {}
    for branch in range(2):
        tensors[f"patch_embed.{branch}.proj.weight"] = _random(4, 3, 4 + branch * 4, 4 + branch * 4)
    for stage, per_branch in enumerate(depths):
        for branch, depth in enumerate(per_branch):
            for index in range(depth):
                tensors[f"blocks.{stage}.blocks.{branch}.{index}.norm1.weight"] = _random(4)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def _metadata(model_type: str) -> ModelMetadata:
    return ModelMetadata(config={"model_type": model_type}, model_index={})


def test_support_claims_only_its_own_identity() -> None:
    assert describe(_metadata("crossvit_9_240")) is not None
    assert describe(_metadata("swin_tiny_patch4_window7_224")) is None


def test_layout_reads_depth_for_each_branch(tmp_path: Path) -> None:
    layout = model._layout(Checkpoint.open(_checkpoint(tmp_path, ((1, 3), (1, 3), (1, 3)))))
    assert layout["branches"] == 2
    assert layout["depths"] == [[1, 3], [1, 3], [1, 3]]


def test_layout_rejects_a_single_branch(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tensors = {"patch_embed.0.proj.weight": _random(4, 3, 4, 4)}
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="at least two branches"):
        model._layout(Checkpoint.open(tmp_path))


def test_layout_rejects_an_empty_branch(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tensors = {
        "patch_embed.0.proj.weight": _random(4, 3, 4, 4),
        "patch_embed.1.proj.weight": _random(4, 3, 8, 8),
        "blocks.0.blocks.0.0.norm1.weight": _random(4),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="is empty"):
        model._layout(Checkpoint.open(tmp_path))


def test_layout_rejects_non_contiguous_stage_indices(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tensors = {
        "patch_embed.0.proj.weight": _random(4, 3, 4, 4),
        "patch_embed.1.proj.weight": _random(4, 3, 8, 8),
    }
    for stage in (0, 2):
        for branch in range(2):
            tensors[f"blocks.{stage}.blocks.{branch}.0.norm1.weight"] = _random(4)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="stage indices are not contiguous"):
        model._layout(Checkpoint.open(tmp_path))


def test_head_count_is_a_documented_constant() -> None:
    """Every projection is square, so the head split leaves no trace."""
    assert model._NUM_HEADS == 4


def test_layer_norm_uses_the_crossvit_epsilon() -> None:
    assert model._LAYER_NORM_EPSILON == 1e-6


def test_read_config_rejects_another_family(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(
        json.dumps({"architecture": "swin_tiny_patch4_window7_224"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unsupported timm CrossViT model identity"):
        model._read_config(tmp_path)
