# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for timm Swin Transformer builds."""

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

from families.timm_swin import model  # noqa: E402
from families.timm_swin.checkpoint import Checkpoint  # noqa: E402
from families.timm_swin.support import describe  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


def _random(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _blocks(tmp_path: Path, depths: tuple[int, ...] = (2, 2)) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, np.ndarray] = {}
    for layer, depth in enumerate(depths):
        for index in range(depth):
            tensors[f"layers.{layer}.blocks.{index}.norm1.weight"] = _random(4)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def _metadata(model_type: str) -> ModelMetadata:
    return ModelMetadata(config={"model_type": model_type}, model_index={})


def test_support_claims_only_its_own_identity() -> None:
    assert describe(_metadata("swin_tiny_patch4_window7_224")) is not None
    assert describe(_metadata("repvgg_a2")) is None


def test_layout_reads_depth_per_layer(tmp_path: Path) -> None:
    assert model._layout(Checkpoint.open(_blocks(tmp_path, (2, 2, 6, 2)))) == [2, 2, 6, 2]


def test_layout_rejects_non_contiguous_block_indices(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tensors = {
        "layers.0.blocks.0.norm1.weight": _random(4),
        "layers.0.blocks.2.norm1.weight": _random(4),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="not contiguous"):
        model._layout(Checkpoint.open(tmp_path))


def test_attention_bias_folds_the_relative_position_table() -> None:
    """Without a mask the bias is the gathered table, transposed to head-major."""
    heads, area = 2, 4
    table = np.arange(9 * heads, dtype=np.float32).reshape(9, heads)
    index = np.arange(area * area, dtype=np.int64).reshape(area, area) % 9
    bias = model._attention_bias(index, table, None, area, heads)
    assert bias.shape == (1, heads, area, area)
    for head in range(heads):
        np.testing.assert_allclose(bias[0, head], table[index, head])


def test_attention_bias_pushes_masked_positions_below_the_real_scores() -> None:
    """The cyclic roll wraps unrelated positions into one window.

    Those entries must be driven far below the real scores, or edge tokens
    attend to the opposite edge and the result is merely plausible.
    """
    heads, area, windows = 1, 2, 2
    table = np.zeros((4, heads), dtype=np.float32)
    index = np.zeros((area, area), dtype=np.int64)
    mask = np.array([[[0.0, 1.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
    bias = model._attention_bias(index, table, mask, area, heads)
    assert bias.shape == (windows, heads, area, area)
    assert bias[0, 0, 0, 1] == model._MASK_FILL
    assert bias[0, 0, 0, 0] == 0.0
    assert np.all(bias[1] == 0.0)


def test_layer_norm_uses_the_transformer_epsilon() -> None:
    assert model._LAYER_NORM_EPSILON == 1e-5


def test_read_config_rejects_another_family(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps({"architecture": "repvgg_a2"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported timm Swin Transformer model identity"):
        model._read_config(tmp_path)
