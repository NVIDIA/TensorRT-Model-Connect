# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for timm HRNet builds."""

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

from families.timm_hrnet import model  # noqa: E402
from families.timm_hrnet.checkpoint import Checkpoint  # noqa: E402
from families.timm_hrnet.support import describe  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


def _random(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _norm(tensors: dict[str, np.ndarray], prefix: str, channels: int = 4) -> None:
    tensors[f"{prefix}.weight"] = _random(channels)
    tensors[f"{prefix}.bias"] = _random(channels)
    tensors[f"{prefix}.running_mean"] = _random(channels)
    tensors[f"{prefix}.running_var"] = np.abs(_random(channels)) + 1.0


def _grid(tmp_path: Path, branches: tuple[int, ...] = (2, 3, 4)) -> Path:
    """A checkpoint carrying only what `_layout` inspects."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, np.ndarray] = {"layer1.0.conv1.weight": _random(4, 4, 1, 1)}
    for stage_index, count in enumerate(branches, start=2):
        for branch in range(count):
            tensors[f"stage{stage_index}.0.branches.{branch}.0.conv1.weight"] = _random(4, 4, 3, 3)
    for index in range(branches[-1]):
        tensors[f"incre_modules.{index}.0.conv1.weight"] = _random(4, 4, 1, 1)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def _metadata(model_type: str) -> ModelMetadata:
    return ModelMetadata(config={"model_type": model_type}, model_index={})


def test_support_claims_only_its_own_identity() -> None:
    assert describe(_metadata("hrnet_w18")) is not None
    assert describe(_metadata("repvgg_a2")) is None


def test_layout_reads_the_branch_grid(tmp_path: Path) -> None:
    layout = model._layout(Checkpoint.open(_grid(tmp_path)))
    assert [stage["branches"] for stage in layout["stages"]] == [2, 3, 4]
    assert layout["layer1"] == 1


def test_layout_requires_the_head_to_match_the_final_branch_count(tmp_path: Path) -> None:
    """`incre_modules` feeds one bottleneck per surviving branch."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, np.ndarray] = {"layer1.0.conv1.weight": _random(4, 4, 1, 1)}
    for stage_index, count in enumerate((2, 3, 4), start=2):
        for branch in range(count):
            tensors[f"stage{stage_index}.0.branches.{branch}.0.conv1.weight"] = _random(4, 4, 3, 3)
    # Only three head modules for four branches.
    for index in range(3):
        tensors[f"incre_modules.{index}.0.conv1.weight"] = _random(4, 4, 1, 1)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="does not match the final"):
        model._layout(Checkpoint.open(tmp_path))


def test_layout_rejects_a_missing_stage(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tensors = {"layer1.0.conv1.weight": _random(4, 4, 1, 1)}
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="has no stage2 modules"):
        model._layout(Checkpoint.open(tmp_path))


def test_fold_absorbs_a_convolution_bias(tmp_path: Path) -> None:
    """HRNet's head convolutions carry their own bias; the fold must keep it.

    Dropping it shifts the logits by a constant per channel and still ranks
    plausibly.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, np.ndarray] = {
        "final_layer.0.weight": _random(4, 4, 1, 1),
        "final_layer.0.bias": _random(4),
    }
    _norm(tensors, "final_layer.1")
    save_file(tensors, str(tmp_path / "model.safetensors"))
    checkpoint = Checkpoint.open(tmp_path)
    _, bias = model._fold(checkpoint, "final_layer.0.weight", "final_layer.1", np.float32)
    gamma = checkpoint.tensor("final_layer.1.weight")
    beta = checkpoint.tensor("final_layer.1.bias")
    mean = checkpoint.tensor("final_layer.1.running_mean")
    variance = checkpoint.tensor("final_layer.1.running_var")
    scale = gamma / np.sqrt(variance + model._BATCH_NORM_EPSILON)
    expected = beta - mean * scale + checkpoint.tensor("final_layer.0.bias") * scale
    np.testing.assert_allclose(bias, expected, rtol=1e-6)


def test_read_config_rejects_another_family(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps({"architecture": "repvgg_a2"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported timm HRNet model identity"):
        model._read_config(tmp_path)
