# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for timm DPN builds."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file


try:
    import tensorrt  # noqa: F401
except ModuleNotFoundError:
    sys.modules["tensorrt"] = SimpleNamespace()

from families.timm_dpn import model  # noqa: E402
from families.timm_dpn.checkpoint import Checkpoint  # noqa: E402
from families.timm_dpn.support import describe  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


def _random(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _norm(tensors: dict[str, np.ndarray], prefix: str, channels: int) -> None:
    tensors[f"{prefix}.weight"] = _random(channels)
    tensors[f"{prefix}.bias"] = _random(channels)
    tensors[f"{prefix}.running_mean"] = _random(channels)
    tensors[f"{prefix}.running_var"] = np.abs(_random(channels)) + 1.0


def _norm_conv(
    tensors: dict[str, np.ndarray], prefix: str, out_ch: int, in_ch: int, kernel: int, groups: int
) -> None:
    """DPN norms sit before their convolution, so the norm sizes the input."""
    _norm(tensors, f"{prefix}.bn", in_ch)
    tensors[f"{prefix}.conv.weight"] = _random(out_ch, in_ch // groups, kernel, kernel)


def _block(
    tensors: dict[str, np.ndarray],
    prefix: str,
    *,
    in_ch: int,
    residual: int,
    increment: int,
    projection: str | None,
    split_convs: bool,
    width: int = 6,
) -> None:
    if projection is not None:
        _norm_conv(tensors, f"{prefix}.{projection}", residual + 2 * increment, in_ch, 1, 1)
    _norm_conv(tensors, f"{prefix}.c1x1_a", width, in_ch, 1, 1)
    _norm_conv(tensors, f"{prefix}.c3x3_b", width, width, 3, 2)
    if split_convs:
        # The "b" widths end in two separate convolutions with no norm of
        # their own; the shared norm still sits in front of both.
        _norm(tensors, f"{prefix}.c1x1_c.bn", width)
        tensors[f"{prefix}.c1x1_c1.weight"] = _random(residual, width, 1, 1)
        tensors[f"{prefix}.c1x1_c2.weight"] = _random(increment, width, 1, 1)
    else:
        _norm_conv(
            tensors, f"{prefix}.c1x1_c", residual + increment, in_ch=width, kernel=1, groups=1
        )


def _checkpoint(tmp_path: Path, *, split_convs: bool = False) -> Path:
    """Build a two-stage DPN whose widths are small but self-consistent."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = {
        "architecture": "dpn68b" if split_convs else "dpn92",
        "num_classes": 5,
        "pretrained_cfg": {
            "input_size": [3, 224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_pct": 0.875,
            "interpolation": "bicubic",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    tensors: dict[str, np.ndarray] = {}
    tensors["features.conv1_1.conv.weight"] = _random(4, 3, 3, 3)
    _norm(tensors, "features.conv1_1.bn", 4)

    # Stage 2 keeps the resolution; stage 3 halves it.
    _block(
        tensors,
        "features.conv2_1",
        in_ch=4,
        residual=4,
        increment=2,
        projection="c1x1_w_s1",
        split_convs=split_convs,
    )
    _block(
        tensors,
        "features.conv2_2",
        in_ch=8,
        residual=4,
        increment=2,
        projection=None,
        split_convs=split_convs,
    )
    _block(
        tensors,
        "features.conv3_1",
        in_ch=10,
        residual=8,
        increment=3,
        projection="c1x1_w_s2",
        split_convs=split_convs,
    )
    _norm(tensors, "features.conv3_bn_ac.bn", 14)
    tensors["classifier.weight"] = _random(5, 14, 1, 1)
    tensors["classifier.bias"] = _random(5)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def _metadata(model_type: str) -> ModelMetadata:
    return ModelMetadata(config={"model_type": model_type}, model_index={})


def test_support_claims_only_its_own_identity() -> None:
    assert describe(_metadata("dpn68b")) is not None
    assert describe(_metadata("dpn131")) is not None
    assert describe(_metadata("repvgg_a2")) is None


def test_layout_numbers_blocks_from_the_stem_onwards(tmp_path: Path) -> None:
    """conv1_1 is the stem, so the dual-path stages start at 2."""
    blocks = model._layout(Checkpoint.open(_checkpoint(tmp_path)))
    assert [block["prefix"] for block in blocks] == [
        "features.conv2_1",
        "features.conv2_2",
        "features.conv3_1",
    ]


def test_layout_takes_each_stride_from_the_projection_name(tmp_path: Path) -> None:
    """c1x1_w_s1 keeps the resolution, c1x1_w_s2 halves it."""
    blocks = model._layout(Checkpoint.open(_checkpoint(tmp_path)))
    assert [block["stride"] for block in blocks] == [1, 1, 2]
    assert [block["projection"] for block in blocks] == ["c1x1_w_s1", None, "c1x1_w_s2"]


def test_layout_detects_the_two_output_convolution_widths(tmp_path: Path) -> None:
    plain = model._layout(Checkpoint.open(_checkpoint(tmp_path / "a", split_convs=False)))
    split = model._layout(Checkpoint.open(_checkpoint(tmp_path / "b", split_convs=True)))
    assert not any(block["split_convs"] for block in plain)
    assert all(block["split_convs"] for block in split)


def test_layout_rejects_a_projection_away_from_the_head_of_a_stage(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, np.ndarray] = {}
    _block(
        tensors,
        "features.conv2_1",
        in_ch=4,
        residual=4,
        increment=2,
        projection="c1x1_w_s1",
        split_convs=False,
    )
    _block(
        tensors,
        "features.conv2_2",
        in_ch=8,
        residual=4,
        increment=2,
        projection="c1x1_w_s1",
        split_convs=False,
    )
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="projects only if it heads its stage"):
        model._layout(Checkpoint.open(tmp_path))


def test_layout_rejects_a_block_missing_a_dual_path_convolution(tmp_path: Path) -> None:
    _norm_conv({}, "unused", 1, 1, 1, 1)
    tensors: dict[str, np.ndarray] = {}
    _norm_conv(tensors, "features.conv2_1.c1x1_a", 4, 4, 1, 1)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="missing a dual-path convolution"):
        model._layout(Checkpoint.open(tmp_path))


def test_residual_width_is_implicit_when_one_convolution_produces_both_paths(
    tmp_path: Path,
) -> None:
    """The projection emits residual + 2*increment, the output residual + increment."""
    checkpoint = Checkpoint.open(_checkpoint(tmp_path, split_convs=False))
    blocks = model._layout(checkpoint)
    weights = model._weights(checkpoint, blocks, np.float32)
    assert model._stage_residual_widths(blocks, weights) == {2: 4, 3: 8}


def test_residual_width_is_explicit_when_two_convolutions_produce_the_paths(
    tmp_path: Path,
) -> None:
    checkpoint = Checkpoint.open(_checkpoint(tmp_path, split_convs=True))
    blocks = model._layout(checkpoint)
    weights = model._weights(checkpoint, blocks, np.float32)
    assert model._stage_residual_widths(blocks, weights) == {2: 4, 3: 8}


def test_residual_width_rejects_a_stage_whose_blocks_disagree(tmp_path: Path) -> None:
    checkpoint = Checkpoint.open(_checkpoint(tmp_path, split_convs=True))
    blocks = model._layout(checkpoint)
    weights = model._weights(checkpoint, blocks, np.float32)
    # Move the split in the second block of stage 2 without moving the first.
    weights["features.conv2_2.c1x1_c1.weight"] = _random(3, 6, 1, 1)
    with pytest.raises(ValueError, match="disagrees with its stage residual width"):
        model._stage_residual_widths(blocks, weights)


def test_batch_norm_uses_the_timm_epsilon() -> None:
    """timm builds every DPN norm with eps=0.001, not the PyTorch default."""
    assert model._BATCH_NORM_EPSILON == 1e-3


def test_norm_reproduces_the_batch_norm_it_replaces(tmp_path: Path) -> None:
    checkpoint = Checkpoint.open(_checkpoint(tmp_path))
    scale, shift = model._norm(checkpoint, "features.conv1_1.bn", np.float32)
    gamma = checkpoint.tensor("features.conv1_1.bn.weight")
    beta = checkpoint.tensor("features.conv1_1.bn.bias")
    mean = checkpoint.tensor("features.conv1_1.bn.running_mean")
    variance = checkpoint.tensor("features.conv1_1.bn.running_var")
    expected = gamma / np.sqrt(variance + model._BATCH_NORM_EPSILON)
    np.testing.assert_allclose(scale, expected, rtol=1e-6)
    np.testing.assert_allclose(shift, beta - mean * expected, rtol=1e-6)


def test_preprocess_config_rejects_a_degenerate_std() -> None:
    raw = {
        "num_classes": 5,
        "pretrained_cfg": {
            "input_size": [3, 224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.0, 0.225],
            "crop_pct": 0.875,
            "interpolation": "bicubic",
        },
    }
    with pytest.raises(ValueError, match="preprocessing or classifier config is invalid"):
        model._preprocess_config(raw)


def test_read_config_rejects_another_family(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"architecture": "repvgg_a2"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unsupported timm DPN model identity"):
        model._read_config(tmp_path)
