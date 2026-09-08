# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for timm Xception builds."""

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

from families.timm_xception import model  # noqa: E402
from families.timm_xception.checkpoint import Checkpoint  # noqa: E402
from families.timm_xception.support import describe  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


def _random(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _batch_norm(tensors: dict[str, np.ndarray], prefix: str, channels: int) -> None:
    tensors[f"{prefix}.weight"] = _random(channels)
    tensors[f"{prefix}.bias"] = _random(channels)
    tensors[f"{prefix}.running_mean"] = _random(channels)
    tensors[f"{prefix}.running_var"] = np.abs(_random(channels)) + 1.0


def _conv_bn(tensors: dict[str, np.ndarray], prefix: str, out_ch: int, in_ch: int, kernel: int):
    tensors[f"{prefix}.conv.weight"] = _random(out_ch, in_ch, kernel, kernel)
    _batch_norm(tensors, f"{prefix}.bn", out_ch)


def _separable(tensors: dict[str, np.ndarray], prefix: str, out_ch: int, in_ch: int) -> None:
    tensors[f"{prefix}.conv_dw.weight"] = _random(in_ch, 1, 3, 3)
    _batch_norm(tensors, f"{prefix}.bn_dw", in_ch)
    tensors[f"{prefix}.conv_pw.weight"] = _random(out_ch, in_ch, 1, 1)
    _batch_norm(tensors, f"{prefix}.bn_pw", out_ch)


def _checkpoint(tmp_path: Path, shortcuts: tuple[bool, ...] = (True, False, True)) -> Path:
    config = {
        "architecture": "xception41",
        "num_classes": 5,
        "pretrained_cfg": {
            "input_size": [3, 299, 299],
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "crop_pct": 0.903,
            "interpolation": "bicubic",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors: dict[str, np.ndarray] = {}
    _conv_bn(tensors, "stem.0", 8, 3, 3)
    _conv_bn(tensors, "stem.1", 8, 8, 3)
    channels = 8
    for index, has_shortcut in enumerate(shortcuts):
        prefix = f"blocks.{index}"
        for leaf in ("conv1", "conv2", "conv3"):
            _separable(tensors, f"{prefix}.stack.{leaf}", channels, channels)
        if has_shortcut:
            _conv_bn(tensors, f"{prefix}.shortcut", channels, channels, 1)
    tensors["head.fc.weight"] = _random(5, channels)
    tensors["head.fc.bias"] = _random(5)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def _metadata(model_type: str) -> ModelMetadata:
    return ModelMetadata(config={"model_type": model_type}, model_index={})


def test_support_claims_only_its_own_identity() -> None:
    """Support matches exact identity, so a sibling family must not match."""
    assert describe(_metadata("xception41")) is not None
    assert describe(_metadata("repvgg_a2")) is None


def test_layout_takes_stride_from_the_projection_shortcut(tmp_path: Path) -> None:
    """Xception downsamples exactly in the blocks that project."""
    blocks = model._layout(Checkpoint.open(_checkpoint(tmp_path, (True, False, True))))
    assert [block["stride"] for block in blocks] == [2, 1, 2]
    assert [block["has_shortcut"] for block in blocks] == [True, False, True]


def test_layout_marks_only_the_last_block_as_the_exit(tmp_path: Path) -> None:
    """The exit block is the one thing the weights cannot tell us.

    It inverts the activation placement and drops the residual, so mislabelling
    it still builds and still produces a plausible answer.
    """
    blocks = model._layout(Checkpoint.open(_checkpoint(tmp_path, (True, False, True))))
    assert [block["is_exit"] for block in blocks] == [False, False, True]


def test_layout_rejects_a_block_without_a_separable_stack(tmp_path: Path) -> None:
    tensors = {
        "blocks.0.shortcut.conv.weight": _random(4, 4, 1, 1),
        "head.fc.weight": _random(5, 4),
        "head.fc.bias": _random(5),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="no separable convolution stack"):
        model._layout(Checkpoint.open(tmp_path))


def test_layout_rejects_non_contiguous_block_indices(tmp_path: Path) -> None:
    tensors: dict[str, np.ndarray] = {}
    for index in (0, 2):
        _separable(tensors, f"blocks.{index}.stack.conv1", 4, 4)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="not contiguous"):
        model._layout(Checkpoint.open(tmp_path))


def test_fold_reproduces_the_batch_norm_it_replaces(tmp_path: Path) -> None:
    """A folded convolution must equal convolution-then-norm on real numbers."""
    checkpoint = Checkpoint.open(_checkpoint(tmp_path))
    weight, bias = model._fold(checkpoint, "stem.0.conv.weight", "stem.0.bn", np.float32)
    raw = checkpoint.tensor("stem.0.conv.weight")
    gamma = checkpoint.tensor("stem.0.bn.weight")
    beta = checkpoint.tensor("stem.0.bn.bias")
    mean = checkpoint.tensor("stem.0.bn.running_mean")
    variance = checkpoint.tensor("stem.0.bn.running_var")
    scale = gamma / np.sqrt(variance + model._BATCH_NORM_EPSILON)
    np.testing.assert_allclose(weight, raw * scale.reshape(-1, 1, 1, 1), rtol=1e-6)
    np.testing.assert_allclose(bias, beta - mean * scale, rtol=1e-6)


def test_fold_uses_the_tensorflow_epsilon() -> None:
    """Xception is a TensorFlow port; the PyTorch default changes the argmax."""
    assert model._BATCH_NORM_EPSILON == 1e-3


def test_preprocess_config_rejects_a_degenerate_std() -> None:
    raw = {
        "num_classes": 5,
        "pretrained_cfg": {
            "input_size": [3, 299, 299],
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.0, 0.5],
            "crop_pct": 0.903,
            "interpolation": "bicubic",
        },
    }
    with pytest.raises(ValueError, match="preprocessing or classifier config is invalid"):
        model._preprocess_config(raw)


def test_read_config_rejects_another_family(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"architecture": "repvgg_a2"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported timm Xception model identity"):
        model._read_config(tmp_path)
