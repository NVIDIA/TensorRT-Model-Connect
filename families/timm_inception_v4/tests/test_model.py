# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for timm Inception-v4 builds."""

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

from families.timm_inception_v4 import model  # noqa: E402
from families.timm_inception_v4.checkpoint import Checkpoint  # noqa: E402
from families.timm_inception_v4.support import describe  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


def _random(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _conv_norm(tensors: dict[str, np.ndarray], prefix: str, out_ch: int = 4, kernel: int = 3):
    tensors[f"{prefix}.conv.weight"] = _random(out_ch, 4, kernel, kernel)
    tensors[f"{prefix}.bn.weight"] = _random(out_ch)
    tensors[f"{prefix}.bn.bias"] = _random(out_ch)
    tensors[f"{prefix}.bn.running_mean"] = _random(out_ch)
    tensors[f"{prefix}.bn.running_var"] = np.abs(_random(out_ch)) + 1.0


def _metadata(model_type: str) -> ModelMetadata:
    return ModelMetadata(config={"model_type": model_type}, model_index={})


def test_support_claims_only_its_own_identity() -> None:
    assert describe(_metadata("inception_v4")) is not None
    assert describe(_metadata("repvgg_a2")) is None


def test_classify_names_each_topology_from_its_keys() -> None:
    assert model._classify(3, {"conv"}, set()) == "mixed3a"
    assert model._classify(9, {"conv"}, set()) == "mixed5a"
    assert model._classify(12, {"branch0", "branch1_0"}, set()) == "inception_c"
    assert (
        model._classify(5, {"branch0", "branch1", "branch2", "branch3"}, set())
        == "inception_ab"
    )
    assert (
        model._classify(8, {"branch0", "branch1"}, {"branch0.conv.weight"}) == "reduction_a"
    )
    assert (
        model._classify(4, {"branch0", "branch1"}, {"branch0.0.conv.weight"}) == "chained_pair"
    )


def test_classify_rejects_an_unknown_topology() -> None:
    with pytest.raises(ValueError, match="unrecognised Inception-v4 block topology"):
        model._classify(7, {"mystery"}, set())


def test_layout_orders_the_two_indistinguishable_blocks(tmp_path: Path) -> None:
    """Mixed4a and Reduction-B differ only in stride, which weights never record.

    Their order is the only honest discriminator: the first is Mixed4a and any
    later one is Reduction-B.
    """
    tensors: dict[str, np.ndarray] = {}
    for index in range(3):
        _conv_norm(tensors, f"features.{index}")
    for index in (3, 4):
        for branch in ("branch0", "branch1"):
            _conv_norm(tensors, f"features.{index}.{branch}.0")
            _conv_norm(tensors, f"features.{index}.{branch}.1")
    tensors["last_linear.weight"] = _random(5, 4)
    tensors["last_linear.bias"] = _random(5)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    kinds = [block["kind"] for block in model._layout(Checkpoint.open(tmp_path))]
    assert kinds == ["stem", "stem", "stem", "mixed4a", "reduction_b"]


def test_layout_rejects_non_contiguous_block_indices(tmp_path: Path) -> None:
    tensors: dict[str, np.ndarray] = {}
    _conv_norm(tensors, "features.0")
    _conv_norm(tensors, "features.2")
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="not contiguous"):
        model._layout(Checkpoint.open(tmp_path))


def test_fold_uses_the_tensorflow_epsilon() -> None:
    """Inception-v4 is a TensorFlow port; the PyTorch default changes the argmax."""
    assert model._BATCH_NORM_EPSILON == 1e-3


def test_fold_reproduces_the_batch_norm_it_replaces(tmp_path: Path) -> None:
    tensors: dict[str, np.ndarray] = {}
    _conv_norm(tensors, "features.0")
    save_file(tensors, str(tmp_path / "model.safetensors"))
    checkpoint = Checkpoint.open(tmp_path)
    weight, bias = model._fold(checkpoint, "features.0", np.float32)
    raw = checkpoint.tensor("features.0.conv.weight")
    gamma = checkpoint.tensor("features.0.bn.weight")
    beta = checkpoint.tensor("features.0.bn.bias")
    mean = checkpoint.tensor("features.0.bn.running_mean")
    variance = checkpoint.tensor("features.0.bn.running_var")
    scale = gamma / np.sqrt(variance + model._BATCH_NORM_EPSILON)
    np.testing.assert_allclose(weight, raw * scale.reshape(-1, 1, 1, 1), rtol=1e-6)
    np.testing.assert_allclose(bias, beta - mean * scale, rtol=1e-6)


def test_read_config_rejects_another_family(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"architecture": "repvgg_a2"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported timm Inception-v4 model identity"):
        model._read_config(tmp_path)
