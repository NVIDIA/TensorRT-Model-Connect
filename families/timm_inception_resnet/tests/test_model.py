# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for timm Inception-ResNet-v2 builds."""

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

from families.timm_inception_resnet import model  # noqa: E402
from families.timm_inception_resnet.checkpoint import Checkpoint  # noqa: E402
from families.timm_inception_resnet.support import describe  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


def _random(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _conv_norm(tensors: dict[str, np.ndarray], prefix: str, out_ch: int = 4) -> None:
    tensors[f"{prefix}.conv.weight"] = _random(out_ch, 4, 3, 3)
    tensors[f"{prefix}.bn.weight"] = _random(out_ch)
    tensors[f"{prefix}.bn.bias"] = _random(out_ch)
    tensors[f"{prefix}.bn.running_mean"] = _random(out_ch)
    tensors[f"{prefix}.bn.running_var"] = np.abs(_random(out_ch)) + 1.0


def _checkpoint(tmp_path: Path, lengths: tuple[int, int, int] = (2, 3, 1)) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, np.ndarray] = {}
    for group, count in zip(("repeat", "repeat_1", "repeat_2"), lengths):
        for index in range(count):
            prefix = f"{group}.{index}"
            _conv_norm(tensors, f"{prefix}.branch0")
            tensors[f"{prefix}.conv2d.weight"] = _random(4, 4, 1, 1)
            tensors[f"{prefix}.conv2d.bias"] = _random(4)
    for required in ("mixed_5b", "mixed_6a", "mixed_7a", "block8", "conv2d_7b"):
        _conv_norm(tensors, f"{required}.branch0" if required.startswith("mixed") else required)
    tensors["block8.conv2d.weight"] = _random(4, 4, 1, 1)
    tensors["block8.conv2d.bias"] = _random(4)
    tensors["classif.weight"] = _random(5, 4)
    tensors["classif.bias"] = _random(5)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def _metadata(model_type: str) -> ModelMetadata:
    return ModelMetadata(config={"model_type": model_type}, model_index={})


def test_support_claims_only_its_own_identity() -> None:
    assert describe(_metadata("inception_resnet_v2")) is not None
    assert describe(_metadata("inception_v4")) is None


def test_layout_reads_each_repeat_group_length(tmp_path: Path) -> None:
    assert model._layout(Checkpoint.open(_checkpoint(tmp_path, (10, 20, 9)))) == {
        "repeat": 10,
        "repeat_1": 20,
        "repeat_2": 9,
    }


def test_layout_rejects_non_contiguous_repeat_indices(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, np.ndarray] = {}
    for index in (0, 2):
        _conv_norm(tensors, f"repeat.{index}.branch0")
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="not contiguous"):
        model._layout(Checkpoint.open(tmp_path))


def test_layout_requires_every_named_stage(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, np.ndarray] = {}
    for group in ("repeat", "repeat_1", "repeat_2"):
        _conv_norm(tensors, f"{group}.0.branch0")
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="missing mixed_5b"):
        model._layout(Checkpoint.open(tmp_path))


def test_residual_scales_are_the_documented_constants() -> None:
    """No checkpoint records these; a wrong value still builds and looks sane."""
    assert model._GROUP_SCALE == {"repeat": 0.17, "repeat_1": 0.10, "repeat_2": 0.20}
    assert model._FINAL_BLOCK_SCALE == 1.0


def test_fold_uses_the_tensorflow_epsilon() -> None:
    assert model._BATCH_NORM_EPSILON == 1e-3


def test_weights_load_the_projection_without_a_norm(tmp_path: Path) -> None:
    """The 1x1 projection is biased and has no batch norm, unlike the rest."""
    checkpoint = Checkpoint.open(_checkpoint(tmp_path))
    weights = model._weights(checkpoint, np.float32)
    np.testing.assert_allclose(
        weights["repeat.0.conv2d.weight"], checkpoint.tensor("repeat.0.conv2d.weight")
    )
    np.testing.assert_allclose(
        weights["repeat.0.conv2d.bias"], checkpoint.tensor("repeat.0.conv2d.bias")
    )


def test_read_config_rejects_another_family(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps({"architecture": "inception_v4"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported timm Inception-ResNet-v2 model identity"):
        model._read_config(tmp_path)
