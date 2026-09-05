# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for fused timm RepVGG builds."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file


try:
    import tensorrt  # noqa: F401
except ModuleNotFoundError:
    sys.modules["tensorrt"] = SimpleNamespace()

from families.timm_repvgg import model  # noqa: E402
from families.timm_repvgg.checkpoint import Checkpoint  # noqa: E402
from families.timm_repvgg.support import describe  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


def _random(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _batch_norm(tensors: dict[str, np.ndarray], prefix: str, channels: int) -> None:
    tensors[f"{prefix}.weight"] = _random(channels)
    tensors[f"{prefix}.bias"] = _random(channels)
    tensors[f"{prefix}.running_mean"] = _random(channels)
    tensors[f"{prefix}.running_var"] = np.abs(_random(channels)) + 1.0


def _branch(
    tensors: dict[str, np.ndarray],
    prefix: str,
    output_channels: int,
    input_channels: int,
    kernel: int,
) -> None:
    tensors[f"{prefix}.conv.weight"] = _random(output_channels, input_channels, kernel, kernel)
    _batch_norm(tensors, f"{prefix}.bn", output_channels)


def _checkpoint(tmp_path: Path, blocks: tuple[int, ...] = (2, 1)) -> Path:
    config = {
        "architecture": "repvgg_a2",
        "num_classes": 5,
        "num_features": 16,
        "pretrained_cfg": {
            "input_size": [3, 224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_pct": 0.875,
            "interpolation": "bilinear",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors: dict[str, np.ndarray] = {}
    _branch(tensors, "stem.conv_kxk", 8, 3, 3)
    _branch(tensors, "stem.conv_1x1", 8, 3, 1)
    channels = 8
    for stage, count in enumerate(blocks):
        for index in range(count):
            prefix = f"stages.{stage}.{index}"
            input_channels = channels if index == 0 else 16
            _branch(tensors, f"{prefix}.conv_kxk", 16, input_channels, 3)
            _branch(tensors, f"{prefix}.conv_1x1", 16, input_channels, 1)
            if index > 0:
                _batch_norm(tensors, f"{prefix}.identity", 16)
        channels = 16
    tensors["head.fc.weight"] = _random(5, channels)
    tensors["head.fc.bias"] = _random(5)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def test_support_owns_only_the_exact_repvgg_identity() -> None:
    assert describe(ModelMetadata({"architecture": "repvgg_a2"}, {})) is not None
    assert describe(ModelMetadata({"architecture": "resnet50"}, {})) is None


def test_layout_derives_stride_from_identity_branch(tmp_path: Path) -> None:
    checkpoint = Checkpoint.open(_checkpoint(tmp_path, (2, 1)))

    blocks = model._layout(checkpoint)

    assert [block["has_identity"] for block in blocks] == [False, True, False]
    assert [block["stride"] for block in blocks] == [2, 1, 2]


def test_reparameterization_fuses_every_training_branch(tmp_path: Path) -> None:
    _checkpoint(tmp_path, (2,))
    tensors = load_file(str(tmp_path / "model.safetensors"))
    prefix = "stages.0.1"
    for branch, kernel in (("conv_kxk", 3), ("conv_1x1", 1)):
        tensors[f"{prefix}.{branch}.conv.weight"] = np.zeros((16, 16, kernel, kernel))
        tensors[f"{prefix}.{branch}.bn.weight"] = np.zeros(16)
        tensors[f"{prefix}.{branch}.bn.bias"] = np.zeros(16)
        tensors[f"{prefix}.{branch}.bn.running_mean"] = np.zeros(16)
        tensors[f"{prefix}.{branch}.bn.running_var"] = np.ones(16)
    tensors[f"{prefix}.identity.weight"] = np.ones(16)
    tensors[f"{prefix}.identity.bias"] = np.zeros(16)
    tensors[f"{prefix}.identity.running_mean"] = np.zeros(16)
    tensors[f"{prefix}.identity.running_var"] = np.ones(16)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    checkpoint = Checkpoint.open(tmp_path)

    weight, bias = model._fused_block(checkpoint, prefix, True, np.float32)

    expected = np.zeros_like(weight)
    indices = np.arange(16)
    expected[indices, indices, 1, 1] = 1.0
    np.testing.assert_allclose(weight, expected, atol=1e-4)
    np.testing.assert_allclose(bias, np.zeros(16), atol=1e-6)


def test_layout_rejects_a_missing_branch(tmp_path: Path) -> None:
    _checkpoint(tmp_path)
    tensors = {
        name: value
        for name, value in load_file(str(tmp_path / "model.safetensors")).items()
        if not name.startswith("stages.0.0.conv_kxk")
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))

    with pytest.raises(ValueError, match="missing a convolution branch"):
        model._layout(Checkpoint.open(tmp_path))


def test_plain_build_publishes_abstract_classification_bundle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _checkpoint(tmp_path)
    monkeypatch.setattr(
        model,
        "_build_engine",
        lambda raw, checkpoint, precision, verbose: (
            b"plan",
            model._preprocess_config(raw),
        ),
    )

    class Writer:
        def __init__(self) -> None:
            self.header = None
            self.sections = {}

        def set_header(self, **value) -> None:
            self.header = value

        def add_bytes(self, name, value) -> None:
            self.sections[name] = value

        def add_json(self, name, value) -> None:
            self.sections[name] = value

    request = SimpleNamespace(
        model_dir=tmp_path,
        backend="trt",
        dynamic_kv_cache=False,
        family="timm_repvgg",
        task="classification",
        precision="fp16",
        max_sequence_length=1,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        tensor_parallel_size=1,
        context_parallel_size=1,
        quantization=None,
        fp32_layers=(),
        verbose=False,
    )
    writer = Writer()

    model.build(request, writer)

    assert writer.header == {
        "family": "timm_repvgg",
        "task": "classification",
        "backend": "trt",
    }
    assert writer.sections["engine.plan"] == b"plan"
    assert writer.sections["runtime.json"]["input_image_h"] == 224
