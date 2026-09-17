# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for timm ResNeSt builds."""

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

from families.timm_resnest import model  # noqa: E402
from families.timm_resnest.checkpoint import Checkpoint  # noqa: E402
from families.timm_resnest.support import describe  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


def _random(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _norm(tensors: dict[str, np.ndarray], prefix: str, channels: int) -> None:
    tensors[f"{prefix}.weight"] = _random(channels)
    tensors[f"{prefix}.bias"] = _random(channels)
    tensors[f"{prefix}.running_mean"] = _random(channels)
    tensors[f"{prefix}.running_var"] = np.abs(_random(channels)) + 1.0


def _checkpoint(tmp_path: Path, depths: tuple[int, ...] = (1, 1, 1, 1)) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = {
        "architecture": "resnest50d",
        "num_classes": 5,
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
    tensors["conv1.0.weight"] = _random(8, 3, 3, 3)
    _norm(tensors, "conv1.1", 8)
    tensors["conv1.3.weight"] = _random(8, 8, 3, 3)
    _norm(tensors, "bn1", 8)
    for stage, depth in zip(("layer1", "layer2", "layer3", "layer4"), depths):
        for index in range(depth):
            prefix = f"{stage}.{index}"
            tensors[f"{prefix}.conv1.weight"] = _random(8, 8, 1, 1)
            _norm(tensors, f"{prefix}.bn1", 8)
            # Split attention: radix 2 over 8 output channels, cardinality 1.
            tensors[f"{prefix}.conv2.conv.weight"] = _random(16, 8, 3, 3)
            _norm(tensors, f"{prefix}.conv2.bn0", 16)
            tensors[f"{prefix}.conv2.fc1.weight"] = _random(4, 8, 1, 1)
            tensors[f"{prefix}.conv2.fc1.bias"] = _random(4)
            _norm(tensors, f"{prefix}.conv2.bn1", 4)
            tensors[f"{prefix}.conv2.fc2.weight"] = _random(16, 4, 1, 1)
            tensors[f"{prefix}.conv2.fc2.bias"] = _random(16)
            tensors[f"{prefix}.conv3.weight"] = _random(8, 8, 1, 1)
            _norm(tensors, f"{prefix}.bn3", 8)
            tensors[f"{prefix}.downsample.1.weight"] = _random(8, 8, 1, 1)
            _norm(tensors, f"{prefix}.downsample.2", 8)
    tensors["fc.weight"] = _random(5, 8)
    tensors["fc.bias"] = _random(5)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def _metadata(model_type: str) -> ModelMetadata:
    return ModelMetadata(config={"model_type": model_type}, model_index={})


def test_support_claims_only_its_own_identity() -> None:
    assert describe(_metadata("resnest50d")) is not None
    assert describe(_metadata("resnet50")) is None


def test_layout_reads_depth_per_stage(tmp_path: Path) -> None:
    assert model._layout(Checkpoint.open(_checkpoint(tmp_path, (3, 4, 6, 3)))) == [3, 4, 6, 3]


def test_layout_requires_a_deep_stem(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, np.ndarray] = {}
    for stage in ("layer1", "layer2", "layer3", "layer4"):
        tensors[f"{stage}.0.conv1.weight"] = _random(4, 4, 1, 1)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="no deep stem"):
        model._layout(Checkpoint.open(tmp_path))


def test_weights_pick_stem_convolutions_by_rank(tmp_path: Path) -> None:
    """Norms interleaved in the stem Sequential also carry a `.weight`."""
    checkpoint = Checkpoint.open(_checkpoint(tmp_path))
    weights = model._weights(checkpoint, model._layout(checkpoint), np.float32)
    assert int(weights["stem_depth"]) == 2
    for position in range(2):
        assert weights[f"stem.{position}.weight"].ndim == 4


def test_fold_absorbs_a_convolution_bias(tmp_path: Path) -> None:
    """The split-attention `fc1` carries a bias on top of the norm it feeds."""
    checkpoint = Checkpoint.open(_checkpoint(tmp_path))
    _, bias = model._fold_norm(
        checkpoint, "layer1.0.conv2.fc1.weight", "layer1.0.conv2.bn1", np.float32
    )
    gamma = checkpoint.tensor("layer1.0.conv2.bn1.weight")
    beta = checkpoint.tensor("layer1.0.conv2.bn1.bias")
    mean = checkpoint.tensor("layer1.0.conv2.bn1.running_mean")
    variance = checkpoint.tensor("layer1.0.conv2.bn1.running_var")
    scale = gamma / np.sqrt(variance + model._BATCH_NORM_EPSILON)
    expected = beta - mean * scale + checkpoint.tensor("layer1.0.conv2.fc1.bias") * scale
    np.testing.assert_allclose(bias, expected, rtol=1e-6)


def test_downsampling_constants_are_the_documented_values() -> None:
    """Neither is recoverable from weights; both still build when wrong."""
    assert model._AVERAGE_DOWN == {"kernel": 3, "stride": 2, "padding": 1}
    assert model._SHORTCUT_DOWN == {"kernel": 2, "stride": 2, "padding": 0}


def test_read_config_rejects_another_family(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps({"architecture": "resnet50"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported timm ResNeSt model identity"):
        model._read_config(tmp_path)


@pytest.mark.parametrize("metadata", [
    {},
    {"vocabulary_id": "test:five-classes",
     "label_names": ["first", "second", "third", "fourth", "fifth"]},
])
def test_build_exports_semantic_task_and_complete_class_metadata(tmp_path: Path, monkeypatch, metadata):
    from tensorrt_model_connect.build import BuildRequest
    _checkpoint(tmp_path)
    monkeypatch.setattr(model, "_build_engine",
                        lambda raw, *_: (b"plan", model._preprocess_config(raw)))
    build_family = model.build
    config_path = tmp_path / "config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw.update(metadata)
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    sections = {}

    class Writer:
        def set_header(self, **header):
            sections["header"] = header

        def add_bytes(self, name, value):
            sections[name] = value

        def add_json(self, name, value):
            sections[name] = value

    request = BuildRequest(model_dir=tmp_path, output_path=tmp_path / "unused.bundle",
                           family="timm_resnest", task="image_to_class_scores", precision="fp32")
    build_family(request, Writer())
    assert sections["header"]["task"] == "image_to_class_scores"
    assert sections["engine.plan"] == b"plan"
    assert sections["runtime.json"]["num_classes"] == 5
    assert sections["runtime.json"]["vocabulary_id"] == metadata.get("vocabulary_id", "")
    assert sections["runtime.json"]["labels"] == metadata.get("label_names", [])


@pytest.mark.parametrize("labels", [["only one"], ["one", "two", "", "four", "five"], 5])
def test_build_rejects_incomplete_class_labels(tmp_path: Path, monkeypatch, labels):
    from tensorrt_model_connect.build import BuildRequest
    _checkpoint(tmp_path)
    monkeypatch.setattr(model, "_build_engine",
                        lambda raw, *_: (b"plan", model._preprocess_config(raw)))
    build_family = model.build
    config_path = tmp_path / "config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["label_names"] = labels
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    request = BuildRequest(model_dir=tmp_path, output_path=tmp_path / "unused.bundle",
                           family="timm_resnest", task="image_to_class_scores", precision="fp32")
    with pytest.raises(ValueError, match="label_names must name every class"):
        build_family(request, object())


def test_support_exposes_only_semantic_task():
    from families.timm_resnest.support import describe
    from tensorrt_model_connect.model_support import ModelMetadata

    support = describe(ModelMetadata(config={"model_type": "timm_resnest"}, model_index={}))
    assert support is not None
    assert support.tasks == ("image_to_class_scores",)
    assert support.default_task == "image_to_class_scores"
