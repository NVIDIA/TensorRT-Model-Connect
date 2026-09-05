# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the timm VGG image-classification family model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file
    from families.timm_vgg.config import ModelConfig
    from families.timm_vgg.model import _TimmVggModel, build as build_family
    from tensorrt_model_connect import BuildRequest
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires TensorRT", allow_module_level=True)


model = _TimmVggModel()


def _rand(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _write_tiny_vgg(
    tmp_path: Path,
    *,
    conv_indices: tuple[int, ...] = (0, 2, 5, 7, 10),
) -> dict[str, np.ndarray]:
    """Write a miniature VGG whose Sequential indices imply the pool positions."""
    classes = 5
    config = {
        "architecture": "vgg16",
        "num_classes": classes,
        "num_features": 16,
        "pretrained_cfg": {
            "input_size": [3, 224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_pct": 0.875,
            "crop_mode": "center",
            "interpolation": "bilinear",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    tensors: dict[str, np.ndarray] = {}
    in_ch = 3
    for index in conv_indices:
        out_ch = 8
        tensors[f"features.{index}.weight"] = _rand(out_ch, in_ch, 3, 3)
        tensors[f"features.{index}.bias"] = _rand(out_ch)
        in_ch = out_ch
    tensors["pre_logits.fc1.weight"] = _rand(16, in_ch, 7, 7)
    tensors["pre_logits.fc1.bias"] = _rand(16)
    tensors["pre_logits.fc2.weight"] = _rand(16, 16, 1, 1)
    tensors["pre_logits.fc2.bias"] = _rand(16)
    tensors["head.fc.weight"] = _rand(classes, 16)
    tensors["head.fc.bias"] = _rand(classes)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tensors


def test_model_config_uses_timm_architecture_when_model_type_absent(tmp_path: Path):
    _write_tiny_vgg(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    assert cfg.model_type == "vgg16"


def test_model_config_requires_the_exact_config_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        ModelConfig.from_dir(tmp_path)


def test_bundle_config_reads_nested_pretrained_cfg(tmp_path: Path):
    _write_tiny_vgg(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    bundle_config = model.get_bundle_config_overrides(cfg)

    assert bundle_config["input_image_h"] == 224
    assert bundle_config["num_classes"] == 5
    assert bundle_config["image_mean"] == [0.485, 0.456, 0.406]
    assert bundle_config["crop_pct"] == pytest.approx(0.875)
    assert bundle_config["interpolation"] == "bilinear"


def test_bundle_config_rejects_missing_pretrained_contract(tmp_path: Path):
    _write_tiny_vgg(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("pretrained_cfg")
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="requires pretrained_cfg"):
        model.get_bundle_config_overrides(ModelConfig.from_dir(tmp_path))


def test_layout_places_a_pool_where_the_sequential_index_jumps(tmp_path: Path):
    """A gap larger than conv+ReLU means torchvision put a MaxPool there."""
    _write_tiny_vgg(tmp_path, conv_indices=(0, 2, 5, 7, 10))
    cfg = ModelConfig.from_dir(tmp_path)

    model.load_weights(str(tmp_path), cfg, precision="fp32")
    layout = cfg.raw["_timm_vgg_config"]

    # 0->2 is conv+ReLU (no pool); 2->5, 5->7 no, 7->10 yes; plus the closing pool.
    assert layout["layers"] == [
        ("conv", 0),
        ("conv", 2),
        ("pool", -1),
        ("conv", 5),
        ("conv", 7),
        ("pool", -1),
        ("conv", 10),
        ("pool", -1),
    ]
    assert layout["num_pools"] == 3


def test_load_weights_reads_every_vgg_tensor_in_requested_precision(tmp_path: Path):
    expected = _write_tiny_vgg(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    weights = model.load_weights(str(tmp_path), cfg, precision="fp16")

    assert set(weights) == set(expected)
    for name, value in weights.items():
        assert value.shape == expected[name].shape
        assert value.dtype == np.float16


def test_load_weights_rejects_checkpoint_without_convolutions(tmp_path: Path):
    _write_tiny_vgg(tmp_path)
    from safetensors.numpy import load_file

    kept = {
        k: v
        for k, v in load_file(str(tmp_path / "model.safetensors")).items()
        if not k.startswith("features.")
    }
    save_file(kept, str(tmp_path / "model.safetensors"))
    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="no features"):
        model.load_weights(str(tmp_path), cfg, precision="fp32")


def test_load_weights_rejects_checkpoint_without_the_exact_head(tmp_path: Path):
    _write_tiny_vgg(tmp_path)
    from safetensors.numpy import load_file

    tensors = dict(load_file(str(tmp_path / "model.safetensors")))
    tensors.pop("head.fc.bias")
    save_file(tensors, str(tmp_path / "model.safetensors"))
    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(KeyError, match="head.fc.bias"):
        model.load_weights(str(tmp_path), cfg, precision="fp32")


def test_build_rejects_batch_norm_variant_the_graph_does_not_implement(tmp_path: Path):
    _write_tiny_vgg(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["architecture"] = "vgg16_bn"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "unused.bundle",
        family="timm_vgg",
        task="classification",
        precision="fp16",
        max_sequence_length=1,
    )

    with pytest.raises(ValueError, match="model_type='vgg16_bn'"):
        build_family(request, object())


def test_build_rejects_quantization(tmp_path: Path):
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "unused.bundle",
        family="timm_vgg",
        task="classification",
        precision="fp16",
        quantization="fp8",
    )

    with pytest.raises(NotImplementedError, match="quantization"):
        build_family(request, object())


def test_build_engine_rejects_input_not_divisible_by_the_pool_count(tmp_path: Path):
    _write_tiny_vgg(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)
    weights = model.load_weights(str(tmp_path), cfg, precision="fp32")
    cfg.raw["_timm_vgg_config"]["image_size_h"] = 225

    with pytest.raises(ValueError, match="divisible by 8"):
        model.build_engine(cfg, weights, precision="fp32")
