# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the timm ResNet image-classification family plugin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.families.timm_resnet import plugin
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires TensorRT", allow_module_level=True)


def _rand(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _bn(tensors: dict, prefix: str, channels: int) -> None:
    tensors[f"{prefix}.weight"] = _rand(channels)
    tensors[f"{prefix}.bias"] = _rand(channels)
    tensors[f"{prefix}.running_mean"] = _rand(channels)
    # Variance must be positive: the builder folds 1/sqrt(var + eps).
    tensors[f"{prefix}.running_var"] = np.abs(_rand(channels)) + 1.0


def _write_tiny_resnet(
    tmp_path: Path,
    *,
    bottleneck: bool = True,
    blocks: tuple[int, int, int, int] = (1, 1, 1, 1),
    groups: int = 1,
) -> dict[str, np.ndarray]:
    """Write a miniature but structurally faithful timm ResNet checkpoint."""
    classes = 5
    width = 8 * groups
    config = {
        "architecture": "resnet50" if bottleneck else "resnet18",
        "num_classes": classes,
        "num_features": 256 if bottleneck else 64,
        "pretrained_cfg": {
            "input_size": [3, 224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_pct": 0.95,
            "interpolation": "bicubic",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    tensors: dict[str, np.ndarray] = {}
    stem = 16
    tensors["conv1.weight"] = _rand(stem, 3, 7, 7)
    _bn(tensors, "bn1", stem)

    in_ch = stem
    for stage_idx, count in enumerate(blocks):
        out_ch = (32 << stage_idx) if bottleneck else (16 << stage_idx)
        for block in range(count):
            prefix = f"layer{stage_idx + 1}.{block}"
            if bottleneck:
                tensors[f"{prefix}.conv1.weight"] = _rand(width, in_ch, 1, 1)
                _bn(tensors, f"{prefix}.bn1", width)
                # Grouped convs store (out, in/groups, kh, kw).
                tensors[f"{prefix}.conv2.weight"] = _rand(width, width // groups, 3, 3)
                _bn(tensors, f"{prefix}.bn2", width)
                tensors[f"{prefix}.conv3.weight"] = _rand(out_ch, width, 1, 1)
                _bn(tensors, f"{prefix}.bn3", out_ch)
            else:
                tensors[f"{prefix}.conv1.weight"] = _rand(out_ch, in_ch, 3, 3)
                _bn(tensors, f"{prefix}.bn1", out_ch)
                tensors[f"{prefix}.conv2.weight"] = _rand(out_ch, out_ch, 3, 3)
                _bn(tensors, f"{prefix}.bn2", out_ch)
            if in_ch != out_ch:
                tensors[f"{prefix}.downsample.0.weight"] = _rand(out_ch, in_ch, 1, 1)
                _bn(tensors, f"{prefix}.downsample.1", out_ch)
            in_ch = out_ch

    tensors["fc.weight"] = _rand(classes, in_ch)
    tensors["fc.bias"] = _rand(classes)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tensors


def test_model_config_uses_timm_architecture_when_model_type_absent(tmp_path: Path):
    _write_tiny_resnet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    # timm config.json carries "architecture", not "model_type".
    assert cfg.model_type == "resnet50"
    assert cfg.architectures == ["resnet50"]
    assert plugin.matches(cfg.model_type)


@pytest.mark.parametrize(
    "model_type",
    ["resnet18", "resnet101", "resnext50_32x4d", "wide_resnet50_2", "timm_resnet"],
)
def test_plugin_matches_resnet_variants(model_type: str):
    assert plugin.matches(model_type)


@pytest.mark.parametrize("model_type", ["vit_base_patch16_224", "llama", "qwen3_5", ""])
def test_plugin_rejects_unrelated_model_types(model_type: str):
    assert not plugin.matches(model_type)


def test_bundle_config_reads_nested_pretrained_cfg(tmp_path: Path):
    """timm nests preprocessing under pretrained_cfg; it must not be flattened away."""
    _write_tiny_resnet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    bundle_config = plugin.get_bundle_config_overrides(cfg)

    assert bundle_config["runtime_strategy"] == "timm_resnet_image_classification"
    assert bundle_config["input_image_h"] == 224
    assert bundle_config["input_image_w"] == 224
    assert bundle_config["num_classes"] == 5
    assert bundle_config["image_mean"] == [0.485, 0.456, 0.406]
    assert bundle_config["image_std"] == [0.229, 0.224, 0.225]
    assert bundle_config["crop_pct"] == pytest.approx(0.95)
    assert bundle_config["interpolation"] == "bicubic"


def test_load_weights_discovers_bottleneck_layout(tmp_path: Path):
    raw = _write_tiny_resnet(tmp_path, bottleneck=True, blocks=(2, 1, 1, 1))
    cfg = ModelConfig.from_dir(tmp_path)

    weights = plugin.load_weights(str(tmp_path), cfg)
    layout = cfg.raw["_timm_resnet_config"]

    assert layout["bottleneck"] is True
    assert layout["blocks"] == [2, 1, 1, 1]
    assert weights["conv1.weight"].shape == raw["conv1.weight"].shape
    # BN statistics stay fp32 so the host-side fold keeps its precision.
    assert weights["bn1.running_var"].dtype == np.float32


def test_load_weights_discovers_basic_block_layout(tmp_path: Path):
    _write_tiny_resnet(tmp_path, bottleneck=False, blocks=(2, 2, 2, 2))
    cfg = ModelConfig.from_dir(tmp_path)

    plugin.load_weights(str(tmp_path), cfg)
    layout = cfg.raw["_timm_resnet_config"]

    assert layout["bottleneck"] is False
    assert layout["blocks"] == [2, 2, 2, 2]


def test_load_weights_rejects_checkpoint_missing_a_stage(tmp_path: Path):
    _write_tiny_resnet(tmp_path)
    tensors = {}
    from safetensors.numpy import load_file

    for key, value in load_file(str(tmp_path / "model.safetensors")).items():
        if not key.startswith("layer4."):
            tensors[key] = value
    save_file(tensors, str(tmp_path / "model.safetensors"))
    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="no blocks for stage layer4"):
        plugin.load_weights(str(tmp_path), cfg)


def test_build_engine_rejects_quantized_context(tmp_path: Path):
    _write_tiny_resnet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)

    with pytest.raises(NotImplementedError, match="quantized"):
        plugin.build_engine(cfg, weights, 0, quant_ctx=object())


def test_build_engine_rejects_input_not_divisible_by_32(tmp_path: Path):
    _write_tiny_resnet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)
    cfg.raw["_timm_resnet_config"]["image_size_h"] = 225

    with pytest.raises(ValueError, match="divisible by 32"):
        plugin.build_engine(cfg, weights, 0)
