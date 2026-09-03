# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the timm SE-ResNet image-classification family plugin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.families.timm_seresnet import plugin
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires TensorRT", allow_module_level=True)


def _rand(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _bn(t: dict, prefix: str, ch: int) -> None:
    t[f"{prefix}.weight"] = _rand(ch)
    t[f"{prefix}.bias"] = _rand(ch)
    t[f"{prefix}.running_mean"] = _rand(ch)
    t[f"{prefix}.running_var"] = np.abs(_rand(ch)) + 1.0


def _write_tiny_seresnet(
    tmp_path: Path,
    *,
    blocks: tuple[int, int, int, int] = (1, 1, 1, 1),
    with_se: bool = True,
) -> None:
    """Write a miniature SE-ResNet bottleneck stack."""
    classes = 5
    config = {
        "architecture": "seresnet50",
        "num_classes": classes,
        "num_features": 256,
        "pretrained_cfg": {
            "input_size": [3, 224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_pct": 0.875,
            "interpolation": "bicubic",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    t: dict[str, np.ndarray] = {}
    stem = 16
    t["conv1.weight"] = _rand(stem, 3, 7, 7)
    _bn(t, "bn1", stem)

    in_ch = stem
    width = 8
    for stage, count in enumerate(blocks, start=1):
        out_ch = 32 << (stage - 1)
        for index in range(count):
            p = f"layer{stage}.{index}"
            t[f"{p}.conv1.weight"] = _rand(width, in_ch, 1, 1)
            _bn(t, f"{p}.bn1", width)
            t[f"{p}.conv2.weight"] = _rand(width, width, 3, 3)
            _bn(t, f"{p}.bn2", width)
            t[f"{p}.conv3.weight"] = _rand(out_ch, width, 1, 1)
            _bn(t, f"{p}.bn3", out_ch)
            if with_se:
                t[f"{p}.se.fc1.weight"] = _rand(4, out_ch, 1, 1)
                t[f"{p}.se.fc1.bias"] = _rand(4)
                t[f"{p}.se.fc2.weight"] = _rand(out_ch, 4, 1, 1)
                t[f"{p}.se.fc2.bias"] = _rand(out_ch)
            if in_ch != out_ch:
                t[f"{p}.downsample.0.weight"] = _rand(out_ch, in_ch, 1, 1)
                _bn(t, f"{p}.downsample.1", out_ch)
            in_ch = out_ch

    t["fc.weight"] = _rand(classes, in_ch)
    t["fc.bias"] = _rand(classes)
    save_file(t, str(tmp_path / "model.safetensors"))


def test_model_config_uses_timm_architecture_when_model_type_absent(tmp_path: Path):
    _write_tiny_seresnet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    assert cfg.model_type == "seresnet50"
    assert plugin.matches(cfg.model_type)


@pytest.mark.parametrize(
    "model_type", ["seresnet50", "seresnet152", "seresnext50_32x4d", "timm_seresnet"]
)
def test_plugin_matches_seresnet_variants(model_type: str):
    assert plugin.matches(model_type)


@pytest.mark.parametrize("model_type", ["resnet50", "resnext50_32x4d", "vgg16", ""])
def test_plugin_does_not_claim_plain_resnets(model_type: str):
    """timm_resnet owns the gate-free variants; the two families must not overlap."""
    assert not plugin.matches(model_type)


def test_bundle_config_reads_nested_pretrained_cfg(tmp_path: Path):
    _write_tiny_seresnet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    bundle_config = plugin.get_bundle_config_overrides(cfg)

    assert bundle_config["runtime_strategy"] == "timm_seresnet_image_classification"
    assert bundle_config["num_classes"] == 5


def test_layout_discovers_the_bottleneck_stack(tmp_path: Path):
    _write_tiny_seresnet(tmp_path, blocks=(2, 1, 1, 1))
    cfg = ModelConfig.from_dir(tmp_path)

    plugin.load_weights(str(tmp_path), cfg)
    layout = cfg.raw["_timm_seresnet_config"]

    assert layout["blocks"] == [2, 1, 1, 1]
    assert layout["bottleneck"] is True


def test_load_weights_reads_the_gate_projections(tmp_path: Path):
    _write_tiny_seresnet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    weights = plugin.load_weights(str(tmp_path), cfg)

    assert "layer1.0.se.fc1.weight" in weights
    assert "layer1.0.se.fc2.bias" in weights


def test_load_weights_rejects_a_checkpoint_without_a_gate(tmp_path: Path):
    """A plain ResNet must not be built here as if it were an SE-ResNet."""
    _write_tiny_seresnet(tmp_path, with_se=False)
    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="no squeeze-excite gate"):
        plugin.load_weights(str(tmp_path), cfg)


def test_build_engine_rejects_quantized_context(tmp_path: Path):
    _write_tiny_seresnet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)

    with pytest.raises(NotImplementedError, match="quantized"):
        plugin.build_engine(cfg, weights, 0, quant_ctx=object())
