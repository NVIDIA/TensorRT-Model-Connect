# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the timm EfficientNet image-classification family plugin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.families.timm_efficientnet import plugin
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires TensorRT", allow_module_level=True)


def _rand(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _bn(t: dict, prefix: str, ch: int) -> None:
    t[f"{prefix}.weight"] = _rand(ch)
    t[f"{prefix}.bias"] = _rand(ch)
    t[f"{prefix}.running_mean"] = _rand(ch)
    t[f"{prefix}.running_var"] = np.abs(_rand(ch)) + 1.0


def _se(t: dict, prefix: str, ch: int, mid: int) -> None:
    t[f"{prefix}.se.conv_reduce.weight"] = _rand(mid, ch, 1, 1)
    t[f"{prefix}.se.conv_reduce.bias"] = _rand(mid)
    t[f"{prefix}.se.conv_expand.weight"] = _rand(ch, mid, 1, 1)
    t[f"{prefix}.se.conv_expand.bias"] = _rand(ch)


def _write_tiny_effnet(tmp_path: Path, *, stages: int = 7) -> None:
    """Write a miniature EfficientNet with one block per stage, all with SE."""
    classes = 5
    config = {
        "architecture": "efficientnet_b0",
        "num_classes": classes,
        "num_features": 16,
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
    ch = 8
    t["conv_stem.weight"] = _rand(ch, 3, 3, 3)
    _bn(t, "bn1", ch)

    for stage in range(stages):
        p = f"blocks.{stage}.0"
        if stage == 0:
            t[f"{p}.conv_dw.weight"] = _rand(ch, 1, 3, 3)
            _bn(t, f"{p}.bn1", ch)
            _se(t, p, ch, 4)
            t[f"{p}.conv_pw.weight"] = _rand(ch, ch, 1, 1)
            _bn(t, f"{p}.bn2", ch)
        else:
            mid = ch * 2
            t[f"{p}.conv_pw.weight"] = _rand(mid, ch, 1, 1)
            _bn(t, f"{p}.bn1", mid)
            t[f"{p}.conv_dw.weight"] = _rand(mid, 1, 3, 3)
            _bn(t, f"{p}.bn2", mid)
            _se(t, p, mid, 4)
            t[f"{p}.conv_pwl.weight"] = _rand(ch, mid, 1, 1)
            _bn(t, f"{p}.bn3", ch)

    t["conv_head.weight"] = _rand(16, ch, 1, 1)
    _bn(t, "bn2", 16)
    t["classifier.weight"] = _rand(classes, 16)
    t["classifier.bias"] = _rand(classes)
    save_file(t, str(tmp_path / "model.safetensors"))


def test_model_config_uses_timm_architecture_when_model_type_absent(tmp_path: Path):
    _write_tiny_effnet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    assert cfg.model_type == "efficientnet_b0"
    assert plugin.matches(cfg.model_type)


@pytest.mark.parametrize(
    "model_type", ["efficientnet_b0", "efficientnet_b3", "timm_efficientnet"]
)
def test_plugin_matches_efficientnet_variants(model_type: str):
    assert plugin.matches(model_type)


@pytest.mark.parametrize(
    "model_type", ["resnet50", "vgg16", "mobilenetv3_large_100", ""]
)
def test_plugin_rejects_unrelated_model_types(model_type: str):
    assert not plugin.matches(model_type)


def test_bundle_config_reads_nested_pretrained_cfg(tmp_path: Path):
    _write_tiny_effnet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    bundle_config = plugin.get_bundle_config_overrides(cfg)

    assert bundle_config["runtime_strategy"] == "timm_efficientnet_image_classification"
    assert bundle_config["num_classes"] == 5
    assert bundle_config["interpolation"] == "bicubic"


def test_layout_classifies_blocks_and_finds_every_se_gate(tmp_path: Path):
    _write_tiny_effnet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    plugin.load_weights(str(tmp_path), cfg)
    blocks = cfg.raw["_timm_efficientnet_config"]["blocks"]

    assert blocks[0]["kind"] == "depthwise_separable"
    assert all(b["kind"] == "inverted_residual" for b in blocks[1:])
    # Every EfficientNet block carries a squeeze-excite gate.
    assert all(b["has_se"] for b in blocks)


def test_layout_applies_the_seven_stage_stride_schedule(tmp_path: Path):
    """Strides are not in the checkpoint; the activation is uniform SiLU."""
    _write_tiny_effnet(tmp_path, stages=7)
    cfg = ModelConfig.from_dir(tmp_path)

    plugin.load_weights(str(tmp_path), cfg)
    blocks = cfg.raw["_timm_efficientnet_config"]["blocks"]

    assert [b["stride"] for b in blocks] == [1, 2, 2, 2, 1, 2, 1]


def test_layout_rejects_a_stage_count_without_a_schedule(tmp_path: Path):
    _write_tiny_effnet(tmp_path, stages=4)
    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="schedule for 4 stages"):
        plugin.load_weights(str(tmp_path), cfg)


def test_build_engine_rejects_quantized_context(tmp_path: Path):
    _write_tiny_effnet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)

    with pytest.raises(NotImplementedError, match="quantized"):
        plugin.build_engine(cfg, weights, 0, quant_ctx=object())
