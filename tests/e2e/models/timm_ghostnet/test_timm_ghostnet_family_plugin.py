# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the timm GhostNet image-classification family plugin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.families.timm_ghostnet import plugin
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires TensorRT", allow_module_level=True)


def _rand(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _bn(t: dict, prefix: str, ch: int) -> None:
    t[f"{prefix}.weight"] = _rand(ch)
    t[f"{prefix}.bias"] = _rand(ch)
    t[f"{prefix}.running_mean"] = _rand(ch)
    t[f"{prefix}.running_var"] = np.abs(_rand(ch)) + 1.0


def _ghost(t: dict, prefix: str, in_ch: int, half: int) -> None:
    """A Ghost module: pointwise primary branch plus a cheap depthwise branch."""
    t[f"{prefix}.primary_conv.0.weight"] = _rand(half, in_ch, 1, 1)
    _bn(t, f"{prefix}.primary_conv.1", half)
    t[f"{prefix}.cheap_operation.0.weight"] = _rand(half, 1, 3, 3)
    _bn(t, f"{prefix}.cheap_operation.1", half)


def _write_tiny_ghostnet(tmp_path: Path, *, strided_stage: int = 1) -> None:
    """Write a miniature GhostNet: two bottlenecks plus a final conv block."""
    classes = 5
    config = {
        "architecture": "ghostnet_100",
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

    half = 4
    mid = half * 2
    for stage in (0, 1):
        p = f"blocks.{stage}.0"
        _ghost(t, f"{p}.ghost1", ch, half)
        if stage == strided_stage:
            t[f"{p}.conv_dw.weight"] = _rand(mid, 1, 3, 3)
            _bn(t, f"{p}.bn_dw", mid)
            t[f"{p}.se.conv_reduce.weight"] = _rand(4, mid, 1, 1)
            t[f"{p}.se.conv_reduce.bias"] = _rand(4)
            t[f"{p}.se.conv_expand.weight"] = _rand(mid, 4, 1, 1)
            t[f"{p}.se.conv_expand.bias"] = _rand(mid)
        _ghost(t, f"{p}.ghost2", mid, half)
        if stage == strided_stage:
            # shortcut: depthwise, norm, pointwise, norm
            t[f"{p}.shortcut.0.weight"] = _rand(ch, 1, 3, 3)
            _bn(t, f"{p}.shortcut.1", ch)
            t[f"{p}.shortcut.2.weight"] = _rand(mid, ch, 1, 1)
            _bn(t, f"{p}.shortcut.3", mid)
        ch = mid

    t["blocks.2.0.conv.weight"] = _rand(16, ch, 1, 1)
    _bn(t, "blocks.2.0.bn1", 16)
    t["conv_head.weight"] = _rand(16, 16, 1, 1)
    t["conv_head.bias"] = _rand(16)
    t["classifier.weight"] = _rand(classes, 16)
    t["classifier.bias"] = _rand(classes)
    save_file(t, str(tmp_path / "model.safetensors"))


def test_model_config_uses_timm_architecture_when_model_type_absent(tmp_path: Path):
    _write_tiny_ghostnet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    assert cfg.model_type == "ghostnet_100"
    assert plugin.matches(cfg.model_type)


@pytest.mark.parametrize(
    "model_type", ["ghostnet_100", "ghostnet_050", "ghostnetv2_100", "timm_ghostnet"]
)
def test_plugin_matches_ghostnet_variants(model_type: str):
    assert plugin.matches(model_type)


@pytest.mark.parametrize("model_type", ["resnet50", "mnasnet_100", "efficientnet_b0", ""])
def test_plugin_rejects_unrelated_model_types(model_type: str):
    assert not plugin.matches(model_type)


def test_bundle_config_reads_nested_pretrained_cfg(tmp_path: Path):
    _write_tiny_ghostnet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    bundle_config = plugin.get_bundle_config_overrides(cfg)

    assert bundle_config["runtime_strategy"] == "timm_ghostnet_image_classification"
    assert bundle_config["num_classes"] == 5


def test_stride_is_derived_from_the_depthwise_convolution(tmp_path: Path):
    """GhostNet needs no stride table: a block downsamples iff it has conv_dw."""
    _write_tiny_ghostnet(tmp_path, strided_stage=1)
    cfg = ModelConfig.from_dir(tmp_path)

    plugin.load_weights(str(tmp_path), cfg)
    blocks = cfg.raw["_timm_ghostnet_config"]["blocks"]

    assert [b["stride"] for b in blocks] == [1, 2, 1]
    assert [b["kind"] for b in blocks] == [
        "ghost_bottleneck", "ghost_bottleneck", "conv_bn_act"
    ]


def test_layout_detects_the_gate_and_the_projection_shortcut(tmp_path: Path):
    _write_tiny_ghostnet(tmp_path, strided_stage=1)
    cfg = ModelConfig.from_dir(tmp_path)

    plugin.load_weights(str(tmp_path), cfg)
    blocks = cfg.raw["_timm_ghostnet_config"]["blocks"]

    assert [b["has_se"] for b in blocks] == [False, True, False]
    assert [b["has_shortcut"] for b in blocks] == [False, True, False]


def test_load_weights_rejects_an_unrecognised_block(tmp_path: Path):
    _write_tiny_ghostnet(tmp_path)
    from safetensors.numpy import load_file

    kept = {
        k: v
        for k, v in load_file(str(tmp_path / "model.safetensors")).items()
        if not k.startswith("blocks.0.0.ghost1")
    }
    save_file(kept, str(tmp_path / "model.safetensors"))
    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="neither a Ghost bottleneck"):
        plugin.load_weights(str(tmp_path), cfg)


def test_build_engine_rejects_quantized_context(tmp_path: Path):
    _write_tiny_ghostnet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)

    with pytest.raises(NotImplementedError, match="quantized"):
        plugin.build_engine(cfg, weights, 0, quant_ctx=object())
