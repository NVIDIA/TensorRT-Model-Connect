# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the timm RegNet image-classification family plugin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.families.timm_regnet import plugin
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires TensorRT", allow_module_level=True)


def _rand(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _conv_bn(t: dict, prefix: str, out_ch: int, in_ch: int, k: int) -> None:
    t[f"{prefix}.conv.weight"] = _rand(out_ch, in_ch, k, k)
    t[f"{prefix}.bn.weight"] = _rand(out_ch)
    t[f"{prefix}.bn.bias"] = _rand(out_ch)
    t[f"{prefix}.bn.running_mean"] = _rand(out_ch)
    t[f"{prefix}.bn.running_var"] = np.abs(_rand(out_ch)) + 1.0


def _write_tiny_regnet(
    tmp_path: Path, *, blocks_per_stage: tuple[int, ...] = (1, 2, 1, 1), se: bool = True
) -> None:
    """Write a miniature RegNetY; the first block of each stage downsamples."""
    classes = 5
    config = {
        "architecture": "regnety_040",
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
    stem = 8
    _conv_bn(t, "stem", stem, 3, 3)

    ch = stem
    for stage, count in enumerate(blocks_per_stage, start=1):
        out = 16
        for index in range(1, count + 1):
            p = f"s{stage}.b{index}"
            mid = 16
            _conv_bn(t, f"{p}.conv1", mid, ch, 1)
            # grouped 3x3: in-channels-per-group is half of mid here
            t[f"{p}.conv2.conv.weight"] = _rand(mid, mid // 2, 3, 3)
            t[f"{p}.conv2.bn.weight"] = _rand(mid)
            t[f"{p}.conv2.bn.bias"] = _rand(mid)
            t[f"{p}.conv2.bn.running_mean"] = _rand(mid)
            t[f"{p}.conv2.bn.running_var"] = np.abs(_rand(mid)) + 1.0
            if se:
                t[f"{p}.se.fc1.weight"] = _rand(4, mid, 1, 1)
                t[f"{p}.se.fc1.bias"] = _rand(4)
                t[f"{p}.se.fc2.weight"] = _rand(mid, 4, 1, 1)
                t[f"{p}.se.fc2.bias"] = _rand(mid)
            _conv_bn(t, f"{p}.conv3", out, mid, 1)
            if index == 1:
                _conv_bn(t, f"{p}.downsample", out, ch, 1)
            ch = out

    t["head.fc.weight"] = _rand(classes, ch)
    t["head.fc.bias"] = _rand(classes)
    save_file(t, str(tmp_path / "model.safetensors"))


def test_model_config_uses_timm_architecture_when_model_type_absent(tmp_path: Path):
    _write_tiny_regnet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    assert cfg.model_type == "regnety_040"
    assert plugin.matches(cfg.model_type)


@pytest.mark.parametrize(
    "model_type", ["regnety_040", "regnetx_032", "regnetz_c16", "timm_regnet"]
)
def test_plugin_matches_regnet_variants(model_type: str):
    assert plugin.matches(model_type)


@pytest.mark.parametrize("model_type", ["resnet50", "vgg16", "densenet121", ""])
def test_plugin_rejects_unrelated_model_types(model_type: str):
    assert not plugin.matches(model_type)


def test_bundle_config_reads_nested_pretrained_cfg(tmp_path: Path):
    _write_tiny_regnet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    bundle_config = plugin.get_bundle_config_overrides(cfg)

    assert bundle_config["runtime_strategy"] == "timm_regnet_image_classification"
    assert bundle_config["num_classes"] == 5


def test_layout_counts_blocks_and_marks_the_downsampling_block(tmp_path: Path):
    _write_tiny_regnet(tmp_path, blocks_per_stage=(1, 2, 1, 1))
    cfg = ModelConfig.from_dir(tmp_path)

    plugin.load_weights(str(tmp_path), cfg)
    blocks = cfg.raw["_timm_regnet_config"]["blocks"]

    assert len(blocks) == 5
    assert cfg.raw["_timm_regnet_config"]["num_stages"] == 4
    # RegNet halves the resolution at the head of every stage.
    assert [b["stride"] for b in blocks] == [2, 2, 1, 2, 2]
    assert [b["has_downsample"] for b in blocks] == [True, True, False, True, True]


def test_layout_detects_the_squeeze_excite_gate(tmp_path: Path):
    _write_tiny_regnet(tmp_path, se=True)
    cfg = ModelConfig.from_dir(tmp_path)
    plugin.load_weights(str(tmp_path), cfg)
    assert all(b["has_se"] for b in cfg.raw["_timm_regnet_config"]["blocks"])


def test_layout_handles_a_checkpoint_without_squeeze_excite(tmp_path: Path):
    """RegNetX has no SE gate; RegNetY does."""
    _write_tiny_regnet(tmp_path, se=False)
    cfg = ModelConfig.from_dir(tmp_path)
    plugin.load_weights(str(tmp_path), cfg)
    assert not any(b["has_se"] for b in cfg.raw["_timm_regnet_config"]["blocks"])


def test_load_weights_rejects_a_checkpoint_without_stages(tmp_path: Path):
    _write_tiny_regnet(tmp_path)
    from safetensors.numpy import load_file

    kept = {
        k: v
        for k, v in load_file(str(tmp_path / "model.safetensors")).items()
        if not k.startswith("s")
    }
    save_file(kept, str(tmp_path / "model.safetensors"))
    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="no s<stage>"):
        plugin.load_weights(str(tmp_path), cfg)


def test_build_engine_rejects_quantized_context(tmp_path: Path):
    _write_tiny_regnet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)

    with pytest.raises(NotImplementedError, match="quantized"):
        plugin.build_engine(cfg, weights, 0, quant_ctx=object())
