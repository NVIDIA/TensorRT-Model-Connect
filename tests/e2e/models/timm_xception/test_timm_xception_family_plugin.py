# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the timm Xception image-classification family plugin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.families.timm_xception import plugin
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires TensorRT", allow_module_level=True)


def _rand(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _bn(t: dict, prefix: str, ch: int) -> None:
    t[f"{prefix}.weight"] = _rand(ch)
    t[f"{prefix}.bias"] = _rand(ch)
    t[f"{prefix}.running_mean"] = _rand(ch)
    t[f"{prefix}.running_var"] = np.abs(_rand(ch)) + 1.0


def _conv_bn(t: dict, prefix: str, out_ch: int, in_ch: int, k: int) -> None:
    t[f"{prefix}.conv.weight"] = _rand(out_ch, in_ch, k, k)
    _bn(t, f"{prefix}.bn", out_ch)


def _separable(t: dict, prefix: str, out_ch: int, in_ch: int) -> None:
    t[f"{prefix}.conv_dw.weight"] = _rand(in_ch, 1, 3, 3)
    _bn(t, f"{prefix}.bn_dw", in_ch)
    t[f"{prefix}.conv_pw.weight"] = _rand(out_ch, in_ch, 1, 1)
    _bn(t, f"{prefix}.bn_pw", out_ch)


def _write_tiny_xception(
    tmp_path: Path, *, shortcut_blocks: tuple[int, ...] = (0,), num_blocks: int = 3
) -> None:
    """Write a miniature aligned Xception."""
    classes = 5
    config = {
        "architecture": "xception41",
        "num_classes": classes,
        "num_features": 16,
        "pretrained_cfg": {
            "input_size": [3, 299, 299],
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "crop_pct": 0.903,
            "interpolation": "bicubic",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    t: dict[str, np.ndarray] = {}
    _conv_bn(t, "stem.0", 8, 3, 3)
    _conv_bn(t, "stem.1", 8, 8, 3)

    ch = 8
    for index in range(num_blocks):
        p = f"blocks.{index}"
        out = 16
        _separable(t, f"{p}.stack.conv1", out, ch)
        _separable(t, f"{p}.stack.conv2", out, out)
        _separable(t, f"{p}.stack.conv3", out, out)
        if index in shortcut_blocks:
            _conv_bn(t, f"{p}.shortcut", out, ch, 1)
        ch = out

    t["head.fc.weight"] = _rand(classes, ch)
    t["head.fc.bias"] = _rand(classes)
    save_file(t, str(tmp_path / "model.safetensors"))


def test_model_config_uses_timm_architecture_when_model_type_absent(tmp_path: Path):
    _write_tiny_xception(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    assert cfg.model_type == "xception41"
    assert plugin.matches(cfg.model_type)


@pytest.mark.parametrize("model_type", ["xception41", "xception65", "xception71", "timm_xception"])
def test_plugin_matches_xception_variants(model_type: str):
    assert plugin.matches(model_type)


@pytest.mark.parametrize("model_type", ["resnet50", "inception_v3", "vgg16", ""])
def test_plugin_rejects_unrelated_model_types(model_type: str):
    assert not plugin.matches(model_type)


def test_bundle_config_uses_the_xception_input_size_and_normalisation(tmp_path: Path):
    _write_tiny_xception(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    bundle_config = plugin.get_bundle_config_overrides(cfg)

    assert bundle_config["runtime_strategy"] == "timm_xception_image_classification"
    assert bundle_config["input_image_h"] == 299
    assert bundle_config["image_mean"] == [0.5, 0.5, 0.5]


def test_stride_is_derived_from_the_projection_shortcut(tmp_path: Path):
    """Xception downsamples exactly in the blocks that project."""
    _write_tiny_xception(tmp_path, shortcut_blocks=(0, 2), num_blocks=3)
    cfg = ModelConfig.from_dir(tmp_path)

    plugin.load_weights(str(tmp_path), cfg)
    blocks = cfg.raw["_timm_xception_config"]["blocks"]

    assert [b["has_shortcut"] for b in blocks] == [True, False, True]
    assert [b["stride"] for b in blocks] == [2, 1, 2]


def test_only_the_last_block_is_marked_as_the_exit_block(tmp_path: Path):
    """The exit block inverts the activation placement and takes no residual."""
    _write_tiny_xception(tmp_path, num_blocks=4)
    cfg = ModelConfig.from_dir(tmp_path)

    plugin.load_weights(str(tmp_path), cfg)
    blocks = cfg.raw["_timm_xception_config"]["blocks"]

    assert [b["is_exit"] for b in blocks] == [False, False, False, True]


def test_load_weights_reads_both_halves_of_each_separable_convolution(tmp_path: Path):
    _write_tiny_xception(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    weights = plugin.load_weights(str(tmp_path), cfg)

    for leaf in ("conv_dw", "conv_pw"):
        assert f"blocks.0.stack.conv1.{leaf}.weight" in weights
    for leaf in ("bn_dw", "bn_pw"):
        assert f"blocks.0.stack.conv1.{leaf}.running_var" in weights


def test_load_weights_rejects_a_block_without_a_stack(tmp_path: Path):
    _write_tiny_xception(tmp_path)
    from safetensors.numpy import load_file

    tensors = load_file(str(tmp_path / "model.safetensors"))
    kept = {k: v for k, v in tensors.items() if not k.startswith("blocks.0.stack")}
    kept["blocks.0.shortcut.conv.weight"] = _rand(16, 8, 1, 1)
    save_file(kept, str(tmp_path / "model.safetensors"))
    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="no separable convolution stack"):
        plugin.load_weights(str(tmp_path), cfg)


def test_build_engine_rejects_quantized_context(tmp_path: Path):
    _write_tiny_xception(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)

    with pytest.raises(NotImplementedError, match="quantized"):
        plugin.build_engine(cfg, weights, 0, quant_ctx=object())
