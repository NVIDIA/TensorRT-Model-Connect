# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the timm CrossViT image-classification family plugin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.families.timm_crossvit import plugin
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires TensorRT", allow_module_level=True)


def _rand(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _bn(t: dict, prefix: str, ch: int, *, unit: bool = False) -> None:
    t[f"{prefix}.weight"] = np.ones(ch, dtype=np.float32) if unit else _rand(ch)
    t[f"{prefix}.bias"] = np.zeros(ch, dtype=np.float32) if unit else _rand(ch)
    t[f"{prefix}.running_mean"] = np.zeros(ch, dtype=np.float32) if unit else _rand(ch)
    t[f"{prefix}.running_var"] = np.ones(ch, dtype=np.float32) if unit else np.abs(_rand(ch)) + 1.0


def _branch(t: dict, prefix: str, out_ch: int, in_ch: int, k: int, *, unit: bool = False) -> None:
    t[f"{prefix}.conv.weight"] = (
        np.zeros((out_ch, in_ch, k, k), dtype=np.float32) if unit else _rand(out_ch, in_ch, k, k)
    )
    _bn(t, f"{prefix}.bn", out_ch, unit=unit)


def _write_tiny_repvgg(tmp_path: Path, *, blocks_per_stage: tuple[int, ...] = (2, 1)) -> None:
    """Write a miniature CrossViT in its multi-branch training form."""
    classes = 5
    config = {
        "architecture": "crossvit_9_240",
        "num_classes": classes,
        "num_features": 16,
        "pretrained_cfg": {
            "input_size": [3, 224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_pct": 0.875,
            "interpolation": "bilinear",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    t: dict[str, np.ndarray] = {}
    stem = 8
    _branch(t, "stem.conv_kxk", stem, 3, 3)
    _branch(t, "stem.conv_1x1", stem, 3, 1)

    ch = stem
    for stage, count in enumerate(blocks_per_stage):
        out = 16
        for index in range(count):
            p = f"stages.{stage}.{index}"
            in_ch = ch if index == 0 else out
            _branch(t, f"{p}.conv_kxk", out, in_ch, 3)
            _branch(t, f"{p}.conv_1x1", out, in_ch, 1)
            if index > 0:
                # identity only exists where the shape is unchanged
                _bn(t, f"{p}.identity", out)
        ch = out

    t["head.fc.weight"] = _rand(classes, ch)
    t["head.fc.bias"] = _rand(classes)
    save_file(t, str(tmp_path / "model.safetensors"))


def test_model_config_uses_timm_architecture_when_model_type_absent(tmp_path: Path):
    _write_tiny_repvgg(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    assert cfg.model_type == "crossvit_9_240"
    assert plugin.matches(cfg.model_type)


@pytest.mark.parametrize("model_type", ["crossvit_9_240", "repvgg_b0", "timm_crossvit"])
def test_plugin_matches_repvgg_variants(model_type: str):
    assert plugin.matches(model_type)


@pytest.mark.parametrize("model_type", ["resnet50", "vgg16", "ghostnet_100", ""])
def test_plugin_rejects_unrelated_model_types(model_type: str):
    assert not plugin.matches(model_type)


def test_stride_is_derived_from_the_identity_branch(tmp_path: Path):
    """Only a block without an identity branch may change shape."""
    _write_tiny_repvgg(tmp_path, blocks_per_stage=(2, 1))
    cfg = ModelConfig.from_dir(tmp_path)

    plugin.load_weights(str(tmp_path), cfg)
    blocks = cfg.raw["_timm_crossvit_config"]["blocks"]

    assert [b["has_identity"] for b in blocks] == [False, True, False]
    assert [b["stride"] for b in blocks] == [2, 1, 2]


def test_load_weights_fuses_every_block_to_one_kernel(tmp_path: Path):
    """Reparameterisation replaces three branches with a single 3x3 convolution."""
    _write_tiny_repvgg(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    weights = plugin.load_weights(str(tmp_path), cfg)

    assert weights["stem.weight"].shape == (8, 3, 3, 3)
    assert weights["stem.bias"].shape == (8,)
    for block in cfg.raw["_timm_crossvit_config"]["blocks"]:
        assert weights[f"{block['prefix']}.weight"].shape[-2:] == (3, 3)
    # the multi-branch names must not survive into the built weights
    assert not any("conv_1x1" in key or "identity" in key for key in weights)


def test_identity_branch_folds_to_a_centre_tap(tmp_path: Path):
    """A unit batch norm on the identity path must contribute exactly 1 at the centre."""
    _write_tiny_repvgg(tmp_path, blocks_per_stage=(2,))
    from safetensors.numpy import load_file

    tensors = load_file(str(tmp_path / "model.safetensors"))
    p = "stages.0.1"
    # zero both convolutions and make the identity batch norm a pass-through
    for leaf, k in ((f"{p}.conv_kxk", 3), (f"{p}.conv_1x1", 1)):
        tensors[f"{leaf}.conv.weight"] = np.zeros_like(tensors[f"{leaf}.conv.weight"])
        ch = tensors[f"{leaf}.bn.weight"].shape[0]
        tensors[f"{leaf}.bn.weight"] = np.zeros(ch, dtype=np.float32)
        tensors[f"{leaf}.bn.bias"] = np.zeros(ch, dtype=np.float32)
        tensors[f"{leaf}.bn.running_mean"] = np.zeros(ch, dtype=np.float32)
        tensors[f"{leaf}.bn.running_var"] = np.ones(ch, dtype=np.float32)
    ch = tensors[f"{p}.identity.weight"].shape[0]
    tensors[f"{p}.identity.weight"] = np.ones(ch, dtype=np.float32)
    tensors[f"{p}.identity.bias"] = np.zeros(ch, dtype=np.float32)
    tensors[f"{p}.identity.running_mean"] = np.zeros(ch, dtype=np.float32)
    tensors[f"{p}.identity.running_var"] = np.ones(ch, dtype=np.float32)
    save_file(tensors, str(tmp_path / "model.safetensors"))

    cfg = ModelConfig.from_dir(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)
    fused = weights[f"{p}.weight"]

    expected = np.zeros_like(fused)
    for channel in range(fused.shape[0]):
        expected[channel, channel, 1, 1] = 1.0
    # 1/sqrt(1 + eps) is marginally below 1, so compare with a tolerance
    np.testing.assert_allclose(fused, expected, atol=1e-4)


def test_load_weights_rejects_a_block_without_the_3x3_branch(tmp_path: Path):
    _write_tiny_repvgg(tmp_path)
    from safetensors.numpy import load_file

    kept = {
        k: v
        for k, v in load_file(str(tmp_path / "model.safetensors")).items()
        if not k.startswith("stages.0.0.conv_kxk")
    }
    save_file(kept, str(tmp_path / "model.safetensors"))
    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="no conv_kxk"):
        plugin.load_weights(str(tmp_path), cfg)


def test_build_engine_rejects_quantized_context(tmp_path: Path):
    _write_tiny_repvgg(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)

    with pytest.raises(NotImplementedError, match="quantized"):
        plugin.build_engine(cfg, weights, 0, quant_ctx=object())
