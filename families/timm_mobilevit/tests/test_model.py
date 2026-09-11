# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for timm MobileViT builds."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file


try:
    import tensorrt  # noqa: F401
except ModuleNotFoundError:
    sys.modules["tensorrt"] = SimpleNamespace()

from families.timm_mobilevit import model  # noqa: E402
from families.timm_mobilevit.checkpoint import Checkpoint  # noqa: E402
from families.timm_mobilevit.support import describe  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


def _random(*shape: int) -> np.ndarray:
    return np.random.RandomState(23).randn(*shape).astype(np.float32)


def _norm(tensors: dict[str, np.ndarray], prefix: str, channels: int) -> None:
    tensors[f"{prefix}.weight"] = _random(channels)
    tensors[f"{prefix}.bias"] = _random(channels)
    tensors[f"{prefix}.running_mean"] = _random(channels)
    tensors[f"{prefix}.running_var"] = np.abs(_random(channels)) + 1.0


def _conv_bn(tensors, prefix: str, out_ch: int, in_ch: int, k: int) -> None:
    tensors[f"{prefix}.conv.weight"] = _random(out_ch, in_ch, k, k)
    _norm(tensors, f"{prefix}.bn", out_ch)


def _checkpoint(tmp_path: Path) -> Path:
    """A tiny MobileViT: one inverted residual stage and one MobileViT block."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architecture": "mobilevit_s",
                "num_classes": 5,
                "pretrained_cfg": {
                    "input_size": [3, 256, 256],
                    "mean": [0.0, 0.0, 0.0],
                    "std": [1.0, 1.0, 1.0],
                    "crop_pct": 0.9,
                    "interpolation": "bicubic",
                },
            }
        ),
        encoding="utf-8",
    )
    tensors: dict[str, np.ndarray] = {}
    _conv_bn(tensors, "stem", 8, 3, 3)
    _conv_bn(tensors, "stages.0.0.conv1_1x1", 8, 8, 1)
    _conv_bn(tensors, "stages.0.0.conv2_kxk", 8, 1, 3)
    _conv_bn(tensors, "stages.0.0.conv3_1x1", 8, 8, 1)
    _conv_bn(tensors, "stages.1.0.conv1_1x1", 8, 8, 1)
    _conv_bn(tensors, "stages.1.0.conv2_kxk", 8, 1, 3)
    _conv_bn(tensors, "stages.1.0.conv3_1x1", 8, 8, 1)

    block = "stages.1.1"
    _conv_bn(tensors, f"{block}.conv_kxk", 8, 8, 3)
    tensors[f"{block}.conv_1x1.weight"] = _random(16, 8, 1, 1)
    _conv_bn(tensors, f"{block}.conv_proj", 8, 16, 1)
    _conv_bn(tensors, f"{block}.conv_fusion", 8, 16, 3)
    tensors[f"{block}.norm.weight"] = _random(16)
    tensors[f"{block}.norm.bias"] = _random(16)
    for layer in range(2):
        source = f"{block}.transformer.{layer}"
        tensors[f"{source}.attn.qkv.weight"] = _random(48, 16)
        tensors[f"{source}.attn.qkv.bias"] = _random(48)
        tensors[f"{source}.attn.proj.weight"] = _random(16, 16)
        tensors[f"{source}.attn.proj.bias"] = _random(16)
        tensors[f"{source}.mlp.fc1.weight"] = _random(32, 16)
        tensors[f"{source}.mlp.fc1.bias"] = _random(32)
        tensors[f"{source}.mlp.fc2.weight"] = _random(16, 32)
        tensors[f"{source}.mlp.fc2.bias"] = _random(16)
        for which in ("norm1", "norm2"):
            tensors[f"{source}.{which}.weight"] = _random(16)
            tensors[f"{source}.{which}.bias"] = _random(16)

    _conv_bn(tensors, "final_conv", 8, 8, 1)
    tensors["head.fc.weight"] = _random(5, 8)
    tensors["head.fc.bias"] = _random(5)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def _metadata(model_type: str) -> ModelMetadata:
    return ModelMetadata(config={"model_type": model_type}, model_index={})


def test_support_claims_only_its_own_identity() -> None:
    assert describe(_metadata("mobilevit_s")) is not None
    assert describe(_metadata("mobilevit_xxs")) is not None
    assert describe(_metadata("vit_base_patch16_224")) is None


def test_read_config_rejects_the_v2_architecture(tmp_path: Path) -> None:
    """MobileViTv2 replaces the attention entirely, so it is not this family."""
    (tmp_path / "config.json").write_text(
        json.dumps({"architecture": "mobilevitv2_100"}), encoding="utf-8"
    )
    with pytest.raises(NotImplementedError, match="separable linear attention"):
        model._read_config(tmp_path)


def test_layout_tells_the_two_block_kinds_apart(tmp_path: Path) -> None:
    blocks = model._layout(Checkpoint.open(_checkpoint(tmp_path)))
    assert [block["kind"] for block in blocks] == ["inverted", "inverted", "mobilevit"]
    assert blocks[-1]["depth"] == 2


def test_layout_rejects_a_block_that_is_neither_kind(tmp_path: Path) -> None:
    tensors: dict[str, np.ndarray] = {}
    _conv_bn(tensors, "stages.0.0.conv1_1x1", 4, 4, 1)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="neither an inverted residual nor a block"):
        model._layout(Checkpoint.open(tmp_path))


def test_transformer_depth_rejects_a_gap(tmp_path: Path) -> None:
    tensors = {"stages.0.0.transformer.1.attn.qkv.weight": _random(12, 4)}
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="not contiguous from 0"):
        model._transformer_depth(Checkpoint.open(tmp_path), "stages.0.0")


def test_the_fused_qkv_splits_into_three_square_projections(tmp_path: Path) -> None:
    """q, k and v are contiguous blocks of the fused weight, in that order."""
    checkpoint = Checkpoint.open(_checkpoint(tmp_path))
    blocks = model._layout(checkpoint)
    weights = model._weights(checkpoint, blocks, np.float32)
    fused = checkpoint.tensor("stages.1.1.transformer.0.attn.qkv.weight")
    for position, name in enumerate(("query", "key", "value")):
        stored = weights[f"stages.1.1.transformer.0.{name}.weight"]
        assert stored.shape == (16, 16)
        np.testing.assert_allclose(stored, fused[position * 16 : (position + 1) * 16], rtol=1e-6)


def test_head_count_is_the_documented_constant() -> None:
    """A fused qkv has the same shape at any head count, so this cannot be read."""
    assert model._ATTENTION_HEADS == 4


def test_patch_size_is_the_documented_constant() -> None:
    assert model._PATCH == 2


def test_fold_reproduces_the_batch_norm_it_replaces(tmp_path: Path) -> None:
    checkpoint = Checkpoint.open(_checkpoint(tmp_path))
    weight, bias = model._fold_norm(checkpoint, "stem.conv.weight", "stem.bn", np.float32)
    raw = checkpoint.tensor("stem.conv.weight")
    gamma = checkpoint.tensor("stem.bn.weight")
    beta = checkpoint.tensor("stem.bn.bias")
    mean = checkpoint.tensor("stem.bn.running_mean")
    variance = checkpoint.tensor("stem.bn.running_var")
    scale = gamma / np.sqrt(variance + model._BATCH_NORM_EPSILON)
    np.testing.assert_allclose(weight, raw * scale.reshape(-1, 1, 1, 1), rtol=1e-6)
    np.testing.assert_allclose(bias, beta - mean * scale, rtol=1e-6)


def test_preprocess_config_keeps_the_identity_normalisation() -> None:
    """MobileViT trains on raw [0, 1] pixels, unlike the ImageNet families."""
    resolved = model._preprocess_config(
        {
            "num_classes": 5,
            "pretrained_cfg": {
                "input_size": [3, 256, 256],
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
                "crop_pct": 0.9,
                "interpolation": "bicubic",
            },
        }
    )
    assert resolved["mean"] == [0.0, 0.0, 0.0]
    assert resolved["std"] == [1.0, 1.0, 1.0]


def test_preprocess_config_rejects_a_degenerate_std() -> None:
    raw = {
        "num_classes": 5,
        "pretrained_cfg": {
            "input_size": [3, 256, 256],
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 0.0, 1.0],
            "crop_pct": 0.9,
            "interpolation": "bicubic",
        },
    }
    with pytest.raises(ValueError, match="preprocessing or classifier config is invalid"):
        model._preprocess_config(raw)


def test_fp32_builds_switch_off_the_reduced_precision_path() -> None:
    import tensorrt as trt

    class _Config:
        def __init__(self) -> None:
            self.cleared: list[object] = []

        def clear_flag(self, flag: object) -> None:
            self.cleared.append(flag)

    config = _Config()
    model._configure_precision(config, "fp32")
    assert config.cleared == [trt.BuilderFlag.TF32]
    config = _Config()
    model._configure_precision(config, "fp16")
    assert config.cleared == []
