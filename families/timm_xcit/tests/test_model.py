# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for timm XCiT builds."""

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

from families.timm_xcit import model  # noqa: E402
from families.timm_xcit.checkpoint import Checkpoint  # noqa: E402
from families.timm_xcit.support import describe  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


WIDTH = 16
HEADS = 4


def _random(*shape: int) -> np.ndarray:
    return np.random.RandomState(29).randn(*shape).astype(np.float32)


def _norm(tensors: dict[str, np.ndarray], prefix: str, channels: int) -> None:
    tensors[f"{prefix}.weight"] = _random(channels)
    tensors[f"{prefix}.bias"] = _random(channels)
    tensors[f"{prefix}.running_mean"] = _random(channels)
    tensors[f"{prefix}.running_var"] = np.abs(_random(channels)) + 1.0


def _layer_norm(tensors: dict[str, np.ndarray], prefix: str) -> None:
    tensors[f"{prefix}.weight"] = _random(WIDTH)
    tensors[f"{prefix}.bias"] = _random(WIDTH)


def _checkpoint(tmp_path: Path, *, stem: int = 4, depth: int = 2, class_depth: int = 2) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architecture": "xcit_tiny_12_p16_224",
                "num_classes": 5,
                "pretrained_cfg": {
                    "input_size": [3, 224, 224],
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "crop_pct": 1.0,
                    "interpolation": "bicubic",
                    "crop_mode": "center",
                },
            }
        ),
        encoding="utf-8",
    )
    tensors: dict[str, np.ndarray] = {}
    for position in range(stem):
        index = position * 2
        channels = WIDTH if position == stem - 1 else 8
        tensors[f"patch_embed.proj.{index}.0.weight"] = _random(
            channels, 3 if position == 0 else 8, 3, 3
        )
        _norm(tensors, f"patch_embed.proj.{index}.1", channels)
    tensors["pos_embed.token_projection.weight"] = _random(WIDTH, 64, 1, 1)
    tensors["pos_embed.token_projection.bias"] = _random(WIDTH)

    for index in range(depth):
        source = f"blocks.{index}"
        tensors[f"{source}.attn.qkv.weight"] = _random(3 * WIDTH, WIDTH)
        tensors[f"{source}.attn.qkv.bias"] = _random(3 * WIDTH)
        tensors[f"{source}.attn.temperature"] = _random(HEADS, 1, 1)
        tensors[f"{source}.attn.proj.weight"] = _random(WIDTH, WIDTH)
        tensors[f"{source}.attn.proj.bias"] = _random(WIDTH)
        tensors[f"{source}.mlp.fc1.weight"] = _random(2 * WIDTH, WIDTH)
        tensors[f"{source}.mlp.fc1.bias"] = _random(2 * WIDTH)
        tensors[f"{source}.mlp.fc2.weight"] = _random(WIDTH, 2 * WIDTH)
        tensors[f"{source}.mlp.fc2.bias"] = _random(WIDTH)
        for which in ("norm1", "norm2", "norm3"):
            _layer_norm(tensors, f"{source}.{which}")
        for which in ("gamma1", "gamma2", "gamma3"):
            tensors[f"{source}.{which}"] = _random(WIDTH)
        for leaf in ("conv1", "conv2"):
            tensors[f"{source}.local_mp.{leaf}.weight"] = _random(WIDTH, 1, 3, 3)
            tensors[f"{source}.local_mp.{leaf}.bias"] = _random(WIDTH)
        _norm(tensors, f"{source}.local_mp.bn", WIDTH)

    for index in range(class_depth):
        source = f"cls_attn_blocks.{index}"
        for leaf in ("attn.q", "attn.k", "attn.v", "attn.proj"):
            tensors[f"{source}.{leaf}.weight"] = _random(WIDTH, WIDTH)
            tensors[f"{source}.{leaf}.bias"] = _random(WIDTH)
        tensors[f"{source}.mlp.fc1.weight"] = _random(2 * WIDTH, WIDTH)
        tensors[f"{source}.mlp.fc1.bias"] = _random(2 * WIDTH)
        tensors[f"{source}.mlp.fc2.weight"] = _random(WIDTH, 2 * WIDTH)
        tensors[f"{source}.mlp.fc2.bias"] = _random(WIDTH)
        for which in ("norm1", "norm2"):
            _layer_norm(tensors, f"{source}.{which}")
        for which in ("gamma1", "gamma2"):
            tensors[f"{source}.{which}"] = _random(WIDTH)

    tensors["cls_token"] = _random(1, 1, WIDTH)
    _layer_norm(tensors, "norm")
    tensors["head.weight"] = _random(5, WIDTH)
    tensors["head.bias"] = _random(5)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def _metadata(model_type: str) -> ModelMetadata:
    return ModelMetadata(config={"model_type": model_type}, model_index={})


def test_support_claims_only_its_own_identities() -> None:
    assert describe(_metadata("xcit_tiny_12_p16_224")) is not None
    assert describe(_metadata("xcit_nano_12_p8_384")) is not None
    assert describe(_metadata("vit_base_patch16_224")) is None


def test_layer_norm_uses_the_epsilon_xcit_builds_with() -> None:
    """XCiT uses 1e-6, where the other families here use 1e-5."""
    assert model._LAYER_NORM_EPSILON == 1e-6


def test_layout_reads_the_head_count_from_the_temperature(tmp_path: Path) -> None:
    """A fused qkv is the same shape at any head count; the temperature is not."""
    layout = model._layout(Checkpoint.open(_checkpoint(tmp_path)))
    assert layout["heads"] == HEADS


def test_layout_reads_the_patch_stem_depth(tmp_path: Path) -> None:
    """Four stem convolutions reach patch 16, three reach patch 8."""
    wide = model._layout(Checkpoint.open(_checkpoint(tmp_path / "a", stem=4)))
    narrow = model._layout(Checkpoint.open(_checkpoint(tmp_path / "b", stem=3)))
    assert wide["stem_convolutions"] == 4
    assert narrow["stem_convolutions"] == 3


def test_layout_counts_both_kinds_of_block(tmp_path: Path) -> None:
    layout = model._layout(Checkpoint.open(_checkpoint(tmp_path, depth=3, class_depth=2)))
    assert layout["depth"] == 3
    assert layout["class_depth"] == 2


def test_layout_rejects_a_gap_in_the_blocks(tmp_path: Path) -> None:
    tensors = {
        "blocks.1.attn.temperature": _random(HEADS, 1, 1),
        "cls_attn_blocks.0.gamma1": _random(WIDTH),
        "patch_embed.proj.0.0.weight": _random(8, 3, 3, 3),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="not contiguous from 0"):
        model._layout(Checkpoint.open(tmp_path))


def test_tokens_norm_is_false_only_for_the_nano_widths() -> None:
    """Not recorded in a checkpoint: the norm has the same weights either way.

    Only its input differs, so reading this wrong changes the class token
    without changing any tensor shape.
    """
    assert model.tokens_norm_of("xcit_nano_12_p16_224") is False
    assert model.tokens_norm_of("xcit_nano_12_p8_384") is False
    assert model.tokens_norm_of("xcit_tiny_12_p16_224") is True
    assert model.tokens_norm_of("xcit_large_24_p8_384") is True


def test_position_encoding_has_one_row_per_token() -> None:
    projection = np.zeros((WIDTH, 64, 1, 1), dtype=np.float32)
    projection[:, 0] = 1.0
    bias = np.zeros((WIDTH,), dtype=np.float32)
    encoded = model.fourier_position_encoding(4, 6, 32, projection, bias)
    assert encoded.shape == (1, WIDTH, 4, 6)


def test_position_encoding_puts_the_row_lanes_first() -> None:
    """Transposing a square grid must swap the two halves of every vector.

    The row half comes from the y coordinate and the column half from x, so
    this fails if they are concatenated the other way round.
    """
    hidden = 32
    projection = np.zeros((2 * hidden, 2 * hidden, 1, 1), dtype=np.float32)
    for index in range(2 * hidden):
        projection[index, index] = 1.0
    bias = np.zeros((2 * hidden,), dtype=np.float32)
    grid = model.fourier_position_encoding(5, 5, hidden, projection, bias)[0]
    rows, columns = grid[:hidden], grid[hidden:]
    np.testing.assert_allclose(rows, columns.transpose(0, 2, 1), rtol=1e-6, atol=1e-6)


def test_read_config_rejects_another_family(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"architecture": "vit_base_patch16_224"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unsupported timm XCiT model identity"):
        model._read_config(tmp_path)


def test_preprocess_config_rejects_a_crop_mode_the_seam_does_not_implement() -> None:
    """Reading this wrong feeds the engine different pixels, silently."""
    raw = {
        "num_classes": 5,
        "pretrained_cfg": {
            "input_size": [3, 224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_pct": 1.0,
            "interpolation": "bicubic",
            "crop_mode": "squash",
        },
    }
    with pytest.raises(NotImplementedError, match="crop_mode"):
        model._preprocess_config(raw)


def test_fold_reproduces_the_batch_norm_it_replaces(tmp_path: Path) -> None:
    checkpoint = Checkpoint.open(_checkpoint(tmp_path))
    weight, bias = model._fold_norm(
        checkpoint, "patch_embed.proj.0.0.weight", "patch_embed.proj.0.1", np.float32
    )
    raw = checkpoint.tensor("patch_embed.proj.0.0.weight")
    gamma = checkpoint.tensor("patch_embed.proj.0.1.weight")
    beta = checkpoint.tensor("patch_embed.proj.0.1.bias")
    mean = checkpoint.tensor("patch_embed.proj.0.1.running_mean")
    variance = checkpoint.tensor("patch_embed.proj.0.1.running_var")
    scale = gamma / np.sqrt(variance + model._BATCH_NORM_EPSILON)
    np.testing.assert_allclose(weight, raw * scale.reshape(-1, 1, 1, 1), rtol=1e-6)
    np.testing.assert_allclose(bias, beta - mean * scale, rtol=1e-6)


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
