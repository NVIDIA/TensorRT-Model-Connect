# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the timm ViT image-classification family model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file
    from tensorrt_model_connect.config import ModelConfig
    import tensorrt_model_connect.models.timm_vit.model as model_module
    from tensorrt_model_connect.parallel_config import ParallelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires TensorRT", allow_module_level=True)


def _rand(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _write_tiny_vit(tmp_path: Path) -> dict[str, np.ndarray]:
    hidden = 8
    mlp = 16
    classes = 5
    config = {
        "architecture": "vit_base_patch16_224",
        "input_size": [3, 224, 224],
        "patch_size": 16,
        "num_features": hidden,
        "depth": 1,
        "num_heads": 2,
        "mlp_ratio": 2.0,
        "num_classes": classes,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    tensors = {
        "patch_embed.proj.weight": _rand(hidden, 3, 16, 16),
        "patch_embed.proj.bias": _rand(hidden),
        "cls_token": _rand(1, 1, hidden),
        "pos_embed": _rand(1, 197, hidden),
        "blocks.0.norm1.weight": _rand(hidden),
        "blocks.0.norm1.bias": _rand(hidden),
        "blocks.0.attn.qkv.weight": _rand(3 * hidden, hidden),
        "blocks.0.attn.qkv.bias": _rand(3 * hidden),
        "blocks.0.attn.proj.weight": _rand(hidden, hidden),
        "blocks.0.attn.proj.bias": _rand(hidden),
        "blocks.0.norm2.weight": _rand(hidden),
        "blocks.0.norm2.bias": _rand(hidden),
        "blocks.0.mlp.fc1.weight": _rand(mlp, hidden),
        "blocks.0.mlp.fc1.bias": _rand(mlp),
        "blocks.0.mlp.fc2.weight": _rand(hidden, mlp),
        "blocks.0.mlp.fc2.bias": _rand(hidden),
        "norm.weight": _rand(hidden),
        "norm.bias": _rand(hidden),
        "head.weight": _rand(classes, hidden),
        "head.bias": _rand(classes),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tensors


def test_model_config_uses_timm_architecture_when_model_type_absent(tmp_path: Path):
    _write_tiny_vit(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    assert cfg.model_type == "vit_base_patch16_224"
    assert cfg.hidden_size == 8
    assert cfg.architectures == ["vit_base_patch16_224"]
    assert model_module.matches(cfg.model_type)


def test_bundle_config_preserves_image_preprocess_contract(tmp_path: Path):
    _write_tiny_vit(tmp_path)
    raw = json.loads((tmp_path / "config.json").read_text())
    raw.update({
        "mean": [0.1, 0.2, 0.3],
        "std": [0.4, 0.5, 0.6],
        "crop_pct": 0.875,
        "interpolation": "bilinear",
    })
    (tmp_path / "config.json").write_text(json.dumps(raw))
    cfg = ModelConfig.from_dir(tmp_path)

    bundle_config = model_module.get_bundle_config_overrides(cfg)

    assert bundle_config["input_image_h"] == 224
    assert bundle_config["input_image_w"] == 224
    assert bundle_config["image_mean"] == [0.1, 0.2, 0.3]
    assert bundle_config["image_std"] == [0.4, 0.5, 0.6]
    assert bundle_config["crop_pct"] == pytest.approx(0.875)
    assert bundle_config["interpolation"] == "bilinear"


def test_load_weights_maps_timm_vit_shapes(tmp_path: Path):
    raw = _write_tiny_vit(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    weights = model_module.load_weights(str(tmp_path), cfg)

    assert weights["patch_embed.proj.weight"].shape == (8, 3, 16, 16)
    assert weights["cls_token"].shape == (1, 1, 8)
    assert weights["pos_embed"].shape == (1, 197, 8)
    assert weights["blocks.0.attn.qkv.weight"].shape == (8, 24)
    assert weights["blocks.0.mlp.fc1.weight"].shape == (8, 16)
    assert weights["blocks.0.mlp.fc2.weight"].shape == (16, 8)
    np.testing.assert_allclose(
        weights["blocks.0.attn.qkv.weight"],
        raw["blocks.0.attn.qkv.weight"].T,
    )


def test_timm_vit_tp_slices_mlp_weights_by_rank(tmp_path: Path):
    raw = _write_tiny_vit(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)
    weights = model_module.load_weights(str(tmp_path), cfg)
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)

    fc1 = model_module._slice_mlp_columns(
        weights["blocks.0.mlp.fc1.weight"], 16, parallel)
    fc1_bias = model_module._slice_mlp_columns(
        weights["blocks.0.mlp.fc1.bias"], 16, parallel)
    fc2 = model_module._slice_mlp_rows(
        weights["blocks.0.mlp.fc2.weight"], 16, parallel)

    assert fc1.shape == (8, 4)
    assert fc1_bias.shape == (4,)
    assert fc2.shape == (4, 8)
    np.testing.assert_allclose(fc1, raw["blocks.0.mlp.fc1.weight"].T[:, 8:12])
    np.testing.assert_allclose(fc1_bias, raw["blocks.0.mlp.fc1.bias"][8:12])
    np.testing.assert_allclose(fc2, raw["blocks.0.mlp.fc2.weight"].T[8:12, :])


def test_timm_vit_tp_validation_requires_concrete_rank(tmp_path: Path):
    _write_tiny_vit(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="concrete rank"):
        model_module._validate_timm_vit_tp(
            cfg,
            ParallelConfig(mode="tensor_parallel", tp_size=4, rank=-1),
        )


def test_timm_vit_plugin_routes_parallel_builds(monkeypatch, tmp_path: Path):
    _write_tiny_vit(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)
    weights = model_module.load_weights(str(tmp_path), cfg)
    calls: dict[str, object] = {}

    def fake_require(parallel, *, feature):
        calls["require"] = (parallel, feature)

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"timm-vit-tp-plan"

    monkeypatch.setattr(
        model_module, "require_tensorrt_11_for_tensor_parallel", fake_require)
    monkeypatch.setattr(model_module, "build_timm_vit_tp_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=1)
    result = model_module.build_engine(
        cfg,
        weights,
        1,
        verbose=True,
        parallel_config=parallel,
    )

    assert result == b"timm-vit-tp-plan"
    assert calls["require"][0] == parallel
    assert "timm_vit tensor-parallel" in calls["require"][1]
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 1
    assert kwargs["parallel_config"] == parallel
    assert kwargs["verbose"] is True
