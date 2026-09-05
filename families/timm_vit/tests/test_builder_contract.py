# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned timm ViT builder and tensor-parallel contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")

from families.timm_vit.config import ModelConfig  # noqa: E402
from families.timm_vit import model as model_module  # noqa: E402
from families.timm_vit import parallel as parallel_module  # noqa: E402
from families.timm_vit import support as support_module  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


def _rand(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _write_tiny_vit(tmp_path: Path) -> dict[str, np.ndarray]:
    hidden, mlp, classes = 8, 16, 5
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
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
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


def test_model_config_uses_timm_architecture_when_model_type_absent(tmp_path: Path) -> None:
    _write_tiny_vit(tmp_path)
    config = ModelConfig.from_dir(tmp_path)

    assert config.model_type == "vit_base_patch16_224"
    assert config.hidden_size == 8
    assert config.architectures == ["vit_base_patch16_224"]
    support = support_module.describe(
        ModelMetadata(config={"architecture": config.model_type}, model_index={})
    )
    assert support is not None
    assert support.tasks == ("classification",)


def test_bundle_config_preserves_image_preprocess_contract(tmp_path: Path) -> None:
    _write_tiny_vit(tmp_path)
    raw = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    raw.update(
        {
            "mean": [0.1, 0.2, 0.3],
            "std": [0.4, 0.5, 0.6],
            "crop_pct": 0.875,
            "interpolation": "bilinear",
        }
    )
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    config = ModelConfig.from_dir(tmp_path)

    bundle_config = model_module._TimmVitModel().get_bundle_config_overrides(config)

    assert bundle_config["input_image_h"] == 224
    assert bundle_config["input_image_w"] == 224
    assert bundle_config["image_mean"] == [0.1, 0.2, 0.3]
    assert bundle_config["image_std"] == [0.4, 0.5, 0.6]
    assert bundle_config["crop_pct"] == pytest.approx(0.875)
    assert bundle_config["interpolation"] == "bilinear"


def test_load_weights_maps_timm_vit_shapes(tmp_path: Path) -> None:
    raw = _write_tiny_vit(tmp_path)
    config = ModelConfig.from_dir(tmp_path)
    weights = model_module._TimmVitModel().load_weights(str(tmp_path), config)

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


def test_tp_slices_mlp_weights_by_rank(tmp_path: Path) -> None:
    raw = _write_tiny_vit(tmp_path)
    config = ModelConfig.from_dir(tmp_path)
    weights = model_module._TimmVitModel().load_weights(str(tmp_path), config)
    parallel = parallel_module.ParallelConfig(tp_size=4, rank=2)

    fc1 = parallel_module._slice_mlp_columns(weights["blocks.0.mlp.fc1.weight"], 16, parallel)
    fc1_bias = parallel_module._slice_mlp_columns(weights["blocks.0.mlp.fc1.bias"], 16, parallel)
    fc2 = parallel_module._slice_mlp_rows(weights["blocks.0.mlp.fc2.weight"], 16, parallel)

    assert fc1.shape == (8, 4)
    assert fc1_bias.shape == (4,)
    assert fc2.shape == (4, 8)
    np.testing.assert_allclose(fc1, raw["blocks.0.mlp.fc1.weight"].T[:, 8:12])
    np.testing.assert_allclose(fc1_bias, raw["blocks.0.mlp.fc1.bias"][8:12])
    np.testing.assert_allclose(fc2, raw["blocks.0.mlp.fc2.weight"].T[8:12, :])


def test_tp_validation_requires_concrete_rank(tmp_path: Path) -> None:
    _write_tiny_vit(tmp_path)
    config = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="concrete rank"):
        parallel_module._validate_timm_vit_tp(
            config,
            parallel_module.ParallelConfig(tp_size=4, rank=-1),
        )


def test_model_routes_parallel_builds(monkeypatch, tmp_path: Path) -> None:
    _write_tiny_vit(tmp_path)
    config = ModelConfig.from_dir(tmp_path)
    model = model_module._TimmVitModel()
    weights = model.load_weights(str(tmp_path), config)
    captured = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        captured["call"] = (config, weights, max_cache_length, kwargs)
        return b"timm-vit-tp-plan"

    monkeypatch.setattr(parallel_module, "build_timm_vit_tp_engine", fake_build)
    parallel = parallel_module.ParallelConfig(tp_size=4, rank=1)

    result = model.build_engine(
        config,
        weights,
        1,
        verbose=True,
        parallel_config=parallel,
    )

    assert result == b"timm-vit-tp-plan"
    _, _, max_cache_length, kwargs = captured["call"]
    assert max_cache_length == 1
    assert kwargs["parallel_config"] == parallel
    assert kwargs["verbose"] is True
