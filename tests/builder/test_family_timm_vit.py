"""Tests for the timm ViT image-classification family plugin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

try:
    from safetensors.numpy import save_file
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.families.timm_vit import plugin
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
    assert plugin.matches(cfg.model_type)


def test_load_weights_maps_timm_vit_shapes(tmp_path: Path):
    raw = _write_tiny_vit(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    weights = plugin.load_weights(str(tmp_path), cfg)

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
