"""Focused GLM family loader tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.families.glm import plugin
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _write_model(path: Path) -> None:
    from safetensors.numpy import save_file

    config = {
        "model_type": "glm",
        "vocab_size": 16,
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    prefix = "model.layers.0"
    tensors = {
        "model.embed_tokens.weight": np.ones((16, 8), dtype=np.float32),
        f"{prefix}.input_layernorm.weight": np.ones(8, dtype=np.float32),
        f"{prefix}.post_attention_layernorm.weight": np.ones(8, dtype=np.float32),
        f"{prefix}.self_attn.q_proj.weight": np.ones((8, 8), dtype=np.float32),
        f"{prefix}.self_attn.q_proj.bias": np.ones(8, dtype=np.float32),
        f"{prefix}.self_attn.k_proj.weight": np.ones((4, 8), dtype=np.float32),
        f"{prefix}.self_attn.k_proj.bias": np.ones(4, dtype=np.float32),
        f"{prefix}.self_attn.v_proj.weight": np.ones((4, 8), dtype=np.float32),
        f"{prefix}.self_attn.v_proj.bias": np.ones(4, dtype=np.float32),
        f"{prefix}.self_attn.o_proj.weight": np.ones((8, 8), dtype=np.float32),
        f"{prefix}.mlp.gate_up_proj.weight": np.ones((24, 8), dtype=np.float32),
        f"{prefix}.mlp.down_proj.weight": np.ones((8, 12), dtype=np.float32),
        "model.norm.weight": np.ones(8, dtype=np.float32),
        "lm_head.weight": np.ones((16, 8), dtype=np.float32),
    }
    save_file(tensors, str(path / "model.safetensors"))


def test_glm_loader_honors_fp16_precision(tmp_path: Path) -> None:
    """GLM uses the standard decoder builder, so load storage can be fp16."""
    _write_model(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    weights = plugin.load_weights(str(tmp_path), cfg, precision="fp16")

    assert weights["embedding"].dtype == np.float16
    assert weights["layer.0.w_q"].dtype == np.float16
    assert weights["layer.0.w_k"].dtype == np.float16
    assert weights["layer.0.k_bias"].dtype == np.float16
    assert weights["layer.0.w_gate"].dtype == np.float16
    assert weights["w_out"].dtype == np.float16
    assert weights["layer.0.input_norm"].dtype == np.float32
    assert weights["final_norm"].dtype == np.float32
