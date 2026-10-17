# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep Qwen3 per-head normalization at the checkpoint/graph boundary."""

import json

import numpy as np
import pytest

pytest.importorskip("tensorrt")
safetensors = pytest.importorskip("safetensors.numpy")

@pytest.mark.parametrize("with_norms", [False, True])
def test_text_loader_preserves_per_head_norm_weights(tmp_path, with_norms):
    from families.internvl.config import ModelConfig
    from families.internvl.model import _load_internvl_text_weights

    config = ModelConfig.from_json(json.dumps({
        "model_type": "internvl", "text_config": {
            "model_type": "qwen3" if with_norms else "qwen2",
            "hidden_size": 4, "vocab_size": 8, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 2, "intermediate_size": 8,
        },
    }))
    prefix = "language_model.model.layers.0"
    tensors = {
        "language_model.model.embed_tokens.weight": np.ones((8, 4), dtype=np.float32),
        "language_model.model.norm.weight": np.ones(4, dtype=np.float32),
        "language_model.lm_head.weight": np.ones((8, 4), dtype=np.float32),
        f"{prefix}.input_layernorm.weight": np.ones(4, dtype=np.float32),
        f"{prefix}.post_attention_layernorm.weight": np.ones(4, dtype=np.float32),
    }
    for name, shape in [("q_proj", (4, 4)), ("k_proj", (2, 4)),
                        ("v_proj", (2, 4)), ("o_proj", (4, 4))]:
        tensors[f"{prefix}.self_attn.{name}.weight"] = np.ones(shape, dtype=np.float32)
    for name, shape in [("gate_proj", (8, 4)), ("up_proj", (8, 4)), ("down_proj", (4, 8))]:
        tensors[f"{prefix}.mlp.{name}.weight"] = np.ones(shape, dtype=np.float32)
    if with_norms:
        tensors[f"{prefix}.self_attn.q_norm.weight"] = np.array([1.25, 0.75], dtype=np.float32)
        tensors[f"{prefix}.self_attn.k_norm.weight"] = np.array([0.5, 1.5], dtype=np.float32)
    safetensors.save_file(tensors, str(tmp_path / "model.safetensors"))
    loaded = _load_internvl_text_weights(str(tmp_path), config)
    for name in ["q_norm", "k_norm"]:
        if with_norms:
            np.testing.assert_array_equal(loaded[f"layer.0.{name}"], tensors[f"{prefix}.self_attn.{name}.weight"])
        else:
            assert f"layer.0.{name}" not in loaded
