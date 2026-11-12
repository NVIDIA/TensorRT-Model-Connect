# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
from safetensors.numpy import save_file

from families.lfm3.checkpoint_mapper import load_lfm3_weights
from .test_config import config_raw


def test_checkpoint_routes_gates_sandwich_norms_and_tied_output(tmp_path):
    raw = config_raw()
    tensors = {
        "model.embed_tokens.weight": np.arange(128, dtype=np.float32).reshape(16, 8),
        "model.embedding_norm.weight": np.ones(8, np.float32),
    }
    for i, kind in enumerate(raw["layer_types"]):
        prefix = f"model.layers.{i}"
        for norm in ("operator_norm", "ffn_norm", "post_operator_norm", "post_ffn_norm"):
            tensors[f"{prefix}.{norm}.weight"] = np.full(8, i + 1, np.float32)
        for name, shape in [("w1", (16, 8)), ("w3", (16, 8)), ("w2", (8, 16))]:
            tensors[f"{prefix}.feed_forward.{name}.weight"] = np.ones(shape, np.float32)
        if kind == "conv":
            for name, shape in [("in_proj", (24, 8)), ("conv", (8, 1, 7)), ("out_proj", (8, 8))]:
                tensors[f"{prefix}.conv.{name}.weight"] = np.ones(shape, np.float32)
        else:
            for name, shape in [
                ("q_proj", (8, 8)),
                ("k_proj", (4, 8)),
                ("v_proj", (4, 8)),
                ("o_proj", (8, 8)),
                ("gate_proj", (2, 8)),
                ("q_norm", (4,)),
                ("k_norm", (4,)),
            ]:
                tensors[f"{prefix}.self_attn.{name}.weight"] = np.arange(
                    np.prod(shape), dtype=np.float32
                ).reshape(shape)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    weights = load_lfm3_weights(str(tmp_path), SimpleNamespace(raw=raw), precision="bf16")
    np.testing.assert_array_equal(
        weights["layer.1.w_gate"], tensors["model.layers.1.self_attn.gate_proj.weight"].T
    )
    np.testing.assert_array_equal(weights["layer.2.post_operator_norm"], np.full(8, 3, np.float32))
    np.testing.assert_array_equal(weights["w_lm_head"], tensors["model.embed_tokens.weight"].T)
    assert weights["layer.0.conv_weight"].shape == (8, 7)
    assert weights["_num_attention_layers"] == 2
