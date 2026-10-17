# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned LFM2.5 MoE geometry and expert-weight orientation."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from families.lfm2_moe.config import validate_lfm2_moe_config
from families.lfm2_moe import checkpoint_mapper


def checkpoint_config():
    # LiquidAI/LFM2.5-8B-A1B, revision 5dd22602c2e9f6a097b1de4c4efe0658b605015c.
    return json.loads((Path(__file__).parent / "fixtures/lfm25-8b-a1b-config.json").read_text())


def test_checkpoint_geometry():
    cfg = validate_lfm2_moe_config(SimpleNamespace(raw=checkpoint_config()))
    assert cfg.rope_theta == 5_000_000
    assert cfg.max_position_embeddings == 128000
    assert cfg.vocab_size == 128000
    assert cfg.intermediate_size == 7168
    assert cfg.moe_intermediate_size == 1792
    assert cfg.num_experts == 32
    assert cfg.num_experts_per_tok == 4
    assert cfg.num_dense_layers == 2
    assert cfg.num_moe_layers == 22
    assert cfg.num_attention_layers == 6
    assert cfg.num_conv_layers == 18


@pytest.mark.parametrize(
    "field,value",
    [
        ("norm_topk_prob", False),
        ("use_expert_bias", False),
        ("routed_scaling_factor", 2),
        ("num_experts_per_tok", 33),
        ("num_dense_layers", 24),
        ("rope_parameters", {"rope_type": "linear", "rope_theta": 5_000_000}),
    ],
)
def test_unsupported_contract_is_rejected(field, value):
    raw = checkpoint_config()
    raw[field] = value
    with pytest.raises(ValueError):
        validate_lfm2_moe_config(SimpleNamespace(raw=raw))


@pytest.mark.parametrize("disk_backed", [False, True])
def test_expert_weight_mapping_preserves_gate_up_and_down(monkeypatch, tmp_path, disk_backed):
    # Distinct, non-symmetric values detect transposes and swapped SwiGLU branches.
    cfg = SimpleNamespace(hidden_size=4, moe_intermediate_size=3, num_experts=2)
    tensors = {}
    for i in range(2):
        for j, (name, shape) in enumerate([("w1", (3, 4)), ("w3", (3, 4)), ("w2", (4, 3))]):
            tensors[f"model.layers.2.feed_forward.experts.{i}.{name}.weight"] = (
                np.arange(12, dtype=np.float32).reshape(shape) + 100 * i + 20 * j
            )
    tensors["model.layers.2.feed_forward.gate.weight"] = np.arange(8, dtype=np.float32).reshape(
        2, 4
    )
    tensors["model.layers.2.feed_forward.expert_bias"] = np.array([0.1, -0.2], dtype=np.float32)
    monkeypatch.setattr(checkpoint_mapper, "_load_tensor", lambda _, name: tensors[name])
    with tempfile.TemporaryFile(dir=tmp_path) as storage:
        weights = checkpoint_mapper.WeightDict(storage_file=storage if disk_backed else None)
        checkpoint_mapper._store_moe_feed_forward(weights, [], "model.layers.2", "layer.2", cfg)
    for i in range(2):
        base = f"model.layers.2.feed_forward.experts.{i}"
        np.testing.assert_array_equal(
            weights["layer.2.moe_w13"][i, :, :3], tensors[base + ".w1.weight"].T
        )
        np.testing.assert_array_equal(
            weights["layer.2.moe_w13"][i, :, 3:], tensors[base + ".w3.weight"].T
        )
        np.testing.assert_array_equal(weights["layer.2.moe_w2"][i], tensors[base + ".w2.weight"].T)
