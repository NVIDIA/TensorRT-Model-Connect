# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contracts for the Cosmos3-Nano dual-stream TensorRT builder."""

from __future__ import annotations

import pytest

from tensorrt_model_connect.families.cosmos3.model_config import COSMOS3_NANO
from tensorrt_model_connect.families.cosmos3.transformer_builder import (
    _parallel_size,
    required_transformer_tensor_names,
    validate_transformer_state_dict,
)


class _Parallel:
    def __init__(self, mode: str, cp_size: int):
        self.mode = mode
        self.cp_size = cp_size


def test_required_weights_cover_both_parameterized_streams() -> None:
    names = set(required_transformer_tensor_names())
    assert len(names) == 11 + 22 * COSMOS3_NANO.num_hidden_layers
    assert "layers.0.self_attn.to_q.weight" in names
    assert "layers.0.self_attn.add_q_proj.weight" in names
    assert "layers.35.mlp.down_proj.weight" in names
    assert "layers.35.mlp_moe_gen.down_proj.weight" in names


def test_missing_dual_stream_weight_fails_closed() -> None:
    state = {name: object() for name in required_transformer_tensor_names()}
    state.pop("layers.17.self_attn.to_add_out.weight")
    with pytest.raises(KeyError, match="layers.17.self_attn.to_add_out.weight"):
        validate_transformer_state_dict(state)


def test_requested_context_parallel_size_is_valid() -> None:
    assert _parallel_size(_Parallel("context_parallel", 4)) == 4
    assert COSMOS3_NANO.num_attention_heads % 4 == 0
    assert COSMOS3_NANO.num_key_value_heads % 4 == 0


def test_tensor_parallel_is_not_silently_treated_as_context_parallel() -> None:
    with pytest.raises(ValueError, match="single-device or context-parallel"):
        _parallel_size(_Parallel("tensor_parallel", 2))
