# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned DeepSeek-V3 MoE router regression coverage."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")

from families.deepseek_v2 import model, moe_routing
from families.deepseek_v2.config import ModelConfig


def _seq(*shape: int, start: int = 0) -> np.ndarray:
    size = int(np.prod(shape))
    return np.arange(start, start + size, dtype=np.float32).reshape(shape)


def _patch_tensor_io(
    monkeypatch: pytest.MonkeyPatch,
    tensors: dict[str, np.ndarray],
) -> None:
    monkeypatch.setattr(model, "_open_safetensors", lambda _: ["reader"])
    monkeypatch.setattr(model, "_has_tensor", lambda _readers, name: name in tensors)

    def load(_readers, name: str):
        if name not in tensors:
            raise KeyError(name)
        return tensors[name]

    monkeypatch.setattr(model, "_load_tensor", load)


def test_load_weights_preserves_router_bias_and_scoring_metadata(monkeypatch) -> None:
    raw = {
        "qk_nope_head_dim": 3,
        "qk_rope_head_dim": 1,
        "v_head_dim": 2,
        "kv_lora_rank": 4,
        "q_lora_rank": None,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "first_k_dense_replace": 0,
        "moe_layer_freq": 1,
        "moe_intermediate_size": 5,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "n_group": 1,
        "topk_group": 1,
    }
    config = ModelConfig(
        model_type="deepseek_v3",
        vocab_size=6,
        hidden_size=8,
        intermediate_size=7,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        raw=raw,
    )
    prefix = "model.layers.0"
    tensors = {
        "model.embed_tokens.weight": _seq(6, 8),
        f"{prefix}.input_layernorm.weight": _seq(8, start=100),
        f"{prefix}.post_attention_layernorm.weight": _seq(8, start=120),
        f"{prefix}.self_attn.q_proj.weight": _seq(8, 8, start=200),
        f"{prefix}.self_attn.kv_a_proj_with_mqa.weight": _seq(5, 8, start=300),
        f"{prefix}.self_attn.kv_a_layernorm.weight": _seq(4, start=400),
        f"{prefix}.self_attn.kv_b_proj.weight": _seq(10, 4, start=450),
        f"{prefix}.self_attn.o_proj.weight": _seq(8, 4, start=500),
        f"{prefix}.mlp.gate.weight": _seq(2, 8, start=600),
        f"{prefix}.mlp.gate.e_score_correction_bias": np.array([0.25, -0.5], dtype=np.float32),
        f"{prefix}.mlp.shared_experts.gate_proj.weight": _seq(5, 8, start=700),
        f"{prefix}.mlp.shared_experts.up_proj.weight": _seq(5, 8, start=750),
        f"{prefix}.mlp.shared_experts.down_proj.weight": _seq(8, 5, start=800),
    }
    for expert in range(2):
        expert_prefix = f"{prefix}.mlp.experts.{expert}"
        tensors[f"{expert_prefix}.gate_proj.weight"] = _seq(5, 8, start=900 + 150 * expert)
        tensors[f"{expert_prefix}.up_proj.weight"] = _seq(5, 8, start=950 + 150 * expert)
        tensors[f"{expert_prefix}.down_proj.weight"] = _seq(8, 5, start=1000 + 150 * expert)
    _patch_tensor_io(monkeypatch, tensors)

    weights = model._DeepSeekV2Model().load_weights("/unused", config)

    np.testing.assert_array_equal(
        weights["layer.0.router_score_bias"],
        tensors[f"{prefix}.mlp.gate.e_score_correction_bias"],
    )
    assert "layer.0.router" in weights
    assert weights["_scoring_func"] == "sigmoid"
    assert weights["_topk_method"] == "noaux_tc"
    assert weights["_n_group"] == 1
    assert weights["_topk_group"] == 1
    assert weights["_norm_topk_prob"] is True
    assert weights["_routed_scaling_factor"] == 1.5


def test_noaux_router_contract_is_supported() -> None:
    moe_routing.validate_router_contract(
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        n_routed_experts=256,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
    )


def test_non_finite_router_score_bias_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        model._validate_router_score_bias(
            np.array([0.0, np.nan], dtype=np.float32),
            "model.layers.1.mlp.gate.e_score_correction_bias",
        )


def test_selected_expert_graph_matches_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The portable selected-expert graph preserves DeepSeek SwiGLU routing."""

    class Tensor:
        def __init__(self, data: np.ndarray):
            self.data = np.asarray(data)
            self.shape = self.data.shape
            self.dtype = self.data.dtype

    class Layer:
        def __init__(self, data: np.ndarray):
            self.output = Tensor(data)

        def get_output(self, _index: int) -> Tensor:
            return self.output

    class Shuffle(Layer):
        @property
        def reshape_dims(self):
            return self.output.shape

        @reshape_dims.setter
        def reshape_dims(self, shape) -> None:
            self.output = Tensor(self.output.data.reshape(shape))

    class Network:
        @staticmethod
        def add_shuffle(tensor: Tensor) -> Shuffle:
            return Shuffle(tensor.data)

        @staticmethod
        def add_cast(tensor: Tensor, dtype) -> Layer:
            return Layer(tensor.data.astype(dtype))

        @staticmethod
        def add_gather(data: Tensor, indices: Tensor, _axis: int) -> Layer:
            return Layer(np.take(data.data, indices.data, axis=0))

        @staticmethod
        def add_matrix_multiply(left: Tensor, _left_op, right: Tensor, _right_op) -> Layer:
            return Layer(np.matmul(left.data, right.data))

        @staticmethod
        def add_activation(tensor: Tensor, _operation) -> Layer:
            return Layer(1.0 / (1.0 + np.exp(-tensor.data)))

        @staticmethod
        def add_elementwise(left: Tensor, right: Tensor, _operation) -> Layer:
            return Layer(left.data * right.data)

        @staticmethod
        def add_reduce(tensor: Tensor, _operation, _axes: int, keep_dims: bool) -> Layer:
            return Layer(np.sum(tensor.data, axis=1, keepdims=keep_dims))

    monkeypatch.setattr(
        model.graph_ops,
        "add_constant",
        lambda _network, _shape, values, dtype: Tensor(np.asarray(values, dtype=dtype)),
    )
    hidden = np.array([[0.25, -0.5], [1.0, 0.75]], dtype=np.float16)
    indices = np.array([[0, 2], [1, 0]], dtype=np.int32)
    routing = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=np.float32)
    weights = {}
    for expert in range(3):
        scale = float(expert + 1)
        weights[f"layer.0.expert.{expert}.w_gate"] = (
            np.array([[0.2, -0.1, 0.3], [0.4, 0.5, -0.2]], dtype=np.float16) * scale
        )
        weights[f"layer.0.expert.{expert}.w_up"] = (
            np.array([[0.3, 0.2, -0.4], [-0.1, 0.6, 0.5]], dtype=np.float16) * scale
        )
        weights[f"layer.0.expert.{expert}.w_down"] = (
            np.array([[0.5, -0.2], [0.1, 0.4], [-0.3, 0.2]], dtype=np.float16) / scale
        )

    actual = model._add_routed_experts(
        Network(),
        Tensor(hidden),
        weights,
        "layer.0",
        hidden_size=2,
        n_routed_experts=3,
        num_experts_per_tok=2,
        top_indices=Tensor(indices),
        scaled_weights=Tensor(routing),
        dtype=np.float16,
    ).data

    expected = np.zeros_like(hidden)
    for token in range(hidden.shape[0]):
        for route in range(indices.shape[1]):
            expert = int(indices[token, route])
            gate = hidden[token] @ weights[f"layer.0.expert.{expert}.w_gate"]
            up = hidden[token] @ weights[f"layer.0.expert.{expert}.w_up"]
            activated = gate / (1.0 + np.exp(-gate))
            down = (activated * up) @ weights[f"layer.0.expert.{expert}.w_down"]
            expected[token] += routing[token, route] * down

    assert actual.dtype == np.float16
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(
    ("scoring_func", "n_routed_experts", "n_group"),
    (
        ("softmax", 256, 8),
        ("sigmoid", 255, 8),
        ("sigmoid", 8, 8),
    ),
)
def test_noaux_router_contract_rejects_invalid_grouping(
    scoring_func: str,
    n_routed_experts: int,
    n_group: int,
) -> None:
    with pytest.raises(ValueError):
        moe_routing.validate_router_contract(
            scoring_func=scoring_func,
            topk_method="noaux_tc",
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=8,
            n_group=n_group,
            topk_group=4,
        )
