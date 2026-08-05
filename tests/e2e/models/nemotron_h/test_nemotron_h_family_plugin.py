# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Branch-focused tests for the Nemotron-H family plugin.

Trace: ARCH-FAM-001, UD-FAM-NEMOTRON-H
Intent: Validate Nemotron-H hybrid mamba/attention layer routing and weight loading branches
Preconditions: Layer type pattern string and synthetic tensors for mamba2/mlp/attention layers are provided
Postconditions: Layer types are correctly parsed and branch-specific weights load with correct fallback behavior
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from tensorrt_model_connect.config import ModelConfig
    import tensorrt_model_connect.families.nemotron_h as nemotron_h
    from tensorrt_model_connect.families.nemotron_h.plugin import plugin
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _seq(*shape: int, start: int = 0) -> np.ndarray:
    size = int(np.prod(shape))
    return np.arange(start, start + size, dtype=np.float32).reshape(shape)


def _patch_tensor_io(monkeypatch: pytest.MonkeyPatch,
                     tensor_map: dict[str, np.ndarray]) -> None:
    monkeypatch.setattr(nemotron_h, "_open_safetensors", lambda _: ["reader"])
    monkeypatch.setattr(
        nemotron_h, "_has_tensor", lambda _readers, name: name in tensor_map)

    def _load(_readers, name: str):
        if name not in tensor_map:
            raise KeyError(name)
        return tensor_map[name]

    monkeypatch.setattr(nemotron_h, "_load_tensor", _load)


def test_plugin_opts_in_to_staged_tp_bundle_loading():
    assert plugin.staged_tp_bundle_loading is True


def test_parse_layer_types_maps_and_filters_pattern_chars():
    """Intent: validate pattern-to-layer-type conversion.
    Preconditions: pattern includes valid markers and unrelated characters.
    Postconditions: only valid markers are retained and mapped to canonical layer names.
    """
    parsed = nemotron_h._parse_layer_types("ME-x*-M")
    assert parsed == [
        "mamba2",
        "moe",
        "mlp",
        "attention",
        "mlp",
        "mamba2",
    ]


def test_modelopt_checkpoint_is_rejected_before_tensor_open(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = ModelConfig(
        model_type="nemotron_h",
        vocab_size=5,
        hidden_size=8,
        intermediate_size=10,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        raw={
            "hybrid_override_pattern": "E",
            "quantization_config": {"quant_method": "modelopt"},
        },
    )
    monkeypatch.setattr(
        nemotron_h,
        "_open_safetensors",
        lambda _: pytest.fail("checkpoint should not be opened"),
    )

    with pytest.raises(NotImplementedError, match="Prepacked ModelOpt"):
        plugin.load_weights("/unused", cfg, precision="bf16")


def test_grouped_moe_routing_is_rejected_before_tensor_open(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = ModelConfig(
        model_type="nemotron_h",
        vocab_size=5,
        hidden_size=8,
        intermediate_size=10,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        raw={
            "hybrid_override_pattern": "E",
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_group": 2,
            "topk_group": 1,
        },
    )
    monkeypatch.setattr(
        nemotron_h,
        "_open_safetensors",
        lambda _: pytest.fail("checkpoint should not be opened"),
    )

    with pytest.raises(NotImplementedError, match="grouped expert routing"):
        plugin.load_weights("/unused", cfg, precision="bf16")


def test_sd_mamba_constants_follow_runtime_tensor_dtype():
    source = inspect.getsource(nemotron_h._add_mamba2_layer)

    assert source.count("_add_constant_like(") == 3
    assert "present_conv,\n        storage_dtype=dtype" in source
    assert "dt_raw,\n        storage_dtype=dtype" in source
    assert "B_3d.get_output(0),\n            storage_dtype=dtype" in source

    softplus_source = inspect.getsource(nemotron_h._add_stable_softplus)
    assert "trt.ActivationType.SOFTPLUS" in softplus_source
    assert "layer.alpha = 1.0" in softplus_source
    assert "layer.beta = 1.0" in softplus_source
    assert "_add_stable_softplus(network, dt_for_state)" in source


def test_load_weights_factorized_latent_moe(
    monkeypatch: pytest.MonkeyPatch,
):
    raw = {
        "hybrid_override_pattern": "E",
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 3,
        "moe_latent_size": 2,
        "moe_shared_expert_intermediate_size": 5,
        "routed_scaling_factor": 2.5,
        "norm_topk_prob": True,
    }
    cfg = ModelConfig(
        model_type="nemotron_h",
        vocab_size=3,
        hidden_size=4,
        intermediate_size=6,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        raw=raw,
    )
    tensors = {
        "backbone.embeddings.weight": _seq(3, 4),
        "backbone.layers.0.norm.weight": _seq(4, start=20),
        "backbone.layers.0.mixer.gate.weight": _seq(
            2, 4, start=30
        ),
        "backbone.layers.0.mixer.gate.e_score_correction_bias": _seq(
            2, start=40
        ),
        "backbone.layers.0.mixer.fc1_latent_proj.weight": _seq(
            2, 4, start=50
        ),
        "backbone.layers.0.mixer.fc2_latent_proj.weight": _seq(
            4, 2, start=60
        ),
        "backbone.layers.0.mixer.experts.0.up_proj.weight": _seq(
            3, 2, start=70
        ),
        "backbone.layers.0.mixer.experts.0.down_proj.weight": _seq(
            2, 3, start=80
        ),
        "backbone.layers.0.mixer.experts.1.up_proj.weight": _seq(
            3, 2, start=90
        ),
        "backbone.layers.0.mixer.experts.1.down_proj.weight": _seq(
            2, 3, start=100
        ),
        "backbone.layers.0.mixer.shared_experts.up_proj.weight": _seq(
            5, 4, start=110
        ),
        "backbone.layers.0.mixer.shared_experts.down_proj.weight": _seq(
            4, 5, start=130
        ),
    }
    _patch_tensor_io(monkeypatch, tensors)

    weights = plugin.load_weights(
        "/unused", cfg, precision="fp16"
    )

    assert weights["_layer_types"] == ["moe"]
    assert weights["_num_moe_layers"] == 1
    assert weights["_num_experts"] == 2
    assert weights["_num_experts_per_tok"] == 1
    assert weights["_moe_intermediate_size"] == 3
    assert weights["_moe_latent_size"] == 2
    assert weights["_shared_expert_intermediate_size"] == 5
    assert weights["_routed_scaling_factor"] == 2.5
    assert weights["_norm_topk_prob"] is True
    assert weights["layer.0.router"].shape == (4, 2)
    assert weights["layer.0.moe_fc1"].shape == (4, 2)
    assert weights["layer.0.moe_fc2"].shape == (2, 4)
    assert weights["layer.0.experts.w_up"].shape == (2, 2, 3)
    assert weights["layer.0.experts.w_down"].shape == (2, 3, 2)
    assert weights["layer.0.experts.w_up"].dtype == np.float16
    assert weights["layer.0.experts.w_down"].dtype == np.float16
    assert weights["layer.0.shared_expert.w_up"].shape == (4, 5)
    assert weights["layer.0.shared_expert.w_down"].shape == (5, 4)
    assert weights["layer.0.shared_expert.w_up"].dtype == np.float16


def test_load_weights_mixed_layer_routing_and_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
):
    """Intent: execute mamba2/mlp/attention branches plus fallback code paths.
    Preconditions: three-layer hybrid pattern with missing optional mamba norm/final/lm_head keys.
    Postconditions: branch-specific keys are loaded and fallback values are synthesized correctly.
    """
    raw = {
        "hybrid_override_pattern": "M-*",
        "mamba_num_heads": 2,
        "mamba_head_dim": 3,
        "n_groups": 1,
        "ssm_state_size": 2,
        "conv_kernel": 2,
    }
    cfg = ModelConfig(
        model_type="nemotron_h",
        vocab_size=5,
        hidden_size=8,
        intermediate_size=10,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        raw=raw,
    )

    tensors: dict[str, np.ndarray] = {
        "backbone.embeddings.weight": _seq(5, 8, start=0),
        "backbone.layers.0.norm.weight": _seq(8, start=100),
        "backbone.layers.1.norm.weight": _seq(8, start=200),
        "backbone.layers.2.norm.weight": _seq(8, start=300),
        # Layer 0 (mamba2)
        "backbone.layers.0.mixer.in_proj.weight": _seq(18, 8, start=400),
        "backbone.layers.0.mixer.conv1d.weight": _seq(10, 1, 2, start=600),
        "backbone.layers.0.mixer.conv1d.bias": _seq(10, start=700),
        "backbone.layers.0.mixer.out_proj.weight": _seq(8, 6, start=800),
        "backbone.layers.0.mixer.A_log": np.log(np.array([1.0, 3.0], dtype=np.float32)),
        "backbone.layers.0.mixer.D": np.array([0.25, 0.5], dtype=np.float32),
        "backbone.layers.0.mixer.dt_bias": np.array([-0.1, 0.2], dtype=np.float32),
        # mixer.norm.weight intentionally omitted to exercise ones fallback.
        # Layer 1 (mlp)
        "backbone.layers.1.mixer.up_proj.weight": _seq(10, 8, start=900),
        "backbone.layers.1.mixer.down_proj.weight": _seq(8, 10, start=1000),
        # Layer 2 (attention)
        "backbone.layers.2.mixer.q_proj.weight": _seq(8, 8, start=1100),
        "backbone.layers.2.mixer.k_proj.weight": _seq(4, 8, start=1200),
        "backbone.layers.2.mixer.v_proj.weight": _seq(4, 8, start=1300),
        "backbone.layers.2.mixer.o_proj.weight": _seq(8, 8, start=1400),
    }
    _patch_tensor_io(monkeypatch, tensors)

    weights = plugin.load_weights("/unused", cfg)

    np.testing.assert_allclose(
        weights["embedding"], tensors["backbone.embeddings.weight"])
    np.testing.assert_allclose(
        weights["layer.0.input_norm"], tensors["backbone.layers.0.norm.weight"])
    np.testing.assert_allclose(
        weights["layer.1.input_norm"], tensors["backbone.layers.1.norm.weight"])
    np.testing.assert_allclose(
        weights["layer.2.input_norm"], tensors["backbone.layers.2.norm.weight"])

    np.testing.assert_allclose(
        weights["layer.0.mamba_norm"], np.ones(6, dtype=np.float32))
    np.testing.assert_allclose(
        weights["layer.0.A"], np.array([-1.0, -3.0], dtype=np.float32))
    np.testing.assert_allclose(
        weights["layer.0.conv1d_weight"],
        tensors["backbone.layers.0.mixer.conv1d.weight"].reshape(10, 2),
    )

    assert "layer.1.w_up" in weights
    assert "layer.1.w_down" in weights
    assert "layer.1.w_q" not in weights

    assert weights["layer.2.w_q"].shape == (8, 8)
    assert weights["layer.2.w_k"].shape == (8, 4)
    assert weights["layer.2.w_v"].shape == (8, 4)
    assert weights["layer.2.w_o"].shape == (8, 8)

    np.testing.assert_allclose(weights["final_norm"], np.ones(8, dtype=np.float32))
    np.testing.assert_allclose(
        weights["w_lm_head"], tensors["backbone.embeddings.weight"].T.astype(np.float32)
    )

    assert weights["_layer_types"] == ["mamba2", "mlp", "attention"]
    assert weights["_num_mamba_layers"] == 1
    assert weights["_num_attention_layers"] == 1
    assert weights["_d_inner"] == 6
    assert weights["_conv_dim"] == 10
    assert weights["_d_conv"] == 2


def test_load_weights_raises_for_pattern_length_mismatch(
    monkeypatch: pytest.MonkeyPatch,
):
    """Intent: ensure malformed hybrid patterns fail fast.
    Preconditions: pattern maps to fewer layer markers than num_hidden_layers.
    Postconditions: load_weights raises AssertionError before tensor mapping proceeds.
    """
    cfg = ModelConfig(
        model_type="nemotron_h",
        vocab_size=5,
        hidden_size=8,
        intermediate_size=10,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        raw={"hybrid_override_pattern": "M-"},
    )
    monkeypatch.setattr(nemotron_h, "_open_safetensors", lambda _: ["reader"])

    with pytest.raises(AssertionError, match="Pattern length"):
        plugin.load_weights("/unused", cfg)


def test_get_bundle_config_overrides_derives_runtime_fields():
    """Intent: validate derived runtime fields for bundle config emission.
    Preconditions: raw config includes custom mamba dimensions and mixed layer pattern.
    Postconditions: overrides include normalized layer routing and derived shape metadata.
    """
    raw = {
        "hybrid_override_pattern": "M-*-x",
        "mamba_num_heads": 3,
        "mamba_head_dim": 2,
        "n_groups": 2,
        "mamba_state_dim": 4,
        "conv_kernel": 5,
    }
    cfg = ModelConfig(
        model_type="nemotron_h",
        vocab_size=5,
        hidden_size=8,
        intermediate_size=10,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        raw=raw,
    )

    overrides = plugin.get_bundle_config_overrides(cfg)
    assert overrides["layer_types"] == ["mamba2", "mlp", "attention", "mlp"]
    assert overrides["num_mamba_layers"] == 1
    assert overrides["num_attention_layers"] == 1
    assert overrides["d_inner"] == 6
    assert overrides["mamba_d_state"] == 4
    assert overrides["mamba_d_conv"] == 5
    assert overrides["mamba_nheads"] == 3
    assert overrides["mamba_head_dim"] == 2
    assert overrides["conv_dim"] == 22
    assert overrides["n_groups"] == 2


def test_get_bundle_config_overrides_includes_latent_moe_fields():
    cfg = ModelConfig(
        model_type="nemotron_h",
        vocab_size=5,
        hidden_size=8,
        intermediate_size=10,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        raw={
            "hybrid_override_pattern": "ME*",
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 12,
            "moe_latent_size": 6,
            "moe_shared_expert_intermediate_size": 16,
            "routed_scaling_factor": 3.0,
            "norm_topk_prob": False,
        },
    )

    overrides = plugin.get_bundle_config_overrides(cfg)

    assert overrides["layer_types"] == [
        "mamba2",
        "moe",
        "attention",
    ]
    assert overrides["num_moe_layers"] == 1
    assert overrides["num_experts"] == 8
    assert overrides["num_experts_per_tok"] == 2
    assert overrides["moe_intermediate_size"] == 12
    assert overrides["moe_latent_size"] == 6
    assert overrides[
        "moe_shared_expert_intermediate_size"
    ] == 16
    assert overrides["routed_scaling_factor"] == 3.0
    assert overrides["norm_topk_prob"] is False


def test_quant_exclusions_keep_router_in_fp32():
    patterns = plugin.quant_exclude_patterns("nvfp4")
    assert "*.router" in patterns


def test_sparse_expert_dispatch_rejects_quantization():
    with pytest.raises(ValueError, match="does not support quantized expert"):
        nemotron_h._add_moe_layer(
            network=None,
            hidden=None,
            eps_tensor=None,
            weights={},
            prefix="layer.0",
            hidden_size=4,
            num_experts=4,
            top_k=2,
            moe_intermediate=3,
            moe_latent=2,
            shared_expert_intermediate=0,
            routed_scaling_factor=1.0,
            norm_topk_prob=True,
            quant_ctx=object(),
        )


def test_sparse_expert_dispatch_gathers_top_k_weight_batches(monkeypatch):
    events = []
    fake_trt = SimpleNamespace(
        float16=nemotron_h.trt.float16,
        bfloat16=nemotron_h.trt.bfloat16,
        int32=nemotron_h.trt.int32,
        MatrixOperation=SimpleNamespace(NONE="none"),
        ElementWiseOperation=SimpleNamespace(PROD="prod"),
    )
    monkeypatch.setattr(nemotron_h, "trt", fake_trt)

    class Tensor:
        def __init__(self, name, dtype):
            self.name = name
            self.dtype = dtype

    class Layer:
        def __init__(self, output):
            self.output = output
            self.reshape_dims = None

        def get_output(self, index):
            assert index == 0
            return self.output

    class Network:
        def add_shuffle(self, tensor):
            events.append(("shuffle", tensor.name))
            return Layer(Tensor(f"shuffle({tensor.name})", tensor.dtype))

        def add_gather(self, tensor, indices, axis):
            events.append(("gather", tensor.name, indices.name, axis))
            return Layer(Tensor(f"gather({tensor.name})", tensor.dtype))

        def add_cast(self, tensor, dtype):
            events.append(("cast", tensor.name, dtype))
            return Layer(Tensor(f"cast({tensor.name})", dtype))

        def add_elementwise(self, lhs, rhs, operation):
            events.append(("elementwise", lhs.name, rhs.name, operation))
            return Layer(Tensor("batched-latent", lhs.dtype))

    constants = []

    def add_constant(_network, shape, _values, dtype):
        name = f"constant-{len(constants)}"
        constants.append((name, shape, dtype))
        return Tensor(name, nemotron_h.trt.float16)

    monkeypatch.setattr(nemotron_h.graph_ops, "add_constant", add_constant)

    def batched_matmul(_network, lhs, _lhs_op, rhs, _rhs_op):
        events.append(("matmul", lhs.name, rhs.name))
        return Tensor(f"matmul-{len(events)}", lhs.dtype)

    monkeypatch.setattr(
        nemotron_h.graph_ops,
        "_add_matrix_multiply_with_fp32_accumulation",
        batched_matmul,
    )
    monkeypatch.setattr(
        nemotron_h.graph_ops,
        "add_activation",
        lambda _network, tensor, *_args, **_kwargs: tensor,
    )

    result = nemotron_h._add_selected_latent_experts(
        Network(),
        Tensor("latent", nemotron_h.trt.bfloat16),
        Tensor("top-indices", nemotron_h.trt.int32),
        np.ones((512, 4, 6), dtype=np.float16),
        np.ones((512, 6, 4), dtype=np.float16),
        top_k=22,
        dtype=np.float16,
    )

    assert result.dtype == nemotron_h.trt.bfloat16
    assert [event for event in events if event[0] == "gather"] == [
        ("gather", "constant-0", "shuffle(top-indices)", 0),
        ("gather", "constant-1", "shuffle(top-indices)", 0),
    ]
    assert constants == [
        ("constant-0", (512, 4, 6), np.float16),
        ("constant-1", (512, 6, 4), np.float16),
        ("constant-2", (22, 1, 1), np.float16),
    ]
    assert len([event for event in events if event[0] == "matmul"]) == 2
    first_gather = next(i for i, event in enumerate(events) if event[0] == "gather")
    first_matmul = next(i for i, event in enumerate(events) if event[0] == "matmul")
    assert first_gather < first_matmul


def test_sd_bf16_constant_is_cast_to_runtime_dtype(monkeypatch):
    class FakeTensor:
        def __init__(self, dtype):
            self.dtype = dtype

    class FakeLayer:
        def __init__(self, output):
            self.output = output

        def get_output(self, index):
            assert index == 0
            return self.output

    class FakeNetwork:
        def __init__(self):
            self.cast_dtypes = []

        def add_cast(self, tensor, dtype):
            self.cast_dtypes.append((tensor.dtype, dtype))
            return FakeLayer(FakeTensor(dtype))

    monkeypatch.setattr(
        nemotron_h.graph_ops,
        "add_constant",
        lambda network, shape, values, dtype: FakeTensor(
            nemotron_h.trt.float16
        ),
    )
    network = FakeNetwork()
    like = FakeTensor(nemotron_h.trt.bfloat16)

    result = nemotron_h._add_constant_like(
        network,
        (1,),
        np.ones(1, dtype=np.float16),
        like,
        storage_dtype=np.float16,
    )

    assert result.dtype == nemotron_h.trt.bfloat16
    assert network.cast_dtypes == [
        (nemotron_h.trt.float16, nemotron_h.trt.bfloat16)
    ]
