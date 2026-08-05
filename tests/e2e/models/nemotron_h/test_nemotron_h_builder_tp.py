# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for Nemotron-H tensor-parallel support."""

from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    nemotron_h_module = importlib.import_module(
        "tensorrt_model_connect.families.nemotron_h.plugin")
    from tensorrt_model_connect.families.nemotron_h import tp_builder
    from tensorrt_model_connect.parallel_config import ParallelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _config(num_heads: int = 4, num_kv_heads: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        raw={},
        hidden_size=8,
        vocab_size=32,
        num_hidden_layers=3,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=2,
        intermediate_size=16,
        rms_norm_eps=1e-5,
    )


def _weights() -> dict:
    hidden = 8
    d_inner = 16
    d_state = 2
    d_conv = 3
    n_groups = 4
    mamba_heads = 4
    conv_dim = d_inner + 2 * n_groups * d_state
    proj_dim = d_inner + conv_dim + mamba_heads
    weights: dict[str, object] = {
        "_layer_types": ["mamba2", "mlp", "attention"],
        "_d_inner": d_inner,
        "_d_state": d_state,
        "_d_conv": d_conv,
        "_conv_dim": conv_dim,
        "_mamba_num_heads": mamba_heads,
        "_mamba_head_dim": 4,
        "_n_groups": n_groups,
        "_num_mamba_layers": 1,
        "_num_attention_layers": 1,
        "_attention_size": 8,
        "_mlp_size": 16,
        "embedding": np.zeros((32, hidden), dtype=np.float32),
        "final_norm": np.ones((hidden,), dtype=np.float32),
        "w_lm_head": np.zeros((hidden, 32), dtype=np.float32),
    }
    weights["layer.0.input_norm"] = np.ones((hidden,), dtype=np.float32)
    weights["layer.0.mamba_in_proj"] = np.arange(
        hidden * proj_dim, dtype=np.float32).reshape(hidden, proj_dim)
    weights["layer.0.conv1d_weight"] = np.arange(
        conv_dim * d_conv, dtype=np.float32).reshape(conv_dim, d_conv)
    weights["layer.0.conv1d_bias"] = np.arange(conv_dim, dtype=np.float32)
    weights["layer.0.dt_bias"] = np.arange(mamba_heads, dtype=np.float32)
    weights["layer.0.A"] = np.arange(mamba_heads, dtype=np.float32)
    weights["layer.0.D"] = np.arange(mamba_heads, dtype=np.float32)
    weights["layer.0.mamba_norm"] = np.arange(d_inner, dtype=np.float32)
    weights["layer.0.mamba_out_proj"] = np.arange(
        d_inner * hidden, dtype=np.float32).reshape(d_inner, hidden)

    weights["layer.1.input_norm"] = np.ones((hidden,), dtype=np.float32)
    weights["layer.1.w_up"] = np.arange(hidden * 16, dtype=np.float32).reshape(hidden, 16)
    weights["layer.1.w_down"] = np.arange(16 * hidden, dtype=np.float32).reshape(16, hidden)

    weights["layer.2.input_norm"] = np.ones((hidden,), dtype=np.float32)
    weights["layer.2.w_q"] = np.arange(hidden * 8, dtype=np.float32).reshape(hidden, 8)
    weights["layer.2.w_k"] = np.arange(hidden * 8, dtype=np.float32).reshape(hidden, 8)
    weights["layer.2.w_v"] = np.arange(hidden * 8, dtype=np.float32).reshape(hidden, 8)
    weights["layer.2.w_o"] = np.arange(8 * hidden, dtype=np.float32).reshape(8, hidden)
    return weights


def _moe_weights() -> dict:
    weights = _weights()
    hidden = 8
    num_experts = 4
    latent = 6
    intermediate = 16
    shared_intermediate = 12
    weights.update({
        "_num_moe_layers": 1,
        "_num_experts": num_experts,
        "_num_experts_per_tok": 2,
        "_moe_intermediate_size": intermediate,
        "_moe_latent_size": latent,
        "_shared_expert_intermediate_size": shared_intermediate,
        "_routed_scaling_factor": 1.0,
        "_norm_topk_prob": True,
        "layer.3.input_norm": np.ones(hidden, dtype=np.float32),
        "layer.3.router": np.arange(
            hidden * num_experts, dtype=np.float32
        ).reshape(hidden, num_experts),
        "layer.3.router_bias": np.arange(num_experts, dtype=np.float32),
        "layer.3.moe_fc1": np.arange(hidden * latent, dtype=np.float32).reshape(
            hidden, latent
        ),
        "layer.3.moe_fc2": np.arange(latent * hidden, dtype=np.float32).reshape(
            latent, hidden
        ),
        "layer.3.experts.w_up": np.arange(
            num_experts * latent * intermediate, dtype=np.float32
        ).reshape(num_experts, latent, intermediate),
        "layer.3.experts.w_down": np.arange(
            num_experts * intermediate * latent, dtype=np.float32
        ).reshape(num_experts, intermediate, latent),
        "layer.3.shared_expert.w_up": np.arange(
            hidden * shared_intermediate, dtype=np.float32
        ).reshape(hidden, shared_intermediate),
        "layer.3.shared_expert.w_down": np.arange(
            shared_intermediate * hidden, dtype=np.float32
        ).reshape(shared_intermediate, hidden),
    })
    return weights


def test_nemotron_h_tp_slices_mamba_mlp_and_attention_weights():
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)
    weights = _weights()

    sharded = tp_builder.shard_nemotron_h_weights(
        _config(), weights, parallel=parallel)

    expected_in_proj = np.concatenate([
        weights["layer.0.mamba_in_proj"][:, 8:12],
        weights["layer.0.mamba_in_proj"][:, 24:28],
        weights["layer.0.mamba_in_proj"][:, 36:38],
        weights["layer.0.mamba_in_proj"][:, 44:46],
        weights["layer.0.mamba_in_proj"][:, 50:51],
    ], axis=-1)
    expected_conv = np.concatenate([
        weights["layer.0.conv1d_weight"][8:12, :],
        weights["layer.0.conv1d_weight"][20:22, :],
        weights["layer.0.conv1d_weight"][28:30, :],
    ], axis=0)

    np.testing.assert_array_equal(sharded["layer.0.mamba_in_proj"], expected_in_proj)
    np.testing.assert_array_equal(sharded["layer.0.conv1d_weight"], expected_conv)
    np.testing.assert_array_equal(sharded["layer.0.mamba_out_proj"], weights["layer.0.mamba_out_proj"][8:12, :])
    np.testing.assert_array_equal(sharded["layer.1.w_up"], weights["layer.1.w_up"][:, 8:12])
    np.testing.assert_array_equal(sharded["layer.1.w_down"], weights["layer.1.w_down"][8:12, :])
    np.testing.assert_array_equal(sharded["layer.2.w_q"], weights["layer.2.w_q"][:, 4:6])
    np.testing.assert_array_equal(sharded["layer.2.w_k"], weights["layer.2.w_k"][:, 4:6])
    np.testing.assert_array_equal(sharded["layer.2.w_v"], weights["layer.2.w_v"][:, 4:6])
    np.testing.assert_array_equal(sharded["layer.2.w_o"], weights["layer.2.w_o"][4:6, :])
    assert sharded["_d_inner"] == 4
    assert sharded["_conv_dim"] == 8
    assert sharded["_mamba_num_heads"] == 1
    assert sharded["_n_groups"] == 1
    assert sharded["_attention_size"] == 2
    assert sharded["_mlp_size"] == 4


def test_nemotron_h_tp_groups_ranks_by_kv_head_for_gqa():
    weights = _weights()
    for key in ("w_k", "w_v"):
        weights[f"layer.2.{key}"] = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)

    kv_slices = (slice(0, 2), slice(0, 2), slice(2, 4), slice(2, 4))
    for rank, kv_slice in enumerate(kv_slices):
        sharded = tp_builder.shard_nemotron_h_weights(
            _config(num_kv_heads=2),
            weights,
            parallel=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=rank),
        )

        np.testing.assert_array_equal(
            sharded["layer.2.w_k"], weights["layer.2.w_k"][:, kv_slice]
        )
        np.testing.assert_array_equal(
            sharded["layer.2.w_v"], weights["layer.2.w_v"][:, kv_slice]
        )
        assert sharded["_num_key_value_heads"] == 1
        assert sharded["_kv_attention_size"] == 2


def test_nemotron_h_tp_rejects_unaligned_kv_head_ratio():
    with pytest.raises(ValueError, match="aligned GQA sharding"):
        tp_builder._validate_nemotron_h_tp(
            _config(num_kv_heads=3),
            _weights(),
            ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
        )


def test_nemotron_h_tp_shards_routed_and_shared_experts():
    weights = _moe_weights()
    sharded = tp_builder.shard_nemotron_h_weights(
        _config(),
        weights,
        parallel=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2),
    )

    np.testing.assert_array_equal(
        sharded["layer.3.experts.w_up"],
        weights["layer.3.experts.w_up"][:, :, 8:12],
    )
    np.testing.assert_array_equal(
        sharded["layer.3.experts.w_down"],
        weights["layer.3.experts.w_down"][:, 8:12, :],
    )
    np.testing.assert_array_equal(
        sharded["layer.3.shared_expert.w_up"],
        weights["layer.3.shared_expert.w_up"][:, 6:9],
    )
    np.testing.assert_array_equal(
        sharded["layer.3.shared_expert.w_down"],
        weights["layer.3.shared_expert.w_down"][6:9, :],
    )
    np.testing.assert_array_equal(
        sharded["layer.3.router"], weights["layer.3.router"]
    )
    np.testing.assert_array_equal(
        sharded["layer.3.moe_fc1"], weights["layer.3.moe_fc1"]
    )
    assert sharded["_moe_intermediate_size"] == 4
    assert sharded["_shared_expert_intermediate_size"] == 3


def test_nemotron_h_tp_moe_shards_numerically_reconstruct_full_outputs():
    weights = _moe_weights()
    latent = np.linspace(-0.3, 0.2, 6, dtype=np.float32).reshape(1, 6)
    hidden = np.linspace(-0.2, 0.5, 8, dtype=np.float32).reshape(1, 8)

    def relu_squared(value):
        return np.maximum(value, 0.0) ** 2

    full_expert_outputs = []
    for expert in range(4):
        full_expert_outputs.append(
            relu_squared(
                latent @ weights["layer.3.experts.w_up"][expert]
            )
            @ weights["layer.3.experts.w_down"][expert]
        )
    full_shared = (
        relu_squared(hidden @ weights["layer.3.shared_expert.w_up"])
        @ weights["layer.3.shared_expert.w_down"]
    )

    rank_expert_outputs = [[] for _expert in range(4)]
    rank_shared_outputs = []
    for rank in range(4):
        sharded = tp_builder.shard_nemotron_h_weights(
            _config(),
            weights,
            parallel=ParallelConfig(
                mode="tensor_parallel", tp_size=4, rank=rank
            ),
        )
        for expert in range(4):
            rank_expert_outputs[expert].append(
                relu_squared(
                    latent @ sharded["layer.3.experts.w_up"][expert]
                )
                @ sharded["layer.3.experts.w_down"][expert]
            )
        rank_shared_outputs.append(
            relu_squared(hidden @ sharded["layer.3.shared_expert.w_up"])
            @ sharded["layer.3.shared_expert.w_down"]
        )

    for expert in range(4):
        np.testing.assert_allclose(
            sum(rank_expert_outputs[expert]),
            full_expert_outputs[expert],
            rtol=2e-7,
            atol=2e-2,
        )
    np.testing.assert_allclose(
        sum(rank_shared_outputs), full_shared, rtol=2e-7, atol=2e-2
    )


def test_nemotron_h_tp_bf16_uses_bf16_runtime_dtype():
    storage_dtype, runtime_dtype = tp_builder._precision_dtypes("bf16")

    assert storage_dtype is np.float16
    assert runtime_dtype == tp_builder.trt.bfloat16


def test_nemotron_h_tp_casts_bf16_constants_to_runtime_dtype(monkeypatch):
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
        tp_builder.graph_ops,
        "add_constant",
        lambda network, shape, values, dtype: FakeTensor(tp_builder.trt.float16),
    )
    network = FakeNetwork()

    result = tp_builder._add_typed_constant(
        network,
        (1,),
        np.ones(1, dtype=np.float16),
        storage_dtype=np.float16,
        runtime_dtype=tp_builder.trt.bfloat16,
    )

    assert result.dtype == tp_builder.trt.bfloat16
    assert network.cast_dtypes == [
        (tp_builder.trt.float16, tp_builder.trt.bfloat16)
    ]


def test_nemotron_h_tp_rejects_quantized_builds():
    with pytest.raises(ValueError, match="do not support quantization"):
        tp_builder.build_nemotron_h_tp_engine(
            _config(),
            _weights(),
            4,
            quant_ctx=object(),
            parallel_config=ParallelConfig(
                mode="tensor_parallel", tp_size=4, rank=0
            ),
        )


def test_nemotron_h_tp_moe_keeps_router_and_reductions_at_expected_boundaries(
    monkeypatch,
):
    events = []
    fake_trt = SimpleNamespace(
        float32=tp_builder.trt.float32,
        bfloat16=tp_builder.trt.bfloat16,
        int32=tp_builder.trt.int32,
        ActivationType=SimpleNamespace(SIGMOID="sigmoid"),
        TopKOperation=SimpleNamespace(MAX="max"),
        ReduceOperation=SimpleNamespace(SUM="sum"),
        ElementWiseOperation=SimpleNamespace(
            SUM="sum",
            PROD="prod",
            DIV="div",
        ),
    )
    monkeypatch.setattr(tp_builder, "trt", fake_trt)

    class Tensor:
        def __init__(self, name, dtype):
            self.name = name
            self.dtype = dtype
            self.shape = (1, 1)

    class Layer:
        def __init__(self, *outputs):
            self.outputs = outputs
            self.reshape_dims = None

        def get_output(self, index):
            return self.outputs[index]

    class Network:
        def add_cast(self, tensor, dtype):
            events.append(("cast", tensor.name, dtype))
            return Layer(Tensor(f"cast({tensor.name})", dtype))

        def add_activation(self, tensor, _op):
            return Layer(Tensor(f"sigmoid({tensor.name})", tensor.dtype))

        def add_topk(self, tensor, *_args):
            events.append(("topk", tensor.name))
            return Layer(
                Tensor("top_values", tensor.dtype),
                Tensor("top_indices", fake_trt.int32),
            )

        def add_shuffle(self, tensor):
            return Layer(Tensor(f"shuffle({tensor.name})", tensor.dtype))

        def add_gather(self, tensor, _indices, _axis):
            return Layer(Tensor(f"gather({tensor.name})", tensor.dtype))

        def add_slice(self, tensor, **_kwargs):
            return Layer(Tensor(f"slice({tensor.name})", tensor.dtype))

        def add_elementwise(self, lhs, rhs, op):
            events.append(("elementwise", lhs.name, rhs.name, op))
            return Layer(Tensor(f"elementwise-{len(events)}", lhs.dtype))

        def add_reduce(self, tensor, *_args, **_kwargs):
            events.append(("reduce", tensor.name, tensor.dtype))
            return Layer(Tensor("routed_latent", tensor.dtype))

    prefix = "layer.0"
    weights = {
        f"{prefix}.input_norm": np.ones(4, dtype=np.float32),
        f"{prefix}.moe_fc1": np.ones((4, 2), dtype=np.float32),
        f"{prefix}.router": np.ones((4, 2), dtype=np.float32),
        f"{prefix}.experts.w_up": np.ones((2, 2, 4), dtype=np.float16),
        f"{prefix}.experts.w_down": np.ones((2, 4, 2), dtype=np.float16),
        f"{prefix}.moe_fc2": np.ones((2, 4), dtype=np.float32),
        f"{prefix}.shared_expert.w_up": np.ones((4, 3), dtype=np.float16),
        f"{prefix}.shared_expert.w_down": np.ones((3, 4), dtype=np.float16),
    }
    names = {
        id(value): key.rsplit(".", 1)[-1]
        for key, value in weights.items()
    }
    names[id(weights[f"{prefix}.moe_fc1"])] = "moe_fc1"
    names[id(weights[f"{prefix}.moe_fc2"])] = "moe_fc2"
    names[id(weights[f"{prefix}.shared_expert.w_up"])] = "shared_up"
    names[id(weights[f"{prefix}.shared_expert.w_down"])] = "shared_down"

    monkeypatch.setattr(
        tp_builder.graph_ops,
        "add_rms_norm",
        lambda *_args, **_kwargs: Tensor("normed", fake_trt.bfloat16),
    )

    def matmul(_network, lhs, _lhs_width, _rhs_width, rhs, dtype=np.float32):
        label = names[id(rhs)]
        events.append(("matmul", label, lhs.dtype, dtype))
        return Tensor(label, lhs.dtype)

    monkeypatch.setattr(
        tp_builder.graph_ops,
        "add_matmul_rhs_constant",
        matmul,
    )
    monkeypatch.setattr(
        tp_builder.graph_ops,
        "add_activation",
        lambda _network, inp, *_args, **_kwargs: Tensor(
            "shared_activation", inp.dtype
        ),
    )

    def selected_experts(
        _network, _latent_in, top_indices, *_args, top_k, **_kwargs
    ):
        events.append(("selected_experts", top_indices.name, top_k))
        return Tensor("selected_experts", fake_trt.bfloat16)

    monkeypatch.setattr(
        tp_builder,
        "_add_selected_latent_experts",
        selected_experts,
    )

    def all_reduce(_network, tensor, _tp_size):
        events.append(("all_reduce", tensor.name, tensor.dtype))
        return Tensor(f"{tensor.name}.reduced", tensor.dtype)

    monkeypatch.setattr(tp_builder, "add_all_reduce_sum", all_reduce)

    result = tp_builder._add_moe_tp_layer(
        network=Network(),
        hidden=Tensor("hidden", fake_trt.bfloat16),
        eps_tensor=Tensor("eps", fake_trt.bfloat16),
        weights=weights,
        prefix=prefix,
        hidden_size=4,
        num_experts=2,
        top_k=1,
        moe_latent=2,
        shared_expert_intermediate=3,
        routed_scaling_factor=1.0,
        norm_topk_prob=False,
        tp_size=2,
        dtype=np.float16,
    )

    assert result["hidden"].dtype == fake_trt.bfloat16
    assert ("matmul", "router", fake_trt.float32, np.float32) in events
    topk_index = next(
        index for index, event in enumerate(events) if event[0] == "topk"
    )
    selected_index = next(
        index
        for index, event in enumerate(events)
        if event[0] == "selected_experts"
    )
    assert events[selected_index][1:] == ("top_indices", 1)
    assert topk_index < selected_index
    assert any(
        event[0] == "cast"
        and event[1] == "selected_experts"
        and event[2] == fake_trt.float32
        for event in events
    )
    assert any(event[0] == "reduce" for event in events)
    reduce_indices = [
        index for index, event in enumerate(events)
        if event[0] == "all_reduce"
    ]
    assert len(reduce_indices) == 2
    assert events[reduce_indices[0]][2] == fake_trt.float32
    assert events[reduce_indices[1]][2] == fake_trt.bfloat16
    fc2_index = next(
        index for index, event in enumerate(events)
        if event[:2] == ("matmul", "moe_fc2")
    )
    shared_down_index = next(
        index for index, event in enumerate(events)
        if event[:2] == ("matmul", "shared_down")
    )
    residual_index = next(
        index for index, event in enumerate(events)
        if event[0] == "elementwise" and event[1] == "hidden"
    )
    assert (
        reduce_indices[0]
        < fc2_index
        < shared_down_index
        < reduce_indices[1]
        < residual_index
    )


def test_nemotron_h_tp_moe_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="MoE top_k"):
        tp_builder._add_moe_tp_layer(
            network=None,
            hidden=None,
            eps_tensor=None,
            weights={},
            prefix="layer.0",
            hidden_size=4,
            num_experts=2,
            top_k=0,
            moe_latent=2,
            shared_expert_intermediate=3,
            routed_scaling_factor=1.0,
            norm_topk_prob=False,
            tp_size=2,
        )


def test_nemotron_h_tp_uses_shared_stable_softplus():
    source = inspect.getsource(tp_builder._add_mamba2_tp_layer)

    assert "_add_stable_softplus(network, dt_for_state)" in source


def test_nemotron_h_plugin_routes_parallel_builds(monkeypatch):
    calls: dict[str, object] = {}

    def fake_require(parallel, *, feature):
        calls["require"] = (parallel, feature)

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"nemotron-h-tp-plan"

    monkeypatch.setattr(
        nemotron_h_module, "require_tensorrt_11_for_tensor_parallel", fake_require)
    monkeypatch.setattr(tp_builder, "build_nemotron_h_tp_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=1)
    result = nemotron_h_module.NemotronHPlugin().build_engine(
        _config(), _weights(), 17,
        precision="bf16",
        verbose=True,
        debug_layer_outputs=True,
        parallel_config=parallel,
    )

    assert result == b"nemotron-h-tp-plan"
    assert calls["require"][0] == parallel
    assert "Nemotron-H tensor-parallel" in calls["require"][1]
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 17
    assert kwargs["parallel_config"] == parallel
    assert kwargs["precision"] == "bf16"
    assert kwargs["verbose"] is True
    assert kwargs["debug_layer_outputs"] is True
