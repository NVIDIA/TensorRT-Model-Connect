# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for DeepSeek-OCR tensor-parallel support."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    deepseek_ocr_module = importlib.import_module(
        "tensorrt_model_connect.families.deepseek_ocr.plugin")
    from tensorrt_model_connect.checkpoint_mapper import WeightDict
    from tensorrt_model_connect import trt_compat
    from tensorrt_model_connect.families.deepseek_ocr import graph_ops, tp_builder
    from tensorrt_model_connect.families.deepseek_ocr.prefill_config import (
        sequence_prefill_profile_lengths,
    )
    from tensorrt_model_connect.parallel_config import ParallelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _config(num_heads: int = 4, num_kv_heads: int | None = None) -> SimpleNamespace:
    kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    return SimpleNamespace(
        raw={},
        hidden_size=8,
        vocab_size=32,
        num_hidden_layers=2,
        num_attention_heads=num_heads,
        num_key_value_heads=kv_heads,
        head_dim=4,
        attention_size=num_heads * 4,
        intermediate_size=16,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )


def _weights() -> WeightDict:
    hidden = 8
    attention = 16
    kv_attention = 16
    inter = 16
    shared = 32
    weights = WeightDict({
        "_attention_size": attention,
        "_kv_attention_size": kv_attention,
        "_n_routed_experts": 2,
        "_n_shared_experts": 2,
        "_num_experts_per_tok": 2,
        "_first_k_dense_replace": 1,
        "_moe_intermediate_size": inter,
        "_shared_intermediate_size": shared,
        "_norm_topk_prob": False,
        "_routed_scaling_factor": 1.0,
        "embedding": np.zeros((32, hidden), dtype=np.float32),
        "final_norm": np.ones((hidden,), dtype=np.float32),
        "w_out": np.zeros((hidden, 32), dtype=np.float32),
    })
    for layer_idx in range(2):
        prefix = f"layer.{layer_idx}"
        weights[f"{prefix}.input_norm"] = np.ones((hidden,), dtype=np.float32)
        weights[f"{prefix}.post_attn_norm"] = np.ones((hidden,), dtype=np.float32)
        weights[f"{prefix}.w_q"] = np.arange(
            hidden * attention, dtype=np.float32).reshape(hidden, attention)
        weights[f"{prefix}.w_k"] = np.arange(
            hidden * kv_attention, dtype=np.float32).reshape(hidden, kv_attention)
        weights[f"{prefix}.w_v"] = np.arange(
            hidden * kv_attention, dtype=np.float32).reshape(hidden, kv_attention)
        weights[f"{prefix}.w_o"] = np.arange(
            attention * hidden, dtype=np.float32).reshape(attention, hidden)

    for key in ("w_gate", "w_up"):
        weights[f"layer.0.{key}"] = np.arange(
            hidden * inter, dtype=np.float32).reshape(hidden, inter)
    weights["layer.0.w_down"] = np.arange(
        inter * hidden, dtype=np.float32).reshape(inter, hidden)

    weights["layer.1.router"] = np.zeros((hidden, 2), dtype=np.float32)
    for key in ("w_gate", "w_up"):
        weights[f"layer.1.experts.{key}"] = np.arange(
            2 * hidden * inter, dtype=np.float32).reshape(2, hidden, inter)
    weights["layer.1.experts.w_down"] = np.arange(
        2 * inter * hidden, dtype=np.float32).reshape(2, inter, hidden)
    for key in ("w_gate", "w_up"):
        weights[f"layer.1.shared.{key}"] = np.arange(
            hidden * shared, dtype=np.float32).reshape(hidden, shared)
    weights["layer.1.shared.w_down"] = np.arange(
        shared * hidden, dtype=np.float32).reshape(shared, hidden)
    return weights


def test_deepseek_ocr_tp_slices_attention_and_moe_weights() -> None:
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=2, rank=1)
    weights = _weights()

    sharded = tp_builder.shard_deepseek_ocr_weights(
        _config(), weights, parallel=parallel)

    np.testing.assert_array_equal(
        sharded["layer.0.w_q"], weights["layer.0.w_q"][:, 8:16])
    np.testing.assert_array_equal(
        sharded["layer.0.w_k"], weights["layer.0.w_k"][:, 8:16])
    np.testing.assert_array_equal(
        sharded["layer.0.w_v"], weights["layer.0.w_v"][:, 8:16])
    np.testing.assert_array_equal(
        sharded["layer.0.w_o"], weights["layer.0.w_o"][8:16, :])
    np.testing.assert_array_equal(
        sharded["layer.0.w_gate"], weights["layer.0.w_gate"][:, 8:16])
    np.testing.assert_array_equal(
        sharded["layer.0.w_down"], weights["layer.0.w_down"][8:16, :])
    np.testing.assert_array_equal(
        sharded["layer.1.experts.w_up"],
        weights["layer.1.experts.w_up"][:, :, 8:16])
    np.testing.assert_array_equal(
        sharded["layer.1.experts.w_down"],
        weights["layer.1.experts.w_down"][:, 8:16, :])
    np.testing.assert_array_equal(
        sharded["layer.1.shared.w_gate"],
        weights["layer.1.shared.w_gate"][:, 16:32])
    assert sharded["_attention_size"] == 8
    assert sharded["_kv_attention_size"] == 8
    assert sharded["_moe_intermediate_size"] == 8
    assert sharded["_shared_intermediate_size"] == 16


def test_deepseek_ocr_tp_validation_rejects_non_divisible_attention_heads() -> None:
    with pytest.raises(ValueError, match="num_attention_heads"):
        tp_builder._validate_deepseek_ocr_tp(
            _config(num_heads=10),
            _weights(),
            ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
        )


def test_deepseek_ocr_plugin_routes_parallel_builds(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_require(parallel, *, feature):
        calls["require"] = (parallel, feature)

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"deepseek-ocr-tp-plan"

    monkeypatch.setattr(
        deepseek_ocr_module,
        "require_tensorrt_11_for_tensor_parallel",
        fake_require,
    )
    monkeypatch.setattr(tp_builder, "build_deepseek_ocr_tp_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=2, rank=1)
    plugin = deepseek_ocr_module.DeepSeekOCRPlugin()
    result = plugin.build_engine(
        _config(), _weights(), 4096,
        verbose=True,
        parallel_config=parallel,
    )

    assert result == b"deepseek-ocr-tp-plan"
    assert calls["require"][0] == parallel
    assert "DeepSeek-OCR tensor-parallel" in calls["require"][1]
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 4096
    assert kwargs["parallel_config"] == parallel
    assert kwargs["verbose"] is True


def test_deepseek_ocr_parallel_build_rejects_debug_outputs(monkeypatch) -> None:
    monkeypatch.setattr(
        deepseek_ocr_module,
        "require_tensorrt_11_for_tensor_parallel",
        lambda parallel, *, feature: None,
    )

    with pytest.raises(ValueError, match="debug layer outputs"):
        deepseek_ocr_module.DeepSeekOCRPlugin().build_engine(
            _config(),
            _weights(),
            max_cache_length=4096,
            parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=2, rank=0),
            debug_layer_outputs=True,
        )


def test_deepseek_ocr_forwards_fp16_to_vision_engine(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_build(model_dir, config, *, precision, verbose):
        calls.update(
            model_dir=model_dir,
            config=config,
            precision=precision,
            verbose=verbose,
        )
        return b"vision-plan"

    monkeypatch.setattr(
        deepseek_ocr_module,
        "_build_deepseek_ocr_vision_engine",
        fake_build,
    )
    config = _config()

    plan = deepseek_ocr_module.DeepSeekOCRPlugin().build_vision_engine(
        "/model", config, _weights(), precision="fp16", verbose=True)

    assert plan == b"vision-plan"
    assert calls == {
        "model_dir": "/model",
        "config": config,
        "precision": "fp16",
        "verbose": True,
    }


def test_deepseek_ocr_loads_weights_at_their_build_precision(monkeypatch) -> None:
    config = _config()
    config.num_hidden_layers = 3
    config.raw.update({
        "n_routed_experts": 2,
        "n_shared_experts": 2,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 1,
        "moe_intermediate_size": 4,
        "_fp32_layers": [1, 3],
    })

    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": np.zeros((32, 8), dtype=np.float32),
        "model.norm.weight": np.ones(8, dtype=np.float32),
        "lm_head.weight": np.zeros((32, 8), dtype=np.float32),
    }
    for layer_idx in range(3):
        prefix = f"model.layers.{layer_idx}"
        tensors[f"{prefix}.input_layernorm.weight"] = np.ones(8, dtype=np.float32)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = np.ones(
            8, dtype=np.float32)
        for projection in ("q_proj", "k_proj", "v_proj"):
            tensors[f"{prefix}.self_attn.{projection}.weight"] = np.zeros(
                (16, 8), dtype=np.float32)
        tensors[f"{prefix}.self_attn.o_proj.weight"] = np.zeros(
            (8, 16), dtype=np.float32)

    dense_prefix = "model.layers.0.mlp"
    for projection in ("gate_proj", "up_proj"):
        tensors[f"{dense_prefix}.{projection}.weight"] = np.zeros(
            (16, 8), dtype=np.float32)
    tensors[f"{dense_prefix}.down_proj.weight"] = np.zeros(
        (8, 16), dtype=np.float32)

    for layer_idx in (1, 2):
        moe_prefix = f"model.layers.{layer_idx}.mlp"
        tensors[f"{moe_prefix}.gate.weight"] = np.zeros(
            (2, 8), dtype=np.float32)
        for expert_idx in range(2):
            expert_prefix = f"{moe_prefix}.experts.{expert_idx}"
            for projection in ("gate_proj", "up_proj"):
                tensors[f"{expert_prefix}.{projection}.weight"] = np.zeros(
                    (4, 8), dtype=np.float32)
            tensors[f"{expert_prefix}.down_proj.weight"] = np.zeros(
                (8, 4), dtype=np.float32)
        shared_prefix = f"{moe_prefix}.shared_experts"
        for projection in ("gate_proj", "up_proj"):
            tensors[f"{shared_prefix}.{projection}.weight"] = np.zeros(
                (8, 8), dtype=np.float32)
        tensors[f"{shared_prefix}.down_proj.weight"] = np.zeros(
            (8, 8), dtype=np.float32)

    monkeypatch.setattr(deepseek_ocr_module, "_open_safetensors", lambda _path: object())
    monkeypatch.setattr(
        deepseek_ocr_module,
        "_has_tensor",
        lambda _readers, name: name in tensors,
    )
    monkeypatch.setattr(
        deepseek_ocr_module,
        "_load_tensor",
        lambda _readers, name: tensors[name].copy(),
    )

    weights = deepseek_ocr_module.DeepSeekOCRPlugin().load_weights(
        "/model", config, precision="fp16")

    assert weights["layer.0.w_q"].dtype == np.float16
    assert weights["layer.0.w_gate"].dtype == np.float16
    assert weights["layer.1.w_q"].dtype == np.float32
    assert weights["layer.1.experts.w_gate"].dtype == np.float32
    assert weights["layer.1.experts.w_gate"].shape == (2, 8, 4)
    assert weights["layer.2.experts.w_gate"].dtype == np.float16
    assert weights["layer.2.experts.w_gate"].shape == (2, 8, 4)
    assert weights["embedding"].dtype == np.float32
    assert weights["final_norm"].dtype == np.float32
    assert weights["w_out"].dtype == np.float32


def test_deepseek_ocr_moe_multiplies_only_routed_experts(monkeypatch) -> None:
    rhs_constant_matmuls: list[tuple[str, tuple[int, ...], bool]] = []
    routed_matmuls: list[tuple[str, str, str, str]] = []
    weight_gathers: list[tuple[str, str, int]] = []

    class Tensor:
        def __init__(self, name: str, shape=(), dtype="fp16") -> None:
            self.name = name
            self.shape = tuple(shape)
            self.dtype = dtype

    class Layer:
        def __init__(self, *outputs: Tensor) -> None:
            self.outputs = outputs

        def get_output(self, index: int) -> Tensor:
            return self.outputs[index]

        @property
        def reshape_dims(self):
            return self.outputs[0].shape

        @reshape_dims.setter
        def reshape_dims(self, shape) -> None:
            self.outputs[0].shape = tuple(shape)

    class Network:
        def add_softmax(self, tensor):
            return Layer(Tensor(f"softmax({tensor.name})", dtype=tensor.dtype))

        def add_topk(self, tensor, _operation, top_k, _axes):
            return Layer(
                Tensor("top_values", (-1, top_k), tensor.dtype),
                Tensor("top_indices", (-1, top_k), "int32"),
            )

        def add_reduce(self, tensor, _operation, _axes, keep_dims):
            del keep_dims
            return Layer(Tensor(f"reduce({tensor.name})", dtype=tensor.dtype))

        def add_elementwise(self, lhs, rhs, _operation):
            return Layer(Tensor(f"elementwise({lhs.name},{rhs.name})", dtype=lhs.dtype))

        def add_activation(self, tensor, _operation):
            return Layer(Tensor(f"activation({tensor.name})", dtype=tensor.dtype))

        def add_cast(self, tensor, dtype):
            return Layer(Tensor(f"cast({tensor.name})", dtype=dtype))

        def add_shuffle(self, tensor):
            return Layer(Tensor(f"shuffle({tensor.name})", tensor.shape, tensor.dtype))

        def add_gather(self, data, indices, axis):
            weight_gathers.append((data.name, indices.name, axis))
            return Layer(Tensor(f"gather({data.name})", dtype=data.dtype))

        def add_matrix_multiply(self, lhs, _lhs_op, rhs, _rhs_op):
            routed_matmuls.append((lhs.name, rhs.name, lhs.dtype, rhs.dtype))
            return Layer(Tensor(f"mm({lhs.name},{rhs.name})", dtype=lhs.dtype))

    def add_matmul_rhs_constant(
        _network, inp, _lhs_width, _rhs_width, values, dtype=np.float32,
        fp32_accumulation=True,
    ):
        del dtype
        rhs_constant_matmuls.append(
            (inp.name, values.shape, fp32_accumulation))
        return Tensor(f"rhs_mm_{len(rhs_constant_matmuls)}", dtype=inp.dtype)

    constants = 0

    def add_constant(_network, shape, _values, dtype=np.float32):
        nonlocal constants
        constants += 1
        tensor_dtype = "int32" if dtype == np.int32 else "fp16"
        return Tensor(f"weight{constants}", shape, tensor_dtype)

    monkeypatch.setattr(
        deepseek_ocr_module.graph_ops,
        "add_matmul_rhs_constant",
        add_matmul_rhs_constant,
    )
    monkeypatch.setattr(deepseek_ocr_module.graph_ops, "add_constant", add_constant)
    monkeypatch.setattr(
        deepseek_ocr_module.graph_ops,
        "trt",
        SimpleNamespace(float16="fp16", float32="fp32"),
    )

    hidden_size = 16
    intermediate_size = 32
    num_experts = 8
    weights = WeightDict({"layer.1.router": np.ones((hidden_size, num_experts))})
    weights["layer.1.experts.w_gate"] = np.ones(
        (num_experts, hidden_size, intermediate_size))
    weights["layer.1.experts.w_up"] = np.ones(
        (num_experts, hidden_size, intermediate_size))
    weights["layer.1.experts.w_down"] = np.ones(
        (num_experts, intermediate_size, hidden_size))
    weights["layer.1.shared.w_gate"] = np.ones((hidden_size, intermediate_size))
    weights["layer.1.shared.w_up"] = np.ones((hidden_size, intermediate_size))
    weights["layer.1.shared.w_down"] = np.ones((intermediate_size, hidden_size))

    output = deepseek_ocr_module._add_moe_with_shared_experts(
        Network(),
        Tensor("input", (-1, hidden_size)),
        weights,
        "layer.1",
        hidden_size,
        num_experts,
        intermediate_size,
        num_experts_per_tok=2,
        shared_intermediate=intermediate_size,
        dtype=np.float16,
    )

    assert output.dtype == "fp16"
    assert len(routed_matmuls) == 3
    assert all(
        lhs_dtype == rhs_dtype == "fp16"
        for _, _, lhs_dtype, rhs_dtype in routed_matmuls
    )
    assert len(rhs_constant_matmuls) == 4  # router + shared SwiGLU
    assert [fp32 for _, _, fp32 in rhs_constant_matmuls] == [
        True, False, False, False
    ]
    assert len(weight_gathers) == 3


def test_deepseek_ocr_vision_mask_keeps_image_attention_bidirectional() -> None:
    mask = deepseek_ocr_module._make_qwen2_vision_attention_mask(3)

    assert mask.shape == (6, 6)
    np.testing.assert_array_equal(mask[:3, :3], np.zeros((3, 3)))
    np.testing.assert_array_equal(mask[:3, 3:], np.full((3, 3), -10000.0))
    np.testing.assert_array_equal(
        mask[3], [0.0, 0.0, 0.0, 0.0, -10000.0, -10000.0])
    np.testing.assert_array_equal(
        mask[4], [0.0, 0.0, 0.0, 0.0, 0.0, -10000.0])
    np.testing.assert_array_equal(mask[5], np.zeros(6))


def test_deepseek_ocr_vision_mask_rejects_empty_image_sequence() -> None:
    with pytest.raises(ValueError, match="image_tokens must be positive"):
        deepseek_ocr_module._make_qwen2_vision_attention_mask(0)


def test_deepseek_ocr_resizes_sam_position_embedding_for_768_view() -> None:
    position_embedding = np.ones((1, 64, 64, 4), dtype=np.float32)

    resized = deepseek_ocr_module._resize_sam_position_embedding(
        position_embedding, 48)

    assert resized.shape == (1, 48, 48, 4)
    assert resized.dtype == np.float32
    np.testing.assert_allclose(resized, 1.0, atol=1e-6)


def test_deepseek_ocr_uses_official_768_single_view_contract() -> None:
    vl_config = deepseek_ocr_module.DeepSeekOCRPlugin().get_vl_config(_config())

    assert vl_config is not None
    assert vl_config["fixed_image_size"] == 768
    assert vl_config["num_image_pad_tokens"] == 145
    assert vl_config["prefill_max_length"] == 256
    assert vl_config["preprocessor_type"] == "simple_chw"


@pytest.mark.parametrize(
    ("max_cache_length", "expected"),
    [(4096, (64, 256)), (128, (64, 128)), (32, (32, 32))],
)
def test_deepseek_ocr_bounds_sequence_prefill_profile(
    max_cache_length: int, expected: tuple[int, int]
) -> None:
    assert sequence_prefill_profile_lengths(max_cache_length) == expected


def test_deepseek_ocr_decomposes_multirow_sequence_attention() -> None:
    trt = trt_compat.get_trt()
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    head_dim = 128
    hidden_size = 2 * head_dim
    q = network.add_input("q", trt.float16, (2, hidden_size))
    k = network.add_input("k", trt.float16, (5, hidden_size))
    v = network.add_input("v", trt.float16, (5, hidden_size))
    mask = network.add_input("mask", trt.float16, (2, 5))
    mask_4d = graph_ops.add_2d_mask_to_4d(network, mask)

    output = graph_ops.add_attention_from_rows(
        network, q, k, v,
        num_heads=2,
        num_kv_heads=2,
        head_dim=head_dim,
        q_seq=2,
        kv_seq=5,
        mask=mask_4d,
        fp32_accumulation=True,
    )

    assert output.shape == (2, hidden_size)
    layer_types = [
        network.get_layer(idx).type for idx in range(network.num_layers)
    ]
    assert layer_types.count(trt.LayerType.MATRIX_MULTIPLY) == 2
