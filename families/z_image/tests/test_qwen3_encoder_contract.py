# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Z-Image Qwen3 encoder builder contracts using owner-local TensorRT fakes."""

from __future__ import annotations

import types

import numpy as np
import pytest

from families.z_image import qwen3_encoder_builder as encoder


class _FakeTensor:
    next_id = 0

    def __init__(self, name: str | None = None, shape: tuple[int, ...] = (1, 4)):
        if name is None:
            type(self).next_id += 1
            name = f"t{type(self).next_id}"
        self.name = name
        self.dtype = None
        self.shape = shape


class _FakeLayer:
    def __init__(self, output: _FakeTensor | None = None, shape: tuple[int, ...] = (1, 4)):
        self._output = output or _FakeTensor()
        self._output.shape = shape
        self.reshape_dims = None
        self.second_transpose = None
        self.first_transpose = None
        self.axes = None
        self.axis = None
        self.output_types: list[tuple[int, object]] = []
        self.inputs: list[tuple[int, object]] = []

    def get_output(self, _index: int) -> _FakeTensor:
        return self._output

    def set_output_type(self, index: int, dtype: object) -> None:
        self.output_types.append((index, dtype))

    def set_input(self, index: int, tensor: object) -> None:
        self.inputs.append((index, tensor))


class _FakeNetwork:
    def __init__(self):
        self.inputs: list[tuple[str, object, tuple[int, ...]]] = []
        self.outputs: list[_FakeTensor] = []
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.layers: list[_FakeLayer] = []

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def _record(self, operation: str, *args, **kwargs) -> _FakeLayer:
        self.calls.append((operation, args, kwargs))
        shape = next((tuple(arg.shape) for arg in args if hasattr(arg, "shape")), (1, 4))
        layer = _FakeLayer(shape=shape)
        self.layers.append(layer)
        return layer

    def add_input(self, name: str, dtype: object, shape: tuple[int, ...]) -> _FakeTensor:
        self.inputs.append((name, dtype, shape))
        output = _FakeTensor(name, shape=shape)
        output.dtype = dtype
        self.calls.append(("add_input", (name, dtype, shape), {}))
        return output

    def add_constant(self, shape: tuple[int, ...], weights: object) -> _FakeLayer:
        self.calls.append(("add_constant", (shape, weights), {}))
        layer = _FakeLayer(shape=tuple(shape))
        self.layers.append(layer)
        return layer

    def add_cast(self, tensor, target_dtype, **kwargs) -> _FakeLayer:
        layer = self._record("add_cast", tensor, target_dtype, **kwargs)
        layer._output.dtype = target_dtype
        return layer

    def add_identity(self, *args, **kwargs):
        return self._record("add_identity", *args, **kwargs)

    def add_gather(self, *args, **kwargs):
        return self._record("add_gather", *args, **kwargs)

    def add_elementwise(self, *args, **kwargs):
        return self._record("add_elementwise", *args, **kwargs)

    def add_shuffle(self, *args, **kwargs):
        return self._record("add_shuffle", *args, **kwargs)

    def add_matrix_multiply(self, *args, **kwargs):
        return self._record("add_matrix_multiply", *args, **kwargs)

    def add_softmax(self, *args, **kwargs):
        return self._record("add_softmax", *args, **kwargs)

    def add_activation(self, *args, **kwargs):
        return self._record("add_activation", *args, **kwargs)

    def add_slice(self, *args, **kwargs):
        return self._record("add_slice", *args, **kwargs)

    def add_reduce(self, *args, **kwargs):
        return self._record("add_reduce", *args, **kwargs)

    def add_unary(self, *args, **kwargs):
        return self._record("add_unary", *args, **kwargs)

    def add_concatenation(self, *args, **kwargs):
        return self._record("add_concatenation", *args, **kwargs)

    def add_normalization_v2(self, *args, **kwargs):
        return self._record("add_normalization_v2", *args, **kwargs)

    def add_attention(self, *args, **kwargs):
        return self._record("add_attention", *args, **kwargs)

    def add_rotary_embedding(self, *args, **kwargs):
        return self._record("add_rotary_embedding", *args, **kwargs)

    def mark_output(self, tensor: _FakeTensor) -> None:
        self.calls.append(("mark_output", (tensor,), {}))
        self.outputs.append(tensor)


class _FakeBuilderConfig:
    def __init__(self):
        self.pool_limits: list[tuple[object, int]] = []
        self.cleared_flags: list[object] = []
        self.builder_optimization_level = 0

    def set_memory_pool_limit(self, pool: object, size: int) -> None:
        self.pool_limits.append((pool, size))

    def clear_flag(self, flag: object) -> None:
        self.cleared_flags.append(flag)


def _make_fake_trt() -> types.SimpleNamespace:
    class Logger:
        VERBOSE = 2
        WARNING = 1

        def __init__(self, level: int):
            self.level = level

    class Weights:
        def __init__(self, values: np.ndarray):
            self.values = np.asarray(values)

    class Builder:
        last_instance = None
        plan_to_return: bytes | None = b"engine-plan"

        def __init__(self, _logger: Logger):
            self.network = _FakeNetwork()
            self.config = _FakeBuilderConfig()
            type(self).last_instance = self

        def create_network(self, flags=0):
            del flags
            return self.network

        def create_builder_config(self):
            return self.config

        def build_serialized_network(self, network, config):
            del network, config
            return type(self).plan_to_return

    return types.SimpleNamespace(
        Logger=Logger,
        Builder=Builder,
        Weights=Weights,
        ElementWiseOperation=types.SimpleNamespace(SUM="sum", SUB="sub", PROD="prod"),
        ReduceOperation=types.SimpleNamespace(AVG="avg"),
        UnaryOperation=types.SimpleNamespace(SQRT="sqrt", RECIP="recip"),
        MatrixOperation=types.SimpleNamespace(NONE="none", TRANSPOSE="transpose"),
        ActivationType=types.SimpleNamespace(SIGMOID="sigmoid"),
        AttentionNormalizationOp=types.SimpleNamespace(SOFTMAX="softmax"),
        MemoryPoolType=types.SimpleNamespace(WORKSPACE="workspace"),
        BuilderFlag=types.SimpleNamespace(TF32="tf32"),
        NetworkDefinitionCreationFlag=types.SimpleNamespace(EXPLICIT_BATCH=0, STRONGLY_TYPED=1),
        Permutation=lambda dimensions: tuple(dimensions),
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
        int32="int32",
    )


def _fake_tensor_fn(prefix: str):
    counter = {"value": 0}

    def function(*_args, **_kwargs):
        counter["value"] += 1
        return _FakeTensor(f"{prefix}_{counter['value']}")

    return function


def _weights(hidden, layers, heads, kv_heads, head_dim, intermediate, vocab):
    values = {"embed_tokens": np.zeros((vocab, hidden), dtype=np.float32)}
    attention = heads * head_dim
    kv_dim = kv_heads * head_dim
    for index in range(layers):
        prefix = f"layer.{index}"
        values[f"{prefix}.q_proj"] = np.zeros((hidden, attention), dtype=np.float32)
        values[f"{prefix}.k_proj"] = np.zeros((hidden, kv_dim), dtype=np.float32)
        values[f"{prefix}.v_proj"] = np.zeros((hidden, kv_dim), dtype=np.float32)
        values[f"{prefix}.o_proj"] = np.zeros((attention, hidden), dtype=np.float32)
        values[f"{prefix}.q_norm"] = np.ones((head_dim,), dtype=np.float32)
        values[f"{prefix}.k_norm"] = np.ones((head_dim,), dtype=np.float32)
        values[f"{prefix}.input_layernorm"] = np.ones((hidden,), dtype=np.float32)
        values[f"{prefix}.post_attn_norm"] = np.ones((hidden,), dtype=np.float32)
        values[f"{prefix}.gate_proj"] = np.zeros((hidden, intermediate), dtype=np.float32)
        values[f"{prefix}.up_proj"] = np.zeros((hidden, intermediate), dtype=np.float32)
        values[f"{prefix}.down_proj"] = np.zeros((intermediate, hidden), dtype=np.float32)
    return values


def _install_common_fakes(monkeypatch, fake_trt):
    monkeypatch.setattr(encoder, "trt", fake_trt)
    monkeypatch.setattr(encoder.graph_ops, "add_constant", _fake_tensor_fn("const"))
    monkeypatch.setattr(encoder.graph_ops, "add_rms_norm", _fake_tensor_fn("rms"))
    monkeypatch.setattr(encoder.graph_ops, "add_matmul_rhs_constant", _fake_tensor_fn("mm"))


def test_build_qwen3_encoder_engine_success_with_gqa_and_negative_output_layer(monkeypatch) -> None:
    fake_trt = _make_fake_trt()
    _install_common_fakes(monkeypatch, fake_trt)
    attention_calls: list[dict[str, object]] = []

    def attention(*_args, **kwargs):
        attention_calls.append(kwargs)
        return _FakeTensor("native_gqa_attention")

    monkeypatch.setattr(encoder.graph_ops, "add_attention_from_rows", attention)
    plan = encoder.build_qwen3_encoder_engine(
        _weights(4, 2, 2, 1, 2, 6, 10),
        hidden_size=4,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        intermediate_size=6,
        vocab_size=10,
        max_seq_len=3,
        output_layer=-1,
    )

    assert plan == b"engine-plan"
    builder = fake_trt.Builder.last_instance
    assert builder.config.pool_limits == [("workspace", 64 << 30)]
    assert [tensor.name for tensor in builder.network.outputs] == ["text_embeddings"]
    assert [tensor.dtype for tensor in builder.network.outputs] == ["float32"]
    assert attention_calls and all(call["num_kv_heads"] == 1 for call in attention_calls)
    assert all(call["causal"] is True and call["mask"] is None for call in attention_calls)
    assert not any(op == "add_concatenation" for op, _args, _kwargs in builder.network.calls)


def test_qwen3_encoder_negative_output_layer_is_captured_after_selected_layer(monkeypatch) -> None:
    fake_trt = _make_fake_trt()
    _install_common_fakes(monkeypatch, fake_trt)
    monkeypatch.setattr(encoder.graph_ops, "add_attention_from_rows", _fake_tensor_fn("attention"))
    residual_outputs: list[_FakeTensor] = []
    original = _FakeNetwork.add_elementwise

    def recording(self, *args, **kwargs):
        layer = original(self, *args, **kwargs)
        if args[-1] == fake_trt.ElementWiseOperation.SUM and any(
            getattr(arg, "name", "").startswith("mm_") for arg in args[:-1]
        ):
            residual_outputs.append(layer.get_output(0))
        return layer

    monkeypatch.setattr(_FakeNetwork, "add_elementwise", recording)
    plan = encoder.build_qwen3_encoder_engine(
        _weights(4, 3, 2, 1, 2, 6, 10),
        hidden_size=4,
        num_layers=3,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        intermediate_size=6,
        vocab_size=10,
        max_seq_len=3,
        output_layer=-2,
    )
    assert plan == b"engine-plan"
    builder = fake_trt.Builder.last_instance
    cast_inputs = [args[0] for op, args, _kwargs in builder.network.calls if op == "add_cast"]
    assert len(residual_outputs) == 6
    assert cast_inputs[-1] is residual_outputs[3]


def test_build_qwen3_encoder_engine_raises_when_builder_returns_none(monkeypatch) -> None:
    fake_trt = _make_fake_trt()
    fake_trt.Builder.plan_to_return = None
    _install_common_fakes(monkeypatch, fake_trt)
    with pytest.raises(RuntimeError, match="Qwen3 encoder TRT engine build failed"):
        encoder.build_qwen3_encoder_engine(
            _weights(4, 1, 2, 2, 2, 6, 8),
            hidden_size=4,
            num_layers=1,
            num_heads=2,
            num_kv_heads=2,
            head_dim=2,
            intermediate_size=6,
            vocab_size=8,
            max_seq_len=2,
        )


def test_build_qwen3_encoder_engine_dynamic_batch_supports_fp16(monkeypatch) -> None:
    fake_trt = _make_fake_trt()
    monkeypatch.setattr(encoder, "trt", fake_trt)
    monkeypatch.setattr(encoder, "add_dynamic_batch_profile", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(encoder.graph_ops, "add_constant", _fake_tensor_fn("const"))
    dtype_calls: list[object] = []

    def dtype_operation(*_args, **kwargs):
        dtype_calls.append(kwargs.get("dtype"))
        return _FakeTensor("typed_op", shape=(-1, 3, 4))

    monkeypatch.setattr(encoder.graph_ops, "add_rms_norm_last_dim", dtype_operation)
    monkeypatch.setattr(encoder.graph_ops, "add_rms_norm_per_head_batched", dtype_operation)
    monkeypatch.setattr(encoder.graph_ops, "add_matmul_rhs_constant", dtype_operation)
    monkeypatch.setattr(encoder, "_add_apply_rope_native_batched", _fake_tensor_fn("rope"))
    monkeypatch.setattr(encoder, "_add_attention_from_batched_rows", _fake_tensor_fn("attention"))
    plan = encoder.build_qwen3_encoder_engine(
        _weights(4, 1, 2, 1, 2, 6, 8),
        hidden_size=4,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        intermediate_size=6,
        vocab_size=8,
        max_seq_len=3,
        precision="fp16",
        max_batch_size=2,
    )

    assert plan == b"engine-plan"
    builder = fake_trt.Builder.last_instance
    assert any(
        operation == "add_cast" and args[1] == fake_trt.float16
        for operation, args, _kwargs in builder.network.calls
    )
    assert dtype_calls and all(dtype == np.float16 for dtype in dtype_calls)
    assert [tensor.dtype for tensor in builder.network.outputs] == [fake_trt.float32]
