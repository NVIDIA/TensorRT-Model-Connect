# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mocked coverage tests for Flux-owned encoder builders.

These tests avoid real TensorRT execution by importing builders with a fake
`tensorrt` module and monkeypatching graph helper ops.

Trace: ARCH-ENG-001, UD-ENG-ENCODER
Intent: Validate Flux encoder builder code paths with fake TRT and graph op stubs
Preconditions: Fake tensorrt module and graph op stubs are injected
Postconditions: Flux encoder builders execute expected branches and mark outputs
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


_PKG_ROOT = Path(__file__).resolve().parents[4] / "python"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))


class _FakeTensor:
    _next_id = 0

    def __init__(self, name: str | None = None, shape: tuple[int, ...] = (1, 4)):
        if name is None:
            type(self)._next_id += 1
            name = f"t{type(self)._next_id}"
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

    def _record(self, op: str, *args, **kwargs) -> _FakeLayer:
        self.calls.append((op, args, kwargs))
        shape = next((tuple(arg.shape) for arg in args if hasattr(arg, "shape")), (1, 4))
        return _FakeLayer(shape=shape)

    def add_input(self, name: str, dtype: object, shape: tuple[int, ...]) -> _FakeTensor:
        self.inputs.append((name, dtype, shape))
        out = _FakeTensor(name, shape=shape)
        out.dtype = dtype
        self.calls.append(("add_input", (name, dtype, shape), {}))
        return out

    def add_constant(self, shape: tuple[int, ...], weights: object) -> _FakeLayer:
        self.calls.append(("add_constant", (shape, weights), {}))
        return _FakeLayer(shape=tuple(shape))

    def add_cast(self, tensor, target_dtype, **kwargs) -> _FakeLayer:
        layer = self._record("add_cast", tensor, target_dtype, **kwargs)
        layer._output.dtype = target_dtype
        return layer

    def add_identity(self, *args, **kwargs) -> _FakeLayer:
        return self._record("add_identity", *args, **kwargs)

    def add_gather(self, *args, **kwargs) -> _FakeLayer:
        return self._record("add_gather", *args, **kwargs)

    def add_elementwise(self, *args, **kwargs) -> _FakeLayer:
        return self._record("add_elementwise", *args, **kwargs)

    def add_shuffle(self, *args, **kwargs) -> _FakeLayer:
        return self._record("add_shuffle", *args, **kwargs)

    def add_matrix_multiply(self, *args, **kwargs) -> _FakeLayer:
        return self._record("add_matrix_multiply", *args, **kwargs)

    def add_softmax(self, *args, **kwargs) -> _FakeLayer:
        return self._record("add_softmax", *args, **kwargs)

    def add_activation(self, *args, **kwargs) -> _FakeLayer:
        return self._record("add_activation", *args, **kwargs)

    def add_slice(self, *args, **kwargs) -> _FakeLayer:
        return self._record("add_slice", *args, **kwargs)

    def add_reduce(self, *args, **kwargs) -> _FakeLayer:
        return self._record("add_reduce", *args, **kwargs)

    def add_unary(self, *args, **kwargs) -> _FakeLayer:
        return self._record("add_unary", *args, **kwargs)

    def add_concatenation(self, *args, **kwargs) -> _FakeLayer:
        return self._record("add_concatenation", *args, **kwargs)

    def add_normalization_v2(self, *args, **kwargs) -> _FakeLayer:
        return self._record("add_normalization_v2", *args, **kwargs)

    def add_attention(self, *args, **kwargs) -> _FakeLayer:
        return self._record("add_attention", *args, **kwargs)

    def mark_output(self, tensor: _FakeTensor) -> None:
        self.calls.append(("mark_output", (tensor,), {}))
        self.outputs.append(tensor)


class _FakeBuilderConfig:
    def __init__(self):
        self.pool_limits: list[tuple[object, int]] = []
        self.cleared_flags: list[object] = []

    def set_memory_pool_limit(self, pool: object, size: int) -> None:
        self.pool_limits.append((pool, size))

    def clear_flag(self, flag: object) -> None:
        self.cleared_flags.append(flag)


def _make_fake_trt() -> types.SimpleNamespace:
    class _Logger:
        VERBOSE = 2
        WARNING = 1

        def __init__(self, level: int):
            self.level = level

    class _Weights:
        def __init__(self, values: np.ndarray):
            self.values = np.asarray(values)

    class _Builder:
        last_instance = None
        plan_to_return: bytes | None = b"engine-plan"

        def __init__(self, _logger: _Logger):
            self.network = _FakeNetwork()
            self.config = _FakeBuilderConfig()
            self.build_calls: list[tuple[_FakeNetwork, _FakeBuilderConfig]] = []
            type(self).last_instance = self

        def create_network(self, flags=0):
            return self.network

        def create_builder_config(self):
            return self.config

        def build_serialized_network(self, network: _FakeNetwork, config: _FakeBuilderConfig):
            self.build_calls.append((network, config))
            return type(self).plan_to_return

    return types.SimpleNamespace(
        Logger=_Logger,
        Builder=_Builder,
        Weights=_Weights,
        ElementWiseOperation=types.SimpleNamespace(SUM="sum", SUB="sub", PROD="prod"),
        ReduceOperation=types.SimpleNamespace(AVG="avg"),
        UnaryOperation=types.SimpleNamespace(SQRT="sqrt", RECIP="recip"),
        MatrixOperation=types.SimpleNamespace(NONE="none", TRANSPOSE="transpose"),
        ActivationType=types.SimpleNamespace(SIGMOID="sigmoid"),
        AttentionNormalizationOp=types.SimpleNamespace(SOFTMAX="softmax"),
        MemoryPoolType=types.SimpleNamespace(WORKSPACE="workspace"),
        BuilderFlag=types.SimpleNamespace(TF32="tf32"),
        NetworkDefinitionCreationFlag=types.SimpleNamespace(EXPLICIT_BATCH=0, STRONGLY_TYPED=1),
        Permutation=lambda dims: tuple(dims),
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
        int32="int32",
    )


def _drop_imported_module(module_name: str) -> None:
    sys.modules.pop(module_name, None)
    package_name, _, attribute_name = module_name.rpartition(".")
    package = sys.modules.get(package_name)
    if package is not None and hasattr(package, attribute_name):
        delattr(package, attribute_name)


def _import_with_fake_trt(module_name: str, fake_trt: types.SimpleNamespace):
    """Import a tensorrt_model_connect module while tensorrt is mocked."""
    for mod_name in (
        module_name,
        "tensorrt_model_connect.graph_ops",
        "tensorrt_model_connect.graph_blocks",
    ):
        _drop_imported_module(mod_name)
    previous_trt = sys.modules.get("tensorrt")
    sys.modules["tensorrt"] = fake_trt
    try:
        return importlib.import_module(module_name)
    finally:
        if previous_trt is None:
            sys.modules.pop("tensorrt", None)
        else:
            sys.modules["tensorrt"] = previous_trt


def _fake_tensor_fn(prefix: str):
    counter = {"n": 0}

    def _fn(*_args, **_kwargs):
        counter["n"] += 1
        return _FakeTensor(f"{prefix}_{counter['n']}")

    return _fn


def _make_clip_weights(hidden: int, intermediate: int, vocab: int, seq_len: int, num_layers: int):
    w: dict[str, np.ndarray] = {
        "text_model.embeddings.token_embedding.weight": np.zeros((vocab, hidden), dtype=np.float32),
        "text_model.embeddings.position_embedding.weight": np.zeros((seq_len, hidden), dtype=np.float32),
        "text_model.final_layer_norm.weight": np.ones((hidden,), dtype=np.float32),
        "text_model.final_layer_norm.bias": np.zeros((hidden,), dtype=np.float32),
    }
    for i in range(num_layers):
        p = f"text_model.encoder.layers.{i}"
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            w[f"{p}.self_attn.{proj}.weight"] = np.zeros((hidden, hidden), dtype=np.float32)
            w[f"{p}.self_attn.{proj}.bias"] = np.zeros((hidden,), dtype=np.float32)
        w[f"{p}.layer_norm1.weight"] = np.ones((hidden,), dtype=np.float32)
        w[f"{p}.layer_norm1.bias"] = np.zeros((hidden,), dtype=np.float32)
        w[f"{p}.layer_norm2.weight"] = np.ones((hidden,), dtype=np.float32)
        w[f"{p}.layer_norm2.bias"] = np.zeros((hidden,), dtype=np.float32)
        w[f"{p}.mlp.fc1.weight"] = np.zeros((hidden, intermediate), dtype=np.float32)
        w[f"{p}.mlp.fc1.bias"] = np.zeros((intermediate,), dtype=np.float32)
        w[f"{p}.mlp.fc2.weight"] = np.zeros((intermediate, hidden), dtype=np.float32)
        w[f"{p}.mlp.fc2.bias"] = np.zeros((hidden,), dtype=np.float32)
    return w


def _make_t5_weights(
    d_model: int,
    num_heads: int,
    d_ff: int,
    num_layers: int,
    vocab_size: int,
    num_buckets: int,
):
    attn = d_model
    w: dict[str, np.ndarray] = {
        "shared.weight": np.zeros((vocab_size, d_model), dtype=np.float32),
        "encoder.final_layer_norm.weight": np.ones((d_model,), dtype=np.float32),
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": np.arange(
            num_buckets * num_heads, dtype=np.float32
        ).reshape(num_buckets, num_heads),
    }
    for i in range(num_layers):
        p = f"encoder.block.{i}"
        for proj in ("q", "k", "v", "o"):
            w[f"{p}.layer.0.SelfAttention.{proj}.weight"] = np.zeros((d_model, attn), dtype=np.float32)
        w[f"{p}.layer.0.layer_norm.weight"] = np.ones((d_model,), dtype=np.float32)
        w[f"{p}.layer.1.layer_norm.weight"] = np.ones((d_model,), dtype=np.float32)
        w[f"{p}.layer.1.DenseReluDense.wi_0.weight"] = np.zeros((d_model, d_ff), dtype=np.float32)
        w[f"{p}.layer.1.DenseReluDense.wi_1.weight"] = np.zeros((d_model, d_ff), dtype=np.float32)
        w[f"{p}.layer.1.DenseReluDense.wo.weight"] = np.zeros((d_ff, d_model), dtype=np.float32)
    return w


@pytest.mark.unit
def test_build_clip_encoder_engine_success_uses_fake_builder_and_marks_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: verify CLIP builder success path with mocked TRT/graph helpers.

    Preconditions: CLIP builder module is imported with fake tensorrt and minimal weights.
    Postconditions: Engine bytes are returned, config flags are set, and both outputs are marked.
    """
    fake_trt = _make_fake_trt()
    mod = _import_with_fake_trt("tensorrt_model_connect.families.flux.model.components.clip_encoder", fake_trt)

    constant_payloads: list[tuple[tuple[int, ...], np.ndarray]] = []

    def _fake_add_constant(_network, shape, values, **_kwargs):
        constant_payloads.append((tuple(shape), np.asarray(values)))
        return _FakeTensor(f"const_{len(constant_payloads)}")

    monkeypatch.setattr(mod.graph_ops, "add_constant", _fake_add_constant)
    monkeypatch.setattr(mod.graph_ops, "add_layer_norm_native", _fake_tensor_fn("ln"))
    monkeypatch.setattr(mod.graph_ops, "add_matmul_rhs_constant", _fake_tensor_fn("mm"))
    monkeypatch.setattr(mod.graph_ops, "add_bias_sum", _fake_tensor_fn("bias"))
    monkeypatch.setattr(mod.graph_ops, "add_attention_from_rows", _fake_tensor_fn("attention"))

    plan = mod.build_clip_encoder_engine(
        _make_clip_weights(hidden=4, intermediate=8, vocab=16, seq_len=3, num_layers=1),
        hidden_size=4,
        num_heads=2,
        intermediate_size=8,
        num_layers=1,
        vocab_size=16,
        max_seq_len=3,
        verbose=True,
    )

    assert plan == b"engine-plan"
    builder = fake_trt.Builder.last_instance
    assert builder.config.pool_limits == [("workspace", 4 << 30)]
    assert builder.config.cleared_flags == []
    assert [t.name for t in builder.network.outputs] == ["text_embeddings", "pooled_output"]
    assert [t.dtype for t in builder.network.outputs] == ["float32", "float32"]
    assert any(shape == (1, 1, 3, 3) for shape, _ in constant_payloads)


@pytest.mark.unit
def test_build_clip_encoder_engine_raises_when_builder_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: validate CLIP builder error path when TRT serialization fails.

    Preconditions: Fake TRT builder returns `None` instead of a serialized plan.
    Postconditions: `RuntimeError` is raised with CLIP-specific failure text.
    """
    fake_trt = _make_fake_trt()
    fake_trt.Builder.plan_to_return = None
    mod = _import_with_fake_trt("tensorrt_model_connect.families.flux.model.components.clip_encoder", fake_trt)

    monkeypatch.setattr(mod.graph_ops, "add_constant", _fake_tensor_fn("const"))
    monkeypatch.setattr(mod.graph_ops, "add_layer_norm_native", _fake_tensor_fn("ln"))
    monkeypatch.setattr(mod.graph_ops, "add_matmul_rhs_constant", _fake_tensor_fn("mm"))
    monkeypatch.setattr(mod.graph_ops, "add_bias_sum", _fake_tensor_fn("bias"))
    monkeypatch.setattr(mod.graph_ops, "add_attention_from_rows", _fake_tensor_fn("attention"))

    with pytest.raises(RuntimeError, match="CLIP encoder"):
        mod.build_clip_encoder_engine(
            _make_clip_weights(hidden=4, intermediate=8, vocab=8, seq_len=2, num_layers=1),
            hidden_size=4,
            num_heads=2,
            intermediate_size=8,
            num_layers=1,
            vocab_size=8,
            max_seq_len=2,
        )


@pytest.mark.unit
def test_load_clip_weights_transposes_projection_matrices_and_keeps_biases() -> None:
    """Intent: verify CLIP weight loader key mapping and transpose behavior.

    Preconditions: checkpoint mapper functions are replaced with deterministic fake tensor readers.
    Postconditions: Projection matrices are transposed while scalar/vector tensors remain untransformed float32.
    """
    fake_trt = _make_fake_trt()
    mod = _import_with_fake_trt("tensorrt_model_connect.families.flux.model.components.clip_encoder", fake_trt)

    tensors: dict[str, np.ndarray] = {
        "text_model.embeddings.token_embedding.weight": np.arange(32, dtype=np.float32).reshape(8, 4),
        "text_model.embeddings.position_embedding.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
        "text_model.final_layer_norm.weight": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "text_model.final_layer_norm.bias": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
    }

    for layer in range(2):
        p = f"text_model.encoder.layers.{layer}"
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            tensors[f"{p}.self_attn.{proj}.weight"] = (
                np.arange(16, dtype=np.float32).reshape(4, 4) + (10 * layer)
            )
            tensors[f"{p}.self_attn.{proj}.bias"] = np.arange(4, dtype=np.float32) + layer
        tensors[f"{p}.layer_norm1.weight"] = np.ones((4,), dtype=np.float32) * (layer + 1)
        tensors[f"{p}.layer_norm1.bias"] = np.zeros((4,), dtype=np.float32)
        tensors[f"{p}.layer_norm2.weight"] = np.ones((4,), dtype=np.float32) * (layer + 2)
        tensors[f"{p}.layer_norm2.bias"] = np.zeros((4,), dtype=np.float32)
        tensors[f"{p}.mlp.fc1.weight"] = np.arange(24, dtype=np.float32).reshape(6, 4) + layer
        tensors[f"{p}.mlp.fc1.bias"] = np.arange(6, dtype=np.float32) + layer
        tensors[f"{p}.mlp.fc2.weight"] = np.arange(24, dtype=np.float32).reshape(4, 6) + layer
        tensors[f"{p}.mlp.fc2.bias"] = np.arange(4, dtype=np.float32) + layer

    cm = importlib.import_module("tensorrt_model_connect.families.flux.weights")

    with patch.object(cm, "_open_safetensors", lambda _path: tensors), patch.object(
        cm, "_load_tensor", lambda readers, name: readers[name]
    ):
        weights = mod.load_clip_weights("unused", hidden_size=4, num_layers=2)

    np.testing.assert_allclose(
        weights["text_model.encoder.layers.1.self_attn.q_proj.weight"],
        tensors["text_model.encoder.layers.1.self_attn.q_proj.weight"].T.astype(np.float32),
    )
    np.testing.assert_allclose(
        weights["text_model.encoder.layers.0.mlp.fc2.weight"],
        tensors["text_model.encoder.layers.0.mlp.fc2.weight"].T.astype(np.float32),
    )
    np.testing.assert_allclose(
        weights["text_model.encoder.layers.1.self_attn.q_proj.bias"],
        tensors["text_model.encoder.layers.1.self_attn.q_proj.bias"],
    )
    assert weights["text_model.final_layer_norm.bias"].dtype == np.float32


@pytest.mark.unit
def test_build_t5_encoder_engine_success_exercises_relative_bias_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: verify T5 builder success path and layer-0 bias fallback behavior.

    Preconditions: Only layer-0 relative-attention bias exists in weights and graph ops are mocked.
    Postconditions: Engine bytes are returned and each layer emits the expected derived bias constant.
    """
    fake_trt = _make_fake_trt()
    mod = _import_with_fake_trt("tensorrt_model_connect.families.flux.model.components.t5_encoder", fake_trt)

    bucket_indices = np.array([[0, 1], [2, 3]], dtype=np.int32)
    constant_payloads: list[tuple[tuple[int, ...], np.ndarray]] = []

    def _fake_add_constant(_network, shape, values, **_kwargs):
        constant_payloads.append((tuple(shape), np.asarray(values)))
        return _FakeTensor(f"const_{len(constant_payloads)}")

    monkeypatch.setattr(mod.graph_ops, "add_constant", _fake_add_constant)
    monkeypatch.setattr(mod.graph_ops, "make_t5_relative_position_bias", lambda *_a, **_k: bucket_indices)
    monkeypatch.setattr(mod.graph_ops, "add_rms_norm", _fake_tensor_fn("rms"))
    monkeypatch.setattr(mod.graph_ops, "add_matmul_rhs_constant", _fake_tensor_fn("mm"))
    monkeypatch.setattr(mod.graph_ops, "add_gelu_new", _fake_tensor_fn("gelu"))
    monkeypatch.setattr(mod.graph_ops, "add_attention_from_rows", _fake_tensor_fn("attention"))

    weights = _make_t5_weights(
        d_model=4,
        num_heads=2,
        d_ff=6,
        num_layers=2,
        vocab_size=8,
        num_buckets=4,
    )

    plan = mod.build_t5_encoder_engine(
        weights,
        d_model=4,
        num_heads=2,
        d_kv=2,
        d_ff=6,
        num_layers=2,
        vocab_size=8,
        max_seq_len=2,
        relative_attention_num_buckets=4,
    )

    assert plan == b"engine-plan"
    builder = fake_trt.Builder.last_instance
    assert builder.config.pool_limits == [("workspace", 1 << 30)]
    assert [t.name for t in builder.network.outputs] == ["text_embeddings"]
    assert [t.dtype for t in builder.network.outputs] == ["float32"]

    layer0_bias = weights["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"]
    expected = np.zeros((2, 2, 2), dtype=np.float32)
    for q in range(2):
        for k in range(2):
            b = bucket_indices[q, k]
            for h in range(2):
                expected[h, q, k] = layer0_bias[b, h]

    bias_constants = [values for shape, values in constant_payloads if shape == (2, 2, 2)]
    assert len(bias_constants) == 2
    np.testing.assert_allclose(bias_constants[0], expected)
    np.testing.assert_allclose(bias_constants[1], expected)


@pytest.mark.unit
def test_build_t5_encoder_engine_raises_when_builder_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: validate T5 builder failure handling when TRT returns no plan.

    Preconditions: Fake TRT builder is configured to return `None` during serialization.
    Postconditions: Builder raises `RuntimeError` with T5-specific failure message.
    """
    fake_trt = _make_fake_trt()
    fake_trt.Builder.plan_to_return = None
    mod = _import_with_fake_trt("tensorrt_model_connect.families.flux.model.components.t5_encoder", fake_trt)

    monkeypatch.setattr(mod.graph_ops, "add_constant", _fake_tensor_fn("const"))
    monkeypatch.setattr(mod.graph_ops, "make_t5_relative_position_bias", lambda *_a, **_k: np.zeros((2, 2), dtype=np.int32))
    monkeypatch.setattr(mod.graph_ops, "add_rms_norm", _fake_tensor_fn("rms"))
    monkeypatch.setattr(mod.graph_ops, "add_matmul_rhs_constant", _fake_tensor_fn("mm"))
    monkeypatch.setattr(mod.graph_ops, "add_gelu_new", _fake_tensor_fn("gelu"))
    monkeypatch.setattr(mod.graph_ops, "add_attention_from_rows", _fake_tensor_fn("attention"))

    with pytest.raises(RuntimeError, match="T5 encoder"):
        mod.build_t5_encoder_engine(
            _make_t5_weights(d_model=4, num_heads=2, d_ff=6, num_layers=1, vocab_size=8, num_buckets=4),
            d_model=4,
            num_heads=2,
            d_kv=2,
            d_ff=6,
            num_layers=1,
            vocab_size=8,
            max_seq_len=2,
            relative_attention_num_buckets=4,
        )


@pytest.mark.unit
def test_load_t5_weights_transposes_and_bias_fallback() -> None:
    """Verify Flux's T5 loader owns its transforms and layer-0 bias fallback."""
    fake_trt = _make_fake_trt()
    mod = _import_with_fake_trt(
        "tensorrt_model_connect.families.flux.model.components.t5_encoder", fake_trt
    )

    tensors: dict[str, np.ndarray] = {
        "shared.weight": np.arange(28, dtype=np.float32).reshape(7, 4),
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": np.arange(
            64, dtype=np.float32
        ).reshape(32, 2),
        "encoder.final_layer_norm.weight": np.array(
            [0.1, 0.2, 0.3, 0.4], dtype=np.float32
        ),
    }

    for layer in range(2):
        prefix = f"encoder.block.{layer}"
        for offset, projection in enumerate(("q", "k", "v", "o")):
            tensors[f"{prefix}.layer.0.SelfAttention.{projection}.weight"] = (
                np.arange(16, dtype=np.float32).reshape(4, 4) + layer + offset
            )
        tensors[f"{prefix}.layer.0.layer_norm.weight"] = (
            np.arange(4, dtype=np.float32) + layer
        )
        for offset, projection in enumerate(("wi_0", "wi_1")):
            tensors[f"{prefix}.layer.1.DenseReluDense.{projection}.weight"] = (
                np.arange(24, dtype=np.float32).reshape(6, 4) + layer + offset
            )
        tensors[f"{prefix}.layer.1.DenseReluDense.wo.weight"] = (
            np.arange(24, dtype=np.float32).reshape(4, 6) + layer + 2
        )
        tensors[f"{prefix}.layer.1.layer_norm.weight"] = (
            np.arange(4, dtype=np.float32) + layer + 10
        )

    checkpoint_mapper = importlib.import_module(
        "tensorrt_model_connect.families.flux.weights"
    )

    def load(precision: str = "fp32"):
        with patch.object(
            checkpoint_mapper, "_open_safetensors", lambda _path: tensors
        ), patch.object(
            checkpoint_mapper, "_load_tensor", lambda readers, name: readers[name]
        ), patch.object(
            checkpoint_mapper, "_has_tensor", lambda readers, name: name in readers
        ), patch.object(
            checkpoint_mapper,
            "_target_np_dtype",
            lambda value: np.float16 if value == "fp16" else np.float32,
        ):
            return mod.load_t5_weights(
                model_dir="unused",
                d_model=4,
                num_heads=2,
                d_kv=2,
                d_ff=6,
                num_layers=2,
                vocab_size=7,
                precision=precision,
            )

    weights = load()
    np.testing.assert_allclose(
        weights["encoder.block.0.layer.0.SelfAttention.q.weight"],
        tensors["encoder.block.0.layer.0.SelfAttention.q.weight"].T,
    )
    np.testing.assert_allclose(
        weights["encoder.block.1.layer.1.DenseReluDense.wi_0.weight"],
        tensors["encoder.block.1.layer.1.DenseReluDense.wi_0.weight"].T,
    )
    assert "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight" in weights
    assert "encoder.block.1.layer.0.SelfAttention.relative_attention_bias.weight" not in weights
    assert weights["encoder.final_layer_norm.weight"].dtype == np.float32

    fp16_weights = load("fp16")
    assert fp16_weights["shared.weight"].dtype == np.float16
    assert (
        fp16_weights["encoder.block.0.layer.0.SelfAttention.q.weight"].dtype
        == np.float16
    )
    assert fp16_weights["encoder.block.0.layer.0.layer_norm.weight"].dtype == np.float32
