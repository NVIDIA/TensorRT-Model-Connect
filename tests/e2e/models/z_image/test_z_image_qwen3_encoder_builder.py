# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mocked coverage tests for the Z-Image-owned Qwen3 encoder builder.

These tests avoid real TensorRT execution by importing builders with a fake
`tensorrt` module and monkeypatching graph helper ops.

Trace: ARCH-ENG-001, UD-ENG-ENCODER
Intent: Validate Z-Image Qwen3 encoder builder code paths with fake TRT stubs
Preconditions: Fake tensorrt module and graph op stubs are injected
Postconditions: Qwen3 encoder builder executes expected branches and marks outputs
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

    def add_rotary_embedding(self, *args, **kwargs) -> _FakeLayer:
        return self._record("add_rotary_embedding", *args, **kwargs)

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


def _import_qwen3_with_fake_trt(fake_trt: types.SimpleNamespace):
    """Import qwen3 builder with a fake family checkpoint mapper."""
    checkpoint_module = "tensorrt_model_connect.families.z_image.checkpoint_mapper"
    fake_cm = types.ModuleType(checkpoint_module)

    class _WeightDict(dict):
        pass

    fake_cm.WeightDict = _WeightDict  # type: ignore[attr-defined]
    fake_cm._open_safetensors = lambda _path: {}  # type: ignore[attr-defined]
    fake_cm._load_tensor = lambda _readers, _name: np.array([], dtype=np.float32)  # type: ignore[attr-defined]
    fake_cm._has_tensor = lambda _readers, _name: False  # type: ignore[attr-defined]

    for mod_name in (
        "tensorrt_model_connect.families.z_image.qwen3_encoder_builder",
        checkpoint_module,
        "tensorrt_model_connect.graph_ops",
        "tensorrt_model_connect.graph_blocks",
    ):
        _drop_imported_module(mod_name)
    with patch.dict(sys.modules, {"tensorrt": fake_trt, checkpoint_module: fake_cm}):
        return importlib.import_module("tensorrt_model_connect.families.z_image.qwen3_encoder_builder")


def _fake_tensor_fn(prefix: str):
    counter = {"n": 0}

    def _fn(*_args, **_kwargs):
        counter["n"] += 1
        return _FakeTensor(f"{prefix}_{counter['n']}")

    return _fn


def _make_qwen3_weights(
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    intermediate_size: int,
    vocab_size: int,
):
    w: dict[str, np.ndarray] = {
        "embed_tokens": np.zeros((vocab_size, hidden_size), dtype=np.float32),
    }
    attn_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    for i in range(num_layers):
        p = f"layer.{i}"
        w[f"{p}.q_proj"] = np.zeros((hidden_size, attn_dim), dtype=np.float32)
        w[f"{p}.k_proj"] = np.zeros((hidden_size, kv_dim), dtype=np.float32)
        w[f"{p}.v_proj"] = np.zeros((hidden_size, kv_dim), dtype=np.float32)
        w[f"{p}.o_proj"] = np.zeros((attn_dim, hidden_size), dtype=np.float32)
        w[f"{p}.q_norm"] = np.ones((head_dim,), dtype=np.float32)
        w[f"{p}.k_norm"] = np.ones((head_dim,), dtype=np.float32)
        w[f"{p}.input_layernorm"] = np.ones((hidden_size,), dtype=np.float32)
        w[f"{p}.post_attn_norm"] = np.ones((hidden_size,), dtype=np.float32)
        w[f"{p}.gate_proj"] = np.zeros((hidden_size, intermediate_size), dtype=np.float32)
        w[f"{p}.up_proj"] = np.zeros((hidden_size, intermediate_size), dtype=np.float32)
        w[f"{p}.down_proj"] = np.zeros((intermediate_size, hidden_size), dtype=np.float32)
    return w


@pytest.mark.unit
def test_build_qwen3_encoder_engine_success_with_gqa_and_negative_output_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: verify Qwen3 builder success path with GQA and negative output-layer selection.

    Preconditions: num_kv_heads differs from num_heads and graph ops return fake tensors.
    Postconditions: Engine bytes are returned, output tensor is marked, and native GQA is requested.
    """
    fake_trt = _make_fake_trt()
    mod = _import_qwen3_with_fake_trt(fake_trt)

    monkeypatch.setattr(mod.graph_ops, "add_constant", _fake_tensor_fn("const"))
    monkeypatch.setattr(mod.graph_ops, "add_rms_norm", _fake_tensor_fn("rms"))
    monkeypatch.setattr(mod.graph_ops, "add_matmul_rhs_constant", _fake_tensor_fn("mm"))
    attention_calls: list[dict[str, object]] = []

    def fake_attention_from_rows(*_args, **kwargs):
        attention_calls.append(kwargs)
        return _FakeTensor("native_gqa_attention")

    monkeypatch.setattr(mod.graph_ops, "add_attention_from_rows", fake_attention_from_rows)

    weights = _make_qwen3_weights(
        hidden_size=4,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        intermediate_size=6,
        vocab_size=10,
    )

    plan = mod.build_qwen3_encoder_engine(
        weights,
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
    assert builder.config.cleared_flags == []
    assert [t.name for t in builder.network.outputs] == ["text_embeddings"]
    assert [t.dtype for t in builder.network.outputs] == ["float32"]
    assert attention_calls
    assert all(call["num_kv_heads"] == 1 for call in attention_calls)
    assert not any(op == "add_concatenation" for op, _args, _kwargs in builder.network.calls)


@pytest.mark.unit
def test_build_qwen3_encoder_engine_output_layer_overflow_falls_back_to_final_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: cover Qwen3 output-layer overflow branch that returns final hidden state.

    Preconditions: Requested output layer is greater than `num_layers` and heads are non-GQA.
    Postconditions: Build succeeds and output tensor is still marked without raising unbound-variable errors.
    """
    fake_trt = _make_fake_trt()
    mod = _import_qwen3_with_fake_trt(fake_trt)

    monkeypatch.setattr(mod.graph_ops, "add_constant", _fake_tensor_fn("const"))
    monkeypatch.setattr(mod.graph_ops, "add_rms_norm", _fake_tensor_fn("rms"))
    monkeypatch.setattr(mod.graph_ops, "add_matmul_rhs_constant", _fake_tensor_fn("mm"))

    plan = mod.build_qwen3_encoder_engine(
        _make_qwen3_weights(
            hidden_size=4,
            num_layers=1,
            num_heads=2,
            num_kv_heads=2,
            head_dim=2,
            intermediate_size=6,
            vocab_size=8,
        ),
        hidden_size=4,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        head_dim=2,
        intermediate_size=6,
        vocab_size=8,
        max_seq_len=2,
        output_layer=99,
    )

    assert plan == b"engine-plan"
    builder = fake_trt.Builder.last_instance
    assert [t.name for t in builder.network.outputs] == ["text_embeddings"]


@pytest.mark.unit
def test_build_qwen3_encoder_engine_raises_when_builder_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: validate Qwen3 builder error path when TRT serialization fails.

    Preconditions: Fake TRT builder is configured to return `None` for serialized engine.
    Postconditions: Qwen3-specific `RuntimeError` is raised.
    """
    fake_trt = _make_fake_trt()
    fake_trt.Builder.plan_to_return = None
    mod = _import_qwen3_with_fake_trt(fake_trt)

    monkeypatch.setattr(mod.graph_ops, "add_constant", _fake_tensor_fn("const"))
    monkeypatch.setattr(mod.graph_ops, "add_rms_norm", _fake_tensor_fn("rms"))
    monkeypatch.setattr(mod.graph_ops, "add_matmul_rhs_constant", _fake_tensor_fn("mm"))

    with pytest.raises(RuntimeError, match="Qwen3 encoder TRT engine build failed"):
        mod.build_qwen3_encoder_engine(
            _make_qwen3_weights(
                hidden_size=4,
                num_layers=1,
                num_heads=2,
                num_kv_heads=2,
                head_dim=2,
                intermediate_size=6,
                vocab_size=8,
            ),
            hidden_size=4,
            num_layers=1,
            num_heads=2,
            num_kv_heads=2,
            head_dim=2,
            intermediate_size=6,
            vocab_size=8,
            max_seq_len=2,
        )
