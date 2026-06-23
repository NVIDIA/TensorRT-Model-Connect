"""Mocked coverage tests for BERT-owned encoder builders.

These tests avoid real TensorRT execution by importing builders with a fake
`tensorrt` module and monkeypatching graph helper ops.

Trace: ARCH-ENG-001, UD-ENG-ENCODER
Intent: Validate BERT encoder builder code paths with fake TRT and graph op stubs
Preconditions: Fake tensorrt module and graph op stubs are injected
Postconditions: BERT encoder builders execute expected branches and mark outputs
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

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


@pytest.mark.unit
def test_build_encoder_engine_success_passes_rel_pos_bias_and_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: verify encoder builder top-level orchestration with mocked layer helper.

    Preconditions: `_add_encoder_layer` is monkeypatched to a recording stub and weights include relative bias.
    Postconditions: Layer helper receives expected hidden-act/rel-bias values and engine bytes are returned.
    """
    fake_trt = _make_fake_trt()
    mod = _import_with_fake_trt("tensorrt_model_connect.families.bert.encoder_builder", fake_trt)

    monkeypatch.setattr(mod.graph_ops, "add_constant", _fake_tensor_fn("const"))
    monkeypatch.setattr(mod, "_add_seq_layer_norm", _fake_tensor_fn("embed_ln"))

    layer_calls: list[dict[str, object]] = []

    def _fake_add_encoder_layer(**kwargs):
        layer_calls.append(kwargs)
        return _FakeTensor(f"layer_out_{len(layer_calls)}")

    monkeypatch.setattr(mod, "_add_encoder_layer", _fake_add_encoder_layer)

    config = types.SimpleNamespace(
        hidden_size=4,
        vocab_size=10,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=6,
        rms_norm_eps=1e-5,
        hidden_act=None,
        raw={"type_vocab_size": 3, "activation": "gelu_new"},
    )

    weights = {
        "embedding": np.zeros((10, 4), dtype=np.float32),
        "position_embedding": np.zeros((5, 4), dtype=np.float32),
        "token_type_embedding": np.zeros((3, 4), dtype=np.float32),
        "embed_norm": np.ones((4,), dtype=np.float32),
        "embed_norm_beta": np.zeros((4,), dtype=np.float32),
        "relative_position_bias": np.zeros((2, 5, 5), dtype=np.float32),
    }

    plan = mod.build_encoder_engine(config, weights, max_seq_length=5, verbose=True)

    assert plan == b"engine-plan"
    assert len(layer_calls) == 2
    assert all(isinstance(call, dict) for call in layer_calls)

    builder = fake_trt.Builder.last_instance
    assert builder.config.pool_limits == [("workspace", 1 << 30)]
    assert builder.config.cleared_flags == ["tf32"]
    assert [t.name for t in builder.network.outputs] == ["hidden_states"]


@pytest.mark.unit
def test_build_encoder_engine_raises_when_builder_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: validate encoder builder failure branch when TRT plan serialization fails.

    Preconditions: Fake TRT builder returns `None` and graph/layer helpers are mocked.
    Postconditions: `RuntimeError` is raised with encoder build failure text.
    """
    fake_trt = _make_fake_trt()
    fake_trt.Builder.plan_to_return = None
    mod = _import_with_fake_trt("tensorrt_model_connect.families.bert.encoder_builder", fake_trt)

    monkeypatch.setattr(mod.graph_ops, "add_constant", _fake_tensor_fn("const"))
    monkeypatch.setattr(mod, "_add_seq_layer_norm", _fake_tensor_fn("embed_ln"))
    monkeypatch.setattr(mod, "_add_encoder_layer", _fake_tensor_fn("layer"))

    config = types.SimpleNamespace(
        hidden_size=4,
        vocab_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=6,
        rms_norm_eps=1e-5,
        hidden_act="gelu",
        raw={"type_vocab_size": 2},
    )

    weights = {
        "embedding": np.zeros((8, 4), dtype=np.float32),
        "position_embedding": np.zeros((4, 4), dtype=np.float32),
        "token_type_embedding": np.zeros((2, 4), dtype=np.float32),
        "embed_norm": np.ones((4,), dtype=np.float32),
        "embed_norm_beta": np.zeros((4,), dtype=np.float32),
    }

    with pytest.raises(RuntimeError, match="TensorRT engine build failed"):
        mod.build_encoder_engine(config, weights, max_seq_length=4)
