"""BERT-owned encoder builder tests using deterministic TRT stubs."""

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


def _make_fake_trt_base() -> types.SimpleNamespace:
    class _Logger:
        VERBOSE = 2
        WARNING = 1

        def __init__(self, _level):
            pass

    return types.SimpleNamespace(
        Logger=_Logger,
        ElementWiseOperation=types.SimpleNamespace(SUM="sum", SUB="sub", PROD="prod"),
        ReduceOperation=types.SimpleNamespace(AVG="avg"),
        UnaryOperation=types.SimpleNamespace(SQRT="sqrt", RECIP="recip"),
        MatrixOperation=types.SimpleNamespace(NONE="none", TRANSPOSE="transpose"),
        ActivationType=types.SimpleNamespace(SIGMOID="sigmoid", TANH="tanh"),
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


def _import_with_fake_trt(module_name: str):
    _drop_imported_module(module_name)
    if module_name.endswith("encoder_builder"):
        _drop_imported_module("tensorrt_model_connect.graph_blocks")
    previous_trt = sys.modules.get("tensorrt")
    sys.modules["tensorrt"] = _make_fake_trt_base()
    try:
        return importlib.import_module(module_name)
    finally:
        if previous_trt is None:
            sys.modules.pop("tensorrt", None)
        else:
            sys.modules["tensorrt"] = previous_trt


@pytest.mark.unit
def test_encoder_seq_layer_norm_uses_native_normalization() -> None:
    """BERT encoder layer norm uses TRT native add_normalization_v2."""
    mod = _import_with_fake_trt("tensorrt_model_connect.families.bert.model.model")

    class _FakeTensor:
        def __init__(self, name: str, dtype=np.float32, shape: tuple[int, ...] = (3, 4)):
            self.name = name
            self.dtype = dtype
            self.shape = shape

    class _FakeNormLayer:
        def __init__(self):
            self.epsilon = 0.0
            self._out = _FakeTensor("norm_out")

        def get_output(self, _i: int):
            return self._out

    class _FakeNetwork:
        def __init__(self):
            self.norm_calls: list[tuple] = []

        def add_normalization_v2(self, inp, gamma, beta, axis_mask):
            self.norm_calls.append((inp, gamma, beta, axis_mask))
            return _FakeNormLayer()

    add_constant_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def _fake_add_constant(_network, shape, values, **_kw):
        add_constant_calls.append((tuple(shape), tuple(np.asarray(values).shape)))
        return _FakeTensor(f"const_{len(add_constant_calls)}", _kw.get("dtype", np.float32))

    with patch.object(mod, "add_constant", side_effect=_fake_add_constant):
        net = _FakeNetwork()
        out = mod._add_seq_layer_norm(
            network=net,
            inp=_FakeTensor("in"),
            hidden_size=4,
            gamma=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            beta=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            eps=1e-6,
        )

    assert isinstance(out, _FakeTensor)
    assert len(net.norm_calls) == 1
    assert net.norm_calls[0][3] == (1 << 1)
    assert add_constant_calls[0][0] == (1, 4)
    assert add_constant_calls[1][0] == (1, 4)
