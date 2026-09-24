# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-only checks of normalization's strongly typed graph boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


class _Layer:
    def __init__(self, dtype):
        self.tensor = SimpleNamespace(dtype=dtype)

    def get_output(self, index):
        assert index == 0
        return self.tensor


class _TypedNetwork:
    def __init__(self):
        self.compute_dtypes = []
        self.casts = []

    def add_cast(self, tensor, dtype):
        self.casts.append((tensor.dtype, dtype))
        return _Layer(dtype)

    def add_constant(self, shape, weights):
        return _Layer(str(weights.dtype))

    def add_shuffle(self, tensor):
        return _Layer(tensor.dtype)

    def add_elementwise(self, lhs, rhs, operation):
        assert lhs.dtype == rhs.dtype, f"strongly typed {operation}: {lhs.dtype} != {rhs.dtype}"
        self.compute_dtypes.append(lhs.dtype)
        return _Layer(lhs.dtype)

    def add_reduce(self, tensor, operation, axes, keep_dims):
        self.compute_dtypes.append(tensor.dtype)
        return _Layer(tensor.dtype)

    def add_unary(self, tensor, operation):
        self.compute_dtypes.append(tensor.dtype)
        return _Layer(tensor.dtype)


@pytest.mark.parametrize("normalization", ["rms", "per_head_rms", "layer"])
@pytest.mark.parametrize(
    ("runtime_dtype", "storage_dtype"),
    [("bfloat16", np.float32), ("float16", np.float16), ("float32", np.float32)],
)
def test_normalization_uses_runtime_dtype_for_fp32_compute(
    monkeypatch, normalization, runtime_dtype, storage_dtype
):
    import sys

    trt = SimpleNamespace(
        float32="float32",
        Weights=lambda values: values,
        ElementWiseOperation=SimpleNamespace(PROD="prod", SUM="sum", SUB="sub"),
        ReduceOperation=SimpleNamespace(AVG="avg"),
        UnaryOperation=SimpleNamespace(SQRT="sqrt", RECIP="recip"),
    )
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    source = Path(__file__).resolve().parents[1] / "embedding_graph_ops.py"
    spec = importlib.util.spec_from_file_location("qwen3_embedding_norm_under_test", source)
    assert spec is not None and spec.loader is not None
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)

    network = _TypedNetwork()
    inp = SimpleNamespace(dtype=runtime_dtype)
    epsilon = SimpleNamespace(dtype="float32")
    gamma = np.ones(4, dtype=storage_dtype)
    if normalization == "rms":
        result = ops.add_rms_norm(network, inp, 4, gamma, epsilon, dtype=storage_dtype)
    elif normalization == "per_head_rms":
        result = ops.add_rms_norm_per_head(network, inp, 2, 2, gamma, epsilon, dtype=storage_dtype)
    else:
        result = ops.add_layer_norm(
            network,
            inp,
            4,
            gamma,
            np.zeros(4, dtype=storage_dtype),
            epsilon,
            dtype=storage_dtype,
        )

    assert network.compute_dtypes and set(network.compute_dtypes) == {"float32"}
    assert result.dtype == runtime_dtype
    if runtime_dtype == "float32":
        assert network.casts == []
    else:
        assert (runtime_dtype, "float32") in network.casts
        assert network.casts[-1] == ("float32", runtime_dtype)
