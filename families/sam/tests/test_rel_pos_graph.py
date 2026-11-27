# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM's encoder attention graph, run on a numpy stand-in for TensorRT.

The real ``_build_windowed_attention`` and ``_build_global_attention`` build
their layers on a fake network whose tensors are evaluated lazily in float64
(layer properties are set after a layer is created). The result is compared
with a numpy port of HF ``SamVisionAttention`` (window partition, padding and
the decomposed relative-position bias included). FP16 builds carry the bias in
Q.K^T; FP32 builds add it to the scores.
"""

from __future__ import annotations

import enum
import importlib
import sys
import types

import numpy as np
import pytest


class _Tensor:
    def __init__(self, evaluate):
        self._evaluate = evaluate
        self._value = None
        self.dtype = "work"

    def value(self):
        if self._value is None:
            self._value = self._evaluate()
        return self._value

    @property
    def shape(self):
        return tuple(self.value().shape)


class _Layer:
    def __init__(self, evaluate):
        self._output = _Tensor(lambda: evaluate(self))

    def get_output(self, index):
        assert index == 0
        return self._output


def _fake_tensorrt():
    trt = types.ModuleType("tensorrt")
    trt.MatrixOperation = enum.Enum("MatrixOperation", "NONE TRANSPOSE")
    trt.ElementWiseOperation = enum.Enum("ElementWiseOperation", "SUM PROD")
    trt.SampleMode = enum.Enum("SampleMode", "STRICT_BOUNDS WRAP FILL")
    trt.Permutation = tuple
    trt.Weights = np.asarray
    return trt


class _Network:
    """The INetworkDefinition calls the encoder attention makes, in float64.

    ``sums`` records the shape of every elementwise SUM, so a test can tell
    whether a [batch, heads, seq, seq] bias is still added to the scores.
    """

    def __init__(self, trt):
        self.trt = trt
        self.sums = []

    def add_constant(self, shape, weights):
        values = np.asarray(weights, dtype=np.float64).reshape(shape)
        return _Layer(lambda layer: values)

    def add_shuffle(self, tensor):
        def evaluate(layer):
            values = tensor.value()
            if getattr(layer, "first_transpose", None) is not None:
                values = values.transpose(layer.first_transpose)
            if getattr(layer, "reshape_dims", None) is not None:
                values = values.reshape(layer.reshape_dims)
            if getattr(layer, "second_transpose", None) is not None:
                values = values.transpose(layer.second_transpose)
            return values

        return _Layer(evaluate)

    def add_matrix_multiply(self, lhs, lhs_op, rhs, rhs_op):
        def evaluate(layer):
            a, b = lhs.value(), rhs.value()
            assert a.ndim == b.ndim, (a.shape, b.shape)
            if lhs_op == self.trt.MatrixOperation.TRANSPOSE:
                a = np.swapaxes(a, -1, -2)
            if rhs_op == self.trt.MatrixOperation.TRANSPOSE:
                b = np.swapaxes(b, -1, -2)
            return np.matmul(a, b)

        return _Layer(evaluate)

    def add_elementwise(self, lhs, rhs, operation):
        def evaluate(layer):
            a, b = lhs.value(), rhs.value()
            assert a.ndim == b.ndim, (a.shape, b.shape)
            if operation == self.trt.ElementWiseOperation.SUM:
                self.sums.append(np.broadcast_shapes(a.shape, b.shape))
                return a + b
            return a * b

        return _Layer(evaluate)

    def add_softmax(self, tensor):
        def evaluate(layer):
            values = tensor.value()
            axis = layer.axes.bit_length() - 1
            exp = np.exp(values - values.max(axis=axis, keepdims=True))
            return exp / exp.sum(axis=axis, keepdims=True)

        return _Layer(evaluate)

    def add_concatenation(self, tensors):
        return _Layer(lambda layer: np.concatenate([t.value() for t in tensors], axis=layer.axis))

    def add_slice(self, tensor, start, shape, stride):
        def evaluate(layer):
            values = tensor.value()
            mode = getattr(layer, "mode", self.trt.SampleMode.STRICT_BOUNDS)
            index = [s + np.arange(n) * k for s, n, k in zip(start, shape, stride)]
            if mode == self.trt.SampleMode.WRAP:
                return values[np.ix_(*[i % d for i, d in zip(index, values.shape)])]
            inside = [(i >= 0) & (i < d) for i, d in zip(index, values.shape)]
            if mode == self.trt.SampleMode.STRICT_BOUNDS:
                assert all(v.all() for v in inside), "slice out of bounds"
                return values[np.ix_(*index)]
            assert mode == self.trt.SampleMode.FILL
            clipped = values[np.ix_(*[np.clip(i, 0, d - 1) for i, d in zip(index, values.shape)])]
            mask = np.logical_and.reduce(np.meshgrid(*inside, indexing="ij"))
            return np.where(mask, clipped, 0.0)

        return _Layer(evaluate)


@pytest.fixture
def sam_model(monkeypatch):
    """families.sam.model imported against the numpy stand-in for TensorRT.

    The modules that import TensorRT are imported afresh; monkeypatch puts the
    originals (or their absence) back afterwards, in sys.modules and on the
    package.
    """
    import families.sam as package

    trt = _fake_tensorrt()
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    for short in ("model", "graph_ops", "parallel"):
        monkeypatch.setitem(sys.modules, f"families.sam.{short}", None)
        del sys.modules[f"families.sam.{short}"]
        monkeypatch.setattr(package, short, None, raising=False)
        delattr(package, short)
    model = importlib.import_module("families.sam.model")
    return model, trt


def _hf_rel_pos(size, table):
    coords = np.arange(size)[:, None] - np.arange(size)[None, :] + (size - 1)
    return table[coords]


def _hf_attention(x, weights, prefix, num_heads):
    """HF SamVisionAttention.forward on [batch, size, size, hidden]."""
    batch, height, width, hidden = x.shape
    head_dim = hidden // num_heads
    rows = x.reshape(batch, height * width, hidden)

    def heads(name):
        projected = rows @ weights[f"{prefix}.attn.{name}.weight"] + weights[f"{prefix}.attn.{name}.bias"]
        projected = projected.reshape(batch, height * width, num_heads, head_dim)
        return projected.transpose(0, 2, 1, 3).reshape(batch * num_heads, height * width, head_dim)

    q, k, v = heads("q"), heads("k"), heads("v")
    logits = (q * head_dim**-0.5) @ k.transpose(0, 2, 1)
    grid_q = q.reshape(batch * num_heads, height, width, head_dim)
    rel_h = np.einsum("bhwc,hkc->bhwk", grid_q, _hf_rel_pos(height, weights[f"{prefix}.attn.rel_pos_h"]))
    rel_w = np.einsum("bhwc,wkc->bhwk", grid_q, _hf_rel_pos(width, weights[f"{prefix}.attn.rel_pos_w"]))
    bias = rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    logits = logits + bias.reshape(logits.shape)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    context = (probs @ v).reshape(batch, num_heads, height, width, head_dim)
    context = context.transpose(0, 2, 3, 1, 4).reshape(batch, height, width, hidden)
    return context @ weights[f"{prefix}.attn.o.weight"] + weights[f"{prefix}.attn.o.bias"]


def _hf_window_attention(x, weights, prefix, num_heads, window):
    batch, size, _, hidden = x.shape
    pad = (window - size % window) % window
    padded = np.pad(x, ((0, 0), (0, pad), (0, pad), (0, 0)))
    count = (size + pad) // window
    windows = padded.reshape(batch, count, window, count, window, hidden)
    windows = windows.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window, window, hidden)
    out = _hf_attention(windows, weights, prefix, num_heads)
    out = out.reshape(batch, count, count, window, window, hidden).transpose(0, 1, 3, 2, 4, 5)
    return out.reshape(batch, size + pad, size + pad, hidden)[:, :size, :size]


def _f16(values):
    """Values the builders' float16 and float32 constants hold exactly."""
    return np.asarray(values).astype(np.float16).astype(np.float64)


def _weights(rng, hidden, head_dim, rel_size, prefix="encoder.layer0"):
    weights = {}
    for name in "qkvo":
        weights[f"{prefix}.attn.{name}.weight"] = _f16(rng.standard_normal((hidden, hidden)) / np.sqrt(hidden))
        weights[f"{prefix}.attn.{name}.bias"] = _f16(0.1 * rng.standard_normal(hidden))
    for axis in "hw":
        weights[f"{prefix}.attn.rel_pos_{axis}"] = _f16(rng.standard_normal((2 * rel_size - 1, head_dim)))
    return prefix, weights


def _graph_input(x):
    return _Layer(lambda layer: x).get_output(0)


# head_dim 64 (every SAM ViT): the logit scale 1/8 and the tables scaled by it
# are exact in the work dtype, so the graph matches HF to float64 rounding.
# Another head_dim rounds the scale (and in FP16 the scaled tables) to it.
_PRECISIONS = pytest.mark.parametrize("dtype", [np.float16, np.float32])


def _rtol(head_dim, dtype):
    if head_dim == 64:
        return 1e-12
    return 1e-2 if dtype == np.float16 else 1e-6


@_PRECISIONS
@pytest.mark.parametrize(
    ("grid", "window", "hidden", "heads"),
    [
        (10, 7, 128, 2),  # padded windows (10 -> 14)
        (14, 14, 128, 2),  # one window, no padding
        (9, 7, 96, 2),  # head_dim 48
    ],
)
def test_windowed_attention_matches_hf(sam_model, dtype, grid, window, hidden, heads) -> None:
    model, trt = sam_model
    rng = np.random.default_rng(grid * 100 + window)
    prefix, weights = _weights(rng, hidden, hidden // heads, window)
    x = _f16(rng.standard_normal((1, grid, grid, hidden)))
    network = _Network(trt)
    out = model._SamModel._build_windowed_attention(
        network, _graph_input(x), weights, prefix, grid, hidden, heads, hidden // heads, window,
        dtype=dtype,
    )
    expected = _hf_window_attention(x, weights, prefix, heads, window)
    rtol = _rtol(hidden // heads, dtype)
    np.testing.assert_allclose(out.value(), expected, rtol=rtol, atol=rtol)


@_PRECISIONS
@pytest.mark.parametrize(("grid", "hidden", "heads"), [(8, 128, 2), (6, 160, 2)])
def test_global_attention_matches_hf(sam_model, dtype, grid, hidden, heads) -> None:
    model, trt = sam_model
    rng = np.random.default_rng(grid)
    prefix, weights = _weights(rng, hidden, hidden // heads, grid)
    x = _f16(rng.standard_normal((1, grid, grid, hidden)))
    network = _Network(trt)
    out = model._SamModel._build_global_attention(
        network, _graph_input(x), weights, prefix, grid, hidden, heads, hidden // heads, grid * grid,
        dtype=dtype,
    )
    expected = _hf_attention(x, weights, prefix, heads)
    rtol = _rtol(hidden // heads, dtype)
    np.testing.assert_allclose(out.value(), expected, rtol=rtol, atol=rtol)


@_PRECISIONS
def test_fp16_folds_the_bias_and_fp32_keeps_the_bias_add(sam_model, dtype) -> None:
    model, trt = sam_model
    rng = np.random.default_rng(3)
    grid, hidden, heads = 6, 128, 2  # seq 36 differs from head_dim 64
    seq = grid * grid
    prefix, weights = _weights(rng, hidden, hidden // heads, grid)
    network = _Network(trt)
    products = []
    multiply = network.add_matrix_multiply

    def recording_multiply(lhs, lhs_op, rhs, rhs_op):
        products.append((lhs, rhs_op, rhs))
        return multiply(lhs, lhs_op, rhs, rhs_op)

    network.add_matrix_multiply = recording_multiply
    out = model._SamModel._build_global_attention(
        network, _graph_input(_f16(rng.standard_normal((1, grid, grid, hidden)))), weights, prefix,
        grid, hidden, heads, hidden // heads, seq, dtype=dtype,
    )
    out.value()
    seq_by_seq_sums = [shape for shape in network.sums if shape[-2:] == (seq, seq)]
    (score_q, _, _), = [p for p in products if p[1] == trt.MatrixOperation.TRANSPOSE]
    (_, _, value), = [p for p in products if p[0].shape[-2:] == (seq, seq)]
    if dtype == np.float16:
        # The bias rides in Q.K^T: no SUM produces a [.., seq, seq] tensor, and
        # P.V reads V at Q.K's width (one head dimension for the fused attention).
        assert not seq_by_seq_sums
        assert value.shape[-1] == score_q.shape[-1] > hidden // heads
    else:
        # FP32 keeps the graph it had: the bias added to the scores.
        assert len(seq_by_seq_sums) == 1
        assert value.shape[-1] == score_q.shape[-1] == hidden // heads
