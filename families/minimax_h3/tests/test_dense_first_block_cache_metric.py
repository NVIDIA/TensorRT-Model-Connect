# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests of dense FirstBlockCache modality guards; no engine construction."""
from __future__ import annotations


import ml_dtypes
import numpy as np
import pytest


@pytest.fixture
def candidate():
    from families.minimax_h3 import trt_compat

    if not trt_compat.is_available("tensorrt"):
        if not trt_compat.is_available("tensorrt_rtx"):
            pytest.skip("TensorRT or TensorRT-RTX bindings are unavailable")
        trt_compat.configure_backend(rtx=True)
    from families.minimax_h3 import dit_builder

    return dit_builder


@pytest.fixture
def graph(candidate, monkeypatch):
    trt = candidate.trt

    class Layer:
        def __init__(self, value):
            self.value = value

        def get_output(self, index):
            assert index == 0
            return self.value.reshape(self.reshape_dims) if hasattr(self, "reshape_dims") else self.value

    class Network:
        def add_elementwise(self, left, right, operation):
            functions = {trt.ElementWiseOperation.MAX: np.maximum,
                         trt.ElementWiseOperation.DIV: np.divide,
                         trt.ElementWiseOperation.EQUAL: np.equal,
                         trt.ElementWiseOperation.OR: np.logical_or}
            with np.errstate(invalid="ignore", divide="ignore"):
                return Layer(functions[operation](left, right))

        def add_unary(self, value, operation):
            return Layer({trt.UnaryOperation.ISNAN: np.isnan,
                          trt.UnaryOperation.ISINF: np.isinf}[operation](value))

        def add_select(self, condition, when_true, when_false):
            return Layer(np.where(condition, when_true, when_false))

        def add_shuffle(self, value):
            return Layer(value)

        def add_reduce(self, value, operation, axes, keep_dims):
            assert value.dtype == np.float32
            assert operation == trt.ReduceOperation.SUM and axes == 3
            return Layer(value.sum(axis=(0, 1), keepdims=keep_dims, dtype=np.float32))

    def slice_rows(_network, value, reference, *, trailing_reference=None):
        tail = 0 if trailing_reference is None else len(trailing_reference)
        end = len(value) - tail
        return value[end - len(reference):end]

    monkeypatch.setattr(candidate.op, "slice_rows_like_from_end", slice_rows, raising=False)
    monkeypatch.setattr(candidate.op, "constant", lambda _network, value, dtype=np.float32: np.asarray(value, dtype=dtype), raising=False)
    return Network()


def evaluate(candidate, graph, text, audio, condition, target, *, global_metric=0.01,
             text_delta=0.0, audio_delta=0.0, condition_delta=0.0, video_delta=0.5):
    rows = text + audio + condition + target
    previous = np.ones((rows, 4), np.float32)
    delta = np.concatenate((np.full((text, 4), text_delta), np.full((audio, 4), audio_delta),
                            np.full((condition, 4), condition_delta), np.full((target, 4), video_delta))).astype(np.float32)
    # Include vision/text index0 to ensure partitioning happens before masking.
    indices = np.concatenate((np.zeros(text, np.int32), np.full(audio, 5, np.int32),
                              np.full(condition, 6, np.int32), np.zeros(target, np.int32)))
    return candidate._guard_dense_cache_metric(graph, np.asarray([[global_metric]], np.float32),
        delta, previous, np.empty((condition + target, 96)), np.empty((audio, 32)), indices)


@pytest.mark.parametrize("text,audio,condition,target", [(1,2,0,3), (583,414,0,14985),
    (1166,414,0,37296), (2641,1150,0,106488), (7,414,810,14985), (71,414,2016,37296)])
def test_dynamic_rows_and_conditioning_do_not_dilute_video(candidate, graph, text, audio, condition, target):
    result = evaluate(candidate, graph, text, audio, condition, target,
                      text_delta=100, condition_delta=100)
    assert result.shape == (1, 1) and result.dtype == np.float32
    assert result.item() == pytest.approx(0.5)


def test_audio_and_existing_global_triggers_are_preserved(candidate, graph):
    assert evaluate(candidate, graph, 5, 2, 1, 3, video_delta=0.1, audio_delta=0.7).item() == pytest.approx(0.7)
    assert evaluate(candidate, graph, 5, 2, 1, 3, global_metric=0.9).item() == pytest.approx(0.9)


@pytest.mark.parametrize("source", ["global_metric", "audio_delta", "video_delta"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_each_invalid_operand_fails_closed(candidate, graph, source, invalid):
    assert np.isposinf(evaluate(candidate, graph, 2, 2, 1, 3, **{source: invalid})).all()


def test_condition_mask_selects_zero_instead_of_multiplying_nan(candidate, graph):
    # The actual full global metric still fails closed on invalid conditions.
    # Isolate the target mask here to check selection, not 0 * NaN.
    assert evaluate(candidate, graph, 2, 2, 1, 3, condition_delta=np.nan).item() == pytest.approx(0.5)
    assert np.isposinf(evaluate(candidate, graph, 2, 2, 1, 3,
                               global_metric=np.nan, condition_delta=np.nan)).all()


def test_epsilon_and_existing_bf16_subtraction_boundary(candidate, graph):
    bf16 = ml_dtypes.bfloat16
    current = np.asarray([[1.0, 0.0078125, 1.0, 0.0078125]], dtype=bf16)
    previous = np.asarray([[0.001953125, 1.0, 0.001953125, 1.0]], dtype=bf16)
    rounded_delta = np.abs((current.astype(np.float32) - previous.astype(np.float32)).astype(bf16)).astype(np.float32)
    assert not np.array_equal(rounded_delta, np.abs(current.astype(np.float32) - previous.astype(np.float32)))
    previous_abs = np.abs(previous).astype(np.float32)
    result = candidate._dense_modality_change(graph, rounded_delta, previous_abs, current)
    expected = rounded_delta.sum(dtype=np.float32) / previous_abs.sum(dtype=np.float32)
    assert result.item() == expected
    first = candidate._dense_modality_change(graph, np.ones((1, 4), np.float32),
                                            np.zeros((1, 4), np.float32), current)
    assert np.isfinite(first).all() and first.item() == pytest.approx(4.0e8)
