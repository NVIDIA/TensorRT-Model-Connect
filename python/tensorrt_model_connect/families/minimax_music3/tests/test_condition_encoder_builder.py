# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the condition-encoder TensorRT graph.

The graph itself was built and run on an A40 with TensorRT 11.1.0.106 and
torch 2.9.1+cu130: three windows against the oracle, agreeing to 3.5e-06
relative with TF32 off. TensorRT is not importable here, so these tests cover
the parts that do not need it -- the folded constant and the graph's calls
against a recording stub.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

builder = importlib.import_module(
    "tensorrt_model_connect.families.minimax_music3.condition_encoder_builder"
)
ce = importlib.import_module(
    "tensorrt_model_connect.families.minimax_music3.condition_encoder"
)


def test_folded_mix_is_a_scaled_softmax() -> None:
    logits = np.array([2.0898, -2.1539, -2.1834, -2.1673, -2.0807, -2.1133,
                       -2.1029, -2.0674], dtype=np.float32)
    scale = np.array([0.06851457], dtype=np.float32)

    mix = builder.folded_mix(logits, scale)

    assert mix.shape == (8,)
    # The published logits put 90.6% of the weight on the language model's own
    # hidden state and split the rest across the seven residual streams.
    softmax = mix / float(scale[0])
    assert softmax.sum() == pytest.approx(1.0, abs=1e-6)
    assert softmax[0] == pytest.approx(0.9061, abs=1e-3)
    assert all(0.012 < value < 0.015 for value in softmax[1:])
    assert mix.sum() == pytest.approx(float(scale[0]), abs=1e-6)


def test_folded_mix_rejects_a_wrong_stream_count() -> None:
    with pytest.raises(ValueError, match="expected 8 layer weights"):
        builder.folded_mix(np.zeros(4, dtype=np.float32), np.ones(1, dtype=np.float32))


def test_folded_mix_rejects_a_vector_scale() -> None:
    with pytest.raises(ValueError, match="one element"):
        builder.folded_mix(np.zeros(8, dtype=np.float32), np.ones(3, dtype=np.float32))


class _Layer:
    def __init__(self, name: str, record: list) -> None:
        self._name = name
        self._record = record

    def get_output(self, index: int):
        return SimpleNamespace(name=f"{self._name}_out{index}")

    def __setattr__(self, key, value):
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        else:
            self._record.append((self._name, key, value))


class _Network:
    """Records the layers a graph builder asks for."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.attrs: list[tuple] = []

    def _add(self, kind: str, *args) -> _Layer:
        self.calls.append((kind, args))
        return _Layer(f"{kind}{len(self.calls)}", self.attrs)

    def add_shuffle(self, *a): return self._add("shuffle", *a)
    def add_constant(self, *a): return self._add("constant", *a)
    def add_elementwise(self, *a): return self._add("elementwise", *a)
    def add_reduce(self, *a, **k): return self._add("reduce", *a, *sorted(k.items()))
    def add_convolution_nd(self, *a): return self._add("convolution", *a)
    def add_resize(self, *a): return self._add("resize", *a)


_TRT = SimpleNamespace(
    Weights=lambda array: ("weights", tuple(np.shape(array))),
    ElementWiseOperation=SimpleNamespace(PROD="PROD"),
    ReduceOperation=SimpleNamespace(SUM="SUM"),
    InterpolationMode=SimpleNamespace(NEAREST="NEAREST"),
    ResizeCoordinateTransformation=SimpleNamespace(ASYMMETRIC="ASYMMETRIC"),
    ResizeRoundMode=SimpleNamespace(FLOOR="FLOOR"),
    MemoryPoolType=SimpleNamespace(WORKSPACE="WORKSPACE"),
    BuilderFlag=SimpleNamespace(TF32="TF32"),
)


def _build(frames: int = 200) -> _Network:
    net = _Network()
    builder.add_condition_encoder(
        net, _TRT, SimpleNamespace(name="input"),
        frames=frames,
        mix=np.ones(8, dtype=np.float32),
        proj_weight=np.zeros((2048, 4096, 3), dtype=np.float32),
        proj_bias=np.zeros(2048, dtype=np.float32),
    )
    return net


def test_graph_layer_order() -> None:
    kinds = [kind for kind, _ in _build().calls]

    assert kinds == [
        "shuffle", "constant", "elementwise", "reduce",
        "shuffle", "convolution", "shuffle", "resize", "shuffle",
    ]


def test_reduction_is_over_the_stream_axis() -> None:
    reduce_args = next(args for kind, args in _build().calls if kind == "reduce")

    assert ("axes", 1 << 1) in reduce_args
    assert ("keep_dims", False) in reduce_args


def test_resize_settings_are_the_ones_the_reference_needs() -> None:
    attrs = dict((key, value) for _, key, value in _build().attrs)

    assert attrs["resize_mode"] == "NEAREST"
    assert attrs["coordinate_transformation"] == "ASYMMETRIC"
    assert attrs["nearest_rounding"] == "FLOOR"
    assert attrs["shape"] == (1, ce.OUT_DIM, ce.latent_length(200))


def test_convolution_weights_are_reshaped_for_a_1x3_kernel() -> None:
    conv_args = next(args for kind, args in _build().calls if kind == "convolution")

    assert conv_args[1] == ce.OUT_DIM
    assert conv_args[2] == (1, ce.PROJ_KERNEL_SIZE)
    assert conv_args[3] == ("weights", (2048, 4096, 1, 3))
    assert dict((k, v) for _, k, v in _build().attrs)["padding_nd"] == (0, 1)


def test_shapes_follow_the_window_length() -> None:
    attrs = [(key, value) for _, key, value in _build(500).attrs]
    reshapes = [value for key, value in attrs if key == "reshape_dims"]

    assert reshapes[0] == (1, 8, 4096, 500)
    assert reshapes[1] == (1, 4096, 1, 500)
    assert reshapes[2] == (1, 2048, 500)


def test_configure_states_the_precision_choice() -> None:
    cleared = []
    limits = []
    config = SimpleNamespace(
        set_memory_pool_limit=lambda pool, size: limits.append((pool, size)),
        clear_flag=cleared.append,
    )

    builder.configure(config, _TRT)

    assert limits == [("WORKSPACE", builder.WORKSPACE_BYTES)]
    assert cleared == ["TF32"]
