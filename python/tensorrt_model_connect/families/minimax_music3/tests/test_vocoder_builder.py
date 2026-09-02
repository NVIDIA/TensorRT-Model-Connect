# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the vocoder TensorRT graph.

TensorRT is not importable here, so the graph is driven against a recording
stub: the layer census, the shapes, and the folded constants.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

vb = importlib.import_module(
    "tensorrt_model_connect.families.minimax_music3.vocoder_builder"
)
voc = importlib.import_module(
    "tensorrt_model_connect.families.minimax_music3.vocoder"
)


class _Layer:
    def __init__(self, name, record):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_record", record)

    def get_output(self, index):
        return SimpleNamespace(name=f"{self._name}#{index}")

    def __setattr__(self, key, value):
        self._record.append((self._name, key, value))


class _Network:
    def __init__(self):
        self.calls = []
        self.attrs = []

    def _add(self, kind, *args):
        self.calls.append((kind, args))
        return _Layer(f"{kind}{len(self.calls)}", self.attrs)

    def add_shuffle(self, *a): return self._add("shuffle", *a)
    def add_constant(self, *a): return self._add("constant", *a)
    def add_elementwise(self, *a): return self._add("elementwise", *a)
    def add_unary(self, *a): return self._add("unary", *a)
    def add_convolution_nd(self, *a): return self._add("conv", *a)
    def add_deconvolution_nd(self, *a): return self._add("deconv", *a)
    def add_activation(self, *a): return self._add("activation", *a)


_TRT = SimpleNamespace(
    Weights=lambda a: ("W", tuple(np.shape(a))),
    ElementWiseOperation=SimpleNamespace(PROD="PROD", SUM="SUM"),
    UnaryOperation=SimpleNamespace(SIN="SIN"),
    ActivationType=SimpleNamespace(TANH="TANH"),
    MemoryPoolType=SimpleNamespace(WORKSPACE="WORKSPACE"),
    BuilderFlag=SimpleNamespace(TF32="TF32"),
)


def _weights():
    """Checkpoint-shaped weights, matching the published tensor inventory."""

    rng = np.random.default_rng(0)

    def wn(out, inp, k):
        return {
            "weight_g": np.abs(rng.standard_normal((out, 1, 1)).astype(np.float32)) + 0.1,
            "weight_v": rng.standard_normal((out, inp, k)).astype(np.float32),
            "bias": rng.standard_normal(out).astype(np.float32),
        }

    w = {}
    w["dec_in_proj.weight"] = rng.standard_normal((1024, 64, 1)).astype(np.float32)
    w["dec_in_proj.bias"] = rng.standard_normal(1024).astype(np.float32)
    for key, value in wn(1536, 1024, 7).items():
        w[f"conv_in.{key}"] = value
    for block in voc.blocks():
        p = f"blocks.{block.index}"
        w[f"{p}.snake1.alpha"] = np.ones((1, block.input_dim, 1), dtype=np.float32)
        # ConvTranspose weights are (in, out, kernel).
        w[f"{p}.conv_t1.weight_g"] = np.ones((block.input_dim, 1, 1), dtype=np.float32)
        w[f"{p}.conv_t1.weight_v"] = rng.standard_normal(
            (block.input_dim, block.output_dim, block.kernel_size)).astype(np.float32)
        w[f"{p}.conv_t1.bias"] = rng.standard_normal(block.output_dim).astype(np.float32)
        for unit in (1, 2, 3):
            u = f"{p}.res_unit{unit}"
            w[f"{u}.snake1.alpha"] = np.ones((1, block.output_dim, 1), dtype=np.float32)
            w[f"{u}.snake2.alpha"] = np.ones((1, block.output_dim, 1), dtype=np.float32)
            for key, value in wn(block.output_dim, block.output_dim, 7).items():
                w[f"{u}.conv1.{key}"] = value
            for key, value in wn(block.output_dim, block.output_dim, 1).items():
                w[f"{u}.conv2.{key}"] = value
    w["snake_out.alpha"] = np.ones((1, 96, 1), dtype=np.float32)
    for key, value in wn(1, 96, 7).items():
        w[f"conv_out.{key}"] = value
    return w


def _build(latent_length: int = 689):
    net = _Network()
    vb.add_vocoder(net, _TRT, SimpleNamespace(name="in"),
                   latent_length=latent_length, weights=_weights())
    return net


def test_weight_fixture_matches_the_published_tensor_count() -> None:
    """121 tensors, the number the published vocoder header carries."""

    assert len(_weights()) == 121


def test_layer_census() -> None:
    kinds = [kind for kind, _ in _build().calls]

    assert kinds.count("conv") == 27
    assert kinds.count("deconv") == 4
    assert kinds.count("unary") == 29  # one SIN per Snake
    assert kinds.count("activation") == 1  # the final tanh
    assert kinds.count("shuffle") == 2  # fold and unfold


def test_stereo_fold_and_unfold_shapes() -> None:
    reshapes = [v for _, k, v in _build().attrs if k == "reshape_dims"]

    assert reshapes[0] == (2, 64, 1, 689)
    assert reshapes[-1] == (1, 2, 689 * 512)


def test_output_shape_helper_agrees_with_the_graph() -> None:
    assert vb.expected_input_shape(689) == (1, 128, 689)
    assert vb.expected_output_shape(689) == (1, 2, 352768)


def test_transposed_convolutions_carry_the_block_strides() -> None:
    attrs = _build().attrs
    strides = [v for _, k, v in attrs if k == "stride_nd"]
    paddings = [v for _, k, v in attrs if k == "padding_nd"]

    assert strides == [(1, 8), (1, 8), (1, 4), (1, 2)]
    # Every block's transposed convolution pads by ceil(stride / 2).
    assert [p for p in paddings if p in {(0, 4), (0, 2), (0, 1)}][:4] == [
        (0, 4), (0, 4), (0, 2), (0, 1)
    ]


def test_residual_dilations_appear_in_order() -> None:
    dilations = [v for _, k, v in _build().attrs if k == "dilation_nd"]
    # Three residual units per block, dilated 1, 3, 9; the 1x1 convolutions
    # and the two stem convolutions stay undilated.
    per_block = [d for d in dilations if d != (1, 1)]

    assert per_block == [(1, 3), (1, 9)] * 4


def test_snake_reciprocal_is_the_folded_divisor() -> None:
    alpha = np.array([0.5, 2.0, 4.0], dtype=np.float32)

    recip = vb.snake_reciprocal(alpha)

    assert recip.shape == (1, 3, 1, 1)
    assert np.allclose(recip.reshape(-1), 1.0 / (alpha + vb.SNAKE_EPSILON), atol=1e-6)


def test_snake_graph_reproduces_the_reference_activation() -> None:
    """Six layers per Snake, and the folded form equals the reference."""

    net = _Network()
    alpha = np.full((1, 4, 1), 1.5, dtype=np.float32)
    vb.add_snake(net, _TRT, SimpleNamespace(name="x"), alpha)
    kinds = [kind for kind, _ in net.calls]

    assert kinds == ["constant", "constant", "elementwise", "unary",
                     "elementwise", "elementwise", "elementwise"]

    x = np.linspace(-2, 2, 9, dtype=np.float32).reshape(1, 1, 9)
    a = np.float32(1.5)
    folded = x + np.sin(a * x) ** 2 * (1.0 / (a + vb.SNAKE_EPSILON))
    assert np.allclose(folded, voc.snake(x, a), atol=1e-6)


def test_configure_states_the_precision_choice() -> None:
    cleared, limits = [], []
    config = SimpleNamespace(
        set_memory_pool_limit=lambda p, s: limits.append((p, s)),
        clear_flag=cleared.append,
    )

    vb.configure(config, _TRT)

    assert limits == [("WORKSPACE", vb.WORKSPACE_BYTES)]
    assert cleared == ["TF32"]
