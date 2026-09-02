# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the diffusion transformer's TensorRT graph."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

dit = importlib.import_module("tensorrt_model_connect.families.minimax_music3.dit")
dib = importlib.import_module(
    "tensorrt_model_connect.families.minimax_music3.dit_builder"
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
        self.calls, self.attrs = [], []

    def _add(self, kind, *args):
        self.calls.append((kind, args))
        return _Layer(f"{kind}{len(self.calls)}", self.attrs)

    def add_constant(self, *a): return self._add("constant", *a)
    def add_elementwise(self, *a): return self._add("elementwise", *a)
    def add_reduce(self, *a, **k): return self._add("reduce", *a, *sorted(k.items()))
    def add_unary(self, *a): return self._add("unary", *a)
    def add_shuffle(self, *a): return self._add("shuffle", *a)
    def add_matrix_multiply(self, *a): return self._add("matmul", *a)
    def add_softmax(self, *a): return self._add("softmax", *a)
    def add_activation(self, *a): return self._add("activation", *a)
    def add_slice(self, *a): return self._add("slice", *a)
    def add_concatenation(self, *a): return self._add("concat", *a)
    def add_convolution_nd(self, *a): return self._add("conv", *a)


def _weights_stub(*args):
    return ("W", tuple(np.shape(args[0]))) if args else ("W", "empty")


_TRT = SimpleNamespace(
    Weights=_weights_stub,
    ElementWiseOperation=SimpleNamespace(PROD="PROD", SUM="SUM", SUB="SUB", DIV="DIV"),
    ReduceOperation=SimpleNamespace(AVG="AVG"),
    UnaryOperation=SimpleNamespace(SQRT="SQRT", NEG="NEG"),
    MatrixOperation=SimpleNamespace(NONE="NONE", TRANSPOSE="TRANSPOSE"),
    ActivationType=SimpleNamespace(SIGMOID="SIGMOID"),
    MemoryPoolType=SimpleNamespace(WORKSPACE="WORKSPACE"),
    BuilderFlag=SimpleNamespace(TF32="TF32"),
)


def _weights(layers: int):
    rng = np.random.default_rng(0)
    w = {
        "preprocess_conv.weight": rng.standard_normal(
            (dit.CONCAT_CHANNELS, dit.CONCAT_CHANNELS, 1)).astype(np.float32) * 0.01,
        "postprocess_conv.weight": rng.standard_normal(
            (dit.IN_CHANNELS, dit.IN_CHANNELS, 1)).astype(np.float32) * 0.01,
        "proj_in.weight": rng.standard_normal(
            (dit.INNER_DIM, dit.CONCAT_CHANNELS)).astype(np.float32) * 0.01,
        "proj_out.weight": rng.standard_normal(
            (dit.IN_CHANNELS, dit.INNER_DIM)).astype(np.float32) * 0.01,
    }
    for layer in range(layers):
        p = f"transformer_blocks.{layer}"
        for name in ("to_q", "to_k", "to_v"):
            w[f"{p}.attn.{name}.weight"] = rng.standard_normal(
                (dit.INNER_DIM, dit.INNER_DIM)).astype(np.float32) * 0.01
        w[f"{p}.attn.to_out.0.weight"] = rng.standard_normal(
            (dit.INNER_DIM, dit.INNER_DIM)).astype(np.float32) * 0.01
        w[f"{p}.norm1.weight"] = np.ones(dit.INNER_DIM, dtype=np.float32)
        w[f"{p}.norm1.bias"] = np.zeros(dit.INNER_DIM, dtype=np.float32)
        w[f"{p}.norm2.weight"] = np.ones(dit.INNER_DIM, dtype=np.float32)
        w[f"{p}.norm2.bias"] = np.zeros(dit.INNER_DIM, dtype=np.float32)
        w[f"{p}.ff_in.weight"] = rng.standard_normal(
            (2 * dit.FF_INNER_DIM, dit.INNER_DIM)).astype(np.float32) * 0.01
        w[f"{p}.ff_in.bias"] = np.zeros(2 * dit.FF_INNER_DIM, dtype=np.float32)
        w[f"{p}.ff_out.weight"] = rng.standard_normal(
            (dit.INNER_DIM, dit.FF_INNER_DIM)).astype(np.float32) * 0.01
        w[f"{p}.ff_out.bias"] = np.zeros(dit.INNER_DIM, dtype=np.float32)
    return w


def _build(latent_length=16, layers=2):
    net = _Network()
    dib.add_dit(net, _TRT, SimpleNamespace(name="lat"), SimpleNamespace(name="cond"),
                SimpleNamespace(name="temb"), latent_length=latent_length,
                weights=_weights(layers), num_layers=layers)
    return net


def test_stem_uses_two_residual_convolutions() -> None:
    net = _build()
    kinds = [k for k, _ in net.calls]

    assert kinds.count("conv") == 2
    # Each convolution's output is summed with its own input.
    assert kinds.count("elementwise") >= 2


def test_zero_block_is_a_build_time_constant() -> None:
    shapes = [args[0] for kind, args in _build(latent_length=16).calls
              if kind == "constant"]

    assert (1, dit.IN_CHANNELS, 16) in shapes


def test_timestep_prefix_extends_then_is_trimmed() -> None:
    net = _build(latent_length=16, layers=1)
    slices = [args for kind, args in net.calls if kind == "slice"]

    # The final trim drops position zero and keeps latent_length tokens.
    assert ((0, 1, 0), (1, 16, dit.INNER_DIM), (1, 1, 1)) in [
        (a[1], a[2], a[3]) for a in slices
    ]


def test_attention_length_includes_the_prefix() -> None:
    reshapes = [v for _, k, v in _build(latent_length=16, layers=1).attrs
                if k == "reshape_dims"]

    assert (17, dit.NUM_ATTENTION_HEADS, dit.ATTENTION_HEAD_DIM) in reshapes
    assert (1, 17, dit.INNER_DIM) in reshapes


def test_partial_rope_slices_the_head_in_two() -> None:
    net = _Network()
    cos, sin = dit.rotary_tables(5)
    dib.add_partial_rope(net, _TRT, SimpleNamespace(name="h"), cos, sin, 5)
    widths = [a[2][2] for kind, a in net.calls if kind == "slice"]

    # Rotated slice, pass-through slice, then the two halves of the rotated one.
    assert widths == [dit.ROTARY_DIM,
                      dit.ATTENTION_HEAD_DIM - dit.ROTARY_DIM,
                      dit.ROTARY_DIM // 2, dit.ROTARY_DIM // 2]


def test_partial_rope_negates_the_second_half() -> None:
    net = _Network()
    cos, sin = dit.rotary_tables(5)
    dib.add_partial_rope(net, _TRT, SimpleNamespace(name="h"), cos, sin, 5)

    assert [k for k, _ in net.calls].count("unary") == 1  # the NEG
    assert any(args[0] == "NEG" or "NEG" in str(args) for kind, args in net.calls
               if kind == "unary")


def test_gated_feed_forward_uses_first_times_silu_of_second() -> None:
    net = _Network()
    dib.add_gated_feed_forward(net, _TRT, SimpleNamespace(name="h"),
                               _weights(1), "transformer_blocks.0", 17)
    kinds = [k for k, _ in net.calls]

    # ff_in matmul + bias, two slices, sigmoid, silu product, gate product,
    # ff_out matmul + bias.
    assert kinds.count("slice") == 2
    assert kinds.count("activation") == 1
    assert kinds.count("matmul") == 2


def test_layer_norm_is_built_from_primitives() -> None:
    net = _Network()
    dib.add_layer_norm(net, _TRT, SimpleNamespace(name="h"),
                       np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.float32))
    kinds = [k for k, _ in net.calls]

    assert kinds.count("reduce") == 2  # mean, then variance
    assert kinds.count("unary") == 1  # the square root


def test_layer_count_scales_the_graph() -> None:
    two = len(_build(layers=2).calls)
    four = len(_build(layers=4).calls)

    assert (four - two) % 2 == 0
    assert four > two


def test_io_shapes() -> None:
    shapes = dib.expected_io_shapes(689)

    assert shapes[dib.LATENTS_NAME] == (1, 128, 689)
    assert shapes[dib.CONDITION_NAME] == (1, 689, 2048)
    assert shapes[dib.OUTPUT_NAME] == (1, 128, 689)


def test_configure_states_the_precision_choice() -> None:
    cleared, limits = [], []
    dib.configure(SimpleNamespace(
        set_memory_pool_limit=lambda p, s: limits.append((p, s)),
        clear_flag=cleared.append), _TRT)

    assert limits == [("WORKSPACE", dib.WORKSPACE_BYTES)]
    assert cleared == ["TF32"]


def test_bias_free_convolutions_use_an_empty_weights_object() -> None:
    """A zero-length array is not an empty bias.

    TensorRT requires a bias's count and pointer to agree; a zero-length numpy
    array still carries a non-null pointer and the parameter check rejects it.
    Both stem convolutions are stored without a bias, so both must pass a
    default-constructed Weights.
    """

    conv_args = [args for kind, args in _build().calls if kind == "conv"]

    assert len(conv_args) == 2
    for args in conv_args:
        assert args[4] == ("W", "empty")
