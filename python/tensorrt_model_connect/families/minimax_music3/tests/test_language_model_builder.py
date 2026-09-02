# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the language model's TensorRT graph."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

lm = importlib.import_module(
    "tensorrt_model_connect.families.minimax_music3.language_model"
)
lb = importlib.import_module(
    "tensorrt_model_connect.families.minimax_music3.language_model_builder"
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


_TRT = SimpleNamespace(
    Weights=lambda a: ("W", tuple(np.shape(a))),
    ElementWiseOperation=SimpleNamespace(PROD="PROD", SUM="SUM", DIV="DIV"),
    ReduceOperation=SimpleNamespace(AVG="AVG"),
    UnaryOperation=SimpleNamespace(SQRT="SQRT", NEG="NEG"),
    MatrixOperation=SimpleNamespace(NONE="NONE", TRANSPOSE="TRANSPOSE"),
    ActivationType=SimpleNamespace(SIGMOID="SIGMOID"),
    MemoryPoolType=SimpleNamespace(WORKSPACE="WORKSPACE"),
    BuilderFlag=SimpleNamespace(TF32="TF32"),
)


def _weight_shapes(layers: int, prefix: str = "model"):
    """The checkpoint's tensor names and shapes, without the tensors.

    One layer of this model is 0.77 GB of float32, so the inventory is kept
    separate from the data: a test that only counts names must not allocate
    36 layers to do it.
    """

    shapes = {f"{prefix}.norm.weight": (lm.HIDDEN_SIZE,)}
    for layer in range(layers):
        b = f"{prefix}.layers.{layer}"
        shapes[f"{b}.self_attn.q_proj.weight"] = (lm.query_width(), lm.HIDDEN_SIZE)
        for name in ("k_proj", "v_proj"):
            shapes[f"{b}.self_attn.{name}.weight"] = (lm.key_value_width(), lm.HIDDEN_SIZE)
        shapes[f"{b}.self_attn.o_proj.weight"] = (lm.HIDDEN_SIZE, lm.HIDDEN_SIZE)
        shapes[f"{b}.self_attn.q_norm.weight"] = (lm.HEAD_DIM,)
        shapes[f"{b}.self_attn.k_norm.weight"] = (lm.HEAD_DIM,)
        for name in ("gate_proj", "up_proj"):
            shapes[f"{b}.mlp.{name}.weight"] = (lm.INTERMEDIATE_SIZE, lm.HIDDEN_SIZE)
        shapes[f"{b}.mlp.down_proj.weight"] = (lm.HIDDEN_SIZE, lm.INTERMEDIATE_SIZE)
        shapes[f"{b}.input_layernorm.weight"] = (lm.HIDDEN_SIZE,)
        shapes[f"{b}.post_attention_layernorm.weight"] = (lm.HIDDEN_SIZE,)
    return shapes


def _weights(layers: int, prefix: str = "model"):
    # Norms are ones and projections are small random values; the builder is
    # exercised for the graph it emits, not for what the numbers come to.
    rng = np.random.default_rng(0)
    return {
        name: (np.ones(shape, dtype=np.float32) if name.endswith("norm.weight")
               else rng.standard_normal(shape).astype(np.float32) * 0.01)
        for name, shape in _weight_shapes(layers, prefix).items()
    }


def test_weight_fixture_matches_the_published_layer_inventory() -> None:
    """Eleven tensors per layer plus the final norm."""

    assert len(_weight_shapes(1)) == lm.TENSORS_PER_LAYER + 1
    assert len(_weight_shapes(36)) == 36 * lm.TENSORS_PER_LAYER + 1


def _build(seq_len=8, layers=2):
    net = _Network()
    lb.add_language_model(net, _TRT, SimpleNamespace(name="h"), seq_len=seq_len,
                          weights=_weights(layers), num_layers=layers)
    return net


def test_layer_census() -> None:
    kinds = [k for k, _ in _build(layers=2).calls]

    assert kinds.count("softmax") == 2
    assert kinds.count("activation") == 2  # one sigmoid per SwiGLU
    # Two block norms plus q_norm and k_norm per layer, plus the final norm.
    assert kinds.count("reduce") == 2 * 4 + 1


def test_query_and_key_value_heads_are_split_differently() -> None:
    reshapes = [v for _, k, v in _build(seq_len=8, layers=1).attrs
                if k == "reshape_dims"]

    assert (8, lm.NUM_ATTENTION_HEADS, lm.HEAD_DIM) in reshapes
    assert (8, lm.NUM_KEY_VALUE_HEADS, lm.HEAD_DIM) in reshapes


def test_key_value_heads_are_repeated_to_the_query_count() -> None:
    reshapes = [v for _, k, v in _build(seq_len=8, layers=1).attrs
                if k == "reshape_dims"]

    # The expand-then-flatten pair that turns 8 heads into 32.
    assert (lm.NUM_KEY_VALUE_HEADS, 1, 8, lm.HEAD_DIM) in reshapes
    assert (lm.NUM_ATTENTION_HEADS, 8, lm.HEAD_DIM) in reshapes


def test_head_norms_run_over_the_head_not_the_model_width() -> None:
    """q_norm and k_norm reduce over axis 2, which is head_dim."""

    net = _Network()
    cos, sin = lm.rope_tables(8)
    lb.add_attention(net, _TRT, SimpleNamespace(name="h"), _weights(1),
                     "model.layers.0", 8, cos, sin)
    axes = [args for kind, args in net.calls if kind == "reduce"]

    assert all(("axes", 1 << 2) in a for a in axes)
    assert len(axes) == 2  # q_norm and k_norm


def test_rope_is_applied_after_the_head_norm() -> None:
    net = _Network()
    cos, sin = lm.rope_tables(8)
    lb.add_attention(net, _TRT, SimpleNamespace(name="h"), _weights(1),
                     "model.layers.0", 8, cos, sin)
    kinds = [k for k, _ in net.calls]

    first_reduce = kinds.index("reduce")       # inside q_norm
    first_neg = next(i for i, (k, a) in enumerate(net.calls)
                     if k == "unary" and "NEG" in str(a))
    assert first_reduce < first_neg


def test_causal_mask_blocks_only_the_future() -> None:
    mask = lb.causal_mask(4)

    assert np.all(np.isfinite(np.tril(mask)))
    assert np.all(mask[np.triu_indices(4, k=1)] == -np.inf)


def test_io_shapes() -> None:
    shapes = lb.expected_io_shapes(64)

    assert shapes[lb.INPUT_NAME] == (1, 64, lm.HIDDEN_SIZE)
    assert shapes[lb.OUTPUT_NAME] == (1, 64, lm.HIDDEN_SIZE)


def test_configure_states_the_precision_choice() -> None:
    cleared, limits = [], []
    lb.configure(SimpleNamespace(
        set_memory_pool_limit=lambda p, s: limits.append((p, s)),
        clear_flag=cleared.append), _TRT)

    assert limits == [("WORKSPACE", lb.WORKSPACE_BYTES)]
    assert cleared == ["TF32"]
