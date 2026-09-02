# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the RVQ depth decoder geometry and its TensorRT graph."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

dd = importlib.import_module(
    "tensorrt_model_connect.families.minimax_music3.depth_decoder"
)
db = importlib.import_module(
    "tensorrt_model_connect.families.minimax_music3.depth_decoder_builder"
)


def test_head_dim_is_derived_not_configured() -> None:
    """4096 over sixteen heads is 256 -- four times the DiT's head width."""

    assert dd.head_dim() == 256
    assert dd.attention_scale() == pytest.approx(256 ** -0.5)


def test_embedding_table_holds_the_seven_residual_codebooks() -> None:
    """audio_embeddings.weight is [7168, 4096] in the published checkpoint."""

    assert dd.NUM_RESIDUAL_CODEBOOKS == 7
    assert dd.embedding_rows() == 7168


def test_code_offsets_lay_the_tables_end_to_end() -> None:
    assert [dd.code_offset(i) for i in range(7)] == [
        0, 1024, 2048, 3072, 4096, 5120, 6144
    ]
    with pytest.raises(ValueError, match="codebook must be"):
        dd.code_offset(7)


def test_depth_sequence_fits_the_position_table() -> None:
    assert dd.steps_for(0) == 1
    assert dd.steps_for(7) == 8
    assert dd.steps_for(7) <= dd.MAX_POSITION_EMBEDDINGS
    with pytest.raises(ValueError, match="codes_sampled"):
        dd.steps_for(8)


def test_causal_mask_blocks_only_the_future() -> None:
    mask = dd.causal_mask(4)

    assert mask.shape == (4, 4)
    assert np.all(np.isfinite(np.tril(mask)))
    assert np.all(mask[np.triu_indices(4, k=1)] == -np.inf)
    assert np.all(np.diag(mask) == 0.0)


def test_causal_mask_rejects_a_sequence_past_the_table() -> None:
    with pytest.raises(ValueError, match="exceeds the position table"):
        dd.causal_mask(dd.MAX_POSITION_EMBEDDINGS + 1)


def test_rms_norm_normalises_then_scales() -> None:
    # A real signal, not zeros: this asserts unit RMS, which zeros cannot show.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3, 8)).astype(np.float32)
    w = np.full(8, 2.0, dtype=np.float32)

    out = dd.rms_norm(x, w)

    unit = dd.rms_norm(x, np.ones(8, dtype=np.float32))
    assert np.allclose(np.sqrt((unit ** 2).mean(axis=-1)), 1.0, atol=1e-3)
    assert np.allclose(out, 2.0 * unit, atol=1e-6)


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


_TRT = SimpleNamespace(
    Weights=lambda a: ("W", tuple(np.shape(a))),
    ElementWiseOperation=SimpleNamespace(PROD="PROD", SUM="SUM", DIV="DIV"),
    ReduceOperation=SimpleNamespace(AVG="AVG"),
    UnaryOperation=SimpleNamespace(SQRT="SQRT"),
    MatrixOperation=SimpleNamespace(NONE="NONE", TRANSPOSE="TRANSPOSE"),
    ActivationType=SimpleNamespace(SIGMOID="SIGMOID"),
    MemoryPoolType=SimpleNamespace(WORKSPACE="WORKSPACE"),
    BuilderFlag=SimpleNamespace(TF32="TF32"),
)


    # np.zeros, not standard_normal: the _TRT.Weights stub records only
    # np.shape and no assertion reads a value, so drawing production-width
    # randoms (and the float64 intermediates standard_normal makes) is
    # pure cost -- 2.28 GB per call here.
def _weights():
    w = {
        "pos_embedding.weight": np.zeros(
            (dd.MAX_POSITION_EMBEDDINGS, dd.HIDDEN_SIZE), dtype=np.float32),
        "norm.weight": np.ones(dd.HIDDEN_SIZE, dtype=np.float32),
    }
    for layer in range(dd.NUM_LAYERS):
        p = f"layers.{layer}"
        for name in ("to_q", "to_k", "to_v", "to_out"):
            w[f"{p}.attn.{name}.weight"] = np.zeros(
                (dd.HIDDEN_SIZE, dd.HIDDEN_SIZE), dtype=np.float32)
        w[f"{p}.gate_proj.weight"] = np.zeros(
            (dd.INTERMEDIATE_SIZE, dd.HIDDEN_SIZE), dtype=np.float32)
        w[f"{p}.up_proj.weight"] = np.zeros(
            (dd.INTERMEDIATE_SIZE, dd.HIDDEN_SIZE), dtype=np.float32)
        w[f"{p}.down_proj.weight"] = np.zeros(
            (dd.HIDDEN_SIZE, dd.INTERMEDIATE_SIZE), dtype=np.float32)
        w[f"{p}.input_layernorm.weight"] = np.ones(dd.HIDDEN_SIZE, dtype=np.float32)
        w[f"{p}.post_attention_layernorm.weight"] = np.ones(
            dd.HIDDEN_SIZE, dtype=np.float32)
    return w


def test_weight_fixture_matches_the_published_tensor_count() -> None:
    """47 tensors, minus the three the pipeline owns rather than the forward."""

    owned_elsewhere = 1 + 7 + 1  # audio_embeddings, seven heads, projection
    assert len(_weights()) == 47 - owned_elsewhere


def _build(steps: int = 8):
    net = _Network()
    db.add_depth_decoder(net, _TRT, SimpleNamespace(name="in"),
                         steps=steps, weights=_weights())
    return net


def test_layer_census() -> None:
    kinds = [k for k, _ in _build().calls]

    # Nine RMS norms: two per block plus the final one.
    assert kinds.count("softmax") == dd.NUM_LAYERS
    assert kinds.count("activation") == dd.NUM_LAYERS  # one sigmoid per SwiGLU
    assert kinds.count("reduce") == 2 * dd.NUM_LAYERS + 1
    # Four projections and two attention matmuls per block, three in SwiGLU.
    assert kinds.count("matmul") == dd.NUM_LAYERS * 9


def test_softmax_runs_over_the_key_axis() -> None:
    axes = [v for _, k, v in _build().attrs if k == "axes"]

    assert axes == [1 << 2] * dd.NUM_LAYERS


def test_heads_are_split_and_merged_around_attention() -> None:
    reshapes = [v for _, k, v in _build(8).attrs if k == "reshape_dims"]

    assert reshapes[0] == (8, dd.NUM_ATTENTION_HEADS, dd.head_dim())
    assert (1, 8, dd.HIDDEN_SIZE) in reshapes


def test_position_table_is_sliced_to_the_step_count() -> None:
    net = _build(3)
    constants = [args[0] for kind, args in net.calls if kind == "constant"]

    assert (1, 3, dd.HIDDEN_SIZE) in constants


def test_io_shapes() -> None:
    shapes = db.expected_io_shapes(8)

    assert shapes[db.INPUT_NAME] == (1, 8, dd.HIDDEN_SIZE)
    assert shapes[db.OUTPUT_NAME] == (1, 8, dd.HIDDEN_SIZE)


def test_configure_states_the_precision_choice() -> None:
    cleared, limits = [], []
    db.configure(SimpleNamespace(
        set_memory_pool_limit=lambda p, s: limits.append((p, s)),
        clear_flag=cleared.append), _TRT)

    assert limits == [("WORKSPACE", db.WORKSPACE_BYTES)]
    assert cleared == ["TF32"]
