# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the FLUX.1 DiT dynamic-batch profile (PR 1).

Verifies the contract added in the diffusion batch-inference foundation
work (design Decisions A and C):

* ``max_batch_size == 1`` (default) preserves today's static-shape
  behaviour and *must not* attach an optimization profile.
* ``max_batch_size > 1`` switches every input's leading dim to ``-1``
  and calls :func:`add_dynamic_batch_profile` exactly once with the
  expected names, shapes, and ``kMIN/kOPT/kMAX``.

The tests fully stub out the network body — we capture the profile call
and the ``add_input`` shapes, then short-circuit before any further graph
construction happens. No engine is compiled.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from tensorrt_model_connect.families.flux import flux_dit_builder
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt",
                allow_module_level=True)


class _CapturedAndStop(Exception):
    """Sentinel raised inside a stub to short-circuit the builder."""


class _FakeTensor:
    def __init__(self, name, dtype, shape):
        self.name = name
        self.dtype = dtype
        self.shape = tuple(shape)


class _RecordingLayer:
    def __init__(self, out):
        self._out = out
        self.reshape_dims = None
        self.first_transpose = None
        self.second_transpose = None
        self.axis = 0

    def get_output(self, _idx=0):
        return self._out

    def set_input(self, *_a, **_k):
        return None


class _FakeNetwork:
    def __init__(self):
        self.inputs: list[tuple[str, object, tuple]] = []
        self.outputs = []

    def add_input(self, name, dtype, shape):
        self.inputs.append((name, dtype, tuple(shape)))
        return _FakeTensor(name, dtype, tuple(shape))

    def mark_output(self, _t):
        self.outputs.append(_t)

    def __getattr__(self, _name):
        # Any other ``add_*`` call returns a tame recording layer; we never
        # reach it because we short-circuit at profile attach time.
        def _stub(*_a, **_k):
            return _RecordingLayer(_FakeTensor("stub", "fp32", (-1,)))
        return _stub


class _FakeBuilderConfig:
    def set_memory_pool_limit(self, *_a, **_k):
        return None

    def add_optimization_profile(self, _p):
        return None


class _FakeBuilder:
    def __init__(self, _logger=None):
        self._net = _FakeNetwork()
        self._cfg = _FakeBuilderConfig()
        self.builds = 0

    def create_network(self, _flags=0):
        return self._net

    def create_builder_config(self):
        return self._cfg

    def create_optimization_profile(self):
        return self._cfg  # any object; not inspected after our short-circuit

    def build_serialized_network(self, *_a, **_k):
        self.builds += 1
        return b"FAKE-PLAN"


def _fake_trt():
    import types
    fake = types.SimpleNamespace()

    class _Logger:
        WARNING = 0
        VERBOSE = 1

        def __init__(self, _level=0):
            self.level = _level

    fake.Logger = _Logger
    fake.Builder = _FakeBuilder
    fake.MemoryPoolType = types.SimpleNamespace(WORKSPACE=0)
    fake.NetworkDefinitionCreationFlag = types.SimpleNamespace(STRONGLY_TYPED=0)
    fake.ElementWiseOperation = types.SimpleNamespace(SUM=0, PROD=1, SUB=2)
    fake.ReduceOperation = types.SimpleNamespace(AVG=0, SUM=1)
    fake.UnaryOperation = types.SimpleNamespace(SQRT=0, RECIP=1)
    fake.ActivationType = types.SimpleNamespace(SIGMOID=0)
    fake.MatrixOperation = types.SimpleNamespace(NONE=0)
    fake.Permutation = lambda perm: perm
    fake.Weights = lambda *_a: object()
    fake.float32 = "fp32"
    fake.float16 = "fp16"
    fake.int32 = "i32"
    return fake


def _install(monkeypatch):
    captured: dict[str, object] = {}
    monkeypatch.setattr(flux_dit_builder, "trt", _fake_trt())

    def stop(_b, _c, _n, **kwargs):
        captured["profile_kwargs"] = kwargs
        raise _CapturedAndStop()

    from tensorrt_model_connect.tvm_ffi import graph_build

    monkeypatch.setattr(graph_build, "add_dynamic_batch_profile", stop)
    return captured


def test_batched_path_attaches_profile_with_expected_shapes(monkeypatch):
    captured = _install(monkeypatch)

    inputs_seen: list[tuple] = []
    original_add_input = _FakeNetwork.add_input

    def patched_add_input(self, name, dtype, shape):
        inputs_seen.append((name, dtype, tuple(shape)))
        return original_add_input(self, name, dtype, shape)

    monkeypatch.setattr(_FakeNetwork, "add_input", patched_add_input)

    # Architecture knobs — small enough that the builder reaches profile
    # attach quickly; weights dict is irrelevant since we short-circuit.
    dim = 16
    num_heads = 2
    head_dim = dim // num_heads
    num_img_tokens = 4
    text_seq_len = 6
    total_seq = num_img_tokens + text_seq_len

    with pytest.raises(_CapturedAndStop):
        flux_dit_builder.build_flux_dit_engine(
            {},
            dim=dim,
            num_heads=num_heads,
            num_layers=1,
            num_single_layers=1,
            num_img_tokens=num_img_tokens,
            text_seq_len=text_seq_len,
            max_batch_size=4,
        )

    pk = captured["profile_kwargs"]
    assert pk["input_names"] == [
        "hidden_states",
        "encoder_hidden_states",
        "temb",
        "rotary_cos",
        "rotary_sin",
    ]
    assert pk["max_batch"] == 4
    assert pk["opt_batch"] == 4
    assert pk["static_shape"] == {
        "hidden_states": (num_img_tokens, dim),
        "encoder_hidden_states": (text_seq_len, dim),
        "temb": (dim,),
        "rotary_cos": (total_seq, head_dim),
        "rotary_sin": (total_seq, head_dim),
    }

    by_name = {name: shape for name, _dt, shape in inputs_seen}
    assert by_name["hidden_states"] == (-1, num_img_tokens, dim)
    assert by_name["encoder_hidden_states"] == (-1, text_seq_len, dim)
    assert by_name["temb"] == (-1, dim)
    assert by_name["rotary_cos"] == (-1, total_seq, head_dim)
    assert by_name["rotary_sin"] == (-1, total_seq, head_dim)


def test_fp16_precision_casts_inputs_and_matmul_constants(monkeypatch):
    """Intent: prevent FP16 manifests from silently rebuilding an FP32 DiT.

    Preconditions: the FLUX.1 builder is configured for FP16.
    Postconditions: FP32 runtime inputs are cast at the graph boundary and
    matrix weights are materialized as FP16 TensorRT constants.
    """
    casts = []
    matmuls = []

    class _CastNetwork:
        def add_cast(self, tensor, dtype):
            casts.append((tensor.dtype, dtype))
            return _RecordingLayer(_FakeTensor("cast", dtype, tensor.shape))

    def record_matmul(network, inp, in_dim, out_dim, weight, *, dtype):
        matmuls.append((network, inp, in_dim, out_dim, weight, dtype))
        return inp

    monkeypatch.setattr(
        flux_dit_builder.graph_ops,
        "add_matmul_rhs_constant",
        record_matmul,
    )

    try:
        flux_dit_builder._configure_compute_precision("fp16")
        network = _CastNetwork()
        fp32_input = _FakeTensor(
            "hidden_states",
            flux_dit_builder.trt.float32,
            (4, 16),
        )

        fp16_input = flux_dit_builder._to_compute_dtype(network, fp32_input)
        flux_dit_builder._matmul(
            network,
            fp16_input,
            16,
            32,
            np.zeros((16, 32), dtype=np.float32),
        )

        assert casts == [
            (flux_dit_builder.trt.float32, flux_dit_builder.trt.float16)
        ]
        assert fp16_input.dtype == flux_dit_builder.trt.float16
        assert matmuls[0][-1] == np.float16
    finally:
        flux_dit_builder._configure_compute_precision("fp32")


def test_fp16_adaln_keeps_modulation_in_fp32(monkeypatch):
    """Intent: prevent FP16 AdaLN overflow from collapsing generated images.

    Preconditions: normalized activations and modulation vectors originate in
    FP16.
    Postconditions: normalization, scale, shift, and both modulation ops use
    FP32 before a single cast returns the block output to FP16.
    """
    casts = []
    elementwise_dtypes = []

    class _TypedNetwork:
        def add_shuffle(self, tensor):
            return _RecordingLayer(_FakeTensor("shuffle", tensor.dtype, tensor.shape))

        def add_cast(self, tensor, dtype):
            casts.append((tensor.dtype, dtype))
            return _RecordingLayer(_FakeTensor("cast", dtype, tensor.shape))

        def add_elementwise(self, left, right, _operation):
            assert left.dtype == right.dtype
            elementwise_dtypes.append(left.dtype)
            return _RecordingLayer(_FakeTensor("elementwise", left.dtype, left.shape))

    monkeypatch.setattr(
        flux_dit_builder.graph_ops,
        "add_layer_norm_native",
        lambda _network, tensor, *_args, **_kwargs: _FakeTensor(
            "normalized", tensor.dtype, tensor.shape
        ),
    )
    monkeypatch.setattr(
        flux_dit_builder.graph_ops,
        "add_constant",
        lambda _network, shape, _values, dtype=np.float32: _FakeTensor(
            "constant",
            (
                flux_dit_builder.trt.float16
                if dtype == np.float16
                else flux_dit_builder.trt.float32
            ),
            shape,
        ),
    )

    try:
        flux_dit_builder._configure_compute_precision("fp16")
        fp16 = flux_dit_builder.trt.float16
        fp32 = flux_dit_builder.trt.float32
        output = flux_dit_builder._adaln_modulate(
            _TypedNetwork(),
            _FakeTensor("x", fp16, (4, 16)),
            _FakeTensor("scale", fp16, (16,)),
            _FakeTensor("shift", fp16, (16,)),
            16,
            _FakeTensor("eps", fp32, (1, 1)),
            4,
        )

        assert output.dtype == fp16
        assert elementwise_dtypes == [fp32, fp32, fp32]
        assert casts.count((fp16, fp32)) == 3
        assert casts[-1] == (fp32, fp16)
    finally:
        flux_dit_builder._configure_compute_precision("fp32")


def test_fp16_block_updates_accumulate_in_fp32_residual():
    """Intent: prevent deep FLUX.1 residual streams from overflowing FP16.

    Preconditions: a block emits an FP16 update for an FP32 residual.
    Postconditions: the update is promoted and the residual sum remains FP32.
    """
    casts = []
    sums = []

    class _TypedNetwork:
        def add_cast(self, tensor, dtype):
            casts.append((tensor.dtype, dtype))
            return _RecordingLayer(_FakeTensor("cast", dtype, tensor.shape))

        def add_elementwise(self, left, right, operation):
            assert left.dtype == right.dtype
            sums.append((left.dtype, right.dtype, operation))
            return _RecordingLayer(
                _FakeTensor("residual_sum", left.dtype, left.shape))

    try:
        flux_dit_builder._configure_compute_precision("fp16")
        fp16 = flux_dit_builder.trt.float16
        fp32 = flux_dit_builder.trt.float32
        output = flux_dit_builder._residual_add(
            _TypedNetwork(),
            _FakeTensor("residual", fp32, (4, 16)),
            _FakeTensor("update", fp16, (4, 16)),
        )

        assert output.dtype == fp32
        assert casts == [(fp16, fp32)]
        assert sums == [
            (fp32, fp32, flux_dit_builder.trt.ElementWiseOperation.SUM)
        ]
    finally:
        flux_dit_builder._configure_compute_precision("fp32")


def test_fp16_block_gate_runs_before_overflow_in_fp32():
    """Intent: keep finite FP16 branch values finite while gating.

    Preconditions: both a residual-branch projection and its gate are FP16.
    Postconditions: both operands are promoted before their product is formed.
    """
    casts = []
    products = []

    class _TypedNetwork:
        def add_shuffle(self, tensor):
            return _RecordingLayer(_FakeTensor("gate_2d", tensor.dtype, (1, 16)))

        def add_cast(self, tensor, dtype):
            casts.append((tensor.dtype, dtype))
            return _RecordingLayer(_FakeTensor("cast", dtype, tensor.shape))

        def add_elementwise(self, left, right, operation):
            assert left.dtype == right.dtype
            products.append((left.dtype, right.dtype, operation))
            return _RecordingLayer(
                _FakeTensor("gated_update", left.dtype, left.shape))

    try:
        flux_dit_builder._configure_compute_precision("fp16")
        fp16 = flux_dit_builder.trt.float16
        fp32 = flux_dit_builder.trt.float32
        output = flux_dit_builder._gate_1d(
            _TypedNetwork(),
            _FakeTensor("update", fp16, (4, 16)),
            _FakeTensor("gate", fp16, (16,)),
            4,
        )

        assert output.dtype == fp32
        assert casts == [(fp16, fp32), (fp16, fp32)]
        assert products == [
            (fp32, fp32, flux_dit_builder.trt.ElementWiseOperation.PROD)
        ]
    finally:
        flux_dit_builder._configure_compute_precision("fp32")
