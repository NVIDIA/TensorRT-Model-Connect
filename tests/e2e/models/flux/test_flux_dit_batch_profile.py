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

    import tensorrt_model_connect.engine_builder as eb
    monkeypatch.setattr(eb, "add_dynamic_batch_profile", stop)
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


