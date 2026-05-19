"""Tests for the diffusion dynamic-batch optimization-profile helper."""

from __future__ import annotations

import pytest

from .conftest import requires_trt


def test_add_dynamic_batch_profile_rejects_max_zero():
    from tensorrt_model_connect.engine_builder import add_dynamic_batch_profile

    class _FakeProfile:
        def set_shape(self, *args, **kwargs):
            pass

    class _FakeBuilder:
        def create_optimization_profile(self):
            return _FakeProfile()

    class _FakeConfig:
        def add_optimization_profile(self, profile):
            pass

    with pytest.raises(ValueError, match="max_batch must be >= 1"):
        add_dynamic_batch_profile(
            _FakeBuilder(), _FakeConfig(), object(),
            input_names=["x"],
            max_batch=0,
            opt_batch=1,
            static_shape={"x": (8,)},
        )


def test_add_dynamic_batch_profile_rejects_opt_above_max():
    from tensorrt_model_connect.engine_builder import add_dynamic_batch_profile

    class _FakeProfile:
        def set_shape(self, *args, **kwargs):
            pass

    class _FakeBuilder:
        def create_optimization_profile(self):
            return _FakeProfile()

    class _FakeConfig:
        def add_optimization_profile(self, profile):
            pass

    with pytest.raises(ValueError, match="opt_batch must satisfy"):
        add_dynamic_batch_profile(
            _FakeBuilder(), _FakeConfig(), object(),
            input_names=["x"],
            max_batch=4,
            opt_batch=8,
            static_shape={"x": (8,)},
        )


def test_add_dynamic_batch_profile_rejects_missing_static_shape():
    from tensorrt_model_connect.engine_builder import add_dynamic_batch_profile

    class _FakeProfile:
        def set_shape(self, *args, **kwargs):
            pass

    class _FakeBuilder:
        def create_optimization_profile(self):
            return _FakeProfile()

    class _FakeConfig:
        def add_optimization_profile(self, profile):
            pass

    with pytest.raises(KeyError, match="static_shape missing"):
        add_dynamic_batch_profile(
            _FakeBuilder(), _FakeConfig(), object(),
            input_names=["hidden_states", "encoder_hidden_states"],
            max_batch=4,
            opt_batch=4,
            static_shape={"hidden_states": (64, 128)},
        )


def test_add_dynamic_batch_profile_sets_min_opt_max_shapes():
    from tensorrt_model_connect.engine_builder import add_dynamic_batch_profile

    class _Profile:
        def __init__(self):
            self.shapes: dict[str, tuple] = {}

        def set_shape(self, name, *, min, opt, max):
            self.shapes[name] = (tuple(min), tuple(opt), tuple(max))

    class _Builder:
        def __init__(self):
            self.profile = _Profile()

        def create_optimization_profile(self):
            return self.profile

    class _Config:
        def __init__(self):
            self.added = []

        def add_optimization_profile(self, profile):
            self.added.append(profile)

    builder = _Builder()
    config = _Config()

    add_dynamic_batch_profile(
        builder, config, object(),
        input_names=["hidden_states", "encoder_hidden_states", "temb"],
        max_batch=4,
        opt_batch=4,
        static_shape={
            "hidden_states": (64, 128),
            "encoder_hidden_states": (32, 256),
            "temb": (768,),
        },
    )

    assert len(config.added) == 1
    assert builder.profile.shapes["hidden_states"] == (
        (1, 64, 128), (4, 64, 128), (4, 64, 128)
    )
    assert builder.profile.shapes["encoder_hidden_states"] == (
        (1, 32, 256), (4, 32, 256), (4, 32, 256)
    )
    assert builder.profile.shapes["temb"] == ((1, 768), (4, 768), (4, 768))


@requires_trt
def test_add_dynamic_batch_profile_builds_a_trivial_engine(tmp_path):
    import tensorrt as trt
    from tensorrt_model_connect.engine_builder import add_dynamic_batch_profile

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    x = network.add_input("x", trt.float32, (-1, 8))
    identity = network.add_identity(x)
    network.mark_output(identity.get_output(0))
    config = builder.create_builder_config()

    add_dynamic_batch_profile(
        builder, config, network,
        input_names=["x"],
        max_batch=4,
        opt_batch=4,
        static_shape={"x": (8,)},
    )

    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(bytes(serialized))
    ctx = engine.create_execution_context()
    for batch in (1, 2, 3, 4):
        assert ctx.set_input_shape("x", (batch, 8))
    assert ctx.set_input_shape("x", (5, 8)) is False
