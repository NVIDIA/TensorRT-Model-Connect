# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image TensorRT 11 LayerNorm and dynamic-batch contracts."""

from __future__ import annotations

import numpy as np

from families.qwen_image import qwen_image_dit_builder as dit


class _FakeTensor:
    def __init__(self, name: str = "tensor", *, dtype=None, shape=(1,)):
        self.name = name
        self.dtype = dtype if dtype is not None else _FakeTRT.float32
        self.shape = shape


class _FakeLayer:
    def __init__(self, name: str = "layer", *, dtype=None):
        self.name = name
        self._out = _FakeTensor(name, dtype=dtype)
        self.reshape_dims = None
        self.epsilon = None
        self.axis = 0
        self.first_transpose = None
        self.second_transpose = None
        self.compute_precision = None

    def get_output(self, _index: int) -> _FakeTensor:
        return self._out

    def set_input(self, *_args, **_kwargs):
        return None


class _FakeLayerWithoutComputePrecision:
    def __init__(self, name: str = "layer", *, dtype=None):
        self.name = name
        self._out = _FakeTensor(name, dtype=dtype)
        self.epsilon = None

    def get_output(self, _index: int) -> _FakeTensor:
        return self._out


class _FakeNetwork:
    def __init__(self):
        self.inputs: list[tuple[str, object, tuple]] = []

    def add_input(self, name, dtype, shape):
        self.inputs.append((name, dtype, tuple(shape)))
        return _FakeTensor(name, dtype=dtype, shape=tuple(shape))

    def __getattr__(self, item):
        if item == "mark_output":
            return lambda _tensor: None
        return lambda *_args, **_kwargs: _FakeLayer(item)


class _FakeBuilderConfig:
    def __init__(self):
        self.profiles: list[object] = []
        self.builder_optimization_level = 0

    def set_memory_pool_limit(self, *_args, **_kwargs):
        return None

    def add_optimization_profile(self, profile):
        self.profiles.append(profile)

    def clear_flag(self, *_args):
        return None


class _FakeProfile:
    def __init__(self):
        self.shapes: dict[str, tuple] = {}

    def set_shape(self, name, *, min, opt, max):
        self.shapes[name] = (tuple(min), tuple(opt), tuple(max))


class _FakeBuilder:
    def __init__(self):
        self.config = _FakeBuilderConfig()
        self.network = _FakeNetwork()

    def create_builder_config(self):
        return self.config

    def create_network(self, _flags):
        return self.network

    def create_optimization_profile(self):
        return _FakeProfile()

    def build_serialized_network(self, _network, _config):
        return b"FAKE_ENGINE_PLAN"


class _FakeTRT:
    float32 = "float32"
    int32 = "int32"
    bfloat16 = "bfloat16"

    class Logger:
        WARNING = 0
        VERBOSE = 1

        def __init__(self, *_args, **_kwargs):
            pass

    class Builder:
        def __init__(self, _logger):
            self.fake = _FakeBuilder()

        def create_builder_config(self):
            return self.fake.create_builder_config()

        def create_network(self, flags):
            return self.fake.create_network(flags)

        def create_optimization_profile(self):
            return self.fake.create_optimization_profile()

        def build_serialized_network(self, network, config):
            return self.fake.build_serialized_network(network, config)

    class NetworkDefinitionCreationFlag:
        STRONGLY_TYPED = 1

    class MemoryPoolType:
        WORKSPACE = 0

    class BuilderFlag:
        TF32 = 0

    class ElementWiseOperation:
        SUM = 0
        PROD = 1
        SUB = 2

    class UnaryOperation:
        SQRT = 0
        RECIP = 1
        SIN = 2
        COS = 3

    class MatrixOperation:
        NONE = 0

    class Permutation:
        def __init__(self, permutation):
            self.permutation = permutation

    @staticmethod
    def Weights(array, *_args, **_kwargs):  # noqa: N802
        return array


def _tiny_cfg():
    return dit.QwenImageDiTConfig(
        in_channels=4,
        out_channels=1,
        patch_size=2,
        hidden_size=12,
        num_joint_blocks=1,
        num_attention_heads=2,
        attention_head_dim=6,
        intermediate_size=24,
        text_embed_dim=6,
        rope_axes_dim=[2, 2, 2],
        rope_theta=10000.0,
        timestep_embed_dim=4,
        max_image_tokens=8,
        max_text_tokens=4,
        guidance_embeds=False,
    )


def _tiny_weights():
    config = _tiny_cfg()
    hidden = config.hidden_size
    patch = config.patch_size
    rng = np.random.default_rng(0)

    def values(shape):
        return rng.normal(0.0, 0.01, shape).astype(np.float32)

    return {
        "img_in.weight": values((hidden, config.in_channels)),
        "img_in.bias": np.zeros((hidden,), dtype=np.float32),
        "txt_norm.weight": np.ones((config.text_embed_dim,), dtype=np.float32),
        "txt_in.weight": values((hidden, config.text_embed_dim)),
        "txt_in.bias": np.zeros((hidden,), dtype=np.float32),
        "time_text_embed.timestep_embedder.linear_1.weight": values(
            (hidden, config.timestep_embed_dim)
        ),
        "time_text_embed.timestep_embedder.linear_1.bias": np.zeros((hidden,), dtype=np.float32),
        "time_text_embed.timestep_embedder.linear_2.weight": values((hidden, hidden)),
        "time_text_embed.timestep_embedder.linear_2.bias": np.zeros((hidden,), dtype=np.float32),
        "norm_out.linear.weight": values((2 * hidden, hidden)),
        "norm_out.linear.bias": np.zeros((2 * hidden,), dtype=np.float32),
        "proj_out.weight": values((config.out_channels * patch * patch, hidden)),
        "proj_out.bias": np.zeros((config.out_channels * patch * patch,), dtype=np.float32),
    }


def _patch_tensorrt(monkeypatch):
    monkeypatch.setattr(dit, "trt", _FakeTRT)

    def stub_tensor(*_args, **_kwargs):
        return _FakeTensor("stub")

    def stub_two_tensors(*_args, **_kwargs):
        return _FakeTensor("img_out"), _FakeTensor("txt_out")

    monkeypatch.setattr(dit, "_add_linear_3d", stub_tensor)
    monkeypatch.setattr(dit, "_add_rms_norm_last_dim_3d", stub_tensor)
    monkeypatch.setattr(dit, "_add_time_text_embed", stub_tensor)
    monkeypatch.setattr(dit, "_add_norm_out_3d", stub_tensor)
    monkeypatch.setattr(dit, "_add_joint_block_graph", stub_two_tensors)
    monkeypatch.setattr(
        dit,
        "_to_fp32",
        lambda _network, tensor: tensor if isinstance(tensor, _FakeTensor) else _FakeTensor("fp32"),
    )

    def stub_rope(axes_dim, image_shapes, n_text, theta):
        del theta
        head_dim = sum(axes_dim)
        total = sum(height * width for height, width in image_shapes) + n_text
        return (
            np.zeros((total, head_dim), dtype=np.float32),
            np.zeros((total, head_dim), dtype=np.float32),
        )

    monkeypatch.setattr(dit, "_precompute_qwen_rope_tables_for_shapes", stub_rope)
    calls: list[dict] = []

    def record(builder, config, *, input_names, max_batch, opt_batch, static_shape):
        calls.append(
            {
                "builder": builder,
                "config": config,
                "input_names": list(input_names),
                "max_batch": max_batch,
                "opt_batch": opt_batch,
                "static_shape": dict(static_shape),
            }
        )

    monkeypatch.setattr(dit, "_add_dynamic_batch_profile", record)
    return calls


def _capture_network(monkeypatch):
    captured = {}
    real = dit.trt.Builder.create_network

    def wrapper(self, flags):
        network = real(self, flags)
        captured["network"] = network
        return network

    monkeypatch.setattr(dit.trt.Builder, "create_network", wrapper)
    return captured


def test_layernorm_no_affine_tolerates_trt11_without_compute_precision(monkeypatch) -> None:
    class Network(_FakeNetwork):
        def __init__(self):
            super().__init__()
            self.norm = _FakeLayerWithoutComputePrecision("norm")

        def add_normalization(self, *_args, **_kwargs):
            return self.norm

    monkeypatch.setattr(dit, "trt", _FakeTRT)
    network = Network()
    output = dit._add_layernorm_no_affine_3d(
        network,
        _FakeTensor("x", dtype=dit._CAST_DTYPE),
        hidden_size=4,
        eps=1e-6,
    )

    assert output is network.norm.get_output(0)
    assert network.norm.epsilon == 1e-6


def test_max_batch_size_four_uses_dynamic_dim_and_calls_profile(monkeypatch, tmp_path) -> None:
    calls = _patch_tensorrt(monkeypatch)
    captured = _capture_network(monkeypatch)
    config = _tiny_cfg()
    h_lat = 2
    w_lat = 2
    n_text = 3

    dit.build_qwen_image_dit_engine(
        config,
        _tiny_weights(),
        tmp_path / "dit.plan",
        h_lat=h_lat,
        w_lat=w_lat,
        n_text=n_text,
        max_batch_size=4,
    )

    assert len(calls) == 1
    call = calls[0]
    n_img = h_lat * w_lat
    assert call["input_names"] == ["img_patched", "txt_hidden", "timestep", "attention_mask"]
    assert call["max_batch"] == 4
    assert call["opt_batch"] == 4
    assert call["static_shape"] == {
        "img_patched": (n_img, config.in_channels),
        "txt_hidden": (n_text, config.text_embed_dim),
        "timestep": (),
        "attention_mask": (1, 1, n_img + n_text),
    }
    shapes = {name: shape for name, _dtype, shape in captured["network"].inputs}
    assert shapes["img_patched"] == (-1, n_img, config.in_channels)
    assert shapes["txt_hidden"] == (-1, n_text, config.text_embed_dim)
    assert shapes["timestep"] == (-1,)
    assert shapes["attention_mask"] == (-1, 1, 1, n_img + n_text)
