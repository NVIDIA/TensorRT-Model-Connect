# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import inspect
import sys
import types

import numpy as np
import pytest
from safetensors.numpy import save_file


def _load_builder(monkeypatch: pytest.MonkeyPatch):
    fake_trt = types.ModuleType("tensorrt")
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    from tensorrt_model_connect import trt_compat

    monkeypatch.setattr(trt_compat, "_module", fake_trt)
    return importlib.import_module("tensorrt_model_connect.families.dinov3.convnext_builder")


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_real_tensorrt_build_marks_hf_outputs(precision: str) -> None:
    trt = pytest.importorskip("tensorrt")
    from tensorrt_model_connect.families.dinov3.convnext_builder import (
        build_convnext_engine,
    )

    dtype = np.float16 if precision == "fp16" else np.float32
    prefix = "stage.0.block.0"
    weights = {
        "stage.0.downsample.weight": np.ones((8, 3, 4, 4), dtype=dtype),
        "stage.0.downsample.bias": np.zeros(8, dtype=dtype),
        "stage.0.downsample_norm.weight": np.ones(8, dtype=dtype),
        "stage.0.downsample_norm.bias": np.zeros(8, dtype=dtype),
        f"{prefix}.depthwise.weight": np.ones((8, 1, 7, 7), dtype=dtype),
        f"{prefix}.depthwise.bias": np.zeros(8, dtype=dtype),
        f"{prefix}.norm.weight": np.ones(8, dtype=dtype),
        f"{prefix}.norm.bias": np.zeros(8, dtype=dtype),
        f"{prefix}.pointwise1.weight": np.ones((8, 32), dtype=dtype),
        f"{prefix}.pointwise1.bias": np.zeros(32, dtype=dtype),
        f"{prefix}.pointwise2.weight": np.ones((32, 8), dtype=dtype),
        f"{prefix}.pointwise2.bias": np.zeros(8, dtype=dtype),
        f"{prefix}.gamma": np.full(8, 1.0e-6, dtype=dtype),
        "final_norm.weight": np.ones(8, dtype=dtype),
        "final_norm.bias": np.zeros(8, dtype=dtype),
    }
    raw = {
        "model_type": "dinov3_convnext",
        "hidden_sizes": [8],
        "depths": [1],
        "image_size": 32,
        "hidden_act": "gelu",
    }

    plan = build_convnext_engine(raw, weights, precision=precision)
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(plan)

    assert engine is not None
    names = {engine.get_tensor_name(index) for index in range(engine.num_io_tensors)}
    assert names == {"pixel_values", "last_hidden_state", "pooler_output"}
    assert tuple(engine.get_tensor_shape("last_hidden_state")) == (1, 65, 8)
    assert tuple(engine.get_tensor_shape("pooler_output")) == (1, 8)
    assert engine.get_tensor_dtype("last_hidden_state") == trt.float32
    assert engine.get_tensor_dtype("pooler_output") == trt.float32


@pytest.mark.parametrize(
    ("hidden_sizes", "depths", "expected_dim", "expected_layers"),
    [
        ([96, 192, 384, 768], [3, 3, 9, 3], 768, 18),
        ([192, 384, 768, 1536], [3, 3, 27, 3], 1536, 36),
    ],
)
def test_resolve_convnext_official_config_shapes(
    monkeypatch: pytest.MonkeyPatch,
    hidden_sizes,
    depths,
    expected_dim,
    expected_layers,
) -> None:
    builder = _load_builder(monkeypatch)
    raw = {
        "model_type": "dinov3_convnext",
        "hidden_sizes": hidden_sizes,
        "depths": depths,
        "image_size": 224,
    }

    resolved = builder.resolve_convnext_config(raw)

    assert resolved["grid_h"] == 7
    assert resolved["grid_w"] == 7
    assert resolved["num_tokens"] == 50
    assert resolved["output_dim"] == expected_dim
    assert resolved["num_layers"] == expected_layers
    assert builder.convnext_bundle_metadata(raw)["num_feature_tokens"] == 50


def _tiny_checkpoint(prefix: str) -> dict[str, np.ndarray]:
    channels = 2

    def values(shape, offset):
        return np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + offset

    tensors = {
        f"{prefix}stages.0.downsample_layers.0.weight": values((2, 3, 4, 4), 1),
        f"{prefix}stages.0.downsample_layers.0.bias": values((2,), 2),
        f"{prefix}stages.0.downsample_layers.1.weight": values((2,), 3),
        f"{prefix}stages.0.downsample_layers.1.bias": values((2,), 4),
        f"{prefix}stages.0.layers.0.depthwise_conv.weight": values((2, 1, 7, 7), 5),
        f"{prefix}stages.0.layers.0.depthwise_conv.bias": values((2,), 6),
        f"{prefix}stages.0.layers.0.layer_norm.weight": values((2,), 7),
        f"{prefix}stages.0.layers.0.layer_norm.bias": values((2,), 8),
        f"{prefix}stages.0.layers.0.pointwise_conv1.weight": values((8, 2), 9),
        f"{prefix}stages.0.layers.0.pointwise_conv1.bias": values((8,), 10),
        f"{prefix}stages.0.layers.0.pointwise_conv2.weight": values((2, 8), 11),
        f"{prefix}stages.0.layers.0.pointwise_conv2.bias": values((2,), 12),
        f"{prefix}stages.0.layers.0.gamma": values((2,), 13),
        f"{prefix}layer_norm.weight": values((2,), 14),
        f"{prefix}layer_norm.bias": values((2,), 15),
    }
    assert channels == tensors[f"{prefix}stages.0.layers.0.gamma"].shape[0]
    return tensors


@pytest.mark.parametrize("prefix", ["", "model."])
def test_load_convnext_weights_accepts_current_and_legacy_prefixes(
    monkeypatch: pytest.MonkeyPatch, tmp_path, prefix: str
) -> None:
    builder = _load_builder(monkeypatch)
    tensors = _tiny_checkpoint(prefix)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    raw = {
        "model_type": "dinov3_convnext",
        "hidden_sizes": [2],
        "depths": [1],
        "image_size": 8,
    }

    weights = builder.load_convnext_weights(tmp_path, raw, precision="fp16")

    assert weights["stage.0.downsample.weight"].shape == (2, 3, 4, 4)
    assert weights["stage.0.block.0.depthwise.weight"].shape == (2, 1, 7, 7)
    assert weights["stage.0.block.0.pointwise1.weight"].shape == (2, 8)
    assert weights["stage.0.block.0.pointwise2.weight"].shape == (8, 2)
    assert weights["final_norm.weight"].shape == (2,)
    assert all(value.dtype == np.float16 for value in weights.values())
    np.testing.assert_array_equal(
        weights["stage.0.block.0.pointwise1.weight"],
        tensors[f"{prefix}stages.0.layers.0.pointwise_conv1.weight"].T.astype(np.float16),
    )


def test_family_load_repairs_generic_bundle_info_fields(tmp_path) -> None:
    pytest.importorskip("tensorrt")
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.families.dinov3.plugin import plugin

    save_file(_tiny_checkpoint(""), str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(
        '{"model_type":"dinov3_convnext","hidden_sizes":[2],'
        '"depths":[1],"image_size":8}',
        encoding="utf-8",
    )
    config = ModelConfig.from_dir(tmp_path)

    plugin.load_weights(str(tmp_path), config, precision="fp32")

    assert config.hidden_size == 2
    assert config.num_hidden_layers == 1
    assert config.num_attention_heads == 0
    assert config.num_key_value_heads == 0


class _FakeTensor:
    shape = (1, 2, 4, 4)
    dtype = "fp32"


class _FakeLayer:
    def __init__(self, output=None):
        self.output = output or _FakeTensor()

    def get_output(self, _index):
        return self.output


class _FakeConv(_FakeLayer):
    stride_nd = None
    padding_nd = None
    num_groups = None


class _FakeShuffle(_FakeLayer):
    first_transpose = None


class _FakeNetwork:
    def __init__(self, events):
        self.events = events
        self.convolution = None

    def add_convolution_nd(self, *_args, **_kwargs):
        self.events.append("depthwise_conv")
        self.convolution = _FakeConv()
        return self.convolution

    def add_shuffle(self, tensor):
        self.events.append("shuffle")
        return _FakeShuffle(tensor)

    def add_elementwise(self, lhs, _rhs, operation):
        self.events.append(("residual", operation))
        return _FakeLayer(lhs)


class _FakeGraphOps:
    def __init__(self, events):
        self.events = events

    def layer_norm(self, _network, tensor, *_args):
        self.events.append("layer_norm")
        return tensor

    def linear(self, _network, tensor, *_args):
        self.events.append("linear")
        return tensor

    def add_bias(self, _network, tensor, *_args):
        self.events.append("bias")
        return tensor

    def activation(self, _network, tensor, name, _dtype):
        self.events.append(("activation", name))
        return tensor

    def multiply_last_dim(self, _network, tensor, *_args):
        self.events.append("gamma")
        return tensor


def test_convnext_block_constructs_depthwise_nhwc_mlp_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder(monkeypatch)
    fake_trt = types.SimpleNamespace(
        Weights=lambda value: value,
        Permutation=lambda value: tuple(value),
        ElementWiseOperation=types.SimpleNamespace(SUM="sum"),
    )
    from tensorrt_model_connect import trt_compat

    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    monkeypatch.setattr(trt_compat, "_module", fake_trt)
    events = []
    network = _FakeNetwork(events)
    graph_ops = _FakeGraphOps(events)
    prefix = "stage.0.block.0"
    weights = {
        f"{prefix}.depthwise.weight": np.ones((2, 1, 7, 7), dtype=np.float32),
        f"{prefix}.depthwise.bias": np.zeros(2, dtype=np.float32),
        f"{prefix}.norm.weight": np.ones(2, dtype=np.float32),
        f"{prefix}.norm.bias": np.zeros(2, dtype=np.float32),
        f"{prefix}.pointwise1.weight": np.ones((2, 8), dtype=np.float32),
        f"{prefix}.pointwise1.bias": np.zeros(8, dtype=np.float32),
        f"{prefix}.pointwise2.weight": np.ones((8, 2), dtype=np.float32),
        f"{prefix}.pointwise2.bias": np.zeros(2, dtype=np.float32),
        f"{prefix}.gamma": np.ones(2, dtype=np.float32),
    }

    output = builder._add_block(
        network,
        _FakeTensor(),
        weights,
        prefix,
        2,
        {"layer_norm_eps": 1.0e-6, "hidden_act": "gelu"},
        np.dtype(np.float32),
        graph_ops,
    )

    assert output is not None
    assert network.convolution.num_groups == 2
    assert network.convolution.padding_nd == (3, 3)
    assert events == [
        "depthwise_conv",
        "shuffle",
        "layer_norm",
        "linear",
        "bias",
        ("activation", "gelu"),
        "linear",
        "bias",
        "gamma",
        "shuffle",
        ("residual", "sum"),
    ]


def test_build_source_is_native_and_marks_hf_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder(monkeypatch)
    source = inspect.getsource(builder.build_convnext_engine)

    assert "add_convolution_nd" not in source  # delegated to the family-owned helper
    assert "_add_conv2d" in source
    assert "add_reduce" in source
    assert 'last_hidden_state.name = "last_hidden_state"' in source
    assert 'pooler_output.name = "pooler_output"' in source
    assert "onnx" not in source.lower()


def test_real_convnext_fp32_matches_transformers(tmp_path) -> None:
    trt = pytest.importorskip("tensorrt")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TensorRT semantic parity")

    from tensorrt_model_connect.families.dinov3.convnext_builder import (
        build_convnext_engine,
        load_convnext_weights,
    )

    rng = np.random.default_rng(37)

    def values(*shape: int) -> np.ndarray:
        return (0.05 * rng.standard_normal(shape)).astype(np.float32)

    channels = 8
    prefix = "stages.0"
    checkpoint = {
        f"{prefix}.downsample_layers.0.weight": values(channels, 3, 4, 4),
        f"{prefix}.downsample_layers.0.bias": values(channels),
        f"{prefix}.downsample_layers.1.weight": values(channels),
        f"{prefix}.downsample_layers.1.bias": values(channels),
        f"{prefix}.layers.0.gamma": values(channels),
        f"{prefix}.layers.0.depthwise_conv.weight": values(channels, 1, 7, 7),
        f"{prefix}.layers.0.depthwise_conv.bias": values(channels),
        f"{prefix}.layers.0.layer_norm.weight": values(channels),
        f"{prefix}.layers.0.layer_norm.bias": values(channels),
        f"{prefix}.layers.0.pointwise_conv1.weight": values(4 * channels, channels),
        f"{prefix}.layers.0.pointwise_conv1.bias": values(4 * channels),
        f"{prefix}.layers.0.pointwise_conv2.weight": values(channels, 4 * channels),
        f"{prefix}.layers.0.pointwise_conv2.bias": values(channels),
        "layer_norm.weight": values(channels),
        "layer_norm.bias": values(channels),
    }
    save_file(checkpoint, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
    raw = {
        "model_type": "dinov3_convnext",
        "hidden_sizes": [channels],
        "depths": [1],
        "image_size": 32,
        "num_channels": 3,
        "hidden_act": "gelu",
        "layer_norm_eps": 1.0e-6,
        "layer_scale_init_value": 1.0e-6,
        "drop_path_rate": 0.0,
    }
    weights = load_convnext_weights(tmp_path, raw, precision="fp32")
    plan = build_convnext_engine(raw, weights, precision="fp32")

    model = transformers.DINOv3ConvNextModel(
        transformers.DINOv3ConvNextConfig(**raw)
    ).eval().cuda()
    expected_names = set(model.state_dict())
    state = {}
    for name, value in checkpoint.items():
        target_name = name
        if target_name not in expected_names and ("model." + name) in expected_names:
            target_name = "model." + name
        if target_name in expected_names:
            state[target_name] = torch.from_numpy(value)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing
    assert not unexpected

    pixels = torch.from_numpy(
        rng.standard_normal((1, 3, 32, 32), dtype=np.float32)
    ).cuda()
    with torch.inference_mode():
        reference = model(pixel_values=pixels)

    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    output = torch.empty((1, 65, channels), dtype=torch.float32, device="cuda")
    pooled = torch.empty((1, channels), dtype=torch.float32, device="cuda")
    for name, tensor in {
        "pixel_values": pixels,
        "last_hidden_state": output,
        "pooler_output": pooled,
    }.items():
        assert context.set_tensor_address(name, tensor.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output, reference.last_hidden_state, atol=5.0e-4, rtol=5.0e-4
    )
    torch.testing.assert_close(pooled, output[:, 0, :], atol=0.0, rtol=0.0)
    cosine = torch.nn.functional.cosine_similarity(
        output.reshape(-1), reference.last_hidden_state.reshape(-1), dim=0
    )
    assert float(cosine) >= 0.999999
