# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 ConvNeXt family-owned builder contracts."""

from __future__ import annotations

import inspect
import types

import numpy as np
import pytest
from safetensors.numpy import save_file

pytest.importorskip("tensorrt", reason="TensorRT is required for DINOv3 builder imports")

from families.dinov3 import convnext_builder  # noqa: E402
from families.dinov3 import model as dinov3_model  # noqa: E402
from families.dinov3.config import ModelConfig  # noqa: E402


def _load_builder(_monkeypatch: pytest.MonkeyPatch):
    return convnext_builder


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


def test_load_convnext_weights_maps_official_prefix(tmp_path) -> None:
    prefix = "model."
    tensors = _tiny_checkpoint(prefix)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    raw = {
        "model_type": "dinov3_convnext",
        "hidden_sizes": [2],
        "depths": [1],
        "image_size": 8,
    }

    weights = convnext_builder.load_convnext_weights(tmp_path, raw, precision="fp16")

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


def test_convnext_load_updates_bundle_architecture_fields(tmp_path) -> None:
    save_file(_tiny_checkpoint("model."), str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(
        '{"model_type":"dinov3_convnext","hidden_sizes":[2],"depths":[1],"image_size":8}',
        encoding="utf-8",
    )
    config = ModelConfig.from_dir(tmp_path)

    dinov3_model._Dinov3Model().load_weights(str(tmp_path), config, precision="fp32")

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

    def shuffle(self, _network, tensor, **_kwargs):
        self.events.append("shuffle")
        return tensor

    def linear_with_bias(self, _network, tensor, *_args):
        self.events.append("linear")
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
    builder = convnext_builder
    fake_trt = types.SimpleNamespace(
        Weights=lambda value: value,
        Permutation=lambda value: tuple(value),
        ElementWiseOperation=types.SimpleNamespace(SUM="sum"),
    )
    monkeypatch.setattr(builder, "trt", fake_trt)
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
    builder = convnext_builder
    source = inspect.getsource(builder.build_convnext_engine)

    assert "add_convolution_nd" not in source  # delegated to the family-owned helper
    assert "_add_conv2d" in source
    assert "add_reduce" in source
    assert 'last_hidden_state.name = "last_hidden_state"' in source
    assert 'pooler_output.name = "pooler_output"' in source
    assert "onnx" not in source.lower()
