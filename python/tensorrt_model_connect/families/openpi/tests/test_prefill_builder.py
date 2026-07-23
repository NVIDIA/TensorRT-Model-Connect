# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import tensorrt_model_connect.families.openpi as openpi_package
from tensorrt_model_connect.families.openpi import prefill_builder as prefill_builder_module
from tensorrt_model_connect.families.openpi.model_config import (
    GemmaConfig,
    VisionConfig,
    get_profile,
)
from tensorrt_model_connect.families.openpi.prefill_builder import (
    _gelu_tanh,
    build_prefill_engine,
    prefill_input_contract,
    prefix_cache_output_name,
    required_prefill_weight_shapes,
    validate_prefill_weights,
)


def _tiny_profile():
    return replace(
        get_profile("pi05_droid"),
        action_dim=3,
        external_state_dim=2,
        external_action_dim=2,
        max_token_length=5,
        vocab_size=11,
        vision=VisionConfig(
            image_size=4,
            patch_size=2,
            width=4,
            depth=2,
            mlp_dim=6,
            num_heads=2,
            output_width=8,
            num_image_slots=3,
        ),
        prefix=GemmaConfig(
            width=8,
            depth=2,
            mlp_dim=10,
            num_heads=2,
            num_kv_heads=1,
            head_dim=4,
        ),
        action_expert=GemmaConfig(
            width=4,
            depth=2,
            mlp_dim=7,
            num_heads=2,
            num_kv_heads=1,
            head_dim=4,
        ),
    )


def _tiny_weights(profile) -> dict[str, np.ndarray]:
    return {
        name: np.full(shape, (index + 1) / 1000.0, dtype=np.float32)
        for index, (name, shape) in enumerate(required_prefill_weight_shapes(profile).items())
    }


def test_production_io_contract_is_fixed_batch_one_and_compact() -> None:
    inputs = prefill_input_contract("pi05_droid")
    assert [(item.name, item.shape, item.dtype) for item in inputs] == [
        ("pixel_values", (3, 3, 224, 224), "float32"),
        ("token_ids", (1, 200), "int32"),
        ("prefix_mask", (1, 968), "bool"),
        ("prefix_position_ids", (1, 968), "int32"),
    ]


def test_production_prefill_builder_exposes_no_trace_plan_switches() -> None:
    parameters = inspect.signature(build_prefill_engine).parameters
    assert "debug_outputs" not in parameters
    assert "debug_prefix_layer" not in parameters
    assert "emit_prefix_hidden" not in parameters


def test_required_production_weights_use_canonical_compact_names() -> None:
    shapes = required_prefill_weight_shapes("pi05_droid")
    assert len(shapes) == 603
    assert shapes["vision.patch_embedding.weight"] == (1152, 3, 14, 14)
    assert shapes["prefix.layer.0.attention.q.weight"] == (2048, 2048)
    assert shapes["prefix.layer.0.attention.k.weight"] == (2048, 256)
    assert shapes["prefix.layer.17.attention.v.weight"] == (2048, 256)
    assert shapes["prefix.layer.17.attention.o.weight"] == (2048, 2048)
    assert all(not name.startswith("action.") for name in shapes)


def test_weight_validation_reports_missing_shape_and_dtype() -> None:
    profile = _tiny_profile()
    weights = _tiny_weights(profile)
    validate_prefill_weights(weights, profile)

    missing = dict(weights)
    del missing["prefix.embedding"]
    with pytest.raises(ValueError, match="missing=prefix.embedding"):
        validate_prefill_weights(missing, profile)

    wrong_shape = dict(weights)
    wrong_shape["prefix.embedding"] = np.zeros((10, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="expected \\(11, 8\\), got \\(10, 8\\)"):
        validate_prefill_weights(wrong_shape, profile)

    wrong_dtype = dict(weights)
    wrong_dtype["prefix.embedding"] = np.zeros((11, 8), dtype=np.int32)
    with pytest.raises(ValueError, match="expected floating point"):
        validate_prefill_weights(wrong_dtype, profile)


class _FakeTensor:
    def __init__(self, shape, dtype):
        self.shape = tuple(int(dim) for dim in shape)
        self.dtype = dtype
        self.name = ""


class _FakeLayer:
    def __init__(self, output):
        self.output = output
        self.name = ""

    def get_output(self, index):
        assert index == 0
        return self.output


class _FakeShuffle:
    def __init__(self, tensor):
        self.tensor = tensor
        self.first_transpose = None
        self.second_transpose = None
        self.reshape_dims = None

    def get_output(self, index):
        assert index == 0
        shape = self.tensor.shape
        if self.first_transpose is not None:
            shape = tuple(shape[index] for index in self.first_transpose)
        if self.reshape_dims is not None:
            reshape = list(self.reshape_dims)
            if -1 in reshape:
                known = int(np.prod([dim for dim in reshape if dim != -1]))
                reshape[reshape.index(-1)] = int(np.prod(shape)) // known
            shape = tuple(reshape)
        if self.second_transpose is not None:
            shape = tuple(shape[index] for index in self.second_transpose)
        return _FakeTensor(shape, self.tensor.dtype)


class _FakeConvolution:
    def __init__(self, tensor, out_channels, kernel):
        self.tensor = tensor
        self.out_channels = out_channels
        self.kernel = kernel
        self._stride = (1, 1)

    @property
    def stride_nd(self):
        return self._stride

    @stride_nd.setter
    def stride_nd(self, value):
        self._stride = tuple(value)

    def get_output(self, index):
        assert index == 0
        batch, _channels, height, width = self.tensor.shape
        out_h = (height - self.kernel[0]) // self._stride[0] + 1
        out_w = (width - self.kernel[1]) // self._stride[1] + 1
        return _FakeTensor((batch, self.out_channels, out_h, out_w), self.tensor.dtype)


class _FakeConcatenation:
    def __init__(self, tensors):
        self.tensors = tensors
        self.axis = 0

    def get_output(self, index):
        assert index == 0
        shape = list(self.tensors[0].shape)
        shape[self.axis] = sum(tensor.shape[self.axis] for tensor in self.tensors)
        return _FakeTensor(shape, self.tensors[0].dtype)


def _broadcast_shape(*shapes):
    return tuple(np.broadcast_shapes(*shapes))


class _FakeNetwork:
    def __init__(self, trt):
        self.trt = trt
        self.inputs = []
        self.outputs = []
        self.softmax_input_dtypes = []
        self.einsum_equations = []
        self.plugin_layers = []
        self.events = []

    def add_input(self, name, dtype, shape):
        tensor = _FakeTensor(shape, dtype)
        tensor.name = name
        self.inputs.append(tensor)
        return tensor

    def add_convolution_nd(self, tensor, out_channels, kernel, weight, bias):
        del weight, bias
        return _FakeConvolution(tensor, out_channels, kernel)

    def add_shuffle(self, tensor):
        return _FakeShuffle(tensor)

    def add_elementwise(self, left, right, operation):
        del operation
        return _FakeLayer(_FakeTensor(_broadcast_shape(left.shape, right.shape), left.dtype))

    def add_reduce(self, tensor, operation, axes, keep_dims):
        del operation
        shape = list(tensor.shape)
        for axis in range(len(shape)):
            if axes & (1 << axis):
                shape[axis] = 1 if keep_dims else 0
        if not keep_dims:
            shape = [dim for dim in shape if dim != 0]
        return _FakeLayer(_FakeTensor(shape, tensor.dtype))

    def add_unary(self, tensor, operation):
        del operation
        return _FakeLayer(_FakeTensor(tensor.shape, tensor.dtype))

    def add_activation(self, tensor, activation):
        del activation
        return _FakeLayer(_FakeTensor(tensor.shape, tensor.dtype))

    def add_matrix_multiply(self, left, left_op, right, right_op):
        left_shape = left.shape
        right_shape = right.shape
        if left_op == self.trt.MatrixOperation.TRANSPOSE:
            left_shape = (*left_shape[:-2], left_shape[-1], left_shape[-2])
        if right_op == self.trt.MatrixOperation.TRANSPOSE:
            right_shape = (*right_shape[:-2], right_shape[-1], right_shape[-2])
        assert left_shape[-1] == right_shape[-2]
        batch = _broadcast_shape(left_shape[:-2], right_shape[:-2])
        return _FakeLayer(_FakeTensor((*batch, left_shape[-2], right_shape[-1]), left.dtype))

    def add_einsum(self, tensors, equation):
        self.einsum_equations.append(equation)
        self.events.append(("einsum", equation))
        left, right = tensors
        if equation == "bsi,iqhd->bsqhd":
            shape = (left.shape[0], left.shape[1], *right.shape[1:])
        elif equation == "bsi,io->bso":
            shape = (left.shape[0], left.shape[1], right.shape[1])
        elif equation == "bhqd,bhkd->bhqk":
            shape = (left.shape[0], left.shape[1], left.shape[2], right.shape[2])
        elif equation == "bhqk,bhkd->bhqd":
            shape = (left.shape[0], left.shape[1], left.shape[2], right.shape[3])
        elif ",z->" in equation and equation.split(",z->", 1)[0] == equation.split(",z->", 1)[1]:
            assert right.shape == (1,)
            shape = left.shape
        elif ",ij->" in equation:
            lhs, output = equation.split(",ij->", 1)
            assert lhs.endswith("i")
            assert output.endswith("j")
            assert left.shape[-1] == right.shape[0]
            shape = (*left.shape[:-1], right.shape[1])
        else:  # pragma: no cover - catches unexpected builder drift.
            raise AssertionError(f"unsupported fake einsum equation {equation!r}")
        return _FakeLayer(_FakeTensor(shape, left.dtype))

    def add_select(self, condition, then_tensor, else_tensor):
        shape = _broadcast_shape(condition.shape, then_tensor.shape, else_tensor.shape)
        return _FakeLayer(_FakeTensor(shape, then_tensor.dtype))

    def add_softmax(self, tensor):
        self.softmax_input_dtypes.append(tensor.dtype)
        self.events.append(("softmax",))
        layer = _FakeLayer(_FakeTensor(tensor.shape, tensor.dtype))
        layer.axes = 0
        return layer

    def add_gather(self, data, indices, axis):
        shape = (*data.shape[:axis], *indices.shape, *data.shape[axis + 1 :])
        return _FakeLayer(_FakeTensor(shape, data.dtype))

    def add_slice(self, tensor, start, shape, stride):
        del start, stride
        return _FakeLayer(_FakeTensor(shape, tensor.dtype))

    def add_concatenation(self, tensors):
        return _FakeConcatenation(tensors)

    def add_plugin_v3(self, inputs, shape_inputs, plugin):
        assert not shape_inputs
        layer = _FakeLayer(_FakeTensor(inputs[0].shape, inputs[0].dtype))
        self.plugin_layers.append((layer, tuple(inputs), plugin))
        self.events.append(("plugin", plugin))
        return layer

    def mark_output(self, tensor):
        self.outputs.append(tensor)


class _FakeBuilder:
    def build_serialized_network(self, network, config):
        del network, config
        return b"fake-prefill-plan"


class _FakeBuilderConfig:
    def __init__(self):
        self.cleared_flags = []

    def clear_flag(self, flag):
        self.cleared_flags.append(flag)


def _fake_graph_ops_module():
    trt = types.SimpleNamespace(
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
        int32="int32",
        bool="bool",
        Weights=lambda value: value,
        MatrixOperation=types.SimpleNamespace(NONE="none", TRANSPOSE="transpose"),
        ElementWiseOperation=types.SimpleNamespace(
            PROD="prod",
            SUM="sum",
            AND="and",
            SUB="sub",
            MAX="max",
            DIV="div",
            POW="pow",
        ),
        ReduceOperation=types.SimpleNamespace(AVG="avg", SUM="sum", MAX="max"),
        UnaryOperation=types.SimpleNamespace(SQRT="sqrt", RECIP="recip", EXP="exp"),
        ActivationType=types.SimpleNamespace(TANH="tanh"),
        BuilderFlag=types.SimpleNamespace(TF32="tf32"),
    )
    module = types.ModuleType("tensorrt_model_connect.families.openpi.graph_ops")
    module.trt = trt
    module.cast_transitions = []
    module.linear_weight_shapes = []

    def create_builder_context(**kwargs):
        module.builder_context_kwargs = kwargs
        module.last_network = _FakeNetwork(trt)
        module.last_builder_config = _FakeBuilderConfig()
        return _FakeBuilder(), module.last_network, module.last_builder_config

    def precision_types(precision):
        return np.dtype(np.float32), {
            "bf16": trt.bfloat16,
            "fp16": trt.float16,
            "fp32": trt.float32,
        }[precision]

    def constant(network, value, *, dtype=None, shape=None):
        del network
        array = np.asarray(value)
        result_shape = tuple(shape) if shape is not None else (array.shape or (1,))
        return _FakeTensor(result_shape, dtype or trt.float32)

    def cast(network, tensor, dtype):
        del network
        if tensor.dtype != dtype:
            module.cast_transitions.append((tensor.dtype, dtype))
        return tensor if tensor.dtype == dtype else _FakeTensor(tensor.shape, dtype)

    def linear(network, tensor, weight, bias=None, *, dtype=None):
        del network, bias
        weight_shape = np.asarray(weight).shape
        module.linear_weight_shapes.append(weight_shape)
        return _FakeTensor((*tensor.shape[:-1], weight_shape[1]), dtype or tensor.dtype)

    def same_shape(network, tensor, *args, **kwargs):
        del network, args, kwargs
        return _FakeTensor(tensor.shape, tensor.dtype)

    def gated_residual(network, residual, update, gate=None):
        del network, update, gate
        return _FakeTensor(residual.shape, residual.dtype)

    module.create_builder_context = create_builder_context
    module.precision_types = precision_types
    module.constant = constant
    module.cast = cast
    module.linear = linear
    module.gelu_tanh = same_shape
    module.apply_rope = same_shape
    module.gated_residual = gated_residual
    return module


def test_siglip_layer_norm_plugin_is_fail_closed_and_names_layer(monkeypatch) -> None:
    fake_ops = _fake_graph_ops_module()
    network = _FakeNetwork(fake_ops.trt)
    plugin = object()
    monkeypatch.setattr(
        prefill_builder_module,
        "_create_openpi_siglip_layer_norm_plugin",
        lambda *, epsilon, trt: plugin,
    )
    activation = _FakeTensor((1, 256, 1152), fake_ops.trt.bfloat16)
    gamma = np.ones((1152,), dtype=np.float32)
    beta = np.zeros((1152,), dtype=np.float32)
    name = "openpi/siglip/camera_2/layer_26/norm2"

    output = prefill_builder_module._apply_siglip_layer_norm_plugin(
        network,
        activation,
        gamma,
        beta,
        epsilon=1.0e-6,
        name=name,
        ops=fake_ops,
        trt=fake_ops.trt,
    )

    assert output.shape == (1, 256, 1152)
    assert output.dtype == fake_ops.trt.bfloat16
    assert len(network.plugin_layers) == 1
    layer, inputs, observed_plugin = network.plugin_layers[0]
    assert layer.name == name
    assert observed_plugin is plugin
    assert [(item.shape, item.dtype) for item in inputs] == [
        ((1, 256, 1152), fake_ops.trt.bfloat16),
        ((1152,), fake_ops.trt.bfloat16),
        ((1152,), fake_ops.trt.bfloat16),
    ]

    with pytest.raises(ValueError, match="epsilon=1e-6"):
        prefill_builder_module._apply_siglip_layer_norm_plugin(
            network,
            activation,
            gamma,
            beta,
            epsilon=1.0e-5,
            name=name,
            ops=fake_ops,
            trt=fake_ops.trt,
        )
    with pytest.raises(ValueError, match=r"BF16 activation \[1,256,1152\]"):
        prefill_builder_module._apply_siglip_layer_norm_plugin(
            network,
            _FakeTensor((1, 256, 1152), fake_ops.trt.float16),
            gamma,
            beta,
            epsilon=1.0e-6,
            name=name,
            ops=fake_ops,
            trt=fake_ops.trt,
        )
    with pytest.raises(ValueError, match=r"BF16 activation \[1,256,1152\]"):
        prefill_builder_module._apply_siglip_layer_norm_plugin(
            network,
            _FakeTensor((1, 255, 1152), fake_ops.trt.bfloat16),
            gamma,
            beta,
            epsilon=1.0e-6,
            name=name,
            ops=fake_ops,
            trt=fake_ops.trt,
        )
    with pytest.raises(ValueError, match=r"gamma/beta \[1152\]"):
        prefill_builder_module._apply_siglip_layer_norm_plugin(
            network,
            activation,
            gamma.reshape(1, 1152),
            beta,
            epsilon=1.0e-6,
            name=name,
            ops=fake_ops,
            trt=fake_ops.trt,
        )
    with pytest.raises(ValueError, match="deterministic layer name"):
        prefill_builder_module._apply_siglip_layer_norm_plugin(
            network,
            activation,
            gamma,
            beta,
            epsilon=1.0e-6,
            name="",
            ops=fake_ops,
            trt=fake_ops.trt,
        )
    assert len(network.plugin_layers) == 1


def test_siglip_attention_residual_plugin_is_fixed_shape_and_names_layer(
    monkeypatch,
) -> None:
    fake_ops = _fake_graph_ops_module()
    network = _FakeNetwork(fake_ops.trt)
    plugin = object()
    monkeypatch.setattr(
        prefill_builder_module,
        "_create_openpi_siglip_attention_residual_plugin",
        lambda *, trt: plugin,
    )
    hidden = _FakeTensor((1, 256, 1152), fake_ops.trt.bfloat16)
    arrays = (
        np.broadcast_to(np.float32(1.0), (1152,)),
        np.broadcast_to(np.float32(0.0), (1152,)),
        np.broadcast_to(np.float32(0.0), (1152, 3456)),
        np.broadcast_to(np.float32(0.0), (3456,)),
        np.broadcast_to(np.float32(0.0), (1152, 1152)),
        np.broadcast_to(np.float32(0.0), (1152,)),
    )
    name = "openpi/siglip/camera_0/layer_00/attention_residual"

    output = prefill_builder_module._apply_siglip_attention_residual_plugin(
        network,
        hidden,
        *arrays,
        name=name,
        ops=fake_ops,
        trt=fake_ops.trt,
    )

    assert output.shape == hidden.shape
    assert output.dtype == fake_ops.trt.bfloat16
    assert len(network.plugin_layers) == 1
    layer, inputs, observed_plugin = network.plugin_layers[0]
    assert layer.name == name
    assert observed_plugin is plugin
    assert [(item.shape, item.dtype) for item in inputs] == [
        ((1, 256, 1152), fake_ops.trt.bfloat16),
        ((1152,), fake_ops.trt.bfloat16),
        ((1152,), fake_ops.trt.bfloat16),
        ((1152, 3456), fake_ops.trt.bfloat16),
        ((3456,), fake_ops.trt.bfloat16),
        ((1152, 1152), fake_ops.trt.bfloat16),
        ((1152,), fake_ops.trt.bfloat16),
    ]

    with pytest.raises(ValueError, match=r"BF16 hidden \[1,256,1152\]"):
        prefill_builder_module._apply_siglip_attention_residual_plugin(
            network,
            _FakeTensor((1, 255, 1152), fake_ops.trt.bfloat16),
            *arrays,
            name=name,
            ops=fake_ops,
            trt=fake_ops.trt,
        )
    with pytest.raises(ValueError, match=r"QKV weight/bias"):
        prefill_builder_module._apply_siglip_attention_residual_plugin(
            network,
            hidden,
            *arrays[:2],
            np.zeros((1152, 3455), dtype=np.float32),
            *arrays[3:],
            name=name,
            ops=fake_ops,
            trt=fake_ops.trt,
        )
    with pytest.raises(ValueError, match="deterministic layer name"):
        prefill_builder_module._apply_siglip_attention_residual_plugin(
            network,
            hidden,
            *arrays,
            name="",
            ops=fake_ops,
            trt=fake_ops.trt,
        )
    assert len(network.plugin_layers) == 1


def test_siglip_attention_residual_plugin_source_is_exact_and_fail_closed() -> None:
    root = Path(__file__).resolve().parents[5]
    source = (
        root / "src/runtime/models/openpi/trt_plugins/siglip_attention_residual_plugin.cu"
    ).read_text(encoding="utf-8")
    cmake = (root / "src/runtime/models/openpi/model.cmake").read_text(encoding="utf-8")

    assert 'kName = "OpenPISiglipAttentionResidual"' in source
    assert 'kVersion = "1"' in source
    assert "nb_inputs != 7" in source
    assert "has_supported_dynamic_shape(input, nb_inputs, output, nb_outputs)" in source
    assert "kQkvCublasWorkspaceBytes = 8552448" in source
    assert "kQkCublasWorkspaceBytes = 1179648" in source
    assert "kPvCublasWorkspaceBytes = 2686976" in source
    assert "kOutputCublasWorkspaceBytes = 3244032" in source
    assert "ex2.approx.f32" in source
    assert "div.full.f32" in source
    assert "CUBLAS_COMPUTE_32F" in source
    assert "CUBLAS_GEMM_DEFAULT" in source
    assert "cublasSetMathMode(plugin->cublas_handle_, CUBLAS_DEFAULT_MATH)" in source
    assert source.count("configure_cublas(") == 5
    assert source.count("cublasSetStream(handle, stream)") == 1
    assert "fields != nullptr && fields->nbFields != 0" in source
    assert '"${CMAKE_CURRENT_LIST_DIR}/trt_plugins/siglip_attention_residual_plugin.cu"' in cmake


def test_siglip_attention_residual_selection_accepts_depth1_and_full_geometry() -> None:
    fake_ops = _fake_graph_ops_module()
    hidden = _FakeTensor((1, 256, 1152), fake_ops.trt.bfloat16)
    full = get_profile("pi05_droid").vision
    depth1 = replace(full, depth=1)

    assert prefill_builder_module._uses_siglip_attention_residual_plugin(
        hidden, full, trt=fake_ops.trt
    )
    assert prefill_builder_module._uses_siglip_attention_residual_plugin(
        hidden, depth1, trt=fake_ops.trt
    )
    assert not prefill_builder_module._uses_siglip_attention_residual_plugin(
        _FakeTensor(hidden.shape, fake_ops.trt.float16), full, trt=fake_ops.trt
    )
    assert not prefill_builder_module._uses_siglip_attention_residual_plugin(
        _FakeTensor((1, 255, 1152), fake_ops.trt.bfloat16), full, trt=fake_ops.trt
    )
    assert not prefill_builder_module._uses_siglip_attention_residual_plugin(
        hidden, _tiny_profile().vision, trt=fake_ops.trt
    )


def test_siglip_fallback_scalar_einsum_scaling_precedes_qk() -> None:
    fake_ops = _fake_graph_ops_module()
    network = _FakeNetwork(fake_ops.trt)
    q = _FakeTensor((1, 16, 256, 72), fake_ops.trt.bfloat16)

    output = prefill_builder_module._softmax_attention(
        network,
        q,
        _FakeTensor(q.shape, q.dtype),
        _FakeTensor(q.shape, q.dtype),
        None,
        head_dim=72,
        fp32_logits=False,
        scale_with_division=True,
        use_einsum=True,
        ops=fake_ops,
        trt=fake_ops.trt,
    )

    assert network.events == [
        ("einsum", "bhqd,z->bhqd"),
        ("einsum", "bhqd,bhkd->bhqk"),
        ("softmax",),
        ("einsum", "bhqk,bhkd->bhqd"),
    ]
    assert output.shape == (1, 256, 1152)
    assert output.dtype == fake_ops.trt.bfloat16


def test_siglip_layer_norm_plugin_topology_covers_every_camera_block_and_postnorm(
    monkeypatch,
) -> None:
    profile = _tiny_profile()
    weights = _tiny_weights(profile)
    fake_ops = _fake_graph_ops_module()
    network = _FakeNetwork(fake_ops.trt)
    observed: list[tuple[str, float]] = []

    def record_plugin(
        network,
        tensor,
        weight,
        bias,
        *,
        epsilon,
        name,
        ops,
        trt,
    ):
        del network, weight, bias, ops, trt
        observed.append((name, epsilon))
        return _FakeTensor(tensor.shape, tensor.dtype)

    monkeypatch.setattr(
        prefill_builder_module,
        "_apply_siglip_layer_norm_plugin",
        record_plugin,
    )
    output = prefill_builder_module._vision_encoder(
        network,
        _FakeTensor((3, 3, 4, 4), fake_ops.trt.float32),
        weights,
        profile,
        activation_dtype=fake_ops.trt.bfloat16,
        ops=fake_ops,
        trt=fake_ops.trt,
    )

    expected_names = []
    for camera in range(profile.vision.num_image_slots):
        for layer in range(profile.vision.depth):
            expected_names.extend(
                [
                    f"openpi/siglip/camera_{camera}/layer_{layer:02d}/norm1",
                    f"openpi/siglip/camera_{camera}/layer_{layer:02d}/norm2",
                ]
            )
        expected_names.append(f"openpi/siglip/camera_{camera}/post_norm")
    assert [name for name, _epsilon in observed] == expected_names
    assert all(epsilon == 1.0e-6 for _name, epsilon in observed)
    assert len(set(expected_names)) == len(expected_names)
    assert output.shape == (1, 12, 8)
    assert output.dtype == fake_ops.trt.bfloat16


def test_siglip_attention_residual_topology_replaces_norm1_through_first_residual(
    monkeypatch,
) -> None:
    profile = _tiny_profile()
    weights = _tiny_weights(profile)
    fake_ops = _fake_graph_ops_module()
    network = _FakeNetwork(fake_ops.trt)
    attention_names: list[str] = []
    norm_names: list[str] = []

    monkeypatch.setattr(
        prefill_builder_module,
        "_uses_siglip_attention_residual_plugin",
        lambda hidden, cfg, *, trt: True,
    )

    def record_attention(network, hidden, *args, name, ops, trt):
        del network, args, ops, trt
        attention_names.append(name)
        return _FakeTensor(hidden.shape, hidden.dtype)

    def record_norm(network, tensor, weight, bias, *, name, **kwargs):
        del network, weight, bias, kwargs
        norm_names.append(name)
        return _FakeTensor(tensor.shape, tensor.dtype)

    monkeypatch.setattr(
        prefill_builder_module,
        "_apply_siglip_attention_residual_plugin",
        record_attention,
    )
    monkeypatch.setattr(
        prefill_builder_module,
        "_apply_siglip_layer_norm_plugin",
        record_norm,
    )
    output = prefill_builder_module._vision_encoder(
        network,
        _FakeTensor((3, 3, 4, 4), fake_ops.trt.float32),
        weights,
        profile,
        activation_dtype=fake_ops.trt.bfloat16,
        ops=fake_ops,
        trt=fake_ops.trt,
    )

    expected_attention_names = [
        f"openpi/siglip/camera_{camera}/layer_{layer:02d}/attention_residual"
        for camera in range(profile.vision.num_image_slots)
        for layer in range(profile.vision.depth)
    ]
    expected_norm_names = []
    for camera in range(profile.vision.num_image_slots):
        expected_norm_names.extend(
            f"openpi/siglip/camera_{camera}/layer_{layer:02d}/norm2"
            for layer in range(profile.vision.depth)
        )
        expected_norm_names.append(f"openpi/siglip/camera_{camera}/post_norm")

    assert attention_names == expected_attention_names
    assert norm_names == expected_norm_names
    assert output.shape == (1, 12, 8)
    assert output.dtype == fake_ops.trt.bfloat16


def test_tiny_graph_build_has_exact_io_compact_cache_and_upstream_softmax_dtypes(
    monkeypatch,
) -> None:
    profile = _tiny_profile()
    weights = _tiny_weights(profile)
    fake_ops = _fake_graph_ops_module()
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.openpi.graph_ops",
        fake_ops,
    )
    # ``from . import graph_ops`` may reuse the package attribute after an
    # earlier real TensorRT test imported the module. Replace both import
    # caches so this structural test remains order-independent.
    monkeypatch.setattr(openpi_package, "graph_ops", fake_ops, raising=False)
    monkeypatch.setattr(
        prefill_builder_module,
        "_apply_siglip_layer_norm_plugin",
        lambda network, tensor, weight, bias, **kwargs: _FakeTensor(tensor.shape, tensor.dtype),
    )

    plan = build_prefill_engine(weights, profile, precision="bf16", workspace_bytes=1234)
    assert plan == b"fake-prefill-plan"
    assert fake_ops.builder_context_kwargs == {"verbose": False, "workspace_bytes": 1234}
    assert fake_ops.last_builder_config.cleared_flags == ["tf32"]
    network = fake_ops.last_network
    assert network.plugin_layers == []
    assert [(tensor.name, tensor.shape, tensor.dtype) for tensor in network.inputs] == [
        ("pixel_values", (3, 3, 4, 4), "float32"),
        ("token_ids", (1, 5), "int32"),
        ("prefix_mask", (1, 17), "bool"),
        ("prefix_position_ids", (1, 17), "int32"),
    ]
    assert [(tensor.name, tensor.shape, tensor.dtype) for tensor in network.outputs] == [
        ("vision_tokens", (1, 3, 4, 8), "bfloat16"),
        *[
            (prefix_cache_output_name(kind, layer), (1, 17, 1, 4), "bfloat16")
            for layer in range(2)
            for kind in ("k", "v")
        ],
    ]
    # Each per-camera SigLIP layer uses TensorRT's native BF16 softmax. The
    # tiny shape does not activate the fixed production materialization
    # plugin. Gemma retains one FP32-logit softmax per prefix layer.
    assert network.softmax_input_dtypes == ["bfloat16"] * 6 + ["float32"] * 2
    # The three camera encoders remain separate M=tokens graphs, matching the
    # upstream per-image loop and preserving its contraction tactics.
    assert network.einsum_equations.count("bsi,iqhd->bsqhd") == 6
    assert network.einsum_equations.count("bsi,io->bso") == 21
    assert network.einsum_equations.count("bhqd,bhkd->bhqk") == 6
    assert network.einsum_equations.count("bhqk,bhkd->bhqd") == 6
    assert network.einsum_equations.count("bhqd,z->bhqd") == 6
    assert network.einsum_equations.count("bhqk,z->bhqk") == 0


def test_bf16_gelu_preserves_every_jax_rounding_boundary() -> None:
    fake_ops = _fake_graph_ops_module()
    network = _FakeNetwork(fake_ops.trt)
    tensor = _FakeTensor((1, 4, 6), fake_ops.trt.bfloat16)

    output = _gelu_tanh(
        network,
        tensor,
        ops=fake_ops,
        trt=fake_ops.trt,
    )

    assert output.shape == tensor.shape
    assert output.dtype == fake_ops.trt.bfloat16
    # Nine JAX BF16 pointwise results plus the final return are explicitly
    # quantized from FP32. This prevents TensorRT from carrying extended
    # precision through its fused GELU kernel.
    assert fake_ops.cast_transitions.count((fake_ops.trt.float32, fake_ops.trt.bfloat16)) == 10
