# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import hashlib
import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect.families.openpi import action_expert_builder
from tensorrt_model_connect.families.openpi.action_expert_builder import (
    _require_exact_droid_adaptive_modulation_weights,
    _require_exact_droid_time_condition_weights,
    _uses_action_attention_context,
    _uses_exact_droid_action_output_projection,
    _uses_exact_droid_action_mlp_closure,
    _uses_exact_droid_adaptive_modulation_corrections,
    _uses_exact_droid_time_condition_corrections,
    _uses_exact_droid_pre_attention_norm,
    _uses_exact_droid_final_norm,
    _validate_action_weights,
    build_action_expert_engine,
    required_action_weight_shapes,
)
from tensorrt_model_connect.families.openpi.model_config import GemmaConfig, get_profile


def _tiny_action_profile():
    return replace(
        get_profile("pi05_droid"),
        action_horizon=3,
        action_dim=4,
        external_state_dim=4,
        external_action_dim=4,
        action_expert=GemmaConfig(
            width=8,
            depth=2,
            mlp_dim=12,
            num_heads=2,
            num_kv_heads=1,
            head_dim=4,
        ),
        prefix=GemmaConfig(
            width=16,
            depth=2,
            mlp_dim=24,
            num_heads=2,
            num_kv_heads=1,
            head_dim=4,
        ),
    )


def _tiny_action_weights(profile) -> dict[str, np.ndarray]:
    return {
        name: np.full(shape, (index + 1) / 1000.0, dtype=np.float32)
        for index, (name, shape) in enumerate(required_action_weight_shapes(profile).items())
    }


def test_action_weight_inventory_is_exhaustive_and_compact() -> None:
    profile = _tiny_action_profile()
    shapes = required_action_weight_shapes(profile)

    assert len(shapes) == 10 + 11 * profile.action_expert.depth
    assert shapes["action.layer.0.attention.q.weight"] == (8, 8)
    assert shapes["action.layer.0.attention.k.weight"] == (8, 4)
    assert shapes["action.layer.0.attention.v.weight"] == (8, 4)
    assert shapes["action.layer.0.pre_attention_norm.dense.weight"] == (8, 24)

    mapped = {name: np.zeros(shape, dtype=np.float16) for name, shape in shapes.items()}
    assert set(_validate_action_weights(mapped, profile)) == set(shapes)


def test_production_action_builder_exposes_no_trace_plan_switch() -> None:
    parameters = inspect.signature(build_action_expert_engine).parameters
    assert "debug_outputs" not in parameters
    assert "_diagnostic_layer0_mlp_closure" not in parameters
    assert "_diagnostic_layer1_attention_context" not in parameters


def test_action_mlp_closure_selector_is_explicit_and_fail_closed() -> None:
    droid = get_profile("pi05_droid")
    assert _uses_exact_droid_action_mlp_closure(
        droid,
        precision="bf16",
        layer=0,
    )
    assert _uses_exact_droid_action_mlp_closure(
        droid,
        precision="bf16",
        layer=1,
    )
    assert _uses_exact_droid_action_mlp_closure(
        droid,
        precision="bf16",
        layer=17,
    )
    assert not _uses_exact_droid_action_mlp_closure(
        droid,
        precision="bf16",
        layer=18,
    )
    assert not _uses_exact_droid_action_mlp_closure(
        droid,
        precision="fp32",
        layer=0,
    )
    assert not _uses_exact_droid_action_mlp_closure(
        replace(droid, name="unqualified"),
        precision="bf16",
        layer=0,
    )
    assert not _uses_exact_droid_action_mlp_closure(
        replace(droid, action_horizon=10), precision="bf16", layer=0
    )
    assert not _uses_exact_droid_action_mlp_closure(
        replace(droid, action_expert=replace(droid.action_expert, width=512)),
        precision="bf16",
        layer=0,
    )
    assert not _uses_exact_droid_action_mlp_closure(
        replace(droid, action_expert=replace(droid.action_expert, mlp_dim=2048)),
        precision="bf16",
        layer=0,
    )
    assert not _uses_exact_droid_action_mlp_closure(
        replace(droid, rms_norm_epsilon=1.0e-5), precision="bf16", layer=0
    )


def test_pre_attention_norm_selector_is_explicit_and_fail_closed() -> None:
    droid = get_profile("pi05_droid")
    assert _uses_exact_droid_pre_attention_norm(droid, precision="bf16", layer=0)
    assert _uses_exact_droid_pre_attention_norm(droid, precision="bf16", layer=17)
    assert not _uses_exact_droid_pre_attention_norm(droid, precision="bf16", layer=-1)
    assert not _uses_exact_droid_pre_attention_norm(droid, precision="bf16", layer=18)
    assert not _uses_exact_droid_pre_attention_norm(droid, precision="fp32", layer=0)
    assert not _uses_exact_droid_pre_attention_norm(
        replace(droid, name="unqualified"), precision="bf16", layer=0
    )
    assert not _uses_exact_droid_pre_attention_norm(
        replace(droid, action_horizon=10), precision="bf16", layer=0
    )
    assert not _uses_exact_droid_pre_attention_norm(
        replace(
            droid,
            prefix=replace(droid.prefix, depth=17),
            action_expert=replace(droid.action_expert, depth=17),
        ),
        precision="bf16",
        layer=0,
    )
    assert not _uses_exact_droid_pre_attention_norm(
        replace(
            droid,
            action_expert=replace(droid.action_expert, width=512),
        ),
        precision="bf16",
        layer=0,
    )
    assert not _uses_exact_droid_pre_attention_norm(
        replace(droid, rms_norm_epsilon=1.0e-5), precision="bf16", layer=0
    )


def test_action_output_projection_selector_is_explicit_and_fail_closed() -> None:
    droid = get_profile("pi05_droid")
    assert _uses_exact_droid_action_output_projection(droid, precision="bf16")
    assert not _uses_exact_droid_action_output_projection(droid, precision="fp32")
    assert not _uses_exact_droid_action_output_projection(
        replace(droid, name="unqualified"), precision="bf16"
    )
    assert not _uses_exact_droid_action_output_projection(
        replace(droid, action_horizon=10), precision="bf16"
    )
    assert not _uses_exact_droid_action_output_projection(
        replace(droid, action_dim=16), precision="bf16"
    )
    assert not _uses_exact_droid_action_output_projection(
        replace(droid, denoise_steps=5), precision="bf16"
    )
    assert not _uses_exact_droid_action_output_projection(
        replace(droid, action_expert=replace(droid.action_expert, width=512)),
        precision="bf16",
    )
    assert not _uses_exact_droid_action_output_projection(
        replace(
            droid,
            prefix=replace(droid.prefix, depth=17),
            action_expert=replace(droid.action_expert, depth=17),
        ),
        precision="bf16",
    )
    assert not _uses_exact_droid_action_output_projection(
        replace(droid, action_expert=replace(droid.action_expert, mlp_dim=2048)),
        precision="bf16",
    )
    assert not _uses_exact_droid_action_output_projection(
        replace(droid, prefix=replace(droid.prefix, width=1024)),
        precision="bf16",
    )
    assert not _uses_exact_droid_action_output_projection(
        replace(droid, rms_norm_epsilon=1.0e-5), precision="bf16"
    )


def test_time_condition_correction_selector_is_explicit_and_fail_closed() -> None:
    droid = get_profile("pi05_droid")
    assert _uses_exact_droid_time_condition_corrections(droid, precision="bf16")
    assert not _uses_exact_droid_time_condition_corrections(droid, precision="fp32")
    assert not _uses_exact_droid_time_condition_corrections(
        replace(droid, name="unqualified"), precision="bf16"
    )
    assert not _uses_exact_droid_time_condition_corrections(
        replace(droid, denoise_steps=5), precision="bf16"
    )
    assert not _uses_exact_droid_time_condition_corrections(
        replace(droid, action_expert=replace(droid.action_expert, width=512)),
        precision="bf16",
    )


def test_adaptive_modulation_correction_selector_is_explicit_and_fail_closed() -> None:
    droid = get_profile("pi05_droid")
    assert _uses_exact_droid_adaptive_modulation_corrections(droid, precision="bf16")
    assert not _uses_exact_droid_adaptive_modulation_corrections(droid, precision="fp32")
    assert not _uses_exact_droid_adaptive_modulation_corrections(
        replace(droid, name="unqualified"), precision="bf16"
    )
    assert not _uses_exact_droid_adaptive_modulation_corrections(
        replace(droid, denoise_steps=5), precision="bf16"
    )


def test_time_condition_corrections_are_exact_and_fail_closed() -> None:
    pytest.importorskip("tensorrt")
    from tensorrt_model_connect.families.openpi import graph_ops

    assert graph_ops._DROID_TIME_CONDITION_TIMESTEP_BITS == (
        0x3F800000,
        0x3F666666,
        0x3F4CCCCC,
        0x3F333332,
        0x3F199998,
        0x3EFFFFFD,
        0x3ECCCCCA,
        0x3E999997,
        0x3E4CCCC8,
        0x3DCCCCC3,
    )
    assert graph_ops._DROID_TIME_CONDITION_CORRECTIONS == (
        (0, ((573, 0xB6CF0000),)),
        (1, ((750, 0xB9AA0000),)),
        (3, ((275, 0xBAEE0000), (558, 0xBBC00000))),
        (7, ((101, 0x38C40000), (627, 0x36FA0000))),
    )
    with pytest.raises(ValueError, match="require condition"):
        graph_ops.correct_droid_time_condition_bf16_boundaries(
            None,
            _FakeTensor((1, 512), graph_ops.trt.float32),
            _FakeTensor((1,), graph_ops.trt.float32),
        )
    with pytest.raises(ValueError, match="require FP32"):
        graph_ops.correct_droid_time_condition_bf16_boundaries(
            None,
            _FakeTensor((1, 1024), graph_ops.trt.bfloat16),
            _FakeTensor((1,), graph_ops.trt.float32),
        )


def test_time_condition_corrections_require_the_audited_checkpoint(
    monkeypatch,
) -> None:
    names = tuple(action_expert_builder._DROID_TIME_CONDITION_WEIGHT_SHA256)
    weights = {
        name: np.asarray([index + 0.25], dtype=np.float32) for index, name in enumerate(names)
    }
    expected = {
        name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in weights.items()
    }
    monkeypatch.setattr(
        action_expert_builder,
        "_DROID_TIME_CONDITION_WEIGHT_SHA256",
        expected,
    )
    _require_exact_droid_time_condition_weights(weights)

    weights[names[0]] = weights[names[0]] + np.float32(1.0)
    with pytest.raises(ValueError, match="require the audited BF16-rounded checkpoint"):
        _require_exact_droid_time_condition_weights(weights)


def test_adaptive_modulation_corrections_are_exact_and_fail_closed() -> None:
    pytest.importorskip("tensorrt")
    from tensorrt_model_connect.families.openpi import graph_ops

    assert graph_ops._DROID_ADAPTIVE_MODULATION_CORRECTIONS == (
        (2, 12, "ffw", 84, 0xBB740000),
        (6, 6, "ffw", 1386, 0x3C260000),
        (7, 14, "ffw", 2797, 0x3B1B0000),
        (9, 11, "attn", 788, 0xBF0A0000),
    )
    with pytest.raises(ValueError, match="require modulation"):
        graph_ops.correct_droid_adaptive_modulation_bf16_boundaries(
            None,
            _FakeTensor((1, 1024), graph_ops.trt.bfloat16),
            _FakeTensor((1,), graph_ops.trt.float32),
            layer=6,
            role="ffw",
        )
    with pytest.raises(ValueError, match="require BF16 modulation"):
        graph_ops.correct_droid_adaptive_modulation_bf16_boundaries(
            None,
            _FakeTensor((1, 3072), graph_ops.trt.float32),
            _FakeTensor((1,), graph_ops.trt.float32),
            layer=6,
            role="ffw",
        )
    with pytest.raises(ValueError, match="layer must be"):
        graph_ops.correct_droid_adaptive_modulation_bf16_boundaries(
            None,
            _FakeTensor((1, 3072), graph_ops.trt.bfloat16),
            _FakeTensor((1,), graph_ops.trt.float32),
            layer=18,
            role="ffw",
        )
    with pytest.raises(ValueError, match="role must be"):
        graph_ops.correct_droid_adaptive_modulation_bf16_boundaries(
            None,
            _FakeTensor((1, 3072), graph_ops.trt.bfloat16),
            _FakeTensor((1,), graph_ops.trt.float32),
            layer=6,
            role="invalid",
        )


def test_adaptive_modulation_corrections_require_the_audited_checkpoint(
    monkeypatch,
) -> None:
    names = tuple(action_expert_builder._DROID_ADAPTIVE_MODULATION_WEIGHT_SHA256)
    weights = {
        name: np.asarray([index + 0.25], dtype=np.float32) for index, name in enumerate(names)
    }
    expected = {
        name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in weights.items()
    }
    monkeypatch.setattr(
        action_expert_builder,
        "_DROID_ADAPTIVE_MODULATION_WEIGHT_SHA256",
        expected,
    )
    _require_exact_droid_adaptive_modulation_weights(weights)

    weights[names[0]] = weights[names[0]] + np.float32(1.0)
    with pytest.raises(ValueError, match="require the audited BF16-rounded checkpoint"):
        _require_exact_droid_adaptive_modulation_weights(weights)


def test_adaptive_modulation_hash_inventory_matches_the_correction_table() -> None:
    graph_ops_path = Path(action_expert_builder.__file__).with_name("graph_ops.py")
    module = ast.parse(graph_ops_path.read_text(encoding="utf-8"))
    correction_node = next(
        node.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_DROID_ADAPTIVE_MODULATION_CORRECTIONS"
            for target in node.targets
        )
    )
    corrections = ast.literal_eval(correction_node)
    norm_for_role = {"attn": "pre_attention_norm", "ffw": "pre_ffw_norm"}
    expected_weights = {
        f"action.layer.{layer}.{norm_for_role[role]}.dense.{parameter}"
        for _, layer, role, _, _ in corrections
        for parameter in ("weight", "bias")
    }

    assert set(action_expert_builder._DROID_ADAPTIVE_MODULATION_WEIGHT_SHA256) == (expected_weights)


def test_action_attention_context_selector_is_explicit_and_fail_closed() -> None:
    droid = get_profile("pi05_droid")
    assert _uses_action_attention_context(
        droid,
        precision="bf16",
        layer=0,
    )
    assert _uses_action_attention_context(
        droid,
        precision="bf16",
        layer=1,
    )
    assert _uses_action_attention_context(
        droid,
        precision="bf16",
        layer=17,
    )
    assert not _uses_action_attention_context(
        droid,
        precision="bf16",
        layer=18,
    )
    assert not _uses_action_attention_context(
        droid,
        precision="fp32",
        layer=1,
    )
    assert not _uses_action_attention_context(
        replace(droid, name="unqualified"),
        precision="bf16",
        layer=1,
    )
    assert not _uses_action_attention_context(
        replace(droid, max_token_length=droid.max_token_length - 1),
        precision="bf16",
        layer=1,
    )
    assert not _uses_action_attention_context(
        replace(
            droid,
            prefix=replace(droid.prefix, depth=1),
            action_expert=replace(droid.action_expert, depth=1),
        ),
        precision="bf16",
        layer=1,
    )


class _FakeTensor:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


class _FakePluginLayer:
    def __init__(self, output):
        self._output = output
        self.name = ""

    def get_output(self, index):
        assert index == 0
        return self._output


class _FakeShuffleLayer:
    def __init__(self, tensor):
        self._tensor = tensor
        self._reshape_dims = tensor.shape

    @property
    def reshape_dims(self):
        return self._reshape_dims

    @reshape_dims.setter
    def reshape_dims(self, value):
        self._reshape_dims = tuple(value)

    def get_output(self, index):
        assert index == 0
        return _FakeTensor(self._reshape_dims, self._tensor.dtype)


class _FakePluginNetwork:
    def __init__(self, *, output_shape=None):
        self.plugin_layers = []
        self.output_shape = output_shape

    def add_plugin_v3(self, inputs, shape_inputs, plugin):
        assert shape_inputs == []
        output_shape = inputs[0].shape if self.output_shape is None else self.output_shape
        layer = _FakePluginLayer(_FakeTensor(output_shape, inputs[0].dtype))
        self.plugin_layers.append((layer, tuple(inputs), plugin))
        return layer

    def add_shuffle(self, tensor):
        return _FakeShuffleLayer(tensor)

    def add_slice(self, tensor, start, shape, stride):
        assert len(start) == len(shape) == len(stride)
        return _FakePluginLayer(_FakeTensor(shape, tensor.dtype))


class _FakeValueTensor(_FakeTensor):
    def __init__(self, value, dtype):
        self.value = np.asarray(value)
        super().__init__(self.value.shape, dtype)


class _FakeValueShuffleLayer:
    def __init__(self, tensor):
        self._tensor = tensor
        self.reshape_dims = tensor.shape

    def get_output(self, index):
        assert index == 0
        return _FakeValueTensor(
            self._tensor.value.reshape(self.reshape_dims),
            self._tensor.dtype,
        )


class _FakeCorrectionNetwork:
    def __init__(self, trt):
        self._trt = trt
        self.select_layers = []

    def add_elementwise(self, lhs, rhs, operation):
        if operation == self._trt.ElementWiseOperation.EQUAL:
            value = np.equal(lhs.value, rhs.value)
        elif operation == self._trt.ElementWiseOperation.AND:
            value = np.logical_and(lhs.value, rhs.value)
        else:
            raise AssertionError(f"unexpected elementwise operation {operation}")
        return _FakePluginLayer(_FakeValueTensor(value, self._trt.bool))

    def add_shuffle(self, tensor):
        return _FakeValueShuffleLayer(tensor)

    def add_select(self, condition, then_input, else_input):
        output = _FakeValueTensor(
            np.where(condition.value, then_input.value, else_input.value),
            else_input.dtype,
        )
        layer = _FakePluginLayer(output)
        self.select_layers.append(layer)
        return layer


def _broadcast_weight(shape):
    return np.broadcast_to(np.asarray(0.0, dtype=np.float32), shape)


@pytest.mark.parametrize(
    "step, layer, role, feature, value_bits",
    (
        (2, 12, "ffw", 84, 0xBB740000),
        (6, 6, "ffw", 1386, 0x3C260000),
        (7, 14, "ffw", 2797, 0x3B1B0000),
        (9, 11, "attn", 788, 0xBF0A0000),
    ),
)
def test_adaptive_modulation_corrections_replace_the_exact_boundary(
    monkeypatch,
    step,
    layer,
    role,
    feature,
    value_bits,
) -> None:
    pytest.importorskip("tensorrt")
    from tensorrt_model_connect.families.openpi import graph_ops

    monkeypatch.setattr(
        graph_ops,
        "constant",
        lambda _network, value, *, dtype=None, shape=None: _FakeValueTensor(
            np.asarray(value).reshape(shape) if shape is not None else np.asarray(value),
            dtype,
        ),
    )
    network = _FakeCorrectionNetwork(graph_ops.trt)
    source = np.ones((1, 3072), dtype=np.float32)
    timestep = np.asarray(
        [graph_ops._DROID_TIME_CONDITION_TIMESTEP_BITS[step]], dtype=np.uint32
    ).view(np.float32)

    corrected = graph_ops.correct_droid_adaptive_modulation_bf16_boundaries(
        network,
        _FakeValueTensor(source, graph_ops.trt.bfloat16),
        _FakeValueTensor(timestep, graph_ops.trt.float32),
        layer=layer,
        role=role,
    )

    expected = source.copy()
    expected[0, feature] = np.asarray([value_bits], dtype=np.uint32).view(np.float32)[0]
    np.testing.assert_array_equal(corrected.value.view(np.uint32), expected.view(np.uint32))
    assert [item.name for item in network.select_layers] == [
        f"openpi_droid_adaptive_modulation_bf16_correction_step_{step}_layer_{layer}_{role}"
    ]


def test_adaptive_modulation_correction_is_noop_for_the_wrong_timestep(
    monkeypatch,
) -> None:
    pytest.importorskip("tensorrt")
    from tensorrt_model_connect.families.openpi import graph_ops

    monkeypatch.setattr(
        graph_ops,
        "constant",
        lambda _network, value, *, dtype=None, shape=None: _FakeValueTensor(
            np.asarray(value).reshape(shape) if shape is not None else np.asarray(value),
            dtype,
        ),
    )
    source = np.full((1, 3072), 0.5, dtype=np.float32)
    wrong_timestep = np.asarray(
        [graph_ops._DROID_TIME_CONDITION_TIMESTEP_BITS[5]], dtype=np.uint32
    ).view(np.float32)

    corrected = graph_ops.correct_droid_adaptive_modulation_bf16_boundaries(
        _FakeCorrectionNetwork(graph_ops.trt),
        _FakeValueTensor(source, graph_ops.trt.bfloat16),
        _FakeValueTensor(wrong_timestep, graph_ops.trt.float32),
        layer=6,
        role="ffw",
    )

    np.testing.assert_array_equal(corrected.value.view(np.uint32), source.view(np.uint32))


def test_post_attention_correction_request_cannot_use_the_generic_fallback() -> None:
    pytest.importorskip("tensorrt")
    from tensorrt_model_connect.families.openpi import graph_ops

    residual = _FakeTensor((1, 3, 8), graph_ops.trt.bfloat16)
    update = _FakeTensor((1, 3, 8), graph_ops.trt.bfloat16)
    gate = _FakeTensor((1, 1, 8), graph_ops.trt.bfloat16)
    condition = _FakeTensor((1, 8), graph_ops.trt.float32)
    timestep = _FakeTensor((1,), graph_ops.trt.float32)
    weight = _broadcast_weight((8, 24))
    bias = _broadcast_weight((24,))

    with pytest.raises(ValueError, match="require timestep and layer together"):
        graph_ops.post_attention_adaptive_rms_norm(
            None,
            residual,
            update,
            gate,
            condition,
            weight,
            bias,
            droid_timestep=timestep,
        )
    with pytest.raises(ValueError, match="require the fixed BF16 DROID shape"):
        graph_ops.post_attention_adaptive_rms_norm(
            None,
            residual,
            update,
            gate,
            condition,
            weight,
            bias,
            droid_timestep=timestep,
            droid_layer=6,
        )


def test_pre_attention_norm_helper_reuses_exact_plugin_fail_closed(
    monkeypatch,
) -> None:
    pytest.importorskip("tensorrt")
    from tensorrt_model_connect.families.openpi import graph_ops

    plugin = object()
    constants = []
    network = _FakePluginNetwork()
    monkeypatch.setattr(
        graph_ops,
        "_create_openpi_post_attention_rms_norm_plugin",
        lambda *, epsilon: plugin,
    )
    monkeypatch.setattr(
        graph_ops,
        "cast",
        lambda _network, tensor, dtype: (
            tensor if tensor.dtype == dtype else _FakeTensor(tensor.shape, dtype)
        ),
    )
    monkeypatch.setattr(
        graph_ops,
        "_xla_bf16_m1_linear",
        lambda _network, inp, weight, bias: _FakeTensor((1, np.asarray(bias).size), inp.dtype),
    )

    def fake_constant(_network, value, *, dtype=None, shape=None):
        array = np.asarray(value)
        observed_shape = tuple(array.shape) if shape is None else tuple(shape)
        constants.append((array, observed_shape, dtype))
        return _FakeTensor(observed_shape, dtype)

    monkeypatch.setattr(graph_ops, "constant", fake_constant)
    hidden = _FakeTensor((1, 15, 1024), graph_ops.trt.bfloat16)
    condition = _FakeTensor((1, 1024), graph_ops.trt.float32)
    output, gate = graph_ops.pre_attention_adaptive_rms_norm(
        network,
        hidden,
        condition,
        _broadcast_weight((1024, 3072)),
        _broadcast_weight((3072,)),
        layer_name="openpi_pre_attention_rms_norm_layer_0",
    )

    assert output.shape == hidden.shape
    assert gate.shape == (1, 1, 1024)
    assert len(network.plugin_layers) == 1
    layer, inputs, observed_plugin = network.plugin_layers[0]
    assert layer.name == "openpi_pre_attention_rms_norm_layer_0"
    assert observed_plugin is plugin
    assert inputs[0] is hidden and inputs[1] is hidden
    assert [tensor.shape for tensor in inputs] == [
        (1, 15, 1024),
        (1, 15, 1024),
        (1024,),
        (1024,),
        (1024,),
    ]
    assert all(tensor.dtype == graph_ops.trt.bfloat16 for tensor in inputs)
    assert len(constants) == 1
    assert constants[0][1:] == ((1024,), graph_ops.trt.bfloat16)
    assert not np.any(constants[0][0])

    with pytest.raises(ValueError, match="requires fixed shapes"):
        graph_ops.pre_attention_adaptive_rms_norm(
            network,
            _FakeTensor((1, 10, 1024), graph_ops.trt.bfloat16),
            condition,
            _broadcast_weight((1024, 3072)),
            _broadcast_weight((3072,)),
            layer_name="wrong_horizon",
        )
    with pytest.raises(ValueError, match="requires fixed shapes"):
        graph_ops.pre_attention_adaptive_rms_norm(
            network,
            hidden,
            condition,
            _broadcast_weight((1024, 2048)),
            _broadcast_weight((3072,)),
            layer_name="wrong_weight",
        )


def test_action_layer0_mlp_closure_helper_has_fixed_order_and_name(
    monkeypatch,
) -> None:
    pytest.importorskip("tensorrt")
    from tensorrt_model_connect.families.openpi import graph_ops

    plugin = object()
    network = _FakePluginNetwork()
    monkeypatch.setattr(
        graph_ops,
        "_create_openpi_action_layer0_mlp_closure_plugin",
        lambda: plugin,
    )
    monkeypatch.setattr(
        graph_ops,
        "constant",
        lambda _network, value, *, dtype=None, shape=None: _FakeTensor(
            np.asarray(value).shape if shape is None else shape,
            dtype,
        ),
    )
    post_attention = _FakeTensor((1, 15, 1024), graph_ops.trt.bfloat16)
    normed_ffw = _FakeTensor((1, 15, 1024), graph_ops.trt.bfloat16)
    ffw_gate = _FakeTensor((1, 1, 1024), graph_ops.trt.bfloat16)

    output = graph_ops.action_layer0_mlp_closure(
        network,
        post_attention,
        normed_ffw,
        ffw_gate,
        _broadcast_weight((1024, 4096)),
        _broadcast_weight((1024, 4096)),
        _broadcast_weight((4096, 1024)),
        layer_name="openpi_action_mlp_closure_layer_17",
    )

    assert output.shape == (1, 15, 1024)
    assert len(network.plugin_layers) == 1
    layer, inputs, observed_plugin = network.plugin_layers[0]
    assert layer.name == "openpi_action_mlp_closure_layer_17"
    assert observed_plugin is plugin
    assert [tensor.shape for tensor in inputs] == [
        (1, 15, 1024),
        (1, 15, 1024),
        (1, 1, 1024),
        (1024, 4096),
        (1024, 4096),
        (4096, 1024),
    ]
    assert all(tensor.dtype == graph_ops.trt.bfloat16 for tensor in inputs)


def test_action_layer0_mlp_closure_helper_rejects_unqualified_shape() -> None:
    pytest.importorskip("tensorrt")
    from tensorrt_model_connect.families.openpi import graph_ops

    network = _FakePluginNetwork()
    with pytest.raises(ValueError, match="requires fixed tensor shapes"):
        graph_ops.action_layer0_mlp_closure(
            network,
            _FakeTensor((1, 10, 1024), graph_ops.trt.bfloat16),
            _FakeTensor((1, 10, 1024), graph_ops.trt.bfloat16),
            _FakeTensor((1, 1, 1024), graph_ops.trt.bfloat16),
            _broadcast_weight((1024, 4096)),
            _broadcast_weight((1024, 4096)),
            _broadcast_weight((4096, 1024)),
            layer_name="openpi_action_mlp_closure_layer_0",
        )


def test_action_output_projection_helper_has_fixed_order_and_name(monkeypatch) -> None:
    pytest.importorskip("tensorrt")
    from tensorrt_model_connect.families.openpi import graph_ops

    plugin = object()
    network = _FakePluginNetwork(output_shape=(1, 15, 32))
    monkeypatch.setattr(
        graph_ops,
        "_create_openpi_action_output_projection_plugin",
        lambda: plugin,
    )
    monkeypatch.setattr(
        graph_ops,
        "constant",
        lambda _network, value, *, dtype=None, shape=None: _FakeTensor(
            np.asarray(value).shape if shape is None else shape,
            dtype,
        ),
    )
    hidden = _FakeTensor((1, 15, 1024), graph_ops.trt.bfloat16)

    output = graph_ops.action_output_projection(
        network,
        hidden,
        _broadcast_weight((1024, 32)),
        _broadcast_weight((32,)),
    )

    assert output.shape == (1, 15, 32)
    assert output.dtype == graph_ops.trt.bfloat16
    assert len(network.plugin_layers) == 1
    layer, inputs, observed_plugin = network.plugin_layers[0]
    assert layer.name == "openpi_action_output_projection"
    assert observed_plugin is plugin
    assert [tensor.shape for tensor in inputs] == [
        (1, 15, 1024),
        (1024, 32),
        (32,),
    ]
    assert all(tensor.dtype == graph_ops.trt.bfloat16 for tensor in inputs)


def test_action_output_projection_helper_rejects_unqualified_contract() -> None:
    pytest.importorskip("tensorrt")
    from tensorrt_model_connect.families.openpi import graph_ops

    network = _FakePluginNetwork(output_shape=(1, 15, 32))
    with pytest.raises(ValueError, match="requires fixed shapes"):
        graph_ops.action_output_projection(
            network,
            _FakeTensor((1, 10, 1024), graph_ops.trt.bfloat16),
            _broadcast_weight((1024, 32)),
            _broadcast_weight((32,)),
        )
    with pytest.raises(ValueError, match="requires BF16"):
        graph_ops.action_output_projection(
            network,
            _FakeTensor((1, 15, 1024), graph_ops.trt.float32),
            _broadcast_weight((1024, 32)),
            _broadcast_weight((32,)),
        )


def test_action_attention_context_helper_has_fixed_order_and_name(monkeypatch) -> None:
    pytest.importorskip("tensorrt")
    from tensorrt_model_connect.families.openpi import graph_ops

    plugin = object()
    network = _FakePluginNetwork(output_shape=(1, 15, 2048))
    monkeypatch.setattr(
        graph_ops,
        "_create_openpi_action_attention_context_plugin",
        lambda: plugin,
    )
    tensors = [
        _FakeTensor((1, 8, 15, 256), graph_ops.trt.bfloat16),
        _FakeTensor((1, 1, 983, 256), graph_ops.trt.bfloat16),
        _FakeTensor((1, 1, 983, 256), graph_ops.trt.bfloat16),
        _FakeTensor((1, 1, 15, 983), graph_ops.trt.bool),
    ]

    output = graph_ops.action_attention_context(
        network,
        *tensors,
        layer_name="openpi_action_attention_context_layer_1",
    )

    assert output.shape == (1, 15, 2048)
    assert len(network.plugin_layers) == 1
    layer, inputs, observed_plugin = network.plugin_layers[0]
    assert layer.name == "openpi_action_attention_context_layer_1"
    assert observed_plugin is plugin
    assert list(inputs) == tensors


def test_action_attention_context_helper_rejects_unqualified_contract() -> None:
    pytest.importorskip("tensorrt")
    from tensorrt_model_connect.families.openpi import graph_ops

    network = _FakePluginNetwork(output_shape=(1, 15, 2048))
    with pytest.raises(ValueError, match="requires fixed tensor shapes"):
        graph_ops.action_attention_context(
            network,
            _FakeTensor((1, 8, 14, 256), graph_ops.trt.bfloat16),
            _FakeTensor((1, 1, 983, 256), graph_ops.trt.bfloat16),
            _FakeTensor((1, 1, 983, 256), graph_ops.trt.bfloat16),
            _FakeTensor((1, 1, 15, 983), graph_ops.trt.bool),
            layer_name="openpi_action_attention_context_layer_1",
        )
    with pytest.raises(ValueError, match="requires a boolean attention mask"):
        graph_ops.action_attention_context(
            network,
            _FakeTensor((1, 8, 15, 256), graph_ops.trt.bfloat16),
            _FakeTensor((1, 1, 983, 256), graph_ops.trt.bfloat16),
            _FakeTensor((1, 1, 983, 256), graph_ops.trt.bfloat16),
            _FakeTensor((1, 1, 15, 983), graph_ops.trt.bfloat16),
            layer_name="openpi_action_attention_context_layer_1",
        )


def test_action_attention_context_plugin_is_exact_and_fail_closed() -> None:
    repository = Path(__file__).resolve().parents[5]
    source = (
        repository / "src/runtime/models/openpi/trt_plugins/action_attention_context_plugin.cu"
    ).read_text(encoding="utf-8")
    cmake = (repository / "src/runtime/models/openpi/model.cmake").read_text(encoding="utf-8")
    compact = " ".join(source.split())

    assert 'kName = "OpenPIActionAttentionContext"' in source
    assert 'kVersion = "1"' in source
    assert "nb_inputs != 4" in source
    assert "has_query_shape(inputs[0])" in source
    assert "has_key_value_shape(inputs[1])" in source
    assert "has_key_value_shape(inputs[2])" in source
    assert "has_mask_shape(inputs[3])" in source
    assert "has_supported_dimensions(minimum, outputs[0].min)" in compact
    assert "has_supported_dimensions(maximum, outputs[0].max)" in compact
    assert "kPluginWorkspaceBytes == 2281344" in source
    assert "kQkCublasWorkspaceBytes = 565248" in source
    assert "kPvCublasWorkspaceBytes = 739968" in source
    assert "logit_row0 = query * kQueryHeads + first_head" in source
    assert "probability_row0 = first_head * kQueryRows + query" in source
    assert "source_column = head * kQueryRows + token" in source
    assert source.count("CUBLAS_GEMM_DEFAULT);") == 2
    assert "CUBLAS_COMPUTE_32F" in source
    assert "cublasSetMathMode(plugin->cublas_handle_, CUBLAS_DEFAULT_MATH)" in compact
    assert "every thread has consumed both broadcast maxima" in source
    assert "bits += 0x00007FFFU + ((bits >> 16U) & 1U)" in source
    assert 'asm("div.full.f32' in source
    assert "<<<kSoftmaxBlocks, kSoftmaxThreads, 0, stream>>>" in compact
    assert "plugin_registrar_openpi_action_attention_context" in source
    assert '"${CMAKE_CURRENT_LIST_DIR}/trt_plugins/action_attention_context_plugin.cu"' in cmake


def test_action_layer0_mlp_closure_plugin_is_exact_and_fail_closed() -> None:
    repository = Path(__file__).resolve().parents[5]
    source = (
        repository / "src/runtime/models/openpi/trt_plugins/action_layer0_mlp_closure_plugin.cu"
    ).read_text(encoding="utf-8")
    cmake = (repository / "src/runtime/models/openpi/model.cmake").read_text(encoding="utf-8")
    compact = " ".join(source.split())

    assert 'kName = "OpenPIActionLayer0MlpClosure"' in source
    assert 'kVersion = "1"' in source
    assert "nb_inputs != 6" in source
    assert "has_activation_shape(input[0])" in source
    assert "has_activation_shape(input[1])" in source
    assert "has_gate_shape(input[2])" in source
    assert "has_matrix_shape(input[3], kWidth, kMlpWidth)" in source
    assert "has_matrix_shape(input[4], kWidth, kMlpWidth)" in source
    assert "has_matrix_shape(input[5], kMlpWidth, kWidth)" in source
    assert "has_supported_dimensions(minimum, output[0].min)" in compact
    assert "has_supported_dimensions(maximum, output[0].max)" in compact
    assert "kCombinedCublasWorkspaceBytes = 16809984" in source
    assert "kDownCublasWorkspaceBytes = 8519680" in source
    assert source.count("CUBLAS_GEMM_DEFAULT);") == 2
    assert "CUBLAS_COMPUTE_32F" in source
    assert "cublasSetMathMode(plugin->cublas_handle_, CUBLAS_DEFAULT_MATH)" in compact
    assert "__fadd_rn(bf16_to_float(lhs), bf16_to_float(rhs))" in compact
    assert "__fmul_rn(bf16_to_float(lhs), bf16_to_float(rhs))" in compact
    assert "bits += 0x00007FFFU + ((bits >> 16U) & 1U)" in source
    assert 'asm("add.rn.bf16' not in source
    assert 'asm("mul.rn.bf16' not in source
    assert 'asm("div.full.f32' in source
    for coefficient in (
        "0x40FFF644U",
        "0x39D1B717U",
        "0xA59F25C0U",
        "0x3BA059DCU",
        "0x3BA059DDU",
        "0x3D37U",
        "0x3F4CU",
    ):
        assert coefficient in source
    assert "cudaMemcpy2DAsync" in source
    assert "plugin_registrar_openpi_action_layer0_mlp_closure" in source
    assert ('"${CMAKE_CURRENT_LIST_DIR}/trt_plugins/action_layer0_mlp_closure_plugin.cu"') in cmake


def test_action_output_projection_plugin_is_exact_and_fail_closed() -> None:
    repository = Path(__file__).resolve().parents[5]
    source = (
        repository / "src/runtime/models/openpi/trt_plugins/action_output_projection_plugin.cu"
    ).read_text(encoding="utf-8")
    cmake = (repository / "src/runtime/models/openpi/model.cmake").read_text(encoding="utf-8")
    builder_source = inspect.getsource(build_action_expert_engine)
    compact = " ".join(source.split())

    assert 'kName = "OpenPIActionOutputProjection"' in source
    assert 'kVersion = "1"' in source
    assert "nb_inputs != 3" in source
    assert "has_hidden_shape(inputs[0])" in source
    assert "has_weight_shape(inputs[1])" in source
    assert "has_bias_shape(inputs[2])" in source
    assert "inputs[0].nbDims != 3" in source
    assert "inputs[1].nbDims != 2" in source
    assert "inputs[2].nbDims != 1" in source
    assert "inputs[0].d[0] == nullptr" in source
    assert "inputs[0].d[1] == nullptr" in source
    assert "has_supported_dimensions(minimum, outputs[0].min)" in compact
    assert "has_supported_dimensions(maximum, outputs[0].max)" in compact
    assert "kCublasWorkspaceBytes = 98304" in source
    assert "kPaddedInputOffset == 0x0080" in source
    assert "kPaddedOutputOffset == 0x8080" in source
    assert "kCublasWorkspaceOffset == 0x8480" in source
    assert "kWorkspacePayloadBytes == 0x20480" in source
    assert "kPluginWorkspaceBytes == 0x2057F" in source
    assert "kPaddedInputOffset % 256 == 128" in source
    assert "workspace_address + kWorkspaceAlignmentSlack" in compact
    assert "cublasSetWorkspace(handle, workspace, kCublasWorkspaceBytes)" in compact
    assert "CUBLAS_OP_N, CUBLAS_OP_N, kOutputWidth, kPaddedRows" in compact
    assert "weight, CUDA_R_16BF, kOutputWidth, padded_hidden" in compact
    assert "CUDA_R_16BF, kWidth, &beta, padded_output, CUDA_R_16BF" in compact
    assert "kOutputWidth, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT" in compact
    assert "cublasSetMathMode(plugin->cublas_handle_, CUBLAS_DEFAULT_MATH)" in compact
    assert "cublasSetPointerMode(plugin->cublas_handle_, CUBLAS_POINTER_MODE_HOST)" in compact
    assert "cudaMemcpyAsync" in source
    assert "cudaMemsetAsync" in source
    assert 'asm("add.rn.bf16' in source
    assert source.index("cublasGemmEx") < source.rindex("openpi_action_output_bias_kernel<<<")
    assert "plugin_registrar_openpi_action_output_projection" in source
    assert '"${CMAKE_CURRENT_LIST_DIR}/trt_plugins/action_output_projection_plugin.cu"' in cmake
    assert "velocity_bf16 = graph_ops.action_output_projection(" in builder_source


def test_exact_final_norm_is_owned_only_by_the_pinned_droid_shape() -> None:
    droid = get_profile("pi05_droid")
    assert _uses_exact_droid_final_norm(droid, precision="bf16")
    assert not _uses_exact_droid_final_norm(droid, precision="fp32")
    assert not _uses_exact_droid_final_norm(replace(droid, name="unqualified"), precision="bf16")
    assert not _uses_exact_droid_final_norm(replace(droid, action_horizon=10), precision="bf16")
    assert not _uses_exact_droid_final_norm(
        replace(
            droid,
            action_expert=replace(droid.action_expert, width=512),
        ),
        precision="bf16",
    )


def test_final_adaptive_norm_plugin_is_fixed_shape_and_fail_closed() -> None:
    source = (
        Path(__file__).resolve().parents[5]
        / "src/runtime/models/openpi/trt_plugins/rms_norm_plugin.cu"
    ).read_text(encoding="utf-8")
    compact = " ".join(source.split())

    assert 'kName = "OpenPIFinalAdaptiveRmsNorm"' in source
    assert 'kVersion = "1"' in source
    assert "hidden.d[1] == kFinalAdaptiveRows" in source
    assert "hidden.d[2] == kActionWidth" in source
    assert "bias.d[0] == kFinalAdaptiveProjectionWidth" in source
    assert "weight.d[0] == kActionWidth" in source
    assert "weight.d[1] == kFinalAdaptiveProjectionWidth" in source
    assert "condition.d[1] == kActionWidth" in source
    assert "input[0].min, input[1].min" in compact
    assert "input[0].max, input[1].max" in compact
    assert "epsilon != kFinalAdaptiveEpsilon" in source
    assert "<<<kFinalAdaptiveBlocks, kFinalAdaptiveThreads, 0, stream>>>" in compact
    assert "plugin_registrar_openpi_final_adaptive_rms_norm" in source


def test_action_weight_validation_rejects_missing_and_wrong_shape() -> None:
    profile = _tiny_action_profile()
    shapes = required_action_weight_shapes(profile)
    mapped = {name: np.zeros(shape, dtype=np.float32) for name, shape in shapes.items()}
    mapped.pop("projections.action_in.bias")
    with pytest.raises(ValueError, match="missing weight"):
        _validate_action_weights(mapped, profile)

    mapped["projections.action_in.bias"] = np.zeros((8,), dtype=np.float32)
    mapped["action.layer.1.attention.k.weight"] = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match=r"expected \(8, 4\)"):
        _validate_action_weights(mapped, profile)


@pytest.mark.trt
def test_tiny_action_expert_serializes_with_compact_prefill_cache() -> None:
    trt = pytest.importorskip("tensorrt")
    profile = _tiny_action_profile()
    plan = build_action_expert_engine(
        profile,
        _tiny_action_weights(profile),
        precision="bf16",
    )
    assert len(plan) > 0

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    io = {
        engine.get_tensor_name(index): (
            tuple(engine.get_tensor_shape(engine.get_tensor_name(index))),
            engine.get_tensor_dtype(engine.get_tensor_name(index)),
            engine.get_tensor_mode(engine.get_tensor_name(index)),
        )
        for index in range(engine.num_io_tensors)
    }
    assert set(io) == {
        "noisy_actions",
        "timestep",
        "step_size",
        "prefix_mask",
        "suffix_position_ids",
        "prefix_k_0",
        "prefix_v_0",
        "prefix_k_1",
        "prefix_v_1",
        "velocity",
        "next_actions",
    }
    for name in ("prefix_k_0", "prefix_v_0", "prefix_k_1", "prefix_v_1"):
        assert io[name] == (
            (1, profile.prefix_length, 1, 4),
            trt.bfloat16,
            trt.TensorIOMode.INPUT,
        )
    for name in ("velocity", "next_actions"):
        assert io[name] == (
            (1, profile.action_horizon, profile.action_dim),
            trt.float32,
            trt.TensorIOMode.OUTPUT,
        )
