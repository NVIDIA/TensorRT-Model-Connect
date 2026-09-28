# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU graph-plumbing tests; these do not claim TensorRT numerical parity."""

from __future__ import annotations

import ast
import gc
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

from families.minimax_h3.fl2va_contract import MultimodalTextProfile, text_encoder_abi


SOURCE = Path(__file__).parents[1] / "multimodal_text_encoder_builder.py"


def _function(name, namespace):
    tree = ast.parse(SOURCE.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function.decorator_list = []
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


def test_five_ranges_partition_all_weights_once_and_embed_only_in_first():
    keys = _function("checkpoint_keys", {"NUM_LAYERS": 50})
    parts = [keys(start, start + 10) for start in range(0, 50, 10)]
    assert tuple(name for part in parts for name in part) == keys()
    assert [len(part) for part in parts] == [111, 110, 110, 110, 110]
    assert all("embed_tokens" not in name for part in parts[1:] for name in part)


@pytest.mark.parametrize(
    "bounds", [(-1, 10), (0, 0), (20, 10), (0, 51), (True, 10), (0, 10.0), (0, None)]
)
def test_invalid_layer_ranges_are_rejected(bounds):
    keys = _function("checkpoint_keys", {"NUM_LAYERS": 50})
    with pytest.raises(ValueError, match="layer bounds"):
        keys(*bounds)


class Tensor:
    def __init__(self, value, dtype="float32", name=""):
        self.value = np.asarray(value, dtype=np.float32)
        self.dtype = dtype
        self.name = name


class Layer:
    def __init__(self, tensor):
        self.tensor = tensor

    def get_output(self, index):
        return self.tensor


class Network:
    def __init__(self, values):
        self.values = values
        self.inputs = {}
        self.attentions = []
        self.injections = []
        self.casts = []
        self.linears = []
        self.output = None

    def add_input(self, name, dtype, shape):
        tensor = Tensor(self.values[name], dtype, name)
        self.inputs[name] = (tensor, shape)
        return tensor

    def add_gather(self, table, ids, axis):
        return Layer(Tensor(table.value[ids.value.astype(np.int32)], table.dtype))

    def add_elementwise(self, left, right, operation):
        operations = {"sum": np.add, "prod": np.multiply, "greater": np.greater}
        return Layer(Tensor(operations[operation](left.value, right.value), left.dtype))

    def add_select(self, condition, positive, negative):
        return Layer(
            Tensor(np.where(condition.value, positive.value, negative.value), positive.dtype)
        )

    def add_attention(self, query, key, value, normalization, causal):
        self.attentions.append(causal)
        return Layer(Tensor((query.value + key.value + value.value) / 3, query.dtype))

    def mark_output(self, output):
        self.output = output


def _builder_probe():
    """Execute the actual range builder with tiny deterministic stand-in ops."""
    profile = MultimodalTextProfile(2, 3, 4, 1, 1, 1)
    values = {
        binding.name: np.zeros((2, 4), np.float32) for binding in text_encoder_abi(profile).inputs
    }
    values.update(
        input_ids=[0, 1],
        vision_count=[1],
        vision_mask=np.ones((2, 1)),
        vision_embeds=np.arange(8, dtype=np.float32).reshape(2, 4) / 13,
    )
    for index in range(3):
        values[f"deepstack_{index}"] = np.full((2, 4), (index + 1) / 31, np.float32)
    builds = []

    class Builder:
        def __init__(self, logger):
            self.network = Network(values)
            self.optimization = SimpleNamespace(shapes={})
            self.optimization.set_shape = lambda name, *shapes: self.optimization.shapes.update(
                {name: shapes}
            )
            self.config = SimpleNamespace(
                clear_flag=lambda flag: None, add_optimization_profile=lambda profile: None
            )
            builds.append(self)

        def create_network(self, flags):
            return self.network

        def create_builder_config(self):
            return self.config

        def create_optimization_profile(self):
            return self.optimization

        def build_serialized_network(self, network, config):
            return network.output.value.tobytes()

    def cast(network, tensor, dtype):
        network.casts.append(dtype)
        value = (
            tensor.value.astype(ml_dtypes.bfloat16).astype(np.float32)
            if dtype == "bfloat16"
            else tensor.value
        )
        return Tensor(value, dtype, tensor.name)

    def scatter(network, hidden, rows, compact, active):
        network.injections.append(compact.name)
        return Tensor(compact.value, hidden.dtype)

    def linear(network, hidden, weights, name, *, turbo_fp32):
        network.linears.append(name)
        return Tensor(hidden.value * weights[f"{name}.weight"], hidden.dtype)

    def validate(network, *, expected_attentions, label):
        assert len(network.attentions) == expected_attentions

    class Logger:
        VERBOSE = 1
        WARNING = 2

        def __init__(self, severity):
            pass

    namespace = {
        "NUM_LAYERS": 50,
        "HIDDEN_SIZE": 5120,
        "HEAD_DIM": 128,
        "NUM_HEADS": 64,
        "NUM_KV_HEADS": 8,
        "NORM_EPS": 1e-6,
        "TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES": 1,
        "MultimodalTextProfile": MultimodalTextProfile,
        "text_encoder_abi": text_encoder_abi,
        "np": np,
        "gc": gc,
        "math": math,
        "Path": Path,
        "sys": sys,
        "_linear": linear,
        "_per_head_norm": lambda network, tensor, *args: tensor,
        "_repeat_kv": lambda network, tensor: tensor,
        "_mrope_cache": lambda *args, **kwargs: (None, None),
        "_visual_count_active": lambda network, value: value,
        "_scatter_visual_rows": scatter,
        "trt": SimpleNamespace(
            Logger=Logger,
            Builder=Builder,
            float32="float32",
            bfloat16="bfloat16",
            int32="int32",
            NetworkDefinitionCreationFlag=SimpleNamespace(STRONGLY_TYPED=0),
            BuilderFlag=SimpleNamespace(TF32="TF32"),
            ElementWiseOperation=SimpleNamespace(PROD="prod", SUM="sum", GREATER="greater"),
            AttentionNormalizationOp=SimpleNamespace(SOFTMAX="softmax"),
        ),
        "op": SimpleNamespace(
            configure_builder=lambda *args, **kwargs: None,
            configure_workspace=lambda *args, **kwargs: None,
            weight_constant=lambda network, value: Tensor(value),
            constant=lambda network, value: Tensor(
                value.reshape(-1)[0] if value.size == 1 else value
            ),
            cast=cast,
            rms_norm=lambda network, tensor, *args: tensor,
            partial_rope=lambda network, tensor, *args, **kwargs: tensor,
            rows_to_heads=lambda network, tensor, *args: tensor,
            heads_to_rows=lambda network, tensor, *args: tensor,
            silu=lambda network, tensor: Tensor(
                tensor.value / (1 + np.exp(-tensor.value)), tensor.dtype
            ),
            validate_native_network=validate,
            release_weight_buffers=lambda network: None,
        ),
    }
    _function("checkpoint_keys", namespace)
    _function("_network_shape", namespace)
    build = _function("build_multimodal_text_encoder_engine", namespace)
    keys = namespace["checkpoint_keys"]
    weights = {name: np.float32(0.015) for name in keys()}
    weights["model.language_model.embed_tokens.weight"] = (
        np.arange(8, dtype=np.float32).reshape(2, 4) / 17
    )
    return build, keys, weights, values, builds, profile


def test_partition_graph_composes_without_extra_rounding_or_repeated_embedding():
    build, keys, weights, values, builds, profile = _builder_probe()
    build(dict(weights), profile, turbo_fp32=True)
    complete = builds[-1].network
    parts = []
    for start, end in ((0, 1), (1, 3), (3, 10), (10, 50)):
        build(
            {name: weights[name] for name in keys(start, end)},
            profile,
            turbo_fp32=True,
            layer_start=start,
            layer_end=end,
        )
        stage = builds[-1].network
        parts.append(stage)
        values["hidden_states"] = stage.output.value.copy()
        assert stage.output.dtype == "float32"
        assert stage.output.name == "encoder_hidden_states"
        assert "bfloat16" not in stage.casts
        assert len(stage.inputs) == (9 if start == 0 else 10)
        if start:
            assert stage.inputs["hidden_states"][1] == (-1, 5120)
            assert builds[-1].optimization.shapes["hidden_states"] == (
                (2, 5120),
                (3, 5120),
                (4, 5120),
            )
    assert [name for part in parts for name in part.linears] == complete.linears
    assert [name for part in parts for name in part.injections] == [
        "vision_embeds",
        "deepstack_0",
        "deepstack_1",
        "deepstack_2",
    ]
    np.testing.assert_array_equal(parts[-1].output.value, complete.output.value)
    # The first boundary retains more than BF16 precision.
    assert np.any(
        parts[0].output.value != parts[0].output.value.astype(ml_dtypes.bfloat16).astype(np.float32)
    )


def test_non_turbo_rejects_partition_before_creating_a_builder():
    build, _keys, _weights, _values, builds, profile = _builder_probe()
    with pytest.raises(ValueError, match="requires Turbo FP32"):
        build({}, profile, layer_start=0, layer_end=10)
    assert not builds


@pytest.mark.parametrize("start", [0, 40])
def test_final_stage_keeps_fp32_output_until_runtime_conditioning_round(start):
    build, keys, weights, values, builds, profile = _builder_probe()
    values["hidden_states"] = np.arange(8, dtype=np.float32).reshape(2, 4) / 29
    build(
        {name: weights[name] for name in keys(start, 50)},
        profile,
        turbo_fp32=True,
        layer_start=start,
        layer_end=50,
    )
    graph = builds[-1].network
    assert graph.output.dtype == "float32"
    assert graph.output.name == "encoder_hidden_states"
    # The final residual must not be narrowed inside the engine. C++ owns the
    # sole conditioning round, after copying the genuine FP32 output.
    assert "bfloat16" not in graph.casts
    rounded = graph.output.value.astype(ml_dtypes.bfloat16).astype(np.float32)
    assert np.any(graph.output.value != rounded)


def test_fixed_sequence_partition_input_needs_no_dynamic_profile_entry():
    build, keys, weights, values, builds, _profile = _builder_probe()
    values["hidden_states"] = np.zeros((2, 4), np.float32)
    profile = MultimodalTextProfile(2, 2, 2, 1, 1, 1)
    build(
        {name: weights[name] for name in keys(10, 20)},
        profile,
        turbo_fp32=True,
        layer_start=10,
        layer_end=20,
    )
    assert builds[-1].network.inputs["hidden_states"][1] == (2, 5120)
    assert "hidden_states" not in builds[-1].optimization.shapes


def test_stage_rejects_weights_from_outside_its_partition():
    build, _keys, weights, _values, builds, profile = _builder_probe()
    with pytest.raises(ValueError, match="partition mismatch"):
        build(weights, profile, turbo_fp32=True, layer_start=0, layer_end=10)
    assert not builds


def test_default_full_stack_still_uses_nine_bindings_and_bf16_embedding():
    build, _keys, weights, _values, builds, profile = _builder_probe()
    build(weights, profile)
    graph = builds[-1].network
    assert len(graph.attentions) == 50 and len(graph.inputs) == 9
    assert graph.casts[0] == "bfloat16"
    assert graph.output.dtype == "float32"
