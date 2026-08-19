# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precision-boundary contracts for the model-owned SAM3 core plan."""

from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pytest

from tensorrt_model_connect.models.sam3 import core_builder


def test_sam3_core_appends_meta_empty_geometry_prompt() -> None:
    source = textwrap.dedent(inspect.getsource(core_builder.build_sam3_core_engine))

    assert "geometry_features = _empty_geometry_prompt(" in source
    assert "[text_features, geometry_features]" in source
    assert "prompt_seq_len = text_seq_len + 1" in source
    assert "[text_mask_in, geometry_attention_mask]" in source
    assert source.count("kv_seq=prompt_seq_len") == 3
    assert "text_seq_len=prompt_seq_len" in source
    assert 'reduced_precision="bf16"' not in source


def test_sam3_empty_geometry_prompt_matches_meta_layer_order() -> None:
    source = textwrap.dedent(inspect.getsource(core_builder._empty_geometry_prompt))
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.FunctionDef)

    layer_norm_prefixes = [
        ast.unparse(node.args[3])
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_layer_norm"
    ]
    assert "'geometry_encoder.prompt_layer_norm'" in layer_norm_prefixes
    assert "f'{prefix}.layer_norm1'" in layer_norm_prefixes
    assert "f'{prefix}.layer_norm2'" in layer_norm_prefixes
    assert "f'{prefix}.layer_norm3'" in layer_norm_prefixes
    assert "'geometry_encoder.output_layer_norm'" in layer_norm_prefixes

    attention_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_attention"
    ]
    assert len(attention_calls) == 2
    assert ast.unparse(attention_calls[0].args[2]) == "normed"
    assert ast.unparse(attention_calls[1].args[2]) == "vision_with_position"
    assert ast.unparse(attention_calls[1].args[3]) == "vision_features"
    for call in attention_calls:
        keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
        assert keywords["reduced_precision"] == "'bf16'"


def test_sam3_core_encoder_alone_uses_fp16_mlp_island() -> None:
    source = textwrap.dedent(inspect.getsource(core_builder.build_sam3_core_engine))
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.FunctionDef)
    calls = sorted(
        (
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_sam3_mlp"
        ),
        key=lambda node: node.lineno,
    )
    assert len(calls) == 2

    encoder_keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in calls[0].keywords}
    decoder_keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in calls[1].keywords}
    assert encoder_keywords == {"fp16_island": "True"}
    assert "fp16_island" not in decoder_keywords


def test_sam3_core_mlp_casts_only_island_boundaries(monkeypatch) -> None:
    class Tensor:
        def __init__(self, name: str, dtype: str = "fp32"):
            self.name = name
            self.dtype = dtype

    class Layer:
        def __init__(self, output: Tensor):
            self.output = output

        def get_output(self, index: int) -> Tensor:
            assert index == 0
            return self.output

    class Network:
        def __init__(self):
            self.casts: list[tuple[str, str]] = []

        def add_cast(self, tensor: Tensor, dtype: str) -> Layer:
            self.casts.append((tensor.name, dtype))
            return Layer(Tensor(f"cast({tensor.name},{dtype})", dtype))

    class GraphOps:
        @staticmethod
        def add_activation(network, tensor, hidden_act):
            del network, hidden_act
            return Tensor(f"activation({tensor.name})")

    linear_inputs: list[str] = []

    def linear(network, tensor, weights, prefix, in_size, out_size):
        del network, weights, in_size, out_size
        linear_inputs.append(tensor.name)
        return Tensor(f"linear({prefix})")

    trt = type("Trt", (), {"float16": "fp16", "float32": "fp32"})
    monkeypatch.setattr(core_builder, "_trt", trt)
    monkeypatch.setattr(core_builder, "_graph_ops", GraphOps)
    monkeypatch.setattr(core_builder, "_linear", linear)
    network = Network()

    output = core_builder._sam3_mlp(
        network,
        Tensor("fp32_norm"),
        {},
        "detr_encoder.layers.0.mlp",
        256,
        2048,
        "relu",
        fp16_island=True,
    )

    assert linear_inputs == [
        "cast(fp32_norm,fp16)",
        "activation(linear(detr_encoder.layers.0.mlp.fc1))",
    ]
    assert network.casts == [
        ("fp32_norm", "fp16"),
        ("linear(detr_encoder.layers.0.mlp.fc2)", "fp32"),
    ]
    assert output.name == "cast(linear(detr_encoder.layers.0.mlp.fc2),fp32)"

    network.casts.clear()
    output = core_builder._sam3_mlp(
        network,
        Tensor("fp16_norm", "fp16"),
        {},
        "detr_encoder.layers.0.mlp",
        256,
        2048,
        "relu",
        fp16_island=True,
    )
    assert network.casts == [
        ("fp16_norm", "fp16"),
        ("linear(detr_encoder.layers.0.mlp.fc2)", "fp16"),
    ]
    assert output.dtype == "fp16"


def test_sam3_core_has_one_batch1_graph_with_native_group_norm(monkeypatch) -> None:
    assert "batch_size" not in inspect.signature(core_builder.build_sam3_core_engine).parameters
    assert "network.add_cast(ctx, query.dtype)" in inspect.getsource(core_builder._attention)

    class Tensor:
        def __init__(self, shape):
            self.shape = shape

    class Normalization:
        def __init__(self, output):
            self.output = output
            self.num_groups = 1
            self.epsilon = 0.0

        def get_output(self, index):
            assert index == 0
            return self.output

    class Network:
        def __init__(self):
            self.normalizations = []

        def add_normalization_v2(self, tensor, gamma, beta, axes):
            layer = Normalization(Tensor(tensor.shape))
            self.normalizations.append((gamma, beta, axes, layer))
            return layer

    monkeypatch.setattr(
        core_builder,
        "_const",
        lambda network, shape, values, dtype: Tensor(shape),
    )
    network = Network()
    output = core_builder._group_norm_4d(
        network,
        Tensor((1, 256, 144, 144)),
        {
            "pixel_norm.weight": np.ones((256,), dtype=np.float32),
            "pixel_norm.bias": np.zeros((256,), dtype=np.float32),
        },
        "pixel_norm",
        channels=256,
        groups=8,
        eps=1e-5,
    )

    assert output.shape == (1, 256, 144, 144)
    gamma, beta, axes, layer = network.normalizations[0]
    assert gamma.shape == beta.shape == (1, 256, 1, 1)
    assert axes == (1 << 2) | (1 << 3)
    assert layer.num_groups == 8
    assert layer.epsilon == pytest.approx(1e-5)
