# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contracts for the model-owned SAM3.0 vision plan builder."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np

from tensorrt_model_connect.families.sam3 import vision_encoder_builder


def test_sam3_vision_builder_exposes_only_the_selected_graph() -> None:
    parameters = inspect.signature(
        vision_encoder_builder.build_sam3_vision_encoder_engine
    ).parameters
    removed_experiments = {
        "selective_fp16_gemms",
        "selective_fp16_gemm_mode",
        "batch5_profile",
        "dynamic_batch1_profile",
        "fp16_mlp_chain",
        "fp16_mlp_chain_last_n",
        "fp16_fc1_accum_last_n",
        "fp16_neck",
        "fpn_pixel_shuffle",
    }
    assert removed_experiments.isdisjoint(parameters)

    source = inspect.getsource(vision_encoder_builder.build_sam3_vision_encoder_engine)
    assert "(-1, 3, image_size, image_size)" in source
    assert "_add_exact_batch1_profile" in source
    assert "_add_attention_with_rope_batched" in source
    assert "build_sam3_serialized_network" in source
    assert "axis=1" in source
    assert "batch5" not in source


def test_sam3_vision_adds_one_exact_batch1_profile() -> None:
    calls: list[tuple[str, tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []

    class Profile:
        def set_shape(self, name, *, min, opt, max) -> None:
            calls.append((name, min, opt, max))

    class Builder:
        def create_optimization_profile(self):
            return Profile()

    class Config:
        def __init__(self):
            self.profiles = []

        def add_optimization_profile(self, profile) -> None:
            self.profiles.append(profile)

    config = Config()
    vision_encoder_builder._add_exact_batch1_profile(Builder(), config, image_size=1008)

    expected = (1, 3, 1008, 1008)
    assert calls == [("pixel_values", expected, expected, expected)]
    assert len(config.profiles) == 1


def test_sam3_selected_matmul_island_restores_fp32(monkeypatch) -> None:
    class Tensor:
        def __init__(self, name: str):
            self.name = name

    class Layer:
        def __init__(self, output: Tensor):
            self.output = output

        def get_output(self, index: int) -> Tensor:
            assert index == 0
            return self.output

    class Network:
        def __init__(self):
            self.casts = []

        def add_cast(self, tensor, dtype):
            self.casts.append((tensor.name, dtype))
            return Layer(Tensor(f"cast({tensor.name},{dtype})"))

    class GraphOps:
        def __init__(self):
            self.calls = []

        def add_matmul_rhs_constant(
            self, network, tensor, input_width, output_width, weight, *, dtype
        ):
            del network, weight
            self.calls.append((tensor.name, input_width, output_width, dtype))
            return Tensor("matmul")

    network = Network()
    graph_ops = GraphOps()
    monkeypatch.setattr(
        vision_encoder_builder,
        "_trt",
        lambda: SimpleNamespace(float16="fp16", float32="fp32"),
    )
    monkeypatch.setattr(vision_encoder_builder, "_graph_ops", lambda: graph_ops)

    output = vision_encoder_builder._add_fp16_matmul_island(
        network,
        Tensor("input"),
        8,
        16,
        np.zeros((16, 8), dtype=np.float32),
    )

    assert graph_ops.calls == [("cast(input,fp16)", 8, 16, np.float16)]
    assert network.casts == [("input", "fp16"), ("matmul", "fp32")]
    assert output.name == "cast(matmul,fp32)"


def test_sam3_vision_keeps_selected_mlp_and_tracker_neck() -> None:
    mlp_source = inspect.getsource(vision_encoder_builder._add_sam3_vision_mlp)
    assert "trt.float16" in mlp_source
    assert "_add_fp16_matmul_island" in mlp_source
    assert "fp16_chain" not in mlp_source
    assert "fp16_accum" not in mlp_source

    build_source = inspect.getsource(vision_encoder_builder.build_sam3_vision_encoder_engine)
    assert "_add_tracker_fpn_level" in build_source
    assert "sam3_tracker_position_2" in build_source


def test_sam3_tracker_neck_publishes_bf16_values_through_fp32_abi(monkeypatch) -> None:
    class Tensor:
        def __init__(self, name: str):
            self.name = name

    class Layer:
        def __init__(self, output: Tensor):
            self.output = output

        def get_output(self, index: int) -> Tensor:
            assert index == 0
            return self.output

    class Network:
        def __init__(self):
            self.casts = []
            self.outputs = []

        def add_cast(self, tensor, dtype):
            output = Tensor(f"cast({tensor.name},{dtype})")
            self.casts.append((tensor, dtype, output))
            return Layer(output)

        def mark_output(self, tensor) -> None:
            self.outputs.append(tensor)

    class GraphOps:
        @staticmethod
        def add_conv2d(network, tensor, *args, **kwargs):
            del network, args, kwargs
            return tensor

    network = Network()
    monkeypatch.setattr(
        vision_encoder_builder,
        "_trt",
        lambda: SimpleNamespace(bfloat16="bf16", float32="fp32"),
    )
    monkeypatch.setattr(vision_encoder_builder, "_graph_ops", lambda: GraphOps())
    weights = {
        "tracker.fpn.2.proj1.weight": None,
        "tracker.fpn.2.proj1.bias": None,
        "tracker.fpn.2.proj2.weight": None,
        "tracker.fpn.2.proj2.bias": None,
    }

    output = vision_encoder_builder._add_tracker_fpn_level(
        network,
        Tensor("tracker_hidden"),
        weights,
        level=2,
        hidden_size=1024,
        fpn_hidden_size=256,
    )

    assert [(tensor.name, dtype) for tensor, dtype, _ in network.casts] == [
        ("tracker_hidden", "bf16"),
        ("sam3_tracker_feature_2_bf16_round", "fp32"),
    ]
    assert output.name == "sam3_tracker_feature_2"
    assert network.outputs == [output]

    tracker_source = inspect.getsource(vision_encoder_builder._add_tracker_fpn_level)
    detector_source = inspect.getsource(vision_encoder_builder._add_fpn_level)

    bf16_round = 'network.add_cast(x, trt.bfloat16).get_output(0)'
    assert tracker_source.count(bf16_round) == 1
    assert 'network.add_cast(rounded, trt.float32).get_output(0)' in tracker_source
    assert 'rounded.name = f"sam3_tracker_feature_{level}_bf16_round"' in tracker_source
    assert bf16_round not in detector_source
