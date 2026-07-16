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

    source = inspect.getsource(
        vision_encoder_builder.build_sam3_vision_encoder_engine
    )
    assert '(-1, 3, image_size, image_size)' in source
    assert "_add_exact_batch1_profile" in source
    assert "_add_attention_with_rope_batched" in source
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

    build_source = inspect.getsource(
        vision_encoder_builder.build_sam3_vision_encoder_engine
    )
    assert "_add_tracker_fpn_level" in build_source
    assert "sam3_tracker_position_2" in build_source
