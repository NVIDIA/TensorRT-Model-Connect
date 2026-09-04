# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8 hybrid graph and runtime-config contracts."""

from __future__ import annotations

import tensorrt as trt

from families.qwen3_8 import engine_builder
from families.qwen3_8.config import ModelConfig


def test_layer_type_aliases_are_normalized() -> None:
    assert engine_builder._parse_layer_types(
        ["linear", "FULL", "linear_attention", "full_attention", "Custom"]
    ) == ["deltanet", "attention", "deltanet", "attention", "custom"]


def test_fp16_runtime_inputs_keep_recurrent_state_in_fp32() -> None:
    class Tensor:
        def __init__(self, name: str, dtype) -> None:
            self.name = name
            self.dtype = dtype

    class Layer:
        def __init__(self, output: Tensor) -> None:
            self.output = output

        def get_output(self, index: int) -> Tensor:
            assert index == 0
            return self.output

    class Network:
        def __init__(self) -> None:
            self.cast_inputs: list[Tensor] = []

        def add_cast(self, tensor: Tensor, dtype) -> Layer:
            self.cast_inputs.append(tensor)
            return Layer(Tensor(f"{tensor.name}_cast", dtype))

    network = Network()
    attention_mask = Tensor("attention_mask", trt.float32)
    conv_state = Tensor("conv_state", trt.float32)
    recurrent_state = Tensor("recurrent_state", trt.float32)
    cache_k = Tensor("cache_k", trt.float16)
    cache_v = Tensor("cache_v", trt.float16)

    prepared = engine_builder._prepare_runtime_inputs(
        network,
        trt.float16,
        attention_mask,
        [conv_state],
        [recurrent_state],
        [cache_k],
        [cache_v],
    )

    prepared_mask, prepared_conv, prepared_recurrent, prepared_k, prepared_v = prepared
    assert prepared_mask.dtype == trt.float16
    assert prepared_conv[0].dtype == trt.float16
    assert prepared_k[0].dtype == trt.float16
    assert prepared_v[0].dtype == trt.float16
    assert prepared_recurrent == [recurrent_state]
    assert recurrent_state not in network.cast_inputs


def test_runtime_config_publishes_flat_hybrid_dimensions() -> None:
    layer_types = ["linear_attention", "full_attention", "unknown"]
    config = ModelConfig(
        model_type="qwen3_8",
        vocab_size=32,
        hidden_size=12,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=3,
        num_key_value_heads=1,
        raw={
            "text_config": {
                "vocab_size": 32,
                "hidden_size": 12,
                "intermediate_size": 16,
                "num_hidden_layers": 3,
                "num_attention_heads": 3,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "layer_types": layer_types,
                "linear_num_value_heads": 3,
                "linear_num_key_heads": 1,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 5,
            }
        },
    )

    runtime = engine_builder.Qwen38Model().get_bundle_config_overrides(config)

    assert runtime["layer_types"] == ["deltanet", "attention", "unknown"]
    assert runtime["num_mamba_layers"] == 1
    assert runtime["num_attention_layers"] == 1
    assert runtime["hidden_size"] == 12
    assert runtime["num_attention_heads"] == 3
    assert runtime["num_key_value_heads"] == 1
    assert runtime["head_dim"] == 4
    assert runtime["d_inner"] == 12
    assert runtime["mamba_d_state"] == 4
    assert runtime["mamba_d_conv"] == 5
    assert runtime["mamba_nheads"] == 3
    assert runtime["mamba_head_dim"] == 4
    assert runtime["conv_dim"] == 20
