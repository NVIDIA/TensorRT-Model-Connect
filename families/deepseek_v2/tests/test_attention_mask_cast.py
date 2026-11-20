# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP32-accumulation MLA attention keeps the FP16 mask FP16 where TensorRT 11.4 needs it."""

from __future__ import annotations

import pytest

trt = pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")

from families.deepseek_v2 import graph_ops, model  # noqa: E402


class _FakeConfig:
    def __init__(self, all_build_routes: str) -> None:
        self.all_build_routes = all_build_routes


@pytest.mark.parametrize(
    ("all_build_routes", "expected"),
    [
        ("-mha_fusion:gemv_mha_v2=on|off -mha_fusion:v2=on|off", True),
        ("-mha_fusion:v2=on|off", False),
        ("", False),
    ],
)
def test_keep_fp16_mask_only_when_the_knob_is_listed(all_build_routes, expected) -> None:
    assert model._keep_fp16_attention_mask(_FakeConfig(all_build_routes)) is expected
    assert model._keep_fp16_attention_mask(object()) is False


def _mask_is_cast_to_fp32(fp32_accumulation: bool, keep_fp16_mask: bool) -> bool:
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(0)
    q = network.add_input("q", trt.float16, (1, 4, 1, 48))
    k = network.add_input("k", trt.float16, (1, 4, 8, 48))
    v = network.add_input("v", trt.float16, (1, 4, 8, 48))
    mask = network.add_input("mask", trt.float16, (1, 1, 1, 8))
    graph_ops.add_attention_core(
        network,
        q,
        k,
        v,
        mask=mask,
        fp32_accumulation=fp32_accumulation,
        fp16_mask_with_fp32_accumulation=keep_fp16_mask,
    )
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    return any(
        layer.type == trt.LayerType.CAST
        and layer.get_input(0).name == "mask"
        and layer.get_output(0).dtype == trt.float32
        for layer in layers
    )


@pytest.mark.parametrize(
    ("fp32_accumulation", "keep_fp16_mask", "expected"),
    [
        (True, False, True),  # default and TensorRT 11.1: the mask is cast to FP32
        (True, True, False),  # TensorRT 11.4: no mixed half/float add
        (False, True, False),  # FP16 attention never casts the mask
        (False, False, False),
    ],
)
def test_attention_mask_cast(fp32_accumulation, keep_fp16_mask, expected) -> None:
    assert _mask_is_cast_to_fp32(fp32_accumulation, keep_fp16_mask) is expected
