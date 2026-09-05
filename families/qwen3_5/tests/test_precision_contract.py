# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precision contract for Qwen3.5 recurrent state inputs."""

from __future__ import annotations

import pytest


trt = pytest.importorskip("tensorrt")

from families.qwen3_5 import model  # noqa: E402


def test_fp16_runtime_inputs_preserve_fp32_recurrent_state() -> None:
    """FP16 storage must not quantize the persistent DeltaNet state."""

    class _Tensor:
        def __init__(self, name: str, dtype) -> None:
            self.name = name
            self.dtype = dtype

    class _Layer:
        def __init__(self, output: _Tensor) -> None:
            self.output = output

        def get_output(self, index: int) -> _Tensor:
            assert index == 0
            return self.output

    class _Network:
        def __init__(self) -> None:
            self.cast_inputs: list[_Tensor] = []

        def add_cast(self, tensor: _Tensor, dtype):
            self.cast_inputs.append(tensor)
            return _Layer(_Tensor(f"{tensor.name}_cast", dtype))

    network = _Network()
    attention_mask = _Tensor("attention_mask", trt.float32)
    conv_state = _Tensor("conv_state", trt.float32)
    ssm_state = _Tensor("ssm_state", trt.float32)
    cache_k = _Tensor("cache_k", trt.float16)
    cache_v = _Tensor("cache_v", trt.float16)

    (
        prepared_mask,
        prepared_conv,
        prepared_ssm,
        prepared_cache_k,
        prepared_cache_v,
    ) = model._prepare_runtime_inputs(
        network,
        trt.float16,
        attention_mask,
        [conv_state],
        [ssm_state],
        [cache_k],
        [cache_v],
    )

    assert prepared_mask.dtype == trt.float16
    assert prepared_conv[0].dtype == trt.float16
    assert prepared_cache_k[0].dtype == trt.float16
    assert prepared_cache_v[0].dtype == trt.float16
    assert prepared_ssm == [ssm_state]
    assert prepared_ssm[0].dtype == trt.float32
    assert ssm_state not in network.cast_inputs
