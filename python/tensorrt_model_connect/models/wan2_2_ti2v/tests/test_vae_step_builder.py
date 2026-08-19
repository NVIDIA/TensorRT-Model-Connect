# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the native recurrent Wan2.2 VAE decoder."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest

from tensorrt_model_connect.models.wan2_2_ti2v import graph_ops
from tensorrt_model_connect.models.wan2_2_ti2v import vae_step_builder as vae_builder
from tensorrt_model_connect.models.wan2_2_ti2v.vae_step_builder import (
    VAE_STEP_CACHE_SPECS,
    Wan22VaeStepProfile,
    select_vae_convolution_precision,
)


def test_vae_step_cache_contract_matches_source_order() -> None:
    profile = Wan22VaeStepProfile(44, 80)
    assert len(VAE_STEP_CACHE_SPECS) == 32
    assert VAE_STEP_CACHE_SPECS[0].logical_name == "decoder.conv_in"
    assert VAE_STEP_CACHE_SPECS[11].logical_name == ("decoder.up_blocks.0.upsampler.time_conv")
    assert VAE_STEP_CACHE_SPECS[18].logical_name == ("decoder.up_blocks.1.upsampler.time_conv")
    assert VAE_STEP_CACHE_SPECS[-1].logical_name == "decoder.conv_out"
    assert all(spec.shape(profile)[2] == 2 for spec in VAE_STEP_CACHE_SPECS)


@pytest.mark.parametrize(
    ("height", "width", "compute_capability", "integrated", "expected"),
    [
        (44, 80, (11, 0), True, "bf16"),
        (44, 80, (11, 0), False, "fp32"),
        (44, 80, (10, 3), False, "fp32"),
        (24, 42, (11, 0), True, "fp32"),
        (44, 79, (11, 0), True, "fp32"),
    ],
    ids=(
        "thor-official-720p",
        "discrete-sm110",
        "gb300",
        "l0",
        "non-official-default",
    ),
)
def test_vae_convolution_precision_is_scoped_to_thor_official_profile(
    height: int,
    width: int,
    compute_capability: tuple[int, int],
    integrated: bool,
    expected: str,
) -> None:
    profile = Wan22VaeStepProfile(height, width)
    assert (
        select_vae_convolution_precision(
            profile,
            compute_capability,
            integrated=integrated,
        )
        == expected
    )


def test_cuda_device_profile_uses_cuda_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimpleNamespace(
        cudaError_t=SimpleNamespace(cudaSuccess=0),
        cudaGetDevice=lambda: (0, 2),
        cudaGetDeviceProperties=lambda device: (
            0,
            SimpleNamespace(major=11 if device == 2 else 0, minor=0, integrated=1),
        ),
    )
    monkeypatch.setattr(vae_builder, "_cuda_runtime", lambda: runtime)
    assert vae_builder._current_cuda_device_profile() == ((11, 0), True)


def test_cuda_device_profile_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimpleNamespace(
        cudaError_t=SimpleNamespace(cudaSuccess=0),
        cudaGetDevice=lambda: (1, 0),
    )
    monkeypatch.setattr(vae_builder, "_cuda_runtime", lambda: runtime)
    with pytest.raises(RuntimeError, match="cudaGetDevice failed"):
        vae_builder._current_cuda_device_profile()


@pytest.mark.parametrize(
    ("convolution_dtype", "expected_input_casts", "expected_constants", "dynamic_weights"),
    [
        (graph_ops.trt.float32, (), 0, False),
        (
            graph_ops.trt.bfloat16,
            (
                graph_ops.trt.bfloat16,
                graph_ops.trt.bfloat16,
                graph_ops.trt.bfloat16,
            ),
            2,
            True,
        ),
    ],
    ids=("fp32-static-weights", "bf16-dynamic-weights"),
)
def test_vae_convolution_graph_contract(
    convolution_dtype,
    expected_input_casts: tuple[object, ...],
    expected_constants: int,
    dynamic_weights: bool,
) -> None:
    network = Mock()
    tensor = SimpleNamespace(shape=(1, 2, 4, 4), dtype=graph_ops.trt.float32)
    cast_outputs: list[SimpleNamespace] = []

    def add_cast(actual_tensor, dtype):
        output = SimpleNamespace(shape=tuple(actual_tensor.shape), dtype=dtype)
        cast_outputs.append(output)
        layer = Mock()
        layer.get_output.return_value = output
        return layer

    def add_constant(shape, weights):
        del weights
        layer = Mock()
        layer.get_output.return_value = SimpleNamespace(
            shape=tuple(shape), dtype=graph_ops.trt.float32
        )
        return layer

    network.add_cast.side_effect = add_cast
    network.add_constant.side_effect = add_constant
    convolution = Mock()
    convolution_output = SimpleNamespace(shape=(1, 3, 4, 4), dtype=convolution_dtype)
    convolution.get_output.return_value = convolution_output
    network.add_convolution_nd.return_value = convolution

    result = graph_ops._add_convolution(
        network,
        tensor,
        np.ones((3, 2, 3, 3), dtype=np.float32),
        np.zeros((3,), dtype=np.float32),
        out_channels=3,
        kernel_shape=(3, 3),
        convolution_dtype=convolution_dtype,
    )

    assert result is convolution
    assert tuple(item.args[1] for item in network.add_cast.call_args_list) == expected_input_casts
    assert network.add_constant.call_count == expected_constants
    if dynamic_weights:
        assert convolution.set_input.call_args_list == [
            call(1, cast_outputs[1]),
            call(2, cast_outputs[2]),
        ]
    else:
        convolution.set_input.assert_not_called()

    network.add_cast.reset_mock()
    output = graph_ops._convolution_output(network, convolution, convolution_dtype)
    if convolution_dtype == graph_ops.trt.bfloat16:
        network.add_cast.assert_called_once_with(convolution_output, graph_ops.trt.float32)
        assert output.dtype == graph_ops.trt.float32
    else:
        network.add_cast.assert_not_called()
        assert output is convolution_output


@pytest.mark.parametrize(("height", "width"), [(0, 2), (2, 0), (-1, 2)])
def test_vae_step_profile_rejects_nonpositive_dimensions(height: int, width: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        Wan22VaeStepProfile(height, width)
