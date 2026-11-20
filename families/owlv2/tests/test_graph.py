# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OWLv2 graph helpers: fp16 matmuls and convolutions accumulate in fp32."""

from __future__ import annotations

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from .. import graph as g  # noqa: E402

# The image tower's fc2 at 3601 tokens: 3072-term sums, where TensorRT picks an
# fp16-accumulating kernel when it is not asked for fp32 accumulation.
_ROWS, _DEPTH, _WIDTH = 3601, 3072, 768


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def _run(torch, build_output, inputs: dict[str, np.ndarray], output_shape) -> np.ndarray:
    """Build a strongly typed fp16 network around ``build_output`` and run it once."""
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    tensors = {
        name: network.add_input(name, trt.float16, value.shape) for name, value in inputs.items()
    }
    out = build_output(network, tensors)
    out.name = "y"
    network.mark_output(out)
    plan = builder.build_serialized_network(network, builder.create_builder_config())
    assert plan is not None
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    device = {name: torch.from_numpy(value).cuda() for name, value in inputs.items()}
    result = torch.empty(output_shape, dtype=torch.float16, device="cuda")
    for name, value in device.items():
        context.set_tensor_address(name, value.data_ptr())
    context.set_tensor_address("y", result.data_ptr())
    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return result.cpu().numpy().astype(np.float64)


def _assert_at_rounding_floor(actual: np.ndarray, exact: np.ndarray) -> None:
    """As accurate as rounding the exact result to fp16; double rounding stays under 2x.

    A kernel that sums the 3072 products in fp16 lands near 10x.
    """
    floor = _rms(exact.astype(np.float16).astype(np.float64) - exact)
    error = _rms(actual - exact)
    assert error <= 2.0 * floor, f"error rms {error:.3g} vs fp16 rounding rms {floor:.3g}"


@pytest.mark.gpu
@pytest.mark.trt
def test_fp16_linear_accumulates_in_fp32() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((_ROWS, _DEPTH)).astype(np.float16)
    weight = (rng.standard_normal((_WIDTH, _DEPTH)) / np.sqrt(_DEPTH)).astype(np.float16)
    bias = (0.02 * rng.standard_normal(_WIDTH)).astype(np.float16)

    actual = _run(
        torch,
        lambda network, t: g.add_linear(network, t["x"], weight, bias, dtype=np.float16),
        {"x": x},
        (_ROWS, _WIDTH),
    )

    exact = x.astype(np.float64) @ weight.astype(np.float64).T + bias.astype(np.float64)
    _assert_at_rounding_floor(actual, exact)


@pytest.mark.gpu
@pytest.mark.trt
def test_fp16_patch_convolution_accumulates_in_fp32() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    patch, grid, channels = 16, 8, 12  # a stride-16 patch embedding, 3072-term sums
    pixels = rng.standard_normal((1, channels, patch * grid, patch * grid)).astype(np.float16)
    depth = channels * patch * patch
    weight = (rng.standard_normal((_WIDTH, channels, patch, patch)) / np.sqrt(depth)).astype(
        np.float16
    )

    actual = _run(
        torch,
        lambda network, t: g.add_conv2d(
            network, t["pixels"], weight, stride=(patch, patch), dtype=np.float16
        ),
        {"pixels": pixels},
        (1, _WIDTH, grid, grid),
    )

    tiles = pixels.astype(np.float64).reshape(1, channels, grid, patch, grid, patch)
    tiles = tiles.transpose(0, 2, 4, 1, 3, 5).reshape(grid * grid, depth)
    exact = tiles @ weight.astype(np.float64).reshape(_WIDTH, depth).T
    exact = exact.T.reshape(1, _WIDTH, grid, grid)
    _assert_at_rounding_floor(actual, exact)
