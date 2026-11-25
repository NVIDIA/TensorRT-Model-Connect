# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The independent oracle must preserve small terms in cancelling dense sums."""

import numpy as np
import pytest
from ..config import Config
from ..reference import Reference


def test_dense_reference_preserves_cancelling_sum():
    oracle = Reference(
        Config(),
        {
            "projection.weight": np.ones((1, 3), np.float32),
            "projection.bias": np.zeros(1, np.float32),
        },
    )
    x = np.asarray([[1e8, 1, -1e8]], np.float32)
    np.testing.assert_array_equal(oracle.linear(x, "projection."), [[1]])


def test_timestamp_boundaries_match_pinned_upstream_cuda():
    from ..reference import timestamp_features

    # Recorded from torch.div(rounding_mode="floor") and torch.remainder
    # in the pinned original model, followed by its periodic encoding.
    timestamps = np.asarray([[1700009984], [2**40]], np.int64)
    actual = timestamp_features(
        timestamps, np.asarray([[3600, 86400]], np.float32), np.asarray([[24, 7]], np.float32)
    )
    expected = np.asarray(
        [
            [[1.0, 0.0, 0.621353030204773, -0.7835307717323303]],
            [[-0.4990803599357605, 0.8665557503700256, -0.901757001876831, -0.4322434365749359]],
        ],
        np.float32,
    )
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-7)


@pytest.mark.gpu
def test_native_timestamp_encoding_matches_upstream_rounding():
    import torch
    import tensorrt as trt
    from ..native import Network
    from ..checkpoint import POST

    rng = np.random.default_rng(12)
    values = np.concatenate(
        [
            np.asarray([[1700009984], [206587478016], [2**40]], np.int64),
            rng.integers(0, 2**40, size=(100000, 1), dtype=np.int64),
        ]
    )
    periods = np.asarray([[3600, 86400]], np.float32)
    cycles = np.asarray([[24, 7]], np.float32)
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    inputs = network.add_input("timestamp", trt.int64, values.shape)
    owner = Network(
        network,
        Config(),
        {POST + "_period_units": periods, POST + "_units_per_period": cycles},
        "fp32",
    )
    output = owner.timestamp_features(inputs)
    output.name = "features"
    network.mark_output(output)
    options = builder.create_builder_config()
    options.builder_optimization_level = 1
    plan = builder.build_serialized_network(network, options)
    assert plan is not None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    timestamp = torch.from_numpy(values).cuda()
    actual = torch.empty((len(values), 1, 4), dtype=torch.float32, device="cuda")
    context.set_tensor_address("timestamp", timestamp.data_ptr())
    context.set_tensor_address("features", actual.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    units = torch.div(
        timestamp.float()[..., None], torch.from_numpy(periods).cuda(), rounding_mode="floor"
    )
    phase = (
        torch.remainder(units, torch.from_numpy(cycles).cuda())
        / torch.from_numpy(cycles).cuda()
        * (2 * 3.14)
    )
    expected = torch.stack([phase.cos(), phase.sin()], -1).reshape(len(values), 1, 4)
    torch.cuda.synchronize()
    np.testing.assert_allclose(actual.cpu().numpy(), expected.cpu().numpy(), atol=1e-6, rtol=1e-6)
