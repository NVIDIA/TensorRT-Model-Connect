# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT round-trip coverage for MiniMax-H3 dynamic row helpers."""

from __future__ import annotations

import pytest


def _trt_available() -> bool:
    try:
        import tensorrt  # noqa: F401
        from cuda.bindings import runtime as cudart

        status, count = cudart.cudaGetDeviceCount()
        return int(status) == 0 and int(count) > 0
    except (ImportError, RuntimeError):
        return False


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.skipif(not _trt_available(), reason="TensorRT + CUDA not available")
def test_dynamic_row_slices_build_and_infer_runtime_shapes() -> None:
    import tensorrt as trt

    from families.minimax_h3 import graph_ops as op

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    value = network.add_input("value", trt.float32, (-1, 8))
    cos = network.add_input("cos", trt.float32, (1, -1, 2))
    sin = network.add_input("sin", trt.float32, (1, -1, 2))
    prefix = op.dynamic_slice(network, value, (0, 0), (None, 4))
    suffix = op.slice_rows_from_end(network, value, offset=2, rows=2)
    rotated = op.partial_rope(
        network,
        value,
        cos,
        sin,
        rows=-1,
        heads=1,
        head_dim=8,
        rotary_dim=4,
    )
    prefix.name = "prefix"
    suffix.name = "suffix"
    rotated.name = "rotated"
    network.mark_output(prefix)
    network.mark_output(suffix)
    network.mark_output(rotated)

    profile = builder.create_optimization_profile()
    profile.set_shape("value", min=(2, 8), opt=(4, 8), max=(8, 8))
    for name in ("cos", "sin"):
        profile.set_shape(name, min=(1, 2, 2), opt=(1, 4, 2), max=(1, 8, 2))
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()
    for rows in (2, 4, 8):
        assert context.set_input_shape("value", (rows, 8))
        assert context.set_input_shape("cos", (1, rows, 2))
        assert context.set_input_shape("sin", (1, rows, 2))
        assert tuple(context.get_tensor_shape("prefix")) == (rows, 4)
        assert tuple(context.get_tensor_shape("suffix")) == (2, 8)
        assert tuple(context.get_tensor_shape("rotated")) == (rows, 8)
