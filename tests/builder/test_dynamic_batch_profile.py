# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for ``add_dynamic_batch_profile`` — single source of truth for
the diffusion engine ``kMIN=1, kOPT=opt, kMAX=max`` profile. The TRT
round-trip exercises validation, shape setting, and the engine's binding
acceptance in one go.
"""

from __future__ import annotations

from .conftest import requires_trt


@requires_trt
def test_builds_a_trivial_engine_and_enforces_max_batch(tmp_path):
    import tensorrt as trt
    from tensorrt_model_connect.engine_builder import add_dynamic_batch_profile

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    x = network.add_input("x", trt.float32, (-1, 8))
    identity = network.add_identity(x)
    network.mark_output(identity.get_output(0))
    config = builder.create_builder_config()

    add_dynamic_batch_profile(
        builder, config, network,
        input_names=["x"], max_batch=4, opt_batch=4,
        static_shape={"x": (8,)},
    )

    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(bytes(serialized))
    ctx = engine.create_execution_context()
    assert ctx.set_input_shape("x", (1, 8))
    assert ctx.set_input_shape("x", (4, 8))
    assert ctx.set_input_shape("x", (5, 8)) is False
