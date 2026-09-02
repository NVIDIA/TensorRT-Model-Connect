# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from tensorrt_model_connect import trt_compat


trt_compat.configure_backend(rtx=True)
trt = trt_compat.get_trt()

from tensorrt_model_connect.families.minimax_h3 import graph_ops as op  # noqa: E402
from tensorrt_model_connect.families.minimax_h3.vae_builder import (  # noqa: E402
    MAX_BATCH,
    MIN_BATCH,
    OPT_BATCH,
    _broadcast_rows,
)


def test_video_vae_tile_profile_covers_public_aspect_envelope() -> None:
    assert (MIN_BATCH, OPT_BATCH, MAX_BATCH) == (15, 28, 33)


@pytest.mark.gpu
def test_video_vae_batch_constant_broadcasts_at_every_profile_point() -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    value = network.add_input("value", trt.float16, (-1, 8, 4))
    output = _broadcast_rows(
        network,
        value,
        np.arange(6, dtype=np.float32).reshape(1, 3, 2),
    )
    output.name = "output"
    network.mark_output(output)

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "value",
        (MIN_BATCH, 8, 4),
        (OPT_BATCH, 8, 4),
        (MAX_BATCH, 8, 4),
    )
    assert config.add_optimization_profile(profile) == 0
    try:
        plan = builder.build_serialized_network(network, config)
    finally:
        op.release_weight_buffers(network)
    assert plan

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    assert tuple(engine.get_tensor_shape("output")) == (-1, 3, 2)
    assert tuple(tuple(shape) for shape in engine.get_tensor_profile_shape("value", 0)) == (
        (MIN_BATCH, 8, 4),
        (OPT_BATCH, 8, 4),
        (MAX_BATCH, 8, 4),
    )
    for batch in (MIN_BATCH, OPT_BATCH, MAX_BATCH):
        context = engine.create_execution_context()
        assert context.set_input_shape("value", (batch, 8, 4))
        assert tuple(context.get_tensor_shape("output")) == (batch, 3, 2)
