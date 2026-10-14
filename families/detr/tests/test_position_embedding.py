# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare dynamic DETR positions with the official, all-valid image mask."""

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_position_embedding_tracks_shape_changes_in_one_context():
    from transformers.models.detr.modeling_detr import DetrSinePositionEmbedding

    from families.detr.graph.model import add_sine_position_embedding

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    feature = network.add_input("features", trt.float32, (1, 1, -1, -1))
    positions = add_sine_position_embedding(network, feature, 16)
    positions.name = "positions"
    network.mark_output(positions)
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape("features", (1, 1, 1, 1), (1, 1, 25, 25), (1, 1, 42, 42))
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    reference = DetrSinePositionEmbedding(num_position_features=8, normalize=True)
    for height, width in ((25, 38), (42, 25), (25, 25), (1, 1), (42, 42), (25, 38)):
        feature_tensor = torch.zeros((1, 1, height, width), device="cuda")
        assert context.set_input_shape("features", tuple(feature_tensor.shape))
        output = torch.empty(tuple(context.get_tensor_shape("positions")), device="cuda")
        assert context.set_tensor_address("features", feature_tensor.data_ptr())
        assert context.set_tensor_address("positions", output.data_ptr())
        assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        mask = torch.ones((1, height, width), dtype=torch.bool, device="cuda")
        expected = reference(shape=feature_tensor.shape, device="cuda", dtype=torch.float32,
                             mask=mask)[0]
        np.testing.assert_allclose(output.cpu().numpy(), expected.cpu().numpy(), atol=1e-5)
