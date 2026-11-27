# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify that reference routing overrides tied GPU TopK results and weights."""

from types import SimpleNamespace
import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.trt
def test_reference_routing_override():
    trt = pytest.importorskip("tensorrt")
    cuda = pytest.importorskip("cuda.bindings.runtime")
    status, count = cuda.cudaGetDeviceCount()
    if int(status) or count == 0:
        pytest.skip("requires a CUDA GPU for the routing graph")
    torch = pytest.importorskip("torch")
    from families.qwen3_next.graph import Graph
    from families.qwen3_next.tests.gpu_runner import run

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    values = network.add_input("probabilities", trt.float32, (1, 512))
    graph = Graph(network, None)
    _, normalized = graph.routing(
        network.add_identity(values).get_output(0),
        SimpleNamespace(num_experts=512, num_experts_per_tok=10),
    )
    normalized.name = "normalized_weights"
    network.mark_output(normalized)
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)
    plan = bytes(builder.build_serialized_network(network, options))
    random = np.random.default_rng(20261111)
    cases = [np.zeros((1, 512), dtype=np.float32)]
    cases += [random.integers(-4, 5, (1, 512)).astype(np.float32) for _ in range(4)]
    for logits in cases:
        probabilities = torch.softmax(torch.from_numpy(logits), dim=-1)
        weights, ids = torch.topk(probabilities, 10, dim=-1)
        expected = weights / weights.sum(dim=-1, keepdim=True)
        actual = run(
            plan,
            {
                "probabilities": probabilities.numpy(),
                "forced_experts": ids[0].numpy().astype(np.int32),
                "force_routing": np.array([1], dtype=np.int32),
            },
        )
        np.testing.assert_array_equal(actual["selected_experts"], ids.numpy())
        np.testing.assert_array_equal(actual["router_probabilities"], probabilities.numpy())
        np.testing.assert_allclose(
            actual["normalized_weights"], expected.numpy(), rtol=1e-6, atol=0
        )
