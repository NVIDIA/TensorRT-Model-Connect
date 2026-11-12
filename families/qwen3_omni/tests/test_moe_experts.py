# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Omni Thinker MoE: dense experts against a per-token reference, in bf16."""

from __future__ import annotations

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")

from .. import model as model_module  # noqa: E402

EXPERTS, HIDDEN, INTER, TOP_K = 8, 64, 32, 2


def _reference(x, ids, route, w_gate, w_up, w_down) -> np.ndarray:
    out = np.zeros((len(x), HIDDEN), np.float32)
    for row in range(len(x)):
        for slot in range(TOP_K):
            expert = ids[row, slot]
            gate = x[row] @ w_gate[expert]
            up = x[row] @ w_up[expert]
            out[row] += route[row, slot] * ((gate / (1 + np.exp(-gate)) * up) @ w_down[expert])
    return out


@pytest.mark.gpu
@pytest.mark.trt
def test_dense_experts_match_reference() -> None:
    """The dense experts give the routed-expert sum for one and several rows."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    rng = np.random.default_rng(0)
    w_gate, w_up = (
        (rng.standard_normal((EXPERTS, HIDDEN, INTER)) * 0.1).astype(np.float16) for _ in range(2)
    )
    w_down = (rng.standard_normal((EXPERTS, INTER, HIDDEN)) * 0.1).astype(np.float16)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    x = network.add_input("x", trt.bfloat16, (-1, HIDDEN))
    ids = network.add_input("ids", trt.int32, (-1, TOP_K))
    route = network.add_input("route", trt.bfloat16, (-1, TOP_K))
    weights = model_module._add_expert_weights(
        network, trt.bfloat16, w_gate, w_up, w_down, np.float16
    )
    out = model_module._add_dense_experts(network, x, ids, route, HIDDEN, EXPERTS, *weights)
    out.name = "out"
    network.mark_output(out)
    config = builder.create_builder_config()
    for low, opt, high in ((1, 16, 64), (1, 1, 1)):
        profile = builder.create_optimization_profile()
        for name, width in (("x", HIDDEN), ("ids", TOP_K), ("route", TOP_K)):
            profile.set_shape(name, (low, width), (opt, width), (high, width))
        config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)

    stream = torch.cuda.current_stream().cuda_stream
    for profile, rows in ((0, 5), (0, 1), (1, 1)):
        context = engine.create_execution_context()
        assert context.set_optimization_profile_async(profile, stream)
        x_np = rng.standard_normal((rows, HIDDEN)).astype(np.float16)
        ids_np = rng.integers(0, EXPERTS, (rows, TOP_K)).astype(np.int32)
        ids_np[:, 1] = (ids_np[:, 0] + 1) % EXPERTS
        route_np = rng.random((rows, TOP_K)).astype(np.float16)
        inputs = {"x": x_np, "ids": ids_np, "route": route_np}
        tensors = {name: torch.from_numpy(value).cuda() for name, value in inputs.items()}
        tensors["x"] = tensors["x"].to(torch.bfloat16)
        tensors["route"] = tensors["route"].to(torch.bfloat16)
        result = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
        for name, value in tensors.items():
            assert context.set_input_shape(name, tuple(value.shape))
            context.set_tensor_address(name, value.data_ptr())
        context.set_tensor_address("out", result.data_ptr())
        assert context.execute_async_v3(stream)
        torch.cuda.synchronize()

        want = _reference(
            x_np.astype(np.float32),
            ids_np,
            route_np.astype(np.float32),
            *(w.astype(np.float32) for w in (w_gate, w_up, w_down)),
        )
        # bf16 keeps 8 mantissa bits; compare against the output scale.
        tolerance = 2e-2 * float(np.abs(want).max())
        np.testing.assert_allclose(result.float().cpu().numpy(), want, atol=tolerance, rtol=0)
