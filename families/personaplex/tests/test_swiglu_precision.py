# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the SwiGLU rounding boundary through the MLP's normal output."""

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")

from families.personaplex import graph_blocks  # noqa: E402
from families.personaplex.default_dual_profile_decoder import _swiglu_mlp  # noqa: E402

WIDTH = 128
# Binary values: exact in Half and BF16. Rounding SiLU(gate) to Half before the
# product gives 0.0865478515625; the FP32 product rounded once gives
# 0.08648681640625, about 600 FP32 ulps from a Half rounding boundary.
GATE, UP = 7.0 / 64.0, 1.5


def _weights():
    # One-hot input selects row 0; each projection has a single unit path, so
    # every GEMM output is exact and only the activation boundary can differ.
    w_gate = np.zeros((WIDTH, WIDTH), np.float32)
    w_up = np.zeros((WIDTH, WIDTH), np.float32)
    w_down = np.zeros((WIDTH, WIDTH), np.float32)
    w_gate[0, 0], w_up[0, 0], w_down[0, 0] = GATE, UP, 1.0
    return {"mlp.w_gate": w_gate, "mlp.w_up": w_up, "mlp.w_down": w_down}


def _build(path):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    x = network.add_input("x", trt.float16, (1, WIDTH))
    weights = _weights()
    if path == "graph_blocks":
        y = graph_blocks.add_swiglu_mlp(
            network,
            x,
            weights=weights,
            prefix="mlp",
            hidden_size=WIDTH,
            mlp_size=WIDTH,
            dtype=np.float16,
        )
    else:
        matmul = graph_blocks.make_matmul_fn(network, np.float16, None)
        y = _swiglu_mlp(
            network,
            x,
            matmul=matmul,
            weights=weights,
            prefix="mlp",
            hidden=WIDTH,
            mlp_size=WIDTH,
        )
    y.name = "y"
    network.mark_output(y)
    config = builder.create_builder_config()
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    runtime = trt.Runtime(logger)
    return runtime, runtime.deserialize_cuda_engine(plan)


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("rebuild", range(3))
@pytest.mark.parametrize("path", ["graph_blocks", "dual_profile"])
def test_swiglu_rounds_the_product_once(path, rebuild):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    del rebuild  # Each parametrized invocation builds a separate engine.
    runtime, engine = _build(path)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    x = torch.zeros((1, WIDTH), dtype=torch.float16, device="cuda")
    x[0, 0] = 1.0
    y = torch.empty((1, WIDTH), dtype=torch.float16, device="cuda")
    assert context.set_tensor_address("x", x.data_ptr())
    assert context.set_tensor_address("y", y.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    actual = y.cpu().reshape(-1)
    del context, engine, runtime
    # Pinned Moshi's compiled gating kernel: Half projections, SiLU and the
    # product in FP32, then one conversion to Half.
    gate = torch.tensor(GATE, dtype=torch.float32, device="cuda")
    up = torch.tensor(UP, dtype=torch.float32, device="cuda")
    expected = torch.zeros(WIDTH, dtype=torch.float16)
    expected[0] = (torch.nn.functional.silu(gate) * up).half().cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
