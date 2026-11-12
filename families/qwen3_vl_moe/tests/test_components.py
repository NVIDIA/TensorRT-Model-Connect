# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small real TensorRT graphs against the official Qwen3-VL MoE modules."""

import os
import gc
import ml_dtypes
import numpy as np
import pytest


def test_bf16_constants_preserve_checkpoint_range_and_storage_lifetime():
    if os.environ.get("TRTMC_QWEN3_VL_MOE_COMPONENTS") != "1":
        pytest.skip("explicit component GPU execution is required")
    import torch
    import tensorrt as trt
    from families.qwen3_vl_moe.graph_ops import add_constant

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    expected = np.array([1, 2**-25, -(2**-25), 65536], dtype=np.float32)
    constant = add_constant(net, (4,), expected.copy(), dtype=ml_dtypes.bfloat16)
    assert constant.dtype == trt.bfloat16
    output = net.add_cast(constant, trt.float32).get_output(0)
    output.name = "output"
    net.mark_output(output)
    gc.collect()
    plan = builder.build_serialized_network(net, config)
    assert plan is not None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    actual = torch.empty(4, device="cuda", dtype=torch.float32)
    context.set_tensor_address("output", actual.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("experts,top_k", [(4, 2), (12, 8)])
def test_routed_experts_against_official_module(precision, rows, experts, top_k):
    if os.environ.get("TRTMC_QWEN3_VL_MOE_COMPONENTS") != "1":
        pytest.skip("explicit component GPU execution is required")
    import torch
    import tensorrt as trt
    from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeTextConfig
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock
    from families.qwen3_vl_moe.moe import add_moe

    assert torch.cuda.is_available()
    torch.manual_seed(813)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    storage = ml_dtypes.bfloat16 if precision == "bf16" else np.float32
    cfg = Qwen3VLMoeTextConfig(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=12,
        num_experts=experts,
        num_experts_per_tok=top_k,
    )
    reference = Qwen3VLMoeTextSparseMoeBlock(cfg).eval()
    with torch.no_grad():
        for p in reference.parameters():
            p.normal_(0, 0.15)
    reference = reference.to(device="cuda", dtype=dtype)
    source = torch.randn(rows, 16, device="cuda", dtype=dtype)
    with torch.inference_mode():
        expected = reference(source.unsqueeze(0))[0].float().cpu().numpy()
    packed = reference.experts.gate_up_proj.float().cpu().detach().numpy()
    weights = {
        "layer.0.router": reference.gate.weight.float().cpu().detach().numpy().T,
        "layer.0.experts.w_gate": packed[:, :12].transpose(0, 2, 1),
        "layer.0.experts.w_up": packed[:, 12:].transpose(0, 2, 1),
        "layer.0.experts.w_down": reference.experts.down_proj.float()
        .cpu()
        .detach()
        .numpy()
        .transpose(0, 2, 1),
    }
    weights = {k: np.ascontiguousarray(v, dtype=storage) for k, v in weights.items()}
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    config.clear_flag(trt.BuilderFlag.TF32)
    hidden = net.add_input(
        "hidden", trt.bfloat16 if precision == "bf16" else trt.float32, (rows, 16)
    )
    output = add_moe(net, hidden, weights, "layer.0", top_k=top_k, dtype=storage)
    output.name = "output"
    net.mark_output(output)
    plan = builder.build_serialized_network(net, config)
    assert plan is not None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    out = torch.empty_like(source)
    context.set_tensor_address("hidden", source.data_ptr())
    context.set_tensor_address("output", out.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    actual = out.float().cpu().numpy()
    if precision == "fp32":
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-4)
    else:
        np.testing.assert_allclose(actual, expected, atol=0.005, rtol=0.02)
