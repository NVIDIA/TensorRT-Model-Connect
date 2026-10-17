# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the Qwen3 DeepStack merger's actual math against its official module."""

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")


@pytest.mark.gpu
@pytest.mark.trt
def test_qwen3_deepstack_merger_matches_exact_hf_gelu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchMerger

    from families.qwen_vl import graph_ops
    from families.qwen_vl.qwen_vl_vision_builder import _add_deepstack_merger

    torch.manual_seed(1731)
    config = Qwen3VLVisionConfig(hidden_size=64, out_hidden_size=32, spatial_merge_size=2)
    reference = Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=True).eval().cuda()
    weights = {"merger." + name: value.detach().cpu().numpy()
               for name, value in reference.state_dict().items()}
    # Wider random weights from the module constructor exercise the nonlinear
    # region independently of a particular checkpoint's activation distribution.
    values = torch.randn((24, 64), device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        expected = reference(values).cpu().numpy()

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    hidden = network.add_input("hidden", trt.float32, (24, 64))
    epsilon = graph_ops.add_constant(network, (1, 1), np.array([1e-6], dtype=np.float32))
    output = _add_deepstack_merger(
        network, hidden, weights, "merger", 64, 4, 6, 32,
        epsilon, np.arange(6, dtype=np.int32), dtype=np.float32)
    output.name = "features"
    network.mark_output(output)
    build_config = builder.create_builder_config()
    build_config.clear_flag(trt.BuilderFlag.TF32)
    plan = builder.build_serialized_network(network, build_config)
    assert plan is not None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    actual = torch.empty((6, 32), device="cuda", dtype=torch.float32)
    assert context.set_tensor_address("hidden", values.data_ptr())
    assert context.set_tensor_address("features", actual.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=2e-5, atol=5e-6)
