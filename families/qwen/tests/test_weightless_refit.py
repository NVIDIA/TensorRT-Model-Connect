# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real TensorRT canary for distinct placeholders and complete refitted plans."""

import gc

import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.trt
def test_equal_sized_placeholders_survive_refit_and_independent_deserialization(monkeypatch):
    trt = pytest.importorskip("tensorrt")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the counted-placeholder canary requires CUDA")
    from families.qwen import graph_ops, weightless

    logger = trt.Logger(trt.Logger.WARNING)

    def build():
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        config = builder.create_builder_config()
        config.builder_optimization_level = 3
        config.clear_flag(trt.BuilderFlag.TF32)
        x = network.add_input("x", trt.float32, (2, 512))
        arrays = (np.eye(512, dtype=np.float32) * 2,
                  np.fliplr(np.eye(512, dtype=np.float32)).copy() * 3)
        original_weights, null_descriptors = trt.Weights, []

        def record_weights(*args):
            if len(args) == 3:
                null_descriptors.append(args)
            return original_weights(*args)

        with weightless.weightless_build():
            with monkeypatch.context() as patch:
                patch.setattr(trt, "Weights", record_weights)
                for name, array in zip(("left", "right"), arrays, strict=True):
                    product = graph_ops.add_matmul_rhs_constant(
                        network, x, 512, 512, array, dtype=np.float32, weight_name=name)
                    product.name = name
                    network.mark_output(product)
            assert null_descriptors == [(trt.float32, 0, 512 * 512)] * 2
            # The helper also verifies the actual refitter catalog and both
            # TensorRT prototypes before setting the two distinct values.
            return weightless.build_serialized_network(
                builder, network, config, logger)

    plan = build()
    assert weightless._SESSION.get() is None
    gc.collect()
    # Only complete saved bytes survive. The source arrays, builder and
    # refitter are unavailable to this new runtime and execution context.
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        inputs = (torch.arange(2 * 512, dtype=torch.float32, device="cuda").reshape(2, 512) % 17 - 8)
        outputs = {name: torch.empty_like(inputs) for name in ("left", "right")}
        assert context.set_tensor_address("x", inputs.data_ptr())
        for name, output in outputs.items():
            assert context.set_tensor_address(name, output.data_ptr())
        assert context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        assert torch.equal(outputs["left"], inputs * 2)
        assert torch.equal(outputs["right"], inputs.flip(-1) * 3)
