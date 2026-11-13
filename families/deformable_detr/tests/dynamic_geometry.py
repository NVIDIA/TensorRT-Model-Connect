# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned GPU regression for dynamic spatial positions and attention."""

from __future__ import annotations


def check_dynamic_geometry():
    import numpy as np
    import tensorrt as trt
    import torch
    import torch.nn.functional as functional

    from families.deformable_detr import encoder_builder, graph as g, position_builder
    from families.deformable_detr.deformable_attention import build_deformable_attention

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    inputs = [network.add_input(f"level_{i}", trt.float32, (1, 256, -1, -1)) for i in range(4)]
    shapes = [g.shape_of(network, x, [2, 3]) for x in inputs]
    embedding = np.arange(4 * 256, dtype=np.float32).reshape(4, 256) / 1000
    position = position_builder.build_dynamic_level_embeddings(network, shapes, embedding)
    reference = encoder_builder.dynamic_reference_points(network, shapes)
    normalised = g.add_group_norm(network, inputs[0], np.ones(256), np.zeros(256), 32, 1e-5)
    hidden = g.concat(network, [g.spatial_to_tokens(network, x) for x in inputs], axis=1)
    width, heads, levels, points = 256, 8, 4, 4
    offsets = np.tile(np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]],
                               dtype=np.float32), (heads, levels, 1, 1))
    weights = {
        "probe.value_proj.weight": np.eye(width, dtype=np.float32),
        "probe.value_proj.bias": np.zeros(width, dtype=np.float32),
        "probe.sampling_offsets.weight": np.zeros((heads * levels * points * 2, width), np.float32),
        "probe.sampling_offsets.bias": offsets.reshape(-1),
        "probe.attention_weights.weight": np.zeros((heads * levels * points, width), np.float32),
        "probe.attention_weights.bias": np.zeros(heads * levels * points, np.float32),
        "probe.output_proj.weight": np.eye(width, dtype=np.float32),
        "probe.output_proj.bias": np.zeros(width, dtype=np.float32),
    }
    attended = build_deformable_attention(network, hidden, hidden, reference, weights,
                                         "probe", shapes, heads, points)
    for tensor, name in [(position, "position"), (reference, "reference"),
                         (normalised, "normalised"), (attended, "attended")]:
        tensor.name = name
        network.mark_output(tensor)
    profile = builder.create_optimization_profile()
    for tensor in inputs:
        profile.set_shape(tensor.name, (1, 256, 1, 1), (1, 256, 4, 6), (1, 256, 8, 8))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None, "dynamic geometry regression engine failed to build"
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    generator = torch.Generator(device="cuda").manual_seed(0)
    for height, image_width in [(4, 6), (6, 4), (5, 5)]:
        sizes = [(max(1, (height + (1 << i) - 1) // (1 << i)),
                  max(1, (image_width + (1 << i) - 1) // (1 << i))) for i in range(levels)]
        tensors = {f"level_{i}": torch.randn((1, width, h, w), generator=generator,
                                             device="cuda") for i, (h, w) in enumerate(sizes)}
        for name, tensor in tensors.items():
            assert context.set_input_shape(name, tuple(tensor.shape))
            assert context.set_tensor_address(name, tensor.data_ptr())
        results = {}
        for name in ("position", "reference", "normalised", "attended"):
            shape = tuple(context.get_tensor_shape(name))
            assert all(d > 0 for d in shape), (name, shape)
            results[name] = torch.empty(shape, dtype=torch.float32, device="cuda")
            assert context.set_tensor_address(name, results[name].data_ptr())
        assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        np.testing.assert_allclose(results["position"].cpu().numpy(),
                                   position_builder.build_level_embeddings(sizes, embedding),
                                   atol=2e-6, rtol=2e-6)
        expected_reference = encoder_builder.reference_points(sizes)
        np.testing.assert_allclose(results["reference"].cpu().numpy(), expected_reference,
                                   atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(results["normalised"],
                                   functional.group_norm(tensors["level_0"], 32, eps=1e-5),
                                   atol=2e-6, rtol=2e-6)
        centres = torch.from_numpy(expected_reference[0]).to("cuda")
        expected = []
        for level, (h, w) in enumerate(sizes):
            value = tensors[f"level_{level}"].reshape(heads, width // heads, h, w)
            delta = torch.from_numpy(offsets[:, level]).to("cuda") / torch.tensor([w, h], device="cuda")
            grid = 2 * (centres[:, level][None, :, None, :] + delta[:, None, :, :]) - 1
            expected.append(functional.grid_sample(value, grid, align_corners=False))
        expected = torch.stack(expected).mean(0).mean(-1).reshape(1, width, -1).transpose(1, 2)
        torch.testing.assert_close(results["attended"], expected, atol=3e-6, rtol=3e-6)
    print("dynamic geometry: landscape, portrait and square match independent NumPy/PyTorch outputs")
