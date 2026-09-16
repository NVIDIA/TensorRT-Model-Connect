# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute native cache updates and both attention semantics on a CUDA device."""

from __future__ import annotations

import pytest

from families.hstu.cache_graph import add_cached_kv, native_sdpa


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("packed", [False, True], ids=["padded", "packed"])
def test_native_cache_append_and_attention_preserve_history(precision, packed):
    trt = pytest.importorskip("tensorrt")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("native cache graph execution requires a CUDA device")

    dtype = {"fp32": trt.float32, "fp16": trt.float16, "bf16": trt.bfloat16}[precision]
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    capacity, heads, width = 7, 2, 8
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    weights = []
    query = network.add_input("query", dtype, (-1, heads, -1, width))
    k_update = network.add_input("k_update", dtype, (-1, heads, -1, width))
    v_update = network.add_input("v_update", dtype, (-1, heads, -1, width))
    write_indices = network.add_input("write_indices", trt.int32, (-1,))
    active = network.add_input("active_mask", trt.bool, (-1, 1, 1, capacity))
    mask = network.add_input("attention_mask", trt.bool, (-1, 1, -1, capacity))
    packed_inputs = {}
    if packed:
        packed_inputs = {
            "packed_rows": network.add_input("packed_rows", trt.int32, (-1,)),
            "update_lengths": network.add_input("update_lengths", trt.int32, (-1,)),
        }
    key, value = add_cached_kv(
        trt, network, k_update, v_update, capacity, write_indices, active, "0", weights,
        **packed_inputs,
    )
    sdpa = native_sdpa(trt, network, query, key, value, mask, 0.5, weights, "sdpa")
    sdpa.name = "sdpa"
    network.mark_output(sdpa)

    # HSTU intentionally has no softmax. Exercise the same updated cache with
    # its independent nonlinear weighting, including masked NaN cache slots.
    scores = network.add_matrix_multiply(
        query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    sigmoid = network.add_activation(scores, trt.ActivationType.SIGMOID).get_output(0)
    scores = network.add_elementwise(scores, sigmoid, trt.ElementWiseOperation.PROD).get_output(0)
    visibility = network.add_cast(mask, dtype).get_output(0)
    scores = network.add_elementwise(scores, visibility, trt.ElementWiseOperation.PROD).get_output(0)
    hstu = network.add_matrix_multiply(
        scores, trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE,
    ).get_output(0)
    hstu.name = "hstu"
    network.mark_output(hstu)

    profile = builder.create_optimization_profile()
    profile.set_shape("query", (1, heads, 1, width), (2, heads, 2, width), (2, heads, 2, width))
    for name in ("k_update", "v_update"):
        profile.set_shape(name, (1, heads, int(packed), width), (2, heads, 2, width), (2, heads, 2, width))
    if packed:
        profile.set_shape("packed_rows", (0,), (4,), (4,))
        profile.set_shape("update_lengths", (2,), (3,), (3,))
    profile.set_shape("write_indices", (1,), (2,), (2,))
    profile.set_shape("active_mask", (1, 1, 1, capacity), (2, 1, 1, capacity), (2, 1, 1, capacity))
    profile.set_shape("attention_mask", (1, 1, 1, capacity), (2, 1, 2, capacity), (2, 1, 2, capacity))
    for name in ("cache_0_k", "cache_0_v"):
        profile.set_shape(name, (1, heads, capacity, width), (2, heads, capacity, width), (2, heads, capacity, width))
    config = builder.create_builder_config()
    config.builder_optimization_level = 0
    # Match the family builder's IEEE FP32 attention contract.
    config.clear_flag(trt.BuilderFlag.TF32)
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None, "native cache and attention graph must build"
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None

    generator = torch.Generator(device="cuda").manual_seed(371)

    def random(shape):
        return torch.randn(shape, generator=generator, device="cuda", dtype=torch_dtype) / 4

    caches = [torch.full((2, heads, capacity, width), float("nan"), dtype=torch_dtype, device="cuda") for _ in range(2)]
    for cache in caches:
        cache[0, :, :2] = random((heads, 2, width))
        cache[1, :, :3] = random((heads, 3, width))
    expected_caches = [cache.clone() for cache in caches]
    # TensorRT requires a non-null binding even when an input has zero elements.
    empty_binding = torch.empty((1,), dtype=torch_dtype, device="cuda")
    tolerances = {"fp32": (1e-5, 1e-6), "fp16": (5e-3, 1e-3), "bf16": (3e-2, 5e-3)}
    rtol, atol = tolerances[precision]

    # Different prefix lengths share a logical batch. A full history hit makes
    # no update, then later calls append to the same persistent buffers.
    calls = (
        (([2, 3], [2, 1], 2), ([4, 4], [0, 2], 2), ([4, 6], [0, 0], 2), ([4, 6], [2, 1], 2))
        if packed
        else (([2, 3], [2, 2], 2), ([4, 5], [0, 0], 2), ([4, 5], [1, 1], 1))
    )
    for positions, counts, query_count in calls:
        q = random((2, heads, query_count, width))
        padded_count = query_count if packed else counts[0]
        new_k, new_v = [random((2, heads, padded_count, width)) for _ in range(2)]
        if packed:
            # Padded query rows must never become cache updates, even when a
            # sequence has no updates or its cache is at the capacity boundary.
            for updates in (new_k, new_v):
                for sample, count in enumerate(counts):
                    updates[sample, :, count:] = float("nan")
        lengths = torch.tensor(
            [position + count for position, count in zip(positions, counts)],
            device="cuda", dtype=torch.int32,
        )
        active_rows = torch.arange(capacity, device="cuda")[None, :] < lengths[:, None]
        allowed = torch.arange(capacity, device="cuda")[None, None, None, :] <= (
            torch.tensor(positions, device="cuda")[:, None, None, None]
            + torch.arange(query_count, device="cuda")[None, None, :, None]
        )
        allowed &= active_rows[:, None, None, :]
        # PyTorch SDPA defines an entirely masked query as a zero output.
        allowed[1, :, 0, :] = False
        inputs = {
            "query": q, "k_update": new_k, "v_update": new_v,
            "write_indices": torch.tensor(positions, device="cuda", dtype=torch.int32),
            "active_mask": active_rows[:, None, None, :].contiguous(),
            "attention_mask": allowed.contiguous(),
            "cache_0_k": caches[0], "cache_0_v": caches[1],
        }
        if packed:
            inputs["packed_rows"] = torch.tensor(
                [sample * padded_count + row for sample, count in enumerate(counts) for row in range(count)],
                dtype=torch.int32, device="cuda",
            )
            inputs["update_lengths"] = torch.tensor(
                [0, counts[0], sum(counts)], dtype=torch.int32, device="cuda",
            )
        for cache, updates in zip(expected_caches, (new_k, new_v)):
            for sample, (position, count) in enumerate(zip(positions, counts)):
                cache[sample, :, position:position + count] = updates[sample, :, :count]
        for name, tensor in inputs.items():
            assert context.set_input_shape(name, tuple(tensor.shape))
            address = tensor.data_ptr() if tensor.numel() else empty_binding.data_ptr()
            assert context.set_tensor_address(name, address)
        assert context.set_tensor_address("present_0_k", caches[0].data_ptr())
        assert context.set_tensor_address("present_0_v", caches[1].data_ptr())
        outputs = {}
        for name in ("sdpa", "hstu"):
            outputs[name] = torch.empty(tuple(context.get_tensor_shape(name)), device="cuda", dtype=torch_dtype)
            assert context.set_tensor_address(name, outputs[name].data_ptr())
        assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        for actual, expected in zip(caches, expected_caches):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        safe_k, safe_v = [torch.where(active_rows[:, None, :, None], cache, 0) for cache in expected_caches]
        expected_sdpa = torch.nn.functional.scaled_dot_product_attention(
            q, safe_k, safe_v, attn_mask=allowed, scale=0.5,
        )
        expected_hstu = torch.nn.functional.silu(q @ safe_k.transpose(-1, -2)) * allowed
        expected_hstu = expected_hstu @ safe_v
        assert torch.isfinite(outputs["sdpa"]).all()
        assert torch.isfinite(outputs["hstu"]).all()
        torch.testing.assert_close(outputs["sdpa"], expected_sdpa, rtol=rtol, atol=atol)
        torch.testing.assert_close(outputs["hstu"], expected_hstu, rtol=rtol, atol=atol)
