# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical graph checks against independent official Transformers modules."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest


pytestmark = [pytest.mark.gpu, pytest.mark.trt]


def _reference_dependencies(*, official_module=True):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("a CUDA GPU is required")
    pytest.importorskip("tensorrt")
    pytest.importorskip("cuda.bindings.runtime")
    if official_module and importlib.util.find_spec("transformers.models.diffusion_gemma") is None:
        pytest.skip("the pinned DiffusionGemma Transformers reference is required")
    return torch


def _run(plan, inputs):
    import tensorrt as trt
    from cuda.bindings import runtime as cuda
    import ml_dtypes

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    allocations = []
    host_buffers = []
    outputs = {}

    def checked(result):
        assert int(result[0]) == 0, result[0]
        return result[1] if len(result) > 1 else None

    try:
        for name, value in inputs.items():
            value = np.ascontiguousarray(value)
            host_buffers.append(value)
            assert context.set_input_shape(name, value.shape)
            pointer = checked(cuda.cudaMalloc(max(value.nbytes, 1)))
            allocations.append(pointer)
            checked(cuda.cudaMemcpy(pointer, value.ctypes.data, value.nbytes,
                                    cuda.cudaMemcpyKind.cudaMemcpyHostToDevice))
            assert context.set_tensor_address(name, pointer)
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                continue
            shape = tuple(context.get_tensor_shape(name))
            assert all(dimension >= 0 for dimension in shape)
            dtype = engine.get_tensor_dtype(name)
            numpy_dtype = ml_dtypes.bfloat16 if dtype == trt.bfloat16 else trt.nptype(dtype)
            value = np.empty(shape, dtype=numpy_dtype)
            pointer = checked(cuda.cudaMalloc(max(value.nbytes, 1)))
            allocations.append(pointer)
            assert context.set_tensor_address(name, pointer)
            outputs[name] = (value, pointer)
        assert context.execute_async_v3(0)
        checked(cuda.cudaDeviceSynchronize())
        for value, pointer in outputs.values():
            checked(cuda.cudaMemcpy(value.ctypes.data, pointer, value.nbytes,
                                    cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        return {name: value for name, (value, _) in outputs.items()}
    finally:
        for pointer in allocations:
            checked(cuda.cudaFree(pointer))


def _tiny_config():
    from transformers.models.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaTextConfig

    return DiffusionGemmaTextConfig(
        hidden_size=16, vocab_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, num_global_key_value_heads=1,
        head_dim=4, global_head_dim=8, intermediate_size=24, moe_intermediate_size=8,
        num_experts=4, top_k_experts=2, layer_types=["sliding_attention", "full_attention"],
        max_position_embeddings=64, sliding_window=16,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0,
                               "partial_rotary_factor": 0.25},
        },
        final_logit_softcapping=30.0, rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh", attention_bias=False,
    )


def _graph_config(config):
    # The official config resolves these into per_layer_config and omits
    # their original constructor kwargs from to_dict(). The checkpoint
    # stores both explicitly.
    return {**config.to_dict(), "global_head_dim": 8, "num_global_key_value_heads": 1}


def _numpy(tensor, precision):
    import ml_dtypes

    array = tensor.detach().float().cpu().numpy()
    return array.astype(ml_dtypes.bfloat16 if precision == "bf16" else np.float32)


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_layer_matches_official_module(index, precision):
    torch = _reference_dependencies()
    from transformers.cache_utils import DynamicCache
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaDecoderTextLayer, DiffusionGemmaEncoderTextLayer, DiffusionGemmaTextRotaryEmbedding,
    )
    from families.diffusion_gemma.graph import build_layer_engine

    torch.manual_seed(72)
    config = _tiny_config()
    config._attn_implementation = "eager"
    # Match from_pretrained(): attention selection does not change its
    # independent default grouped-expert FP32 reduction contract.
    config._experts_implementation = "grouped_mm"
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    layer = DiffusionGemmaEncoderTextLayer(config, index).eval().to(dtype=dtype, device="cuda")
    # Independent official module initialization is retained, including every
    # norm/router scale and the expert accumulation implementation.
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            if "norm" in name or name.endswith("scale"):
                parameter.uniform_(0.8, 1.2)
            else:
                parameter.normal_(0, 0.03)
        layer.layer_scalar.fill_(0.125)
    weights = {name: _numpy(value, precision) for name, value in layer.state_dict().items()}
    plan = build_layer_engine(_graph_config(config), weights, index, precision=precision, max_sequence_length=32)
    hidden = torch.randn((1, 5, 16), dtype=dtype, device="cuda")
    positions = torch.arange(3, 8, device="cuda").unsqueeze(0)
    full = index == 1
    heads, width = (1, 8) if full else (2, 4)
    past_key = torch.randn((1, heads, 3, width), dtype=dtype, device="cuda")
    past_value = torch.randn_like(past_key)
    cache = DynamicCache(config=config)
    cache.update(past_key, past_value, index)
    mask = torch.full((1, 1, 5, 8), float("-inf"), dtype=dtype, device="cuda")
    for row in range(5):
        mask[:, :, row, :row + 4] = 0
    rotary = DiffusionGemmaTextRotaryEmbedding(config).to("cuda")
    with torch.no_grad():
        expected = layer(hidden, position_embeddings=rotary(hidden, positions, config.layer_types[index]),
                         attention_mask=mask, position_ids=positions, past_key_values=cache)
    actual = _run(plan, {
        "hidden_states": _numpy(hidden.squeeze(0), precision),
        "position_ids": positions.squeeze(0).cpu().numpy().astype(np.int32),
        "attention_mask": mask.squeeze(0).squeeze(0).float().cpu().numpy(),
        "past_key": _numpy(past_key.squeeze(0), precision),
        "past_value": _numpy(past_value.squeeze(0), precision),
    })
    tolerance = 0.006 if precision == "bf16" else 2e-5
    np.testing.assert_allclose(actual["hidden_states_out"].astype(np.float32),
                               expected.squeeze(0).float().cpu().numpy(), atol=tolerance, rtol=tolerance)
    for name, tensor in (("present_key", cache.layers[index].keys),
                         ("present_value", cache.layers[index].values)):
        np.testing.assert_allclose(actual[name].astype(np.float32),
                                   tensor.squeeze(0).float().cpu().numpy(), atol=tolerance, rtol=tolerance)
    # The same mathematical layer implements decoder attention with an
    # external bidirectional mask, while the official decoder never commits
    # its canvas to the encoder cache.
    decoder = DiffusionGemmaDecoderTextLayer(config, index).eval().to(dtype=dtype, device="cuda")
    decoder.load_state_dict(layer.state_dict())
    read_only = DynamicCache(config=config)
    read_only.update(past_key, past_value, index)
    bidirectional = torch.zeros_like(mask)
    with torch.no_grad():
        expected_decoder = decoder(
            hidden, position_embeddings=rotary(hidden, positions, config.layer_types[index]),
            attention_mask=bidirectional, position_ids=positions, past_key_values=read_only,
        )
    decoder_actual = _run(plan, {
        "hidden_states": _numpy(hidden.squeeze(0), precision),
        "position_ids": positions.squeeze(0).cpu().numpy().astype(np.int32),
        "attention_mask": bidirectional.squeeze(0).squeeze(0).float().cpu().numpy(),
        "past_key": _numpy(past_key.squeeze(0), precision),
        "past_value": _numpy(past_value.squeeze(0), precision),
    })
    np.testing.assert_allclose(decoder_actual["hidden_states_out"].astype(np.float32),
                               expected_decoder.squeeze(0).float().cpu().numpy(),
                               atol=tolerance, rtol=tolerance)
    assert read_only.layers[index].keys.shape[-2] == 3
    np.testing.assert_array_equal(decoder_actual["present_key"][:, :3].astype(np.float32),
                                   past_key.squeeze(0).float().cpu().numpy())


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_embedding_and_output_match_official_modules(precision):
    torch = _reference_dependencies()
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaSelfConditioning, DiffusionGemmaRMSNorm, DiffusionGemmaTextScaledWordEmbedding,
    )
    from families.diffusion_gemma.graph import build_embedding_engine, build_output_engine

    torch.manual_seed(91)
    config = _tiny_config()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    embedding = DiffusionGemmaTextScaledWordEmbedding(32, 16, 0, embed_scale=4).eval().to(dtype=dtype, device="cuda")
    conditioning = DiffusionGemmaSelfConditioning(config).eval().to(dtype=dtype, device="cuda")
    norm = DiffusionGemmaRMSNorm(16, eps=config.rms_norm_eps).eval().to(dtype=dtype, device="cuda")
    weights = {"embed_tokens.weight": _numpy(embedding.weight, precision)}
    weights.update({f"self_conditioning.{name}": _numpy(value, precision)
                    for name, value in conditioning.state_dict().items()})
    plan = build_embedding_engine(_graph_config(config), weights, precision=precision, max_sequence_length=16)
    ids = torch.tensor([1, 7, 12, 9], device="cuda")
    signal = torch.randn((4, 16), dtype=dtype, device="cuda")
    with torch.no_grad():
        expected_embedding = embedding(ids)
        expected_conditioned = conditioning(expected_embedding, signal)
    actual = _run(plan, {"input_ids": ids.cpu().numpy().astype(np.int32),
                         "self_conditioning": _numpy(signal, precision)})
    tolerance = 0.025 if precision == "bf16" else 2e-5
    for name, expected in (("encoder_embeddings", expected_embedding),
                            ("decoder_embeddings", expected_conditioned)):
        np.testing.assert_allclose(actual[name].astype(np.float32), expected.float().cpu().numpy(),
                                   atol=tolerance, rtol=tolerance)
    output_weights = {"embed_tokens.weight": weights["embed_tokens.weight"],
                      "norm.weight": _numpy(norm.weight, precision)}
    plan = build_output_engine(_graph_config(config), output_weights, precision=precision, max_sequence_length=16)
    hidden = torch.randn((4, 16), dtype=dtype, device="cuda")
    with torch.no_grad():
        raw = torch.nn.functional.linear(norm(hidden), embedding.weight).float()
        raw = torch.tanh(raw / 30.0) * 30.0
        previous = (raw / 0.7).to(dtype)
        probabilities = previous.softmax(-1, dtype=torch.float32).to(dtype)
        expected_signal = (probabilities @ embedding.weight) * embedding.embed_scale.to(dtype)
    actual = _run(plan, {"hidden_states": _numpy(hidden, precision),
                         "temperature": np.array([[0.7]], dtype=np.float32)})
    np.testing.assert_allclose(actual["raw_logits"].astype(np.float32), raw.cpu().numpy(), atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(actual["self_conditioning"].astype(np.float32),
                               expected_signal.float().cpu().numpy(), atol=tolerance, rtol=tolerance)


def test_router_python_scalar_preserves_opmath_precision():
    torch = _reference_dependencies(official_module=False)
    import tensorrt as trt
    from families.diffusion_gemma.graph import Graph, _create, _mark, _serialize

    logger, builder, network, config = _create("bf16", False)
    graph = Graph(network, {}, {}, "bf16")
    value = network.add_input("input", trt.bfloat16, (1024,))
    scaled = graph.python_scalar(value, 2816 ** -0.5)
    _mark(graph, scaled, "scaled")
    plan = _serialize(builder, network, config, builder.create_optimization_profile(), graph.buffers)
    source = (torch.arange(-512, 512, dtype=torch.float32) / 53).to(torch.bfloat16)
    expected = source * (2816 ** -0.5)
    # The checkpoint's H=2816 factor is not BF16 representable. This case
    # discriminates Python scalar semantics from an early BF16 scalar cast.
    rounded = source * torch.tensor(2816 ** -0.5, dtype=torch.bfloat16)
    assert (expected != rounded).sum().item() > 100
    actual = _run(plan, {"input": _numpy(source, "bf16")})["scaled"]
    np.testing.assert_array_equal(actual.astype(np.float32), expected.float().numpy())
