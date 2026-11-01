# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual TensorRT decoder/visual engines against independent PyTorch equations.

These component tests are not full-checkpoint or end-to-end qualification.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.trt]


def _config():
    from families.cosmos3_edge.config import ModelConfig

    return ModelConfig.from_dict({
        "model_type": "cosmos3_edge",
        "text_config": {
            "model_type": "cosmos3_edge_text", "hidden_act": "relu2", "hidden_size": 32,
            "intermediate_size": 48, "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 8, "vocab_size": 71,
            "max_position_embeddings": 128, "rms_norm_eps": 1e-5,
            "rope_parameters": {"rope_type": "default", "rope_theta": 1e8, "mrope_section": [2, 1, 1]},
        },
        "vision_config": {
            "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_channels": 3, "num_patches": 16,
            "patch_size": 2, "spatial_merge_size": 2, "layer_norm_eps": 1e-6,
            "hidden_act": "gelu_pytorch_tanh",
        },
        "projector_config": {
            "input_hidden_size": 16, "out_hidden_size": 32, "spatial_merge_size": 2,
            "use_postshuffle_norm": False, "merger_intermediate_size": 40,
        },
    })


def _weights(config):
    rng = np.random.default_rng(19)
    weights = {}

    def linear(name, output, input_, bias=False):
        weights[name + ".weight"] = rng.normal(0, 0.1, (output, input_)).astype(np.float32)
        if bias:
            weights[name + ".bias"] = rng.normal(0, 0.03, output).astype(np.float32)

    def norm(name, width, bias=False):
        weights[name + ".weight"] = rng.uniform(0.8, 1.2, width).astype(np.float32)
        if bias:
            weights[name + ".bias"] = rng.normal(0, 0.03, width).astype(np.float32)

    h, kv = config.hidden_size, config.num_key_value_heads * config.head_dim
    for index in range(config.num_hidden_layers):
        p = f"layers.{index}"
        norm(p + ".input_layernorm", h)
        norm(p + ".post_attention_layernorm", h)
        for name, size in (("q", h), ("k", kv), ("v", kv), ("out", h)):
            linear(p + ".self_attn.to_" + name, size, h)
        linear(p + ".mlp.up_proj", config.intermediate_size, h)
        linear(p + ".mlp.down_proj", h, config.intermediate_size)
    norm("norm", h)
    linear("lm_head", config.vocab_size, h)
    v = config.vision
    width = v["hidden_size"]
    linear("model.visual.embeddings.patch_embedding", width, v["num_channels"] * v["patch_size"] ** 2, True)
    weights["model.visual.embeddings.position_embedding.weight"] = rng.normal(0, 0.2, (v["num_patches"], width)).astype(np.float32)
    for index in range(v["num_hidden_layers"]):
        p = f"model.visual.encoder.layers.{index}"
        norm(p + ".layer_norm1", width, True)
        norm(p + ".layer_norm2", width, True)
        for name in ("q", "k", "v", "out"):
            linear(p + ".self_attn." + name + "_proj", width, width, True)
        linear(p + ".mlp.fc1", v["intermediate_size"], width, True)
        linear(p + ".mlp.fc2", width, v["intermediate_size"], True)
    norm("model.visual.post_layernorm", width, True)
    norm("model.projector.norm", width, True)
    linear("model.projector.linear_fc1", 40, width * 4, True)
    linear("model.projector.linear_fc2", h, 40, True)
    return weights


def _torch():
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("Cosmos native mathematical tests require CUDA")
    return torch


def _execute(plan, inputs):
    import torch
    import tensorrt as trt

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    tensors = {name: torch.from_numpy(np.ascontiguousarray(value)).cuda() for name, value in inputs.items()}
    for name, tensor in tensors.items():
        assert context.set_input_shape(name, tuple(tensor.shape))
        assert context.set_tensor_address(name, tensor.data_ptr())
    assert context.infer_shapes() == []
    names = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            assert engine.get_tensor_dtype(name) == trt.float32
            tensors[name] = torch.empty(tuple(context.get_tensor_shape(name)), dtype=torch.float32, device="cuda")
            assert context.set_tensor_address(name, tensors[name].data_ptr())
            names.append(name)
    stream = torch.cuda.current_stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return {name: tensors[name].cpu().numpy() for name in names}


def _decoder_reference(config, weights, inputs):
    import torch
    import torch.nn.functional as f

    w = {name: torch.from_numpy(array) for name, array in weights.items()}
    x = torch.from_numpy(inputs["inputs_embeds"])
    cos, sin = [torch.from_numpy(inputs[name])[None] for name in ("rope_cos", "rope_sin")]
    outputs = {}

    def norm(x, name):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + config.rms_norm_eps) * w[name + ".weight"]

    def rotary(x):
        half = x.shape[-1] // 2
        return x * cos + torch.cat((-x[..., half:], x[..., :half]), -1) * sin

    for index in range(config.num_hidden_layers):
        p = f"layers.{index}"
        y = norm(x, p + ".input_layernorm")
        q, k, v = [f.linear(y, w[p + ".self_attn.to_" + name + ".weight"])
                   .reshape(-1, heads, config.head_dim).transpose(0, 1)
                   for name, heads in (("q", config.num_attention_heads), ("k", config.num_key_value_heads),
                                       ("v", config.num_key_value_heads))]
        q, k = rotary(q), rotary(k)
        outputs[f"present_k_{index}"] = k.numpy()
        outputs[f"present_v_{index}"] = v.numpy()
        k = torch.cat((torch.from_numpy(inputs[f"cache_k_{index}"]), k), 1)
        v = torch.cat((torch.from_numpy(inputs[f"cache_v_{index}"]), v), 1)
        k, v = [a.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, 0) for a in (k, v)]
        probabilities = (q @ k.transpose(-1, -2) / config.head_dim ** 0.5
                         + torch.from_numpy(inputs["attention_mask"])).softmax(-1)
        attended = (probabilities @ v).transpose(0, 1).reshape(-1, config.hidden_size)
        x = x + f.linear(attended, w[p + ".self_attn.to_out.weight"])
        y = f.relu(f.linear(norm(x, p + ".post_attention_layernorm"), w[p + ".mlp.up_proj.weight"])).square()
        x = x + f.linear(y, w[p + ".mlp.down_proj.weight"])
    outputs["logits"] = f.linear(norm(x, "norm"), w["lm_head.weight"]).numpy()
    return outputs


def test_native_prefill_decode_matches_independent_equations():
    _torch()
    from families.cosmos3_edge.native_graph import build_decoder

    config = _config()
    weights = _weights(config)
    plan = build_decoder(config, weights, 8)
    rng = np.random.default_rng(29)
    cache = {f"cache_{kind}_{layer}": np.zeros((2, 8, 8), np.float32)
             for layer in range(2) for kind in ("k", "v")}
    offset = 0
    for length in (3, 1, 2):
        # Distinct per-channel angles exercise multimodal rotary values, not
        # just ordinary one-dimensional token positions.
        angles = rng.uniform(-2, 2, (length, 4)).astype(np.float32)
        angles = np.concatenate((angles, angles), axis=-1)
        mask = np.full((1, length, 8 + length), -10000, np.float32)
        mask[:, :, :offset] = 0
        mask[:, :, 8:] = np.triu(np.full((length, length), -10000, np.float32), 1)
        inputs = dict(cache, inputs_embeds=rng.normal(size=(length, 32)).astype(np.float32),
                      rope_cos=np.cos(angles), rope_sin=np.sin(angles), attention_mask=mask)
        expected = _decoder_reference(config, weights, inputs)
        actual = _execute(plan, inputs)
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], rtol=2e-4, atol=2e-5, err_msg=name)
        for layer in range(2):
            for kind in ("k", "v"):
                cache[f"cache_{kind}_{layer}"][:, offset:offset + length] = actual[f"present_{kind}_{layer}"]
        offset += length


def test_native_visual_matches_independent_siglip_and_merger():
    torch = _torch()
    import torch.nn.functional as f
    from families.cosmos3_edge.native_graph import build_visual

    config = _config()
    weights = _weights(config)
    plan = build_visual(config, weights, 16)
    w = {name: torch.from_numpy(array) for name, array in weights.items()}
    rng = np.random.default_rng(31)

    def linear(x, name):
        return f.linear(x, w[name + ".weight"], w[name + ".bias"])

    def norm(x, name):
        return f.layer_norm(x, (16,), w[name + ".weight"], w[name + ".bias"], 1e-6)

    for height, width in ((2, 2), (2, 6), (4, 4)):
        patches = height * width
        pixels = rng.normal(size=(patches, 12)).astype(np.float32)

        def resize_table(table):
            source = table.reshape(4, 4, -1).permute(2, 0, 1).unsqueeze(0)
            resized = f.interpolate(source, size=(height, width), mode="bilinear",
                                    align_corners=False, antialias=True)
            return resized[0].permute(1, 2, 0).reshape(height // 2, 2, width // 2, 2, -1).permute(
                0, 2, 1, 3, 4).reshape(patches, -1)

        fractions = resize_table(torch.eye(16)).numpy()
        positions = resize_table(w["model.visual.embeddings.position_embedding.weight"])
        x = linear(torch.from_numpy(pixels), "model.visual.embeddings.patch_embedding") + positions
        for i in range(2):
            p = f"model.visual.encoder.layers.{i}"
            y = norm(x, p + ".layer_norm1")
            q, k, v = [linear(y, p + ".self_attn." + kind + "_proj").reshape(-1, 4, 4).transpose(0, 1)
                       for kind in ("q", "k", "v")]
            attended = ((q @ k.transpose(-1, -2) / 2).softmax(-1) @ v).transpose(0, 1).reshape(-1, 16)
            x = x + linear(attended, p + ".self_attn.out_proj")
            x = x + linear(f.gelu(linear(norm(x, p + ".layer_norm2"), p + ".mlp.fc1"), approximate="tanh"),
                           p + ".mlp.fc2")
        x = norm(norm(x, "model.visual.post_layernorm"), "model.projector.norm").reshape(-1, 64)
        expected = linear(f.gelu(linear(x, "model.projector.linear_fc1")), "model.projector.linear_fc2").numpy()
        actual = _execute(plan, {"pixel_values": pixels, "position_weights": fractions})["image_features"]
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
