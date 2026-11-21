# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small real TensorRT engines vs independent equations, not full-model evidence."""

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.trt]


def _torch():
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("Omni native mathematical tests require CUDA")
    return torch


def _execute(plan, inputs):
    import torch
    import tensorrt as trt

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    tensors = {
        name: torch.from_numpy(np.ascontiguousarray(value)).cuda() for name, value in inputs.items()
    }
    for name, tensor in tensors.items():
        assert context.set_input_shape(name, tuple(tensor.shape))
        assert context.set_tensor_address(name, tensor.data_ptr())
    assert context.infer_shapes() == []
    names = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            assert engine.get_tensor_dtype(name) == trt.float32
            tensors[name] = torch.empty(
                tuple(context.get_tensor_shape(name)), dtype=torch.float32, device="cuda"
            )
            assert context.set_tensor_address(name, tensors[name].data_ptr())
            names.append(name)
    stream = torch.cuda.current_stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return {name: tensors[name].cpu().numpy() for name in names}


def _tiny_config():
    from families.nemotron_omni.config import OmniConfig
    from .test_config import fixture

    raw = fixture()
    raw.update(patch_size=2, vit_hidden_size=16, projector_hidden_size=32)
    raw["vision_config"].update(
        num_hidden_layers=2,
        num_attention_heads=4,
        video_temporal_patch_size=2,
        separate_video_embedder=True,
    )
    return OmniConfig.from_dict(raw)


def test_image_and_real_temporal_patch_graphs_and_radio():
    torch = _torch()
    import torch.nn.functional as functional
    from families.nemotron_omni.visual_graph import build_patch, build_visual

    config = _tiny_config()
    rng = np.random.default_rng(73)
    weights = {}
    root = "vision_model.radio_model.model"

    def linear(prefix, rows, columns, bias=False):
        weights[prefix + ".weight"] = rng.normal(0, 0.15, (rows, columns)).astype(np.float32)
        if bias:
            weights[prefix + ".bias"] = rng.normal(0, 0.05, rows).astype(np.float32)

    def norm(prefix, width, centered=False):
        weights[prefix + ".weight"] = rng.uniform(0.8, 1.2, width).astype(np.float32)
        if centered:
            weights[prefix + ".bias"] = rng.normal(0, 0.05, width).astype(np.float32)

    linear(root + ".patch_generator.embedder", 16, 12)
    linear(root + ".patch_generator.video_embedder", 16, 24)
    weights[root + ".patch_generator.cls_token.token"] = rng.normal(size=(3, 16)).astype(np.float32)
    for layer in range(2):
        p = f"{root}.blocks.{layer}"
        for n in ("norm1", "norm2"):
            norm(p + "." + n, 16, True)
        linear(p + ".attn.qkv", 48, 16, True)
        linear(p + ".attn.proj", 16, 16, True)
        linear(p + ".mlp.fc1", 32, 16, True)
        linear(p + ".mlp.fc2", 16, 32, True)
    norm("mlp1.0", 64)
    linear("mlp1.1", 32, 64)
    linear("mlp1.3", 16, 32)
    w = {key: torch.from_numpy(value) for key, value in weights.items()}
    trunk = build_visual(config, weights, 16)

    def project(value, prefix, bias=False):
        return functional.linear(
            value, w[prefix + ".weight"], w[prefix + ".bias"] if bias else None
        )

    for video in (False, True):
        patch_plan = build_patch(config, weights, 16, video=video)
        for height, width in ((2, 2), (2, 6), (4, 4)):
            pixels = rng.normal(size=(height * width, 24 if video else 12)).astype(np.float32)
            prefix = root + ".patch_generator." + ("video_embedder" if video else "embedder")
            expected_patches = project(torch.from_numpy(pixels), prefix)
            actual_patches = _execute(patch_plan, {"patch_pixels": pixels})["patch_embeddings"]
            np.testing.assert_allclose(
                actual_patches, expected_patches.numpy(), rtol=2e-4, atol=2e-5
            )
            positions = rng.normal(0, 0.2, expected_patches.shape).astype(np.float32)
            x = torch.cat(
                (
                    w[root + ".patch_generator.cls_token.token"],
                    expected_patches + torch.from_numpy(positions),
                ),
                0,
            )
            for layer in range(2):
                p = f"{root}.blocks.{layer}"
                normalized = functional.layer_norm(
                    x, (16,), w[p + ".norm1.weight"], w[p + ".norm1.bias"], 1e-6
                )
                q, k, v = (
                    project(normalized, p + ".attn.qkv", True)
                    .reshape(-1, 3, 4, 4)
                    .permute(1, 2, 0, 3)
                )
                attended = (
                    ((q @ k.transpose(-1, -2) / 2).softmax(-1) @ v).transpose(0, 1).reshape(-1, 16)
                )
                x = x + project(attended, p + ".attn.proj", True)
                normalized = functional.layer_norm(
                    x, (16,), w[p + ".norm2.weight"], w[p + ".norm2.bias"], 1e-6
                )
                x = x + project(
                    functional.gelu(project(normalized, p + ".mlp.fc1", True)), p + ".mlp.fc2", True
                )
            # Independent v2 pixel shuffle, not the graph's gather implementation.
            image = x[3:].reshape(height, width, 16)
            image = image.reshape(height, width // 2, 32).permute(1, 0, 2)
            image = image.reshape(width // 2, height // 2, 64).permute(1, 0, 2).reshape(-1, 64)
            image = (
                image
                * torch.rsqrt(image.square().mean(-1, keepdim=True) + 1e-5)
                * w["mlp1.0.weight"]
            )
            expected = project(functional.relu(project(image, "mlp1.1")).square(), "mlp1.3")
            indices = np.array(
                [
                    [(2 * y + dy) * width + 2 * z + dx for dy in range(2) for dx in range(2)]
                    for y in range(height // 2)
                    for z in range(width // 2)
                ],
                np.int32,
            )
            actual = _execute(
                trunk,
                {
                    "patch_embeddings": actual_patches,
                    "position_embeddings": positions,
                    "shuffle_indices": indices,
                },
            )["image_features"]
            np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-4, atol=2e-5)


def test_decoder_consumes_supplied_media_embedding_not_text_lookup(tmp_path):
    torch = _torch()
    from safetensors.numpy import save_file
    from families.nemotron_omni.config import DecoderConfig
    from families.nemotron_omni.decoder import OmniDecoder

    config = DecoderConfig.from_dict(
        {
            "model_type": "nemotron_h",
            "vocab_size": 17,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "max_position_embeddings": 32,
            "hybrid_override_pattern": "--",
        }
    )
    rng = np.random.default_rng(82)
    state = {
        "language_model.backbone.embeddings.weight": np.zeros((17, 16), np.float32),
        "language_model.backbone.norm_f.weight": rng.uniform(0.8, 1.2, 16).astype(np.float32),
        "language_model.lm_head.weight": rng.normal(0, 0.2, (17, 16)).astype(np.float32),
    }
    for index in range(2):
        p = f"language_model.backbone.layers.{index}"
        state[p + ".norm.weight"] = rng.uniform(0.8, 1.2, 16).astype(np.float32)
        state[p + ".mixer.up_proj.weight"] = rng.normal(0, 0.1, (24, 16)).astype(np.float32)
        state[p + ".mixer.down_proj.weight"] = rng.normal(0, 0.1, (16, 24)).astype(np.float32)
    save_file(state, tmp_path / "model.safetensors")
    decoder = OmniDecoder()
    mapped = decoder.load_weights(str(tmp_path), config)
    plan = decoder.build_engine(config, mapped, 8)
    w = {key: torch.from_numpy(value) for key, value in state.items()}

    def norm(value, key):
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5) * w[key]

    for scale in (0.1, 1.0, 3.0):
        embedding = rng.normal(0, scale, (1, 16)).astype(np.float32)
        x = torch.from_numpy(embedding)
        for index in range(2):
            p = f"language_model.backbone.layers.{index}"
            up = torch.relu(norm(x, p + ".norm.weight") @ w[p + ".mixer.up_proj.weight"].T).square()
            x = x + up @ w[p + ".mixer.down_proj.weight"].T
        expected = (
            norm(x, "language_model.backbone.norm_f.weight") @ w["language_model.lm_head.weight"].T
        )
        actual = _execute(
            plan,
            {
                "input_embedding": embedding,
                "position_id": np.array([0], np.int32),
                "attention_mask": np.zeros((1, 9), np.float32),
            },
        )["logits"]
        np.testing.assert_allclose(
            actual.reshape(-1), expected.numpy().reshape(-1), rtol=2e-4, atol=2e-5
        )


def test_audio_encoder_and_projector_match_independent_hf_parakeet():
    torch = _torch()
    import torch.nn.functional as functional
    from transformers import ParakeetEncoder, ParakeetEncoderConfig
    from families.nemotron_omni.audio_graph import build_audio, encoded_length, relative_inputs
    from families.nemotron_omni.config import OmniConfig
    from .test_config import fixture

    torch.manual_seed(95)
    audio = {
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 4,
        "num_hidden_layers": 2,
        "intermediate_size": 32,
        "conv_kernel_size": 9,
        "subsampling_conv_channels": 4,
        "subsampling_conv_kernel_size": 3,
        "subsampling_conv_stride": 2,
        "subsampling_factor": 8,
        "num_mel_bins": 16,
        "attention_bias": False,
        "convolution_bias": False,
        "projection_bias": False,
        "projection_hidden_size": 32,
        "scale_input": False,
    }
    reference_config = ParakeetEncoderConfig(**audio)
    reference_config._attn_implementation = "eager"
    reference = ParakeetEncoder(reference_config).float().eval()
    weights = {
        "sound_encoder.encoder." + key: value.detach().numpy()
        for key, value in reference.state_dict().items()
    }
    rng = np.random.default_rng(96)
    weights["sound_projection.norm.weight"] = rng.uniform(0.8, 1.2, 16).astype(np.float32)
    weights["sound_projection.linear1.weight"] = rng.normal(0, 0.1, (32, 16)).astype(np.float32)
    weights["sound_projection.linear2.weight"] = rng.normal(0, 0.1, (16, 32)).astype(np.float32)
    raw = fixture()
    raw["sound_config"] = audio
    config = OmniConfig.from_dict(raw)
    plan = build_audio(config, weights, 65)
    for frames in (8, 17, 32, 65):
        mel = rng.normal(0, 0.5, (frames, 16)).astype(np.float32)
        positions, indices = relative_inputs(encoded_length(frames), 16, 4)
        with torch.inference_mode():
            expected = reference(torch.from_numpy(mel)[None]).last_hidden_state[0]
            expected = expected * torch.rsqrt(expected.square().mean(-1, keepdim=True) + 1e-5)
            expected = expected * torch.from_numpy(weights["sound_projection.norm.weight"])
            expected = functional.relu(
                functional.linear(
                    expected, torch.from_numpy(weights["sound_projection.linear1.weight"])
                )
            ).square()
            expected = functional.linear(
                expected, torch.from_numpy(weights["sound_projection.linear2.weight"])
            )
        actual = _execute(
            plan,
            {"mel_features": mel, "relative_positions": positions, "relative_indices": indices},
        )["sound_features"]
        np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-4, atol=2e-5)


def test_cpp_hybrid_state_progression_reset_and_capacity(tmp_path):
    """Same-engine transport parity, not an independent full-model quality oracle."""
    import json
    import os
    import subprocess

    _torch()
    from pathlib import Path
    from .test_e2e import _runner

    probe = os.environ.get("TRTMC_OMNI_DECODER_PROBE")
    if not probe:
        probe = _runner(
            Path(os.environ.get("TRTMC_RUNTIME_ROOT", ".")), "nemotron_omni_decoder_probe"
        )
    from safetensors.numpy import save_file
    from tensorrt_model_connect.bundle_writer import BundleWriter
    from families.nemotron_omni.config import DecoderConfig
    from families.nemotron_omni.decoder import OmniDecoder

    raw = {
        "model_type": "nemotron_h",
        "vocab_size": 17,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 32,
        "hybrid_override_pattern": "M*",
        "mamba_num_heads": 2,
        "mamba_head_dim": 8,
        "n_groups": 2,
        "ssm_state_size": 4,
        "conv_kernel": 4,
    }
    config = DecoderConfig.from_dict(raw)
    rng = np.random.default_rng(94)

    def weight(shape):
        return rng.normal(0, 0.1, shape).astype(np.float32)

    state = {
        "language_model.backbone.embeddings.weight": weight((17, 16)),
        "language_model.backbone.norm_f.weight": np.ones(16, np.float32),
        "language_model.lm_head.weight": weight((17, 16)),
    }
    for i in range(2):
        state[f"language_model.backbone.layers.{i}.norm.weight"] = np.ones(16, np.float32)
    p = "language_model.backbone.layers.0.mixer."
    for key, shape in [
        ("in_proj.weight", (50, 16)),
        ("out_proj.weight", (16, 16)),
        ("conv1d.weight", (32, 1, 4)),
        ("conv1d.bias", (32,)),
        ("A_log", (2,)),
        ("dt_bias", (2,)),
    ]:
        state[p + key] = weight(shape)
    state[p + "D"] = np.ones(2, np.float32)
    state[p + "norm.weight"] = np.ones(16, np.float32)
    p = "language_model.backbone.layers.1.mixer."
    for projection, shape in [("q", (16, 16)), ("k", (8, 16)), ("v", (8, 16)), ("o", (16, 16))]:
        state[p + projection + "_proj.weight"] = weight(shape)
    save_file(state, tmp_path / "model.safetensors")
    decoder = OmniDecoder()
    mapped = decoder.load_weights(tmp_path, config)
    plan = decoder.build_engine(config, mapped, 8)
    writer = BundleWriter(tmp_path / "probe.bundle")
    writer.set_header(family="nemotron_omni", task="text_continuation", backend="trt")
    writer.add_bytes("decoder.plan", plan)
    writer.add_json(
        "runtime.json",
        {
            "format": 1,
            "max_cache_length": 8,
            "config": {"model_type": "NemotronH_Nano_Omni_Reasoning_V3", "llm_config": raw},
            "state": decoder.get_bundle_config_overrides(config),
        },
    )
    writer.finish()
    embeddings = weight((4, 16))
    inputs = {
        "conv_state_0": np.zeros((32, 4), np.float32),
        "ssm_state_0": np.zeros((2, 8, 4), np.float32),
        "cache_k_0": np.zeros((8, 8), np.float32),
        "cache_v_0": np.zeros((8, 8), np.float32),
    }
    expected = []
    for index, embedding in enumerate(embeddings):
        mask = np.full((1, 9), np.finfo(np.float32).min, np.float32)
        mask[0, :index] = 0
        mask[0, -1] = 0
        output = _execute(
            plan,
            dict(
                inputs,
                input_embedding=embedding[None],
                position_id=np.array([index], np.int32),
                attention_mask=mask,
            ),
        )
        expected.append(output["logits"].reshape(-1))
        inputs["conv_state_0"] = output["present_conv_0"]
        inputs["ssm_state_0"] = output["present_ssm_0"]
        for name in ("k", "v"):
            inputs[f"cache_{name}_0"][index] = output[f"present_{name}_0"].reshape(-1)
    source, result = tmp_path / "embeddings.json", tmp_path / "result.json"
    source.write_text(json.dumps(embeddings.tolist()))
    subprocess.run([probe, str(tmp_path / "probe.bundle"), str(source), str(result)], check=True)
    actual = np.asarray(json.loads(result.read_text()), np.float32)
    np.testing.assert_array_equal(actual[0], expected)
    np.testing.assert_array_equal(actual[1], expected)
