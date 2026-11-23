# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused independent official-module oracles for dynamic LightOn graphs."""

from __future__ import annotations

import numpy as np
import pytest

from families.lighton_ocr.config import Config


def _dependencies():
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensorrt")
    pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("The TensorRT graph oracle requires a CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch


def _config():
    return Config.from_dict({
        "architectures": ["LightOnOCRForConditionalGeneration"], "model_type": "mistral3",
        "spatial_merge_size": 2, "image_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
        "projector_hidden_act": "gelu", "vision_feature_layer": -1,
        "text_config": {
            "model_type": "qwen3", "hidden_size": 8, "intermediate_size": 24,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
            "num_hidden_layers": 1, "max_position_embeddings": 32, "vocab_size": 32,
            "rms_norm_eps": 1e-6, "hidden_act": "silu", "tie_word_embeddings": True,
            "rope_parameters": {"rope_type": "default", "rope_theta": 1000000},
        },
        "vision_config": {
            "model_type": "pixtral", "hidden_size": 8, "intermediate_size": 32,
            "num_attention_heads": 2, "head_dim": 4, "num_hidden_layers": 1,
            "image_size": 112, "patch_size": 14, "num_channels": 3, "hidden_act": "silu",
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000},
        },
    })


def _numpy(tensor):
    return tensor.detach().float().cpu().numpy()


def _run(plan, inputs):
    import tensorrt as trt
    import torch

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    buffers = {}
    for name, value in inputs.items():
        value = np.ascontiguousarray(value)
        assert context.set_input_shape(name, value.shape)
        if value.size:
            buffers[name] = torch.as_tensor(value, device="cuda")
        else:
            # Empty K/V has zero logical volume but a valid storage address.
            buffers[name] = torch.empty(1, dtype=torch.float32, device="cuda")
        assert context.set_tensor_address(name, buffers[name].data_ptr())
    assert not context.infer_shapes()
    outputs = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            continue
        shape = tuple(context.get_tensor_shape(name))
        assert all(value >= 0 for value in shape)
        assert engine.get_tensor_dtype(name) == trt.float32
        buffers[name] = torch.empty(shape, dtype=torch.float32, device="cuda")
        assert context.set_tensor_address(name, buffers[name].data_ptr())
        outputs.append(name)
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return {name: _numpy(buffers[name]) for name in outputs}


def _compare(actual, expected, absolute):
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    relative = np.linalg.norm(difference) / max(np.linalg.norm(expected.astype(np.float64)), 1e-12)
    assert relative <= 1e-4
    assert np.abs(difference).max() <= absolute


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("image_shape", [(28, 56), (42, 70)])
def test_pixtral_and_projector_match_official_even_and_odd_patch_grids(image_shape):
    torch = _dependencies()
    from transformers import LightOnOcrConfig
    from transformers.models.lighton_ocr.modeling_lighton_ocr import LightOnOcrMultiModalProjector
    from transformers.models.pixtral.modeling_pixtral import PixtralVisionModel
    from families.lighton_ocr.vision import build_vision_engine

    torch.manual_seed(71)
    config = _config()
    reference_config = LightOnOcrConfig.from_dict(config.raw)
    reference_config.vision_config._attn_implementation = "eager"
    vision = PixtralVisionModel(reference_config.vision_config).eval().float().cuda()
    projector = LightOnOcrMultiModalProjector(reference_config).eval().float().cuda()
    pixels = torch.randn(1, 3, *image_shape, dtype=torch.float32, device="cuda")
    sizes = torch.tensor([image_shape], device="cuda")
    with torch.inference_mode():
        features = vision(pixel_values=pixels, image_sizes=sizes).last_hidden_state.squeeze(0)
        expected = _numpy(projector(features, sizes))
    weights = {name: _numpy(value) for name, value in vision.state_dict().items()}
    weights.update({"vision_projection." + name: _numpy(value) for name, value in projector.state_dict().items()})
    side = config.vision_config["image_size"] // config.vision_config["patch_size"]
    rows, columns = (dimension // config.vision_config["patch_size"] for dimension in image_shape)
    positions = (np.arange(rows)[:, None] * side + np.arange(columns)[None]).reshape(-1).astype(np.int32)
    plan = build_vision_engine(config, weights)
    actual = _run(plan, {"pixel_values": _numpy(pixels), "vision_position_ids": positions})["image_features"]
    assert actual.shape == ((rows // 2) * (columns // 2), config.text_config["hidden_size"])
    _compare(actual, expected, 0.005)


@pytest.mark.gpu
@pytest.mark.trt
def test_qwen3_decoder_matches_official_empty_and_nonempty_compact_cache():
    torch = _dependencies()
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from families.lighton_ocr.graph import build_decoder_engine

    torch.manual_seed(93)
    config = _config()
    reference_config = Qwen3Config(**config.text_config)
    reference_config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(reference_config).eval().float().cuda()
    weights = {name.removeprefix("model."): _numpy(value) for name, value in model.state_dict().items()}
    plan = build_decoder_engine(config, weights, 16)
    cache = None
    actual_cache = {name: np.empty((1, 2, 0, 4), np.float32) for name in ("past_key", "past_value")}
    past = 0
    for length in (3, 1):
        embeddings = torch.randn(1, length, 8, dtype=torch.float32, device="cuda")
        positions = np.arange(past, past + length, dtype=np.int32)
        mask = np.where(np.arange(past + length)[None] <= positions[:, None], 0, -np.inf).astype(np.float32)
        with torch.inference_mode():
            reference = model(inputs_embeds=embeddings, past_key_values=cache, use_cache=True,
                              position_ids=torch.as_tensor(positions[None], device="cuda"), return_dict=True)
        cache = reference.past_key_values
        actual = _run(plan, {"inputs_embeds": _numpy(embeddings.squeeze(0)), "position_ids": positions,
                             "attention_mask": mask, **actual_cache})
        logits = _numpy(reference.logits[:, -1])
        _compare(actual["logits"], logits, 0.01)
        np.testing.assert_array_equal(actual["logits"].argmax(-1), logits.argmax(-1))
        for name, expected in (("key", cache.layers[0].keys), ("value", cache.layers[0].values)):
            observed = actual["present_" + name]
            _compare(observed, _numpy(expected), 0.005)
            if past:
                np.testing.assert_array_equal(observed[:, :, :past], actual_cache["past_" + name])
            actual_cache["past_" + name] = observed
        past += length
