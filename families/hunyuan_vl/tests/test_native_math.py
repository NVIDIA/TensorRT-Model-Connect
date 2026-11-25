# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT decoder and vision engines against the reference HunYuanVL modules (tiny, random, FP32).

These component tests are not full-checkpoint qualification; ``test_e2e.py`` owns that.
"""

import re

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.trt]


@pytest.fixture(autouse=True)
def _exact_fp32_reference():
    """TF32 would perturb the FP32 reference products (NGC PyTorch enables it by default)."""
    torch = pytest.importorskip("torch")
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


# Reference module names -> released checkpoint names (the inverse of the reference loader's renames).
_RENAMES = (
    (r"^model\.language_model\.", "model."),
    (r"^model\.vision_tower\.", "vit."),
    (r"^vit\.layers\.(\d+)\.layer_norm1\.", r"vit.layers.\1.input_layernorm."),
    (r"^vit\.layers\.(\d+)\.layer_norm2\.", r"vit.layers.\1.post_attention_layernorm."),
    (r"\.mlp\.fc1\.", ".mlp.dense_h_to_4h."),
    (r"\.mlp\.fc2\.", ".mlp.dense_4h_to_h."),
    (r"\.patch_merger\.proj_conv\.", ".perceive.proj.0."),
    (r"\.patch_merger\.proj_out\.", ".perceive.proj.2."),
    (r"\.patch_merger\.", ".perceive."),
)


def _reference():
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensorrt")
    transformers = pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("HunYuanVL native mathematical tests require CUDA")
    if not hasattr(transformers, "HunYuanVLForConditionalGeneration"):
        pytest.skip("this transformers release has no HunYuanVL reference")
    torch.manual_seed(7)
    config = transformers.HunYuanVLConfig(
        text_config={
            "hidden_size": 64,
            "intermediate_size": 96,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 300,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-5,
            "hidden_act": "silu",
            "rope_parameters": {
                "rope_type": "xdrope",
                "alpha": 1000.0,
                "rope_theta": 10000.0,
                "factor": 1.0,
                "xdrope_section": [4, 4, 4, 4],
            },
        },
        vision_config={
            "hidden_size": 48,
            "intermediate_size": 80,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "patch_size": 4,
            "spatial_merge_size": 2,
            "max_image_size": 32,
            "hidden_act": "gelu",
            "rms_norm_eps": 1e-5,
        },
        image_token_id=290,
        im_start_id=288,
        im_end_id=289,
        tie_word_embeddings=True,
    )
    model = transformers.HunYuanVLForConditionalGeneration(config).eval().float().cuda()
    with torch.no_grad():  # Non-unit norm scales and nonzero biases exercise every parameter.
        for name, parameter in model.named_parameters():
            if (
                parameter.dim() == 1
                and name.endswith(".weight")
                and ("norm" in name or "_rms" in name)
            ):
                parameter.uniform_(0.8, 1.2)
            elif name.endswith(".bias"):
                parameter.normal_(0, 0.05)
    weights = {}
    for name, tensor in model.state_dict().items():
        for pattern, replacement in _RENAMES:
            name = re.sub(pattern, replacement, name)
        weights[name] = tensor.detach().float().cpu().numpy()
    raw = config.to_dict()
    raw["architectures"] = ["HunYuanVLForConditionalGeneration"]
    from families.hunyuan_vl.config import ModelConfig

    return torch, model, ModelConfig.from_dict(raw), weights


def _execute(plan, inputs, bindings=None):
    import tensorrt as trt
    import torch

    dtypes = {trt.float32: torch.float32, trt.bfloat16: torch.bfloat16, trt.int32: torch.int32}
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    tensors = {
        name: torch.from_numpy(np.ascontiguousarray(value)).cuda() for name, value in inputs.items()
    }
    tensors.update(bindings or {})
    for name, tensor in tensors.items():
        assert context.set_input_shape(name, tuple(tensor.shape)), name
        assert context.set_tensor_address(name, tensor.data_ptr())
    assert context.infer_shapes() == []
    outputs = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            outputs[name] = torch.empty(
                tuple(context.get_tensor_shape(name)),
                dtype=dtypes[engine.get_tensor_dtype(name)],
                device="cuda",
            )
            assert context.set_tensor_address(name, outputs[name].data_ptr())
    stream = torch.cuda.current_stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return outputs


def _corners(rows, cols, edge):
    """The family runtime's bilinear corner rows and FP32 weights, in 2x2 merge-block order."""

    def axis(size):
        scale = np.float32(edge) / np.float32(size)
        # Fused FP32 multiply-add, evaluated exactly in FP64 and rounded once.
        source = (np.float64(scale) * (np.arange(size) + 0.5) - 0.5).astype(np.float32)
        source = np.maximum(source, np.float32(0))
        low = source.astype(np.int32)
        take = (source - low).astype(np.float32)
        return low, np.where(low < edge - 1, low + 1, low), np.float32(1) - take, take

    ylow, yhigh, ykeep, ytake = axis(rows)
    xlow, xhigh, xkeep, xtake = axis(cols)
    ids, lambdas, order = [], [], []
    for by in range(0, rows, 2):
        for bx in range(0, cols, 2):
            for dy in (0, 1):
                for dx in (0, 1):
                    r, c = by + dy, bx + dx
                    order.append(r * cols + c)
                    ids.append(
                        [1 + y * edge + x for y in (ylow[r], yhigh[r]) for x in (xlow[c], xhigh[c])]
                    )
                    lambdas.append([ykeep[r], ytake[r], xkeep[c], xtake[c]])
    return np.array(ids, np.int32), np.array(lambdas, np.float32), np.array(order)


@pytest.mark.parametrize("rows, cols", [(2, 2), (4, 6), (10, 8)])
def test_vision_engine_matches_reference_tower(rows, cols):
    torch, model, config, weights = _reference()
    from families.hunyuan_vl.graph import build_visual

    plan = build_visual(config, weights, 128, precision="fp32")
    patch_width = 3 * config.patch_size**2
    pixels = (
        np.random.default_rng(rows * cols)
        .normal(size=(rows * cols, patch_width))
        .astype(np.float32)
    )
    with torch.no_grad():
        expected = (
            model.model.vision_tower(
                torch.from_numpy(pixels).cuda(), grid_thw=torch.tensor([[1, rows, cols]]).cuda()
            )
            .pooler_output[0]
            .cpu()
            .numpy()
        )
    ids, lambdas, order = _corners(rows, cols, config.position_edge)
    actual = (
        _execute(
            plan, {"pixel_values": pixels[order], "position_ids": ids, "position_lambdas": lambdas}
        )["image_features"]
        .cpu()
        .numpy()
    )
    cells = rows // 2 * (cols // 2)
    assembled = [actual[cells + 1]]
    for y in range(rows // 2):
        assembled.extend(actual[y * (cols // 2) : (y + 1) * (cols // 2)])
        assembled.append(actual[cells])
    assembled.append(actual[cells + 2])
    np.testing.assert_allclose(np.stack(assembled), expected, rtol=2e-4, atol=2e-4)


def _rope(config, positions):
    half = config.head_dim // 2
    axis = np.repeat(np.arange(len(config.rope_sections)), 2 * np.array(config.rope_sections))
    frequencies = np.array(config.inverse_frequencies(), np.float32)[
        np.arange(config.head_dim) % half
    ]
    angles = positions[axis].T.astype(np.float32) * frequencies  # [tokens, head_dim]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def test_decoder_engine_matches_reference_language_model():
    torch, model, config, weights = _reference()
    from families.hunyuan_vl.graph import build_decoder

    capacity = 16
    plan = build_decoder(config, weights, capacity, max_input_length=8, precision="fp32")
    rng = np.random.default_rng(11)
    total = 11
    embeds = rng.normal(size=(total, config.hidden_size)).astype(np.float32)
    # Sequence axis plus image-like column/row/ordinal axes over a grid span.
    positions = np.tile(np.arange(total), (4, 1))
    positions[1:, 3:9] = [[0, 1, 2, 0, 1, 2], [0, 0, 0, 1, 1, 1], [0] * 6]
    with torch.no_grad():
        hidden = model.model.language_model(
            inputs_embeds=torch.from_numpy(embeds)[None].cuda(),
            position_ids=torch.from_numpy(positions)[:, None].cuda(),
        ).last_hidden_state
        expected = model.lm_head(hidden)[0].cpu().numpy()
    cos, sin = _rope(config, positions)
    caches = {
        f"cache_{kind}_{layer}": torch.zeros(
            config.num_key_value_heads, capacity, config.head_dim, device="cuda"
        )
        for layer in range(config.num_hidden_layers)
        for kind in ("k", "v")
    }
    offset = 0
    for count in (6, 4, 1):  # chunked prefill, then a one-token decode step
        mask = np.full((1, count, capacity + count), np.finfo(np.float32).min, np.float32)
        mask[:, :, :offset] = 0
        mask[0, :, capacity:] = np.where(
            np.tril(np.ones((count, count))) > 0, 0, np.finfo(np.float32).min
        )
        span = slice(offset, offset + count)
        outputs = _execute(
            plan,
            {
                "inputs_embeds": embeds[span],
                "rope_cos": cos[span],
                "rope_sin": sin[span],
                "attention_mask": mask,
            },
            caches,
        )
        np.testing.assert_allclose(
            outputs["logits"][0].cpu().numpy(), expected[offset + count - 1], rtol=2e-4, atol=2e-4
        )
        for layer in range(config.num_hidden_layers):
            for kind in ("k", "v"):
                caches[f"cache_{kind}_{layer}"][:, span] = outputs[f"present_{kind}_{layer}"]
        offset += count
