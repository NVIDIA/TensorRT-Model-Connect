# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An FP16 InternViT engine computes its attention and MLP blocks in FP16 and matches FP32."""

from __future__ import annotations

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")

from families.internvl import internvit_vision_builder as vision_module  # noqa: E402
from families.internvl.internvit_vision_builder import build_internvit_vision_engine  # noqa: E402

IMAGE, PATCH, EMBED, HEADS, MLP, LAYERS, LLM = 56, 14, 64, 4, 128, 2, 48


def _weights(rng: np.random.Generator) -> dict[str, np.ndarray]:
    def rand(*shape: int, scale: float = 0.05) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    patches = (IMAGE // PATCH) ** 2
    weights = {
        "vision_tower.embeddings.patch_embeddings.projection.weight": rand(EMBED, 3, PATCH, PATCH),
        "vision_tower.embeddings.patch_embeddings.projection.bias": rand(EMBED),
        "vision_tower.embeddings.cls_token": rand(1, 1, EMBED),
        "vision_tower.embeddings.position_embeddings": rand(1, patches + 1, EMBED),
    }
    for layer in range(LAYERS):
        prefix = f"vision_tower.encoder.layer.{layer}"
        weights.update(
            {
                f"{prefix}.layernorm_before.weight": 1.0 + rand(EMBED),
                f"{prefix}.layernorm_before.bias": rand(EMBED),
                f"{prefix}.layernorm_after.weight": 1.0 + rand(EMBED),
                f"{prefix}.layernorm_after.bias": rand(EMBED),
                f"{prefix}.lambda_1": 0.1 + rand(EMBED),
                f"{prefix}.lambda_2": 0.1 + rand(EMBED),
                f"{prefix}.mlp.fc1.weight": rand(MLP, EMBED),
                f"{prefix}.mlp.fc1.bias": rand(MLP),
                f"{prefix}.mlp.fc2.weight": rand(EMBED, MLP),
                f"{prefix}.mlp.fc2.bias": rand(EMBED),
            }
        )
        for name in ("q_proj", "k_proj", "v_proj", "projection_layer"):
            weights[f"{prefix}.attention.{name}.weight"] = rand(EMBED, EMBED)
            weights[f"{prefix}.attention.{name}.bias"] = rand(EMBED)
    concat = 4 * EMBED
    weights.update(
        {
            "multi_modal_projector.layer_norm.weight": 1.0 + rand(concat),
            "multi_modal_projector.layer_norm.bias": rand(concat),
            "multi_modal_projector.linear_1.weight": rand(LLM, concat),
            "multi_modal_projector.linear_1.bias": rand(LLM),
            "multi_modal_projector.linear_2.weight": rand(LLM, LLM),
            "multi_modal_projector.linear_2.bias": rand(LLM),
        }
    )
    return weights


def _features(plan: bytes, pixels: torch.Tensor) -> np.ndarray:
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    out = torch.empty(tuple(engine.get_tensor_shape("image_features")), device="cuda")
    context.set_tensor_address("pixel_values", pixels.data_ptr())
    context.set_tensor_address("image_features", out.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return out.cpu().numpy()


def _matmul_input_types(network) -> list:
    layers = (network.get_layer(i) for i in range(network.num_layers))
    return [
        layer.get_input(0).dtype for layer in layers if layer.type == trt.LayerType.MATRIX_MULTIPLY
    ]


@pytest.mark.gpu
@pytest.mark.trt
def test_fp16_blocks_match_the_fp32_engine(monkeypatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    rng = np.random.default_rng(0)
    weights = _weights(rng)
    top = {"downsample_ratio": 0.5, "vision_feature_layer": -1}
    vision = {
        "hidden_size": EMBED,
        "num_attention_heads": HEADS,
        "num_hidden_layers": LAYERS,
        "intermediate_size": MLP,
        "patch_size": PATCH,
        "layer_norm_eps": 1e-6,
    }
    pixels = torch.from_numpy(rng.standard_normal((3, IMAGE, IMAGE)).astype(np.float32)).cuda()

    contexts = []
    real_context = vision_module.create_builder_context

    def record(**kwargs):
        contexts.append(real_context(**kwargs))
        return contexts[-1]

    monkeypatch.setattr(vision_module, "create_builder_context", record)
    plans = {
        precision: build_internvit_vision_engine(
            top, vision, weights, fixed_image_size=IMAGE, precision=precision
        )
        for precision in ("fp32", "fp16")
    }
    fp32, fp16 = (_features(plans[precision], pixels) for precision in ("fp32", "fp16"))

    assert fp16.shape == fp32.shape == ((IMAGE // PATCH) ** 2 // 4, LLM)
    assert np.isfinite(fp16).all()
    assert np.linalg.norm(fp16 - fp32) / np.linalg.norm(fp32) < 1e-2
    # q, k, v, o, fc1 and fc2 of every layer run in FP16; the projector's two matmuls stay FP32.
    assert _matmul_input_types(contexts[0].network) == [trt.float32] * (6 * LAYERS + 2)
    assert (
        _matmul_input_types(contexts[1].network) == [trt.float16] * (6 * LAYERS) + [trt.float32] * 2
    )


def test_unknown_precision_is_rejected() -> None:
    with pytest.raises(ValueError, match="int8"):
        build_internvit_vision_engine({}, {}, {}, precision="int8")
