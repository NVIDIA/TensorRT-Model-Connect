# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free Chroma identity and default-task policy."""

from families.chroma.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def _pipeline(name: str) -> ModelMetadata:
    return ModelMetadata(config={}, model_index={"_class_name": name})


def test_chroma_pipeline_is_text_to_image_in_bf16():
    support = describe(_pipeline("ChromaPipeline"))
    assert support is not None
    assert support.tasks == ("text_to_image",)
    assert support.default_task == "text_to_image"
    # The checkpoint stores BF16; FP32 is the parity build.
    assert support.default_precision == "bf16"


def test_other_chroma_and_flux_pipelines_are_not_claimed():
    # Img2img and inpainting need their own Task contracts; Radiance denoises
    # pixels instead of VAE latents; FLUX.1 keeps per-block modulation and CLIP.
    for name in (
        "ChromaImg2ImgPipeline",
        "ChromaInpaintPipeline",
        "ChromaRadiancePipeline",
        "FluxPipeline",
        "Flux2Pipeline",
    ):
        assert describe(_pipeline(name)) is None


def test_a_bare_transformer_config_is_not_claimed():
    # Only the complete pipeline (text encoder, tokenizer, VAE) can be built.
    metadata = ModelMetadata(config={"_class_name": "ChromaTransformer2DModel"}, model_index={})
    assert describe(metadata) is None
