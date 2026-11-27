# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free identity and default-task checks for stable_diffusion_3."""

from tensorrt_model_connect.model_support import ModelMetadata

from families.stable_diffusion_3.support import describe

# The component layout of a StableDiffusion3Pipeline snapshot such as SD3.5 Medium.
_SD35_MEDIUM_INDEX = {
    "_class_name": "StableDiffusion3Pipeline",
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "CLIPTextModelWithProjection"],
    "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
    "text_encoder_3": ["transformers", "T5EncoderModel"],
    "tokenizer": ["transformers", "CLIPTokenizer"],
    "tokenizer_2": ["transformers", "CLIPTokenizer"],
    "tokenizer_3": ["transformers", "T5TokenizerFast"],
    "transformer": ["diffusers", "SD3Transformer2DModel"],
    "vae": ["diffusers", "AutoencoderKL"],
}


def test_sd3_pipeline_resolves_to_this_family_with_text_to_image():
    support = describe(ModelMetadata({}, _SD35_MEDIUM_INDEX))
    assert support is not None
    assert support.default_task == "text_to_image"
    assert support.tasks == ("text_to_image",)
    assert support.default_precision == "bf16"


def test_other_diffusers_pipelines_are_not_claimed():
    for pipeline in ("StableDiffusionPipeline", "StableDiffusionXLPipeline", "FluxPipeline",
                     "StableDiffusion3Img2ImgPipeline", "StableDiffusion3ControlNetPipeline"):
        assert describe(ModelMetadata({}, {"_class_name": pipeline})) is None


def test_transformers_configs_are_not_claimed():
    assert describe(ModelMetadata({"model_type": "t5", "architectures": ["T5EncoderModel"]}, {})) is None
