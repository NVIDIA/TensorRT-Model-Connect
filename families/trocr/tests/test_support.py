# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free ownership of TrOCR checkpoints."""

from families.trocr.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def _metadata(encoder: str = "vit", decoder: str = "trocr", **overrides) -> ModelMetadata:
    config = {
        "model_type": "vision-encoder-decoder",
        "architectures": ["VisionEncoderDecoderModel"],
        "encoder": {"model_type": encoder},
        "decoder": {"model_type": decoder},
    }
    config.update(overrides)
    return ModelMetadata(config=config, model_index={})


def test_vit_encoder_with_trocr_decoder_is_claimed() -> None:
    support = describe(_metadata())
    assert support is not None
    assert support.tasks == ("images_text_to_text",)
    assert support.default_task == "images_text_to_text"
    assert support.default_precision == "fp32"


def test_other_vision_encoder_decoder_models_are_not_claimed() -> None:
    # The generic model type is shared: Donut and Nougat pair a Swin encoder
    # with an mBART decoder, captioners pair ViT with GPT-2, and the small
    # TrOCR checkpoints use a DeiT encoder with a distillation token.
    for encoder, decoder in (("donut-swin", "mbart"), ("vit", "gpt2"), ("deit", "trocr"),
                             ("swin", "trocr"), ("vit", "bert")):
        assert describe(_metadata(encoder, decoder)) is None, (encoder, decoder)


def test_the_parts_alone_are_not_claimed() -> None:
    for model_type in ("trocr", "vit", "vision_encoder_decoder_model"):
        metadata = ModelMetadata(config={"model_type": model_type}, model_index={})
        assert describe(metadata) is None


def test_the_identity_is_the_whole_root_shape() -> None:
    assert describe(_metadata(architectures=["TrOCRForCausalLM"])) is None
    assert describe(_metadata(architectures=[])) is None
    assert describe(_metadata(encoder=None)) is None
    assert describe(_metadata(decoder="TROCR")) is None
    assert describe(ModelMetadata(config={"model_type": "vision-encoder-decoder"},
                                  model_index={})) is None
