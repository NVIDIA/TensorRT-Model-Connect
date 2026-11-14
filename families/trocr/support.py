# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact ownership of TrOCR: a ViT image encoder composed with the TrOCR text decoder.

TrOCR checkpoints are published as ``VisionEncoderDecoderModel`` with the
generic ``model_type`` ``vision-encoder-decoder``, which other image-to-text
models share (Donut, Nougat, ViT-GPT2 captioners). The owner is therefore
identified by the root config's own shape: that model type and architecture,
a ``vit`` encoder and a ``trocr`` decoder. The family reads one line of text
from one image; the text part of the request carries no instruction.
"""

from tensorrt_model_connect.model_support import FamilySupport

_SUPPORT = FamilySupport(
    tasks=("images_text_to_text",),
    default_task="images_text_to_text",
    # The checkpoint is stored in float32 and the fp32 engine reproduces the
    # reference's greedy tokens exactly (tests/test_e2e.py).
    default_precision="fp32",
)


def _model_type(config: object) -> str:
    if not isinstance(config, dict):
        return ""
    value = config.get("model_type", "")
    return value if isinstance(value, str) else ""


def describe(metadata):
    """Claim ``VisionEncoderDecoderModel`` checkpoints with a ViT encoder and a TrOCR decoder."""
    config = metadata.config
    if _model_type(config) != "vision-encoder-decoder":
        return None
    if "VisionEncoderDecoderModel" not in metadata.architectures:
        return None
    if _model_type(config.get("encoder")) != "vit":
        return None
    if _model_type(config.get("decoder")) != "trocr":
        return None
    return _SUPPORT
