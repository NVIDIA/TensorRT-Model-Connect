# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PaddleOCR-VL native fixed image profile and multimodal causal decoder."""

import json
from pathlib import Path

from .decoder_builder import build_decoder
from .vision_builder import build_vision


def build(request, writer):
    if request.task != "images_text_to_text" or request.backend != "trt":
        raise ValueError("PaddleOCR requires images_text_to_text with backend=trt")
    if (
        request.precision != "fp32"
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.max_batch_size != 1
        or request.quantization not in (None, "none")
        or request.fp32_layers
        or request.dynamic_kv_cache
        or request.video_num_frames is not None
        or request.mtp_seq_len is not None
    ):
        raise NotImplementedError(
            "PaddleOCR supports one fp32 image/question with a fixed image profile"
        )
    directory = Path(request.model_dir)
    from transformers import AutoConfig, AutoProcessor

    config = AutoConfig.from_pretrained(directory)
    cfg = config.text_config.to_dict()
    cfg["rope_theta"] = config.text_config.rope_parameters["rope_theta"]
    cfg["mrope_section"] = config.text_config.rope_parameters.get("mrope_section", [16, 24, 24])
    height, width = request.image_height or 280, request.image_width or 560
    length = request.max_sequence_length or 512
    if (
        config.model_type != "paddleocr_vl"
        or cfg["use_bias"]
        or cfg["hidden_act"] != "silu"
        or config.vision_config.patch_size != 14
        or config.vision_config.spatial_merge_size != 2
        or config.vision_config.hidden_act != "gelu_pytorch_tanh"
        or config.vision_config.rope_parameters["rope_type"] != "axial"
    ):
        raise ValueError("Unsupported PaddleOCR topology")
    if height <= 0 or width <= 0 or height % 28 or width % 28:
        raise ValueError("PaddleOCR image profile axes must be divisible by 28")
    image_sequence = height // 28 * (width // 28)
    if not image_sequence + 32 <= length <= cfg["max_position_embeddings"]:
        raise ValueError("PaddleOCR sequence profile needs room for image tokens and text")
    raw = json.loads((directory / "preprocessor_config.json").read_text())
    if (
        raw["resample"] != 3
        or raw["temporal_patch_size"] != 1
        or raw["image_mean"] != [0.5] * 3
        or raw["image_std"] != [0.5] * 3
    ):
        raise ValueError("Unsupported PaddleOCR image preprocessing")
    processor = AutoProcessor.from_pretrained(directory, use_fast=False)
    marker = "<|IMAGE_PLACEHOLDER|>"
    sentinel = "TRTMC_QUESTION_PLACEHOLDER"
    template = processor.apply_chat_template(
        [dict(role="user", content=[dict(type="image"), dict(type="text", text=sentinel)])],
        add_generation_prompt=True,
    )
    prefix, rest = template.split(marker)
    between, suffix = rest.split(sentinel)
    from safetensors import safe_open

    with safe_open(str(directory / "model.safetensors"), framework="pt", device="cpu") as handle:
        weights = {key: handle.get_tensor(key).float().numpy() for key in handle.keys()}
    writer.set_header(family="paddleocr_vl", task=request.task, backend=request.backend)
    writer.add_bytes(
        "vision.plan", build_vision(weights, config.vision_config, height, width, request.verbose)
    )
    writer.add_bytes("decoder.plan", build_decoder(weights, cfg, length, request.verbose))
    writer.add_bytes("tokenizer.json", processor.tokenizer.backend_tokenizer.to_str().encode())
    generation = json.loads((directory / "generation_config.json").read_text())
    writer.add_json(
        "runtime.json",
        dict(
            sequence_length=length,
            hidden_size=cfg["hidden_size"],
            vocab_size=cfg["vocab_size"],
            pad_id=generation["pad_token_id"],
            eos_ids=[generation["eos_token_id"]],
            image_id=config.image_token_id,
            image_sequence=image_sequence,
            image_height=height,
            image_width=width,
            min_pixels=raw["min_pixels"],
            max_pixels=raw["max_pixels"],
            image_token=marker,
            chat_prefix=prefix,
            between_image_text=between,
            chat_suffix=suffix,
        ),
    )
