# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PaddleOCR-VL native fixed image profile and multimodal causal decoder."""

import json
from pathlib import Path

from .decoder_builder import build_decoder
from .vision_builder import build_vision


def build(request, writer):
    if request.dynamic_kv_cache:
        raise NotImplementedError("paddleocr_vl does not support dynamic_kv_cache")
    if request.task != "images_text_to_text" or request.backend != "trt":
        raise ValueError("PaddleOCR requires images_text_to_text with backend=trt")
    if (
        request.precision != "fp32"
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.max_batch_size != 1
        or request.quantization not in (None, "none")
        or request.fp32_layers
        or request.video_num_frames is not None
        or request.mtp_seq_len is not None
    ):
        raise NotImplementedError(
            "PaddleOCR supports one fp32 image/question with a fixed image profile"
        )
    directory = Path(request.model_dir)
    raw_config = json.loads((directory / "config.json").read_text())
    if not isinstance(raw_config, dict) or raw_config.get("model_type") != "paddleocr_vl":
        raise ValueError("PaddleOCR requires model_type=paddleocr_vl")
    from transformers import AutoConfig, AutoProcessor

    config = AutoConfig.from_pretrained(directory, trust_remote_code=False)
    vision_raw = raw_config.get("vision_config", {})
    vision_rope = vision_raw.get("rope_parameters")
    if vision_rope is None:
        vision_rope = vision_raw.get("rope_scaling")
    if vision_rope is None:
        vision_rope = {}
    if not isinstance(vision_rope, dict):
        raise ValueError("PaddleOCR requires axial vision rotary parameters")
    vision_rope_type = vision_rope.get("rope_type", vision_rope.get("type", "axial"))
    vision_theta = vision_rope.get("rope_theta", vision_raw.get("rope_theta", 10000.0))
    if vision_raw.get("head_dim") not in (
        None,
        config.vision_config.hidden_size // config.vision_config.num_attention_heads,
    ):
        raise ValueError("PaddleOCR requires vision head dimensions to match its hidden size")
    if (
        vision_rope_type not in ("axial", "default")
        or type(vision_theta) not in (int, float)
        or not 0 < vision_theta < float("inf")
    ):
        raise ValueError("PaddleOCR requires finite positive axial rotary frequencies")
    cfg = config.text_config.to_dict()
    cfg["rope_theta"] = config.text_config.rope_parameters["rope_theta"]
    cfg["mrope_section"] = config.text_config.rope_parameters.get("mrope_section", [16, 24, 24])
    height, width = request.image_height or 280, request.image_width or 560
    length = request.max_sequence_length or 512
    if (
        config.model_type != "paddleocr_vl"
        or config.text_config.rope_parameters.get("rope_type", "default") != "default"
        or cfg["use_bias"]
        or cfg["hidden_act"] != "silu"
        or config.vision_config.patch_size != 14
        or config.vision_config.spatial_merge_size != 2
        or config.vision_config.hidden_act != "gelu_pytorch_tanh"
    ):
        raise ValueError("Unsupported PaddleOCR topology")
    if height <= 0 or width <= 0 or height % 28 or width % 28:
        raise ValueError("PaddleOCR image profile axes must be divisible by 28")
    image_sequence = height // 28 * (width // 28)
    # The native decoder materializes dense causal attention for this profile.
    if not image_sequence + 32 <= length <= min(4096, cfg["max_position_embeddings"]):
        raise ValueError("PaddleOCR sequence profile needs room for image tokens and text")
    raw = json.loads((directory / "preprocessor_config.json").read_text())
    if (
        not raw.get("do_normalize", True)
        or raw["resample"] != 3
        or raw["temporal_patch_size"] != 1
        or raw["image_mean"] != [0.5] * 3
        or raw["image_std"] != [0.5] * 3
    ):
        raise ValueError("Unsupported PaddleOCR image preprocessing")
    area = height * width
    minimum, maximum = raw["min_pixels"], raw["max_pixels"]
    # Resize rounds each axis to 28 pixels. The ceil/floor branches can cross
    # a pixel limit, so retain those boundary intervals as reachable profiles.
    if not (
        minimum <= area <= maximum
        or area <= maximum < (height + 28) * (width + 28)
        or (height - 28) * (width - 28) < minimum <= area
    ):
        raise ValueError("PaddleOCR image profile cannot be produced by checkpoint resize")
    generation = json.loads((directory / "generation_config.json").read_text())
    eos = generation["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]
    if not eos or any(
        type(value) is not int or not 0 <= value < cfg["vocab_size"] for value in eos
    ):
        raise ValueError("PaddleOCR requires a nonempty scalar or flat list of vocabulary EOS IDs")
    processor = AutoProcessor.from_pretrained(directory, use_fast=False, trust_remote_code=False)
    marker = "<|IMAGE_PLACEHOLDER|>"
    sentinel = "<|QUESTION_PLACEHOLDER|>"
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
        "vision.plan",
        build_vision(
            weights,
            config.vision_config,
            height,
            width,
            request.verbose,
            rotary_theta=vision_theta,
        ),
    )
    writer.add_bytes("decoder.plan", build_decoder(weights, cfg, length, request.verbose))
    writer.add_bytes("tokenizer.json", processor.tokenizer.backend_tokenizer.to_str().encode())
    writer.add_json(
        "runtime.json",
        dict(
            sequence_length=length,
            hidden_size=cfg["hidden_size"],
            vocab_size=cfg["vocab_size"],
            pad_id=generation["pad_token_id"],
            eos_ids=eos,
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
