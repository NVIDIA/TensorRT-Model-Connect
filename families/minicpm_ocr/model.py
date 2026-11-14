# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniCPM-OCR (``MiniCPMOcrForConditionalGeneration``) bundle builder for ``images_text_to_text``.

The bundle holds a vision engine (one image piece per call), a one-token hybrid decoder engine, the checkpoint
tokenizer and ``runtime.json`` with the decoder geometry and the image-processor settings the native runtime
reproduces.
"""

from __future__ import annotations

from pathlib import Path

from .checkpoint import Checkpoint, load_config
from .decoder import build_decoder_engine
from .checkpoint_contract import tokenizer_bytes
from .vision import build_vision_engine

FAMILY = "minicpm_ocr"
TASK = "images_text_to_text"
DEFAULT_SEQUENCE_LENGTH = 8192
DEFAULT_MAX_NEW_TOKENS = 1024
# Patches of one overview or slice. The checkpoint's slicing keeps a piece near
# scale_resolution^2 pixels; this bound covers aspect ratios up to about 1:1000.
MAX_PIECE_PATCHES = 2048
_TOKENS = {
    "image_pad": "<|image_pad|>",
    "image_start": "<image>",
    "image_end": "</image>",
    "slice_start": "<slice>",
    "slice_end": "</slice>",
    "image_id_start": "<image_id>",
    "image_id_end": "</image_id>",
}


def _image_settings(processor: dict, config) -> dict:
    """Validate the preprocessor contract the native image processor implements."""
    expected = {
        "image_processor_type": "MiniCPMOcrImageProcessor",
        "downsample_mode": "4x",
        "slice_mode": True,
        "use_image_id": True,
        "patch_size": config.vision.patch_size,
    }
    for key, value in expected.items():
        if processor.get(key, value) != value:
            raise ValueError(f"{FAMILY}: preprocessor {key}={processor.get(key)!r} is not supported")
    # Unsliced images are upscaled to scale_resolution; other modes are a different contract.
    if processor.get("small_image_upsize", "always") != "always":
        raise ValueError(f"{FAMILY}: only small_image_upsize='always' is supported")
    if processor.get("do_resize", True) is not True or processor.get("do_normalize", True) is not True:
        raise ValueError(f"{FAMILY}: the image processor must resize and normalize")
    if processor.get("do_rescale", True) is not True or float(processor.get("rescale_factor", 1 / 255)) != 1 / 255:
        raise ValueError(f"{FAMILY}: the image processor must rescale by 1/255")
    resample = processor.get("resample", 3)
    if resample != 3:  # PIL BICUBIC
        raise ValueError(f"{FAMILY}: only bicubic resampling is supported")
    mean = [float(v) for v in processor.get("image_mean", [0.5, 0.5, 0.5])]
    std = [float(v) for v in processor.get("image_std", [0.5, 0.5, 0.5])]
    if len(mean) != 3 or len(std) != 3 or min(std) <= 0:
        raise ValueError(f"{FAMILY}: invalid image normalization")
    return {
        "patch_size": config.vision.patch_size,
        "merge_size": config.merge_kernel[0],
        "scale_resolution": int(processor.get("scale_resolution", 448)),
        "max_slice_nums": int(processor.get("max_slice_nums", 9)),
        "slice_mode": True,
        "use_image_id": True,
        "position_grid": config.vision.position_grid,
        "max_piece_patches": MAX_PIECE_PATCHES,
        "image_mean": mean,
        "image_std": std,
    }


def build(request, writer) -> None:
    """Build one MiniCPM-OCR bundle; every shared request option is consumed or rejected."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("minicpm_ocr does not support dynamic_kv_cache")
    if request.task != TASK:
        raise ValueError(f"{FAMILY} supports only the {TASK} task")
    if request.backend != "trt":
        raise ValueError(f"{FAMILY} supports only the trt backend")
    precision = str(request.precision).lower()
    if precision not in {"fp16", "fp32"}:
        raise ValueError(f"{FAMILY} precision must be fp16 or fp32")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise ValueError(f"{FAMILY} runs on one GPU without tensor or context parallelism")
    if request.max_batch_size != 1:
        raise ValueError(f"{FAMILY} supports max_batch_size 1 only")
    if request.quantization is not None:
        raise ValueError(f"{FAMILY} does not support quantization")
    if request.fp32_layers:
        raise ValueError(f"{FAMILY} does not support fp32_layers")
    if request.image_height is not None or request.image_width is not None:
        raise ValueError(f"{FAMILY} sizes every image from its own aspect ratio; image_height/width are not used")
    if request.video_num_frames is not None:
        raise ValueError(f"{FAMILY} does not support video input")
    if request.mtp_seq_len is not None:
        raise ValueError(f"{FAMILY} has no multi-token prediction head")
    if request.graph_transform is not None:
        raise ValueError(f"{FAMILY} does not support graph transforms")

    model_dir = Path(request.model_dir)
    config = load_config(model_dir)
    capacity = int(request.max_sequence_length or DEFAULT_SEQUENCE_LENGTH)
    if not 1 < capacity <= config.text.max_position_embeddings:
        raise ValueError(f"{FAMILY} max_sequence_length must be in (1, {config.text.max_position_embeddings}]")
    image = _image_settings(config.processor, config)
    tokenizer = tokenizer_bytes(model_dir)
    checkpoint = Checkpoint(model_dir)
    verbose = bool(request.verbose)

    writer.set_header(family="minicpm_ocr", task=request.task, backend=request.backend)
    writer.add_bytes("vision.plan", build_vision_engine(checkpoint, config, precision, MAX_PIECE_PATCHES, verbose))
    writer.add_bytes("decoder.plan", build_decoder_engine(checkpoint, config, precision, capacity, verbose))
    text = config.text
    key_dim = text.linear_key_heads * text.linear_key_dim
    writer.add_json("runtime.json", {
        "schema_version": 1,
        "precision": precision,
        "hidden_size": text.hidden_size,
        "vocab_size": text.vocab_size,
        "max_cache_length": capacity,
        "linear_layers": text.layer_types.count("linear_attention"),
        "attention_layers": text.layer_types.count("full_attention"),
        "conv_dim": 2 * key_dim + text.linear_value_heads * text.linear_value_dim,
        "conv_kernel": text.linear_conv_kernel,
        "linear_heads": text.linear_value_heads,
        "linear_key_dim": text.linear_key_dim,
        "linear_value_dim": text.linear_value_dim,
        "kv_dim": text.num_kv_heads * text.head_dim,
        "eos_token_ids": list(config.eos_token_ids),
        "image_token_id": config.image_token_id,
        "default_max_new_tokens": min(DEFAULT_MAX_NEW_TOKENS, capacity),
        "image": image,
        "tokens": _TOKENS,
    })
    writer.add_bytes("tokenizer.json", tokenizer)
