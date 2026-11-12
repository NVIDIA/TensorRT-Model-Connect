# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-OCR bundle builder.

A bundle holds three engines (vision tower with merger, decoder prefill,
one-token decoder), the BF16 token-embedding table the runtime gathers rows
from, the checkpoint's tokenizer.json, and ``runtime.json`` with every value
the native runtime needs to preprocess images, render the chat prompt, place
multimodal rotary positions and stop generation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from .checkpoint import Checkpoint, GlmOcrConfig, load_config
from .decoder_engine import build_decoder_engines
from .vision_engine import build_vision_engine

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter

_TASK = "images_text_to_text"
_PRECISIONS = ("bf16", "fp32")
# Prompt plus generated tokens of one request: room for the processor's largest
# image (6144 image tokens) and several thousand tokens of output.
_DEFAULT_MAX_SEQUENCE_LENGTH = 16384
_RUNTIME_FORMAT = 1
_EMBEDDING = "model.language_model.embed_tokens.weight"


def _patch_budget(height: int, width: int, config: GlmOcrConfig) -> int:
    """Vision patches of an image of at most ``height x width`` pixels after the processor's rounding."""
    p = config.processor
    factor = p.patch_size * p.merge_size
    budget = math.ceil(height / factor) * math.ceil(width / factor) * p.merge_size * p.merge_size
    return min(budget, config.max_image_patches)


def _runtime_contract(config: GlmOcrConfig, precision: str, max_rows: int, max_patches: int) -> dict:
    t, v, p = config.text, config.vision, config.processor
    return {
        "format": _RUNTIME_FORMAT,
        "precision": precision,
        "max_sequence_length": max_rows,
        "text": {
            "hidden_size": t.hidden_size,
            "num_layers": t.num_hidden_layers,
            "num_key_value_heads": t.num_key_value_heads,
            "head_dim": t.head_dim,
            "vocab_size": t.vocab_size,
            "rope_theta": t.rope_theta,
            "mrope_section": list(t.mrope_section),
        },
        "vision": {
            "patch_size": v.patch_size,
            "temporal_patch_size": v.temporal_patch_size,
            "merge_size": v.spatial_merge_size,
            "in_channels": v.in_channels,
            "head_dim": v.head_dim,
            "rope_theta": v.rope_theta,
            "out_hidden_size": v.out_hidden_size,
            "max_patches": max_patches,
        },
        "image": {
            "min_pixels": p.min_pixels,
            "max_pixels": p.max_pixels,
            "image_mean": list(p.image_mean),
            "image_std": list(p.image_std),
        },
        "tokens": dict(config.prompt_token_ids),
        "eos_token_ids": list(config.eos_token_ids),
        "special_token_ids": list(config.special_token_ids),
    }


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build a GLM-OCR images-and-text-to-text bundle."""
    if request.task != _TASK:
        raise ValueError(f"glm_ocr supports only task={_TASK}")
    if request.backend != "trt":
        raise ValueError("glm_ocr supports only the trt backend")
    if request.precision not in _PRECISIONS:
        raise ValueError(f"glm_ocr supports precision {_PRECISIONS}, not {request.precision!r}")
    if request.quantization not in (None, "none"):
        raise NotImplementedError(f"glm_ocr does not support quantization={request.quantization!r}")
    if request.max_batch_size != 1:
        raise ValueError("glm_ocr supports only max_batch_size=1")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise ValueError("glm_ocr runs on one GPU: tensor and context parallel sizes must be 1")
    if request.fp32_layers:
        raise NotImplementedError("glm_ocr does not support fp32_layers")
    if request.dynamic_kv_cache:
        raise NotImplementedError("glm_ocr does not support dynamic_kv_cache")
    if request.video_num_frames is not None:
        raise NotImplementedError("glm_ocr does not support video inputs")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("glm_ocr does not run the checkpoint's multi-token prediction layer")
    if (request.image_height is None) != (request.image_width is None):
        raise ValueError("glm_ocr takes image_height and image_width together (largest image size)")
    max_rows = int(request.max_sequence_length or _DEFAULT_MAX_SEQUENCE_LENGTH)
    if max_rows < 2:
        raise ValueError("glm_ocr max_sequence_length must hold a prompt and one generated token")

    model_dir = Path(request.model_dir)
    config = load_config(model_dir)
    max_patches = config.max_image_patches
    if request.image_height is not None:
        max_patches = _patch_budget(int(request.image_height), int(request.image_width), config)
    checkpoint = Checkpoint(model_dir, config)

    writer.set_header(family="glm_ocr", task=request.task, backend=request.backend)
    writer.add_json("runtime.json", _runtime_contract(config, request.precision, max_rows, max_patches))
    writer.add_bytes("vision.plan", build_vision_engine(
        checkpoint, config, request.precision, max_patches, verbose=request.verbose))
    prefill, decode = build_decoder_engines(
        checkpoint, config, request.precision, max_rows, verbose=request.verbose)
    writer.add_bytes("prefill.plan", prefill)
    writer.add_bytes("decode.plan", decode)
    table = checkpoint.raw(_EMBEDDING)
    if str(table.dtype) != "bfloat16":
        raise ValueError(f"glm_ocr expects the token embedding in BF16, found {table.dtype}")
    with writer.open_section("token_embedding.bf16") as section:
        section.write(table.tobytes())
    writer.add_bytes("tokenizer.json", (model_dir / "tokenizer.json").read_bytes())
