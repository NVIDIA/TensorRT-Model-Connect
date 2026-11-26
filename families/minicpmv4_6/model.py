# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniCPM-V 4.6 bundle builder.

``MiniCPMV4_6ForConditionalGeneration`` pairs a SigLIP-style vision tower
(with a windowed token merger inside the encoder and an MLP projector) with a
Qwen3.5 hybrid decoder (Gated DeltaNet + gated attention). The bundle holds:

``engine.plan``   one-token decoder (``text_model``)
``prefill.plan``  chunked prefill that accepts vision feature rows (``text_model``)
``vision.plan``   per-slice vision tower, merger and projector (``vision``)
``runtime.json``  decoder geometry and generation ids
``vision.json``   image-processor settings, markup token ids and prompt recipes
``tokenizer.json``, ``chat_template.jinja`` and the processor configuration.
"""

from __future__ import annotations

import gc
from pathlib import Path

from .checkpoint_mapper import _open_safetensors
from .config import CheckpointConfig, capacity_rows, generation_ids, load_json, positive_when_stored
from .prompt import check_tokenizer_layout, image_metadata, token_count
from .rope import rope_positions, rope_tables
from .text_model import PREFILL_MAX_ROWS, build_text_engine, load_text_weights, text_runtime_config
from .vision import MAX_PATCHES, build_vision_engine, load_vision_weights

_PRECISIONS = ("fp16", "fp32")
_BUNDLE_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json")


def build(request, writer) -> None:
    """Build one MiniCPM-V 4.6 bundle; every request option is consumed or rejected."""
    if request.task != "images_text_to_text":
        raise ValueError("minicpmv4_6 supports only the images_text_to_text task")
    if request.backend != "trt":
        raise ValueError("minicpmv4_6 supports only the trt backend")
    if request.precision not in _PRECISIONS:
        raise ValueError(f"minicpmv4_6 precision must be one of {', '.join(_PRECISIONS)}")
    if type(request.tensor_parallel_size) is not int or request.tensor_parallel_size != 1:
        raise ValueError("minicpmv4_6 runs on one GPU; tensor_parallel_size must be 1")
    if type(request.context_parallel_size) is not int or request.context_parallel_size != 1:
        raise ValueError("minicpmv4_6 does not support context parallelism")
    if type(request.max_batch_size) is not int or request.max_batch_size != 1:
        raise ValueError("minicpmv4_6 serves one request at a time; max_batch_size must be 1")
    if type(request.verbose) is not bool:
        raise ValueError("minicpmv4_6 verbose must be a boolean")
    if request.quantization is not None:
        raise ValueError("minicpmv4_6 does not support quantization")
    if request.fp32_layers:
        raise ValueError("minicpmv4_6 does not support per-layer FP32 overrides")
    if request.dynamic_kv_cache:
        raise NotImplementedError("minicpmv4_6 does not support dynamic_kv_cache")
    if request.mtp_seq_len is not None:
        raise ValueError("minicpmv4_6 has no multi-token prediction head")
    if request.image_height is not None or request.image_width is not None:
        raise ValueError("minicpmv4_6 sizes every image slice at runtime; image dimensions are not build options")
    if request.video_num_frames is not None:
        raise ValueError("minicpmv4_6 does not support video input")

    model_dir = Path(request.model_dir)
    config = CheckpointConfig.from_dir(model_dir)
    capacity = capacity_rows(request.max_sequence_length, config.text.max_position_embeddings)
    precision = request.precision
    # Everything a bundle needs is checked before the first engine build. The RMSNorm epsilon is
    # an FP32 constant at both precisions.
    if not positive_when_stored(config.text.rms_norm_eps, "f"):
        raise ValueError(f"minicpmv4_6 rms_norm_eps {config.text.rms_norm_eps} vanishes in FP32")
    rope_tables(config.text, rope_positions(capacity))  # finite angles for every cached position
    for name in _BUNDLE_FILES:
        if not (model_dir / name).is_file():
            raise FileNotFoundError(f"MiniCPM-V 4.6 checkpoint is missing {name}")
    tokenizer_json = load_json(model_dir / "tokenizer.json")
    check_tokenizer_layout(tokenizer_json)
    eos_token_ids = generation_ids(model_dir, config, token_count(tokenizer_json))
    metadata = image_metadata(model_dir, config, tokenizer_json, capacity, precision, MAX_PATCHES)
    verbose = request.verbose

    writer.set_header(family="minicpmv4_6", task=request.task, backend=request.backend)
    readers = _open_safetensors(model_dir)
    text_weights = load_text_weights(readers, config.text)
    writer.add_bytes("engine.plan", build_text_engine(
        config.text, text_weights, capacity, precision=precision, verbose=verbose))
    writer.add_bytes("prefill.plan", build_text_engine(
        config.text, text_weights, capacity, precision=precision, prefill_rows=PREFILL_MAX_ROWS,
        verbose=verbose))
    del text_weights
    gc.collect()
    vision_weights = load_vision_weights(readers, config.vision, config.text.hidden_size)
    writer.add_bytes("vision.plan", build_vision_engine(config.vision, vision_weights, precision=precision,
                                                        verbose=verbose))
    del vision_weights
    writer.add_json("vision.json", metadata)
    writer.add_json("runtime.json", {
        **text_runtime_config(config.text),
        "precision": precision,
        "max_cache_length": capacity,
        "eos_token_id": eos_token_ids,
    })
    for name in _BUNDLE_FILES:
        writer.add_bytes(name, (model_dir / name).read_bytes())
