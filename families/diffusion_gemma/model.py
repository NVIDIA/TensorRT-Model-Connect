# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the exact checkpoint's text encoder and block diffusion denoiser."""

from __future__ import annotations

import gc
import json
from pathlib import Path

from .config import Config
from .weights import Checkpoint


def _generation_config(directory: Path, config: Config) -> dict:
    document = json.loads((directory / "generation_config.json").read_text())
    if not isinstance(document, dict):
        raise ValueError("diffusion_gemma generation_config.json must be an object")
    if document.get("sampler_config", {}).get("_cls_name") != "EntropyBoundSamplerConfig":
        raise ValueError("diffusion_gemma requires its entropy-bound sampler")
    eos = document.get("eos_token_id")
    if not isinstance(eos, list) or not eos or any(
        type(value) is not int or not 0 <= value < config.text_config["vocab_size"] for value in eos
    ):
        raise ValueError("diffusion_gemma generation config requires valid EOS token IDs")
    return document


def build(request, writer):
    if request.family != "diffusion_gemma" or request.task != "text_continuation":
        raise ValueError("diffusion_gemma builds the text_continuation task")
    if request.backend != "trt":
        raise ValueError("diffusion_gemma currently requires the TensorRT backend")
    if request.precision != "bf16":
        raise ValueError("diffusion_gemma requires the checkpoint's BF16 compute precision")
    if (request.tensor_parallel_size, request.context_parallel_size, request.max_batch_size) != (1, 1, 1):
        raise ValueError("diffusion_gemma supports TP1, CP1, and batch size 1")
    if request.quantization not in {None, "none"}:
        raise ValueError("diffusion_gemma does not implement quantization")
    if request.dynamic_kv_cache:
        raise NotImplementedError("diffusion_gemma does not support dynamic_kv_cache")
    if request.fp32_layers or request.mtp_seq_len is not None:
        raise ValueError("diffusion_gemma does not support layer precision overrides or MTP")
    if any(value is not None for value in (request.image_height, request.image_width, request.video_num_frames)):
        raise ValueError("diffusion_gemma text continuation does not accept media build options")

    directory = Path(request.model_dir)
    config = Config.from_dir(directory)
    generation = _generation_config(directory, config)
    capacity = request.max_sequence_length if request.max_sequence_length is not None else 1024
    if type(capacity) is not int or not config.canvas_length < capacity <= config.text_config["max_position_embeddings"]:
        raise ValueError("diffusion_gemma sequence capacity must fit a prompt and the complete diffusion canvas")
    tokenizer = (directory / "tokenizer.json").read_bytes()
    checkpoint = Checkpoint(directory, config)
    checkpoint.validate()
    from .graph import build_embedding_engine, build_layer_engine, build_output_engine

    writer.set_header(family="diffusion_gemma", task=request.task, backend=request.backend)
    writer.add_bytes("tokenizer.json", tokenizer)
    for name in ("tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"):
        path = directory / name
        if path.is_file():
            writer.add_bytes(name, path.read_bytes())
    writer.add_json("runtime.json", {
        "format_version": 1,
        "text_config": config.text_config,
        "canvas_length": config.canvas_length,
        "max_sequence_length": capacity,
        "precision": request.precision,
        "generation_config": generation,
        "unsupported_token_ids": sorted({
            config.raw[key] for key in ("image_token_id", "boi_token_id", "eoi_token_id")
            if type(config.raw.get(key)) is int
        }),
    })
    options = {
        "precision": request.precision,
        "max_sequence_length": capacity,
        "verbose": request.verbose,
    }
    print("diffusion_gemma: building token embedding and self-conditioning engine", flush=True)
    weights = checkpoint.embedding()
    writer.add_bytes("embedding.plan", build_embedding_engine(config.text_config, weights, **options))
    del weights
    gc.collect()
    for index in range(config.text_config["num_hidden_layers"]):
        print(f"diffusion_gemma: building transformer layer {index + 1}/{config.text_config['num_hidden_layers']}", flush=True)
        weights = checkpoint.layer(index)
        writer.add_bytes(f"layer_{index}.plan", build_layer_engine(config.text_config, weights, index, **options))
        del weights
        gc.collect()
    print("diffusion_gemma: building output and soft self-conditioning engine", flush=True)
    weights = checkpoint.output()
    writer.add_bytes("output.plan", build_output_engine(config.text_config, weights, **options))
