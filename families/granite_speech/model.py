# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one Granite Speech bundle: the audio engine, the decoder engine, the prompt and the vocabulary.

The bundle answers ``speech_transcription``. Its token budget is the decoder's KV capacity
(``max_sequence_length``, by default the checkpoint's ``max_position_embeddings``): the prompt,
one placeholder per projector output and every generated token must fit in it, so the longest
accepted clip follows from the capacity and is recorded for the runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

import tensorrt as trt

from . import config as config_module
from .audio import build_audio_network
from .checkpoint import Checkpoint
from .decoder import build_decoder_network
from .graph import Graph
from .prompt import INSTRUCTION, prompt_ids, vocabulary

_SUPPORTED_TASKS = ("speech_transcription",)
_PRECISIONS = {"bf16": trt.bfloat16, "fp32": trt.float32}


def _builder(verbose: bool):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    # TF32 would round FP32 products to 10 mantissa bits and hide graph errors from the FP32 oracle.
    config.clear_flag(trt.BuilderFlag.TF32)
    return builder, network, config


def _serialize(builder, network, config, what: str):
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"granite_speech {what} engine build failed")
    return plan


def _build_audio(ckpt: Checkpoint, cfg, work, limits: dict, verbose: bool):
    builder, network, config = _builder(verbose)
    graph = Graph(network, work)
    build_audio_network(graph, ckpt, cfg, limits["max_frames"])
    pad = cfg.front_end.n_fft
    profile = builder.create_optimization_profile()
    profile.set_shape("waveform", (limits["min_samples"] + pad,), (limits["opt_samples"] + pad,),
                      (limits["max_samples"] + pad,))
    config.add_optimization_profile(profile)
    return _serialize(builder, network, config, "audio")


def _build_decoder(ckpt: Checkpoint, cfg, work, capacity: int, opt_prompt: int, verbose: bool):
    builder, network, config = _builder(verbose)
    graph = Graph(network, work)
    build_decoder_network(graph, ckpt, cfg, capacity)
    text = cfg.text
    opt_prompt = min(opt_prompt, capacity - 1)
    # Profile 0 prefills the whole prompt with an empty cache; profile 1 decodes one token.
    for rows, past in (((1, opt_prompt, capacity), (0, 0, 0)), ((1, 1, 1), (1, opt_prompt, capacity - 1))):
        profile = builder.create_optimization_profile()
        for name in ("input_ids", "audio_mask"):
            profile.set_shape(name, *((value,) for value in rows))
        profile.set_shape("audio_embeds", *((value, text.hidden_size) for value in rows))
        for index in range(text.num_layers):
            for name in (f"past_key_{index}", f"past_value_{index}"):
                profile.set_shape(name, *((value, text.kv_width) for value in past))
        config.add_optimization_profile(profile)
    return _serialize(builder, network, config, "decoder")


def build(request, writer) -> None:
    """Build one Granite Speech bundle."""
    if request.task not in _SUPPORTED_TASKS:
        raise ValueError("granite_speech supports only task=" + ", ".join(_SUPPORTED_TASKS))
    if request.backend != "trt":
        raise ValueError("granite_speech supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("granite_speech does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("granite_speech transcribes one clip per request (max_batch_size=1)")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("granite_speech does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("granite_speech does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("granite_speech does not support video_num_frames")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("granite_speech takes audio, not an image size")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("granite_speech does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("granite_speech does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("granite_speech does not support MTP speculative decoding")
    precision = str(request.precision).lower()
    if precision not in _PRECISIONS:
        raise NotImplementedError(f"granite_speech builds {' or '.join(_PRECISIONS)}, not {precision}")

    model_dir = Path(request.model_dir)
    cfg = config_module.resolve(model_dir)
    capacity = int(request.max_sequence_length or cfg.text.max_position_embeddings)
    if capacity > cfg.text.max_position_embeddings:
        raise ValueError(f"granite_speech max_sequence_length {capacity} exceeds the checkpoint's "
                         f"max_position_embeddings {cfg.text.max_position_embeddings}")
    prompt, prefix, suffix = prompt_ids(model_dir, cfg.audio_token_id)
    limits = config_module.sample_limits(cfg, len(prefix) + len(suffix), capacity)
    vocab = vocabulary(model_dir, cfg.text.vocab_size)
    work = _PRECISIONS[precision]
    verbose = bool(request.verbose)

    writer.set_header(family="granite_speech", task=request.task, backend=request.backend)
    with Checkpoint(model_dir, cfg) as ckpt:
        if verbose:
            print(f"[granite_speech] building the {precision} audio engine", file=sys.stderr)
        plan = _build_audio(ckpt, cfg, work, limits, verbose)
        with writer.open_section("audio.plan") as section:
            section.write(memoryview(plan))
        del plan
        if verbose:
            print(f"[granite_speech] building the {precision} decoder engine", file=sys.stderr)
        opt_prompt = len(prefix) + len(suffix) + cfg.audio_tokens(limits["opt_samples"])
        plan = _build_decoder(ckpt, cfg, work, capacity, opt_prompt, verbose)
        with writer.open_section("decoder.plan") as section:
            section.write(memoryview(plan))
        del plan
    writer.add_json("runtime.json", {
        "precision": precision,
        "sample_rate": cfg.front_end.sample_rate,
        "n_fft": cfg.front_end.n_fft,
        "hop_length": cfg.front_end.hop_length,
        "window_size": cfg.projector.window_size,
        "num_queries": cfg.projector.num_queries,
        "min_samples": limits["min_samples"],
        "max_samples": limits["max_samples"],
        "capacity": capacity,
        "hidden_size": cfg.text.hidden_size,
        "num_layers": cfg.text.num_layers,
        "kv_width": cfg.text.kv_width,
        "vocab_size": cfg.text.vocab_size,
        "audio_token_id": cfg.audio_token_id,
        "eos_token_id": cfg.text.eos_token_id,
        "prompt_prefix_ids": prefix,
        "prompt_suffix_ids": suffix,
        "prompt": prompt,
        "instruction": INSTRUCTION,
    })
    writer.add_json("vocabulary.json", vocab)
