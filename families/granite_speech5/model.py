# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one Granite Speech 5 CTC bundle: the encoder engine, its limits and the CTC vocabulary.

The bundle answers ``speech_transcription``. ``max_sequence_length`` is the longest clip in samples
(default 30 s); every clip from one CTC frame up to it is accepted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import tensorrt as trt

from . import config as config_module
from .checkpoint import Checkpoint
from .encoder import build_network
from .graph import Graph
from .vocabulary import read_pieces, read_vocabulary

_SUPPORTED_TASKS = ("speech_transcription",)
_PRECISIONS = {"bf16": trt.bfloat16, "fp32": trt.float32}


def build_engine(ckpt: Checkpoint, cfg, limits: dict, precision: str, verbose: bool = False) -> bytes:
    """Serialized encoder engine for one precision and waveform profile."""
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    builder_config.builder_optimization_level = 3
    # TF32 would round FP32 products to 10 mantissa bits: the front end and the FP32 parts of a
    # BF16 engine are computed in FP32 on purpose.
    builder_config.clear_flag(trt.BuilderFlag.TF32)
    # The graph holds the host buffers its constants point to: it must outlive the engine build.
    graph = Graph(network, _PRECISIONS[precision])
    build_network(graph, ckpt, cfg, limits["max_waveform"])
    profile = builder.create_optimization_profile()
    profile.set_shape("waveform", (limits["min_waveform"],), (limits["opt_waveform"],), (limits["max_waveform"],))
    builder_config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, builder_config)
    del graph
    if plan is None:
        raise RuntimeError("granite_speech5 encoder engine build failed")
    return plan


def build(request, writer) -> None:
    """Build one Granite Speech 5 CTC bundle."""
    if request.task not in _SUPPORTED_TASKS:
        raise ValueError("granite_speech5 supports only task=" + ", ".join(_SUPPORTED_TASKS))
    if request.backend != "trt":
        raise ValueError("granite_speech5 supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("granite_speech5 does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("granite_speech5 transcribes one clip per request (max_batch_size=1)")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("granite_speech5 does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("granite_speech5 does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("granite_speech5 does not support video_num_frames")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("granite_speech5 takes audio, not an image size")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("granite_speech5 does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("granite_speech5 does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("granite_speech5 does not support MTP speculative decoding")
    precision = str(request.precision).lower()
    if precision not in _PRECISIONS:
        raise NotImplementedError(f"granite_speech5 builds {' or '.join(_PRECISIONS)}, not {precision}")
    model_dir = Path(request.model_dir)
    pieces = read_pieces(model_dir)
    cfg = config_module.resolve(model_dir, pieces)
    max_samples = request.max_sequence_length
    limits = config_module.sample_limits(cfg, None if max_samples is None else max_samples)
    vocabulary = read_vocabulary(model_dir, cfg.vocab_size, cfg.blank_id)
    verbose = bool(request.verbose)

    writer.set_header(family="granite_speech5", task=request.task, backend=request.backend)
    with Checkpoint(model_dir, cfg) as ckpt:
        if verbose:
            print(f"[granite_speech5] building the {precision} encoder engine", file=sys.stderr)
        plan = build_engine(ckpt, cfg, limits, precision, verbose)
    with writer.open_section("encoder.plan") as section:
        section.write(memoryview(plan))
    del plan
    fe = cfg.front_end
    writer.add_json("runtime.json", {
        "precision": precision,
        "sample_rate": fe.sample_rate,
        "n_fft": fe.n_fft,
        "hop_length": fe.hop_length,
        "frame_stacking": config_module.FRAME_STACKING,
        "subsampling_layers": len(cfg.encoder.subsample_layers),
        "min_samples": limits["min_samples"],
        "max_samples": limits["max_samples"],
        "vocab_size": cfg.vocab_size,
    })
    writer.add_json("vocabulary.json", vocabulary.to_json())
