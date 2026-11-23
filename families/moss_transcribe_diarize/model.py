# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a MOSS-Transcribe-Diarize bundle for the ``speech_transcription`` Task.

The bundle holds two engines and the host contract of the reference processor:

- ``audio.plan``: Whisper encoder + frame merge + adaptor for one 30 s chunk;
- ``text.plan``: the Qwen3 decoder with in-engine audio injection and a KV cache;
- ``runtime.json``: front-end geometry, prompt ids, time-marker rule, limits;
- ``mel_filters.bin``: the checkpoint feature extractor's mel filter bank;
- ``token_bytes.bin``: per-id decode bytes and skip flags.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .audio_encoder import audio_weight_names, build_audio_engine
from .checkpoint import Checkpoint
from .config import MODEL_TYPE, load_config
from .contract import runtime_json
from .decoder import build_text_engine, text_weight_names
from .graph import PRECISIONS
from .tokens import decode_table, mel_filters, prompt_contract

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter

# KV-cache rows when the request names no max_sequence_length: about six
# minutes of audio with its transcript.
DEFAULT_SEQUENCE_LENGTH = 8192


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    if request.task != "speech_transcription":
        raise ValueError(f"{MODEL_TYPE} supports only task=speech_transcription")
    if request.backend != "trt":
        raise ValueError(f"{MODEL_TYPE} supports only the trt backend")
    if request.precision not in PRECISIONS:
        raise NotImplementedError(
            f"{MODEL_TYPE} supports precision fp32 or bf16 (the checkpoint's own), "
            f"not {request.precision!r}"
        )
    if request.max_batch_size != 1:
        raise NotImplementedError(f"{MODEL_TYPE} supports only max_batch_size=1")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise NotImplementedError(f"{MODEL_TYPE} runs on a single GPU without parallelism")
    if request.quantization not in (None, "none"):
        raise NotImplementedError(
            f"{MODEL_TYPE} does not support quantization={request.quantization!r}"
        )
    if request.fp32_layers:
        raise NotImplementedError(f"{MODEL_TYPE} does not support fp32_layers")
    if request.dynamic_kv_cache:
        raise NotImplementedError("moss_transcribe_diarize does not support dynamic_kv_cache")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError(f"{MODEL_TYPE} takes audio, not image dimensions")
    if request.video_num_frames is not None:
        raise NotImplementedError(f"{MODEL_TYPE} takes audio, not video frames")
    if request.mtp_seq_len is not None:
        raise NotImplementedError(f"{MODEL_TYPE} has no multi-token-prediction head")

    model_dir = Path(request.model_dir)
    config = load_config(model_dir)
    checkpoint = Checkpoint(model_dir)
    expected = set(audio_weight_names(config)) | set(text_weight_names(config))
    if checkpoint.names() != expected:
        missing = sorted(expected - checkpoint.names())
        unused = sorted(checkpoint.names() - expected)
        raise ValueError(
            f"{MODEL_TYPE} checkpoint tensors differ from the supported layout: "
            f"missing={missing[:5]} unused={unused[:5]}"
        )
    capacity = int(request.max_sequence_length or DEFAULT_SEQUENCE_LENGTH)
    prompt = prompt_contract(model_dir, config)
    fixed_rows = len(prompt["prompt_prefix_ids"]) + len(prompt["prompt_suffix_ids"])
    if capacity <= fixed_rows + 1:
        raise ValueError(
            f"max_sequence_length={capacity} cannot hold the {fixed_rows}-token prompt, "
            "one audio token and one generated token"
        )

    writer.set_header(family="moss_transcribe_diarize", task=request.task, backend=request.backend)
    writer.add_bytes(
        "audio.plan",
        build_audio_engine(checkpoint, config, request.precision, verbose=request.verbose),
    )
    writer.add_bytes(
        "text.plan",
        build_text_engine(checkpoint, config, request.precision, capacity,
                          verbose=request.verbose),
    )
    writer.add_bytes("mel_filters.bin", mel_filters(model_dir))
    writer.add_bytes("token_bytes.bin", decode_table(model_dir, config.text.vocab_size))
    writer.add_json(
        "runtime.json", runtime_json(config, prompt, capacity, request.precision)
    )
