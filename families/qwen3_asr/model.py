# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-ASR family build.

Checkpoint layout (Transformers ``Qwen3ASRForConditionalGeneration``):
  model.audio_tower            Qwen3ASREncoder: 3 strided Conv2d, 18 layers, d_model 896
  model.multi_modal_projector  896 -> 896, GELU, -> decoder width
  model.language_model         Qwen3 decoder; the LM head is tied to the token embedding

A bundle holds the audio encoder engine, the decoder engine (prefill and decode profiles over one
KV cache of ``max_sequence_length`` positions), the mel filterbank, a vocabulary byte table and
``runtime.json``. The runtime computes log-mel features, encodes the audio, fills the transcription
prompt and decodes greedily, as ``generate(do_sample=False)`` does.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

from .config import load_config, read_json
from .frontend import slaney_mel_filters

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter

DEFAULT_CAPACITY = 4096
# Rows of the decoder's key range are bound in multiples of this, to bound shape changes.
KEY_ROW_BUCKET = 64
# Activation memory per launch stays bounded whatever max_sequence_length is: the prompt is
# prefilled in chunks of at most this many tokens (prefill attention scores grow with chunk x
# cache length), and the encoder runs at most this many attention windows (64 s of audio) at a time.
PREFILL_CHUNK_TOKENS = 1024
ENCODER_WINDOWS_PER_LAUNCH = 8
# The official parser collapses runs longer than this (``_detect_and_fix_repetitions``).
REPETITION_THRESHOLD = 20
# ``Qwen3ASRFeatureExtractor`` builds its filters up to 8 kHz.
MEL_MAX_FREQUENCY = 8000.0


def _generation_defaults(model_dir: Path) -> int:
    """``max_new_tokens`` of the checkpoint's greedy generation configuration."""
    generation = read_json(model_dir / "generation_config.json")
    if generation.get("do_sample", False):
        raise ValueError("qwen3_asr implements greedy decoding; the checkpoint samples")
    value = int(generation.get("max_new_tokens", 0))
    if value < 1:
        raise ValueError("qwen3_asr checkpoint declares no max_new_tokens")
    return value


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one Qwen3-ASR speech-transcription bundle."""
    if request.task != "speech_transcription":
        raise ValueError("qwen3_asr supports only task=speech_transcription")
    if request.backend != "trt":
        raise ValueError("qwen3_asr supports only backend=trt")
    if request.precision not in {"bf16", "fp32"}:
        raise ValueError("qwen3_asr supports precision bf16 (the checkpoint's) or fp32")
    if request.max_batch_size != 1:
        raise NotImplementedError("qwen3_asr does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("qwen3_asr does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("qwen3_asr does not support context parallelism")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("qwen3_asr does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("qwen3_asr does not support fp32_layers")
    if request.dynamic_kv_cache:
        raise NotImplementedError("qwen3_asr does not support dynamic_kv_cache")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("qwen3_asr does not take image dimensions")
    if request.video_num_frames is not None:
        raise NotImplementedError("qwen3_asr does not support video_num_frames")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("qwen3_asr does not support MTP speculative decoding")
    if request.graph_transform is not None:
        raise NotImplementedError("qwen3_asr does not support graph transforms")
    model_dir = Path(request.model_dir)
    verbose = bool(request.verbose)
    capacity = int(request.max_sequence_length or DEFAULT_CAPACITY)

    # Heavy TensorRT / NumPy modules load only when a bundle is actually built.
    from . import tokens
    from .decoder_builder import build_decoder_plan
    from .encoder_builder import build_encoder_plan
    from .trt_ops import BF16, FP32
    from .weights import read_safetensors

    config = load_config(model_dir)
    max_output_tokens = _generation_defaults(model_dir)
    tokenizer = tokens.load_tokenizer(model_dir)
    prefix, suffix = tokens.prompt_pieces(tokenizer, config.audio_token_id)
    languages = tokens.language_suffixes(tokenizer)
    encoder = config.encoder
    longest_language = max(len(entry["ids"]) for entry in languages)
    # One full chunk of audio, the prompt, the longest forced language and one generated token.
    minimum = len(prefix) + len(suffix) + longest_language + encoder.tokens_per_chunk + 1
    if capacity < minimum:
        raise ValueError(f"qwen3_asr needs max_sequence_length >= {minimum}, got {capacity}")
    max_audio_tokens = capacity - len(prefix) - len(suffix) - 1
    encoder_windows = min(-(-max_audio_tokens // encoder.window_tokens), ENCODER_WINDOWS_PER_LAUNCH)
    prefill_tokens = min(capacity, PREFILL_CHUNK_TOKENS)
    compute = BF16 if request.precision == "bf16" else FP32

    weights = read_safetensors(model_dir)
    writer.set_header(family="qwen3_asr", task=request.task, backend=request.backend)
    with writer.open_section("audio_encoder.plan") as section:
        build_encoder_plan(
            encoder, weights, section, max_windows=encoder_windows, verbose=verbose, compute=compute
        )
    with writer.open_section("decoder.plan") as section:
        build_decoder_plan(
            config.decoder,
            weights,
            section,
            capacity=capacity,
            prefill_tokens=prefill_tokens,
            verbose=verbose,
            compute=compute,
        )
    del weights

    features = config.features
    filters = slaney_mel_filters(
        num_frequency_bins=features.n_fft // 2 + 1,
        num_mel_filters=features.num_mel_bins,
        max_frequency=MEL_MAX_FREQUENCY,
        sampling_rate=features.sampling_rate,
    )
    writer.add_bytes(
        "mel_filters.bin",
        struct.pack("<ii", filters.shape[0], filters.shape[1]) + filters.tobytes(),
    )
    writer.add_bytes(
        "vocab.bin", tokens.vocabulary_table(tokenizer, config.decoder.vocab_size)
    )
    decoder = config.decoder
    writer.add_json(
        "runtime.json",
        {
            "tensor_parallel_size": 1,
            "sample_rate": features.sampling_rate,
            "n_fft": features.n_fft,
            "hop_length": features.hop_length,
            "num_mel_bins": features.num_mel_bins,
            "min_samples": features.min_samples,
            "chunk_frames": encoder.chunk_frames,
            "tokens_per_chunk": encoder.tokens_per_chunk,
            "window_chunks": encoder.window_chunks,
            "encoder_windows_per_launch": encoder_windows,
            "prefill_chunk_tokens": prefill_tokens,
            "audio_embed_size": encoder.output_dim,
            "vocab_size": decoder.vocab_size,
            "num_layers": decoder.layers,
            "num_kv_heads": decoder.kv_heads,
            "head_dim": decoder.head_dim,
            "max_cache_length": capacity,
            "cache_dtype": request.precision,
            "key_row_bucket": KEY_ROW_BUCKET,
            "audio_token_id": config.audio_token_id,
            "prompt_prefix_ids": prefix,
            "prompt_suffix_ids": suffix,
            "stop_token_ids": list(config.eos_token_ids),
            "languages": languages,
            "asr_text_marker": tokens.ASR_TEXT_MARKER,
            "repetition_threshold": REPETITION_THRESHOLD,
            "default_max_output_tokens": max_output_tokens,
        },
    )
