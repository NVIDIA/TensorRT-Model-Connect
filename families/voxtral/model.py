# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voxtral speech transcription bundle builder.

Voxtral pairs a Whisper-style audio encoder with a Llama decoder. Each 30 s
chunk of 16 kHz audio becomes 3000 log-mel frames, 1500 encoder frames and,
after stacking four frames per row and a two-layer projector, 375 decoder
embeddings that replace the ``[AUDIO]`` placeholders of the transcription
prompt. The supported checkpoints are MLX affine-quantized exports; their
weights are dequantized exactly at build time (see ``checkpoint.py``).

Bundle sections: ``encoder.plan`` (audio tower and projector), ``decoder.plan``
(prefill profile 0, one-row decode profile 1, explicit KV caches),
``embed_tokens.bin`` (token embeddings in the engine dtype, gathered on the
host), ``mel_filters.f32`` (``[n_fft / 2 + 1, mels]`` Slaney filter bank),
``vocabulary.bin`` (token bytes for decoding) and ``runtime.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .checkpoint import MlxCheckpoint, engine_array
from .config import (
    CHUNK_SECONDS,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    HOP_LENGTH,
    N_FFT,
    PREFILL_MAX_ROWS,
    SAMPLE_RATE,
    VoxtralConfig,
    max_audio_chunks,
    parse_config,
    validate_preprocessor,
)
from .prompt import prompt_contract, vocabulary_table

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter

TASK = "speech_transcription"
PRECISIONS = ("fp32", "fp16", "bf16")
_DUPLICATE_EMBEDDING = ("embed_tokens", "language_model.embed_tokens")
_EMBEDDING_ROWS_PER_WRITE = 8192


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"voxtral checkpoint file is missing: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _mel_filters(model_dir: Path) -> np.ndarray:
    """The checkpoint feature extractor's own filter bank, as float32 ``[freq_bins, mels]``."""
    from transformers import WhisperFeatureExtractor

    extractor = WhisperFeatureExtractor.from_pretrained(str(model_dir))
    filters = np.asarray(extractor.mel_filters, dtype=np.float32)
    if filters.shape != (N_FFT // 2 + 1, extractor.feature_size):
        raise ValueError(f"unexpected mel filter bank shape {filters.shape}")
    return filters


def _write_embeddings(writer: "BundleWriter", weights: MlxCheckpoint, config: VoxtralConfig,
                      precision: str) -> None:
    table = weights.linear("language_model.embed_tokens", config.text.vocab_size,
                           config.text.hidden_size)
    with writer.open_section("embed_tokens.bin") as output:
        for start in range(0, table.shape[0], _EMBEDDING_ROWS_PER_WRITE):
            rows = table[start : start + _EMBEDDING_ROWS_PER_WRITE]
            output.write(engine_array(rows, precision).tobytes())


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build a Voxtral speech-transcription bundle."""
    if request.task != TASK:
        raise ValueError(f"voxtral supports only task={TASK}")
    if request.backend != "trt":
        raise ValueError("voxtral supports only the trt backend")
    if request.precision not in PRECISIONS:
        raise ValueError(f"voxtral precision must be one of {PRECISIONS}, not {request.precision!r}")
    if request.quantization not in (None, "none"):
        raise NotImplementedError(
            "voxtral builds the checkpoint's own MLX weights; quantization is not supported")
    if request.dynamic_kv_cache:
        raise NotImplementedError("voxtral does not support dynamic_kv_cache")
    unsupported = {
        "image_height": request.image_height is not None,
        "image_width": request.image_width is not None,
        "video_num_frames": request.video_num_frames is not None,
        "max_batch_size": request.max_batch_size != 1,
        "tensor_parallel_size": request.tensor_parallel_size != 1,
        "context_parallel_size": request.context_parallel_size != 1,
        "fp32_layers": bool(request.fp32_layers),
        "graph_transform": request.graph_transform is not None,
        "mtp_seq_len": request.mtp_seq_len is not None,
    }
    rejected = sorted(name for name, present in unsupported.items() if present)
    if rejected:
        raise NotImplementedError(f"voxtral does not support: {', '.join(rejected)}")
    from .graph import build_decoder, build_encoder

    model_dir = Path(request.model_dir)
    config = parse_config(_read_json(model_dir / "config.json"))
    validate_preprocessor(_read_json(model_dir / "preprocessor_config.json"), config)
    contract = prompt_contract(model_dir, config)
    longest_suffix = max(len(ids) for ids in contract["language_suffix_ids"].values())
    fixed_tokens = len(contract["prompt_prefix_ids"]) + max(len(contract["prompt_suffix_ids"]),
                                                            longest_suffix)
    capacity = request.max_sequence_length or DEFAULT_MAX_SEQUENCE_LENGTH
    if capacity > config.text.max_position_embeddings:
        raise ValueError(f"max_sequence_length {capacity} exceeds the decoder's "
                         f"{config.text.max_position_embeddings} positions")
    chunks = max_audio_chunks(capacity, config, fixed_tokens)
    prefill_rows = min(PREFILL_MAX_ROWS, capacity)

    weights = MlxCheckpoint(model_dir, config.quantization)
    if not weights.packed_equal(*_DUPLICATE_EMBEDDING):
        raise ValueError("the export's top-level embed_tokens differs from the decoder's table")
    precision = request.precision
    writer.set_header(family="voxtral", task=request.task, backend=request.backend)
    writer.add_bytes("encoder.plan", build_encoder(config, weights, precision, chunks,
                                                   verbose=request.verbose))
    writer.add_bytes("decoder.plan", build_decoder(config, weights, precision, capacity,
                                                   prefill_rows, verbose=request.verbose))
    _write_embeddings(writer, weights, config, precision)
    leftover = weights.unconsumed()
    if leftover:
        raise ValueError(f"voxtral did not consume checkpoint tensors: {leftover[:8]}")
    writer.add_bytes("mel_filters.f32", _mel_filters(model_dir).tobytes())
    writer.add_bytes("vocabulary.bin", vocabulary_table(
        model_dir, config.text.vocab_size, contract["num_special_tokens"]))
    writer.add_json("runtime.json", {
        "version": 1,
        "precision": precision,
        "sample_rate": SAMPLE_RATE,
        "chunk_samples": SAMPLE_RATE * CHUNK_SECONDS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "num_mel_bins": config.audio.num_mel_bins,
        "mel_frames_per_chunk": config.mel_frames_per_chunk,
        "audio_tokens_per_chunk": config.audio_tokens_per_chunk,
        "max_audio_chunks": chunks,
        "hidden_size": config.text.hidden_size,
        "vocab_size": config.text.vocab_size,
        "num_hidden_layers": config.text.num_hidden_layers,
        "kv_width": config.text.num_key_value_heads * config.text.head_dim,
        "max_cache_length": capacity,
        "prefill_max_rows": prefill_rows,
        "audio_token_id": config.audio_token_id,
        **contract,
    })
