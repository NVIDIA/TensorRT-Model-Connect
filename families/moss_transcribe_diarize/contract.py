# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bundle's ``runtime.json``: everything the native runtime needs besides the engines."""

from __future__ import annotations

from .config import MossConfig
from .decoder import PREFILL_CHUNK_TOKENS
from .tokens import DEFAULT_PROMPT


def runtime_json(config: MossConfig, prompt: dict[str, object], capacity: int,
                 precision: str) -> dict[str, object]:
    frontend = config.frontend
    return {
        "precision": precision,
        "sample_rate": frontend.sampling_rate,
        "chunk_samples": frontend.chunk_samples,
        "n_fft": frontend.n_fft,
        "hop_length": frontend.hop_length,
        "num_mel_bins": config.audio.num_mel_bins,
        "mel_frames": frontend.mel_frames,
        "tokens_per_chunk": config.tokens_per_chunk,
        "samples_per_token": config.samples_per_token,
        "hidden_size": config.text.hidden_size,
        "num_layers": config.text.num_layers,
        "num_kv_heads": config.text.num_kv_heads,
        "head_dim": config.text.head_dim,
        "vocab_size": config.text.vocab_size,
        "max_sequence_length": capacity,
        "prefill_chunk_tokens": min(PREFILL_CHUNK_TOKENS, capacity),
        "audio_token_id": config.audio_token_id,
        "time_markers": {
            "enabled": config.markers.enabled,
            "audio_tokens_per_second": config.markers.audio_tokens_per_second,
            "every_seconds": config.markers.every_seconds,
            "digit_token_ids": prompt["digit_token_ids"],
        },
        "prompt": DEFAULT_PROMPT,
        "prompt_prefix_ids": prompt["prompt_prefix_ids"],
        "prompt_suffix_ids": prompt["prompt_suffix_ids"],
        "eos_token_ids": list(config.eos_token_ids),
        "default_max_new_tokens": config.default_max_new_tokens,
    }
