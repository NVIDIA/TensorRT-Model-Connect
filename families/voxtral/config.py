# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voxtral checkpoint configuration and transcription geometry (standard library only)."""

from __future__ import annotations

import math
from dataclasses import dataclass

ARCHITECTURE = "VoxtralForConditionalGeneration"
MODEL_TYPE = "voxtral"
MLX_CODE_WIDTHS = (2, 3, 4, 5, 6, 8)
# Whisper-style front end of the checkpoint's WhisperFeatureExtractor.
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_SECONDS = 30
# Bundle capacity (prompt plus generated tokens) when the request names none.
DEFAULT_MAX_SEQUENCE_LENGTH = 4096
# Rows of one prefill launch; longer prompts are prefilled in several launches.
PREFILL_MAX_ROWS = 512


@dataclass(frozen=True)
class TextConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int


@dataclass(frozen=True)
class AudioConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    num_mel_bins: int
    max_source_positions: int


@dataclass(frozen=True)
class VoxtralConfig:
    text: TextConfig
    audio: AudioConfig
    audio_token_id: int
    quantization: dict

    @property
    def frames_per_token(self) -> int:
        """Encoder frames stacked into one projector input row."""
        return self.audio.intermediate_size // self.audio.hidden_size

    @property
    def mel_frames_per_chunk(self) -> int:
        """Mel frames of one 30 s chunk (the encoder's stride-2 convolution halves them)."""
        return 2 * self.audio.max_source_positions

    @property
    def audio_tokens_per_chunk(self) -> int:
        return self.audio.max_source_positions // self.frames_per_token


def _integer(source: dict, key: str, where: str) -> int:
    value = source.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"voxtral {where}.{key} must be a positive integer, got {value!r}")
    return value


def _number(source: dict, key: str, where: str) -> float:
    value = source.get(key)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"voxtral {where}.{key} must be a positive number, got {value!r}")
    return float(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"voxtral: {message}")


def _rope_theta(text: dict) -> float:
    parameters = text.get("rope_parameters")
    if isinstance(parameters, dict):
        _require(parameters.get("rope_type", "default") == "default",
                 "only the default rotary embedding is supported")
        return _number(parameters, "rope_theta", "text_config.rope_parameters")
    _require(text.get("rope_scaling") is None, "rope_scaling is not supported")
    return _number(text, "rope_theta", "text_config")


def _quantization(raw: dict) -> dict:
    spec = raw.get("quantization")
    _require(isinstance(spec, dict) and spec == raw.get("quantization_config", spec),
             "the checkpoint must be an MLX affine-quantized export (config 'quantization')")
    for module, entry in spec.items():
        if module in ("group_size", "bits"):
            _require(type(entry) is int and entry > 0, f"quantization.{module} must be positive")
            continue
        if entry is False:
            continue
        _require(
            isinstance(entry, dict)
            and set(entry) <= {"bits", "group_size", "mode"}
            and entry.get("mode", "affine") == "affine"
            and entry.get("bits") in MLX_CODE_WIDTHS
            and type(entry.get("group_size")) is int
            and entry["group_size"] > 0
            and entry["group_size"] % 8 == 0,
            f"unsupported MLX quantization entry for {module}: {entry!r}",
        )
    _require(raw.get("quantization", {}).get("mode", "affine") == "affine",
             "only MLX affine quantization is supported")
    return spec


def parse_config(raw: dict) -> VoxtralConfig:
    """Validate a Voxtral ``config.json`` and return the fields the family uses."""
    _require(isinstance(raw, dict), "config.json must be an object")
    _require(raw.get("model_type") == MODEL_TYPE, f"model_type must be {MODEL_TYPE!r}")
    _require(raw.get("architectures") == [ARCHITECTURE], f"architectures must be [{ARCHITECTURE!r}]")
    text = raw.get("text_config")
    audio = raw.get("audio_config")
    _require(isinstance(text, dict) and isinstance(audio, dict),
             "text_config and audio_config are required")
    _require(text.get("model_type") == "llama", "text_config.model_type must be 'llama'")
    _require(text.get("hidden_act", "silu") == "silu", "the decoder MLP must use silu")
    _require(not text.get("attention_bias", False) and not text.get("mlp_bias", False),
             "decoder attention and MLP biases are not supported")
    _require(text.get("sliding_window") is None, "sliding-window attention is not supported")
    _require(not text.get("tie_word_embeddings", False), "tied output embeddings are not supported")
    _require(text.get("pretraining_tp", 1) == 1, "pretraining_tp must be 1")
    text_config = TextConfig(
        hidden_size=_integer(text, "hidden_size", "text_config"),
        intermediate_size=_integer(text, "intermediate_size", "text_config"),
        num_hidden_layers=_integer(text, "num_hidden_layers", "text_config"),
        num_attention_heads=_integer(text, "num_attention_heads", "text_config"),
        num_key_value_heads=_integer(text, "num_key_value_heads", "text_config"),
        head_dim=_integer(text, "head_dim", "text_config"),
        vocab_size=_integer(text, "vocab_size", "text_config"),
        rms_norm_eps=_number(text, "rms_norm_eps", "text_config"),
        rope_theta=_rope_theta(text),
        max_position_embeddings=_integer(text, "max_position_embeddings", "text_config"),
    )
    _require(text_config.num_attention_heads % text_config.num_key_value_heads == 0,
             "attention heads must be a multiple of key/value heads")
    _require(text_config.head_dim % 2 == 0, "head_dim must be even for rotary embeddings")
    _require(raw.get("vocab_size", text_config.vocab_size) == text_config.vocab_size,
             "top-level and text vocab sizes differ")
    _require(raw.get("hidden_size", text_config.hidden_size) == text_config.hidden_size,
             "top-level and text hidden sizes differ")

    _require(audio.get("model_type") == "voxtral_encoder",
             "audio_config.model_type must be 'voxtral_encoder'")
    _require(audio.get("activation_function", "gelu") == "gelu", "the encoder MLP must use gelu")
    _require(not audio.get("scale_embedding", False), "scale_embedding is not supported")
    audio_config = AudioConfig(
        hidden_size=_integer(audio, "hidden_size", "audio_config"),
        intermediate_size=_integer(audio, "intermediate_size", "audio_config"),
        num_hidden_layers=_integer(audio, "num_hidden_layers", "audio_config"),
        num_attention_heads=_integer(audio, "num_attention_heads", "audio_config"),
        head_dim=_integer(audio, "head_dim", "audio_config"),
        num_mel_bins=_integer(audio, "num_mel_bins", "audio_config"),
        max_source_positions=_integer(audio, "max_source_positions", "audio_config"),
    )
    _require(audio.get("num_key_value_heads", audio_config.num_attention_heads)
             == audio_config.num_attention_heads, "the encoder must use multi-head attention")
    _require(audio_config.num_attention_heads * audio_config.head_dim == audio_config.hidden_size,
             "encoder heads times head_dim must equal its hidden size")
    _require(audio_config.intermediate_size % audio_config.hidden_size == 0,
             "the projector input must stack whole encoder frames")
    stack = audio_config.intermediate_size // audio_config.hidden_size
    _require(audio_config.max_source_positions % stack == 0,
             "encoder frames per chunk must be a multiple of the frame stack")
    _require(raw.get("projector_hidden_act", "gelu") == "gelu", "the projector must use gelu")
    audio_token_id = raw.get("audio_token_id")
    _require(type(audio_token_id) is int and 0 <= audio_token_id < text_config.vocab_size,
             "audio_token_id must be a token id")
    return VoxtralConfig(text_config, audio_config, audio_token_id, _quantization(raw))


def validate_preprocessor(preprocessor: dict, config: VoxtralConfig) -> None:
    """The front end the runtime implements must be the one the checkpoint declares."""
    expected = {
        "feature_extractor_type": "WhisperFeatureExtractor",
        "feature_size": config.audio.num_mel_bins,
        "sampling_rate": SAMPLE_RATE,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "chunk_length": CHUNK_SECONDS,
        "n_samples": SAMPLE_RATE * CHUNK_SECONDS,
        "nb_max_frames": config.mel_frames_per_chunk,
    }
    for key, value in expected.items():
        _require(preprocessor.get(key) == value,
                 f"preprocessor_config.{key} must be {value!r}, got {preprocessor.get(key)!r}")
    _require(float(preprocessor.get("dither", 0.0)) == 0.0, "dithered features are not supported")
    _require(not preprocessor.get("do_normalize", False), "normalized waveforms are not supported")
    _require(config.mel_frames_per_chunk * HOP_LENGTH == SAMPLE_RATE * CHUNK_SECONDS,
             "mel frames per chunk do not cover the 30 s chunk")


def max_audio_chunks(capacity: int, config: VoxtralConfig, fixed_prompt_tokens: int) -> int:
    """Largest number of 30 s chunks whose prompt still leaves one generated token."""
    chunks = (capacity - fixed_prompt_tokens - 1) // config.audio_tokens_per_chunk
    _require(chunks >= 1, f"max_sequence_length {capacity} cannot hold one 30 s audio chunk")
    return chunks
