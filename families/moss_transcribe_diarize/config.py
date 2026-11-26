# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact checkpoint contract of MOSS-Transcribe-Diarize.

The checkpoint combines four files: ``config.json`` (Qwen3 text decoder, Whisper
audio encoder, frame merge), ``preprocessor_config.json`` (the Whisper log-mel
front end), ``processor_config.json`` (audio placeholder expansion and the
numeric time markers written into the audio span) and ``generation_config.json``
(end-of-turn token and the default generation length). Every value the family
implements is read and checked here; any other variant is refused by name.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_TYPE = "moss_transcribe_diarize"
ARCHITECTURE = "MossTranscribeDiarizeForConditionalGeneration"

# The processor derives one audio token from hop_length * this stride * merge
# samples; it is the stride of the Whisper encoder's second convolution.
WHISPER_ENCODER_STRIDE = 2

# nn.LayerNorm default used by every Whisper encoder norm.
WHISPER_LAYER_NORM_EPS = 1e-5

# Generation settings that keep Hugging Face greedy decoding a plain argmax.
_NEUTRAL_GENERATION = {
    "do_sample": False,
    "num_beams": 1,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "length_penalty": 1.0,
}
_GENERATION_IDENTITY_KEYS = {
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "max_new_tokens",
    "transformers_version",
    "_from_model_config",
}


@dataclass(frozen=True)
class TextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float


@dataclass(frozen=True)
class AudioConfig:
    num_mel_bins: int
    d_model: int
    num_layers: int
    num_heads: int
    ffn_dim: int
    max_source_positions: int

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads


@dataclass(frozen=True)
class FrontendConfig:
    sampling_rate: int
    n_fft: int
    hop_length: int
    chunk_samples: int
    mel_frames: int


@dataclass(frozen=True)
class MarkerConfig:
    audio_tokens_per_second: float
    every_seconds: int
    enabled: bool


@dataclass(frozen=True)
class MossConfig:
    text: TextConfig
    audio: AudioConfig
    frontend: FrontendConfig
    markers: MarkerConfig
    audio_token_id: int
    merge_size: int
    adaptor_input_dim: int
    eos_token_ids: tuple[int, ...]
    default_max_new_tokens: int

    @property
    def encoder_frames(self) -> int:
        return self.audio.max_source_positions

    @property
    def tokens_per_chunk(self) -> int:
        return self.encoder_frames // self.merge_size

    @property
    def samples_per_token(self) -> int:
        return self.frontend.hop_length * WHISPER_ENCODER_STRIDE * self.merge_size


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{MODEL_TYPE} checkpoint is missing {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must hold a JSON object")
    return value


def _int(container: dict[str, Any], key: str, where: str) -> int:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where}.{key} must be a positive integer, got {value!r}")
    return value


def _float(container: dict[str, Any], key: str, where: str) -> float:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}.{key} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{where}.{key} must be finite and positive, got {value!r}")
    return value


def _require(container: dict[str, Any], key: str, expected: Any, where: str) -> None:
    actual = container.get(key, expected)
    if actual != expected:
        raise ValueError(
            f"{MODEL_TYPE} supports only {where}.{key}={expected!r}, got {actual!r}"
        )


def _rope_theta(text: dict[str, Any]) -> float:
    parameters = text.get("rope_parameters")
    if parameters is not None:
        if not isinstance(parameters, dict) or parameters.get("rope_type", "default") != "default":
            raise ValueError(f"{MODEL_TYPE} supports only the default RoPE schedule")
        if "rope_theta" in parameters:
            return _float(parameters, "rope_theta", "text_config.rope_parameters")
    scaling = text.get("rope_scaling")
    if scaling not in (None, {}) and (
        not isinstance(scaling, dict) or scaling.get("rope_type", scaling.get("type")) != "default"
    ):
        raise ValueError(f"{MODEL_TYPE} supports only the default RoPE schedule")
    return _float(text, "rope_theta", "text_config")


def _text_config(raw: dict[str, Any]) -> TextConfig:
    text = raw.get("text_config")
    if not isinstance(text, dict):
        raise ValueError(f"{MODEL_TYPE} config.json has no text_config object")
    _require(text, "model_type", "qwen3", "text_config")
    _require(text, "hidden_act", "silu", "text_config")
    _require(text, "attention_bias", False, "text_config")
    _require(text, "use_sliding_window", False, "text_config")
    num_layers = _int(text, "num_hidden_layers", "text_config")
    layer_types = text.get("layer_types")
    if layer_types is not None and list(layer_types) != ["full_attention"] * num_layers:
        raise ValueError(f"{MODEL_TYPE} supports only full-attention decoder layers")
    config = TextConfig(
        vocab_size=_int(text, "vocab_size", "text_config"),
        hidden_size=_int(text, "hidden_size", "text_config"),
        intermediate_size=_int(text, "intermediate_size", "text_config"),
        num_layers=num_layers,
        num_heads=_int(text, "num_attention_heads", "text_config"),
        num_kv_heads=_int(text, "num_key_value_heads", "text_config"),
        head_dim=_int(text, "head_dim", "text_config"),
        rms_norm_eps=_float(text, "rms_norm_eps", "text_config"),
        rope_theta=_rope_theta(text),
    )
    if config.num_heads % config.num_kv_heads != 0:
        raise ValueError("text_config attention heads must be a multiple of the KV heads")
    if config.head_dim % 2 != 0:
        raise ValueError("text_config.head_dim must be even for rotary embeddings")
    return config


def _audio_config(raw: dict[str, Any]) -> AudioConfig:
    audio = raw.get("audio_config")
    if not isinstance(audio, dict):
        raise ValueError(f"{MODEL_TYPE} config.json has no audio_config object")
    _require(audio, "model_type", "whisper", "audio_config")
    _require(audio, "activation_function", "gelu", "audio_config")
    _require(audio, "scale_embedding", False, "audio_config")
    config = AudioConfig(
        num_mel_bins=_int(audio, "num_mel_bins", "audio_config"),
        d_model=_int(audio, "d_model", "audio_config"),
        num_layers=_int(audio, "encoder_layers", "audio_config"),
        num_heads=_int(audio, "encoder_attention_heads", "audio_config"),
        ffn_dim=_int(audio, "encoder_ffn_dim", "audio_config"),
        max_source_positions=_int(audio, "max_source_positions", "audio_config"),
    )
    if config.d_model % config.num_heads != 0:
        raise ValueError("audio_config.d_model must be a multiple of the attention heads")
    return config


def _frontend_config(preprocessor: dict[str, Any], audio: AudioConfig) -> FrontendConfig:
    where = "preprocessor_config"
    _require(preprocessor, "feature_extractor_type", "WhisperFeatureExtractor", where)
    _require(preprocessor, "dither", 0.0, where)
    _require(preprocessor, "padding_value", 0.0, where)
    sampling_rate = _int(preprocessor, "sampling_rate", where)
    chunk_length = _int(preprocessor, "chunk_length", where)
    frontend = FrontendConfig(
        sampling_rate=sampling_rate,
        n_fft=_int(preprocessor, "n_fft", where),
        hop_length=_int(preprocessor, "hop_length", where),
        chunk_samples=_int(preprocessor, "n_samples", where),
        mel_frames=_int(preprocessor, "nb_max_frames", where),
    )
    if _int(preprocessor, "feature_size", where) != audio.num_mel_bins:
        raise ValueError("preprocessor feature_size must equal audio_config.num_mel_bins")
    if frontend.chunk_samples != chunk_length * sampling_rate:
        raise ValueError("preprocessor n_samples must equal chunk_length * sampling_rate")
    if frontend.mel_frames != frontend.chunk_samples // frontend.hop_length:
        raise ValueError("preprocessor nb_max_frames must equal n_samples // hop_length")
    if frontend.mel_frames != audio.max_source_positions * WHISPER_ENCODER_STRIDE:
        raise ValueError("the Whisper encoder must see one full chunk of mel frames")
    return frontend


def _marker_config(processor: dict[str, Any], merge_size: int, samples_per_token: int,
                   sampling_rate: int) -> MarkerConfig:
    where = "processor_config"
    _require(processor, "processor_class", "MossTranscribeDiarizeProcessor", where)
    if _int(processor, "audio_merge_size", where) != merge_size:
        raise ValueError("processor_config.audio_merge_size must equal config.audio_merge_size")
    enabled = processor.get("enable_time_marker", True)
    if not isinstance(enabled, bool):
        raise ValueError("processor_config.enable_time_marker must be a boolean")
    markers = MarkerConfig(
        audio_tokens_per_second=_float(processor, "audio_tokens_per_second", where),
        every_seconds=_int(processor, "time_marker_every_seconds", where),
        enabled=enabled,
    )
    if markers.audio_tokens_per_second * samples_per_token != float(sampling_rate):
        raise ValueError(
            "processor_config.audio_tokens_per_second must equal the encoder token rate "
            f"{sampling_rate / samples_per_token}"
        )
    return markers


def _generation(generation: dict[str, Any]) -> tuple[tuple[int, ...], int]:
    for key, value in generation.items():
        if key in _GENERATION_IDENTITY_KEYS:
            continue
        if key not in _NEUTRAL_GENERATION or value != _NEUTRAL_GENERATION[key]:
            raise ValueError(
                f"{MODEL_TYPE} implements greedy decoding only; "
                f"generation_config.{key}={value!r} is not supported"
            )
    eos = generation.get("eos_token_id")
    eos_ids = (eos,) if isinstance(eos, int) else tuple(eos or ())
    if not eos_ids or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in eos_ids):
        raise ValueError("generation_config.eos_token_id must name the end-of-turn token")
    return eos_ids, _int(generation, "max_new_tokens", "generation_config")


def load_config(model_dir: str | Path) -> MossConfig:
    """Read and validate the four configuration files of one checkpoint."""
    root = Path(model_dir)
    raw = _read_json(root / "config.json")
    if raw.get("model_type") != MODEL_TYPE:
        raise ValueError(f"{MODEL_TYPE} cannot build model_type={raw.get('model_type')!r}")
    if ARCHITECTURE not in (raw.get("architectures") or ()):
        raise ValueError(f"{MODEL_TYPE} builds only the {ARCHITECTURE} head")
    if raw.get("tie_word_embeddings", True) is not True:
        raise ValueError(f"{MODEL_TYPE} expects the LM head tied to the token embedding")
    if raw.get("text_config", {}).get("tie_word_embeddings", True) is not True:
        raise ValueError(f"{MODEL_TYPE} expects the LM head tied to the token embedding")

    text = _text_config(raw)
    audio = _audio_config(raw)
    merge_size = _int(raw, "audio_merge_size", "config")
    adaptor_input_dim = raw.get("adaptor_input_dim") or audio.d_model * merge_size
    if adaptor_input_dim != audio.d_model * merge_size:
        raise ValueError("config.adaptor_input_dim must equal d_model * audio_merge_size")
    if audio.max_source_positions % merge_size != 0:
        raise ValueError("the encoder frame count must be a multiple of audio_merge_size")
    audio_token_id = raw.get("audio_token_id")
    if isinstance(audio_token_id, bool) or not isinstance(audio_token_id, int):
        raise ValueError("config.audio_token_id must be an integer")
    if not 0 <= audio_token_id < text.vocab_size:
        raise ValueError("config.audio_token_id must lie inside the vocabulary")

    frontend = _frontend_config(_read_json(root / "preprocessor_config.json"), audio)
    samples_per_token = frontend.hop_length * WHISPER_ENCODER_STRIDE * merge_size
    markers = _marker_config(
        _read_json(root / "processor_config.json"),
        merge_size,
        samples_per_token,
        frontend.sampling_rate,
    )
    eos_ids, max_new_tokens = _generation(_read_json(root / "generation_config.json"))
    if any(i >= text.vocab_size for i in eos_ids):
        raise ValueError("generation_config.eos_token_id must lie inside the vocabulary")

    config = MossConfig(
        text=text,
        audio=audio,
        frontend=frontend,
        markers=markers,
        audio_token_id=audio_token_id,
        merge_size=merge_size,
        adaptor_input_dim=adaptor_input_dim,
        eos_token_ids=eos_ids,
        default_max_new_tokens=max_new_tokens,
    )
    full_chunk_tokens = (frontend.chunk_samples - 1) // config.samples_per_token + 1
    if full_chunk_tokens != config.tokens_per_chunk:
        raise ValueError("a full audio chunk must map onto every merged encoder frame")
    return config
