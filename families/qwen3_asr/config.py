# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint configuration of Qwen3-ASR (``config.json`` and ``processor_config.json``).

Every field the engines or the runtime depend on is read here and checked against the one layout
this family implements; anything else is rejected rather than ignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .frontend import post_cnn_length

ARCHITECTURE = "Qwen3ASRForConditionalGeneration"


@dataclass(frozen=True)
class EncoderConfig:
    num_mel_bins: int
    d_model: int
    layers: int
    heads: int
    ffn: int
    conv_channels: int
    n_window: int
    n_window_infer: int
    max_positions: int
    output_dim: int

    @property
    def head_dim(self) -> int:
        return self.d_model // self.heads

    @property
    def chunk_frames(self) -> int:
        return 2 * self.n_window

    @property
    def tokens_per_chunk(self) -> int:
        return post_cnn_length(self.chunk_frames)

    @property
    def window_chunks(self) -> int:
        return self.n_window_infer // self.chunk_frames

    @property
    def window_tokens(self) -> int:
        return self.window_chunks * self.tokens_per_chunk

    @property
    def freq_bins_after_cnn(self) -> int:
        return (((self.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2


@dataclass(frozen=True)
class DecoderConfig:
    vocab_size: int
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    rope_theta: float
    eps: float
    max_positions: int


@dataclass(frozen=True)
class FeatureConfig:
    sampling_rate: int
    n_fft: int
    hop_length: int
    num_mel_bins: int
    min_samples: int
    n_window: int


@dataclass(frozen=True)
class Qwen3AsrConfig:
    encoder: EncoderConfig
    decoder: DecoderConfig
    features: FeatureConfig
    audio_token_id: int
    eos_token_ids: tuple[int, ...]


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Qwen3-ASR checkpoint is missing {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"unsupported Qwen3-ASR checkpoint: {message}")


def _encoder(raw: dict) -> EncoderConfig:
    _require(raw.get("model_type") == "qwen3_asr_encoder", "audio_config.model_type")
    _require(raw.get("activation_function", "gelu") == "gelu", "audio activation")
    _require(not raw.get("scale_embedding", False), "scaled audio embeddings")
    config = EncoderConfig(
        num_mel_bins=int(raw["num_mel_bins"]),
        d_model=int(raw["d_model"]),
        layers=int(raw["encoder_layers"]),
        heads=int(raw["encoder_attention_heads"]),
        ffn=int(raw["encoder_ffn_dim"]),
        conv_channels=int(raw["downsample_hidden_size"]),
        n_window=int(raw["n_window"]),
        n_window_infer=int(raw["n_window_infer"]),
        max_positions=int(raw["max_position_embeddings"]),
        output_dim=int(raw["output_dim"]),
    )
    _require(config.d_model % config.heads == 0, "encoder head width")
    _require(int(raw.get("num_key_value_heads", config.heads)) == config.heads, "encoder GQA")
    _require(config.n_window > 0 and config.n_window_infer % config.chunk_frames == 0,
             "attention window is not a whole number of chunks")
    _require(config.window_chunks >= 1, "attention window")
    # The sinusoid table restarts in every chunk; it must cover one chunk's tokens.
    _require(config.max_positions >= config.tokens_per_chunk, "encoder position table")
    return config


def _decoder(raw: dict) -> DecoderConfig:
    _require(raw.get("model_type") == "qwen3", "text_config.model_type")
    _require(raw.get("hidden_act") == "silu", "decoder activation")
    _require(not raw.get("attention_bias", False), "decoder attention bias")
    _require(bool(raw.get("tie_word_embeddings", False)), "untied LM head")
    _require(not raw.get("use_sliding_window", False) and raw.get("sliding_window") is None,
             "sliding-window attention")
    layer_types = raw.get("layer_types")
    _require(layer_types is None or set(layer_types) == {"full_attention"}, "decoder layer types")
    rope = raw.get("rope_parameters") or {}
    _require(rope.get("rope_type", "default") == "default", "rope scaling")
    theta = rope.get("rope_theta", raw.get("rope_theta"))
    _require(theta is not None, "rope_theta")
    config = DecoderConfig(
        vocab_size=int(raw["vocab_size"]),
        hidden=int(raw["hidden_size"]),
        layers=int(raw["num_hidden_layers"]),
        heads=int(raw["num_attention_heads"]),
        kv_heads=int(raw["num_key_value_heads"]),
        head_dim=int(raw["head_dim"]),
        intermediate=int(raw["intermediate_size"]),
        rope_theta=float(theta),
        eps=float(raw["rms_norm_eps"]),
        max_positions=int(raw["max_position_embeddings"]),
    )
    _require(config.heads % config.kv_heads == 0 and config.head_dim % 2 == 0, "attention heads")
    return config


def _features(raw: dict) -> FeatureConfig:
    extractor = raw.get("feature_extractor", raw)
    _require(extractor.get("feature_extractor_type") == "Qwen3ASRFeatureExtractor",
             "feature extractor type")
    _require(float(extractor.get("dither", 0.0)) == 0.0, "dithered features")
    _require(float(extractor.get("padding_value", 0.0)) == 0.0, "audio padding value")
    return FeatureConfig(
        sampling_rate=int(extractor["sampling_rate"]),
        n_fft=int(extractor["n_fft"]),
        hop_length=int(extractor["hop_length"]),
        num_mel_bins=int(extractor["feature_size"]),
        min_samples=int(extractor.get("min_length", 0)),
        n_window=int(extractor["n_window"]),
    )


def load_config(model_dir: Path) -> Qwen3AsrConfig:
    raw = read_json(model_dir / "config.json")
    _require(raw.get("model_type") == "qwen3_asr", "model_type")
    _require(tuple(raw.get("architectures") or (ARCHITECTURE,)) == (ARCHITECTURE,), "architecture")
    _require(
        "thinker_config" not in raw
        and isinstance(raw.get("audio_config"), dict)
        and isinstance(raw.get("text_config"), dict),
        "this is the original release layout (thinker_config); use the Transformers-native "
        "checkpoint (the -hf repository)",
    )
    encoder = _encoder(raw["audio_config"])
    decoder = _decoder(raw["text_config"])
    features = _features(read_json(model_dir / "processor_config.json"))
    _require(encoder.output_dim == decoder.hidden, "projector width")
    _require(features.num_mel_bins == encoder.num_mel_bins, "mel bins")
    _require(features.n_window == encoder.n_window, "feature chunking")
    _require(features.n_fft % 2 == 0 and features.hop_length > 0, "STFT geometry")
    eos = raw.get("eos_token_id")
    eos_ids = tuple(int(token) for token in (eos if isinstance(eos, list) else [eos]))
    _require(bool(eos_ids) and all(0 <= token < decoder.vocab_size for token in eos_ids), "eos")
    audio_token = int(raw["audio_token_id"])
    _require(0 <= audio_token < decoder.vocab_size, "audio_token_id")
    return Qwen3AsrConfig(
        encoder=encoder,
        decoder=decoder,
        features=features,
        audio_token_id=audio_token,
        eos_token_ids=eos_ids,
    )
