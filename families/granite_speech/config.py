# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve Granite Speech geometry and refuse every variant the graphs do not build.

Each refusal names a setting that loads without error in Transformers and changes the
numbers: an adapter, a scaled rotary embedding, a different feature front end. Building such a
checkpoint silently would produce a plausible but wrong transcript.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

ARCHITECTURE = "GraniteSpeechForConditionalGeneration"
# Engine profiles are tuned for a 30 s clip; every length between the limits is accepted.
OPT_SECONDS = 30


def _read_json(path: Path, what: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required granite_speech {what} is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"granite_speech {what} must contain one JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NotImplementedError(f"granite_speech {message}")


def _int(section: dict, key: str, where: str) -> int:
    value = section.get(key)
    _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
             f"needs a positive integer {where}.{key}, not {value!r}")
    return int(value)


def _float(section: dict, key: str, where: str) -> float:
    value = section.get(key)
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(float(value)), f"needs a finite {where}.{key}, not {value!r}")
    return float(value)


@dataclass(frozen=True)
class FrontEnd:
    """Effective GraniteSpeechFeatureExtractor arguments (its constructor ignores melspec_kwargs)."""

    sample_rate: int
    n_fft: int
    win_length: int
    hop_length: int
    n_mels: int


@dataclass(frozen=True)
class Encoder:
    input_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    dim_head: int
    context_size: int
    max_pos_emb: int
    conv_kernel_size: int
    conv_expansion_factor: int
    feedforward_mult: int
    output_dim: int


@dataclass(frozen=True)
class Projector:
    hidden_size: int
    num_heads: int
    num_layers: int
    intermediate_size: int
    layer_norm_eps: float
    cross_attention_frequency: int
    window_size: int
    downsample_rate: int

    @property
    def num_queries(self) -> int:
        return self.window_size // self.downsample_rate

    def has_cross_attention(self, layer: int) -> bool:
        return layer % self.cross_attention_frequency == 0


@dataclass(frozen=True)
class Text:
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    embedding_multiplier: float
    attention_multiplier: float
    residual_multiplier: float
    logits_scaling: float
    eos_token_id: int

    @property
    def kv_width(self) -> int:
        return self.num_kv_heads * self.head_dim


@dataclass(frozen=True)
class GraniteSpeechConfig:
    front_end: FrontEnd
    encoder: Encoder
    projector: Projector
    text: Text
    audio_token_id: int

    def stacked_frames(self, samples: int) -> int:
        """Encoder frames for ``samples`` PCM values, as the reference computes them."""
        return (samples // self.front_end.hop_length + 1) // 2

    def audio_tokens(self, samples: int) -> int:
        """Projector outputs (= audio placeholder tokens) for ``samples`` PCM values."""
        windows = math.ceil(self.stacked_frames(samples) / self.projector.window_size)
        return windows * self.projector.num_queries


def _front_end(pre: dict, projector_window: int, downsample: int) -> FrontEnd:
    _require(pre.get("feature_extractor_type") == "GraniteSpeechFeatureExtractor",
             f"builds GraniteSpeechFeatureExtractor features, not {pre.get('feature_extractor_type')!r}")
    # The extractor's constructor reads only these top-level keys and falls back to its defaults;
    # a melspec_kwargs entry in the JSON is overwritten by those values.
    front = FrontEnd(
        sample_rate=int(pre.get("sampling_rate", 16000)),
        n_fft=int(pre.get("n_fft", 512)),
        win_length=int(pre.get("win_length", 400)),
        hop_length=int(pre.get("hop_length", 160)),
        n_mels=int(pre.get("n_mels", 80)),
    )
    _require(front.n_fft % 2 == 0 and 0 < front.win_length <= front.n_fft,
             f"needs an even n_fft and 0 < win_length <= n_fft, not {front}")
    _require(front.hop_length > 0 and front.n_mels > 0 and front.sample_rate > 0,
             f"needs positive hop_length, n_mels and sampling_rate, not {front}")
    _require(int(pre.get("projector_window_size", 15)) == projector_window,
             "needs the feature extractor's projector_window_size to equal config.window_size")
    _require(int(pre.get("projector_downsample_rate", 5)) == downsample,
             "needs the feature extractor's projector_downsample_rate to equal config.downsample_rate")
    return front


def _encoder(section: dict) -> Encoder:
    where = "encoder_config"
    _require(section.get("model_type") == "granite_speech_encoder",
             f"builds the granite_speech_encoder, not {section.get('model_type')!r}")
    encoder = Encoder(
        input_dim=_int(section, "input_dim", where),
        hidden_dim=_int(section, "hidden_dim", where),
        num_layers=_int(section, "num_layers", where),
        num_heads=_int(section, "num_heads", where),
        dim_head=_int(section, "dim_head", where),
        context_size=_int(section, "context_size", where),
        max_pos_emb=_int(section, "max_pos_emb", where),
        conv_kernel_size=_int(section, "conv_kernel_size", where),
        conv_expansion_factor=_int(section, "conv_expansion_factor", where),
        feedforward_mult=_int(section, "feedforward_mult", where),
        output_dim=_int(section, "output_dim", where),
    )
    _require(encoder.context_size <= encoder.max_pos_emb,
             "needs encoder context_size <= max_pos_emb (the reference refuses it too)")
    return encoder


def _projector(section: dict, top: dict, encoder: Encoder) -> Projector:
    where = "projector_config"
    _require(section.get("model_type") == "blip_2_qformer",
             f"builds a blip_2_qformer projector, not {section.get('model_type')!r}")
    _require(not section.get("use_qformer_text_input", False),
             "builds the query-only Q-Former, not one with text input")
    _require(section.get("hidden_act") == "gelu",
             f"builds the exact GELU Q-Former, not {section.get('hidden_act')!r}")
    projector = Projector(
        hidden_size=_int(section, "hidden_size", where),
        num_heads=_int(section, "num_attention_heads", where),
        num_layers=_int(section, "num_hidden_layers", where),
        intermediate_size=_int(section, "intermediate_size", where),
        layer_norm_eps=_float(section, "layer_norm_eps", where),
        cross_attention_frequency=_int(section, "cross_attention_frequency", where),
        window_size=_int(top, "window_size", "config"),
        downsample_rate=_int(top, "downsample_rate", "config"),
    )
    _require(projector.hidden_size % projector.num_heads == 0,
             "needs the Q-Former width divisible by its head count")
    _require(int(section.get("encoder_hidden_size", 0)) == encoder.hidden_dim,
             "needs projector encoder_hidden_size equal to the encoder width")
    _require(projector.window_size % projector.downsample_rate == 0,
             "needs window_size divisible by downsample_rate (the reference reshapes by it)")
    return projector


def _rope_theta(section: dict) -> float:
    parameters = section.get("rope_parameters")
    scaling = section.get("rope_scaling")
    _require(scaling is None, f"builds the default rotary embedding, not rope_scaling={scaling!r}")
    if parameters is not None:
        _require(isinstance(parameters, dict) and parameters.get("rope_type", "default") == "default",
                 f"builds the default rotary embedding, not rope_parameters={parameters!r}")
        theta = parameters.get("rope_theta", section.get("rope_theta"))
    else:
        theta = section.get("rope_theta")
    _require(isinstance(theta, (int, float)) and float(theta) > 0, f"needs a positive rope_theta, not {theta!r}")
    _require(float(section.get("partial_rotary_factor", 1.0)) == 1.0,
             "builds a full-width rotary embedding")
    return float(theta)


def _text(section: dict, top: dict) -> Text:
    where = "text_config"
    _require(section.get("model_type") == "granite",
             f"builds a granite language model, not {section.get('model_type')!r}")
    _require(section.get("hidden_act") == "silu", "builds the SiLU-gated MLP")
    _require(not section.get("attention_bias", False), "builds attention without bias")
    _require(not section.get("mlp_bias", False), "builds the MLP without bias")
    _require(not top.get("tie_word_embeddings", False) and not section.get("tie_word_embeddings", False),
             "builds an untied language-model head")
    hidden = _int(section, "hidden_size", where)
    heads = _int(section, "num_attention_heads", where)
    kv_heads = _int(section, "num_key_value_heads", where)
    head_dim = int(section.get("head_dim") or hidden // heads)
    _require(heads % kv_heads == 0, "needs the query heads divisible by the key/value heads")
    _require(head_dim % 2 == 0, "needs an even attention head width for the rotary embedding")
    eos = section.get("eos_token_id")
    _require(isinstance(eos, int) and not isinstance(eos, bool), f"needs one integer eos_token_id, not {eos!r}")
    return Text(
        hidden_size=hidden,
        num_layers=_int(section, "num_hidden_layers", where),
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        intermediate_size=_int(section, "intermediate_size", where),
        vocab_size=_int(section, "vocab_size", where),
        rms_norm_eps=_float(section, "rms_norm_eps", where),
        rope_theta=_rope_theta(section),
        max_position_embeddings=_int(section, "max_position_embeddings", where),
        embedding_multiplier=_float(section, "embedding_multiplier", where),
        attention_multiplier=_float(section, "attention_multiplier", where),
        residual_multiplier=_float(section, "residual_multiplier", where),
        logits_scaling=_float(section, "logits_scaling", where),
        eos_token_id=int(eos),
    )


def resolve(model_dir: Path) -> GraniteSpeechConfig:
    """Read config.json and preprocessor_config.json; raise naming what is not built."""
    model_dir = Path(model_dir)
    top = _read_json(model_dir / "config.json", "config.json")
    pre = _read_json(model_dir / "preprocessor_config.json", "preprocessor_config.json")
    _require(top.get("model_type") == "granite_speech",
             f"builds model_type granite_speech, not {top.get('model_type')!r}")
    _require(tuple(top.get("architectures") or ()) == (ARCHITECTURE,),
             f"builds {ARCHITECTURE}, not {top.get('architectures')!r}")
    # A LoRA adapter is enabled only while audio is present; the base weights alone would
    # transcribe differently, so an adapter checkpoint is refused rather than ignored.
    _require(not top.get("has_lora_adapter", False), "does not build the LoRA adapter variant")
    encoder = _encoder(top.get("encoder_config") or {})
    projector = _projector(top.get("projector_config") or {}, top, encoder)
    text = _text(top.get("text_config") or {}, top)
    _require(projector.hidden_size == encoder.hidden_dim,
             "needs the Q-Former width equal to the encoder width")
    audio_token = top.get("audio_token_index", top.get("audio_token_id"))
    _require(isinstance(audio_token, int) and 0 <= audio_token < text.vocab_size,
             f"needs an audio token id inside the vocabulary, not {audio_token!r}")
    front = _front_end(pre, projector.window_size, projector.downsample_rate)
    _require(2 * front.n_mels == encoder.input_dim,
             "needs encoder input_dim equal to two stacked mel frames")
    return GraniteSpeechConfig(front_end=front, encoder=encoder, projector=projector, text=text,
                               audio_token_id=int(audio_token))


def sample_limits(cfg: GraniteSpeechConfig, prompt_length: int, capacity: int) -> dict:
    """Sample range of one request for a decoder holding ``capacity`` positions.

    Reflect padding needs ``n_fft / 2 + 1`` samples; the placeholders of the longest clip leave
    room for the ``prompt_length`` fixed ids and one generated token.
    """
    fe, p = cfg.front_end, cfg.projector
    windows = (capacity - prompt_length - 1) // p.num_queries
    if windows < 1:
        raise ValueError(f"granite_speech capacity {capacity} leaves no room for audio after the prompt")
    max_frames = windows * p.window_size
    max_samples = (2 * max_frames + 1) * fe.hop_length - 1
    min_samples = fe.n_fft // 2 + 1
    opt_samples = min(max(OPT_SECONDS * fe.sample_rate, min_samples), max_samples)
    return {"min_samples": min_samples, "max_samples": max_samples, "opt_samples": opt_samples,
            "max_frames": max_frames}
