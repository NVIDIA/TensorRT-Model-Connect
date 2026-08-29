# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-ASR configuration.

The checkpoint keeps the decoder under ``text_config`` (model_type "llama") and
the encoder under ``audio_config`` (model_type "glmasr_encoder"). The shared
``ModelConfig`` merges ``text_config`` up to the top level, so decoder fields
arrive through the ordinary accessors and this module owns only the encoder and
projector shapes.

Encoder output frames are stacked in groups of ``MERGE_FACTOR`` before the
projector runs, which is why the projector's input width is the encoder's
``intermediate_size`` rather than its ``hidden_size``.
"""

from __future__ import annotations

from dataclasses import dataclass

# The decoder builders ported into this family import ModelConfig from their
# own package. GLM-ASR's decoder fields arrive through the shared parser
# unchanged, so re-export it rather than carrying a second copy.
from tensorrt_model_connect.config import ModelConfig

__all__ = [
    "MERGE_FACTOR",
    "CONV_STAGES",
    "AudioEncoderConfig",
    "ProjectorConfig",
    "ModelConfig",
    "audio_encoder_config",
    "projector_config",
    "audio_token_id",
    "encoder_output_length",
    "projected_length",
]

# Encoder frames merged into one projector input, per
# GlmAsrForConditionalGeneration.get_audio_features.
MERGE_FACTOR = 4

# Conv front-end geometry, as (padding, kernel_size, stride) per layer. Used to
# predict the encoder output length for a given mel length.
CONV_STAGES = ((1, 3, 1), (1, 3, 2))


@dataclass(frozen=True)
class AudioEncoderConfig:
    """Shape of the GLM-ASR audio encoder."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    head_dim: int
    num_mel_bins: int
    max_position_embeddings: int
    rope_theta: float
    partial_rotary_factor: float
    layer_norm_eps: float
    hidden_act: str

    @property
    def attention_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def rotary_dim(self) -> int:
        """Head dimensions that rotate; the remainder passes through."""
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def kv_attention_size(self) -> int:
        return self.num_key_value_heads * self.head_dim


@dataclass(frozen=True)
class ProjectorConfig:
    """Shape of the audio-to-decoder projector."""

    in_features: int
    hidden_features: int
    out_features: int
    hidden_act: str


def audio_encoder_config(raw: dict) -> AudioEncoderConfig | None:
    """Parse ``audio_config``; None when the key is absent."""
    audio = raw.get("audio_config")
    if not isinstance(audio, dict):
        return None
    hidden = int(audio.get("hidden_size", 0))
    heads = int(audio.get("num_attention_heads", 0))
    head_dim = int(audio.get("head_dim") or (hidden // heads if heads else 0))
    # Rotary settings live under "rope_parameters" for this checkpoint; the
    # flat spellings are accepted as a fallback.
    rope = audio.get("rope_parameters")
    rope = rope if isinstance(rope, dict) else {}
    rope_theta = rope.get("rope_theta", audio.get("rope_theta", 10000.0))
    partial = rope.get(
        "partial_rotary_factor", audio.get("partial_rotary_factor", 1.0))
    return AudioEncoderConfig(
        hidden_size=hidden,
        num_hidden_layers=int(audio.get("num_hidden_layers", 0)),
        num_attention_heads=heads,
        num_key_value_heads=int(audio.get("num_key_value_heads") or heads),
        intermediate_size=int(audio.get("intermediate_size", 0)),
        head_dim=head_dim,
        num_mel_bins=int(audio.get("num_mel_bins", 0)),
        max_position_embeddings=int(audio.get("max_position_embeddings", 0)),
        rope_theta=float(rope_theta),
        partial_rotary_factor=float(partial),
        # nn.LayerNorm's default eps; the checkpoint declares no override.
        layer_norm_eps=float(audio.get("layer_norm_eps", 1e-5)),
        hidden_act=str(audio.get("hidden_act", "gelu")),
    )


def projector_config(raw: dict) -> ProjectorConfig | None:
    """Derive the projector shape from the encoder and decoder widths."""
    encoder = audio_encoder_config(raw)
    text = raw.get("text_config")
    if encoder is None or not isinstance(text, dict):
        return None
    decoder_hidden = int(text.get("hidden_size", 0))
    if not decoder_hidden:
        return None
    return ProjectorConfig(
        in_features=encoder.intermediate_size,
        hidden_features=decoder_hidden * 2,
        out_features=decoder_hidden,
        hidden_act=str(raw.get("projector_hidden_act", "gelu")),
    )


def audio_token_id(raw: dict) -> int | None:
    """Decoder token whose embedding the projected audio frames replace."""
    value = raw.get("audio_token_id")
    return None if value is None else int(value)


def encoder_output_length(mel_length: int) -> int:
    """Frames the conv front-end emits for ``mel_length`` mel frames."""
    length = mel_length
    for padding, kernel_size, stride in CONV_STAGES:
        length = (length + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    return length


def projected_length(mel_length: int) -> int:
    """Audio embeddings the projector emits for ``mel_length`` mel frames."""
    frames = encoder_output_length(mel_length)
    if frames < MERGE_FACTOR:
        return 0
    return (frames - MERGE_FACTOR) // MERGE_FACTOR + 1
