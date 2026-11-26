# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voxtral checkpoint configuration and transcription geometry (standard library only)."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass

ARCHITECTURE = "VoxtralForConditionalGeneration"
MODEL_TYPE = "voxtral"
# Kernel width of both encoder convolutions.
CONV_KERNEL = 3
# Whisper-style front end of the checkpoint's WhisperFeatureExtractor.
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_SECONDS = 30
# Bundle capacity (prompt plus generated tokens) when the request names none.
DEFAULT_MAX_SEQUENCE_LENGTH = 4096
# Rows of one prefill launch; longer prompts are prefilled in several launches.
PREFILL_MAX_ROWS = 512
# The runtime's load limits (runtime/runtime_config.cpp): widths, layer count, token ids and cache positions.
MAX_WIDTH = 1 << 20
MAX_LAYERS = 4096
MAX_INT32 = (1 << 31) - 1


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


def load_json(text: str | bytes, what: str):
    """``json.loads`` that refuses an object repeating a key and the NaN / Infinity literals.

    The json module would keep the last of two equal keys and accept the non-standard literals.
    """
    def pairs(items: list) -> dict:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"voxtral {what} repeats the key {key!r}")
            result[key] = value
        return result

    def constant(name: str):
        raise ValueError(f"voxtral {what} holds the non-standard number {name}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def _integer(source: dict, key: str, where: str, maximum: int = MAX_WIDTH) -> int:
    value = source.get(key)
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError(f"voxtral {where}.{key} must be an integer in [1, {maximum}], got {value!r}")
    return value


def _number(source: dict, key: str, where: str) -> float:
    value = source.get(key)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"voxtral {where}.{key} must be a positive number, got {value!r}")
    return float(value)


def _float32(source: dict, key: str, where: str) -> float:
    """A positive number that stays finite and nonzero once stored as float32 (a graph constant)."""
    value = _number(source, key, where)
    try:
        stored = struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        stored = math.inf
    if not math.isfinite(stored) or stored <= 0.0:
        raise ValueError(f"voxtral {where}.{key} = {value!r} is not a positive finite float32 value")
    return stored


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"voxtral: {message}")


def _optional(source: dict, key: str, allowed, message: str) -> None:
    """An option that may be absent; when present it must be exactly ``allowed``, of the same JSON type
    (null, 0 or 1.0 do not stand for false or 1)."""
    if key in source:
        value = source[key]
        _require(type(value) is type(allowed) and value == allowed, f"{message} (got {value!r})")


def _rope_theta(text: dict) -> float:
    if "rope_parameters" in text:
        parameters = text["rope_parameters"]
        _require(isinstance(parameters, dict), "text_config.rope_parameters must be an object")
        _optional(parameters, "rope_type", "default", "only the default rotary embedding is supported")
        return _number(parameters, "rope_theta", "text_config.rope_parameters")
    _require(text.get("rope_scaling") is None, "rope_scaling is not supported")
    return _number(text, "rope_theta", "text_config")


def _check_rotary(theta: float, head_dim: int, positions: int) -> None:
    """Every inverse frequency ``theta ** (-2 i / head_dim)`` and the angle at the last position must be finite."""
    for index in range(0, head_dim, 2):
        try:
            angle = (positions - 1) / theta ** (index / head_dim)
        except (OverflowError, ZeroDivisionError):
            angle = math.inf
        _require(math.isfinite(angle), f"rope_theta {theta!r} gives non-finite rotary angles")


def parse_config(raw: dict) -> VoxtralConfig:
    """Validate a Voxtral ``config.json`` and return the fields the family uses."""
    _require(isinstance(raw, dict), "config.json must be an object")
    _require(raw.get("model_type") == MODEL_TYPE, f"model_type must be {MODEL_TYPE!r}")
    _require(raw.get("architectures") == [ARCHITECTURE], f"architectures must be [{ARCHITECTURE!r}]")
    quantized = sorted(key for key in ("quantization", "quantization_config") if key in raw)
    _require(not quantized, f"quantized checkpoints are not supported (config declares {quantized}); "
             "build the original floating-point checkpoint")
    text = raw.get("text_config")
    audio = raw.get("audio_config")
    _require(isinstance(text, dict) and isinstance(audio, dict),
             "text_config and audio_config are required")
    _require(text.get("model_type") == "llama", "text_config.model_type must be 'llama'")
    _optional(text, "hidden_act", "silu", "the decoder MLP must use silu")
    _optional(text, "attention_bias", False, "decoder attention and MLP biases are not supported")
    _optional(text, "mlp_bias", False, "decoder attention and MLP biases are not supported")
    _require(text.get("sliding_window") is None, "sliding-window attention is not supported")
    _optional(text, "tie_word_embeddings", False, "tied output embeddings are not supported")
    _optional(text, "pretraining_tp", 1, "pretraining_tp must be 1")
    text_config = TextConfig(
        hidden_size=_integer(text, "hidden_size", "text_config"),
        intermediate_size=_integer(text, "intermediate_size", "text_config"),
        num_hidden_layers=_integer(text, "num_hidden_layers", "text_config", MAX_LAYERS),
        num_attention_heads=_integer(text, "num_attention_heads", "text_config"),
        num_key_value_heads=_integer(text, "num_key_value_heads", "text_config"),
        head_dim=_integer(text, "head_dim", "text_config"),
        vocab_size=_integer(text, "vocab_size", "text_config", MAX_INT32),
        rms_norm_eps=_float32(text, "rms_norm_eps", "text_config"),
        rope_theta=_rope_theta(text),
        max_position_embeddings=_integer(text, "max_position_embeddings", "text_config", MAX_INT32),
    )
    _require(text_config.num_attention_heads % text_config.num_key_value_heads == 0,
             "attention heads must be a multiple of key/value heads")
    _require(text_config.num_attention_heads * text_config.head_dim <= MAX_WIDTH,
             f"decoder attention width must not exceed {MAX_WIDTH}")
    _require(text_config.head_dim % 2 == 0, "head_dim must be even for rotary embeddings")
    _check_rotary(text_config.rope_theta, text_config.head_dim, text_config.max_position_embeddings)
    _optional(raw, "vocab_size", text_config.vocab_size, "top-level and text vocab sizes differ")
    _optional(raw, "hidden_size", text_config.hidden_size, "top-level and text hidden sizes differ")

    _require(audio.get("model_type") == "voxtral_encoder",
             "audio_config.model_type must be 'voxtral_encoder'")
    _optional(audio, "activation_function", "gelu", "the encoder MLP must use gelu")
    _optional(audio, "scale_embedding", False, "scale_embedding is not supported")
    audio_config = AudioConfig(
        hidden_size=_integer(audio, "hidden_size", "audio_config"),
        intermediate_size=_integer(audio, "intermediate_size", "audio_config"),
        num_hidden_layers=_integer(audio, "num_hidden_layers", "audio_config", MAX_LAYERS),
        num_attention_heads=_integer(audio, "num_attention_heads", "audio_config"),
        head_dim=_integer(audio, "head_dim", "audio_config"),
        num_mel_bins=_integer(audio, "num_mel_bins", "audio_config"),
        max_source_positions=_integer(audio, "max_source_positions", "audio_config"),
    )
    _optional(audio, "num_key_value_heads", audio_config.num_attention_heads,
              "the encoder must use multi-head attention")
    _require(audio_config.num_attention_heads * audio_config.head_dim == audio_config.hidden_size,
             "encoder heads times head_dim must equal its hidden size")
    _require(audio_config.intermediate_size % audio_config.hidden_size == 0,
             "the projector input must stack whole encoder frames")
    stack = audio_config.intermediate_size // audio_config.hidden_size
    _require(audio_config.max_source_positions % stack == 0,
             "encoder frames per chunk must be a multiple of the frame stack")
    _optional(raw, "projector_hidden_act", "gelu", "the projector must use gelu")
    audio_token_id = raw.get("audio_token_id")
    _require(type(audio_token_id) is int and 0 <= audio_token_id < text_config.vocab_size,
             "audio_token_id must be a token id")
    return VoxtralConfig(text_config, audio_config, audio_token_id)


def expected_tensors(config: VoxtralConfig) -> dict[str, tuple[int, ...]]:
    """Every tensor of a Transformers Voxtral checkpoint with its shape, derived from the configuration.

    Convolution kernels use the PyTorch ``[out, in, kernel]`` layout. The checkpoint must hold exactly these tensors.
    """
    audio, text = config.audio, config.text
    width, inner = audio.hidden_size, audio.intermediate_size
    shapes: dict[str, tuple[int, ...]] = {
        "audio_tower.conv1.weight": (width, audio.num_mel_bins, CONV_KERNEL),
        "audio_tower.conv1.bias": (width,),
        "audio_tower.conv2.weight": (width, width, CONV_KERNEL),
        "audio_tower.conv2.bias": (width,),
        "audio_tower.embed_positions.weight": (audio.max_source_positions, width),
        "audio_tower.layer_norm.weight": (width,),
        "audio_tower.layer_norm.bias": (width,),
        "multi_modal_projector.linear_1.weight": (text.hidden_size, audio.intermediate_size),
        "multi_modal_projector.linear_2.weight": (text.hidden_size, text.hidden_size),
        "language_model.model.embed_tokens.weight": (text.vocab_size, text.hidden_size),
        "language_model.model.norm.weight": (text.hidden_size,),
        "language_model.lm_head.weight": (text.vocab_size, text.hidden_size),
    }
    for index in range(audio.num_hidden_layers):
        prefix = f"audio_tower.layers.{index}"
        for norm in ("self_attn_layer_norm", "final_layer_norm"):
            shapes[f"{prefix}.{norm}.weight"] = (width,)
            shapes[f"{prefix}.{norm}.bias"] = (width,)
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            shapes[f"{prefix}.self_attn.{projection}.weight"] = (width, width)
            if projection != "k_proj":
                shapes[f"{prefix}.self_attn.{projection}.bias"] = (width,)
        shapes[f"{prefix}.fc1.weight"] = (inner, width)
        shapes[f"{prefix}.fc1.bias"] = (inner,)
        shapes[f"{prefix}.fc2.weight"] = (width, inner)
        shapes[f"{prefix}.fc2.bias"] = (width,)
    hidden, query = text.hidden_size, text.num_attention_heads * text.head_dim
    kv = text.num_key_value_heads * text.head_dim
    for index in range(text.num_hidden_layers):
        prefix = f"language_model.model.layers.{index}"
        shapes[f"{prefix}.input_layernorm.weight"] = (hidden,)
        shapes[f"{prefix}.post_attention_layernorm.weight"] = (hidden,)
        shapes[f"{prefix}.self_attn.q_proj.weight"] = (query, hidden)
        shapes[f"{prefix}.self_attn.k_proj.weight"] = (kv, hidden)
        shapes[f"{prefix}.self_attn.v_proj.weight"] = (kv, hidden)
        shapes[f"{prefix}.self_attn.o_proj.weight"] = (hidden, query)
        shapes[f"{prefix}.mlp.gate_proj.weight"] = (text.intermediate_size, hidden)
        shapes[f"{prefix}.mlp.up_proj.weight"] = (text.intermediate_size, hidden)
        shapes[f"{prefix}.mlp.down_proj.weight"] = (hidden, text.intermediate_size)
    return shapes


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
    _require(isinstance(preprocessor, dict), "preprocessor_config.json must be an object")
    for key, value in expected.items():
        actual = preprocessor.get(key)
        # Exact JSON types: 400.0 or true are not the integers 400 or 1.
        _require(type(actual) is type(value) and actual == value,
                 f"preprocessor_config.{key} must be {value!r}, got {actual!r}")
    dither = preprocessor.get("dither", 0.0)
    _require(type(dither) in (int, float) and dither == 0.0, "dithered features are not supported")
    _require(preprocessor.get("do_normalize", False) is False, "normalized waveforms are not supported")
    # The native front end pads every clip with zeros on the right.
    padding_value = preprocessor.get("padding_value", 0.0)
    _require(type(padding_value) in (int, float) and padding_value == 0.0, "only zero padding is supported")
    _require(preprocessor.get("padding_side", "right") == "right", "only right padding is supported")
    _require(config.mel_frames_per_chunk * HOP_LENGTH == SAMPLE_RATE * CHUNK_SECONDS,
             "mel frames per chunk do not cover the 30 s chunk")


def max_audio_chunks(capacity: int, config: VoxtralConfig, fixed_prompt_tokens: int) -> int:
    """Largest number of 30 s chunks whose prompt still leaves one generated token."""
    chunks = (capacity - fixed_prompt_tokens - 1) // config.audio_tokens_per_chunk
    _require(chunks >= 1, f"max_sequence_length {capacity} cannot hold one 30 s audio chunk")
    return chunks
