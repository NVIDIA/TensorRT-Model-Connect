# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma 3n checkpoint configuration, validated before any engine is built.

Only the standard library is imported so the support tests can exercise the
parsing rules without TensorRT, NumPy or the checkpoint weights. Defaults follow
the reference ``Gemma3nTextConfig`` when a published config omits a field.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

LAYER_TYPES = ("sliding_attention", "full_attention")
VISION_ARCHITECTURE = "mobilenetv5_300m_enc"
DEFAULT_THETA = {"full_attention": 1_000_000.0, "sliding_attention": 10_000.0}


def _integer(raw: dict, name: str, default=None) -> int:
    value = raw.get(name, default)
    if type(value) is not int or value <= 0:
        raise ValueError(f"Gemma 3n config field {name} must be a positive integer, got {value!r}")
    return value


def _number(raw: dict, name: str, default=None) -> float:
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Gemma 3n config field {name} must be a finite number, got {value!r}")
    return float(value)


def _rope_thetas(raw: dict) -> dict[str, float]:
    """Resolve the per-attention-type RoPE base the reference model uses."""
    if raw.get("rope_scaling") is not None:
        raise NotImplementedError("Gemma 3n RoPE scaling is not implemented")
    parameters = raw.get("rope_parameters")
    if parameters is not None and not isinstance(parameters, dict):
        raise ValueError("Gemma 3n rope_parameters must be an object")
    parameters = parameters or {}
    legacy = {"full_attention": "rope_theta", "sliding_attention": "rope_local_base_freq"}
    thetas = {}
    for kind in LAYER_TYPES:
        entry = parameters.get(kind) or {}
        if not isinstance(entry, dict) or entry.get("rope_type", "default") != "default":
            raise NotImplementedError(f"Gemma 3n {kind} RoPE must be the default rotation")
        if set(entry) - {"rope_type", "rope_theta"}:
            raise NotImplementedError(f"Gemma 3n {kind} RoPE parameters are not implemented")
        theta = entry.get("rope_theta", raw.get(legacy[kind], DEFAULT_THETA[kind]))
        thetas[kind] = _number({"theta": theta}, "theta")
        if thetas[kind] <= 1.0:
            raise ValueError(f"Gemma 3n {kind} RoPE base must exceed one")
    return thetas


@dataclass(frozen=True)
class TextConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_sizes: tuple[int, ...]
    layer_types: tuple[str, ...]
    sliding_window: int
    num_kv_shared_layers: int
    altup_num_inputs: int
    altup_active_idx: int
    altup_correct_scale: bool
    laurel_rank: int
    per_layer_hidden: int
    vocab_size: int
    per_layer_vocab_size: int
    rms_norm_eps: float
    final_logit_softcapping: float | None
    rope_theta: dict
    activation_sparsity: tuple[float, ...]
    max_position_embeddings: int

    @classmethod
    def from_raw(cls, raw: dict) -> "TextConfig":
        if raw.get("model_type") != "gemma3n_text":
            raise ValueError("Gemma 3n text_config must have model_type gemma3n_text")
        if raw.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
            raise NotImplementedError("Gemma 3n requires the gelu_pytorch_tanh activation")
        if raw.get("attention_bias", False):
            raise NotImplementedError("Gemma 3n biased attention projections are not implemented")
        if not raw.get("tie_word_embeddings", True):
            raise NotImplementedError("Gemma 3n requires the tied language-model head")
        layers = _integer(raw, "num_hidden_layers", 35)
        sizes = raw.get("intermediate_size", 16_384)
        sizes = [sizes] * layers if type(sizes) is int else sizes
        if not isinstance(sizes, list) or len(sizes) != layers or any(
                type(size) is not int or size <= 0 for size in sizes):
            raise ValueError("Gemma 3n intermediate_size must give one positive size per layer")
        types = raw.get("layer_types") or [
            "full_attention" if (index + 1) % 5 == 0 else "sliding_attention"
            for index in range(layers)
        ]
        if len(types) != layers or any(kind not in LAYER_TYPES for kind in types):
            raise ValueError("Gemma 3n layer_types must name one known attention type per layer")
        sparsity = raw.get("activation_sparsity_pattern")
        if sparsity is None:
            sparse = 10 if layers > 10 else 0
            sparsity = [0.95] * sparse + [0.0] * (layers - sparse)
        if not isinstance(sparsity, list) or len(sparsity) != layers or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not 0.0 <= value < 1.0 for value in sparsity):
            raise ValueError("Gemma 3n activation_sparsity_pattern must give one value in [0, 1)")
        heads = _integer(raw, "num_attention_heads", 8)
        kv_heads = _integer(raw, "num_key_value_heads", 2)
        if heads % kv_heads:
            raise ValueError("Gemma 3n attention heads must be a multiple of the KV heads")
        head_dim = _integer(raw, "head_dim", 256)
        if head_dim % 2:
            raise ValueError("Gemma 3n head_dim must be even for rotary embeddings")
        shared = raw.get("num_kv_shared_layers", 15)
        if type(shared) is not int or not 0 <= shared < layers:
            raise ValueError("Gemma 3n num_kv_shared_layers must be in [0, num_hidden_layers)")
        altup = _integer(raw, "altup_num_inputs", 4)
        active = raw.get("altup_active_idx", 0)
        if type(active) is not int or not 0 <= active < altup:
            raise ValueError("Gemma 3n altup_active_idx must select one AltUp input")
        if active != 0:
            # The reference adds the per-layer update to inputs 1..N-1, not to
            # every inactive input, so only the published layout is exact.
            raise NotImplementedError("Gemma 3n requires altup_active_idx 0")
        cap = raw.get("final_logit_softcapping", 30.0)
        if cap is not None:
            cap = _number({"cap": cap}, "cap")
            if cap <= 0:
                raise ValueError("Gemma 3n final_logit_softcapping must be positive")
        vocab = _integer(raw, "vocab_size", 262_400)
        per_layer_vocab = _integer(raw, "vocab_size_per_layer_input", 262_144)
        if per_layer_vocab > vocab:
            raise ValueError("Gemma 3n per-layer vocabulary exceeds the text vocabulary")
        return cls(
            hidden_size=_integer(raw, "hidden_size", 2048),
            num_layers=layers,
            num_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            intermediate_sizes=tuple(sizes),
            layer_types=tuple(types),
            sliding_window=_integer(raw, "sliding_window", 512),
            num_kv_shared_layers=shared,
            altup_num_inputs=altup,
            altup_active_idx=active,
            altup_correct_scale=bool(raw.get("altup_correct_scale", True)),
            laurel_rank=_integer(raw, "laurel_rank", 64),
            per_layer_hidden=_integer(raw, "hidden_size_per_layer_input", 256),
            vocab_size=vocab,
            per_layer_vocab_size=per_layer_vocab,
            rms_norm_eps=_number(raw, "rms_norm_eps", 1e-6),
            final_logit_softcapping=cap,
            rope_theta=_rope_thetas(raw),
            activation_sparsity=tuple(float(value) for value in sparsity),
            max_position_embeddings=_integer(raw, "max_position_embeddings", 32_768),
        )

    def kv_owners(self) -> tuple[int, ...]:
        """Layer whose keys and values each layer attends to.

        Layers from the sharing point on reuse the last earlier layer of the
        same attention type, exactly as the reference model selects it.
        """
        first_shared = self.num_layers - self.num_kv_shared_layers
        owners, last = [], {}
        for index, kind in enumerate(self.layer_types):
            if index < first_shared or first_shared == 0:
                last[kind] = index
                owners.append(index)
            elif kind not in last:
                raise ValueError(f"Gemma 3n shared {kind} layer {index} has no earlier owner")
            else:
                owners.append(last[kind])
        return tuple(owners)

    def cache_layers(self) -> tuple[int, ...]:
        """Layers that own a persistent key/value cache."""
        owners = self.kv_owners()
        return tuple(index for index, owner in enumerate(owners) if owner == index)


@dataclass(frozen=True)
class EmbedderConfig:
    """One multimodal embedder: its vocabulary slice and soft-token width."""

    hidden_size: int
    vocab_offset: int
    vocab_size: int
    rms_norm_eps: float

    @classmethod
    def from_raw(cls, raw: dict, label: str) -> "EmbedderConfig":
        offset = raw.get("vocab_offset")
        if type(offset) is not int or offset < 0:
            raise ValueError(f"Gemma 3n {label} vocab_offset must be a non-negative integer")
        return cls(
            hidden_size=_integer(raw, "hidden_size"),
            vocab_offset=offset,
            vocab_size=_integer(raw, "vocab_size"),
            rms_norm_eps=_number(raw, "rms_norm_eps", 1e-6),
        )


@dataclass(frozen=True)
class ModelConfig:
    raw: dict
    text: TextConfig
    vision: EmbedderConfig
    audio: EmbedderConfig
    image_token_id: int
    boi_token_id: int
    eoi_token_id: int
    soft_tokens_per_image: int

    @classmethod
    def from_raw(cls, raw: dict) -> "ModelConfig":
        if raw.get("model_type") != "gemma3n":
            raise ValueError("Gemma 3n family requires model_type gemma3n")
        if "Gemma3nForConditionalGeneration" not in (raw.get("architectures") or []):
            raise ValueError("Gemma 3n family requires Gemma3nForConditionalGeneration")
        for name in ("text_config", "vision_config", "audio_config"):
            if not isinstance(raw.get(name), dict):
                raise ValueError(f"Gemma 3n checkpoint config has no {name}")
        # The conditional-generation model ties its language-model head to the
        # token embeddings by this top-level setting; the decoder computes the
        # logits with the token embeddings.
        if not raw.get("tie_word_embeddings", True):
            raise NotImplementedError("Gemma 3n requires the tied language-model head")
        vision_raw = raw["vision_config"]
        if vision_raw.get("model_type") != "gemma3n_vision":
            raise ValueError("Gemma 3n vision_config must have model_type gemma3n_vision")
        if vision_raw.get("architecture") != VISION_ARCHITECTURE:
            raise NotImplementedError(
                f"Gemma 3n vision tower {vision_raw.get('architecture')!r} is not implemented")
        if vision_raw.get("model_args") not in (None, {}):
            raise NotImplementedError("Gemma 3n vision tower model_args are not implemented")
        text = TextConfig.from_raw(raw["text_config"])
        vision = EmbedderConfig.from_raw(vision_raw, "vision")
        audio = EmbedderConfig.from_raw(raw["audio_config"], "audio")
        if vision.hidden_size != 2048:
            raise ValueError("Gemma 3n MobileNetV5 encoder produces 2048 channels")
        # Text ids below the vision slice use the text table; the vision and
        # audio slices tile the rest of the vocabulary, in this order.
        if not (text.per_layer_vocab_size <= vision.vocab_offset
                and vision.vocab_offset + vision.vocab_size == audio.vocab_offset
                and audio.vocab_offset + audio.vocab_size == text.vocab_size):
            raise ValueError("Gemma 3n multimodal vocabularies do not tile the text vocabulary")
        tokens = {}
        for name in ("image_token_id", "boi_token_id", "eoi_token_id"):
            value = raw.get(name)
            if type(value) is not int or not 0 <= value < text.vocab_size:
                raise ValueError(f"Gemma 3n {name} must be a token id")
            tokens[name] = value
        if not vision.vocab_offset <= tokens["image_token_id"] < audio.vocab_offset:
            raise ValueError("Gemma 3n image placeholder must lie in the vision vocabulary")
        soft = raw.get("vision_soft_tokens_per_image", 256)
        if soft != 256:
            raise NotImplementedError("Gemma 3n MobileNetV5 encoder produces 256 soft tokens")
        return cls(raw=raw, text=text, vision=vision, audio=audio,
                   soft_tokens_per_image=soft, **tokens)

    @classmethod
    def from_dir(cls, root: Path) -> "ModelConfig":
        return cls.from_raw(json.loads((Path(root) / "config.json").read_text(encoding="utf-8")))
