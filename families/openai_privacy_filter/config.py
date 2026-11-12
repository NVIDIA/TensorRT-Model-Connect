# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint configuration of the privacy-filter token classifiers.

The Transformers ``openai_privacy_filter`` model is a bidirectional encoder:
every layer attends within ``|i - j| <= sliding_window`` with one learned sink
logit per head, routes each token through ``num_experts_per_tok`` of
``num_local_experts`` clamped-SwiGLU experts, and a linear ``score`` head maps
the final RMS-normalized states to one logit per label.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_TYPE = "openai_privacy_filter"
ARCHITECTURE = "OpenAIPrivacyFilterForTokenClassification"
# Defaults of the Transformers configuration class for keys a checkpoint may omit.
_DEFAULT_SWIGLU_ALPHA = 1.702
_DEFAULT_SWIGLU_LIMIT = 7.0


def _positive_int(raw: dict, key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{MODEL_TYPE} config requires a positive integer {key!r}, got {value!r}")
    return value


def _finite_float(raw: dict, key: str, default: float | None = None) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{MODEL_TYPE} config requires a finite number {key!r}, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class RopeConfig:
    """Rotary schedule: ``default`` or ``yarn`` (NTK-by-parts with a cos/sin gain)."""

    rope_type: str
    theta: float
    factor: float = 1.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    original_max_position_embeddings: int = 0
    truncate: bool = True
    attention_factor: float = 1.0

    @classmethod
    def from_raw(cls, raw: dict) -> "RopeConfig":
        params = raw.get("rope_parameters")
        if not isinstance(params, dict):
            raise ValueError(f"{MODEL_TYPE} config requires rope_parameters")
        rope_type = params.get("rope_type", "default")
        theta = _finite_float(params, "rope_theta", raw.get("rope_theta", 150000.0))
        if rope_type == "default":
            return cls(rope_type="default", theta=theta)
        if rope_type != "yarn":
            raise NotImplementedError(
                f"{MODEL_TYPE} supports default or yarn RoPE, got {rope_type!r}"
            )
        unsupported = {"mscale", "mscale_all_dim", "partial_rotary_factor"} & set(params)
        if unsupported:
            raise NotImplementedError(f"{MODEL_TYPE} YaRN does not implement {sorted(unsupported)}")
        factor = _finite_float(params, "factor")
        original = _positive_int(params, "original_max_position_embeddings")
        if factor <= 0:
            raise ValueError(f"{MODEL_TYPE} YaRN factor must be positive")
        attention_factor = params.get("attention_factor")
        if attention_factor is None:
            attention_factor = 1.0 if factor <= 1 else 0.1 * math.log(factor) + 1.0
        return cls(
            rope_type="yarn",
            theta=theta,
            factor=factor,
            beta_fast=float(params.get("beta_fast") or 32.0),
            beta_slow=float(params.get("beta_slow") or 1.0),
            original_max_position_embeddings=original,
            truncate=bool(params.get("truncate", True)),
            attention_factor=float(attention_factor),
        )

    def inverse_frequencies(self, head_dim: int) -> np.ndarray:
        """Inverse frequencies with the float32 operation order of the Transformers reference."""
        exponents = np.arange(0, head_dim, 2, dtype=np.int64).astype(np.float32) / np.float32(
            head_dim
        )
        pos_freqs = np.power(np.float32(self.theta), exponents, dtype=np.float32)
        extrapolation = np.float32(1.0) / pos_freqs
        if self.rope_type == "default":
            return extrapolation.astype(np.float32)
        interpolation = np.float32(1.0) / (np.float32(self.factor) * pos_freqs)

        def correction_dim(rotations: float) -> float:
            ratio = self.original_max_position_embeddings / (rotations * 2 * math.pi)
            return (head_dim * math.log(ratio)) / (2 * math.log(self.theta))

        low = correction_dim(self.beta_fast)
        high = correction_dim(self.beta_slow)
        if self.truncate:
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, head_dim - 1)
        if low == high:
            high += 0.001
        ramp = (np.arange(head_dim // 2, dtype=np.float32) - np.float32(low)) / np.float32(
            high - low
        )
        extrapolation_factor = np.float32(1.0) - np.clip(ramp, 0, 1).astype(np.float32)
        return (
            interpolation * (np.float32(1.0) - extrapolation_factor)
            + extrapolation * extrapolation_factor
        ).astype(np.float32)

    def tables(self, head_dim: int, positions: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``cos`` and ``sin`` of shape ``[positions, head_dim // 2]`` including the YaRN gain."""
        angles = (
            np.arange(positions, dtype=np.float32)[:, None]
            * self.inverse_frequencies(head_dim)[None, :]
        )
        gain = np.float32(self.attention_factor)
        cos = np.cos(angles.astype(np.float64)).astype(np.float32) * gain
        sin = np.sin(angles.astype(np.float64)).astype(np.float32) * gain
        return np.ascontiguousarray(cos, dtype=np.float32), np.ascontiguousarray(
            sin, dtype=np.float32
        )


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    num_local_experts: int
    num_experts_per_tok: int
    vocab_size: int
    max_position_embeddings: int
    sliding_window: int
    rms_norm_eps: float
    swiglu_alpha: float
    swiglu_limit: float
    attention_bias: bool
    labels: tuple[str, ...]
    rope: RopeConfig

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    @classmethod
    def from_dir(cls, model_dir: Path) -> "ModelConfig":
        raw = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        return cls.from_raw(raw)

    @classmethod
    def from_raw(cls, raw: dict) -> "ModelConfig":
        if raw.get("model_type") != MODEL_TYPE:
            raise ValueError(
                f"{MODEL_TYPE} builder requires model_type {MODEL_TYPE!r}, got {raw.get('model_type')!r}"
            )
        if ARCHITECTURE not in (raw.get("architectures") or ()):
            raise ValueError(f"{MODEL_TYPE} builder requires the {ARCHITECTURE} head")
        id2label = raw.get("id2label")
        if not isinstance(id2label, dict) or not id2label:
            raise ValueError(f"{MODEL_TYPE} config requires id2label")
        try:
            indexed = {int(key): str(value) for key, value in id2label.items()}
        except (TypeError, ValueError) as error:
            raise ValueError(f"{MODEL_TYPE} id2label keys must be integers") from error
        if sorted(indexed) != list(range(len(indexed))):
            raise ValueError(f"{MODEL_TYPE} id2label must number the labels 0..N-1")
        config = cls(
            hidden_size=_positive_int(raw, "hidden_size"),
            num_hidden_layers=_positive_int(raw, "num_hidden_layers"),
            num_attention_heads=_positive_int(raw, "num_attention_heads"),
            num_key_value_heads=_positive_int(raw, "num_key_value_heads"),
            head_dim=_positive_int(raw, "head_dim"),
            intermediate_size=_positive_int(raw, "intermediate_size"),
            num_local_experts=_positive_int(raw, "num_local_experts"),
            num_experts_per_tok=_positive_int(raw, "num_experts_per_tok"),
            vocab_size=_positive_int(raw, "vocab_size"),
            max_position_embeddings=_positive_int(raw, "max_position_embeddings"),
            sliding_window=_positive_int(raw, "sliding_window"),
            rms_norm_eps=_finite_float(raw, "rms_norm_eps"),
            swiglu_alpha=_finite_float(raw, "swiglu_alpha", _DEFAULT_SWIGLU_ALPHA),
            swiglu_limit=_finite_float(raw, "swiglu_limit", _DEFAULT_SWIGLU_LIMIT),
            attention_bias=bool(raw.get("attention_bias", True)),
            labels=tuple(indexed[index] for index in range(len(indexed))),
            rope=RopeConfig.from_raw(raw),
        )
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError(f"{MODEL_TYPE} query heads must be a multiple of the key/value heads")
        if config.num_experts_per_tok > config.num_local_experts:
            raise ValueError(f"{MODEL_TYPE} num_experts_per_tok exceeds num_local_experts")
        if config.head_dim % 2:
            raise ValueError(f"{MODEL_TYPE} rotary embedding requires an even head_dim")
        if raw.get("num_labels") not in (None, config.num_labels):
            raise ValueError(f"{MODEL_TYPE} num_labels disagrees with id2label")
        return config
