# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint contract of a bidirectional ``Ministral3Model`` sentence encoder.

The family accepts exactly the configuration it implements and rejects every
other value: a causal backbone, sliding-window attention, an unknown RoPE type
or a Sentence Transformers head other than mean pooling plus L2 normalization.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RopeConfig:
    """Rotary embedding parameters, in the form Transformers reads them."""

    rope_type: str
    theta: float
    factor: float = 1.0
    original_max_position_embeddings: int = 0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float | None = None
    mscale_all_dim: float | None = None
    attention_factor: float | None = None
    truncate: bool = True
    llama_4_scaling_beta: float = 0.0


@dataclass(frozen=True)
class SentenceHead:
    """The Sentence Transformers modules stacked on the backbone."""

    prompts: dict[str, str]
    max_seq_length: int


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    max_position_embeddings: int
    rope: RopeConfig
    sentence: SentenceHead

    @property
    def attention_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_attention_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "ModelConfig":
        root = Path(model_dir)
        raw = _read_json(root / "config.json")
        return cls.from_dicts(
            raw,
            modules=_read_json(root / "modules.json"),
            pooling=_read_json(root / "1_Pooling" / "config.json"),
            sentence=_read_json(root / "config_sentence_transformers.json"),
            sentence_bert=_read_json(root / "sentence_bert_config.json"),
        )

    @classmethod
    def from_dicts(
        cls,
        raw: dict[str, Any],
        *,
        modules: Any,
        pooling: dict[str, Any],
        sentence: dict[str, Any],
        sentence_bert: dict[str, Any],
    ) -> "ModelConfig":
        if raw.get("model_type") != "ministral3":
            raise ValueError(
                f"ministral3 requires model_type 'ministral3', got {raw.get('model_type')!r}"
            )
        if raw.get("architectures") != ["Ministral3Model"]:
            raise ValueError("ministral3 requires architectures ['Ministral3Model']")
        if raw.get("is_causal") is not False:
            raise ValueError(
                "ministral3 implements the bidirectional encoder only (is_causal: false)"
            )
        if raw.get("sliding_window") is not None:
            raise ValueError("ministral3 does not implement sliding-window attention")
        if raw.get("hidden_act") != "silu":
            raise ValueError(
                f"ministral3 requires hidden_act 'silu', got {raw.get('hidden_act')!r}"
            )
        hidden = _positive_int(raw, "hidden_size")
        heads = _positive_int(raw, "num_attention_heads")
        kv_heads = _positive_int(raw, "num_key_value_heads")
        head_dim = raw.get("head_dim")
        head_dim = hidden // heads if head_dim is None else _positive_int(raw, "head_dim")
        if heads % kv_heads != 0:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        eps = raw.get("rms_norm_eps")
        if (
            not isinstance(eps, (int, float))
            or isinstance(eps, bool)
            or not math.isfinite(eps)
            or not eps > 0
        ):
            raise ValueError("rms_norm_eps must be a positive finite number")
        max_positions = _positive_int(raw, "max_position_embeddings")
        rope = _rope(raw)
        head = _sentence_head(modules, pooling, sentence, sentence_bert, hidden)
        if head.max_seq_length > max_positions:
            raise ValueError("Sentence Transformers max_seq_length exceeds max_position_embeddings")
        return cls(
            vocab_size=_positive_int(raw, "vocab_size"),
            hidden_size=hidden,
            intermediate_size=_positive_int(raw, "intermediate_size"),
            num_hidden_layers=_positive_int(raw, "num_hidden_layers"),
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            rms_norm_eps=float(eps),
            max_position_embeddings=max_positions,
            rope=rope,
            sentence=head,
        )


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"ministral3 checkpoint is missing {path.name} ({path})")
    return json.loads(path.read_text(encoding="utf-8"))


def _positive_int(raw: dict[str, Any], name: str) -> int:
    value = raw.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"ministral3 config requires positive integer {name}")
    return value


def _number(raw: dict[str, Any], name: str, default: float | None = None) -> float | None:
    value = raw.get(name, default)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"rope_parameters.{name} must be a finite number")
    return float(value)


_YARN_KEYS = {
    "rope_type",
    "type",
    "rope_theta",
    "factor",
    "original_max_position_embeddings",
    "beta_fast",
    "beta_slow",
    "mscale",
    "mscale_all_dim",
    "attention_factor",
    "truncate",
    "llama_4_scaling_beta",
    "apply_yarn_scaling",
}


def _rope(raw: dict[str, Any]) -> RopeConfig:
    params = raw.get("rope_parameters")
    if not isinstance(params, dict):
        raise ValueError("ministral3 config requires rope_parameters")
    unknown = set(params) - _YARN_KEYS
    if unknown:
        raise ValueError(f"unsupported rope_parameters keys: {sorted(unknown)}")
    rope_type = params.get("rope_type", params.get("type", "default"))
    # Transformers ignores this key (it only warns); the family follows that
    # reading and refuses the value whose meaning it cannot reproduce.
    if params.get("apply_yarn_scaling", False) is not False:
        raise ValueError("rope_parameters.apply_yarn_scaling must be false or absent")
    theta = _number(params, "rope_theta", raw.get("rope_theta"))
    if theta is None or theta <= 0:
        raise ValueError("rope_theta must be positive")
    beta = _number(params, "llama_4_scaling_beta")
    if beta is None or beta < 0:
        raise ValueError("ministral3 requires rope_parameters.llama_4_scaling_beta >= 0")
    original = params.get("original_max_position_embeddings")
    if not isinstance(original, int) or isinstance(original, bool) or original < 1:
        raise ValueError(
            "rope_parameters.original_max_position_embeddings must be a positive integer"
        )
    # Transformers moves these top-level keys into the RoPE parameters, the
    # original context taking priority over the rope_parameters value.
    if raw.get("partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("ministral3 implements full rotary dimensions only")
    if raw.get("original_max_position_embeddings", original) != original:
        raise ValueError("original_max_position_embeddings disagrees with rope_parameters")
    top_level = raw.get("llama_4_scaling")
    if top_level is not None and (
        not isinstance(top_level, dict)
        or top_level.get("beta") != params.get("llama_4_scaling_beta")
        or top_level.get("original_max_position_embeddings") != original
    ):
        raise ValueError("llama_4_scaling disagrees with rope_parameters")
    if rope_type == "default":
        return RopeConfig(
            rope_type="default",
            theta=theta,
            original_max_position_embeddings=original,
            llama_4_scaling_beta=beta,
        )
    if rope_type != "yarn":
        raise ValueError(
            f"unsupported rope_type {rope_type!r}; ministral3 implements default and yarn"
        )
    factor = _number(params, "factor")
    if factor is None or factor <= 0:
        raise ValueError("yarn rope_parameters.factor must be positive")
    truncate = params.get("truncate", True)
    if not isinstance(truncate, bool):
        raise ValueError("rope_parameters.truncate must be a boolean")
    return RopeConfig(
        rope_type="yarn",
        theta=theta,
        factor=factor,
        original_max_position_embeddings=original,
        beta_fast=_number(params, "beta_fast") or 32.0,
        beta_slow=_number(params, "beta_slow") or 1.0,
        mscale=_number(params, "mscale"),
        mscale_all_dim=_number(params, "mscale_all_dim"),
        attention_factor=_number(params, "attention_factor"),
        truncate=truncate,
        llama_4_scaling_beta=beta,
    )


_POOLING_MODES = (
    "pooling_mode_cls_token",
    "pooling_mode_max_tokens",
    "pooling_mode_mean_tokens",
    "pooling_mode_mean_sqrt_len_tokens",
    "pooling_mode_weightedmean_tokens",
    "pooling_mode_lasttoken",
)


def _sentence_head(
    modules: Any,
    pooling: dict[str, Any],
    sentence: dict[str, Any],
    sentence_bert: dict[str, Any],
    hidden: int,
) -> SentenceHead:
    stack = (
        [(item.get("path"), item.get("type")) for item in modules]
        if isinstance(modules, list)
        else None
    )
    if stack != [
        ("", "sentence_transformers.models.Transformer"),
        ("1_Pooling", "sentence_transformers.models.Pooling"),
        ("2_Normalize", "sentence_transformers.models.Normalize"),
    ]:
        raise ValueError("ministral3 requires the Transformer -> Pooling -> Normalize module stack")
    enabled = [mode for mode in _POOLING_MODES if pooling.get(mode, False)]
    if enabled != ["pooling_mode_mean_tokens"]:
        raise ValueError(f"ministral3 implements mean pooling only, got {enabled}")
    if pooling.get("include_prompt", True) is not True:
        raise ValueError("ministral3 implements pooling that includes the prompt tokens")
    if pooling.get("word_embedding_dimension") != hidden:
        raise ValueError("pooling word_embedding_dimension must equal hidden_size")
    if sentence_bert.get("do_lower_case", False) is not False:
        raise ValueError("ministral3 does not lower-case its input")
    max_seq_length = sentence_bert.get("max_seq_length")
    if (
        not isinstance(max_seq_length, int)
        or isinstance(max_seq_length, bool)
        or max_seq_length < 1
    ):
        raise ValueError("sentence_bert_config.max_seq_length must be a positive integer")
    prompts = sentence.get("prompts")
    if (
        not isinstance(prompts, dict)
        or set(prompts) != {"query", "document"}
        or not all(isinstance(value, str) for value in prompts.values())
    ):
        raise ValueError("config_sentence_transformers.json must define query and document prompts")
    if sentence.get("default_prompt_name") is not None:
        raise ValueError("ministral3 requires default_prompt_name null")
    if sentence.get("similarity_fn_name", "cosine") != "cosine":
        raise ValueError("ministral3 embeddings are compared by cosine similarity")
    return SentenceHead(prompts=dict(prompts), max_seq_length=max_seq_length)
