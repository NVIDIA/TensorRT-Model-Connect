# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict checkpoint mapping for the native Qwen3-Omni Talker stages.

The official checkpoint stores three independently trained pieces used by the
audio-generation path:

* a Thinker-token embedding projection,
* the 20-layer sparse-MoE Talker and its primary codec head, and
* the 5-layer residual code predictor with 15 embedding tables and heads.

This module deliberately maps only those checkpoint tensors.  It has no model
It never imports an implementation from another family.
Missing tensors and shape mismatches fail the build instead of silently tying
weights or substituting defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np

from .checkpoint_mapper import (
    WeightDict,
    _load_tensor_as_dtype,
    _load_transposed_tensor,
    _open_safetensors,
)
from .config import ModelConfig


PREDICTOR_MAX_CACHE_LENGTH = 32
TALKER_MAX_FRAMES = 32


@dataclass(frozen=True)
class NativeTalkerConfigs:
    """Validated decoder and codec configuration used by the native plans."""

    talker: ModelConfig
    predictor: ModelConfig
    num_codebooks: int
    codebook_size: int
    num_experts: int
    experts_per_token: int
    moe_intermediate_size: int
    shared_intermediate_size: int


def storage_dtype(precision: str) -> np.dtype:
    """Return the checkpoint-storage dtype without routing BF16 through FP16."""
    if precision.lower() != "bf16":
        raise ValueError("Qwen3-Omni Talker supports only bf16")
    return np.dtype(ml_dtypes.bfloat16)


def _required_dict(mapping: dict, name: str, owner: str) -> dict:
    value = mapping.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{owner} requires object {name}")
    return value


def _required_int(mapping: dict, name: str, owner: str) -> int:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{owner} requires integer {name}")
    return int(value)


def _required_positive_int(mapping: dict, name: str, owner: str) -> int:
    value = _required_int(mapping, name, owner)
    if value < 1:
        raise ValueError(f"{owner}.{name} must be positive")
    return value


def _required_positive_float(mapping: dict, name: str, owner: str) -> float:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{owner} requires numeric {name}")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{owner}.{name} must be finite and positive")
    return result


def _decoder_config(model_type: str, raw: dict, owner: str) -> ModelConfig:
    hidden = _required_positive_int(raw, "hidden_size", owner)
    heads = _required_positive_int(raw, "num_attention_heads", owner)
    kv_heads = _required_positive_int(raw, "num_key_value_heads", owner)
    head_dim = _required_positive_int(raw, "head_dim", owner)
    if heads % kv_heads != 0:
        raise ValueError(f"{owner}.num_attention_heads must be divisible by num_key_value_heads")
    if head_dim % 2:
        raise ValueError(f"{owner}.head_dim must be even for rotary embedding")
    if raw.get("attention_bias", False) is not False:
        raise ValueError(f"{owner}.attention_bias=true is not supported")
    hidden_act = raw.get("hidden_act")
    if hidden_act != "silu":
        raise ValueError(f"{owner}.hidden_act must be 'silu'")
    return ModelConfig(
        model_type=model_type,
        architectures=[],
        vocab_size=_required_positive_int(raw, "vocab_size", owner),
        hidden_size=hidden,
        intermediate_size=_required_positive_int(raw, "intermediate_size", owner),
        num_hidden_layers=_required_positive_int(raw, "num_hidden_layers", owner),
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        rms_norm_eps=_required_positive_float(raw, "rms_norm_eps", owner),
        rope_theta=_required_positive_float(raw, "rope_theta", owner),
        max_position_embeddings=_required_positive_int(raw, "max_position_embeddings", owner),
        _head_dim=head_dim,
        raw=dict(raw),
    )


def parse_native_talker_configs(root_config: ModelConfig) -> NativeTalkerConfigs:
    """Parse and validate the nested official Talker configuration."""
    root_raw = root_config.raw
    talker_raw = _required_dict(root_raw, "talker_config", "Qwen3-Omni config")
    talker_text = _required_dict(talker_raw, "text_config", "Qwen3-Omni talker_config")
    predictor_raw = _required_dict(talker_raw, "code_predictor_config", "Qwen3-Omni talker_config")
    talker = _decoder_config("qwen3_omni_talker", talker_text, "Qwen3-Omni Talker text_config")
    predictor = _decoder_config(
        "qwen3_omni_code_predictor",
        predictor_raw,
        "Qwen3-Omni code_predictor_config",
    )

    thinker_hidden = _required_positive_int(
        talker_raw, "thinker_hidden_size", "Qwen3-Omni talker_config"
    )
    if thinker_hidden != root_config.hidden_size:
        raise ValueError("Qwen3-Omni Talker thinker_hidden_size does not match the Thinker")

    num_codebooks = _required_positive_int(
        talker_raw, "num_code_groups", "Qwen3-Omni talker_config"
    )
    predictor_groups = _required_positive_int(
        predictor_raw,
        "num_code_groups",
        "Qwen3-Omni code_predictor_config",
    )
    if num_codebooks != predictor_groups or num_codebooks < 2:
        raise ValueError("Qwen3-Omni Talker and code predictor code-group counts do not match")
    codebook_size = predictor.vocab_size

    num_experts = _required_positive_int(
        talker_text, "num_experts", "Qwen3-Omni Talker text_config"
    )
    experts_per_token = _required_positive_int(
        talker_text, "num_experts_per_tok", "Qwen3-Omni Talker text_config"
    )
    if experts_per_token > num_experts:
        raise ValueError("Qwen3-Omni Talker top-k exceeds its expert count")
    if talker_text.get("norm_topk_prob") is not True:
        raise ValueError("Qwen3-Omni Talker requires normalized top-k routing")
    moe_intermediate_size = _required_positive_int(
        talker_text, "moe_intermediate_size", "Qwen3-Omni Talker text_config"
    )
    shared_intermediate_size = _required_positive_int(
        talker_text,
        "shared_expert_intermediate_size",
        "Qwen3-Omni Talker text_config",
    )

    return NativeTalkerConfigs(
        talker=talker,
        predictor=predictor,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        moe_intermediate_size=moe_intermediate_size,
        shared_intermediate_size=shared_intermediate_size,
    )


def validate_thinker_embedding(
    embedding: np.ndarray,
    root_config: ModelConfig,
    *,
    precision: str,
) -> None:
    """Validate the embedding table consumed by ``text_projection.plan``."""
    expected = (root_config.vocab_size, root_config.hidden_size)
    if not isinstance(embedding, np.ndarray) or embedding.shape != expected:
        shape = getattr(embedding, "shape", None)
        raise ValueError(f"Thinker embedding shape {shape} does not match {expected}")
    if precision.lower() == "bf16" and embedding.dtype.name != "bfloat16":
        raise ValueError(
            "BF16 Qwen3-Omni text projection requires an exact bfloat16 "
            "Thinker embedding; an FP16 intermediate loses checkpoint bits"
        )


def _expect_shape(name: str, value: np.ndarray, expected: tuple[int, ...]) -> np.ndarray:
    if value.shape != expected:
        raise ValueError(f"{name} shape {value.shape} does not match {expected}")
    return value


def _load_array(readers, name: str, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    return _expect_shape(name, _load_tensor_as_dtype(readers, name, dtype), shape)


def _load_linear(
    readers,
    name: str,
    *,
    out_features: int,
    in_features: int,
    dtype: np.dtype,
) -> np.ndarray:
    value = _load_transposed_tensor(readers, name, name, dtype)
    return _expect_shape(name, value, (in_features, out_features))


def load_text_projection_weights(
    model_dir: str | Path,
    root_config: ModelConfig,
    configs: NativeTalkerConfigs,
    *,
    precision: str,
) -> dict[str, np.ndarray]:
    """Load the trained Thinker-token to Talker-hidden projection."""
    readers = _open_safetensors(Path(model_dir))
    dtype = storage_dtype(precision)
    thinker_hidden = root_config.hidden_size
    intermediate = configs.talker.intermediate_size
    talker_hidden = configs.talker.hidden_size
    prefix = "talker.text_projection"
    return {
        "fc1": _load_linear(
            readers,
            f"{prefix}.linear_fc1.weight",
            out_features=intermediate,
            in_features=thinker_hidden,
            dtype=dtype,
        ),
        "fc1_bias": _load_array(readers, f"{prefix}.linear_fc1.bias", (intermediate,), dtype),
        "fc2": _load_linear(
            readers,
            f"{prefix}.linear_fc2.weight",
            out_features=talker_hidden,
            in_features=intermediate,
            dtype=dtype,
        ),
        "fc2_bias": _load_array(readers, f"{prefix}.linear_fc2.bias", (talker_hidden,), dtype),
    }


def _map_attention_layer(
    readers,
    weights: WeightDict,
    *,
    source: str,
    target: str,
    config: ModelConfig,
    dtype: np.dtype,
) -> None:
    hidden = config.hidden_size
    attention = config.num_attention_heads * config.head_dim
    kv_attention = config.num_key_value_heads * config.head_dim
    weights[f"{target}.input_norm"] = _load_array(
        readers, f"{source}.input_layernorm.weight", (hidden,), np.dtype(np.float32)
    )
    weights[f"{target}.post_attn_norm"] = _load_array(
        readers,
        f"{source}.post_attention_layernorm.weight",
        (hidden,),
        np.dtype(np.float32),
    )
    for checkpoint_name, target_name, out_features in (
        ("q_proj", "w_q", attention),
        ("k_proj", "w_k", kv_attention),
        ("v_proj", "w_v", kv_attention),
    ):
        weights[f"{target}.{target_name}"] = _load_linear(
            readers,
            f"{source}.self_attn.{checkpoint_name}.weight",
            out_features=out_features,
            in_features=hidden,
            dtype=dtype,
        )
    weights[f"{target}.w_o"] = _load_linear(
        readers,
        f"{source}.self_attn.o_proj.weight",
        out_features=hidden,
        in_features=attention,
        dtype=dtype,
    )
    q_norm = _load_array(
        readers,
        f"{source}.self_attn.q_norm.weight",
        (config.head_dim,),
        np.dtype(np.float32),
    )
    k_norm = _load_array(
        readers,
        f"{source}.self_attn.k_norm.weight",
        (config.head_dim,),
        np.dtype(np.float32),
    )
    weights[f"{target}.q_norm"] = np.tile(q_norm, config.num_attention_heads).astype(
        np.float32, copy=False
    )
    weights[f"{target}.k_norm"] = np.tile(k_norm, config.num_key_value_heads).astype(
        np.float32, copy=False
    )


def load_talker_weights(
    model_dir: str | Path,
    configs: NativeTalkerConfigs,
    *,
    precision: str,
) -> tuple[WeightDict, np.ndarray]:
    """Load the complete sparse-MoE Talker and its distinct primary embedding."""
    readers = _open_safetensors(Path(model_dir))
    dtype = storage_dtype(precision)
    config = configs.talker
    hidden = config.hidden_size
    vocab = config.vocab_size

    codec_embedding = _load_array(
        readers,
        "talker.model.codec_embedding.weight",
        (vocab, hidden),
        np.dtype(np.float32),
    )
    weights = WeightDict()
    weights["w_out"] = _load_linear(
        readers,
        "talker.codec_head.weight",
        out_features=vocab,
        in_features=hidden,
        dtype=dtype,
    )
    weights["final_norm"] = _load_array(
        readers, "talker.model.norm.weight", (hidden,), np.dtype(np.float32)
    )

    for layer in range(config.num_hidden_layers):
        source = f"talker.model.layers.{layer}"
        target = f"layer.{layer}"
        _map_attention_layer(
            readers,
            weights,
            source=source,
            target=target,
            config=config,
            dtype=dtype,
        )
        weights[f"{target}.router"] = _load_linear(
            readers,
            f"{source}.mlp.gate.weight",
            out_features=configs.num_experts,
            in_features=hidden,
            dtype=dtype,
        )

        expert_gate = np.empty(
            (configs.num_experts, hidden, configs.moe_intermediate_size), dtype=dtype
        )
        expert_up = np.empty_like(expert_gate)
        expert_down = np.empty(
            (configs.num_experts, configs.moe_intermediate_size, hidden), dtype=dtype
        )
        for expert in range(configs.num_experts):
            expert_source = f"{source}.mlp.experts.{expert}"
            expert_gate[expert] = _load_linear(
                readers,
                f"{expert_source}.gate_proj.weight",
                out_features=configs.moe_intermediate_size,
                in_features=hidden,
                dtype=dtype,
            )
            expert_up[expert] = _load_linear(
                readers,
                f"{expert_source}.up_proj.weight",
                out_features=configs.moe_intermediate_size,
                in_features=hidden,
                dtype=dtype,
            )
            expert_down[expert] = _load_linear(
                readers,
                f"{expert_source}.down_proj.weight",
                out_features=hidden,
                in_features=configs.moe_intermediate_size,
                dtype=dtype,
            )
        weights[f"{target}.experts.w_gate"] = expert_gate
        weights[f"{target}.experts.w_up"] = expert_up
        weights[f"{target}.experts.w_down"] = expert_down

        shared_source = f"{source}.mlp.shared_expert"
        for checkpoint_name, target_name in (
            ("gate_proj", "w_gate"),
            ("up_proj", "w_up"),
        ):
            weights[f"{target}.shared.{target_name}"] = _load_linear(
                readers,
                f"{shared_source}.{checkpoint_name}.weight",
                out_features=configs.shared_intermediate_size,
                in_features=hidden,
                dtype=dtype,
            )
        weights[f"{target}.shared.w_down"] = _load_linear(
            readers,
            f"{shared_source}.down_proj.weight",
            out_features=hidden,
            in_features=configs.shared_intermediate_size,
            dtype=dtype,
        )
        weights[f"{target}.shared_expert_gate"] = _load_linear(
            readers,
            f"{source}.mlp.shared_expert_gate.weight",
            out_features=1,
            in_features=hidden,
            dtype=dtype,
        )

    weights["_attention_size"] = config.num_attention_heads * config.head_dim
    weights["_kv_attention_size"] = config.num_key_value_heads * config.head_dim
    weights["_num_experts"] = configs.num_experts
    weights["_num_experts_per_tok"] = configs.experts_per_token
    weights["_moe_intermediate_size"] = configs.moe_intermediate_size
    weights["_shared_intermediate_size"] = configs.shared_intermediate_size
    return weights, codec_embedding


def load_predictor_weights(
    model_dir: str | Path,
    configs: NativeTalkerConfigs,
    *,
    precision: str,
) -> tuple[WeightDict, list[np.ndarray]]:
    """Load all five predictor layers and all 15 residual tables/heads."""
    readers = _open_safetensors(Path(model_dir))
    dtype = storage_dtype(precision)
    config = configs.predictor
    hidden = config.hidden_size
    vocab = config.vocab_size
    residual_groups = configs.num_codebooks - 1

    codec_embeddings = [
        _load_array(
            readers,
            f"talker.code_predictor.model.codec_embedding.{group}.weight",
            (vocab, hidden),
            np.dtype(np.float32),
        )
        for group in range(residual_groups)
    ]
    output_heads = [
        _load_linear(
            readers,
            f"talker.code_predictor.lm_head.{group}.weight",
            out_features=vocab,
            in_features=hidden,
            dtype=dtype,
        )
        for group in range(residual_groups)
    ]

    weights = WeightDict()
    weights["final_norm"] = _load_array(
        readers,
        "talker.code_predictor.model.norm.weight",
        (hidden,),
        np.dtype(np.float32),
    )
    for layer in range(config.num_hidden_layers):
        source = f"talker.code_predictor.model.layers.{layer}"
        target = f"layer.{layer}"
        _map_attention_layer(
            readers,
            weights,
            source=source,
            target=target,
            config=config,
            dtype=dtype,
        )
        for checkpoint_name, target_name in (
            ("gate_proj", "w_gate"),
            ("up_proj", "w_up"),
        ):
            weights[f"{target}.{target_name}"] = _load_linear(
                readers,
                f"{source}.mlp.{checkpoint_name}.weight",
                out_features=config.intermediate_size,
                in_features=hidden,
                dtype=dtype,
            )
        weights[f"{target}.w_down"] = _load_linear(
            readers,
            f"{source}.mlp.down_proj.weight",
            out_features=hidden,
            in_features=config.intermediate_size,
            dtype=dtype,
        )

    weights["_attention_size"] = config.num_attention_heads * config.head_dim
    weights["_kv_attention_size"] = config.num_key_value_heads * config.head_dim
    weights["_output_heads"] = output_heads
    return weights, codec_embeddings


def _runtime_int(mapping: dict, name: str, owner: str) -> int:
    return _required_int(mapping, name, owner)


def native_talker_runtime_fields(
    root_config: ModelConfig,
    configs: NativeTalkerConfigs,
    *,
    max_cache_length: int,
) -> dict[str, object]:
    """Return the flat Talker fields consumed by the family-owned runtime."""
    if isinstance(max_cache_length, bool) or max_cache_length < 1:
        raise ValueError("Qwen3-Omni Talker max_cache_length must be positive")
    if max_cache_length > configs.talker.max_position_embeddings:
        raise ValueError("Qwen3-Omni Talker cache exceeds checkpoint capacity")

    raw = root_config.raw
    talker_raw = _required_dict(raw, "talker_config", "Qwen3-Omni config")
    speakers = _required_dict(talker_raw, "speaker_id", "Qwen3-Omni talker_config")
    speaker_id = _runtime_int(speakers, "ethan", "Qwen3-Omni speaker_id")
    return {
        "talker_hidden_size": configs.talker.hidden_size,
        "talker_num_layers": configs.talker.num_hidden_layers,
        "talker_num_attention_heads": configs.talker.num_attention_heads,
        "talker_num_key_value_heads": configs.talker.num_key_value_heads,
        "talker_head_dim": configs.talker.head_dim,
        "talker_vocab_size": configs.talker.vocab_size,
        "talker_max_cache_length": max_cache_length,
        "predictor_hidden_size": configs.predictor.hidden_size,
        "predictor_num_layers": configs.predictor.num_hidden_layers,
        "predictor_num_attention_heads": configs.predictor.num_attention_heads,
        "predictor_num_key_value_heads": configs.predictor.num_key_value_heads,
        "predictor_head_dim": configs.predictor.head_dim,
        "predictor_vocab_size": configs.predictor.vocab_size,
        "predictor_max_cache_length": PREDICTOR_MAX_CACHE_LENGTH,
        "num_codebooks": configs.num_codebooks,
        "codebook_size": configs.codebook_size,
        "talker_max_frames": TALKER_MAX_FRAMES,
        "im_start_token_id": _runtime_int(raw, "im_start_token_id", "Qwen3-Omni config"),
        "system_token_id": _runtime_int(raw, "system_token_id", "Qwen3-Omni config"),
        "user_token_id": _runtime_int(raw, "user_token_id", "Qwen3-Omni config"),
        "assistant_token_id": _runtime_int(raw, "assistant_token_id", "Qwen3-Omni config"),
        "tts_bos_token_id": _runtime_int(raw, "tts_bos_token_id", "Qwen3-Omni config"),
        "tts_eos_token_id": _runtime_int(raw, "tts_eos_token_id", "Qwen3-Omni config"),
        "tts_pad_token_id": _runtime_int(raw, "tts_pad_token_id", "Qwen3-Omni config"),
        "codec_bos_id": _runtime_int(talker_raw, "codec_bos_id", "Qwen3-Omni talker_config"),
        "codec_eos_token_id": _runtime_int(
            talker_raw, "codec_eos_token_id", "Qwen3-Omni talker_config"
        ),
        "codec_nothink_id": _runtime_int(
            talker_raw, "codec_nothink_id", "Qwen3-Omni talker_config"
        ),
        "codec_pad_id": _runtime_int(talker_raw, "codec_pad_id", "Qwen3-Omni talker_config"),
        "codec_think_bos_id": _runtime_int(
            talker_raw, "codec_think_bos_id", "Qwen3-Omni talker_config"
        ),
        "codec_think_eos_id": _runtime_int(
            talker_raw, "codec_think_eos_id", "Qwen3-Omni talker_config"
        ),
        "speaker_id": speaker_id,
    }


__all__ = [
    "NativeTalkerConfigs",
    "PREDICTOR_MAX_CACHE_LENGTH",
    "TALKER_MAX_FRAMES",
    "load_predictor_weights",
    "load_talker_weights",
    "load_text_projection_weights",
    "native_talker_runtime_fields",
    "parse_native_talker_configs",
    "storage_dtype",
    "validate_thinker_embedding",
]
