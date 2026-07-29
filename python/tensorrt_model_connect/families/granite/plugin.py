# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Granite family plugin with a single native-KV TensorRT build route.

Granite models (IBM) use the standard LLaMA-style decoder pattern but with
four extra scaling factors that differ from vanilla LLaMA:

  - embedding_multiplier:  scales embedding output (default 1.0)
  - attention_multiplier:  replaces 1/sqrt(head_dim) attention scaling
  - residual_multiplier:   scales attention and MLP outputs before residual add
  - logits_scaling:        divides final logits (default 1.0)

All four are absorbed into the weight tensors at load time so the native
split-engine builder stays on a single KISS FP16 graph.
"""

from __future__ import annotations

import math

import numpy as np

from .config import ModelConfig
from .checkpoint_mapper import WeightDict, load_standard_weights
from .build_routing import (
    native_kv_architecture_capability,
    native_kv_build_capability,
)
from .native_kv_contract import validate_native_kv_weights
from ...parallel_config import normalize_parallel_config
from .native_decoder_builder import build_native_decoder_engine


class GranitePlugin:
    name = "granite"
    runtime_strategy = "granite_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        return model_type.lower().startswith("granite")

    def default_build_precision(self, config: ModelConfig) -> str:
        capability = native_kv_architecture_capability(config)
        return "fp16" if capability.eligible else "fp32"

    def default_max_cache_length(self, config: ModelConfig) -> int:
        """Build eligible Granite models for their complete HF context."""
        capability = native_kv_architecture_capability(config)
        return int(config.max_position_embeddings) if capability.eligible else 256

    def supports_split_decoder_roles(self, config: ModelConfig) -> bool:
        return native_kv_architecture_capability(config).eligible

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        weights = load_standard_weights(model_dir, config, precision=precision)

        raw = config.raw
        embedding_multiplier = raw.get("embedding_multiplier", 1.0)
        attention_multiplier = raw.get("attention_multiplier")
        residual_multiplier = raw.get("residual_multiplier", 1.0)
        logits_scaling = raw.get("logits_scaling", 1.0)

        head_dim = config.head_dim
        standard_attention_scale = 1.0 / math.sqrt(max(head_dim, 1))

        if embedding_multiplier != 1.0:
            weights["embedding"] = weights["embedding"].astype(np.float32) * embedding_multiplier

        if attention_multiplier is not None and attention_multiplier != standard_attention_scale:
            q_scale = attention_multiplier / standard_attention_scale
            for layer_idx in range(config.num_hidden_layers):
                key = f"layer.{layer_idx}.w_q"
                weights[key] = weights[key].astype(np.float32) * q_scale

        if residual_multiplier != 1.0:
            for layer_idx in range(config.num_hidden_layers):
                output_key = f"layer.{layer_idx}.w_o"
                down_key = f"layer.{layer_idx}.w_down"
                weights[output_key] = weights[output_key].astype(np.float32) * residual_multiplier
                weights[down_key] = weights[down_key].astype(np.float32) * residual_multiplier

        if logits_scaling != 1.0:
            weights["w_out"] = weights["w_out"].astype(np.float32) / logits_scaling

        return weights

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
        debug_layer_outputs: bool = False,
        parallel_config=None,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        capability = native_kv_build_capability(
            config,
            precision=precision,
            max_cache_length=max_cache_length,
            parallel_enabled=parallel.enabled,
            quantized=quant_ctx is not None,
            debug_layer_outputs=debug_layer_outputs,
        )
        if not capability.eligible:
            raise NotImplementedError(
                "Granite uses only the TensorRT native KV-cache path; " + capability.reason
            )

        validate_native_kv_weights(config, weights)
        config.raw["_decoder_engine_layout_supported"] = True
        config.raw["_native_kv_cache_metadata"] = {
            "native_kv_contract_version": 1,
            "native_kv_cache": True,
        }
        role = str(config.raw.get("_decoder_engine_role", ""))
        if role not in ("prefill", "decode"):
            raise ValueError(
                "native Granite requires explicit split engine role "
                f"'prefill' or 'decode', got {role!r}"
            )
        return build_native_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision="fp16",
            verbose=verbose,
            profile_mode=role,
        )

    def get_bundle_config_overrides(
        self,
        config: ModelConfig,
    ) -> dict | None:
        """Mark bundles that use the native KV runtime contract."""
        metadata = config.raw.get("_native_kv_cache_metadata")
        return dict(metadata) if isinstance(metadata, dict) else None


plugin = GranitePlugin()
