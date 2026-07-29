# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mistral family plugin with a native-only TensorRT KV contract."""

from __future__ import annotations

from .config import ModelConfig
from .checkpoint_mapper import WeightDict, load_standard_weights
from ...parallel_config import normalize_parallel_config
from .build_routing import native_kv_build_capability
from .native_decoder_builder import build_native_decoder_engine
from .native_kv_contract import validate_native_kv_weights


class MistralPlugin:
    name = "mistral"
    runtime_strategy = "mistral_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        return model_type.lower().startswith("mistral")

    def default_build_precision(self, config: ModelConfig) -> str:
        del config
        return "bf16"

    def default_max_cache_length(self, config: ModelConfig) -> int:
        """Use the checkpoint's official context without a user build flag."""
        return int(config.max_position_embeddings)

    def supports_split_decoder_roles(self, config: ModelConfig) -> bool:
        del config
        return True

    def load_weights(
        self, model_dir: str, config: ModelConfig,
        *, precision: str = "fp32",
    ) -> WeightDict:
        return load_standard_weights(model_dir, config, precision=precision)

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "bf16",
        quant_ctx=None, verbose: bool = False, debug_layer_outputs: bool = False,
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
            config.raw.pop("_native_kv_cache_metadata", None)
            raise ValueError(
                "Mistral native KV build rejected: " + capability.reason
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
                "native Mistral requires explicit split engine role "
                f"'prefill' or 'decode', got {role!r}"
            )
        return build_native_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision="bf16",
            verbose=verbose,
            profile_mode=role,
        )

    def get_bundle_config_overrides(
        self, config: ModelConfig,
    ) -> dict | None:
        """Mark bundles that satisfy the native runtime contract."""
        metadata = config.raw.get("_native_kv_cache_metadata")
        return dict(metadata) if isinstance(metadata, dict) else None


plugin = MistralPlugin()
