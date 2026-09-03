# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin registry adapter for the model-owned K2-Horizon implementation."""

from __future__ import annotations

from .weights import WeightDict, load_standard_weights
from .config import validate_config


def _build_engine(*args, **kwargs) -> bytes:
    from .model import build_engine

    return build_engine(*args, **kwargs)


class K2HorizonPlugin:
    name = "k2_horizon"
    runtime_strategy = "k2_horizon_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        return str(model_type).lower().replace("-", "_") == "k2_horizon"

    def default_build_precision(self, config: object) -> str:
        validate_config(config)
        return "bf16"

    def default_max_cache_length(self, config: object) -> int:
        resolved = validate_config(config)
        return min(256, resolved.max_position_embeddings)

    def validate_build_request(self, config: object) -> None:
        validate_config(config)
        precision = str(config.raw.get("_resolved_build_precision", "bf16")).lower()
        if precision != "bf16":
            raise ValueError("K2-Horizon currently supports only BF16 builds")
        layout = str(config.raw.get("_decoder_engine_layout", "split"))
        if layout != "split":
            raise NotImplementedError(
                "K2-Horizon supports the single-engine fallback selected from "
                "decoder_engine_layout='split'; dual_profile is not supported"
            )
        unsupported = [
            name
            for name, enabled in (
                ("tensor parallel", config.raw.get("_parallel_build_enabled")),
                ("quantization", config.raw.get("_quantized_build_requested")),
                ("RTX specialization", config.raw.get("_rtx_build_requested")),
                ("dynamic KV cache", config.raw.get("_runtime_dynamic_kv_requested")),
                ("mixed FP32 layers", config.raw.get("_fp32_layers")),
            )
            if enabled
        ]
        if unsupported:
            raise NotImplementedError(
                "K2-Horizon does not support build options: " + ", ".join(unsupported)
            )

    def load_weights(
        self,
        model_dir: str,
        config: object,
        *,
        precision: str = "bf16",
    ) -> WeightDict:
        validate_config(config)
        if precision != "bf16":
            raise ValueError("K2-Horizon currently supports only BF16 builds")
        return load_standard_weights(model_dir, config, precision=precision)

    def build_engine(
        self,
        config: object,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "bf16",
        quant_ctx=None,
        verbose: bool = False,
        debug_layer_outputs: bool = False,
        parallel_config=None,
    ) -> bytes:
        if quant_ctx is not None:
            raise NotImplementedError("K2-Horizon does not support quantized builds")
        if debug_layer_outputs:
            raise NotImplementedError("K2-Horizon does not support debug layer outputs")
        if parallel_config is not None and bool(getattr(parallel_config, "enabled", False)):
            raise NotImplementedError("K2-Horizon does not support tensor-parallel builds")
        plan = _build_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            verbose=verbose,
        )
        config.raw["_k2_horizon_bundle_overrides"] = {
            "native_kv_cache": True,
            "native_kv_contract_version": 1,
        }
        return plan

    def get_bundle_config_overrides(self, config: object) -> dict[str, object]:
        validate_config(config)
        overrides = {
            "native_kv_cache": True,
            "native_kv_contract_version": 1,
        }
        stored = config.raw.get("_k2_horizon_bundle_overrides")
        if isinstance(stored, dict):
            overrides.update(stored)
        return overrides


plugin = K2HorizonPlugin()
