# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone Qwen3-Embedding family plugin."""

from __future__ import annotations

from ...parallel_config import normalize_parallel_config
from .checkpoint_mapper import WeightDict, load_standard_weights
from .config import ModelConfig
from .embedding_builder import build_qwen3_embedding_engine
from .embedding_contract import Qwen3EmbeddingContract, detect_qwen3_embedding_contract


class Qwen3EmbeddingPlugin:
    name = "qwen3_embedding"
    runtime_strategy = "qwen_embedding"
    runtime_capabilities: set[str] = set()

    def matches(self, model_type: str) -> bool:
        return str(model_type).lower().replace("-", "_") == "qwen3_embedding"

    def matches_config(self, config: object) -> bool:
        if not hasattr(config, "raw") or not hasattr(config, "model_type"):
            return False
        return detect_qwen3_embedding_contract(config) is not None

    @staticmethod
    def _contract(config: ModelConfig) -> Qwen3EmbeddingContract:
        cached = getattr(config, "_qwen3_embedding_contract", None)
        if isinstance(cached, Qwen3EmbeddingContract):
            return cached
        contract = detect_qwen3_embedding_contract(config)
        if contract is None:
            raise ValueError(
                "qwen3_embedding requires the pinned Qwen3-Embedding-0.6B "
                "sentence-transformers last-token pooling contract"
            )
        setattr(config, "_qwen3_embedding_contract", contract)
        return contract

    def default_build_precision(self, config: ModelConfig) -> str:
        self._contract(config)
        return "bf16"

    def default_max_cache_length(self, config: ModelConfig) -> int:
        self._contract(config)
        return int(config.max_position_embeddings)

    def supports_split_decoder_roles(self, config: ModelConfig) -> bool:
        self._contract(config)
        return False

    def validate_build_request(self, config: ModelConfig) -> None:
        self._contract(config)
        if config.raw.get("_quantized_build_requested"):
            raise ValueError("Qwen3-Embedding quantized builds are not supported")
        if config.raw.get("_parallel_build_enabled"):
            raise ValueError("Qwen3-Embedding tensor-parallel builds are not supported")
        if config.raw.get("_runtime_dynamic_kv_requested"):
            raise ValueError("Qwen3-Embedding does not use a KV cache")

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        self.validate_build_request(config)
        return load_standard_weights(
            model_dir,
            config,
            precision=precision,
            include_lm_head=False,
        )

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "bf16",
        quant_ctx=None,
        verbose: bool = False,
        parallel_config=None,
        debug_layer_outputs: bool = False,
    ) -> bytes:
        self.validate_build_request(config)
        if normalize_parallel_config(parallel_config).enabled:
            raise ValueError("Qwen3-Embedding tensor-parallel builds are not supported")
        if quant_ctx is not None:
            raise ValueError("Qwen3-Embedding quantized builds are not supported")
        if debug_layer_outputs:
            raise ValueError("Qwen3-Embedding debug layer outputs are not supported")
        return build_qwen3_embedding_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            verbose=verbose,
        )

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        contract = self._contract(config)
        return {
            "runtime_strategy": self.runtime_strategy,
            "embedding_pooling": contract.pooling,
            "embedding_normalize": contract.normalize,
            "embedding_dimension": contract.embedding_dimension,
            "embedding_input_format": contract.input_format,
            "embedding_eos_token_id": contract.eos_token_id,
        }


plugin = Qwen3EmbeddingPlugin()
