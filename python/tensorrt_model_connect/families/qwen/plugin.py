# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen family plugin — Qwen, Qwen2, Qwen3, QwQ (text-only, not VL).

Dense Qwen3 uses the family-owned TensorRT native KV path. Other Qwen
variants retain their existing legacy graph routes.
"""

from __future__ import annotations

from .config import ModelConfig
from .checkpoint_mapper import WeightDict, load_standard_weights
from ...parallel_config import normalize_parallel_config
from ...quantization.adapters import StandardDecoderCalibrationAdapter
from .build_routing import (
    native_kv_architecture_capability,
    native_kv_build_capability,
)
from .native_kv_contract import validate_native_kv_weights
from .dual_profile_decoder_builder import build_dual_profile_decoder_engine
from .standard_decoder_builder import build_standard_decoder_engine
from .dual_profile_decoder_tp_builder import build_dual_profile_tp_decoder_engine


class QwenPlugin:
    name = "qwen"
    runtime_strategy = "qwen_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    _CALIBRATION_PROMPTS = [
        "What is the capital of France? Answer in one sentence.",
        "Summarize why photosynthesis is important for life on Earth.",
        "Translate 'Good morning, how are you?' into Chinese.",
        "Write a Python function that checks whether a string is a palindrome.",
        "Explain the difference between RAM and storage in simple terms.",
        "What causes the seasons to change on Earth?",
        "Give three bullet points about the benefits of exercise.",
        "Write a short email asking to reschedule a meeting.",
        "What is the derivative of x^2 + 3x + 1?",
        "Solve this: If a train travels 60 miles in 1.5 hours, what is its average speed?",
        "Describe the plot of Romeo and Juliet in three sentences.",
        "What is the purpose of unit testing in software engineering?",
        "List five countries in South America.",
        "Explain what a GPU does in machine learning.",
        "Write a haiku about the ocean.",
        "What is the boiling point of water at sea level?",
        "Compare democracy and monarchy in two sentences.",
        "Generate a SQL query to select all users created in the last 7 days.",
        "What is Newton's second law?",
        "Describe how to make a peanut butter sandwich.",
        "Why do programmers use version control?",
        "Name three applications of linear algebra.",
        "What is the tallest mountain in the world?",
        "Explain recursion to a beginner.",
        "What is the difference between a list and a tuple in Python?",
        "Write a short product description for wireless headphones.",
        "How does a solar panel generate electricity?",
        "What are the main themes of 1984 by George Orwell?",
        "Give a one-paragraph summary of the water cycle.",
        "Write a polite response declining an invitation.",
        "What is the role of the mitochondria in a cell?",
        "Convert the fraction 3/4 into a percentage.",
    ]

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        # Exclude vision-language variants (handled by qwen_vl plugin),
        # MoE variants (handled by qwen_moe plugin),
        # omni variants (handled by qwen3_omni plugin),
        # Qwen-Image diffusion variants (handled by qwen_image plugin), and
        # Qwen3.5 hybrid variants (handled by qwen3_5 plugin).
        if "vl" in mt or "moe" in mt or "omni" in mt or "image" in mt:
            return False
        if mt in {"qwen3_5", "qwen3.5"}:
            return False
        return mt.startswith("qwen") or mt.startswith("qwq")

    def default_build_precision(self, config: ModelConfig) -> str:
        capability = native_kv_architecture_capability(config)
        return "bf16" if capability.eligible else "fp32"

    def default_max_cache_length(self, config: ModelConfig) -> int:
        """Use the model's complete context for native Qwen3."""
        capability = native_kv_architecture_capability(config)
        return int(config.max_position_embeddings) if capability.eligible else 256

    def supports_split_decoder_roles(self, config: ModelConfig) -> bool:
        """Keep quantized Qwen on the single-engine correctness path."""
        return not bool(config.raw.get("_quantized_build_requested"))

    def load_weights(
        self, model_dir: str, config: ModelConfig,
        *, precision: str = "fp32",
    ) -> WeightDict:
        return load_standard_weights(model_dir, config, precision=precision)

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False, parallel_config=None,
        debug_layer_outputs: bool = False,
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
        if capability.eligible:
            validate_native_kv_weights(config, weights)
            config.raw["_decoder_engine_layout_supported"] = True
            config.raw["_native_kv_cache_metadata"] = {
                "native_kv_contract_version": 1,
                "native_kv_cache": True,
            }
            role = str(
                config.raw.get("_decoder_engine_role", "")
            )
            if role not in ("prefill", "decode"):
                raise ValueError(
                    "native Qwen3 requires explicit split engine role "
                    f"'prefill' or 'decode', got {role!r}"
                )
            return build_dual_profile_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=None,
                verbose=verbose,
                profile_mode=role,
                native_kv_cache=True,
            )

        config.raw.pop("_native_kv_cache_metadata", None)
        if parallel.enabled:
            if debug_layer_outputs:
                raise NotImplementedError(
                    "Qwen tensor-parallel debug layer outputs are not supported")
            return build_dual_profile_tp_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                parallel_config=parallel,
            )
        return build_standard_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
        )

    def get_bundle_config_overrides(
        self, config: ModelConfig,
    ) -> dict | None:
        """Mark bundles that use the native KV runtime contract."""
        metadata = config.raw.get("_native_kv_cache_metadata")
        return dict(metadata) if isinstance(metadata, dict) else None

    def calibration_data(self, format_name: str) -> list[str] | None:
        return list(self._CALIBRATION_PROMPTS)

    def quant_exclude_patterns(self, format_name: str) -> list[str]:
        patterns = [
            "embedding", "final_norm", "w_out", "lm_head",
            "*.input_norm", "*.post_attn_norm", "*_norm*",
        ]
        if format_name == "fp8":
            patterns.extend([
                "layer.*.w_q",
                "layer.*.w_k",
                "layer.*.w_v",
                "layer.*.w_o",
                "layer.*.w_gate",
                "layer.*.w_down",
            ])
        return patterns

    def supports_parallel_quantization(self, format_name: str | None) -> bool:
        return format_name == "fp8"

    def quant_adapter(self, format_name: str) -> StandardDecoderCalibrationAdapter:
        return StandardDecoderCalibrationAdapter(family=self.name)


plugin = QwenPlugin()
