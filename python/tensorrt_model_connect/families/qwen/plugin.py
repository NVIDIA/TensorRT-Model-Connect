# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen family plugin — Qwen, Qwen2, Qwen3, QwQ (text-only, not VL).

Calls the standard decoder builder, which now dispatches to the
dual-profile builder by default (one engine, two optimization profiles —
profile 0 = batched prefill, profile 1 = single-token decode). Quantized
and TriAttention (``dynamic_kv_cache``) bundles fall back to the legacy
single-profile graph automatically inside the standard builder.
"""

from __future__ import annotations

from .config import ModelConfig
from .checkpoint_mapper import WeightDict, load_standard_weights
from ...parallel_config import normalize_parallel_config
from ...quantization.adapters import StandardDecoderCalibrationAdapter
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
        if quant_ctx is not None:
            for scales in quant_ctx.profile.scale_map.scales.values():
                scales.input_scale = 1.0e6
                scales.weight_scale = 1.0e6
        parallel = normalize_parallel_config(parallel_config)
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

    def calibration_data(self, format_name: str) -> list[str] | None:
        return list(self._CALIBRATION_PROMPTS)

    def quant_exclude_patterns(self, format_name: str) -> list[str]:
        patterns = [
            "embedding", "final_norm", "w_out", "lm_head",
            "*.input_norm", "*.post_attn_norm", "*_norm*",
        ]
        if format_name == "fp8":
            patterns.extend([
                "layer.*.w_o",
                "layer.*.w_gate",
                "layer.*.w_up",
                "layer.*.w_down",
            ])
        return patterns

    def supports_parallel_quantization(self, format_name: str | None) -> bool:
        return format_name == "fp8"

    def quant_adapter(self, format_name: str) -> StandardDecoderCalibrationAdapter:
        return StandardDecoderCalibrationAdapter(family=self.name)


plugin = QwenPlugin()
