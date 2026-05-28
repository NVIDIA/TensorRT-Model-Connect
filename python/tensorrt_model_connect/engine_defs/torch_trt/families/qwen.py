"""Qwen family plugin for Torch-TRT — handles Qwen2, Qwen2.5, Qwen3 (non-VL, non-MoE).

This is the reference plugin for standard decoder-only models. Most decoder
families can follow the same pattern: load with AutoModelForCausalLM and
export with the standard cache_config.make_export_args().
"""

from __future__ import annotations

import torch

from ..config import ModelConfig


class QwenTorchTrtPlugin:
    name = "qwen"
    runtime_strategy = "decoder"

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        # Exclude VL, MoE, and Omni variants (need separate plugins)
        if any(x in mt for x in ("vl", "moe", "omni")):
            return False
        return mt.startswith("qwen") or mt.startswith("qwq")

    def load_model(
        self,
        model_dir: str,
        config: ModelConfig,
        max_cache_length: int,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.nn.Module:
        from transformers import AutoModelForCausalLM

        if dtype is None:
            dtype = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=dtype,
            device_map="cuda",
            attn_implementation="eager",
        )
        model.eval()
        return model

    def get_export_args(
        self,
        model: torch.nn.Module,
        config: ModelConfig,
        max_cache_length: int,
        *,
        precision: str = "fp16",
    ) -> tuple:
        from ..cache_config import make_export_args
        # Pass model.config (HF PretrainedConfig) for StaticCache creation
        return make_export_args(model.config, max_cache_length, precision=precision)


plugin = QwenTorchTrtPlugin()
