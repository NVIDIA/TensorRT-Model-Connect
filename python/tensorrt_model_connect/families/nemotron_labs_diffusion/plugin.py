"""Nemotron Labs Diffusion family plugin.

The HF checkpoint is a dense Ministral-style decoder wrapped as
``NemotronLabsDiffusionModel``. Its tensors use ``encoder.*`` and
``diffusion_head.weight`` names, and runtime generation needs full per-position
logits from the prefill profile for diffusion denoising.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from ...checkpoint_mapper import WeightDict, load_standard_weights
from ...config import ModelConfig

build_standard_decoder_engine = None


def _get_standard_decoder_builder():
    global build_standard_decoder_engine
    if build_standard_decoder_engine is None:
        from ..qwen.standard_decoder_builder import (
            build_standard_decoder_engine as imported_builder,
        )
        build_standard_decoder_engine = imported_builder
    return build_standard_decoder_engine


class NemotronLabsDiffusionPlugin:
    name = "nemotron_labs_diffusion"
    runtime_strategy = "nemotron_labs_diffusion"
    lora_engine_section = "linear_spec_lora_engine_plan"

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "nemotron_labs_diffusion"

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        return load_standard_weights(
            model_dir,
            config,
            precision=precision,
            model_prefix="encoder",
            lm_head_key="diffusion_head.weight",
        )

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
    ) -> bytes:
        if config.raw.get("_decoder_engine_role") in (None, "decode"):
            config.raw["_decoder_engine_role"] = "dual_profile"
        config.raw["_decoder_full_logits_output"] = True
        config.raw.setdefault("runtime_strategy", self.runtime_strategy)
        return _get_standard_decoder_builder()(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            norm_type="rmsnorm",
            mlp_type="swiglu",
            position_type="rope",
            activation="silu",
            verbose=verbose,
            full_logits_output=True,
        )

    def build_extra_engines(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
    ) -> dict[str, bytes]:
        lora_dir = Path(str(config.raw.get("_model_dir", ""))) / "linear_spec_lora"
        if not lora_dir.is_dir():
            return {}
        lora_weights = _merge_linear_spec_lora(weights, config, lora_dir, precision=precision)
        previous_role = config.raw.get("_decoder_engine_role")
        previous_full_logits = config.raw.get("_decoder_full_logits_output")
        config.raw["_decoder_engine_role"] = "dual_profile"
        config.raw["_decoder_full_logits_output"] = True
        config.raw.setdefault("runtime_strategy", self.runtime_strategy)
        try:
            plan = _get_standard_decoder_builder()(
                config,
                lora_weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                norm_type="rmsnorm",
                mlp_type="swiglu",
                position_type="rope",
                activation="silu",
                verbose=verbose,
                full_logits_output=True,
            )
        finally:
            if previous_role is None:
                config.raw.pop("_decoder_engine_role", None)
            else:
                config.raw["_decoder_engine_role"] = previous_role
            if previous_full_logits is None:
                config.raw.pop("_decoder_full_logits_output", None)
            else:
                config.raw["_decoder_full_logits_output"] = previous_full_logits
        return {self.lora_engine_section: plan}

    def get_lora_config(self, config: ModelConfig) -> dict | None:
        model_dir = Path(str(config.raw.get("_model_dir", "")))
        if (model_dir / "linear_spec_lora" / "adapter_config.json").is_file():
            return {"linear_spec_lora_engine_section": self.lora_engine_section}
        return None


def _target_np_dtype(precision: str) -> np.dtype:
    return np.float16 if precision in ("fp16", "bf16") else np.float32


def _load_lora_config(lora_dir: Path) -> dict:
    cfg_path = lora_dir / "adapter_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing LoRA adapter config: {cfg_path}")
    return json.loads(cfg_path.read_text())


def _merge_linear_spec_lora(
    weights: WeightDict,
    config: ModelConfig,
    lora_dir: Path,
    *,
    precision: str,
) -> WeightDict:
    adapter_path = lora_dir / "adapter_model.safetensors"
    if not adapter_path.is_file():
        raise FileNotFoundError(f"Missing LoRA adapter weights: {adapter_path}")
    lora_cfg = _load_lora_config(lora_dir)
    target_modules = set(lora_cfg.get("target_modules") or [])
    if target_modules != {"o_proj"}:
        raise ValueError(
            "Nemotron Labs Diffusion linear_spec_lora currently supports only "
            f"target_modules=['o_proj'], got {sorted(target_modules)}")
    rank = int(lora_cfg.get("r", 0))
    if rank <= 0:
        raise ValueError("LoRA rank must be positive")
    scale = float(lora_cfg.get("lora_alpha", rank)) / float(rank)
    out_dtype = _target_np_dtype(precision)
    merged = WeightDict(weights)
    with safe_open(str(adapter_path), framework="numpy") as reader:
        for layer_idx in range(config.num_hidden_layers):
            prefix = f"base_model.model.encoder.layers.{layer_idx}.self_attn.o_proj"
            a_key = f"{prefix}.lora_A.weight"
            b_key = f"{prefix}.lora_B.weight"
            if a_key not in reader.keys() or b_key not in reader.keys():
                raise KeyError(f"Missing LoRA tensors for layer {layer_idx}: {a_key}, {b_key}")
            lora_a = reader.get_tensor(a_key).astype(np.float32)
            lora_b = reader.get_tensor(b_key).astype(np.float32)
            delta_hf = (lora_b @ lora_a) * scale
            weight_key = f"layer.{layer_idx}.w_o"
            merged[weight_key] = (
                weights[weight_key].astype(np.float32, copy=True) + delta_hf.T
            ).astype(out_dtype)
    return merged


plugin = NemotronLabsDiffusionPlugin()
