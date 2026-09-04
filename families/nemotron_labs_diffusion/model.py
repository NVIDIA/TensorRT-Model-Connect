# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron Labs Diffusion family plugin.

The HF checkpoint is a dense Ministral-style decoder wrapped as
``NemotronLabsDiffusionModel``. Its tensors use ``encoder.*`` and
``diffusion_head.weight`` names, and runtime generation needs full per-position
logits from the prefill profile for diffusion denoising.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .checkpoint_mapper import WeightDict, load_standard_weights
from .config import ModelConfig
from .default_decoder import build_standard_decoder_engine


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _NemotronLabsDiffusionModel:
    lora_engine_section = "linear_spec_lora.plan"

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
        return build_standard_decoder_engine(
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
        try:
            plan = build_standard_decoder_engine(
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
            f"target_modules=['o_proj'], got {sorted(target_modules)}"
        )
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


_BUNDLE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.model",
)


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one Nemotron Labs Diffusion bundle."""
    if request.image_height is not None:
        raise NotImplementedError("nemotron_labs_diffusion does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("nemotron_labs_diffusion does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("nemotron_labs_diffusion does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("nemotron_labs_diffusion does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_generation":
        raise ValueError("nemotron_labs_diffusion supports only task=text_generation")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() != "nemotron_labs_diffusion":
        raise ValueError(
            f"Nemotron Labs Diffusion does not support model_type={config.model_type!r}"
        )
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Nemotron Labs Diffusion precision must be fp32, fp16, or bf16")
    max_sequence_length = int(
        request.max_sequence_length or min(config.max_position_embeddings, 256)
    )
    if max_sequence_length < 1 or max_sequence_length > config.max_position_embeddings:
        raise ValueError(
            "Nemotron Labs Diffusion max_sequence_length is outside checkpoint capacity"
        )
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("Nemotron Labs Diffusion does not support tensor parallelism")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("Nemotron Labs Diffusion has no family-owned quantized build")
    if request.fp32_layers:
        raise NotImplementedError("Nemotron Labs Diffusion does not expose mixed-precision layers")
    model = _NemotronLabsDiffusionModel()
    config.raw["_model_dir"] = str(model_dir)
    weights = model.load_weights(str(model_dir), config, precision=precision)
    plan = model.build_engine(
        config,
        weights,
        max_sequence_length,
        precision=precision,
        quant_ctx=None,
        verbose=bool(request.verbose),
    )
    extra = model.build_extra_engines(
        config,
        weights,
        max_sequence_length,
        precision=precision,
        quant_ctx=None,
        verbose=bool(request.verbose),
    )
    runtime = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": config.pad_token_id,
        "max_cache_length": max_sequence_length,
        "precision": precision,
    }
    lora_config = model.get_lora_config(config)
    if lora_config:
        runtime.update(lora_config)
    writer.set_header(family="nemotron_labs_diffusion", task=request.task, backend="trt")
    writer.add_bytes("engine.plan", plan)
    for name, extra_plan in (extra or {}).items():
        writer.add_bytes(name, extra_plan)
    writer.add_json("runtime.json", runtime)
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
