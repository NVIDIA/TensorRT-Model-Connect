# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron Labs Diffusion family-owned build implementation.

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

from .checkpoint_mapper import WeightDict, load_standard_weights
from .config import ModelConfig

build_standard_decoder_engine = None


def _get_standard_decoder_builder():
    global build_standard_decoder_engine
    if build_standard_decoder_engine is None:
        from .default_decoder import (
            build_standard_decoder_engine as imported_builder,
        )

        build_standard_decoder_engine = imported_builder
    return build_standard_decoder_engine


name = "nemotron_labs_diffusion"
runtime_strategy = "nemotron_labs_diffusion"
lora_engine_section = "linear_spec_lora_engine_plan"


def matches(config) -> bool:
    model_type = str(getattr(config, "model_type", config))
    return model_type.lower() == "nemotron_labs_diffusion"


def load_weights(model_dir: str, config: ModelConfig, *, precision: str = "fp32") -> WeightDict:
    return load_standard_weights(
        model_dir,
        config,
        precision=precision,
        model_prefix="encoder",
        lm_head_key="diffusion_head.weight",
    )


def build_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx=None,
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    if config.raw.get("_decoder_engine_role") in (None, "decode"):
        config.raw["_decoder_engine_role"] = "dual_profile"
    config.raw["_decoder_full_logits_output"] = True
    config.raw.setdefault("runtime_strategy", runtime_strategy)
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
    config.raw.setdefault("runtime_strategy", runtime_strategy)
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
    return {lora_engine_section: plan}


def get_lora_config(config: ModelConfig) -> dict | None:
    model_dir = Path(str(config.raw.get("_model_dir", "")))
    if (model_dir / "linear_spec_lora" / "adapter_config.json").is_file():
        return {"linear_spec_lora_engine_section": lora_engine_section}
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


# Model-owned build entry
requires_tokenizer = True
embed_input = False


def _build_local_engine(
    config,
    weights,
    max_cache_length,
    precision,
    quant_ctx,
    verbose,
    parallel,
):
    from tensorrt_model_connect.tvm_ffi.graph_build import engine_role, inspection_role

    def build_role(role: str):
        previous = config.raw.get("_decoder_engine_role")
        config.raw["_decoder_engine_role"] = role
        try:
            with engine_role(role):
                return build_engine(
                    config,
                    weights,
                    max_cache_length,
                    precision=precision,
                    quant_ctx=quant_ctx,
                    verbose=verbose,
                )
        finally:
            if previous is None:
                config.raw.pop("_decoder_engine_role", None)
            else:
                config.raw["_decoder_engine_role"] = previous

    target_role = inspection_role()
    if target_role is not None:
        build_role(target_role)
        raise RuntimeError("graph inspection did not reach TensorRT serialization")
    selected_role = (
        "dual_profile"
        if str(config.raw.get("_decoder_engine_layout") or "split") == "dual_profile"
        else "decode"
    )
    return build_role(selected_role), (
        "dual_profile" if selected_role == "dual_profile" else "single"
    )


def build(model_dir: str, output_path: str, **options) -> None:
    """Build the complete nemotron_labs_diffusion bundle in the owning family module."""
    import json
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    from tensorrt_model_connect.build_timing import (
        add_build_timing,
        build_timing_phase,
        new_build_timing,
        untracked_phase_time,
        write_build_timing,
    )
    from tensorrt_model_connect.bundle_writer import (
        BundleInfo,
        BundleSection,
        gpu_name,
        tensorrt_abi,
        tensorrt_version,
        write_bundle,
    )
    from tensorrt_model_connect.parallel_config import (
        normalize_parallel_config,
    )
    from tensorrt_model_connect.tokenizer_conversion import (
        detect_tokenizer_add_special_tokens,
        prepare_tokenizer_special_frame,
    )

    model_path = Path(model_dir)
    decoder_engine_layout = str(options.get("decoder_engine_layout") or "split")
    if decoder_engine_layout not in {"split", "dual_profile"}:
        raise ValueError(
            "decoder_engine_layout must be 'split' or 'dual_profile', "
            f"got {decoder_engine_layout!r}"
        )
    parallel = normalize_parallel_config(options.get("parallel_config"))
    if parallel.cp_enabled:
        raise NotImplementedError(f"{name} does not support context-parallel builds")
    if options.get("dynamic_kv_cache") or options.get("triattention_stats_path"):
        raise ValueError(f"{name} does not use a decoder KV-cache runtime")

    config = ModelConfig.from_dir(model_path)
    config.raw["_model_dir"] = str(model_path)
    config.raw["_decoder_engine_layout"] = decoder_engine_layout
    config.raw["_fp32_layers"] = sorted(set(options.get("fp32_layers") or ()))
    config.raw["_family_build_options"] = dict(options.get("family_build_options") or {})
    config.raw["_parallel_build_enabled"] = bool(parallel.enabled)
    config.raw["_rtx_build_requested"] = bool(options.get("rtx"))
    config.raw["_runtime_dynamic_kv_requested"] = False
    config.raw["_quantized_build_requested"] = bool(options.get("quantize"))
    precision = str(options.get("precision") or "fp32").lower()
    config.raw["_resolved_build_precision"] = precision
    requested_cache_length = options.get("max_cache_length")
    max_cache_length = int(256 if requested_cache_length is None else requested_cache_length)
    if max_cache_length < 1:
        raise ValueError("max_cache_length must be >= 1")

    timing = new_build_timing(options.get("build_timing_path"))
    timing["model_dir"] = str(model_path)
    timing["output_path"] = str(output_path)
    started = time.monotonic()
    write_build_timing(timing)

    weights_started = time.monotonic()
    weights = load_weights(str(model_path), config, precision=precision)
    add_build_timing(timing, "weights_loading_s", time.monotonic() - weights_started)
    write_build_timing(timing)

    quantize = options.get("quantize")
    quant_ctx = None
    quant_plan = None
    if quantize:
        from dataclasses import replace
        from tensorrt_model_connect.quantization import QuantPlan, build_quant_context
        from . import graph_ops as family_graph_ops

        quant_plan = QuantPlan.from_build_args(
            precision=precision,
            quantize=str(quantize),
            quant_scales=options.get("quant_scales"),
            quant_calibration_samples=int(options.get("quant_calibration_samples") or 512),
        )
        quant_method = str(
            config.raw.get("quantization_config", {}).get("quant_method", "")
        ).lower()
        if quant_plan.scale_source == "modelopt" and quant_method in {
            "awq",
            "gptq",
            "compressed-tensors",
            "compressed_tensors",
        }:
            quant_plan = replace(quant_plan, scale_source="prequantized")
        quant_ctx = build_quant_context(
            format_name=quant_plan.quant_format,
            model_dir=str(model_path),
            config=config,
            scales_json=options.get("quant_scales"),
            num_calibration_samples=int(options.get("quant_calibration_samples") or 512),
            quant_plan=quant_plan,
            graph_ops=family_graph_ops,
        )

    if parallel.enabled:
        raise ValueError(f"{name} does not support tensor-parallel builds")

    verbose = bool(options.get("verbose"))
    compile_started = time.monotonic()

    sections = []
    plan, decoder_layout = _build_local_engine(
        config, weights, max_cache_length, precision, quant_ctx, verbose, parallel
    )
    sections.append(BundleSection("engine_plan", plan))

    compile_elapsed = time.monotonic() - compile_started
    add_build_timing(timing, "trt_compile_s", compile_elapsed)
    add_build_timing(timing, "trt_compile_main_engine_s", compile_elapsed)
    write_build_timing(timing)

    vision_plan = None

    extra_started = time.monotonic()
    compile_before_extra = build_timing_phase(timing, "trt_compile_s")
    extra_engines = (
        build_extra_engines(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            verbose=verbose,
        )
        or {}
    )
    extra_elapsed = time.monotonic() - extra_started
    add_build_timing(
        timing,
        "trt_compile_s",
        untracked_phase_time(extra_elapsed, compile_before_extra, timing, "trt_compile_s"),
    )
    add_build_timing(timing, "trt_compile_extra_engines_s", extra_elapsed)
    write_build_timing(timing)
    sections.extend(
        BundleSection(section_name, section_data)
        for section_name, section_data in extra_engines.items()
    )

    tokenizer_frame = prepare_tokenizer_special_frame(
        model_path,
        source_model_id_or_path=str(options.get("tokenizer_source_model_id_or_path") or model_path),
        source_revision=(
            str(options["tokenizer_source_revision"])
            if options.get("tokenizer_source_revision")
            else None
        ),
    )

    if tokenizer_frame is None:
        prefix_ids, suffix_ids = [], []
        add_special_tokens = detect_tokenizer_add_special_tokens(model_path)
    else:
        prefix_ids, suffix_ids = tokenizer_frame
        add_special_tokens = bool(prefix_ids or suffix_ids)

    trt_version = tensorrt_version()
    trt_abi_value = tensorrt_abi(trt_version)
    info = BundleInfo(
        model_id=model_path.name,
        model_type=config.model_type,
        family=name,
        trt_version=trt_version,
        trt_abi=trt_abi_value,
        gpu_name=gpu_name(),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_cache_length=max_cache_length,
        runtime_strategy=runtime_strategy,
        precision=precision,
        quantization=(quant_plan.quant_format if quant_plan else "none"),
        tokenizer_add_special_tokens=add_special_tokens,
    )

    source_config = model_path / "config.json"
    runtime_config = (
        json.loads(source_config.read_text(encoding="utf-8"))
        if source_config.is_file()
        else {key: value for key, value in config.raw.items() if not str(key).startswith("_")}
    )
    generation_config = model_path / "generation_config.json"
    if generation_config.is_file():
        generation = json.loads(generation_config.read_text(encoding="utf-8"))
        if "eos_token_id" in generation:
            runtime_config["eos_token_id"] = generation["eos_token_id"]
    runtime_config.update(
        {
            "runtime_strategy": runtime_strategy,
            "engine_backend": "trt_rtx" if options.get("rtx") else "trt",
            "trt_version": trt_version,
            "precision": precision,
            "tokenizer_add_special_tokens": int(add_special_tokens),
            "decoder_engine_layout": decoder_layout,
        }
    )
    if trt_abi_value:
        runtime_config["trt_abi"] = trt_abi_value
    if tokenizer_frame is not None:
        runtime_config["tokenizer_special_prefix_ids"] = prefix_ids
        runtime_config["tokenizer_special_suffix_ids"] = suffix_ids
    if options.get("fp32_layers"):
        runtime_config["fp32_layers"] = sorted(set(options["fp32_layers"]))
    if quant_plan is not None:
        runtime_config["quantization"] = quant_plan.as_config_dict()
    runtime_config.update(parallel.to_bundle_config_fields())
    if vision_plan is not None:
        runtime_config["has_vision_engine"] = True
    lora_config = get_lora_config(config)
    if lora_config is not None:
        runtime_config.update(lora_config)

    from tensorrt_model_connect.tvm_ffi.graph_build import kernel_slots_section

    slot_section = kernel_slots_section()
    if slot_section is not None:
        sections.append(BundleSection("kernel_slots.json", slot_section))

    sections.append(
        BundleSection("config.json", json.dumps(runtime_config, indent=2).encode("utf-8"))
    )
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.model",
        "preprocessor_config.json",
        "processor_config.json",
    ):
        path = model_path / filename
        if path.is_file():
            sections.append(BundleSection(filename, path.read_bytes()))

    kernel_manifest = []
    for global_name, library in options.get("kernel_artifacts") or ():
        section_name = f"kernel_{global_name.replace('.', '_')}.so"
        sections.append(BundleSection(section_name, Path(library).read_bytes()))
        kernel_manifest.append(
            {
                "global_name": global_name,
                "func_name": "run",
                "section": section_name,
            }
        )
    if kernel_manifest:
        sections.append(
            BundleSection(
                "kernel_manifest.json",
                json.dumps({"kernels": kernel_manifest}).encode("utf-8"),
            )
        )

    write_started = time.monotonic()
    write_bundle(output_path, info, sections)
    add_build_timing(timing, "bundle_write_s", time.monotonic() - write_started)
    timing["total_s"] = time.monotonic() - started
    write_build_timing(timing)
