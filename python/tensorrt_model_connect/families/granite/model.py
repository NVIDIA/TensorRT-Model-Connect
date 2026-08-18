# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Granite family model — absorbs Granite-specific multipliers into weights.

Granite models (IBM) use the standard LLaMA-style decoder pattern but with
four extra scaling factors that differ from vanilla LLaMA:

  - embedding_multiplier:  scales embedding output (default 1.0)
  - attention_multiplier:  replaces 1/sqrt(head_dim) attention scaling
  - residual_multiplier:   scales attention and MLP outputs before residual add
  - logits_scaling:        divides final logits (default 1.0)

All four are absorbed into the weight tensors at load time so the standard
decoder builder can be reused without modification.
"""

from __future__ import annotations

import math

import numpy as np

from .config import ModelConfig
from .checkpoint_mapper import WeightDict, load_standard_weights
from ...parallel_config import normalize_parallel_config
from .dual_profile_decoder_tp_builder import build_dual_profile_tp_decoder_engine
from .standard_decoder_builder import build_standard_decoder_engine


name = "granite"
runtime_strategy = "granite_decoder_kv_cache"
runtime_capabilities = {"decoder_kv"}


def matches(config: object) -> bool:
    """Return whether this module owns the parsed model config."""
    model_type = str(getattr(config, "model_type", config))
    return model_type.lower().startswith("granite")


def load_weights(model_dir: str, config: ModelConfig, *, precision: str = "fp32") -> WeightDict:
    weights = load_standard_weights(model_dir, config, precision=precision)

    raw = config.raw
    embedding_multiplier = raw.get("embedding_multiplier", 1.0)
    attention_multiplier = raw.get("attention_multiplier", None)
    residual_multiplier = raw.get("residual_multiplier", 1.0)
    logits_scaling = raw.get("logits_scaling", 1.0)

    head_dim = config.head_dim
    standard_attn_scale = 1.0 / math.sqrt(max(head_dim, 1))

    # Fix 1: Granite scales embedding output by embedding_multiplier.
    if embedding_multiplier != 1.0:
        weights["embedding"] = weights["embedding"].astype(np.float32) * embedding_multiplier

    # Fix 2: Granite uses attention_multiplier instead of 1/sqrt(head_dim).
    # Absorb the ratio into Q projection weights so the standard builder's
    # 1/sqrt(head_dim) scaling produces the correct result.
    if attention_multiplier is not None and attention_multiplier != standard_attn_scale:
        q_scale = attention_multiplier / standard_attn_scale
        for layer_idx in range(config.num_hidden_layers):
            key = f"layer.{layer_idx}.w_q"
            weights[key] = weights[key].astype(np.float32) * q_scale

    # Fix 3: Granite multiplies attention and MLP outputs by
    # residual_multiplier before the residual add:
    #   hidden = residual + attn_out * residual_multiplier
    #   hidden = residual + mlp_out * residual_multiplier
    # Absorb into the output projections (w_o and w_down).
    if residual_multiplier != 1.0:
        for layer_idx in range(config.num_hidden_layers):
            o_key = f"layer.{layer_idx}.w_o"
            d_key = f"layer.{layer_idx}.w_down"
            weights[o_key] = weights[o_key].astype(np.float32) * residual_multiplier
            weights[d_key] = weights[d_key].astype(np.float32) * residual_multiplier

    # Fix 4: Granite divides final logits by logits_scaling.
    # Absorb into the output (lm_head) weight matrix.
    if logits_scaling != 1.0:
        weights["w_out"] = weights["w_out"].astype(np.float32) / logits_scaling

    return weights


def build_engine(
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
    if parallel.enabled:
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


def supports_split_decoder_roles(config: ModelConfig) -> bool:
    return True


def _dynamic_kv_profile_rows(
    max_cache_length: int,
    kv_budget: int,
    *,
    bucket_rows: int = 32,
    preferred_rows: list[int] | None = None,
) -> list[int]:
    if max_cache_length < 1:
        return [1]
    start = ((max(kv_budget, 1) + bucket_rows - 1) // bucket_rows) * bucket_rows
    start = max(bucket_rows, min(start, max_cache_length))
    rows: list[int] = []

    def add_row(value: int) -> None:
        rounded = (
            (min(max(value, 1), max_cache_length) + bucket_rows - 1) // bucket_rows
        ) * bucket_rows
        rounded = max(bucket_rows, min(rounded, max_cache_length))
        if rounded not in rows:
            rows.append(rounded)

    for value in preferred_rows or ():
        add_row(value)
    row = start
    while row < max_cache_length:
        add_row(row)
        row = (
            (min(max(row + bucket_rows, row * 2), max_cache_length) + bucket_rows - 1)
            // bucket_rows
        ) * bucket_rows
    add_row(max_cache_length)
    return sorted(rows)


def _sanitize_dynamic_kv_profile_rows(
    rows: list[int] | None,
    max_cache_length: int,
) -> list[int] | None:
    if rows is None:
        return None
    sanitized = sorted({max(1, min(int(value), max_cache_length)) for value in rows})
    if not sanitized:
        raise ValueError("dynamic_kv_profile_rows_override must contain at least one row")
    return sanitized


def build(model_dir: str, output_path: str, **options) -> None:
    """Build a complete granite bundle from checkpoint to serialized artifact."""
    import json
    import time
    from dataclasses import replace
    from datetime import datetime, timezone
    from pathlib import Path

    from tensorrt_model_connect import trt_compat
    from tensorrt_model_connect.build_timing import (
        add_build_timing,
        new_build_timing,
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
        rank_engine_section,
        require_tensorrt_11_for_tensor_parallel,
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
        raise NotImplementedError("granite does not support context-parallel builds")

    config = ModelConfig.from_dir(model_path)
    config.raw["_model_dir"] = str(model_path)
    config.raw["_decoder_engine_layout"] = decoder_engine_layout
    config.raw["_fp32_layers"] = sorted(set(options.get("fp32_layers") or ()))
    config.raw["_family_build_options"] = dict(options.get("family_build_options") or {})
    config.raw["_parallel_build_enabled"] = bool(parallel.enabled)
    config.raw["_rtx_build_requested"] = bool(options.get("rtx"))
    config.raw["_runtime_dynamic_kv_requested"] = bool(
        options.get("dynamic_kv_cache") or options.get("triattention_stats_path")
    )
    config.raw["_quantized_build_requested"] = bool(options.get("quantize"))

    precision = str(options.get("precision") or "fp32").lower()
    config.raw["_resolved_build_precision"] = precision
    requested_cache_length = options.get("max_cache_length")
    max_cache_length = int(256) if requested_cache_length is None else int(requested_cache_length)
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
        from tensorrt_model_connect.quantization import QuantPlan, build_quant_context
        from . import graph_ops

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
            graph_ops=graph_ops,
        )

    dynamic_kv_cache = bool(options.get("dynamic_kv_cache"))
    dynamic_kv_budget = 1
    triattention_config = None
    triattention_section = None
    if options.get("triattention_stats_path"):
        from tensorrt_model_connect.triattention_export import (
            TriAttentionBundleConfig,
            export_triattention_stats_section,
        )

        recent_window = int(options.get("triattention_recent_window", 128))
        divide_length = int(options.get("triattention_divide_length", 128))
        score_aggregation = str(options.get("triattention_score_aggregation", "mean"))
        if recent_window < 0:
            raise ValueError("TriAttention recent_window must be >= 0")
        if divide_length < 1:
            raise ValueError("TriAttention divide_length must be >= 1")
        if score_aggregation not in {"mean", "max"}:
            raise ValueError("TriAttention score_aggregation must be 'mean' or 'max'")
        requested_budget = options.get("triattention_kv_budget")
        dynamic_kv_budget = max_cache_length if requested_budget is None else int(requested_budget)
        if not 1 <= dynamic_kv_budget <= max_cache_length:
            raise ValueError("TriAttention kv_budget must fit max_cache_length")
        triattention_config = TriAttentionBundleConfig(
            kv_budget=dynamic_kv_budget,
            divide_length=divide_length,
            recent_window=recent_window,
            score_aggregation=score_aggregation,
            count_prompt_tokens=bool(options.get("triattention_count_prompt_tokens", True)),
            protect_prefill=bool(options.get("triattention_protect_prefill", True)),
            disable_mlr=bool(options.get("triattention_disable_mlr", False)),
            disable_trig=bool(options.get("triattention_disable_trig", False)),
        )
        triattention_section = export_triattention_stats_section(
            str(options["triattention_stats_path"]),
            config=config,
        )
        dynamic_kv_cache = True

    if dynamic_kv_cache:
        rows = _sanitize_dynamic_kv_profile_rows(
            options.get("dynamic_kv_profile_rows_override"),
            max_cache_length,
        )
        if rows is None:
            preferred_rows = (
                [max(32, dynamic_kv_budget // 2)]
                if triattention_config is not None and dynamic_kv_budget >= 4096
                else None
            )
            rows = _dynamic_kv_profile_rows(
                max_cache_length,
                dynamic_kv_budget,
                preferred_rows=preferred_rows,
            )
        config.raw["dynamic_kv_cache"] = True
        config.raw["_dynamic_kv_opt_length"] = max_cache_length
        config.raw["_dynamic_kv_profile_rows"] = rows

    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(parallel, feature="granite tensor-parallel builds")

        if quant_ctx is not None:
            raise ValueError("granite tensor-parallel builds do not support quantization")
        if dynamic_kv_cache:
            raise NotImplementedError(
                "Tensor-parallel granite builds do not support dynamic KV cache or TriAttention"
            )

    verbose = bool(options.get("verbose"))

    def build_role(role: str, *, rank_parallel=parallel) -> bytes:
        from tensorrt_model_connect.tvm_ffi.graph_build import engine_role

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
                    parallel_config=rank_parallel,
                )
        finally:
            if previous is None:
                config.raw.pop("_decoder_engine_role", None)
            else:
                config.raw["_decoder_engine_role"] = previous

    from tensorrt_model_connect.tvm_ffi.graph_build import inspection_role

    inspection = inspection_role()
    if inspection is not None:
        build_role(inspection)
        raise RuntimeError("graph inspection did not reach TensorRT serialization")

    compile_started = time.monotonic()
    if parallel.enabled:
        plans = {
            rank: build_role("dual_profile", rank_parallel=parallel.for_rank(rank))
            for rank in range(parallel.tp_size)
        }
        sections = [
            BundleSection(rank_engine_section(rank), plan) for rank, plan in sorted(plans.items())
        ]
        decoder_layout = "dual_profile"
    else:
        split = (
            decoder_engine_layout == "split"
            and not dynamic_kv_cache
            and supports_split_decoder_roles(config)
        )
        if split:
            previous_active = config.raw.get("_active_split_decoder_build")
            config.raw["_active_split_decoder_build"] = True
            try:
                quant_label = str(quantize or "noquant")

                def build_split_role(role: str) -> bytes:
                    scope = (
                        f"split-{config.model_type}-h{config.hidden_size}"
                        f"-l{config.num_hidden_layers}-{precision}-{quant_label}-{role}"
                    )
                    with trt_compat.scoped_timing_cache(scope):
                        return build_role(role)

                prefill_started = time.monotonic()
                prefill_plan = build_split_role("prefill")
                add_build_timing(
                    timing,
                    "trt_compile_prefill_engine_s",
                    time.monotonic() - prefill_started,
                )
                decode_started = time.monotonic()
                plan = build_split_role("decode")
                add_build_timing(
                    timing,
                    "trt_compile_decode_engine_s",
                    time.monotonic() - decode_started,
                )
            finally:
                if previous_active is None:
                    config.raw.pop("_active_split_decoder_build", None)
                else:
                    config.raw["_active_split_decoder_build"] = previous_active
            sections = [
                BundleSection("engine_plan", plan),
                BundleSection("prefill_engine_plan", prefill_plan),
            ]
            decoder_layout = "split"
        else:
            role = "dual_profile" if decoder_engine_layout == "dual_profile" else "decode"
            plan = build_role(role)
            sections = [BundleSection("engine_plan", plan)]
            decoder_layout = "dual_profile" if role == "dual_profile" else "single"
    compile_elapsed = time.monotonic() - compile_started
    add_build_timing(timing, "trt_compile_s", compile_elapsed)
    add_build_timing(timing, "trt_compile_main_engine_s", compile_elapsed)
    write_build_timing(timing)

    if triattention_config is not None and triattention_section is not None:
        sections.append(BundleSection(triattention_config.stats_section, triattention_section))

    from tensorrt_model_connect.tokenizer_conversion import (
        prepare_tokenizer_special_frame,
    )

    tokenizer_frame = prepare_tokenizer_special_frame(
        model_path,
        source_model_id_or_path=options.get("tokenizer_source_model_id_or_path"),
        source_revision=options.get("tokenizer_source_revision"),
    )
    prefix_ids, suffix_ids = tokenizer_frame or ([], [])
    add_special_tokens = bool(prefix_ids or suffix_ids)

    trt_version = tensorrt_version()
    trt_abi = tensorrt_abi(trt_version)
    info = BundleInfo(
        model_id=model_path.name,
        model_type=config.model_type,
        family=name,
        trt_version=trt_version,
        trt_abi=trt_abi,
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
    if trt_abi:
        runtime_config["trt_abi"] = trt_abi
    if options.get("fp32_layers"):
        runtime_config["fp32_layers"] = sorted(set(options["fp32_layers"]))
    if tokenizer_frame is not None:
        runtime_config["tokenizer_special_prefix_ids"] = prefix_ids
        runtime_config["tokenizer_special_suffix_ids"] = suffix_ids
    if quant_plan is not None:
        runtime_config["quantization"] = quant_plan.as_config_dict()
    if dynamic_kv_cache:
        runtime_config["dynamic_kv_cache"] = True
        runtime_config["dynamic_kv_profile_rows"] = config.raw["_dynamic_kv_profile_rows"]
    if triattention_config is not None:
        runtime_config["triattention"] = triattention_config.to_dict()
    runtime_config.update(parallel.to_bundle_config_fields())

    tokenizer_override = None
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "vocab.txt",
        "special_tokens_map.json",
        "tokenizer.model",
    ):
        path = model_path / filename
        if filename == "tokenizer.json" and tokenizer_override is not None:
            sections.append(BundleSection(filename, tokenizer_override))
        elif path.is_file():
            sections.append(BundleSection(filename, path.read_bytes()))
    sections.append(
        BundleSection(
            "config.json",
            json.dumps(runtime_config, indent=2).encode("utf-8"),
        )
    )

    from tensorrt_model_connect.tvm_ffi.graph_build import kernel_slots_section

    slot_section = kernel_slots_section()
    if slot_section is not None:
        sections.append(BundleSection("kernel_slots.json", slot_section))
    kernel_manifest = []
    for global_name, library in options.get("kernel_artifacts") or ():
        section_name = f"kernel_{global_name.replace('.', '_')}.so"
        sections.append(BundleSection(section_name, Path(library).read_bytes()))
        kernel_manifest.append(
            {"global_name": global_name, "func_name": "run", "section": section_name}
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
