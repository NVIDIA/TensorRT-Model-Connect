# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen family model — Qwen, Qwen2, Qwen3, QwQ (text-only, not VL).

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


def matches(config: object) -> bool:
    """Return whether this module owns the parsed model config."""
    model_type = str(getattr(config, "model_type", config))
    mt = model_type.lower()
    # Exclude variants owned by the qwen_vl, qwen_moe, qwen3_omni,
    # qwen_image, and qwen3_5 family modules.
    if "vl" in mt or "moe" in mt or "omni" in mt or "image" in mt:
        return False
    if mt in {"qwen3_5", "qwen3.5"}:
        return False
    return mt.startswith("qwen") or mt.startswith("qwq")


def default_build_precision(config: ModelConfig) -> str:
    capability = native_kv_architecture_capability(config)
    return "bf16" if capability.eligible else "fp32"


def default_max_cache_length(config: ModelConfig) -> int:
    """Use the model's complete context for native Qwen3."""
    capability = native_kv_architecture_capability(config)
    return int(config.max_position_embeddings) if capability.eligible else 256


def supports_split_decoder_roles(config: ModelConfig) -> bool:
    """Keep quantized Qwen on the single-engine correctness path."""
    return not bool(config.raw.get("_quantized_build_requested"))


def load_weights(model_dir: str, config: ModelConfig, *, precision: str = "fp32") -> WeightDict:
    return load_standard_weights(model_dir, config, precision=precision)


def build_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx=None,
    verbose: bool = False,
    parallel_config=None,
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
        role = str(config.raw.get("_decoder_engine_role", ""))
        if role not in ("prefill", "decode"):
            raise ValueError(
                "native Qwen3 requires explicit split engine role "
                f"'prefill' or 'decode', got {role!r}"
            )
        return build_dual_profile_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision="bf16",
            quant_ctx=None,
            verbose=verbose,
            profile_mode=role,
            native_kv_cache=True,
        )

    config.raw.pop("_native_kv_cache_metadata", None)
    if parallel.enabled:
        if debug_layer_outputs:
            raise NotImplementedError("Qwen tensor-parallel debug layer outputs are not supported")
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
    config: ModelConfig,
) -> dict | None:
    """Mark bundles that use the native KV runtime contract."""
    metadata = config.raw.get("_native_kv_cache_metadata")
    return dict(metadata) if isinstance(metadata, dict) else None


def calibration_data(format_name: str) -> list[str] | None:
    return list(_CALIBRATION_PROMPTS)


def quant_exclude_patterns(format_name: str) -> list[str]:
    patterns = [
        "embedding",
        "final_norm",
        "w_out",
        "lm_head",
        "*.input_norm",
        "*.post_attn_norm",
        "*_norm*",
    ]
    if format_name == "fp8":
        patterns.extend(
            [
                "layer.*.w_q",
                "layer.*.w_k",
                "layer.*.w_v",
                "layer.*.w_o",
                "layer.*.w_gate",
                "layer.*.w_down",
            ]
        )
    return patterns


def supports_parallel_quantization(format_name: str | None) -> bool:
    return format_name == "fp8"


def quant_adapter(format_name: str) -> StandardDecoderCalibrationAdapter:
    return StandardDecoderCalibrationAdapter(family=name)


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


def _optimized_value(value):
    from pathlib import Path

    from tensorrt_model_connect.parallel_config import ParallelConfig

    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ParallelConfig):
        return value.to_config_dict()
    if isinstance(value, dict):
        return {str(key): _optimized_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_optimized_value(item) for item in value]
    raise TypeError(
        f"Qwen build option cannot be represented for its runtime adapter: {type(value).__name__}"
    )


def _optimized_public_options(options: dict) -> dict:
    """Normalize the established public build values for Qwen's adapter."""
    public_options = {
        key: _optimized_value(value)
        for key, value in options.items()
        if key
        not in {
            "tokenizer_source_model_id_or_path",
            "tokenizer_source_revision",
        }
    }
    if public_options.get("precision") is None:
        public_options["precision"] = "fp32"
    if public_options.get("max_cache_length") is None:
        public_options["max_cache_length"] = 256
    return public_options


def _try_optimized_runtime(model_dir: str, output_path: str, options: dict) -> bool:
    from tensorrt_model_connect.runtime_provider.orchestrator import (
        try_build_optimized_runtime,
    )

    return (
        try_build_optimized_runtime(
            model_dir,
            output_path,
            family_name=name,
            parameters={"public_options": _optimized_public_options(options)},
        )
        is not None
    )


def build(model_dir: str, output_path: str, **options) -> None:
    """Build a complete qwen bundle from checkpoint to serialized artifact."""
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
        raise NotImplementedError("qwen does not support context-parallel builds")

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

    from .build_routing import prefer_native_default

    if not prefer_native_default(config) and _try_optimized_runtime(
        str(model_path), output_path, options
    ):
        return
    precision = str(options.get("precision") or default_build_precision(config)).lower()
    config.raw["_resolved_build_precision"] = precision
    requested_cache_length = options.get("max_cache_length")
    max_cache_length = (
        int(default_max_cache_length(config))
        if requested_cache_length is None
        else int(requested_cache_length)
    )
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
            exclude_patterns=quant_exclude_patterns(quant_plan.quant_format),
            calibration_prompts=calibration_data(quant_plan.quant_format),
            calibration_adapter=quant_adapter(quant_plan.quant_format),
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
        require_tensorrt_11_for_tensor_parallel(parallel, feature="qwen tensor-parallel builds")

        if quant_ctx is not None and not supports_parallel_quantization(
            quant_plan.quant_format if quant_plan is not None else None
        ):
            raise ValueError("Qwen tensor-parallel builds support only FP8 quantization")
        if dynamic_kv_cache:
            raise NotImplementedError(
                "Tensor-parallel qwen builds do not support dynamic KV cache or TriAttention"
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
    overrides = get_bundle_config_overrides(config)
    if overrides is not None:
        merged = dict(overrides)
        merged.update(runtime_config)
        merged.update(overrides)
        runtime_config = merged
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
