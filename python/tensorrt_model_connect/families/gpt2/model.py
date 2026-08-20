# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete model-owned GPT-2 bundle build orchestration."""

from __future__ import annotations

import sys


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
        next_row = max(row + bucket_rows, row * 2)
        row = (
            (min(next_row, max_cache_length) + bucket_rows - 1) // bucket_rows
        ) * bucket_rows
    add_row(max_cache_length)
    rows.sort()
    return rows


def _sanitize_dynamic_kv_profile_rows(
    rows: list[int] | None,
    max_cache_length: int,
) -> list[int] | None:
    if rows is None:
        return None
    sanitized = sorted(
        {max(1, min(int(value), max_cache_length)) for value in rows}
    )
    if not sanitized:
        raise ValueError("dynamic_kv_profile_rows_override must contain at least one row")
    return sanitized


def _build_call_supports(function, name: str) -> bool:
    import inspect

    return name in inspect.signature(function).parameters


def _detect_build_tokenizer_frame(
    source: str,
    *,
    revision: str | None = None,
) -> tuple[list[int], list[int]] | None:
    from pathlib import Path

    try:
        from transformers import AutoTokenizer

        kwargs = {"trust_remote_code": True}
        if revision:
            kwargs["revision"] = revision
        if not Path(source).is_dir():
            kwargs["local_files_only"] = True
        tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        default_ids = list(tokenizer.encode("hello"))
        plain_ids = list(tokenizer.encode("hello", add_special_tokens=False))
    except Exception:
        return None
    if default_ids == plain_ids:
        return [], []
    if not plain_ids:
        return default_ids, []
    for start in range(len(default_ids) - len(plain_ids) + 1):
        if default_ids[start : start + len(plain_ids)] == plain_ids:
            return default_ids[:start], default_ids[start + len(plain_ids) :]
    return None


def _ensure_build_tokenizer(model_dir) -> None:
    import tempfile
    from pathlib import Path

    tokenizer_path = model_dir / "tokenizer.json"
    if tokenizer_path.is_file():
        return
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        with tempfile.TemporaryDirectory(prefix="trtmc-tokenizer-") as temporary:
            generated = Path(temporary) / "tokenizer.json"
            backend = getattr(tokenizer, "backend_tokenizer", None)
            if backend is None:
                backend = getattr(tokenizer, "_tokenizer", None)
            if backend is not None and hasattr(backend, "save"):
                backend.save(str(generated))
            if not generated.is_file():
                tokenizer.save_pretrained(temporary)
            if not generated.is_file():
                raise RuntimeError("tokenizer conversion did not create tokenizer.json")
            with tempfile.NamedTemporaryFile(
                dir=model_dir,
                prefix=".trtmc-tokenizer-",
                suffix=".json",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                output.write(generated.read_bytes())
            temporary_path.replace(tokenizer_path)
    except Exception as exc:
        print(
            "[trtmc build] Warning: could not generate tokenizer.json "
            f"(C++ runtime may fail to create tokenizer): {exc}",
            file=sys.stderr,
        )


def build(model_dir: str, output_path: str, *, options: dict[str, object]) -> None:
    """Build one complete GPT-2 bundle without shared model orchestration."""

    import json
    import re
    import time
    from dataclasses import replace
    from datetime import datetime, timezone
    from pathlib import Path

    from tensorrt_model_connect import trt_compat as build_trt_compat
    from tensorrt_model_connect.build_timing import (
        add_build_timing,
        new_build_timing,
        write_build_timing,
    )
    from tensorrt_model_connect.bundle_writer import BundleInfo, BundleSection, write_bundle
    from .config import ModelConfig
    from tensorrt_model_connect.parallel_config import (
        normalize_parallel_config,
        rank_engine_section,
        require_tensorrt_11_for_tensor_parallel,
    )
    from tensorrt_model_connect.families.gpt2.plugin import plugin

    model_path = Path(model_dir)
    parallel = normalize_parallel_config(options.get("parallel_config"))
    if parallel.cp_enabled:
        raise NotImplementedError("GPT-2 does not support context-parallel builds")

    config = ModelConfig.from_dir(model_path)
    config.raw["_model_dir"] = str(model_path)
    config.raw["_fp32_layers"] = sorted(set(options.get("fp32_layers") or ()))
    config.raw["_family_build_options"] = dict(options.get("family_build_options") or {})
    config.raw["_parallel_build_enabled"] = bool(parallel.enabled)
    config.raw["_rtx_build_requested"] = bool(options.get("rtx"))
    precision = str(options.get("precision") or "fp32").lower()
    config.raw["_resolved_build_precision"] = precision
    max_cache_length = int(options.get("max_cache_length") or 256)
    if max_cache_length < 1:
        raise ValueError("max_cache_length must be >= 1")

    dynamic_kv_cache = bool(options.get("dynamic_kv_cache"))
    dynamic_kv_budget = 1
    triattention_config = None
    triattention_section = None
    if options.get("triattention_stats_path"):
        from tensorrt_model_connect.triattention_export import (
            TriAttentionBundleConfig,
            export_triattention_stats_section,
        )

        recent_window = int(options.get("triattention_recent_window") or 0)
        divide_length = int(options.get("triattention_divide_length") or 0)
        score_aggregation = str(
            options.get("triattention_score_aggregation") or "mean"
        )
        if recent_window < 0:
            raise ValueError("TriAttention recent_window must be >= 0")
        if divide_length < 1:
            raise ValueError("TriAttention divide_length must be >= 1")
        if score_aggregation not in {"mean", "max"}:
            raise ValueError("TriAttention score_aggregation must be 'mean' or 'max'")
        dynamic_kv_budget = int(
            options.get("triattention_kv_budget") or max_cache_length
        )
        if dynamic_kv_budget < 1 or dynamic_kv_budget > max_cache_length:
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

    timing = new_build_timing(options.get("build_timing_path"))
    timing["model_dir"] = str(model_path)
    timing["output_path"] = str(output_path)
    started = time.monotonic()
    write_build_timing(timing)

    weights_started = time.monotonic()
    weight_kwargs = {"precision": precision} if _build_call_supports(
        plugin.load_weights, "precision"
    ) else {}
    weights = plugin.load_weights(str(model_path), config, **weight_kwargs)
    add_build_timing(timing, "weights_loading_s", time.monotonic() - weights_started)
    write_build_timing(timing)

    quantize = options.get("quantize")
    quant_ctx = None
    quant_plan = None
    if quantize:
        from tensorrt_model_connect.quantization import QuantPlan, build_quant_context

        quant_plan = QuantPlan.from_build_args(
            precision=precision,
            quantize=str(quantize),
            quant_scales=options.get("quant_scales"),
            quant_calibration_samples=int(options.get("quant_calibration_samples") or 512),
        )
        quant_method = str(config.raw.get("quantization_config", {}).get("quant_method", "")).lower()
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
            plugin=plugin,
            quant_plan=quant_plan,
            graph_ops=__import__(
                "tensorrt_model_connect.families.gpt2.graph_ops",
                fromlist=["*"],
            ),
        )

    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(parallel, feature="GPT-2 tensor-parallel builds")
        if quant_ctx is not None:
            raise ValueError("GPT-2 tensor-parallel builds do not support quantization")

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
    if parallel.enabled and dynamic_kv_cache:
        raise NotImplementedError(
            "Tensor-parallel GPT-2 builds do not support dynamic_kv_cache"
        )

    verbose = bool(options.get("verbose"))
    compile_started = time.monotonic()
    if parallel.enabled:
        plans = {
            rank: plugin.build_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                parallel_config=parallel.for_rank(rank),
            )
            for rank in range(parallel.tp_size)
        }
        sections = [
            BundleSection(rank_engine_section(rank), plan)
            for rank, plan in sorted(plans.items())
        ]
        decoder_layout = "dual_profile"
    else:
        from tensorrt_model_connect.tvm_ffi.graph_build import (
            engine_role,
            inspection_role,
        )

        def build_role(role: str) -> bytes:
            previous = config.raw.get("_decoder_engine_role")
            config.raw["_decoder_engine_role"] = role
            try:
                with engine_role(role):
                    return plugin.build_engine(
                        config,
                        weights,
                        max_cache_length,
                        precision=precision,
                        quant_ctx=quant_ctx,
                        verbose=verbose,
                        parallel_config=parallel,
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

        split = (
            str(options.get("decoder_engine_layout") or "split") == "split"
            and not dynamic_kv_cache
        )
        if split:
            config.raw["_active_split_decoder_build"] = True
            try:
                quant_label = str(quantize or "noquant")

                def build_split_role(selected_role: str) -> bytes:
                    scope = (
                        f"split-{config.model_type}-h{config.hidden_size}"
                        f"-l{config.num_hidden_layers}-{precision}-{quant_label}"
                        f"-{selected_role}"
                    )
                    with build_trt_compat.scoped_timing_cache(scope):
                        return build_role(selected_role)

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
                config.raw.pop("_active_split_decoder_build", None)
            sections = [
                BundleSection("engine_plan", plan),
                BundleSection("prefill_engine_plan", prefill_plan),
            ]
            decoder_layout = "split"
        else:
            role = (
                "dual_profile"
                if str(options.get("decoder_engine_layout") or "split") == "dual_profile"
                else "decode"
            )
            plan = build_role(role)
            sections = [BundleSection("engine_plan", plan)]
            decoder_layout = "dual_profile" if role == "dual_profile" else "single"
    compile_elapsed = time.monotonic() - compile_started
    add_build_timing(timing, "trt_compile_s", compile_elapsed)
    add_build_timing(timing, "trt_compile_main_engine_s", compile_elapsed)
    write_build_timing(timing)
    if triattention_config is not None and triattention_section is not None:
        sections.append(
            BundleSection(triattention_config.stats_section, triattention_section)
        )

    tokenizer_source = str(options.get("tokenizer_source_model_id_or_path") or model_path)
    tokenizer_frame = _detect_build_tokenizer_frame(
        tokenizer_source,
        revision=(
            str(options["tokenizer_source_revision"])
            if options.get("tokenizer_source_revision")
            else None
        ),
    )
    _ensure_build_tokenizer(model_path)
    if tokenizer_frame is None:
        tokenizer_frame = _detect_build_tokenizer_frame(str(model_path))
    prefix_ids, suffix_ids = tokenizer_frame or ([], [])
    add_special_tokens = bool(prefix_ids or suffix_ids)

    trt_version = build_trt_compat.tensorrt_version() or "unknown"
    version_match = re.search(r"(\d+)\.(\d+)", trt_version)
    trt_abi = f"{version_match.group(1)}.{version_match.group(2)}" if version_match else ""
    try:
        from tensorrt_model_connect.runtime_provider.target import (
            _probe_current_target_with_device,
        )

        gpu_name = str(_probe_current_target_with_device()[0]["gpu_name"])
    except Exception:
        gpu_name = ""
    info = BundleInfo(
        model_id=model_path.name,
        model_type=config.model_type,
        family=plugin.name,
        trt_version=trt_version,
        trt_abi=trt_abi,
        gpu_name=gpu_name,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_cache_length=max_cache_length,
        runtime_strategy=plugin.runtime_strategy,
        precision=precision,
        quantization=(quant_plan.quant_format if quant_plan else "none"),
        tokenizer_add_special_tokens=add_special_tokens,
    )

    source_config = model_path / "config.json"
    runtime_config = (
        json.loads(source_config.read_text(encoding="utf-8"))
        if source_config.is_file()
        else dict(config.raw)
    )
    runtime_config.update(
        {
            "runtime_strategy": plugin.runtime_strategy,
            "engine_backend": "trt_rtx" if options.get("rtx") else "trt",
            "trt_version": trt_version,
            "precision": precision,
            "tokenizer_add_special_tokens": int(add_special_tokens),
            "decoder_engine_layout": decoder_layout,
        }
    )
    if trt_abi:
        runtime_config["trt_abi"] = trt_abi
    if tokenizer_frame is not None:
        runtime_config["tokenizer_special_prefix_ids"] = prefix_ids
        runtime_config["tokenizer_special_suffix_ids"] = suffix_ids
    if quant_plan is not None:
        runtime_config["quantization"] = quant_plan.as_config_dict()
    if dynamic_kv_cache:
        runtime_config["dynamic_kv_cache"] = True
        runtime_config["dynamic_kv_profile_rows"] = config.raw[
            "_dynamic_kv_profile_rows"
        ]
    if triattention_config is not None:
        runtime_config["triattention"] = triattention_config.to_dict()
    runtime_config.update(parallel.to_bundle_config_fields())

    generation_config = model_path / "generation_config.json"
    if generation_config.is_file():
        generation = json.loads(generation_config.read_text(encoding="utf-8"))
        if "eos_token_id" in generation:
            runtime_config["eos_token_id"] = generation["eos_token_id"]

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
        if path.is_file():
            sections.append(BundleSection(filename, path.read_bytes()))
    sections.append(
        BundleSection("config.json", json.dumps(runtime_config, indent=2).encode("utf-8"))
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
