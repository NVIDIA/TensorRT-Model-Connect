# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Falcon family model — LayerNorm + GELU FC + RoPE + GQA.

Falcon-3 (TII) uses:
  - LayerNorm (with beta) instead of RMSNorm
  - 2-projection MLP (dense_h_to_4h / dense_4h_to_h) with GELU activation
  - RoPE for positional encoding
  - GQA (grouped query attention)
  - Separate Q/K/V projections (no fused QKV)
  - No QKV biases, no output projection bias
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from ...parallel_config import normalize_parallel_config
from .dual_profile_decoder_tp_builder import build_dual_profile_tp_decoder_engine
from .standard_decoder_builder import build_standard_decoder_engine


name = "falcon"
runtime_strategy = "falcon_decoder_kv_cache"
runtime_capabilities = {"decoder_kv"}


def matches(config: object) -> bool:
    """Return whether this module owns the parsed model config."""
    model_type = str(getattr(config, "model_type", config))
    mt = model_type.lower()
    return mt == "falcon" or mt.startswith("falcon") or mt in ("refinedweb", "refinedwebmodel")


def load_weights(model_dir: str, config: ModelConfig, *, precision: str = "fp32") -> WeightDict:
    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    weights = WeightDict()

    # Detect RW-style naming (falcon-rw-1b uses transformer.* prefix)
    rw_style = _has_tensor(readers, "transformer.word_embeddings.weight")

    # Embedding
    embed_key = "transformer.word_embeddings.weight" if rw_style else "model.embed_tokens.weight"
    embedding = _load_tensor(readers, embed_key)
    assert embedding.shape == (vocab, hidden), (
        f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
    )
    weights["embedding"] = embedding.astype(np.float32)

    mlp_size = 0
    attention_size = 0

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        if rw_style:
            hf_prefix = f"transformer.h.{layer_idx}"
        else:
            hf_prefix = f"model.layers.{layer_idx}"

        # LayerNorm weights + biases
        # RW models use ln_attn/ln_mlp; Falcon-3 uses input_layernorm/post_attention_layernorm
        if rw_style:
            input_norm_key = f"{hf_prefix}.ln_attn.weight"
            input_norm_beta_key = f"{hf_prefix}.ln_attn.bias"
            post_norm_key = f"{hf_prefix}.ln_mlp.weight"
            post_norm_beta_key = f"{hf_prefix}.ln_mlp.bias"
            # RW may use input_layernorm instead of ln_attn
            if not _has_tensor(readers, input_norm_key):
                input_norm_key = f"{hf_prefix}.input_layernorm.weight"
                input_norm_beta_key = f"{hf_prefix}.input_layernorm.bias"
                post_norm_key = f"{hf_prefix}.post_attention_layernorm.weight"
                post_norm_beta_key = f"{hf_prefix}.post_attention_layernorm.bias"
        else:
            input_norm_key = f"{hf_prefix}.input_layernorm.weight"
            input_norm_beta_key = f"{hf_prefix}.input_layernorm.bias"
            post_norm_key = f"{hf_prefix}.post_attention_layernorm.weight"
            post_norm_beta_key = f"{hf_prefix}.post_attention_layernorm.bias"

        input_norm = _load_tensor(readers, input_norm_key)
        input_norm_beta = _load_tensor(readers, input_norm_beta_key)
        post_norm = _load_tensor(readers, post_norm_key)
        post_norm_beta = _load_tensor(readers, post_norm_beta_key)

        weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32)
        weights[f"{prefix}.input_norm_beta"] = input_norm_beta.astype(np.float32)
        weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)
        weights[f"{prefix}.post_attn_norm_beta"] = post_norm_beta.astype(np.float32)

        # Q/K/V projections (separate)
        # RW models use self_attention.query_key_value (fused) or
        # self_attention.{query,key,value}; Falcon-3 uses self_attn.{q,k,v}_proj
        if rw_style:
            attn_prefix = f"{hf_prefix}.self_attention"
            # Check for fused QKV
            fused_qkv_key = f"{attn_prefix}.query_key_value.weight"
            if _has_tensor(readers, fused_qkv_key):
                fused_qkv = _load_tensor(readers, fused_qkv_key)
                # Falcon-RW uses HEAD-INTERLEAVED fused QKV layout:
                # [Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1, ...]
                # Shape: [num_heads * 3 * head_dim, hidden]
                # Reshape to [num_heads, 3, head_dim, hidden] then extract
                fused_qkv = fused_qkv.reshape(num_heads, 3, head_dim, hidden)
                q_raw = fused_qkv[:, 0, :, :].reshape(q_dim, hidden)
                k_raw = fused_qkv[:, 1, :, :].reshape(kv_dim, hidden)
                v_raw = fused_qkv[:, 2, :, :].reshape(kv_dim, hidden)
            else:
                q_raw = _load_tensor(readers, f"{attn_prefix}.q_proj.weight")
                k_raw = _load_tensor(readers, f"{attn_prefix}.k_proj.weight")
                v_raw = _load_tensor(readers, f"{attn_prefix}.v_proj.weight")
            o_raw = _load_tensor(readers, f"{attn_prefix}.dense.weight")
        else:
            q_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.q_proj.weight")
            k_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.k_proj.weight")
            v_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.v_proj.weight")
            o_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.o_proj.weight")

        if attention_size == 0:
            attention_size = q_raw.shape[0]

        q_t = _transpose_2d(q_raw, "q_proj")
        k_t = _transpose_2d(k_raw, "k_proj")
        v_t = _transpose_2d(v_raw, "v_proj")
        o_t = _transpose_2d(o_raw, "o_proj")

        # Keep compact GQA/MQA K/V

        weights[f"{prefix}.w_q"] = q_t
        weights[f"{prefix}.w_k"] = k_t
        weights[f"{prefix}.w_v"] = v_t
        weights[f"{prefix}.w_o"] = o_t

        # QKV biases (fused or separate)
        if rw_style:
            fused_qkv_bias_key = f"{attn_prefix}.query_key_value.bias"
            if _has_tensor(readers, fused_qkv_bias_key):
                fused_bias = _load_tensor(readers, fused_qkv_bias_key).astype(np.float32)
                # Same head-interleaved layout as weight
                fused_bias = fused_bias.reshape(num_heads, 3, head_dim)
                weights[f"{prefix}.q_bias"] = fused_bias[:, 0, :].reshape(-1)
                weights[f"{prefix}.k_bias"] = fused_bias[:, 1, :].reshape(-1)
                weights[f"{prefix}.v_bias"] = fused_bias[:, 2, :].reshape(-1)
            dense_bias_key = f"{attn_prefix}.dense.bias"
            if _has_tensor(readers, dense_bias_key):
                weights[f"{prefix}.o_bias"] = _load_tensor(readers, dense_bias_key).astype(
                    np.float32
                )

        # MLP: Falcon uses dense_h_to_4h / dense_4h_to_h
        if rw_style:
            mlp_prefix = f"{hf_prefix}.mlp"
        else:
            mlp_prefix = f"{hf_prefix}.mlp"
        fc1_raw = _load_tensor(readers, f"{mlp_prefix}.dense_h_to_4h.weight")
        fc2_raw = _load_tensor(readers, f"{mlp_prefix}.dense_4h_to_h.weight")
        if mlp_size == 0:
            mlp_size = fc1_raw.shape[0]

        weights[f"{prefix}.w_fc1"] = _transpose_2d(fc1_raw, "fc1")
        weights[f"{prefix}.w_fc2"] = _transpose_2d(fc2_raw, "fc2")

        # MLP biases (if present)
        fc1_bias_key = f"{mlp_prefix}.dense_h_to_4h.bias"
        fc2_bias_key = f"{mlp_prefix}.dense_4h_to_h.bias"
        if _has_tensor(readers, fc1_bias_key):
            weights[f"{prefix}.fc1_bias"] = _load_tensor(readers, fc1_bias_key).astype(np.float32)
        if _has_tensor(readers, fc2_bias_key):
            weights[f"{prefix}.fc2_bias"] = _load_tensor(readers, fc2_bias_key).astype(np.float32)

    # Final LayerNorm
    if rw_style:
        final_norm_key = "transformer.ln_f.weight"
        final_norm_beta_key = "transformer.ln_f.bias"
    else:
        final_norm_key = "model.norm.weight"
        final_norm_beta_key = "model.norm.bias"

    if _has_tensor(readers, final_norm_key):
        weights["final_norm"] = _load_tensor(readers, final_norm_key).astype(np.float32)
    else:
        weights["final_norm"] = np.ones(hidden, dtype=np.float32)

    if _has_tensor(readers, final_norm_beta_key):
        weights["final_norm_beta"] = _load_tensor(readers, final_norm_beta_key).astype(np.float32)

    # LM head
    lm_head_key = "lm_head.weight"
    if _has_tensor(readers, lm_head_key):
        weights["w_out"] = _transpose_2d(_load_tensor(readers, lm_head_key), "lm_head")
    else:
        weights["w_out"] = _transpose_2d(embedding.copy(), "embedding_tied")

    weights["_attention_size"] = attention_size  # type: ignore[assignment]
    weights["_kv_attention_size"] = kv_dim  # type: ignore[assignment]
    weights["_mlp_size"] = mlp_size  # type: ignore[assignment]

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
    use_alibi = config.raw.get("alibi", False)
    position_type = "alibi" if use_alibi else "rope"
    alibi_bias_scale = float(1.0 / np.sqrt(max(config.head_dim, 1))) if use_alibi else 1.0
    activation = config.raw.get("activation") or config.hidden_act or "gelu"
    parallel = normalize_parallel_config(parallel_config)
    if parallel.enabled:
        return build_dual_profile_tp_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            position_type=position_type,
            activation=activation,
            alibi_bias_scale=alibi_bias_scale,
            verbose=verbose,
            parallel_config=parallel,
        )

    return build_standard_decoder_engine(
        config,
        weights,
        max_cache_length,
        precision=precision,
        quant_ctx=quant_ctx,
        position_type=position_type,
        activation=activation,
        alibi_bias_scale=alibi_bias_scale,
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
    """Build a complete falcon bundle from checkpoint to serialized artifact."""
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
        raise NotImplementedError("falcon does not support context-parallel builds")

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
        require_tensorrt_11_for_tensor_parallel(parallel, feature="falcon tensor-parallel builds")

        if quant_ctx is not None:
            raise ValueError("falcon tensor-parallel builds do not support quantization")
        if dynamic_kv_cache:
            raise NotImplementedError(
                "Tensor-parallel falcon builds do not support dynamic KV cache or TriAttention"
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
