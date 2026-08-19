# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ConvBERT family model — encoder-only with mixed attention and span-based dynamic convolution.

ConvBERT uses:
  - Learned absolute position embeddings
  - Token type embeddings (segment A/B)
  - LayerNorm (with beta) instead of RMSNorm
  - HYBRID attention: standard multi-head self-attention on HALF the heads,
    span-based dynamic convolution on the other half
  - SeparableConv1D (depthwise + pointwise) for key_conv_attn
  - Dynamic convolution kernels generated per-position via softmax
  - Unfold/im2col for sliding window feature extraction
  - POST-norm (residual then LayerNorm), not pre-norm
  - 2-projection MLP (fc1/fc2) with GELU activation
  - Output projection maps concatenated [attn_context, conv_context] back to hidden_size

Architecture details (conv-bert-base):
  - hidden_size=768, num_attention_heads=12, head_ratio=2
  - Effective attention heads = num_attention_heads // head_ratio = 6
  - attention_head_size = (hidden_size // effective_heads) // 2 = 64
  - all_head_size = effective_heads * attention_head_size = 384
  - Q,K,V project to all_head_size=384 (not full hidden_size)
  - conv_kernel_size=9 for dynamic convolution
  - Output: concat(attn[seq,384], conv[seq,384]) = [seq,768] -> dense -> hidden
"""

from __future__ import annotations

import sys
import json
import re
import tempfile
import time

from pathlib import Path

import numpy as np

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
)
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)


name = "convbert"
runtime_strategy = "convbert_encoder_only"


def matches(config) -> bool:
    """Return whether this module owns the resolved model config."""
    model_type = getattr(config, "model_type", config)
    model_type = str(model_type)
    return model_type.lower() == "convbert"


def load_weights(model_dir: str, config: ModelConfig) -> WeightDict:
    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    max_pos = config.max_position_embeddings
    type_vocab_size = config.raw.get("type_vocab_size", 2)
    embedding_size = config.raw.get("embedding_size", hidden)

    # ConvBERT specific config
    head_ratio = config.raw.get("head_ratio", 2)
    num_attention_heads = config.num_attention_heads
    conv_kernel_size = config.raw.get("conv_kernel_size", 9)

    new_num_attention_heads = num_attention_heads // head_ratio
    if new_num_attention_heads < 1:
        new_num_attention_heads = 1
    attention_head_size = (hidden // new_num_attention_heads) // 2
    all_head_size = new_num_attention_heads * attention_head_size

    weights = WeightDict()

    # Store ConvBERT-specific config in weights for the builder
    weights["_convbert_new_num_heads"] = np.array([new_num_attention_heads], dtype=np.int32)
    weights["_convbert_head_size"] = np.array([attention_head_size], dtype=np.int32)
    weights["_convbert_all_head_size"] = np.array([all_head_size], dtype=np.int32)
    weights["_convbert_conv_kernel_size"] = np.array([conv_kernel_size], dtype=np.int32)

    # Detect prefix
    if _has_tensor(readers, "convbert.embeddings.word_embeddings.weight"):
        root = "convbert"
    elif _has_tensor(readers, "embeddings.word_embeddings.weight"):
        root = ""
    else:
        root = "convbert"

    def _pfx(key):
        return f"{root}.{key}" if root else key

    # Word embedding
    embedding = _load_tensor(readers, _pfx("embeddings.word_embeddings.weight"))
    assert embedding.shape == (vocab, embedding_size), (
        f"Embedding shape {embedding.shape} != ({vocab}, {embedding_size})"
    )
    weights["embedding"] = embedding.astype(np.float32)

    # Position embedding
    pos_embed = _load_tensor(readers, _pfx("embeddings.position_embeddings.weight"))
    assert pos_embed.shape == (max_pos, embedding_size), (
        f"Position embedding shape {pos_embed.shape} != ({max_pos}, {embedding_size})"
    )
    weights["position_embedding"] = pos_embed.astype(np.float32)

    # Token type embedding
    tt_embed = _load_tensor(readers, _pfx("embeddings.token_type_embeddings.weight"))
    assert tt_embed.shape == (type_vocab_size, embedding_size), (
        f"Token type embedding shape {tt_embed.shape} != ({type_vocab_size}, {embedding_size})"
    )
    weights["token_type_embedding"] = tt_embed.astype(np.float32)

    # Embedding LayerNorm
    embed_ln_w = _load_tensor(readers, _pfx("embeddings.LayerNorm.weight"))
    embed_ln_b = _load_tensor(readers, _pfx("embeddings.LayerNorm.bias"))
    weights["embed_norm"] = embed_ln_w.astype(np.float32)
    weights["embed_norm_beta"] = embed_ln_b.astype(np.float32)

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        hf_prefix = _pfx(f"encoder.layer.{layer_idx}")

        # Q, K, V projections
        q_w = _load_tensor(readers, f"{hf_prefix}.attention.query.weight")
        k_w = _load_tensor(readers, f"{hf_prefix}.attention.key.weight")
        v_w = _load_tensor(readers, f"{hf_prefix}.attention.value.weight")

        weights[f"{prefix}.w_q"] = np.ascontiguousarray(q_w.T.astype(np.float32))
        weights[f"{prefix}.w_k"] = np.ascontiguousarray(k_w.T.astype(np.float32))
        weights[f"{prefix}.w_v"] = np.ascontiguousarray(v_w.T.astype(np.float32))

        # QKV biases
        weights[f"{prefix}.q_bias"] = _load_tensor(
            readers, f"{hf_prefix}.attention.query.bias"
        ).astype(np.float32)
        weights[f"{prefix}.k_bias"] = _load_tensor(
            readers, f"{hf_prefix}.attention.key.bias"
        ).astype(np.float32)
        weights[f"{prefix}.v_bias"] = _load_tensor(
            readers, f"{hf_prefix}.attention.value.bias"
        ).astype(np.float32)

        # ConvBERT-specific: SeparableConv1D weights
        sep_dw = _load_tensor(
            readers, f"{hf_prefix}.attention.key_conv_attn_layer.depthwise.weight"
        )
        sep_pw = _load_tensor(
            readers, f"{hf_prefix}.attention.key_conv_attn_layer.pointwise.weight"
        )
        sep_bias = _load_tensor(readers, f"{hf_prefix}.attention.key_conv_attn_layer.bias")

        weights[f"{prefix}.sep_conv_dw"] = sep_dw.astype(np.float32)
        weights[f"{prefix}.sep_conv_pw"] = sep_pw.astype(np.float32)
        weights[f"{prefix}.sep_conv_bias"] = sep_bias.squeeze(-1).astype(np.float32)

        # conv_kernel_layer: linear [all_head_size -> num_heads * kernel_size]
        ck_w = _load_tensor(readers, f"{hf_prefix}.attention.conv_kernel_layer.weight")
        ck_b = _load_tensor(readers, f"{hf_prefix}.attention.conv_kernel_layer.bias")
        weights[f"{prefix}.conv_kernel_w"] = np.ascontiguousarray(ck_w.T.astype(np.float32))
        weights[f"{prefix}.conv_kernel_bias"] = ck_b.astype(np.float32)

        # conv_out_layer: linear [hidden -> all_head_size]
        co_w = _load_tensor(readers, f"{hf_prefix}.attention.conv_out_layer.weight")
        co_b = _load_tensor(readers, f"{hf_prefix}.attention.conv_out_layer.bias")
        weights[f"{prefix}.conv_out_w"] = np.ascontiguousarray(co_w.T.astype(np.float32))
        weights[f"{prefix}.conv_out_bias"] = co_b.astype(np.float32)

        # Output projection
        o_w = _load_tensor(readers, f"{hf_prefix}.attention.output.dense.weight")
        weights[f"{prefix}.w_o"] = np.ascontiguousarray(o_w.T.astype(np.float32))
        weights[f"{prefix}.o_bias"] = _load_tensor(
            readers, f"{hf_prefix}.attention.output.dense.bias"
        ).astype(np.float32)

        # Post-attention LayerNorm
        attn_ln_w = _load_tensor(readers, f"{hf_prefix}.attention.output.LayerNorm.weight")
        attn_ln_b = _load_tensor(readers, f"{hf_prefix}.attention.output.LayerNorm.bias")
        weights[f"{prefix}.post_attn_norm"] = attn_ln_w.astype(np.float32)
        weights[f"{prefix}.post_attn_norm_beta"] = attn_ln_b.astype(np.float32)

        # FFN
        fc1_w = _load_tensor(readers, f"{hf_prefix}.intermediate.dense.weight")
        fc1_b = _load_tensor(readers, f"{hf_prefix}.intermediate.dense.bias")
        fc2_w = _load_tensor(readers, f"{hf_prefix}.output.dense.weight")
        fc2_b = _load_tensor(readers, f"{hf_prefix}.output.dense.bias")

        weights[f"{prefix}.w_fc1"] = np.ascontiguousarray(fc1_w.T.astype(np.float32))
        weights[f"{prefix}.fc1_bias"] = fc1_b.astype(np.float32)
        weights[f"{prefix}.w_fc2"] = np.ascontiguousarray(fc2_w.T.astype(np.float32))
        weights[f"{prefix}.fc2_bias"] = fc2_b.astype(np.float32)

        # Output LayerNorm
        out_ln_w = _load_tensor(readers, f"{hf_prefix}.output.LayerNorm.weight")
        out_ln_b = _load_tensor(readers, f"{hf_prefix}.output.LayerNorm.bias")
        weights[f"{prefix}.output_norm"] = out_ln_w.astype(np.float32)
        weights[f"{prefix}.output_norm_beta"] = out_ln_b.astype(np.float32)

    return weights


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
    parallel = normalize_parallel_config(parallel_config)
    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(parallel, feature="ConvBERT tensor-parallel builds")
        if quant_ctx is not None:
            raise ValueError("ConvBERT tensor-parallel builds do not support quantization")
        from .tp_builder import build_tp_convbert_encoder_engine

        return build_tp_convbert_encoder_engine(
            config,
            weights,
            max_seq_length=max_cache_length,
            verbose=verbose,
            parallel_config=parallel,
        )

    from .builder import build_convbert_encoder_engine

    return build_convbert_encoder_engine(
        config, weights, max_seq_length=max_cache_length, precision=precision, verbose=verbose
    )


requires_tokenizer = True


def _detect_tokenizer_frame(
    source: str, *, revision: str | None = None
) -> tuple[list[int], list[int]] | None:
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


def _ensure_tokenizer_json(model_dir: Path) -> None:
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
                dir=model_dir, prefix=".trtmc-tokenizer-", suffix=".json", delete=False
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


def _apply_generation_config_eos(model_dir: Path, config: dict) -> None:
    path = model_dir / "generation_config.json"
    if not path.is_file():
        return
    generation_config = json.loads(path.read_text(encoding="utf-8"))
    if "eos_token_id" in generation_config:
        config["eos_token_id"] = generation_config["eos_token_id"]


def _build_local_engine(
    config, weights, max_cache_length, precision, quant_ctx, verbose, parallel, options
):
    from tensorrt_model_connect.tvm_ffi.graph_build import engine_role, inspection_role

    role = (
        "dual_profile"
        if str(options.get("decoder_engine_layout") or "split") == "dual_profile"
        else "decode"
    )

    def build_role(selected_role: str) -> bytes:
        with engine_role(selected_role):
            return build_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                parallel_config=parallel,
            )

    target_role = inspection_role()
    if target_role is not None:
        build_role(target_role)
        raise RuntimeError("graph inspection did not reach TensorRT serialization")
    return build_role(role), ("dual_profile" if role == "dual_profile" else "single")


def build(model_dir: str, output_path: str, **options) -> None:
    """Build the complete convbert bundle inside its owning family module."""
    from dataclasses import replace
    from datetime import datetime, timezone

    from tensorrt_model_connect import trt_compat as build_trt_compat
    from tensorrt_model_connect.build_timing import (
        add_build_timing,
        new_build_timing,
        write_build_timing,
    )
    from tensorrt_model_connect.bundle_writer import BundleInfo, BundleSection, write_bundle
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
        raise NotImplementedError("convbert does not support context-parallel builds")
    if options.get("dynamic_kv_cache") or options.get("triattention_stats_path"):
        raise ValueError("convbert does not use a decoder KV-cache runtime")

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
    weights = load_weights(str(model_path), config)
    add_build_timing(timing, "weights_loading_s", time.monotonic() - weights_started)
    write_build_timing(timing)

    quantize = options.get("quantize")
    quant_ctx = None
    quant_plan = None
    if quantize:
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
        require_tensorrt_11_for_tensor_parallel(parallel, feature="convbert tensor-parallel builds")
        if quant_ctx is not None:
            raise ValueError("convbert tensor-parallel builds do not support quantization")

    verbose = bool(options.get("verbose"))
    compile_started = time.monotonic()
    if parallel.enabled:
        plans = {
            rank: build_engine(
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
            BundleSection(rank_engine_section(rank), plan) for rank, plan in sorted(plans.items())
        ]
        decoder_layout = "dual_profile"
    else:
        plan, decoder_layout = _build_local_engine(
            config, weights, max_cache_length, precision, quant_ctx, verbose, parallel, options
        )
        sections = [BundleSection("engine_plan", plan)]
    compile_elapsed = time.monotonic() - compile_started
    add_build_timing(timing, "trt_compile_s", compile_elapsed)
    add_build_timing(timing, "trt_compile_main_engine_s", compile_elapsed)
    write_build_timing(timing)

    tokenizer_source = str(options.get("tokenizer_source_model_id_or_path") or model_path)
    tokenizer_frame = _detect_tokenizer_frame(
        tokenizer_source,
        revision=(
            str(options["tokenizer_source_revision"])
            if options.get("tokenizer_source_revision")
            else None
        ),
    )
    _ensure_tokenizer_json(model_path)
    if tokenizer_frame is None:
        tokenizer_frame = _detect_tokenizer_frame(str(model_path))
    prefix_ids, suffix_ids = tokenizer_frame or ([], [])
    add_special_tokens = bool(prefix_ids or suffix_ids)

    trt_version = build_trt_compat.tensorrt_version() or "unknown"
    version_match = re.search(r"(\d+)\.(\d+)", trt_version)
    trt_abi = f"{version_match.group(1)}.{version_match.group(2)}" if version_match else ""
    try:
        from tensorrt_model_connect.runtime_provider.target import _probe_current_target_with_device

        gpu_name = str(_probe_current_target_with_device()[0]["gpu_name"])
    except Exception:
        gpu_name = ""
    info = BundleInfo(
        model_id=model_path.name,
        model_type=config.model_type,
        family=name,
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
        runtime_strategy=runtime_strategy,
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
    _apply_generation_config_eos(model_path, runtime_config)
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
    if tokenizer_frame is not None:
        runtime_config["tokenizer_special_prefix_ids"] = prefix_ids
        runtime_config["tokenizer_special_suffix_ids"] = suffix_ids
    if options.get("fp32_layers"):
        runtime_config["fp32_layers"] = sorted(set(options["fp32_layers"]))
    if quant_plan is not None:
        runtime_config["quantization"] = quant_plan.as_config_dict()
    runtime_config.update(parallel.to_bundle_config_fields())

    from tensorrt_model_connect.tvm_ffi.graph_build import kernel_slots_section

    slot_section = kernel_slots_section()
    if slot_section is not None:
        sections.append(BundleSection("kernel_slots.json", slot_section))

    embedded_config = False
    for filename in (
        "config.json",
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
        if filename == "config.json":
            sections.append(
                BundleSection(filename, json.dumps(runtime_config, indent=2).encode("utf-8"))
            )
            embedded_config = True
        elif path.is_file():
            sections.append(BundleSection(filename, path.read_bytes()))
    if not embedded_config:
        sections.append(
            BundleSection("config.json", json.dumps(runtime_config, indent=2).encode("utf-8"))
        )

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
