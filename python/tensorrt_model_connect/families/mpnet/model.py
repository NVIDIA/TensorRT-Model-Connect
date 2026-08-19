# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MPNet family model — encoder-only bidirectional transformer.

MPNet (Masked and Permuted Pre-training) shares the same encoder
architecture as BERT but with different weight naming:
  - No token type embeddings (no segment A/B)
  - QKV projections use .attn.q/.attn.k/.attn.v (not .self.query etc.)
  - Post-attention LayerNorm is at .attention.LayerNorm (not .attention.output.LayerNorm)
  - Weight prefix may be absent or "mpnet."

Detection: model_type == "mpnet"
"""

from __future__ import annotations

import json
import re
import tempfile
import time

import sys
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
from .encoder_builder import build_encoder_engine


def _detect_prefix(readers) -> str:
    """Detect weight prefix: '' (sentence-transformers) or 'mpnet.'."""
    if _has_tensor(readers, "mpnet.embeddings.word_embeddings.weight"):
        return "mpnet"
    if _has_tensor(readers, "embeddings.word_embeddings.weight"):
        return ""
    return "mpnet"


def _pfx(root, key):
    """Join root prefix with key, handling empty root."""
    return f"{root}.{key}" if root else key


def _compute_relative_position_bias(
    seq_length: int,
    num_buckets: int,
    num_heads: int,
    bias_table: np.ndarray,
) -> np.ndarray:
    """Pre-compute relative position bias matrix [num_heads, seq_len, seq_len].

    Uses the T5-style bucketing scheme (bidirectional):
    - Half the buckets for positive relative positions, half for negative
    - Exact buckets for small relative distances, log-spaced for larger ones
    """
    half_buckets = num_buckets // 2
    max_distance = 128  # T5/MPNet default

    # Relative positions: query_pos - key_pos
    context_position = np.arange(seq_length)[:, None]
    memory_position = np.arange(seq_length)[None, :]
    relative_position = memory_position - context_position  # [seq_len, seq_len]

    # Bidirectional bucketing — mirrors HF's _relative_position_bucket.
    # HF computes: n = -relative_position; offset = (n < 0) * half_buckets.
    # So: positive relative_position → n < 0 → offset = half_buckets
    #     zero/negative rel_pos → n >= 0 → offset = 0
    n = -relative_position  # negate to match HF convention
    buckets = np.zeros_like(n, dtype=np.int32)
    pos_mask = n < 0  # positive relative_position gets offset
    buckets[pos_mask] = half_buckets
    n_abs = np.abs(n)

    max_exact = half_buckets // 2
    is_small = n_abs < max_exact

    val_if_large = max_exact + (
        np.log(n_abs.astype(np.float64) / max_exact + 1e-12)
        / np.log(max_distance / max_exact)
        * (half_buckets - max_exact)
    ).astype(np.int32)
    val_if_large = np.minimum(val_if_large, half_buckets - 1)

    buckets += np.where(is_small, n_abs, val_if_large)

    # Look up bias: bias_table[bucket, head] -> [seq_len, seq_len, num_heads]
    bias = bias_table[buckets]  # [seq_len, seq_len, num_heads]
    # Transpose to [num_heads, seq_len, seq_len]
    return bias.transpose(2, 0, 1).astype(np.float32)


name = "mpnet"
runtime_strategy = "mpnet_encoder_only"


def matches(config) -> bool:
    """Return whether this module owns the resolved model config."""
    model_type = getattr(config, "model_type", config)
    model_type = str(model_type)
    return model_type.lower() == "mpnet"


def load_weights(model_dir: str, config: ModelConfig) -> WeightDict:
    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    _intermediate = config.intermediate_size
    _max_pos = config.max_position_embeddings

    root = _detect_prefix(readers)

    weights = WeightDict()

    # Word embedding
    embedding = _load_tensor(readers, _pfx(root, "embeddings.word_embeddings.weight"))
    assert embedding.shape == (vocab, hidden)
    weights["embedding"] = embedding.astype(np.float32)

    # Position embedding — MPNet uses padding_idx=1, positions start at 2
    pos_embed_raw = _load_tensor(readers, _pfx(root, "embeddings.position_embeddings.weight"))
    pad_idx = config.raw.get("pad_token_id", 1)
    pos_offset = pad_idx + 1
    pos_embed = pos_embed_raw[pos_offset:].astype(np.float32)
    weights["position_embedding"] = pos_embed

    # No token type embedding — synthesize zeros matching type_vocab_size
    type_vocab_size = config.raw.get("type_vocab_size", 2)
    weights["token_type_embedding"] = np.zeros((type_vocab_size, hidden), dtype=np.float32)

    # Embedding LayerNorm
    ln_w = _load_tensor(readers, _pfx(root, "embeddings.LayerNorm.weight"))
    ln_b = _load_tensor(readers, _pfx(root, "embeddings.LayerNorm.bias"))
    weights["embed_norm"] = ln_w.astype(np.float32)
    weights["embed_norm_beta"] = ln_b.astype(np.float32)

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        hf = _pfx(root, f"encoder.layer.{layer_idx}")

        # Q, K, V — MPNet uses .attention.attn.{q,k,v}
        q_w = _load_tensor(readers, f"{hf}.attention.attn.q.weight")
        k_w = _load_tensor(readers, f"{hf}.attention.attn.k.weight")
        v_w = _load_tensor(readers, f"{hf}.attention.attn.v.weight")

        weights[f"{prefix}.w_q"] = np.ascontiguousarray(q_w.T.astype(np.float32))
        weights[f"{prefix}.w_k"] = np.ascontiguousarray(k_w.T.astype(np.float32))
        weights[f"{prefix}.w_v"] = np.ascontiguousarray(v_w.T.astype(np.float32))

        weights[f"{prefix}.q_bias"] = _load_tensor(readers, f"{hf}.attention.attn.q.bias").astype(
            np.float32
        )
        weights[f"{prefix}.k_bias"] = _load_tensor(readers, f"{hf}.attention.attn.k.bias").astype(
            np.float32
        )
        weights[f"{prefix}.v_bias"] = _load_tensor(readers, f"{hf}.attention.attn.v.bias").astype(
            np.float32
        )

        # Output projection — .attention.attn.o
        o_w = _load_tensor(readers, f"{hf}.attention.attn.o.weight")
        weights[f"{prefix}.w_o"] = np.ascontiguousarray(o_w.T.astype(np.float32))
        weights[f"{prefix}.o_bias"] = _load_tensor(readers, f"{hf}.attention.attn.o.bias").astype(
            np.float32
        )

        # Post-attention LayerNorm — .attention.LayerNorm
        attn_ln_w = _load_tensor(readers, f"{hf}.attention.LayerNorm.weight")
        attn_ln_b = _load_tensor(readers, f"{hf}.attention.LayerNorm.bias")
        weights[f"{prefix}.post_attn_norm"] = attn_ln_w.astype(np.float32)
        weights[f"{prefix}.post_attn_norm_beta"] = attn_ln_b.astype(np.float32)

        # FFN: intermediate.dense -> output.dense
        fc1_w = _load_tensor(readers, f"{hf}.intermediate.dense.weight")
        fc1_b = _load_tensor(readers, f"{hf}.intermediate.dense.bias")
        fc2_w = _load_tensor(readers, f"{hf}.output.dense.weight")
        fc2_b = _load_tensor(readers, f"{hf}.output.dense.bias")

        weights[f"{prefix}.w_fc1"] = np.ascontiguousarray(fc1_w.T.astype(np.float32))
        weights[f"{prefix}.fc1_bias"] = fc1_b.astype(np.float32)
        weights[f"{prefix}.w_fc2"] = np.ascontiguousarray(fc2_w.T.astype(np.float32))
        weights[f"{prefix}.fc2_bias"] = fc2_b.astype(np.float32)

        # Output LayerNorm — .output.LayerNorm
        out_ln_w = _load_tensor(readers, f"{hf}.output.LayerNorm.weight")
        out_ln_b = _load_tensor(readers, f"{hf}.output.LayerNorm.bias")
        weights[f"{prefix}.output_norm"] = out_ln_w.astype(np.float32)
        weights[f"{prefix}.output_norm_beta"] = out_ln_b.astype(np.float32)

    # Relative attention bias — shared across all layers
    # Shape: [num_buckets, num_heads] -> pre-compute [num_heads, seq_len, seq_len]
    rel_bias_key = _pfx(root, "encoder.relative_attention_bias.weight")
    if _has_tensor(readers, rel_bias_key):
        rel_bias_w = _load_tensor(readers, rel_bias_key).astype(np.float32)
        num_buckets = rel_bias_w.shape[0]
        _num_attn_heads = rel_bias_w.shape[1]
        weights["_relative_attention_bias"] = rel_bias_w
        weights["_relative_attention_num_buckets"] = num_buckets

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
    public_module = sys.modules.get(__package__)
    compute_relative_position_bias = getattr(
        public_module,
        "_compute_relative_position_bias",
        _compute_relative_position_bias,
    )
    builder = getattr(public_module, "build_encoder_engine", build_encoder_engine)

    # Pre-compute relative position bias if present
    if "_relative_attention_bias" in weights:
        bias_table = weights.pop("_relative_attention_bias")
        num_buckets = weights.pop("_relative_attention_num_buckets")
        num_heads = bias_table.shape[1]
        bias_matrix = compute_relative_position_bias(
            max_cache_length, num_buckets, num_heads, bias_table
        )
        weights["relative_position_bias"] = bias_matrix

    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(parallel, feature="MPNet tensor-parallel builds")
        if quant_ctx is not None:
            raise ValueError("MPNet tensor-parallel builds do not support quantization")
        from .tp_builder import build_tp_encoder_engine

        return build_tp_encoder_engine(
            config,
            weights,
            max_seq_length=max_cache_length,
            verbose=verbose,
            parallel_config=parallel,
        )

    return builder(
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
    """Build the complete mpnet bundle inside its owning family module."""
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
        raise NotImplementedError("mpnet does not support context-parallel builds")
    if options.get("dynamic_kv_cache") or options.get("triattention_stats_path"):
        raise ValueError("mpnet does not use a decoder KV-cache runtime")

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
        require_tensorrt_11_for_tensor_parallel(parallel, feature="mpnet tensor-parallel builds")
        if quant_ctx is not None:
            raise ValueError("mpnet tensor-parallel builds do not support quantization")

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
