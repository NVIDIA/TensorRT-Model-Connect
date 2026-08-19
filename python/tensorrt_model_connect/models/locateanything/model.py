# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LocateAnything family model.

LocateAnything-3B is a custom-code vision-language model:
  - Vision: MoonViT patch encoder.
  - Projector: LayerNorm + two-layer MLP (``mlp1``).
  - Text: Qwen2.5 decoder under ``language_model.model.*``.

This plugin wires a fixed single-image TRT MC contract: 448x448 image input,
14x14 MoonViT patches, 32x32 patch grid, 2x2 merge, and 256 image tokens.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import time
from pathlib import Path

from typing import TYPE_CHECKING

import numpy as np

from .checkpoint_mapper import (
    WeightDict,
    _has_tensor,
    _load_tensor,
    _open_safetensors,
    _transpose_2d,
)
from .config import ModelConfig
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)
from .default_decoder import build_standard_decoder_engine

if TYPE_CHECKING:
    from ...quantization.context import QuantContext


_DEFAULT_FIXED_IMAGE_SIZE = 448


name = "locateanything"
runtime_strategy = "locateanything_vision_language"
embed_input = True


def matches(config) -> bool:
    model_type = getattr(config, "model_type", config)
    model_type = str(model_type)
    return model_type.lower() == "locateanything"


def load_weights(model_dir: str, config: ModelConfig) -> WeightDict:
    return _load_locateanything_text_weights(model_dir, config)


def build_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx: "QuantContext | None" = None,
    verbose: bool = False,
    debug_layer_outputs: bool = False,
    parallel_config=None,
) -> bytes:
    parallel = normalize_parallel_config(parallel_config)
    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(
            parallel, feature="LocateAnything tensor-parallel builds"
        )
        if debug_layer_outputs:
            raise ValueError(
                "LocateAnything tensor-parallel builds do not support debug layer outputs"
            )
        from .decoder_tp_builder import build_qwen_vl_tp_decoder_engine

        return build_qwen_vl_tp_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            deepstack_num_levels=0,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
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


def build_vision_engine(
    model_dir: str,
    config: ModelConfig,
    weights: WeightDict,
    *,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes | None:
    from .vision_builder import build_locateanything_vision_engine

    return build_locateanything_vision_engine(
        model_dir, config, fixed_image_size=_DEFAULT_FIXED_IMAGE_SIZE, verbose=verbose
    )


def get_vl_config(config: ModelConfig) -> dict | None:
    vision_config = config.raw.get("vision_config")
    if not isinstance(vision_config, dict):
        return None

    patch_size = _first_int(vision_config.get("patch_size", 14), 14)
    merge_kernel = vision_config.get("merge_kernel_size", [2, 2])
    if isinstance(merge_kernel, (list, tuple)) and len(merge_kernel) >= 2:
        merge_h = _first_int(merge_kernel[0], 2)
        merge_w = _first_int(merge_kernel[1], 2)
    else:
        merge_h = merge_w = 2

    fixed_image_size = _DEFAULT_FIXED_IMAGE_SIZE
    grid_h = fixed_image_size // patch_size
    grid_w = fixed_image_size // patch_size
    num_image_tokens = (grid_h * grid_w) // max(merge_h * merge_w, 1)

    return {
        "image_token_id": config.raw.get(
            "image_token_index", config.raw.get("image_token_id", 151665)
        ),
        "fixed_image_size": fixed_image_size,
        "patch_size": patch_size,
        "merge_size": merge_h,
        "num_image_pad_tokens": num_image_tokens,
        "vision_output_dim": config.hidden_size,
        "preprocessor_type": "patchify_chw",
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "temporal_patch_size": 1,
        "interpolation": "bicubic",
        "vl_prompt_template": (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "<img>{image_pads}</img>{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "image_token_str": "<IMG_CONTEXT>",
        "locateanything_fixed_vision_grid_h": grid_h,
        "locateanything_fixed_vision_grid_w": grid_w,
        "box_start_token_id": config.raw.get("box_start_token_id", 151668),
        "box_end_token_id": config.raw.get("box_end_token_id", 151669),
        "coord_start_token_id": config.raw.get("coord_start_token_id", 151677),
        "coord_end_token_id": config.raw.get("coord_end_token_id", 152677),
        "ref_start_token_id": config.raw.get("ref_start_token_id", 151672),
        "ref_end_token_id": config.raw.get("ref_end_token_id", 151673),
        "none_token_id": config.raw.get("none_token_id", 4064),
    }


def get_bundle_config_overrides(config: ModelConfig) -> dict | None:
    return {
        "model_type": "locateanything",
        "runtime_strategy": "locateanything_vision_language",
        "embed_input": True,
    }


def _first_int(value: object, default: int) -> int:
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[0]
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_locateanything_text_weights(
    model_dir: str,
    config: ModelConfig,
) -> WeightDict:
    """Load the Qwen text decoder from LocateAnything safetensors."""
    from pathlib import Path

    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    weights = WeightDict()

    embed_key = _first_existing_tensor(
        readers,
        [
            "language_model.model.embed_tokens.weight",
            "model.language_model.model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ],
        "text token embedding",
    )
    embedding = _load_tensor(readers, embed_key)
    assert embedding.shape == (vocab, hidden), (
        f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
    )
    weights["embedding"] = embedding.astype(np.float32)

    layer_prefix = _detect_layer_prefix(readers)
    attention_size = 0
    kv_attention_size = 0
    mlp_size = 0

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        hf_prefix = f"{layer_prefix}.{layer_idx}"

        weights[f"{prefix}.input_norm"] = _load_tensor(
            readers, f"{hf_prefix}.input_layernorm.weight"
        ).astype(np.float32)
        weights[f"{prefix}.post_attn_norm"] = _load_tensor(
            readers, f"{hf_prefix}.post_attention_layernorm.weight"
        ).astype(np.float32)

        q_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.q_proj.weight")
        k_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.k_proj.weight")
        v_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.v_proj.weight")
        o_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.o_proj.weight")

        if attention_size == 0:
            attention_size = q_raw.shape[0]
        if kv_attention_size == 0:
            kv_attention_size = k_raw.shape[0]

        weights[f"{prefix}.w_q"] = _transpose_2d(q_raw, "q_proj")
        weights[f"{prefix}.w_k"] = _transpose_2d(k_raw, "k_proj")
        weights[f"{prefix}.w_v"] = _transpose_2d(v_raw, "v_proj")
        weights[f"{prefix}.w_o"] = _transpose_2d(o_raw, "o_proj")

        for proj_name, weight_key in [
            ("q_bias", "self_attn.q_proj.bias"),
            ("k_bias", "self_attn.k_proj.bias"),
            ("v_bias", "self_attn.v_proj.bias"),
        ]:
            full_key = f"{hf_prefix}.{weight_key}"
            if _has_tensor(readers, full_key):
                weights[f"{prefix}.{proj_name}"] = _load_tensor(readers, full_key).astype(
                    np.float32
                )

        gate_raw = _load_tensor(readers, f"{hf_prefix}.mlp.gate_proj.weight")
        up_raw = _load_tensor(readers, f"{hf_prefix}.mlp.up_proj.weight")
        down_raw = _load_tensor(readers, f"{hf_prefix}.mlp.down_proj.weight")

        if mlp_size == 0:
            mlp_size = gate_raw.shape[0]

        weights[f"{prefix}.w_gate"] = _transpose_2d(gate_raw, "gate")
        weights[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up")
        weights[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down")

    final_norm_key = _first_existing_tensor(
        readers,
        [
            f"{layer_prefix.rsplit('.layers', 1)[0]}.norm.weight",
            "language_model.model.norm.weight",
            "model.norm.weight",
        ],
        "text final norm",
        required=False,
    )
    if final_norm_key is not None:
        weights["final_norm"] = _load_tensor(readers, final_norm_key).astype(np.float32)
    else:
        weights["final_norm"] = np.ones(hidden, dtype=np.float32)

    lm_head_key = _first_existing_tensor(
        readers,
        [
            "language_model.lm_head.weight",
            "lm_head.weight",
            "model.lm_head.weight",
        ],
        "LM head",
        required=False,
    )
    if lm_head_key is not None:
        weights["w_out"] = _transpose_2d(_load_tensor(readers, lm_head_key), "lm_head")
    else:
        weights["w_out"] = _transpose_2d(embedding.copy(), "embedding_tied")

    weights["_attention_size"] = attention_size
    weights["_kv_attention_size"] = kv_attention_size
    weights["_mlp_size"] = mlp_size
    return weights


def _first_existing_tensor(
    readers,
    names: list[str],
    description: str,
    *,
    required: bool = True,
) -> str | None:
    for name in names:
        if _has_tensor(readers, name):
            return name
    if required:
        raise RuntimeError(f"Cannot find LocateAnything {description} weights")
    return None


def _detect_layer_prefix(readers) -> str:
    for prefix in [
        "language_model.model.layers",
        "model.language_model.model.layers",
        "model.layers",
    ]:
        if _has_tensor(readers, f"{prefix}.0.input_layernorm.weight"):
            return prefix
    raise RuntimeError("Cannot find LocateAnything text decoder layer weights")


requires_tokenizer = True
embed_input = True


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
    """Build the complete locateanything bundle inside its owning family module."""
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
        raise NotImplementedError("locateanything does not support context-parallel builds")
    if options.get("dynamic_kv_cache") or options.get("triattention_stats_path"):
        raise ValueError("locateanything does not use a decoder KV-cache runtime")

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
        require_tensorrt_11_for_tensor_parallel(
            parallel, feature="locateanything tensor-parallel builds"
        )
        if quant_ctx is not None:
            raise ValueError("locateanything tensor-parallel builds do not support quantization")

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

    vision_started = time.monotonic()
    vision_plan = build_vision_engine(
        str(model_path), config, weights, precision=precision, verbose=verbose
    )
    vision_elapsed = time.monotonic() - vision_started
    add_build_timing(timing, "trt_compile_s", vision_elapsed)
    add_build_timing(timing, "trt_compile_vision_engine_s", vision_elapsed)
    write_build_timing(timing)
    if vision_plan is not None:
        sections.append(BundleSection("vision_engine_plan", vision_plan))

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
    if vision_plan is not None:
        runtime_config["has_vision_engine"] = True
    runtime_config["embed_input"] = True
    vl_config = get_vl_config(config)
    if vl_config is not None:
        runtime_config.update(vl_config)
    overrides = get_bundle_config_overrides(config)
    if overrides is not None:
        merged = dict(overrides)
        merged.update(runtime_config)
        merged.update(overrides)
        runtime_config = merged

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
