# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lance family model — ByteDance ``bytedance-research/Lance`` unified model.

Scope (Stage 1): the **understanding** path only — ``x2t_image`` and
``x2t_video``. Lance's understanding sub-model is a Qwen2.5-VL ViT vision
encoder feeding a Lance text decoder, which maps onto the existing
``lance_vision_language`` runtime strategy.

Lance is a Mixture-of-Transformer-Experts model: every decoder layer carries a
second ``*_moe_gen`` parameter set, plus ``llm2vae`` / ``vae2llm`` /
``time_embedder`` / ``latent_pos_embed`` tensors. Those drive flow-matching
image/video **generation** and are intentionally NOT consumed here:
``load_standard_weights`` only reads the unsuffixed understanding-expert keys
(``self_attn.q_proj``, ``mlp.*``, ``input_layernorm`` …), so the generation
expert is dropped automatically. Generation/editing is a later stage that needs
a new runtime strategy and is out of scope for this plugin.

Architecture (confirmed against ``modeling/lance/qwen2_navit.py``): the
understanding decoder is GQA (16/2) with **QKV bias** (Qwen2 style) **and**
per-head **QK-norm** over ``head_dim`` (``qk_norm_und``) + SwiGLU + standard
RoPE; ViT is the standard Qwen2.5-VL encoder shipped with bare ``blocks.*`` /
``merger.*`` / ``patch_embed.*`` names (we re-add the ``visual.`` prefix the
shared vision builder expects). The shared decoder builder applies QKV-bias and
QK-norm conditionally when the weights are present.

Numerical validation: the TRT decoder matches an independent eager reference
exactly (per-layer and logits), and end-to-end ``trtmc run`` at **bf16** is
verified correct ("White car driving on the street." / "White"). Reduced
precision relies on the #184 builder fix (now in main): for embed bundles
``input_embed`` is bound as fp32 and cast inside the graph, and ``build_engine``
forwards ``precision`` so bf16/fp16 build true reduced-precision engines.

Checkpoint layout: the Lance HF repo is not a flat HF checkpoint (it nests
``Lance_3B/llm_config.json`` and a separate ``Qwen2.5-VL-ViT/`` dir). Run
``python -m tensorrt_model_connect.families.lance.prepare_model`` to stage a
directory this plugin can build:
``config.json`` (model_type=lance), ``model.safetensors``, the tokenizer files,
and the ViT at ``vision/model.safetensors``.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import time

from pathlib import Path

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    load_standard_weights,
    _open_safetensors,
    _load_tensor,
)

# Reuse the Qwen-VL vision encoder shape. The decoder builder is local so the
# Lance family does not depend on another family's text-builder package.
from .default_decoder import build_standard_decoder_engine
from .qwen_vl_vision_builder import build_qwen_vl_vision_engine

# Standard Qwen2.5-VL ViT input size; the runtime resizes images to this.
_DEFAULT_FIXED_IMAGE_SIZE = 448
# Lance LLM weights live under this prefix. The generation expert (``*_moe_gen``)
# and the VAE/time-embedder/latent-pos tensors are deliberately not requested.
_LLM_PREFIX = "language_model.model"
_LM_HEAD_KEY = "language_model.lm_head.weight"


name = "lance"
runtime_strategy = "lance_vision_language"
# During VL prefill the decoder consumes ViT features as input_embed in
# place of the image-pad token embeddings.
embed_input = True


def matches(config) -> bool:
    model_type = getattr(config, "model_type", config)
    model_type = str(model_type)
    return model_type.lower() == "lance"


def load_weights(model_dir: str, config: ModelConfig) -> WeightDict:
    return load_standard_weights(
        model_dir,
        config,
        model_prefix=_LLM_PREFIX,
        lm_head_key=_LM_HEAD_KEY,
    )


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
    return build_standard_decoder_engine(
        config,
        weights,
        max_cache_length,
        precision=precision,
        verbose=verbose,
        quant_ctx=quant_ctx,
        round_rope_inv_freq_to_bf16=precision == "bf16",
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
    vision_config = config.raw.get("vision_config")
    if vision_config is None:
        return None
    vision_weights = _load_lance_vision_weights(model_dir)
    return build_qwen_vl_vision_engine(
        vision_config,
        vision_weights,
        fixed_image_size=_DEFAULT_FIXED_IMAGE_SIZE,
        verbose=verbose,
    )


def get_vl_config(config: ModelConfig) -> dict | None:
    vision_config = config.raw.get("vision_config")
    if vision_config is None:
        return None

    patch_size = vision_config.get("patch_size", 14)
    merge_size = vision_config.get("spatial_merge_size", 2)
    fixed = _DEFAULT_FIXED_IMAGE_SIZE
    num_patches = (fixed // patch_size) ** 2
    num_merged = num_patches // (merge_size * merge_size)

    return {
        # Lance's pinned x2t_image reference intentionally routes image
        # features through Qwen2.5-VL's video placeholder.
        "image_token_id": config.raw.get("video_token_id", 151656),
        "fixed_image_size": fixed,
        "num_image_pad_tokens": num_merged,
        "vision_output_dim": config.hidden_size,
        "preprocessor_type": "merge_group_chw",
        "vl_prompt_template": (
            "<|im_start|>system\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|>{image_pads}<|vision_end|>"
            "{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "image_token_str": "<|video_pad|>",
    }


def _load_lance_vision_weights(model_dir: str) -> WeightDict:
    """Load the Qwen2.5-VL ViT weights, adding the ``visual.`` prefix the shared
    vision builder expects. The staged ViT lives at ``<model_dir>/vision/``."""
    vit_dir = Path(model_dir) / "vision"
    if not (vit_dir / "model.safetensors").exists():
        raise FileNotFoundError(
            f"Lance ViT weights not found at {vit_dir}/model.safetensors. "
            "Run python -m tensorrt_model_connect.families.lance.prepare_model "
            "to stage the model."
        )
    readers = _open_safetensors(vit_dir)
    weights = WeightDict()
    for reader in readers:
        for key in reader.keys():
            weights[f"visual.{key}"] = _load_tensor([reader], key)
    return weights


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
    """Build the complete lance bundle inside its owning family module."""
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
        raise NotImplementedError("lance does not support context-parallel builds")
    if options.get("dynamic_kv_cache") or options.get("triattention_stats_path"):
        raise ValueError("lance does not use a decoder KV-cache runtime")

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
        require_tensorrt_11_for_tensor_parallel(parallel, feature="lance tensor-parallel builds")
        if quant_ctx is not None:
            raise ValueError("lance tensor-parallel builds do not support quantization")

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
