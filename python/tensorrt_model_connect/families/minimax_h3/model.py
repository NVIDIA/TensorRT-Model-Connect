# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-Model-Connect family model for MiniMaxAI/MiniMax-H3."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path

from .checkpoint import (
    load_selected_component_state_dict,
    numpy_state,
    validate_component_key_partition,
)
from .config import (
    SOL_ENGINE_1344X768_124F,
    default_workspace_limit_bytes,
)
from .provenance import (
    builder_source_sha256,
    checkpoint_snapshot_record,
    validate_source_revision,
    validate_workspace_limit_bytes,
)


def _build_source_revision() -> str:
    for name in (
        "TRTMC_MINIMAX_H3_SOURCE_REVISION",
        "TRTMC_ENGINE_BUILD_REVISION",
        "GITHUB_SHA",
    ):
        revision = os.environ.get(name, "").strip().lower()
        if revision:
            return validate_source_revision(revision)
    raise ValueError(
        "MiniMax-H3 native builds require TRTMC_MINIMAX_H3_SOURCE_REVISION "
        "(or TRTMC_ENGINE_BUILD_REVISION / GITHUB_SHA in CI)"
    )


def _effective_build_config(raw: dict) -> dict:
    family_options = raw.get("_family_build_options", {})
    minimax_options = (
        family_options.get("minimax_h3", {}) if isinstance(family_options, dict) else {}
    )
    if not isinstance(minimax_options, dict):
        raise ValueError("minimax_h3 build options must be an object")
    return {**raw, **minimax_options}


def _fixed_profile(raw: dict):
    expected = {
        "text_rows": SOL_ENGINE_1344X768_124F.text_rows,
        "audio_rows": SOL_ENGINE_1344X768_124F.audio_rows,
        "video_rows": SOL_ENGINE_1344X768_124F.video_rows,
        "padded_sequence_length": SOL_ENGINE_1344X768_124F.padded_sequence_length,
    }
    mismatches = {
        name: (raw[name], value)
        for name, value in expected.items()
        if name in raw and int(raw[name]) != value
    }
    if mismatches:
        raise ValueError(f"Unsupported MiniMax-H3 packed-row profile: {mismatches}")
    explicit_flag = raw.get("first_block_cache")
    mode = raw.get(
        "denoiser_cache_mode",
        "first_block" if explicit_flag is True else "monolithic",
    )
    if mode not in ("monolithic", "first_block"):
        raise ValueError(f"Unsupported MiniMax-H3 denoiser_cache_mode: {mode!r}")
    if explicit_flag is not None and not isinstance(explicit_flag, bool):
        raise ValueError("MiniMax-H3 first_block_cache must be a boolean")
    mode_flag = mode == "first_block"
    if explicit_flag is not None and explicit_flag != mode_flag:
        raise ValueError("MiniMax-H3 cache mode and first_block_cache flag disagree")
    if not mode_flag:
        return SOL_ENGINE_1344X768_124F
    return replace(SOL_ENGINE_1344X768_124F, first_block_cache=True)


def _first_block_cache_threshold(raw: dict) -> float:
    value = raw.get("first_block_cache_threshold", 0.025)
    if isinstance(value, bool):
        raise ValueError("MiniMax-H3 first_block_cache_threshold must be finite and positive")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "MiniMax-H3 first_block_cache_threshold must be finite and positive"
        ) from error
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("MiniMax-H3 first_block_cache_threshold must be finite and positive")
    return threshold


name = "minimax_h3"
default_build_precision = "bf16"
runtime_strategy = "diffusion_minimax_h3"
pipeline_classes = ("MiniMaxH3ModularPipeline", "MiniMaxH3Pipeline")


def matches(config) -> bool:
    """Return whether this module owns the resolved model config."""
    pipeline_class = str(getattr(config, "raw", {}).get("_class_name", ""))
    if pipeline_class in pipeline_classes:
        return True
    model_type = str(getattr(config, "model_type", config))
    return model_type.lower().replace("-", "_") in {
        "minimax_h3",
        "minimaxh3",
        "minimaxh3modularpipeline",
        "minimaxh3pipeline",
    }


def load_weights(model_dir: str, config, **_kwargs) -> dict:
    del config
    root = Path(model_dir)
    required_dirs = ("transformer", "text_encoder", "vae", "audio_vae", "tokenizer")
    missing = [str(root / name) for name in required_dirs if not (root / name).is_dir()]
    if missing:
        raise FileNotFoundError("Incomplete MiniMax-H3 Diffusers checkpoint: " + ", ".join(missing))
    transformer_config = json.loads((root / "transformer" / "config.json").read_text())
    expected = {
        "hidden_size": 5376,
        "num_layers": 50,
        "num_attention_heads": 56,
        "attention_head_dim": 128,
        "ffn_dim": 14336,
    }
    mismatches = {
        name: (transformer_config.get(name), value)
        for name, value in expected.items()
        if transformer_config.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Unsupported MiniMax-H3 transformer architecture: {mismatches}")
    return {
        "_model_dir": str(root),
        "_transformer_dir": str(root / "transformer"),
        "_text_encoder_dir": str(root / "text_encoder"),
        "_vae_dir": str(root / "vae"),
        "_audio_vae_dir": str(root / "audio_vae"),
        "_tokenizer_dir": str(root / "tokenizer"),
    }


def build_components(
    model_dir: str,
    config,
    weights: dict,
    *,
    precision: str = "bf16",
    verbose: bool = False,
    parallel_config=None,
    **_kwargs,
) -> dict:
    del model_dir
    if precision.lower() != "bf16":
        raise ValueError("MiniMax-H3 native builds require BF16 checkpoint weights")
    cp_size = int(getattr(parallel_config, "cp_size", 1))
    mode = str(getattr(parallel_config, "mode", "single"))
    if mode != "single" or cp_size != 1:
        raise ValueError("MiniMax-H3 requires parallel.mode=single and cp_size=1")

    raw = _effective_build_config(getattr(config, "raw", {}))
    profile = _fixed_profile(raw)
    profile.validate()
    workspace_limits = default_workspace_limit_bytes(first_block_cache=profile.first_block_cache)
    source_revision = _build_source_revision()
    snapshot = checkpoint_snapshot_record(Path(weights["_model_dir"]))
    from .adaln_builder import build_adaln_precompute_engine
    from .adaln_builder import checkpoint_keys as adaln_checkpoint_keys
    from .dit_builder import (
        build_dit_engine,
        build_dit_finish_engine,
        build_dit_head_engine,
        build_dit_tail_engine,
        checkpoint_keys as dit_checkpoint_keys,
        finish_checkpoint_keys,
        head_checkpoint_keys,
        tail_checkpoint_keys,
    )
    from .text_encoder_builder import (
        build_text_encoder_engine,
        checkpoint_keys as text_encoder_checkpoint_keys,
    )

    if profile.first_block_cache:
        denoiser_specs = (
            (
                "denoiser_head",
                "denoiser_head.plan",
                build_dit_head_engine,
                head_checkpoint_keys(profile),
            ),
            (
                "denoiser_tail",
                "denoiser_tail.plan",
                build_dit_tail_engine,
                tail_checkpoint_keys(profile),
            ),
            (
                "denoiser_finish",
                "denoiser_finish.plan",
                build_dit_finish_engine,
                finish_checkpoint_keys(profile),
            ),
        )
        checkpoint_groups = (
            adaln_checkpoint_keys(profile),
            *(spec[3] for spec in denoiser_specs),
        )
    else:
        denoiser_specs = (
            (
                "denoiser",
                "denoiser.plan",
                build_dit_engine,
                dit_checkpoint_keys(profile),
            ),
        )
        checkpoint_groups = (
            adaln_checkpoint_keys(profile),
            dit_checkpoint_keys(profile),
        )
    validate_component_key_partition(weights["_transformer_dir"], checkpoint_groups)

    text_state = load_selected_component_state_dict(
        weights["_text_encoder_dir"], text_encoder_checkpoint_keys()
    )
    text_weights = numpy_state(text_state)
    del text_state
    text_encoder_plan = build_text_encoder_engine(
        text_weights,
        sequence_length=profile.text_rows,
        verbose=verbose,
        consume_weights=True,
        workspace_bytes=workspace_limits["text_encoder.plan"],
    )
    del text_weights
    gc.collect()

    adaln_state = load_selected_component_state_dict(
        weights["_transformer_dir"], adaln_checkpoint_keys(profile)
    )
    adaln_weights = numpy_state(adaln_state)
    del adaln_state
    adaln_plan = build_adaln_precompute_engine(
        adaln_weights,
        profile,
        verbose=verbose,
        consume_weights=True,
        workspace_bytes=workspace_limits["adaln_precompute.plan"],
    )
    del adaln_weights
    gc.collect()

    denoiser_components = {}
    plan_sha256 = {
        "text_encoder.plan": hashlib.sha256(text_encoder_plan).hexdigest(),
        "adaln_precompute.plan": hashlib.sha256(adaln_plan).hexdigest(),
    }
    for component_name, filename, denoiser_builder, selected_keys in denoiser_specs:
        dit_state = load_selected_component_state_dict(weights["_transformer_dir"], selected_keys)
        dit_weights = numpy_state(dit_state)
        del dit_state
        denoiser_plan = denoiser_builder(
            dit_weights,
            profile,
            verbose=verbose,
            consume_weights=True,
            workspace_bytes=workspace_limits[filename],
        )
        del dit_weights
        gc.collect()
        denoiser_components[component_name] = denoiser_plan
        plan_sha256[filename] = hashlib.sha256(denoiser_plan).hexdigest()

    from .vae_builder import (
        build_vae_tile_decoder_engine,
        checkpoint_keys as vae_checkpoint_keys,
    )

    vae_state = load_selected_component_state_dict(weights["_vae_dir"], vae_checkpoint_keys())
    vae_weights = numpy_state(vae_state)
    del vae_state
    vae_decoder_plan = build_vae_tile_decoder_engine(
        vae_weights,
        verbose=verbose,
        consume_weights=True,
        workspace_bytes=workspace_limits["vae_tile_decoder.plan"],
    )
    tokenizer_json = (Path(weights["_tokenizer_dir"]) / "tokenizer.json").read_bytes()

    plan_sha256["vae_tile_decoder.plan"] = hashlib.sha256(vae_decoder_plan).hexdigest()

    return {
        "text_encoder": text_encoder_plan,
        "adaln_precompute": adaln_plan,
        **denoiser_components,
        "vae_decoder": vae_decoder_plan,
        "profile": profile,
        # Text/VAE paths remain explicit so follow-on native component
        # builders cannot silently substitute a different checkpoint.
        "vae_dir": weights["_vae_dir"],
        "audio_vae_dir": weights["_audio_vae_dir"],
        "tokenizer_dir": weights["_tokenizer_dir"],
        "tokenizer_json": tokenizer_json,
        "provenance": {
            "source_revision": source_revision,
            "builder_source_sha256": builder_source_sha256(),
            "checkpoint_inventory_sha256": snapshot["inventory_sha256"],
            "workspace_limit_bytes": workspace_limits,
            "plan_sha256": plan_sha256,
        },
    }


def diffusion_bundle_sections(components: dict, *, parallel_config=None) -> list[tuple[str, bytes]]:
    del parallel_config
    shared = [
        ("text_encoder_plan", components["text_encoder"]),
        ("adaln_precompute_plan", components["adaln_precompute"]),
    ]
    if components["profile"].first_block_cache:
        denoiser = [
            ("denoiser_head_plan", components["denoiser_head"]),
            ("denoiser_tail_plan", components["denoiser_tail"]),
            ("denoiser_finish_plan", components["denoiser_finish"]),
        ]
    else:
        denoiser = [("denoiser_plan", components["denoiser"])]
    return [
        *shared,
        *denoiser,
        ("vae_tile_decoder_plan", components["vae_decoder"]),
        ("tokenizer.json", components["tokenizer_json"]),
    ]


def diffusion_bundle_config(config, *, components: dict) -> dict:
    raw = _effective_build_config(getattr(config, "raw", {}))
    profile = components["profile"]
    fixed_request = {
        "video_height": 768,
        "video_width": 1344,
        "video_num_frames": 124,
        "num_inference_steps": 50,
    }
    mismatches = {
        name: (raw[name], value)
        for name, value in fixed_request.items()
        if name in raw and int(raw[name]) != value
    }
    if mismatches:
        raise ValueError(f"Unsupported MiniMax-H3 runtime profile: {mismatches}")
    provenance = components.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("MiniMax-H3 components are missing exact build provenance")
    validate_workspace_limit_bytes(provenance.get("workspace_limit_bytes"), profile=profile)
    if profile.first_block_cache:
        denoiser_sections = [
            "denoiser_head_plan",
            "denoiser_tail_plan",
            "denoiser_finish_plan",
        ]
    else:
        denoiser_sections = ["denoiser_plan"]
    return {
        "checkpoint_revision": "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc",
        **provenance,
        "height": 768,
        "width": 1344,
        "num_frames": 124,
        "fps": 24,
        "num_inference_steps": 50,
        "seed": int(raw.get("seed", 0)),
        "bundle_loading": {
            "mode": "staged",
            "eager_sections": ["tokenizer.json", "config.json"],
            "lazy_sections": [
                "text_encoder_plan",
                "adaln_precompute_plan",
                *denoiser_sections,
                "vae_tile_decoder_plan",
            ],
        },
        "first_block_cache": profile.first_block_cache,
        "denoiser_cache_mode": ("first_block" if profile.first_block_cache else "monolithic"),
        "first_block_cache_threshold": _first_block_cache_threshold(raw),
        "text_rows": profile.text_rows,
        "audio_rows": profile.audio_rows,
        "video_rows": profile.video_rows,
        "padded_sequence_length": profile.padded_sequence_length,
        "max_timestep_count": profile.max_timestep_count,
        "context_parallel_size": profile.context_parallel_size,
        "vae_tile_batch": 28,
        "vae_tile_size": 256,
        "vae_tile_overlap": 64,
    }


def diffusion_tokenizer_add_special_tokens(*_args, **_kwargs) -> bool:
    return False


def diffusion_tokenizer_bundle_sections(*_args, **_kwargs):
    # tokenizer.json is emitted with the model-owned engine sections.
    return []


def build(model_dir: str, output_path: str, **options) -> None:
    """Build this family's complete diffusion bundle."""
    import json
    import time
    from datetime import datetime, timezone
    from pathlib import Path
    from types import SimpleNamespace

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
        require_tensorrt_11_for_distributed,
    )
    from tensorrt_model_connect.tokenizer_conversion import (
        detect_tokenizer_add_special_tokens,
        ensure_tokenizer_json,
    )

    model_path = Path(model_dir)
    index_path = model_path / "model_index.json"
    config_path = index_path if index_path.is_file() else model_path / "config.json"
    pipeline_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(pipeline_config, dict):
        raise ValueError(f"Diffusion config must be an object: {config_path}")
    config = SimpleNamespace(model_type=name, raw=dict(pipeline_config))

    requested_cache = options.get("max_cache_length")
    max_cache_length = 256 if requested_cache is None else int(requested_cache)
    if max_cache_length < 1:
        raise ValueError("max_cache_length must be >= 1")
    config.raw["max_cache_length"] = max_cache_length
    config.raw["_fp32_layers"] = sorted(set(options.get("fp32_layers") or ()))
    config.raw["_family_build_options"] = dict(options.get("family_build_options") or {})
    config.raw.update(options.get("diffusion_overrides") or {})
    config.raw["_source_model_ref"] = str(
        options.get("tokenizer_source_model_id_or_path") or model_path
    )

    precision = str(options.get("precision") or default_build_precision).lower()
    verbose = bool(options.get("verbose"))
    parallel = normalize_parallel_config(options.get("parallel_config"))
    if parallel.distributed:
        require_tensorrt_11_for_distributed(parallel, feature=f"{name} distributed builds")
    max_batch_size = int(options.get("max_batch_size") or 1)
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be >= 1")

    timing = new_build_timing(options.get("build_timing_path"))
    timing["model_dir"] = str(model_path)
    timing["output_path"] = str(output_path)
    started = time.monotonic()
    write_build_timing(timing)

    weights_started = time.monotonic()
    weights = load_weights(str(model_path), config, precision=precision)
    add_build_timing(timing, "weights_loading_s", time.monotonic() - weights_started)
    write_build_timing(timing)
    if "_transformer_config" in weights:
        config.raw["_transformer_config"] = weights["_transformer_config"]

    fp8_scales = options.get("fp8_scales")
    if fp8_scales == "auto":
        raise ValueError("this family does not support FP8 auto-calibration")
    save_fp8_scales = options.get("save_fp8_scales")
    if save_fp8_scales and isinstance(fp8_scales, dict):
        Path(str(save_fp8_scales)).write_text(json.dumps(fp8_scales, indent=2), encoding="utf-8")

    components_started = time.monotonic()
    weights_before = build_timing_phase(timing, "weights_loading_s")
    compile_before = build_timing_phase(timing, "trt_compile_s")
    components = build_components(
        str(model_path),
        config,
        weights,
        verbose=verbose,
        precision=precision,
        fp8_scales=fp8_scales,
        build_timing=timing,
        parallel_config=parallel,
        max_batch_size=max_batch_size,
    )
    components_elapsed = time.monotonic() - components_started
    component_weights = max(0.0, build_timing_phase(timing, "weights_loading_s") - weights_before)
    compile_elapsed = max(0.0, components_elapsed - component_weights)
    add_build_timing(
        timing,
        "trt_compile_s",
        untracked_phase_time(compile_elapsed, compile_before, timing, "trt_compile_s"),
    )
    add_build_timing(timing, "trt_compile_diffusion_components_s", compile_elapsed)
    write_build_timing(timing)
    if components is None:
        raise ValueError("build_components() returned None")

    sections = [
        BundleSection(name, data)
        for name, data in diffusion_bundle_sections(components, parallel_config=parallel)
    ]
    special_frame = None
    if special_frame is None:
        prefix_ids: list[int] = []
        suffix_ids: list[int] = []
        add_special_tokens = diffusion_tokenizer_add_special_tokens(
            model_path,
            detect_tokenizer_add_special_tokens=(detect_tokenizer_add_special_tokens),
        )
    else:
        prefix_ids, suffix_ids = special_frame
        add_special_tokens = bool(prefix_ids or suffix_ids)

    version = tensorrt_version()
    abi = tensorrt_abi(version)
    if "config_json" in components:
        config_data = components["config_json"]
        if not isinstance(config_data, (bytes, bytearray)):
            raise TypeError("components['config_json'] must be bytes")
        config_data = bytes(config_data)
    else:
        config_dict = {
            "model_type": name,
            "runtime_strategy": runtime_strategy,
            "precision": "bf16" if fp8_scales else precision,
            "engine_backend": "trt_rtx" if options.get("rtx") else "trt",
            "trt_version": version,
            "tokenizer_add_special_tokens": int(add_special_tokens),
        }
        if abi:
            config_dict["trt_abi"] = abi
        if special_frame is not None:
            config_dict["tokenizer_special_prefix_ids"] = prefix_ids
            config_dict["tokenizer_special_suffix_ids"] = suffix_ids
        if fp8_scales:
            config_dict["quantization"] = {"format": "fp8"}
        config_dict.update(diffusion_bundle_config(config, components=components) or {})
        config_dict.update(parallel.to_bundle_config_fields())
        config_data = json.dumps(config_dict, indent=2).encode("utf-8")
    sections.append(BundleSection("config.json", config_data))
    sections.extend(
        BundleSection(name, data)
        for name, data in diffusion_tokenizer_bundle_sections(
            model_path, ensure_tokenizer_json=ensure_tokenizer_json
        )
    )

    batch_envelope = components.get("max_batch_size_envelope")
    if batch_envelope is None and max_batch_size > 1:
        batch_envelope = {
            "dit": max_batch_size,
            "text_encoder": min(max_batch_size * 2, 8),
            "vae": 1,
        }
    info = BundleInfo(
        model_id=model_path.name,
        model_type=name,
        family=name,
        trt_version=version,
        trt_abi=abi,
        gpu_name=gpu_name(),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        runtime_strategy=runtime_strategy,
        precision=precision,
        quantization="fp8" if fp8_scales else "none",
        max_cache_length=max_cache_length,
        tokenizer_add_special_tokens=add_special_tokens,
        max_batch_size=batch_envelope,
    )
    write_started = time.monotonic()
    write_bundle(output_path, info, sections)
    add_build_timing(timing, "bundle_write_s", time.monotonic() - write_started)
    timing["total_s"] = time.monotonic() - started
    write_build_timing(timing)
