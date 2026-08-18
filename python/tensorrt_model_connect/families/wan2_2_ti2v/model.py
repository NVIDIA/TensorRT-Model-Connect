# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-Model-Connect plugin for Wan2.2 TI2V-5B."""

from __future__ import annotations

import json
from pathlib import Path

from .model_config import (
    OFFICIAL_NEGATIVE_PROMPT,
    select_generation_profile,
    validate_native_config,
)


WAN22_MODEL_OWNED_BUNDLE_SECTIONS = (
    "text_encoder_0_plan",
    "denoiser_plan",
    "vae_decoder_plan",
    "vae_decoder_first_frame_plan",
    "tokenizer.json",
)
WAN22_EAGER_BUNDLE_SECTIONS = ("tokenizer.json", "config.json")
WAN22_LAZY_BUNDLE_SECTIONS = (
    "text_encoder_0_plan",
    "denoiser_plan",
    "vae_decoder_plan",
    "vae_decoder_first_frame_plan",
)


name = "wan2_2_ti2v"
default_build_precision = "bf16"
runtime_strategy = "diffusion_wan2_2_ti2v"
pipeline_classes = ("WanModel",)


def matches(config) -> bool:
    """Return whether this module owns the resolved model config."""
    pipeline_class = str(getattr(config, "raw", {}).get("_class_name", ""))
    if pipeline_class in pipeline_classes:
        return True
    model_type = str(getattr(config, "model_type", config))
    return model_type.lower() in {
        "ti2v",
        "wan2_2_ti2v",
    }


def load_weights(model_dir: str, config, **_kwargs) -> dict:
    root = Path(model_dir)
    config_path = root / "config.json"
    if not config_path.exists():
        raise ValueError(f"Wan2.2 TI2V requires native config.json in {root}")
    native_config = json.loads(config_path.read_text())
    validate_native_config(native_config)

    required = {
        "_vae_checkpoint": root / "Wan2.2_VAE.pth",
        "_text_encoder_checkpoint": root / "models_t5_umt5-xxl-enc-bf16.pth",
        "_tokenizer_dir": root / "google" / "umt5-xxl",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Incomplete Wan2.2-TI2V-5B checkpoint; missing: " + ", ".join(missing)
        )
    tokenizer_json = required["_tokenizer_dir"] / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError(f"Incomplete Wan2.2-TI2V-5B checkpoint; missing: {tokenizer_json}")

    return {key: str(path) for key, path in required.items()}


def fp8_precomputed_scales(model_dir: str, config) -> dict:
    """Load the packaged scale profile after fail-closed qualification."""

    from .fp8_profile import load_packaged_fp8_scales

    return load_packaged_fp8_scales(model_dir, config)


def build_components(
    model_dir: str,
    config,
    weights: dict,
    *,
    precision: str = "bf16",
    verbose: bool = False,
    **kwargs,
) -> dict:
    from .trt_builder import build_wan22_components

    return build_wan22_components(
        model_dir,
        config=config,
        weights=weights,
        precision=precision,
        verbose=verbose,
        **kwargs,
    )


def diffusion_bundle_sections(components: dict, *, parallel_config=None) -> list[tuple[str, bytes]]:
    del parallel_config
    payloads = {
        "text_encoder_0_plan": components["text_encoders"][0][1],
        "denoiser_plan": components["denoiser"],
        "vae_decoder_plan": components["vae_decoder"],
        "vae_decoder_first_frame_plan": components["vae_decoder_first_frame"],
        "tokenizer.json": components["tokenizer_json"],
    }
    return [(name, payloads[name]) for name in WAN22_MODEL_OWNED_BUNDLE_SECTIONS]


def diffusion_bundle_config(config, *, components: dict) -> dict:
    del components
    result = get_diffusion_config(config)
    result["bundle_loading"] = {
        "mode": "staged",
        "eager_sections": list(WAN22_EAGER_BUNDLE_SECTIONS),
        "lazy_sections": list(WAN22_LAZY_BUNDLE_SECTIONS),
    }
    return result


def diffusion_tokenizer_add_special_tokens(
    model_dir_path, *, detect_tokenizer_add_special_tokens
) -> bool:
    del model_dir_path, detect_tokenizer_add_special_tokens
    return False


def diffusion_tokenizer_bundle_sections(
    model_dir_path, *, ensure_tokenizer_json
) -> list[tuple[str, bytes]]:
    # tokenizer.json is already emitted with the model-owned sections.
    del model_dir_path, ensure_tokenizer_json
    return []


def get_diffusion_config(config) -> dict:
    raw = config.raw
    arch = select_generation_profile(raw)
    seed = int(raw.get("seed", 42))
    if not 0 <= seed <= 2_147_483_647:
        raise ValueError("Wan2.2-TI2V-5B bundle seed must be between 0 and 2147483647")
    return {
        "num_inference_steps": arch.num_inference_steps,
        "guidance_scale": arch.guidance_scale,
        "flow_shift": arch.flow_shift,
        "video_height": arch.video_height,
        "video_width": arch.video_width,
        "video_num_frames": arch.video_num_frames,
        "frame_rate": arch.frame_rate,
        "negative_prompt": str(raw.get("negative_prompt", OFFICIAL_NEGATIVE_PROMPT)),
        "text_seq_len": arch.text_seq_len,
        "seed": seed,
    }


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
        fp8_scales = fp8_precomputed_scales(str(model_path), config)
        if not isinstance(fp8_scales, dict) or not fp8_scales:
            raise ValueError("family FP8 scales must be a non-empty dictionary")
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
