# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resumable, component-at-a-time MiniMax-H3 TensorRT build."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence

from tensorrt_model_connect.bundle_writer import BundleWriter

from . import trt_compat
from .config import (
    ADALN_PRECOMPUTE_DEFAULT_WORKSPACE_BYTES,
    AUDIO_LATENT_FRAMES_MAX,
    AUDIO_LATENT_FRAMES_MIN,
    AUDIO_LATENT_FRAMES_OPT,
    CANVAS_MAX_ASPECT_RATIO,
    CANVAS_MAX_PIXELS,
    CANVAS_MIN_ASPECT_RATIO,
    CANVAS_MULTIPLE,
    CANVAS_SHORT_EDGE,
    DENOISER_DEFAULT_WORKSPACE_BYTES,
    NATIVE_EXPLICIT_CANVAS_SIZES,
    RTX_STAGED_WORKSPACE_BYTES,
    RTX_WEIGHT_STREAMING_BUDGET_BYTES,
    SOL_ENGINE_1344X768_124_TO_345F,
    TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES,
    TRT_DEFAULT_WORKSPACE_POLICY,
    VIDEO_NUM_FRAMES_MAX,
    VIDEO_NUM_FRAMES_MIN,
    VIDEO_NUM_FRAMES_OPT,
    VISION_ENCODER_DEFAULT_WORKSPACE_BYTES,
)
from .provenance import (
    CHECKPOINT_REVISION,
    QUANTIZED_TRANSFORMER_CONFIG,
    SUPER_RESOLUTION_LEARNED_RESIDUAL_STRENGTH,
    super_resolution_bundle_config,
    super_resolution_source_identity,
    validate_quantized_transformer_metadata,
    validate_super_resolution_source_identity,
)
from .ref2va_bundle_contract import REF2VA_PLAN_SECTIONS as _REF2VA_COMPONENTS


_MODULE = "families.minimax_h3.staged_build"
_DENSE_FBC_COMPONENTS = (
    ("adaln_precompute", "adaln_precompute.plan", "adaln_precompute_plan"),
    ("denoiser_head", "denoiser_head.plan", "denoiser_head_plan"),
    ("denoiser_tail", "denoiser_tail.plan", "denoiser_tail_plan"),
    ("denoiser_finish", "denoiser_finish.plan", "denoiser_finish_plan"),
)
_QUANTIZED_TRANSFORMER_COMPONENTS = frozenset(
    component for component, _filename, _section in _DENSE_FBC_COMPONENTS
)
_SUPER_RESOLUTION_COMPONENT = (
    "video_super_resolution",
    "video_super_resolution.plan",
    "video_super_resolution_plan",
)
_COMPONENTS = (
    ("text_encoder", "text_encoder.plan", "text_encoder_plan"),
    ("vision_encoder", "vision_encoder.plan", "vision_encoder_plan"),
    *_DENSE_FBC_COMPONENTS,
    (
        "fl2va_keyframe_vae_encoder",
        "fl2va_keyframe_vae_encoder.plan",
        "fl2va_keyframe_vae_encoder_plan",
    ),
    ("vae_tile_decoder", "vae_tile_decoder.plan", "vae_tile_decoder_plan"),
    ("audio_vae_decoder", "audio_vae_decoder.plan", "audio_vae_decoder_plan"),
)


def _profile():
    return replace(SOL_ENGINE_1344X768_124_TO_345F, first_block_cache=True)


def _component_workspace_bytes(component: str, *, ref2va: bool) -> int:
    if not ref2va:
        return RTX_STAGED_WORKSPACE_BYTES
    return {
        "text_encoder": TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES,
        "vision_encoder": VISION_ENCODER_DEFAULT_WORKSPACE_BYTES,
        "ref2va_denoiser": DENOISER_DEFAULT_WORKSPACE_BYTES,
        "ref2va_adaln_precompute": ADALN_PRECOMPUTE_DEFAULT_WORKSPACE_BYTES,
        "ref2va_video_vae_encoder": 32 << 30,
        "ref2va_audio_vae_encoder": 32 << 30,
    }.get(component, RTX_STAGED_WORKSPACE_BYTES)


def _workspace_limits(
    components: Sequence[tuple[str, str, str]], *, ref2va: bool
) -> dict[str, int | str]:
    default_max = {name for name, _filename, _section in _DENSE_FBC_COMPONENTS}
    return {
        filename: (
            TRT_DEFAULT_WORKSPACE_POLICY
            if component == _SUPER_RESOLUTION_COMPONENT[0] or component in default_max
            else _component_workspace_bytes(component, ref2va=ref2va)
        )
        for component, filename, _section in components
    }


def _valid_plan_record(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"bytes"}
        and isinstance(value["bytes"], int)
        and not isinstance(value["bytes"], bool)
        and value["bytes"] > 0
    )


def _run_component(
    component: str,
    model: Path,
    output: Path,
    *,
    verbose: bool,
    transformer_ref_path: Path | None = None,
    quantized_transformer_path: Path | None = None,
    super_resolution_model: Path | None = None,
    super_resolution_weak_model: Path | None = None,
) -> dict[str, int]:
    command = [
        sys.executable,
        "-m",
        _MODULE,
        "--child",
        "--component",
        component,
        "--model-dir",
        str(model),
        "--output",
        str(output),
    ]
    if verbose:
        command.append("--verbose")
    if transformer_ref_path is not None:
        command.extend(("--transformer-ref", str(transformer_ref_path)))
    if quantized_transformer_path is not None:
        if component not in _QUANTIZED_TRANSFORMER_COMPONENTS:
            raise ValueError(
                "MiniMax-H3 quantized transformer may only build AdaLN and denoiser plans"
            )
        command.extend(("--quantized-transformer", str(quantized_transformer_path)))
    if super_resolution_model is not None:
        if component != _SUPER_RESOLUTION_COMPONENT[0]:
            raise ValueError("MiniMax-H3 super-resolution input was sent to a non-SR component")
        command.extend(("--super-resolution-model", str(super_resolution_model)))
        if super_resolution_weak_model is not None:
            command.extend(("--super-resolution-weak-model", str(super_resolution_weak_model)))
    subprocess.run(command, check=True)
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError(f"MiniMax-H3 staged child did not publish {component}")
    return {"bytes": output.stat().st_size}


def _runtime_config(
    *,
    trt_version: str,
    trt_abi: str,
    audio_vae_config: dict,
    components: Sequence[tuple[str, str, str]],
    transformer_ref_identity=None,
    quantized_transformer_identity=None,
    super_resolution_identity=None,
    runtime_defaults: dict[str, int | float] | None = None,
) -> dict[str, object]:
    profile = _profile()
    rates = audio_vae_config.get("decoder_rates")
    latent_mean = audio_vae_config.get("latents_mean")
    latent_std = audio_vae_config.get("latents_std")
    if (
        not isinstance(rates, list)
        or not rates
        or not isinstance(latent_mean, list)
        or not isinstance(latent_std, list)
        or len(latent_mean) != profile.audio_in_channels
        or len(latent_std) != profile.audio_in_channels
    ):
        raise ValueError("MiniMax-H3 AudioVAE config has invalid latent normalization")
    try:
        hop_length = math.prod(int(value) for value in rates)
        sampling_rate = int(audio_vae_config["sampling_rate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("MiniMax-H3 AudioVAE config has invalid decoder metadata") from error
    if hop_length <= 0 or sampling_rate <= 0:
        raise ValueError("MiniMax-H3 AudioVAE config has invalid decoder metadata")

    if transformer_ref_identity is None:
        text_sequence_profile = [1, 1144, 2641]
        vision_patch_profile = [2040, 4032, 4176]
        vision_row_profile = [1, 1008, 2088]
    else:
        from .ref2va_qwen_contract import ref2va_shared_qwen_profile_metadata

        shared_qwen = ref2va_shared_qwen_profile_metadata()
        text_sequence_profile = shared_qwen["text_encoder_plan"]["sequence_rows"]
        vision_patch_profile = shared_qwen["vision_encoder_plan"]["patch_rows_per_call"]
        vision_row_profile = shared_qwen["text_encoder_plan"]["compact_vision_rows"]

    extra: dict[str, object] = {}
    if quantized_transformer_identity is not None:
        extra["quantized_transformer"] = validate_quantized_transformer_metadata(
            quantized_transformer_identity.bundle_metadata()
        )
        extra["quantization"] = dict(QUANTIZED_TRANSFORMER_CONFIG)
    if super_resolution_identity is not None:
        extra["super_resolution"] = super_resolution_bundle_config(
            validate_super_resolution_source_identity(super_resolution_identity)
        )

    defaults = runtime_defaults or {}
    config: dict[str, object] = {
        "model_type": "minimax_h3",
        "runtime_strategy": "diffusion_minimax_h3",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "precision": "bf16",
        "engine_backend": "trt_rtx",
        "trt_version": trt_version,
        "trt_abi": trt_abi,
        **extra,
        "runtime_memory": {
            "mode": "staged",
            "weight_streaming_budget_bytes": RTX_WEIGHT_STREAMING_BUDGET_BYTES,
        },
        "workspace_limit_bytes": _workspace_limits(
            components, ref2va=transformer_ref_identity is not None
        ),
        "bundle_loading": {
            "mode": "staged",
            "eager_sections": ["tokenizer.json", "runtime.json"],
            "lazy_sections": [section for _component, _filename, section in components],
        },
        "height": int(defaults.get("height", 768)),
        "width": int(defaults.get("width", 1344)),
        "canvas_multiple": CANVAS_MULTIPLE,
        "canvas_short_edge": CANVAS_SHORT_EDGE,
        "canvas_max_pixels": CANVAS_MAX_PIXELS,
        "explicit_canvas_sizes": [list(size) for size in NATIVE_EXPLICIT_CANVAS_SIZES],
        "min_aspect_ratio": CANVAS_MIN_ASPECT_RATIO,
        "max_aspect_ratio": CANVAS_MAX_ASPECT_RATIO,
        "public_workflows": [
            "t2va",
            "fl2va",
            *(["ref2va"] if transformer_ref_identity is not None else []),
        ],
        "conditioning": {
            "implementation": "shared_native_qwen3_vl",
            "text_encoder_section": "text_encoder_plan",
            "vision_encoder_section": "vision_encoder_plan",
            "keyframe_vae_encoder_section": "fl2va_keyframe_vae_encoder_plan",
            "text_sequence_profile": text_sequence_profile,
            "vision_patch_profile": vision_patch_profile,
            "vision_row_profile": vision_row_profile,
            "t2va_dummy_vision_rows": 1,
            "t2va_vision_count": 0,
            "t2va_vision_mask_nonzero": 0,
            "keyframe_vae_tile_batch_profile": [1, 28, 33],
            "reachable_canvas_count": 95,
            "max_rounded_canvas": [576, 1856],
            "max_condition_video_rows": 2088,
            "mode_coupled_profile_required": True,
        },
        "num_frames": int(defaults.get("num_frames", VIDEO_NUM_FRAMES_OPT)),
        "num_frames_min": VIDEO_NUM_FRAMES_MIN,
        "num_frames_opt": VIDEO_NUM_FRAMES_OPT,
        "num_frames_max": VIDEO_NUM_FRAMES_MAX,
        "fps": 24,
        "num_inference_steps": 50,
        "seed": 0,
        "first_block_cache": True,
        "denoiser_cache_mode": "first_block",
        "denoiser_profile_count": 3,
        "denoiser_profile_layout": "five_second_t2va_then_fl2va_then_public_dynamic",
        "first_block_cache_threshold": float(
            defaults.get("first_block_cache_threshold", 0.08)
        ),
        "text_rows": profile.text_rows,
        "text_rows_min": profile.min_text_rows,
        "text_rows_opt": profile.opt_text_rows,
        "text_rows_max": profile.text_rows,
        "audio_rows": profile.opt_audio_rows,
        "audio_rows_min": profile.min_audio_rows,
        "audio_rows_opt": profile.opt_audio_rows,
        "audio_rows_max": profile.audio_rows,
        "audio_latent_frames": AUDIO_LATENT_FRAMES_OPT,
        "audio_latent_frames_min": AUDIO_LATENT_FRAMES_MIN,
        "audio_latent_frames_opt": AUDIO_LATENT_FRAMES_OPT,
        "audio_latent_frames_max": AUDIO_LATENT_FRAMES_MAX,
        "audio_sample_rate": sampling_rate,
        "audio_hop_length": hop_length,
        "audio_channels": 2,
        "audio_vae_precision": "fp32",
        "audio_vae_input_normalized": False,
        "audio_latents_mean": [float(value) for value in latent_mean],
        "audio_latents_std": [float(value) for value in latent_std],
        "video_rows": profile.opt_video_rows,
        "video_rows_min": profile.min_video_rows,
        "video_rows_opt": profile.opt_video_rows,
        "video_rows_max": profile.video_rows,
        "packed_sequence_length_min": profile.min_sequence_length,
        "packed_sequence_length_opt": profile.opt_sequence_length,
        "packed_sequence_length_max": profile.sequence_length,
        "padded_sequence_length": profile.padded_sequence_length,
        "max_timestep_count": profile.max_timestep_count,
        "context_parallel_size": profile.context_parallel_size,
        "vae_tile_batch": 28,
        "vae_tile_batch_min": 15,
        "vae_tile_batch_opt": 28,
        "vae_tile_batch_max": 33,
        "vae_tile_size": 256,
        "vae_tile_overlap": 64,
        "guidance_scale": 1.0,
        "scheduler_grid_points": 50,
        "transformer_forwards": 49,
        "attention_mode": "dense",
    }
    if transformer_ref_identity is not None:
        from .ref2va_bundle_contract import ref2va_bundle_metadata

        config.update(ref2va_bundle_metadata(transformer_ref_identity))
    return config


def _write_plan_section(writer: BundleWriter, section: str, path: Path) -> None:
    with writer.open_section(section) as output, path.open("rb") as source:
        shutil.copyfileobj(source, output, length=8 << 20)


def build_staged_bundle(
    model_dir: str | Path,
    writer: BundleWriter,
    *,
    plans_dir: str | Path,
    verbose: bool = False,
    transformer_ref: str | Path | None = None,
    quantized_transformer: str | Path | None = None,
    super_resolution_model: str | Path | None = None,
    super_resolution_weak_model: str | Path | None = None,
    runtime_defaults: dict[str, int | float] | None = None,
) -> None:
    """Build or resume each plan, then stream it through the current bundle writer."""

    model = Path(model_dir).resolve(strict=True)
    plans = Path(plans_dir).absolute()
    plans.mkdir(parents=True, exist_ok=True)
    tokenizer = model / "tokenizer" / "tokenizer.json"
    audio_config_path = model / "audio_vae" / "config.json"
    if not tokenizer.is_file() or not audio_config_path.is_file():
        raise FileNotFoundError("MiniMax-H3 tokenizer or AudioVAE config is missing")

    transformer_ref_path = None
    transformer_ref_identity = None
    if transformer_ref is not None:
        from .ref2va_checkpoint import COMPONENT_NAME, validate_transformer_ref_checkpoint

        supplied = Path(transformer_ref).resolve(strict=True)
        transformer_ref_identity = validate_transformer_ref_checkpoint(supplied)
        transformer_ref_path = (
            supplied if supplied.name == COMPONENT_NAME else supplied / COMPONENT_NAME
        ).resolve(strict=True)

    quantized_transformer_path = None
    quantized_transformer_identity = None
    if quantized_transformer is not None:
        from .quantized_checkpoint import validate_quantized_transformer_checkpoint

        quantized_transformer_path = Path(quantized_transformer).absolute()
        quantized_transformer_identity = validate_quantized_transformer_checkpoint(
            quantized_transformer_path
        )

    if super_resolution_weak_model is not None and super_resolution_model is None:
        raise ValueError("MiniMax-H3 super_resolution_weak_model requires super_resolution_model")
    super_resolution_model_path = (
        Path(super_resolution_model).absolute() if super_resolution_model is not None else None
    )
    super_resolution_weak_model_path = (
        Path(super_resolution_weak_model).absolute()
        if super_resolution_weak_model is not None
        else None
    )
    super_resolution_identity = None
    if super_resolution_model_path is not None:
        super_resolution_identity = super_resolution_source_identity(
            super_resolution_model_path,
            super_resolution_weak_model_path,
            denoise_strength=0.5 if super_resolution_weak_model_path is not None else 1.0,
        )

    components = (
        (*_COMPONENTS, *_REF2VA_COMPONENTS)
        if transformer_ref_identity is not None
        else _COMPONENTS
    )
    if super_resolution_identity is not None:
        components = (*components, _SUPER_RESOLUTION_COMPONENT)

    version = trt_compat.tensorrt_version()
    abi = trt_compat.tensorrt_abi(version)
    if not version or not abi:
        raise RuntimeError("Cannot determine TensorRT-RTX version and ABI")

    state = {
        "format": 1,
        "trt_version": version,
        "ref2va": transformer_ref_identity is not None,
        "quantized_transformer": quantized_transformer_identity is not None,
        "super_resolution": super_resolution_identity is not None,
        "components": [filename for _component, filename, _section in components],
    }
    state_path = plans / "build_state.json"
    if state_path.is_file():
        if json.loads(state_path.read_text(encoding="utf-8")) != state:
            raise ValueError(
                "MiniMax-H3 staged plan directory belongs to different build options"
            )
    elif any((plans / filename).exists() for _component, filename, _section in components):
        raise ValueError("MiniMax-H3 staged plan directory is missing build_state.json")
    else:
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    for component, filename, _section in components:
        plan = plans / filename
        if plan.is_file() and plan.stat().st_size > 0:
            continue
        options: dict[str, object] = {
            "verbose": verbose,
            "transformer_ref_path": transformer_ref_path,
        }
        if quantized_transformer_path is not None and component in _QUANTIZED_TRANSFORMER_COMPONENTS:
            options["quantized_transformer_path"] = quantized_transformer_path
        if component == _SUPER_RESOLUTION_COMPONENT[0]:
            options["super_resolution_model"] = super_resolution_model_path
            options["super_resolution_weak_model"] = super_resolution_weak_model_path
        _run_component(component, model, plan, **options)

    for _component, filename, section in components:
        _write_plan_section(writer, section, plans / filename)
    writer.add_bytes("tokenizer.json", tokenizer.read_bytes())
    writer.add_json(
        "runtime.json",
        _runtime_config(
            trt_version=version,
            trt_abi=abi,
            audio_vae_config=json.loads(audio_config_path.read_text(encoding="utf-8")),
            components=components,
            transformer_ref_identity=transformer_ref_identity,
            quantized_transformer_identity=quantized_transformer_identity,
            super_resolution_identity=super_resolution_identity,
            runtime_defaults=runtime_defaults,
        ),
    )


def _build_component(
    component: str,
    model: Path,
    output: Path,
    *,
    verbose: bool,
    transformer_ref_path: Path | None = None,
    quantized_transformer_path: Path | None = None,
    super_resolution_model: Path | None = None,
    super_resolution_weak_model: Path | None = None,
) -> dict[str, int | str]:
    trt_compat.configure_backend(rtx=True)
    from .checkpoint import (
        load_selected_component_state_dict,
        numpy_state,
    )

    if component.startswith("ref2va_") and transformer_ref_path is None:
        raise FileNotFoundError(
            "MiniMax-H3 Ref2VA components require the distinct transformer_ref checkpoint"
        )
    if transformer_ref_path is not None:
        from .ref2va_checkpoint import validate_transformer_ref_checkpoint

        validate_transformer_ref_checkpoint(transformer_ref_path)

    quantized_loader = None
    if quantized_transformer_path is not None:
        if component not in _QUANTIZED_TRANSFORMER_COMPONENTS:
            raise ValueError(
                "MiniMax-H3 quantized transformer may only build AdaLN and denoiser plans"
            )
        from .quantized_checkpoint import (
            load_selected_quantized_transformer_weights,
            validate_quantized_transformer_checkpoint,
        )

        validate_quantized_transformer_checkpoint(quantized_transformer_path)
        quantized_loader = load_selected_quantized_transformer_weights
    if super_resolution_model is not None and component != _SUPER_RESOLUTION_COMPONENT[0]:
        raise ValueError("MiniMax-H3 super-resolution source was sent to a non-SR component")
    if super_resolution_weak_model is not None and super_resolution_model is None:
        raise ValueError("MiniMax-H3 super_resolution_weak_model requires super_resolution_model")

    def transformer_weights(keys: Sequence[str]) -> dict:
        if quantized_loader is not None:
            return quantized_loader(quantized_transformer_path, keys)
        state = load_selected_component_state_dict(model / "transformer", keys)
        result = numpy_state(state)
        del state
        return result

    profile = _profile()

    dense_default_workspace_components = {
        component_name for component_name, _filename, _section in (*_DENSE_FBC_COMPONENTS,)
    }
    common = {
        "verbose": verbose,
        "consume_weights": True,
        "workspace_bytes": (
            None
            if profile.first_block_cache and component in dense_default_workspace_components
            else _component_workspace_bytes(component, ref2va=transformer_ref_path is not None)
        ),
        "weight_streaming": True,
        "output_path": output,
    }
    if component == "text_encoder":
        from .multimodal_text_encoder_builder import (
            build_multimodal_text_encoder_engine,
            checkpoint_keys,
        )

        state = load_selected_component_state_dict(model / "text_encoder", checkpoint_keys())
        weights = numpy_state(state)
        del state
        if transformer_ref_path is None:
            result = build_multimodal_text_encoder_engine(weights, **common)
        else:
            from .ref2va_qwen_builder import build_ref2va_shared_text_encoder_engine

            result = build_ref2va_shared_text_encoder_engine(weights, **common)
    elif component == "vision_encoder":
        from .multimodal_vision_builder import (
            build_multimodal_vision_encoder_engine,
            checkpoint_keys,
        )

        state = load_selected_component_state_dict(model / "text_encoder", checkpoint_keys())
        weights = numpy_state(state)
        del state
        if transformer_ref_path is None:
            result = build_multimodal_vision_encoder_engine(weights, **common)
        else:
            from .ref2va_qwen_builder import build_ref2va_shared_vision_encoder_engine

            result = build_ref2va_shared_vision_encoder_engine(weights, **common)
    elif component == "adaln_precompute":
        from .adaln_builder import build_adaln_precompute_engine, checkpoint_keys

        weights = transformer_weights(checkpoint_keys(profile))
        result = build_adaln_precompute_engine(weights, profile, **common)
    elif component in {
        "denoiser_head",
        "denoiser_tail",
        "denoiser_finish",
    }:
        from .dit_builder import (
            build_dit_finish_engine,
            build_dit_head_engine,
            build_dit_tail_engine,
            finish_checkpoint_keys,
            head_checkpoint_keys,
            tail_checkpoint_keys,
        )

        builders = {
            "denoiser_head": (build_dit_head_engine, head_checkpoint_keys),
            "denoiser_tail": (build_dit_tail_engine, tail_checkpoint_keys),
            "denoiser_finish": (build_dit_finish_engine, finish_checkpoint_keys),
        }
        builder, key_fn = builders[component]
        keys = key_fn() if component == "denoiser_finish" else key_fn(profile)
        weights = transformer_weights(keys)
        result = builder(weights, profile, **common)
    elif component == "fl2va_keyframe_vae_encoder":
        from .fl2va_vae_encoder_builder import (
            build_keyframe_vae_encoder_engine,
            checkpoint_keys,
        )

        state = load_selected_component_state_dict(model / "vae", checkpoint_keys())
        weights = numpy_state(state)
        del state
        result = build_keyframe_vae_encoder_engine(weights, **common)
    elif component == "vae_tile_decoder":
        from .vae_builder import build_vae_tile_decoder_engine, checkpoint_keys

        state = load_selected_component_state_dict(model / "vae", checkpoint_keys())
        weights = numpy_state(state)
        del state
        result = build_vae_tile_decoder_engine(weights, **common)
    elif component == "audio_vae_decoder":
        from .audio_vae_builder import (
            build_audio_vae_decoder_engine,
            checkpoint_keys,
            decoder_config_from_checkpoint,
        )

        audio_vae_dir = model / "audio_vae"
        audio_vae_config = json.loads((audio_vae_dir / "config.json").read_text())
        audio_decoder_profile = decoder_config_from_checkpoint(
            audio_vae_config,
            latent_frames=AUDIO_LATENT_FRAMES_OPT,
            min_latent_frames=AUDIO_LATENT_FRAMES_MIN,
            max_latent_frames=AUDIO_LATENT_FRAMES_MAX,
        )
        state = load_selected_component_state_dict(
            audio_vae_dir, checkpoint_keys(audio_decoder_profile)
        )
        weights = numpy_state(state)
        del state
        result = build_audio_vae_decoder_engine(
            weights,
            audio_decoder_profile,
            **common,
        )
    elif component == "ref2va_denoiser":
        if transformer_ref_path is None:
            raise FileNotFoundError(
                "MiniMax-H3 Ref2VA denoiser requires the distinct transformer_ref checkpoint"
            )
        from .ref2va_dit_builder import build_ref2va_dit_engine, checkpoint_keys

        state = load_selected_component_state_dict(transformer_ref_path, checkpoint_keys())
        weights = numpy_state(state)
        del state
        result = build_ref2va_dit_engine(weights, **common)
    elif component == "ref2va_adaln_precompute":
        if transformer_ref_path is None:
            raise FileNotFoundError(
                "MiniMax-H3 Ref2VA AdaLN requires the distinct transformer_ref checkpoint"
            )
        from .ref2va_dit_builder import (
            adaln_checkpoint_keys,
            build_ref2va_adaln_precompute_engine,
        )

        state = load_selected_component_state_dict(transformer_ref_path, adaln_checkpoint_keys())
        weights = numpy_state(state)
        del state
        result = build_ref2va_adaln_precompute_engine(weights, **common)
    elif component == "ref2va_video_vae_encoder":
        from .ref2va_video_encoder_builder import (
            build_ref2va_video_encoder_engine,
            checkpoint_keys,
        )

        state = load_selected_component_state_dict(model / "vae", checkpoint_keys())
        weights = numpy_state(state)
        del state
        result = build_ref2va_video_encoder_engine(weights, **common)
    elif component == "ref2va_audio_vae_encoder":
        from .ref2va_audio_encoder_builder import (
            build_ref2va_audio_encoder_engine,
            checkpoint_keys,
        )

        state = load_selected_component_state_dict(model / "audio_vae", checkpoint_keys())
        weights = numpy_state(state)
        del state
        result = build_ref2va_audio_encoder_engine(weights, **common)
    elif component == _SUPER_RESOLUTION_COMPONENT[0]:
        if super_resolution_model is None:
            raise FileNotFoundError(
                "MiniMax-H3 video super-resolution requires super_resolution_model"
            )
        from .super_resolution_builder import build_super_resolution_engine

        result = build_super_resolution_engine(
            super_resolution_model,
            super_resolution_weak_model,
            denoise_strength=(0.5 if super_resolution_weak_model is not None else 1.0),
            verbose=verbose,
            workspace_bytes=None,
            weight_streaming=False,
            learned_residual_strength=(SUPER_RESOLUTION_LEARNED_RESIDUAL_STRENGTH),
            output_path=output,
        )
    else:
        raise ValueError(f"Unknown MiniMax-H3 staged component: {component}")

    valid_result = (
        _valid_plan_record(result) and output.is_file() and output.stat().st_size == result["bytes"]
    )
    if not valid_result:
        raise RuntimeError(f"MiniMax-H3 staged builder returned an invalid record: {component}")
    return dict(result)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument(
        "--component",
        choices=sorted(
            {
                item[0]
                for item in (*_COMPONENTS, *_REF2VA_COMPONENTS, _SUPER_RESOLUTION_COMPONENT)
            }
        ),
    )
    parser.add_argument("--model-dir")
    parser.add_argument("--output")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--transformer-ref")
    parser.add_argument("--quantized-transformer")
    parser.add_argument("--super-resolution-model")
    parser.add_argument("--super-resolution-weak-model")
    args = parser.parse_args(argv)
    if not args.child or not args.component or not args.model_dir or not args.output:
        parser.error("this module is a staged-build child")
    _build_component(
        args.component,
        Path(args.model_dir),
        Path(args.output),
        verbose=args.verbose,
        transformer_ref_path=(
            Path(args.transformer_ref).resolve(strict=True) if args.transformer_ref else None
        ),
        quantized_transformer_path=(
            Path(args.quantized_transformer).absolute() if args.quantized_transformer else None
        ),
        super_resolution_model=(
            Path(args.super_resolution_model).absolute() if args.super_resolution_model else None
        ),
        super_resolution_weak_model=(
            Path(args.super_resolution_weak_model).absolute()
            if args.super_resolution_weak_model
            else None
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
