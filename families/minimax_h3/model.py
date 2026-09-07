# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-Model-Connect family plugin for MiniMaxAI/MiniMax-H3."""

from __future__ import annotations

import gc
from fractions import Fraction
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from .checkpoint import (
    load_selected_component_state_dict,
    numpy_state,
    validate_component_key_partition,
)
from .config import (
    AUDIO_LATENT_FRAMES_MAX,
    AUDIO_LATENT_FRAMES_MIN,
    AUDIO_LATENT_FRAMES_OPT,
    CANVAS_MAX_ASPECT_RATIO,
    CANVAS_MAX_PIXELS,
    CANVAS_MIN_ASPECT_RATIO,
    CANVAS_MULTIPLE,
    CANVAS_SHORT_EDGE,
    NATIVE_EXPLICIT_CANVAS_SIZES,
    RTX_WEIGHT_STREAMING_BUDGET_BYTES,
    SOL_ENGINE_1344X768_124_TO_345F,
    TRT_DEFAULT_WORKSPACE_POLICY,
    VIDEO_NUM_FRAMES_MAX,
    VIDEO_NUM_FRAMES_MIN,
    VIDEO_NUM_FRAMES_OPT,
    default_workspace_limit_bytes,
)
from .provenance import (
    QUANTIZED_TRANSFORMER_CONFIG,
    SUPER_RESOLUTION_LEARNED_RESIDUAL_STRENGTH,
    super_resolution_bundle_config,
    super_resolution_source_identity,
    validate_quantized_transformer_metadata,
    validate_super_resolution_source_identity,
    validate_workspace_limit_bytes,
)

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


def _effective_build_config(raw: dict) -> dict:
    family_options = raw.get("_family_build_options", {})
    options = family_options.get("minimax_h3", {}) if isinstance(family_options, dict) else {}
    if not isinstance(options, dict):
        raise ValueError("minimax_h3 build options must be an object")
    from .runtime_config_schema import normalize_build_options

    return {**raw, **normalize_build_options(options)}


def _fixed_profile(raw: dict):
    profile = SOL_ENGINE_1344X768_124_TO_345F
    expected = {
        "text_rows": profile.text_rows,
        "text_rows_min": profile.min_text_rows,
        "text_rows_opt": profile.opt_text_rows,
        "text_rows_max": profile.text_rows,
        "audio_rows": profile.opt_audio_rows,
        "audio_rows_min": profile.min_audio_rows,
        "audio_rows_opt": profile.opt_audio_rows,
        "audio_rows_max": profile.audio_rows,
        "video_rows": profile.opt_video_rows,
        "video_rows_min": profile.min_video_rows,
        "video_rows_opt": profile.opt_video_rows,
        "video_rows_max": profile.video_rows,
        "packed_sequence_length_min": profile.min_sequence_length,
        "packed_sequence_length_opt": profile.opt_sequence_length,
        "packed_sequence_length_max": profile.sequence_length,
        "padded_sequence_length": profile.padded_sequence_length,
    }
    mismatches = {
        name: (raw[name], value)
        for name, value in expected.items()
        if name in raw and int(raw[name]) != value
    }
    if mismatches:
        raise ValueError(f"Unsupported MiniMax-H3 packed-row profile: {mismatches}")
    explicit_flag = raw.get("first_block_cache", True)
    if explicit_flag is not True:
        raise ValueError("MiniMax-H3 only supports the dense FirstBlockCache build")
    mode = raw.get("denoiser_cache_mode", "first_block")
    if mode != "first_block":
        raise ValueError("MiniMax-H3 only supports denoiser_cache_mode='first_block'")
    return replace(profile, first_block_cache=True)


def _default_num_frames(raw: dict) -> int:
    value = raw.get("video_num_frames", raw.get("num_frames", VIDEO_NUM_FRAMES_OPT))
    if isinstance(value, bool):
        raise ValueError("MiniMax-H3 video_num_frames must be a valid 5--15 second geometry")
    try:
        frames = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "MiniMax-H3 video_num_frames must be a valid 5--15 second geometry"
        ) from error
    if not VIDEO_NUM_FRAMES_MIN <= frames <= VIDEO_NUM_FRAMES_MAX or frames % 17 != 5:
        raise ValueError("MiniMax-H3 video_num_frames must be a valid 5--15 second geometry")
    return frames


def _resolve_canvas_size(aspect_width: float, aspect_height: float) -> tuple[int, int]:
    """Mirror the public H3 resolver with Python ties-to-even 32 rounding."""

    if not math.isfinite(aspect_width) or not math.isfinite(aspect_height):
        raise ValueError("MiniMax-H3 canvas aspect must be finite and positive")
    if aspect_width <= 0.0 or aspect_height <= 0.0:
        raise ValueError("MiniMax-H3 canvas aspect must be finite and positive")
    ratio = aspect_width / aspect_height
    if not CANVAS_MIN_ASPECT_RATIO <= ratio <= CANVAS_MAX_ASPECT_RATIO:
        raise ValueError("MiniMax-H3 canvas aspect must be within 1:4 through 4:1")
    if ratio >= 1.0:
        height = float(CANVAS_SHORT_EDGE)
        width = height * ratio
    else:
        width = float(CANVAS_SHORT_EDGE)
        height = width / ratio
    pixels = width * height
    if pixels > CANVAS_MAX_PIXELS:
        scale = math.sqrt(CANVAS_MAX_PIXELS / pixels)
        width *= scale
        height *= scale
    resolved_width = int(round(width / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE
    resolved_height = int(round(height / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE
    return resolved_height, resolved_width


def _reachable_canvas_sizes() -> tuple[tuple[int, int], ...]:
    """Enumerate the exact finite image of the continuous 32-rounded resolver."""

    landscape = {
        (CANVAS_SHORT_EDGE, width)
        for width in range(
            CANVAS_SHORT_EDGE,
            int(CANVAS_SHORT_EDGE * 1.75) + CANVAS_MULTIPLE,
            CANVAS_MULTIPLE,
        )
    }
    half = CANVAS_MULTIPLE // 2
    maximum_dimension = CANVAS_SHORT_EDGE * int(CANVAS_MAX_ASPECT_RATIO)
    for height in range(CANVAS_MULTIPLE, CANVAS_SHORT_EDGE + 1, CANVAS_MULTIPLE):
        for width in range(CANVAS_SHORT_EDGE, maximum_dimension + 1, CANVAS_MULTIPLE):
            # In the area-limited landscape branch raw dimensions satisfy
            # h*w=max_pixels and r=w/h. Intersect the two nearest-multiple
            # rounding cells with the resolver's exact r interval.
            lower = max(
                Fraction(7, 4),
                Fraction(CANVAS_MAX_PIXELS, (height + half) ** 2),
                Fraction((width - half) ** 2, CANVAS_MAX_PIXELS),
            )
            upper = min(
                Fraction(4, 1),
                Fraction(CANVAS_MAX_PIXELS, (height - half) ** 2),
                Fraction((width + half) ** 2, CANVAS_MAX_PIXELS),
            )
            if lower >= upper:
                continue
            sample = float((lower + upper) / 2)
            if _resolve_canvas_size(sample, 1.0) == (height, width):
                landscape.add((height, width))
    reachable = landscape | {(width, height) for height, width in landscape}
    return tuple(sorted(reachable))


def _default_canvas_size(raw: dict) -> tuple[int, int]:
    height_value = raw.get("video_height", raw.get("height", 768))
    width_value = raw.get("video_width", raw.get("width", 1344))
    if isinstance(height_value, bool) or isinstance(width_value, bool):
        raise ValueError("MiniMax-H3 video dimensions must match the public canvas resolver")
    try:
        height = int(height_value)
        width = int(width_value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "MiniMax-H3 video dimensions must match the public canvas resolver"
        ) from error
    if (
        height <= 0
        or width <= 0
        or (
            (height, width) not in NATIVE_EXPLICIT_CANVAS_SIZES
            and (height, width) != _resolve_canvas_size(width, height)
        )
    ):
        raise ValueError("MiniMax-H3 video dimensions must match the public canvas resolver")
    return height, width


def _first_block_cache_threshold(raw: dict, *, default: float = 0.08) -> float:
    value = raw.get("first_block_cache_threshold", default)
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


def _transformer_ref_build_input(raw: dict) -> Path | None:
    """Resolve an explicit Ref2VA checkpoint without falling back to transformer."""

    value = raw.get("transformer_ref", raw.get("transformer_ref_path"))
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(
            "MiniMax-H3 transformer_ref must be an explicit checkpoint directory, not a flag"
        )
    return Path(value).resolve(strict=True)


def _quantized_transformer_build_input(raw: dict) -> Path | None:
    """Resolve the optional build-only Comfy FL2VA INT8 ConvRot checkpoint."""

    value = raw.get("quantized_transformer")
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(
            "MiniMax-H3 quantized_transformer must be an explicit .safetensors file, not a flag"
        )
    try:
        path = Path(value).absolute()
    except TypeError as error:
        raise ValueError(
            "MiniMax-H3 quantized_transformer must be an explicit .safetensors file"
        ) from error
    # Preserve the lexical filename for Hugging Face snapshot symlinks. The
    # strict checkpoint validator accepts the standard symlink-to-blob layout
    # and binds the released filename before following it.
    if not path.is_file():
        raise FileNotFoundError(f"MiniMax-H3 quantized transformer is missing: {path}")
    return path


def _super_resolution_build_inputs(raw: dict) -> tuple[Path | None, Path | None, float]:
    """Resolve optional official Real-ESRGAN build-only checkpoints."""

    def checkpoint(field: str) -> Path | None:
        value = raw.get(field)
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            raise ValueError(f"MiniMax-H3 {field} must be an explicit .pth file, not a flag")
        try:
            path = Path(value).absolute()
        except TypeError as error:
            raise ValueError(f"MiniMax-H3 {field} must be an explicit .pth file") from error
        if not path.is_file():
            raise FileNotFoundError(f"MiniMax-H3 {field} checkpoint is missing: {path}")
        return path

    primary = checkpoint("super_resolution_model")
    weak = checkpoint("super_resolution_weak_model")
    if weak is not None and primary is None:
        raise ValueError("MiniMax-H3 super_resolution_weak_model requires super_resolution_model")
    # The public inference recipe uses 0.5 dynamic-network interpolation. A
    # main-only build remains supported and selects that checkpoint exactly.
    return primary, weak, (0.5 if weak is not None else 1.0)


class MiniMaxH3Plugin:
    name = "minimax_h3"
    default_build_precision = "bf16"
    runtime_strategy = "diffusion_minimax_h3"
    pipeline_classes = ("MiniMaxH3ModularPipeline", "MiniMaxH3Pipeline")

    def matches(self, model_type: str) -> bool:
        return model_type.lower().replace("-", "_") in {
            "minimax_h3",
            "minimaxh3",
            "minimaxh3modularpipeline",
            "minimaxh3pipeline",
        }

    def load_weights(self, model_dir: str, config, **_kwargs) -> dict:
        del config
        root = Path(model_dir)
        required_dirs = ("transformer", "text_encoder", "vae", "audio_vae", "tokenizer")
        missing = [str(root / name) for name in required_dirs if not (root / name).is_dir()]
        if missing:
            raise FileNotFoundError(
                "Incomplete MiniMax-H3 Diffusers checkpoint: " + ", ".join(missing)
            )
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

    def build_engine(self, *_args, **_kwargs) -> bytes:
        raise NotImplementedError("MiniMax-H3 uses build_components(), not build_engine()")

    def build_staged_bundle(
        self,
        model_dir: str,
        writer: "BundleWriter",
        config,
        weights: dict,
        *,
        plans_dir: str | Path,
        precision: str,
        verbose: bool = False,
        parallel_config=None,
        max_batch_size: int = 1,
    ) -> None:
        """Build a staged RTX bundle without retaining serialized plans in RAM."""

        if precision.lower() != "bf16":
            raise ValueError("MiniMax-H3 TensorRT-RTX staged builds require BF16")
        if max_batch_size != 1:
            raise ValueError("MiniMax-H3 TensorRT-RTX staged builds require max_batch_size=1")
        mode = str(getattr(parallel_config, "mode", "single"))
        if mode != "single":
            raise ValueError("MiniMax-H3 TensorRT-RTX staged builds require one GPU")

        raw = _effective_build_config(getattr(config, "raw", {}))
        if raw.get("_fp32_layers"):
            raise ValueError("MiniMax-H3 TensorRT-RTX staged builds do not support FP32 layers")
        transformer_ref_path = _transformer_ref_build_input(raw)
        quantized_transformer_path = _quantized_transformer_build_input(raw)
        (
            super_resolution_model,
            super_resolution_weak_model,
            _super_resolution_strength,
        ) = _super_resolution_build_inputs(raw)
        staged_raw = dict(raw)
        staged_raw.setdefault("first_block_cache", True)
        staged_raw.setdefault("denoiser_cache_mode", "first_block")
        _fixed_profile(staged_raw)
        expected_request = {
            "num_inference_steps": 50,
            "seed": 0,
        }
        mismatches = {
            name: (raw[name], value)
            for name, value in expected_request.items()
            if name in raw and int(raw[name]) != value
        }
        if mismatches:
            raise ValueError(f"Unsupported MiniMax-H3 staged profile: {mismatches}")
        _default_canvas_size(raw)
        _default_num_frames(raw)

        from .staged_build import build_staged_bundle

        root = Path(weights.get("_model_dir", model_dir))
        staged_options = {"verbose": verbose}
        staged_options["runtime_defaults"] = {
            "height": _default_canvas_size(raw)[0],
            "width": _default_canvas_size(raw)[1],
            "num_frames": _default_num_frames(raw),
            "first_block_cache_threshold": _first_block_cache_threshold(raw),
        }
        if transformer_ref_path is not None:
            staged_options["transformer_ref"] = transformer_ref_path
        if quantized_transformer_path is not None:
            staged_options["quantized_transformer"] = quantized_transformer_path
        if super_resolution_model is not None:
            staged_options["super_resolution_model"] = super_resolution_model
        if super_resolution_weak_model is not None:
            staged_options["super_resolution_weak_model"] = super_resolution_weak_model
        build_staged_bundle(root, writer, plans_dir=plans_dir, **staged_options)

    def build_components(
        self,
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
        quantized_transformer_path = _quantized_transformer_build_input(raw)
        (
            super_resolution_model,
            super_resolution_weak_model,
            super_resolution_strength,
        ) = _super_resolution_build_inputs(raw)
        quantized_transformer_identity = None
        if quantized_transformer_path is not None:
            from .quantized_checkpoint import validate_quantized_transformer_checkpoint

            quantized_transformer_identity = validate_quantized_transformer_checkpoint(
                quantized_transformer_path
            )
        profile = _fixed_profile(raw)
        profile.validate()
        workspace_limits = default_workspace_limit_bytes()
        super_resolution_identity = None
        if super_resolution_model is not None:
            workspace_limits["video_super_resolution.plan"] = TRT_DEFAULT_WORKSPACE_POLICY
            super_resolution_identity = super_resolution_source_identity(
                super_resolution_model,
                super_resolution_weak_model,
                denoise_strength=super_resolution_strength,
            )
        from .adaln_builder import build_adaln_precompute_engine
        from .adaln_builder import checkpoint_keys as adaln_checkpoint_keys
        from .dit_builder import (
            build_dit_finish_engine,
            build_dit_head_engine,
            build_dit_tail_engine,
            finish_checkpoint_keys,
            head_checkpoint_keys,
            tail_checkpoint_keys,
        )
        from .multimodal_text_encoder_builder import (
            build_multimodal_text_encoder_engine,
            checkpoint_keys as text_encoder_checkpoint_keys,
        )
        from .multimodal_vision_builder import (
            build_multimodal_vision_encoder_engine,
            checkpoint_keys as vision_encoder_checkpoint_keys,
        )

        adaln_specs = (
            (
                "adaln_precompute",
                "adaln_precompute.plan",
                build_adaln_precompute_engine,
                adaln_checkpoint_keys(profile),
                None,
            ),
        )
        denoiser_specs = (
            (
                "denoiser_head",
                "denoiser_head.plan",
                build_dit_head_engine,
                head_checkpoint_keys(profile),
                None,
            ),
            (
                "denoiser_tail",
                "denoiser_tail.plan",
                build_dit_tail_engine,
                tail_checkpoint_keys(profile),
                None,
            ),
            (
                "denoiser_finish",
                "denoiser_finish.plan",
                build_dit_finish_engine,
                finish_checkpoint_keys(),
                None,
            ),
        )
        checkpoint_groups = (
            *(spec[3] for spec in adaln_specs),
            *(spec[3] for spec in denoiser_specs),
        )
        if quantized_transformer_path is None:
            validate_component_key_partition(weights["_transformer_dir"], checkpoint_groups)

        def load_transformer_weights(selected_keys) -> dict:
            if quantized_transformer_path is not None:
                from .quantized_checkpoint import (
                    load_selected_quantized_transformer_weights,
                )

                return load_selected_quantized_transformer_weights(
                    quantized_transformer_path, selected_keys
                )
            state = load_selected_component_state_dict(weights["_transformer_dir"], selected_keys)
            result = numpy_state(state)
            del state
            return result

        text_state = load_selected_component_state_dict(
            weights["_text_encoder_dir"], text_encoder_checkpoint_keys()
        )
        text_weights = numpy_state(text_state)
        del text_state
        text_encoder_plan = build_multimodal_text_encoder_engine(
            text_weights,
            verbose=verbose,
            consume_weights=True,
            workspace_bytes=workspace_limits["text_encoder.plan"],
        )
        del text_weights
        gc.collect()

        vision_state = load_selected_component_state_dict(
            weights["_text_encoder_dir"], vision_encoder_checkpoint_keys()
        )
        vision_weights = numpy_state(vision_state)
        del vision_state
        vision_encoder_plan = build_multimodal_vision_encoder_engine(
            vision_weights,
            verbose=verbose,
            consume_weights=True,
            workspace_bytes=workspace_limits["vision_encoder.plan"],
        )
        del vision_weights
        gc.collect()

        adaln_components = {}
        denoiser_components = {}
        for component_name, filename, adaln_builder, selected_keys, projection_index in adaln_specs:
            adaln_weights = load_transformer_weights(selected_keys)
            adaln_options = {
                "verbose": verbose,
                "consume_weights": True,
                "workspace_bytes": None,
                "weight_streaming": True,
            }
            if projection_index is None:
                adaln_plan = adaln_builder(
                    adaln_weights,
                    profile,
                    **adaln_options,
                )
            else:
                adaln_plan = adaln_builder(
                    adaln_weights,
                    profile,
                    projection_index,
                    **adaln_options,
                )
            del adaln_weights
            gc.collect()
            adaln_components[component_name] = adaln_plan

        for (
            component_name,
            filename,
            denoiser_builder,
            selected_keys,
            transition_index,
        ) in denoiser_specs:
            dit_weights = load_transformer_weights(selected_keys)
            denoiser_options = {
                "verbose": verbose,
                "consume_weights": True,
                "workspace_bytes": None,
                "weight_streaming": True,
            }
            if transition_index is None:
                denoiser_plan = denoiser_builder(
                    dit_weights,
                    profile,
                    **denoiser_options,
                )
            else:
                denoiser_plan = denoiser_builder(
                    dit_weights,
                    profile,
                    transition_index,
                    **denoiser_options,
                )
            del dit_weights
            gc.collect()
            denoiser_components[component_name] = denoiser_plan

        from .vae_builder import (
            build_vae_tile_decoder_engine,
            checkpoint_keys as vae_checkpoint_keys,
        )

        from .fl2va_vae_encoder_builder import (
            build_keyframe_vae_encoder_engine,
            checkpoint_keys as keyframe_vae_checkpoint_keys,
        )

        keyframe_vae_state = load_selected_component_state_dict(
            weights["_vae_dir"], keyframe_vae_checkpoint_keys()
        )
        keyframe_vae_weights = numpy_state(keyframe_vae_state)
        del keyframe_vae_state
        keyframe_vae_encoder_plan = build_keyframe_vae_encoder_engine(
            keyframe_vae_weights,
            verbose=verbose,
            consume_weights=True,
            workspace_bytes=workspace_limits["fl2va_keyframe_vae_encoder.plan"],
        )
        del keyframe_vae_weights
        gc.collect()

        vae_state = load_selected_component_state_dict(weights["_vae_dir"], vae_checkpoint_keys())
        vae_weights = numpy_state(vae_state)
        del vae_state
        vae_decoder_plan = build_vae_tile_decoder_engine(
            vae_weights,
            verbose=verbose,
            consume_weights=True,
            workspace_bytes=workspace_limits["vae_tile_decoder.plan"],
        )
        del vae_weights
        gc.collect()

        from .audio_vae_builder import (
            build_audio_vae_decoder_engine,
            checkpoint_keys as audio_vae_checkpoint_keys,
            decoder_config_from_checkpoint,
        )

        audio_vae_config = json.loads((Path(weights["_audio_vae_dir"]) / "config.json").read_text())
        audio_decoder_profile = decoder_config_from_checkpoint(
            audio_vae_config,
            latent_frames=AUDIO_LATENT_FRAMES_OPT,
            min_latent_frames=AUDIO_LATENT_FRAMES_MIN,
            max_latent_frames=AUDIO_LATENT_FRAMES_MAX,
        )
        audio_vae_state = load_selected_component_state_dict(
            weights["_audio_vae_dir"], audio_vae_checkpoint_keys(audio_decoder_profile)
        )
        audio_vae_weights = numpy_state(audio_vae_state)
        del audio_vae_state
        audio_vae_decoder_plan = build_audio_vae_decoder_engine(
            audio_vae_weights,
            audio_decoder_profile,
            verbose=verbose,
            consume_weights=True,
            workspace_bytes=workspace_limits["audio_vae_decoder.plan"],
        )
        del audio_vae_weights
        gc.collect()

        video_super_resolution_plan = None
        if super_resolution_model is not None:
            from .super_resolution_builder import build_super_resolution_engine

            video_super_resolution_plan = build_super_resolution_engine(
                super_resolution_model,
                super_resolution_weak_model,
                denoise_strength=super_resolution_strength,
                verbose=verbose,
                workspace_bytes=None,
                weight_streaming=False,
                learned_residual_strength=(SUPER_RESOLUTION_LEARNED_RESIDUAL_STRENGTH),
            )
            if not isinstance(video_super_resolution_plan, bytes):
                raise RuntimeError(
                    "MiniMax-H3 in-memory super-resolution builder did not return a plan"
                )
            gc.collect()
        tokenizer_json = (Path(weights["_tokenizer_dir"]) / "tokenizer.json").read_bytes()

        if quantized_transformer_identity is not None:
            current_quantized_transformer_identity = validate_quantized_transformer_checkpoint(
                quantized_transformer_path
            )
            if current_quantized_transformer_identity != quantized_transformer_identity:
                raise ValueError(
                    "MiniMax-H3 quantized_transformer changed while component plans were built"
                )
        if super_resolution_identity is not None:
            current_super_resolution_identity = super_resolution_source_identity(
                super_resolution_model,
                super_resolution_weak_model,
                denoise_strength=super_resolution_strength,
            )
            if current_super_resolution_identity != super_resolution_identity:
                raise ValueError(
                    "MiniMax-H3 super-resolution checkpoints changed while component plans "
                    "were built"
                )

        return {
            "text_encoder": text_encoder_plan,
            "vision_encoder": vision_encoder_plan,
            **adaln_components,
            **denoiser_components,
            "vae_decoder": vae_decoder_plan,
            "keyframe_vae_encoder": keyframe_vae_encoder_plan,
            "audio_vae_decoder": audio_vae_decoder_plan,
            **(
                {"video_super_resolution": video_super_resolution_plan}
                if video_super_resolution_plan is not None
                else {}
            ),
            "audio_vae_config": audio_vae_config,
            "audio_decoder_profile": audio_decoder_profile,
            "profile": profile,
            # Text/VAE paths remain explicit so follow-on native component
            # builders cannot silently substitute a different checkpoint.
            "vae_dir": weights["_vae_dir"],
            "audio_vae_dir": weights["_audio_vae_dir"],
            "tokenizer_dir": weights["_tokenizer_dir"],
            "tokenizer_json": tokenizer_json,
            "provenance": {
                "workspace_limit_bytes": workspace_limits,
                **(
                    {
                        "quantized_transformer": validate_quantized_transformer_metadata(
                            quantized_transformer_identity.bundle_metadata()
                        )
                    }
                    if quantized_transformer_identity is not None
                    else {}
                ),
                **(
                    {
                        "super_resolution": validate_super_resolution_source_identity(
                            super_resolution_identity
                        )
                    }
                    if super_resolution_identity is not None
                    else {}
                ),
            },
        }

    def diffusion_bundle_sections(
        self, components: dict, *, parallel_config=None
    ) -> list[tuple[str, bytes]]:
        del parallel_config
        shared = [
            ("text_encoder_plan", components["text_encoder"]),
            ("vision_encoder_plan", components["vision_encoder"]),
            ("adaln_precompute_plan", components["adaln_precompute"]),
        ]
        denoiser = [
            ("denoiser_head_plan", components["denoiser_head"]),
            ("denoiser_tail_plan", components["denoiser_tail"]),
            ("denoiser_finish_plan", components["denoiser_finish"]),
        ]
        sections = [
            *shared,
            *denoiser,
            ("fl2va_keyframe_vae_encoder_plan", components["keyframe_vae_encoder"]),
            ("vae_tile_decoder_plan", components["vae_decoder"]),
            ("audio_vae_decoder_plan", components["audio_vae_decoder"]),
        ]
        if "video_super_resolution" in components:
            sections.append(("video_super_resolution_plan", components["video_super_resolution"]))
        sections.append(("tokenizer.json", components["tokenizer_json"]))
        return sections

    def diffusion_bundle_config(self, config, *, components: dict) -> dict:
        raw = _effective_build_config(getattr(config, "raw", {}))
        profile = components["profile"]
        expected_profile = replace(SOL_ENGINE_1344X768_124_TO_345F, first_block_cache=True)
        if profile != expected_profile:
            raise ValueError("Unsupported MiniMax-H3 dynamic media profile")
        fixed_request = {"num_inference_steps": 50}
        mismatches = {
            name: (raw[name], value)
            for name, value in fixed_request.items()
            if name in raw and int(raw[name]) != value
        }
        if mismatches:
            raise ValueError(f"Unsupported MiniMax-H3 runtime profile: {mismatches}")
        default_height, default_width = _default_canvas_size(raw)
        default_num_frames = _default_num_frames(raw)
        provenance = components.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("MiniMax-H3 components are missing exact build provenance")
        quantized_fields: dict[str, object] = {}
        if "quantized_transformer" in provenance:
            validate_quantized_transformer_metadata(provenance["quantized_transformer"])
            quantized_fields["quantization"] = dict(QUANTIZED_TRANSFORMER_CONFIG)
        has_super_resolution_plan = "video_super_resolution" in components
        has_super_resolution_provenance = "super_resolution" in provenance
        if has_super_resolution_plan != has_super_resolution_provenance:
            raise ValueError(
                "MiniMax-H3 super-resolution plan and provenance must be present together"
            )
        super_resolution_fields: dict[str, object] = {}
        additional_plan_filenames: tuple[str, ...] = ()
        if has_super_resolution_plan:
            super_resolution_fields["super_resolution"] = super_resolution_bundle_config(
                provenance["super_resolution"]
            )
            additional_plan_filenames = ("video_super_resolution.plan",)
        validate_workspace_limit_bytes(
            provenance.get("workspace_limit_bytes"),
            profile=profile,
            additional_plan_filenames=additional_plan_filenames,
        )
        adaln_sections = ["adaln_precompute_plan"]
        denoiser_sections = [
            "denoiser_head_plan",
            "denoiser_tail_plan",
            "denoiser_finish_plan",
        ]
        audio_vae_config = components.get("audio_vae_config")
        audio_decoder_profile = components.get("audio_decoder_profile")
        if not isinstance(audio_vae_config, dict) or audio_decoder_profile is None:
            raise ValueError("MiniMax-H3 components are missing AudioVAE metadata")
        latent_mean = audio_vae_config.get("latents_mean")
        latent_std = audio_vae_config.get("latents_std")
        if (
            not isinstance(latent_mean, list)
            or not isinstance(latent_std, list)
            or len(latent_mean) != profile.audio_in_channels
            or len(latent_std) != profile.audio_in_channels
        ):
            raise ValueError("MiniMax-H3 AudioVAE config has invalid latent normalization")
        return {
            "checkpoint_revision": "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc",
            **provenance,
            **quantized_fields,
            **super_resolution_fields,
            "height": default_height,
            "width": default_width,
            "canvas_multiple": CANVAS_MULTIPLE,
            "canvas_short_edge": CANVAS_SHORT_EDGE,
            "canvas_max_pixels": CANVAS_MAX_PIXELS,
            "explicit_canvas_sizes": [list(size) for size in NATIVE_EXPLICIT_CANVAS_SIZES],
            "min_aspect_ratio": CANVAS_MIN_ASPECT_RATIO,
            "max_aspect_ratio": CANVAS_MAX_ASPECT_RATIO,
            "public_workflows": ["t2va", "fl2va"],
            "conditioning": {
                "implementation": "shared_native_qwen3_vl",
                "text_encoder_section": "text_encoder_plan",
                "vision_encoder_section": "vision_encoder_plan",
                "keyframe_vae_encoder_section": "fl2va_keyframe_vae_encoder_plan",
                "text_sequence_profile": [1, 1144, 2641],
                "vision_patch_profile": [2040, 4032, 4176],
                "vision_row_profile": [1, 1008, 2088],
                "t2va_dummy_vision_rows": 1,
                "t2va_vision_count": 0,
                "t2va_vision_mask_nonzero": 0,
                "keyframe_vae_tile_batch_profile": [1, 28, 33],
                "reachable_canvas_count": len(_reachable_canvas_sizes()),
                "max_rounded_canvas": [576, 1856],
                "max_condition_video_rows": 2088,
                "mode_coupled_profile_required": True,
            },
            "num_frames": default_num_frames,
            "num_frames_min": VIDEO_NUM_FRAMES_MIN,
            "num_frames_opt": VIDEO_NUM_FRAMES_OPT,
            "num_frames_max": VIDEO_NUM_FRAMES_MAX,
            "fps": 24,
            "num_inference_steps": 50,
            "guidance_scale": 1.0,
            "scheduler_grid_points": 50,
            "transformer_forwards": 49,
            "attention_mode": "dense",
            "runtime_memory": {
                "mode": "staged",
                "weight_streaming_budget_bytes": RTX_WEIGHT_STREAMING_BUDGET_BYTES,
            },
            "seed": int(raw.get("seed", 0)),
            "bundle_loading": {
                "mode": "staged",
                "eager_sections": ["tokenizer.json", "config.json"],
                "lazy_sections": [
                    "text_encoder_plan",
                    "vision_encoder_plan",
                    *adaln_sections,
                    *denoiser_sections,
                    "fl2va_keyframe_vae_encoder_plan",
                    "vae_tile_decoder_plan",
                    "audio_vae_decoder_plan",
                    *(["video_super_resolution_plan"] if has_super_resolution_plan else []),
                ],
            },
            "first_block_cache": True,
            "denoiser_cache_mode": "first_block",
            "denoiser_profile_count": 3,
            "denoiser_profile_layout": "five_second_t2va_then_fl2va_then_public_dynamic",
            "first_block_cache_threshold": _first_block_cache_threshold(raw),
            "text_rows": profile.text_rows,
            "text_rows_min": profile.min_text_rows,
            "text_rows_opt": profile.opt_text_rows,
            "text_rows_max": profile.text_rows,
            "audio_rows": profile.opt_audio_rows,
            "audio_rows_min": profile.min_audio_rows,
            "audio_rows_opt": profile.opt_audio_rows,
            "audio_rows_max": profile.audio_rows,
            "audio_latent_frames": audio_decoder_profile.latent_frames,
            "audio_latent_frames_min": audio_decoder_profile.min_latent_frames,
            "audio_latent_frames_opt": audio_decoder_profile.latent_frames,
            "audio_latent_frames_max": audio_decoder_profile.max_latent_frames,
            "audio_sample_rate": audio_decoder_profile.sampling_rate,
            "audio_hop_length": audio_decoder_profile.hop_length,
            "audio_channels": audio_decoder_profile.batch_size,
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
        }

    def diffusion_tokenizer_add_special_tokens(self, *_args, **_kwargs) -> bool:
        return False

    def diffusion_tokenizer_bundle_sections(self, *_args, **_kwargs):
        return []


plugin = MiniMaxH3Plugin()


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build the unified T2VA/FL2VA/Ref2VA bundle through TensorRT-RTX."""

    if request.task != "image_generation":
        raise ValueError("minimax_h3 supports only task=image_generation")
    if request.backend != "trt_rtx":
        raise ValueError("MiniMax-H3 requires backend=trt_rtx")
    if request.precision != "bf16":
        raise ValueError("MiniMax-H3 requires precision=bf16")
    if request.dynamic_kv_cache:
        raise NotImplementedError("minimax_h3 does not support dynamic_kv_cache")
    if request.max_sequence_length is not None:
        raise NotImplementedError("minimax_h3 does not support max_sequence_length")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise ValueError("MiniMax-H3 requires single-device parallel settings")
    if request.max_batch_size != 1:
        raise ValueError("MiniMax-H3 requires max_batch_size=1")
    if request.fp32_layers:
        raise NotImplementedError("MiniMax-H3 does not support fp32_layers")

    family_options = dict(getattr(request, "family_options", ()))
    quantized_path = family_options.get("quantized_transformer")
    if request.quantization not in {None, "none", "int8_tensorwise_convrot"}:
        raise ValueError("MiniMax-H3 supports only int8_tensorwise_convrot quantization")
    if request.quantization == "int8_tensorwise_convrot" and not quantized_path:
        raise ValueError(
            "MiniMax-H3 int8_tensorwise_convrot requires "
            "--set minimax_h3.quantized_transformer=<checkpoint>"
        )

    raw = {
        "_family_build_options": {"minimax_h3": family_options},
        "height": int(request.image_height or 768),
        "width": int(request.image_width or 1344),
        "video_num_frames": int(request.video_num_frames or VIDEO_NUM_FRAMES_OPT),
        "num_inference_steps": 50,
        "seed": 0,
    }
    config = SimpleNamespace(raw=raw)
    _default_canvas_size(raw)
    _default_num_frames(raw)
    weights = plugin.load_weights(str(request.model_dir), config)

    writer.set_header(family="minimax_h3", task=request.task, backend=request.backend)
    plans_dir = request.output_path.with_name(f"{request.output_path.name}.plans")
    plugin.build_staged_bundle(
        str(request.model_dir),
        writer,
        config,
        weights,
        plans_dir=plans_dir,
        precision=request.precision,
        verbose=request.verbose,
        max_batch_size=request.max_batch_size,
    )
