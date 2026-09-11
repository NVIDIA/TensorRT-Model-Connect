# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-Model-Connect family plugin for MiniMaxAI/MiniMax-H3."""

from __future__ import annotations

from fractions import Fraction
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from .config import (
    CANVAS_MAX_ASPECT_RATIO,
    CANVAS_MAX_PIXELS,
    CANVAS_MIN_ASPECT_RATIO,
    CANVAS_MULTIPLE,
    CANVAS_SHORT_EDGE,
    NATIVE_EXPLICIT_CANVAS_SIZES,
    SOL_ENGINE_1344X768_124_TO_345F,
    VIDEO_NUM_FRAMES_MAX,
    VIDEO_NUM_FRAMES_MIN,
    VIDEO_NUM_FRAMES_OPT,
)
from .runtime_config_schema import normalize_build_options as validate_build_options
from .delivery import SR_SOURCE_SHAPE

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


def _effective_build_config(raw: dict) -> dict:
    family_options = raw.get("_family_build_options", {})
    options = family_options.get("minimax_h3", {}) if isinstance(family_options, dict) else {}
    if not isinstance(options, dict):
        raise ValueError("minimax_h3 build options must be an object")
    return {**raw, **validate_build_options(options)}


def _public_dynamic_profile(raw: dict):
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
    sr = raw.get("super_resolution", False)
    default_height, default_width = SR_SOURCE_SHAPE if sr else (768, 1344)
    height_value = raw.get("video_height", raw.get("height", default_height))
    width_value = raw.get("video_width", raw.get("width", default_width))
    if isinstance(height_value, bool) or isinstance(width_value, bool):
        raise ValueError("MiniMax-H3 video dimensions must match the public canvas resolver")
    try:
        height = int(height_value)
        width = int(width_value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "MiniMax-H3 video dimensions must match the public canvas resolver"
        ) from error
    if sr and (height, width) != SR_SOURCE_SHAPE:
        raise ValueError("MiniMax-H3 super_resolution=true requires height=480 and width=864")
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


class MiniMaxH3Plugin:
    def load_weights(self, model_dir: str, config, **_kwargs) -> dict:
        del config
        root = Path(model_dir)
        required_dirs = ("transformer", "vae", "audio_vae", "tokenizer")
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
        return {"_model_dir": str(root)}

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
        from .delivery import resolve_quantized_sources, resolve_super_resolution_sources

        root = Path(weights.get("_model_dir", model_dir))
        staged_raw = dict(raw)
        staged_raw.setdefault("first_block_cache", True)
        staged_raw.setdefault("denoiser_cache_mode", "first_block")
        _public_dynamic_profile(staged_raw)
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

        # All public builds use the same quantized checkpoints for all three
        # workflows. Only the explicit SR flag changes the delivery mode.
        staged_options = {"verbose": verbose, **resolve_quantized_sources(root, raw)}
        super_resolution_model, super_resolution_weak_model = resolve_super_resolution_sources(raw)
        staged_options["runtime_defaults"] = {
            "height": _default_canvas_size(raw)[0],
            "width": _default_canvas_size(raw)[1],
            "num_frames": _default_num_frames(raw),
            "first_block_cache_threshold": _first_block_cache_threshold(raw),
        }
        staged_options["runtime_defaults"].update(
            ref2va_first_block_cache=raw.get("ref2va_first_block_cache", True),
            ref2va_first_block_cache_threshold=raw.get("ref2va_first_block_cache_threshold", 0.08),
        )
        if super_resolution_model is not None:
            staged_options["super_resolution_model"] = super_resolution_model
        if super_resolution_weak_model is not None:
            staged_options["super_resolution_weak_model"] = super_resolution_weak_model
        build_staged_bundle(root, writer, plans_dir=plans_dir, **staged_options)


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

    family_options = validate_build_options(dict(getattr(request, "family_options", ())))
    if request.quantization not in {None, "int8_tensorwise_convrot"}:
        raise ValueError("MiniMax-H3 delivery requires Comfy INT8 denoisers and the NVFP4 text checkpoint")
    if (request.image_height is None) != (request.image_width is None):
        raise ValueError("MiniMax-H3 requires both image height and width, or neither")
    sr = family_options.get("super_resolution", False)
    default_height, default_width = SR_SOURCE_SHAPE if sr else (768, 1344)

    raw = {
        "_family_build_options": {"minimax_h3": family_options},
        "height": int(request.image_height or default_height),
        "width": int(request.image_width or default_width),
        "video_num_frames": int(request.video_num_frames or VIDEO_NUM_FRAMES_OPT),
        "num_inference_steps": 50,
        "seed": 0,
    }
    config = SimpleNamespace(raw=raw)
    _default_canvas_size(_effective_build_config(raw))
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
