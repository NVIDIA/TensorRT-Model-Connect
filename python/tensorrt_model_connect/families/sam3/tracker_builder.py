# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the stateful SAM3 video tracker with TensorRT Network API layers.

The detector and vision plans own the shared frame backbone. These seven
tracker plans own the learned initialization, recurrent conditioning, mask
decoding, pointer projection, and spatial-memory encoding operations. Session
history, object identity, association, and memory policy remain in the native
C++ runtime.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tensorrt_model_connect import trt_compat

from .tracker_weights import TrackerWeights, load_tracker_weights

if TYPE_CHECKING:
    from .tracker_memory_ffi_builder import TrackerMemoryPlanSpec
    from .tracker_step_ffi_builder import TrackerStepPlanSpec


TRACKER_INIT_SECTION = "sam3_tracker_init_engine_plan"
TRACKER_STEP_SECTION = "sam3_tracker_step_engine_plan"
TRACKER_STEP_BATCH2_SECTION = "sam3_tracker_step_batch2_engine_plan"
TRACKER_MEMORY_SECTION = "sam3_tracker_memory_engine_plan"
TRACKER_MEMORY_BATCH2_SECTION = "sam3_tracker_memory_batch2_engine_plan"
TRACKER_HARD_MEMORY_SECTION = "sam3_tracker_hard_memory_engine_plan"
TRACKER_HARD_MEMORY_BATCH2_SECTION = "sam3_tracker_hard_memory_batch2_engine_plan"

# Meta selects at most four temporally closest conditioning pointers and then
# appends up to fifteen quality-filtered non-conditioning pointers.
SAM3_TRACKER_MAX_VIDEO_FRAMES = 1024
SAM3_TRACKER_RECONDITION_CADENCE = 16
SAM3_TRACKER_MAX_CONDITIONING_POINTERS = 4
SAM3_TRACKER_MAX_POINTER_INPUTS = SAM3_TRACKER_MAX_CONDITIONING_POINTERS + 15

_SPATIAL_TOKENS = 72 * 72
_MIN_MEMORY_FRAMES = 1
_OPT_MEMORY_FRAMES = 3
_NUM_MASK_MEMORY_FRAMES = 7
_MAX_CONDITIONING_FRAMES = 4
_MAX_MEMORY_FRAMES = _MAX_CONDITIONING_FRAMES + _NUM_MASK_MEMORY_FRAMES - 1
_OPT_POINTER_OBJECTS = 4
_POINTER_RECENT_WINDOW = 16
_MAX_POINTER_OBJECTS = SAM3_TRACKER_MAX_POINTER_INPUTS
_BATCH2_OBJECTS = 2
_BATCH2_OPT_MEMORY_FRAMES = _NUM_MASK_MEMORY_FRAMES
_BATCH2_OPT_POINTER_OBJECTS = _POINTER_RECENT_WINDOW

_FEATURE_SHAPES = (
    (1, 32, 288, 288),
    (1, 64, 144, 144),
    (1, 256, 72, 72),
)
_POSITION_SHAPE = (1, 256, 72, 72)
_DETECTOR_MASK_SHAPE = (1, 1, 288, 288)

# The direct TensorRT graphs below intentionally specialize the official SAM3
# tracker architecture.  Keep every config-controlled constant used by the
# decoder, recurrent attention, and memory encoder in one fail-closed contract
# so a different checkpoint cannot silently execute with the wrong graph.
_TRACKER_ARCHITECTURE_CONTRACT: tuple[tuple[tuple[str, ...], object], ...] = (
    (("vision_config", "fpn_hidden_size"), 256),
    (("vision_config", "num_feature_levels"), 3),
    (("prompt_encoder_config", "hidden_size"), 256),
    (("prompt_encoder_config", "image_size"), 1008),
    (("prompt_encoder_config", "patch_size"), 14),
    (("prompt_encoder_config", "mask_input_channels"), 16),
    (("prompt_encoder_config", "num_point_embeddings"), 4),
    (("prompt_encoder_config", "layer_norm_eps"), 1e-6),
    (("prompt_encoder_config", "hidden_act"), "gelu"),
    (("prompt_encoder_config", "scale"), 1),
    (("mask_decoder_config", "hidden_size"), 256),
    (("mask_decoder_config", "num_attention_heads"), 8),
    (("mask_decoder_config", "num_hidden_layers"), 2),
    (("mask_decoder_config", "attention_downsample_rate"), 2),
    (("mask_decoder_config", "mlp_dim"), 2048),
    (("mask_decoder_config", "num_multimask_outputs"), 3),
    (("mask_decoder_config", "iou_head_depth"), 3),
    (("mask_decoder_config", "iou_head_hidden_dim"), 256),
    (("memory_attention_hidden_size",), 256),
    (("memory_attention_num_attention_heads",), 1),
    (("memory_attention_num_layers",), 4),
    (("memory_attention_feed_forward_hidden_size",), 2048),
    (("memory_attention_feed_forward_hidden_act",), "relu"),
    (("memory_attention_downsample_rate",), 1),
    (("memory_attention_rope_feat_sizes",), (72, 72)),
    (("memory_attention_rope_theta",), 10000),
    (("memory_encoder_hidden_size",), 256),
    (("memory_encoder_output_channels",), 64),
    (("mask_downsampler_embed_dim",), 256),
    (("mask_downsampler_hidden_act",), "gelu"),
    (("mask_downsampler_kernel_size",), 3),
    (("mask_downsampler_padding",), 1),
    (("mask_downsampler_stride",), 2),
    (("mask_downsampler_total_stride",), 16),
    (("memory_fuser_embed_dim",), 256),
    (("memory_fuser_hidden_act",), "gelu"),
    (("memory_fuser_intermediate_dim",), 1024),
    (("memory_fuser_kernel_size",), 7),
    (("memory_fuser_padding",), 3),
    (("memory_fuser_num_layers",), 2),
    (("memory_fuser_layer_scale_init_value",), 1e-6),
    (("sigmoid_scale_for_mem_enc",), 20.0),
    (("sigmoid_bias_for_mem_enc",), -10.0),
    (("enable_occlusion_spatial_embedding",), True),
    (("enable_temporal_pos_encoding_for_object_pointers",), True),
    (("multimask_output_for_tracking",), True),
    (("multimask_output_in_sam",), True),
    (("multimask_min_pt_num",), 0),
    (("multimask_max_pt_num",), 1),
)

_ROOT_TRACKER_ARCHITECTURE_CONTRACT: tuple[tuple[tuple[str, ...], object], ...] = (
    (("low_res_mask_size",), 288),
)

_MISSING = object()


def _step_profile_shapes(
    name: str,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
    if name in {"memory_features", "memory_position"}:
        return (
            (1, _MIN_MEMORY_FRAMES, _SPATIAL_TOKENS, 64),
            (1, _OPT_MEMORY_FRAMES, _SPATIAL_TOKENS, 64),
            (1, _MAX_MEMORY_FRAMES, _SPATIAL_TOKENS, 64),
        )
    if name == "memory_temporal_offsets":
        return (
            (1, _MIN_MEMORY_FRAMES),
            (1, _OPT_MEMORY_FRAMES),
            (1, _MAX_MEMORY_FRAMES),
        )
    if name == "object_pointers":
        return (
            (1, 1, 256),
            (1, _OPT_POINTER_OBJECTS, 256),
            (1, _MAX_POINTER_OBJECTS, 256),
        )
    if name == "object_pointer_temporal_offsets":
        return (
            (1, 1),
            (1, _OPT_POINTER_OBJECTS),
            (1, _MAX_POINTER_OBJECTS),
        )
    return None


def _step_batch2_profile_shapes(
    name: str,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
    if name in {"memory_features", "memory_position"}:
        return (
            (_BATCH2_OBJECTS, _MIN_MEMORY_FRAMES, _SPATIAL_TOKENS, 64),
            (_BATCH2_OBJECTS, _BATCH2_OPT_MEMORY_FRAMES, _SPATIAL_TOKENS, 64),
            (_BATCH2_OBJECTS, _MAX_MEMORY_FRAMES, _SPATIAL_TOKENS, 64),
        )
    if name == "memory_temporal_offsets":
        return (
            (_BATCH2_OBJECTS, _MIN_MEMORY_FRAMES),
            (_BATCH2_OBJECTS, _BATCH2_OPT_MEMORY_FRAMES),
            (_BATCH2_OBJECTS, _MAX_MEMORY_FRAMES),
        )
    if name == "object_pointers":
        return (
            (_BATCH2_OBJECTS, 1, 256),
            (_BATCH2_OBJECTS, _BATCH2_OPT_POINTER_OBJECTS, 256),
            (_BATCH2_OBJECTS, _MAX_POINTER_OBJECTS, 256),
        )
    if name == "object_pointer_temporal_offsets":
        return (
            (_BATCH2_OBJECTS, 1),
            (_BATCH2_OBJECTS, _BATCH2_OPT_POINTER_OBJECTS),
            (_BATCH2_OBJECTS, _MAX_POINTER_OBJECTS),
        )
    return None


def _validate_tracker_model(tracker: Any) -> None:
    """Validate a tracker-like object's fixed architecture contract."""

    _validate_tracker_config(tracker.config)


def _config_value(config: Any, path: tuple[str, ...], *, prefix: str) -> Any:
    current = config
    for name in path:
        if isinstance(current, Mapping):
            value = current.get(name, _MISSING)
        else:
            value = getattr(current, name, _MISSING)
        if value is _MISSING:
            field = ".".join((prefix, *path)) if prefix else ".".join(path)
            raise RuntimeError(f"SAM3 tracker configuration is missing required field {field}")
        current = value
    return current


def _normalized_config_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_normalized_config_value(item) for item in value)
    return value


def _config_values_match(actual: Any, expected: object) -> bool:
    actual = _normalized_config_value(actual)
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, float):
        return (
            not isinstance(actual, bool)
            and isinstance(actual, (int, float))
            and float(actual) == expected
        )
    if isinstance(expected, int):
        return (
            not isinstance(actual, bool) and isinstance(actual, (int, float)) and actual == expected
        )
    return actual == expected


def _validate_config_contract(
    config: Any,
    contract: tuple[tuple[tuple[str, ...], object], ...],
    *,
    prefix: str,
) -> None:
    for path, expected in contract:
        actual = _config_value(config, path, prefix=prefix)
        if not _config_values_match(actual, expected):
            field = ".".join((prefix, *path)) if prefix else ".".join(path)
            raise RuntimeError(
                "SAM3 tracker TensorRT builder supports only the official architecture; "
                f"expected {field}={expected!r}, got {actual!r}"
            )


def _validate_tracker_config(config: Any) -> None:
    """Reject tracker configs that do not match the directly rebuilt graph."""

    try:
        feature_sizes = tuple(
            tuple(int(value) for value in size)
            for size in _config_value(
                config,
                ("vision_config", "backbone_feature_sizes"),
                prefix="tracker_config",
            )
        )
        _validate_tracker_values(
            image_size=int(_config_value(config, ("image_size",), prefix="tracker_config")),
            feature_sizes=feature_sizes,
            num_maskmem=int(_config_value(config, ("num_maskmem",), prefix="tracker_config")),
            max_cond_frames=int(
                _config_value(config, ("max_cond_frame_num",), prefix="tracker_config")
            ),
            max_pointers=int(
                _config_value(
                    config,
                    ("max_object_pointers_in_encoder",),
                    prefix="tracker_config",
                )
            ),
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("SAM3 tracker configuration has an incomplete shape contract") from error
    _validate_config_contract(
        config,
        _TRACKER_ARCHITECTURE_CONTRACT,
        prefix="tracker_config",
    )


def _validate_tracker_values(
    *,
    image_size: int,
    feature_sizes: tuple[tuple[int, ...], ...],
    num_maskmem: int,
    max_cond_frames: int,
    max_pointers: int,
) -> None:
    expected_sizes = ((288, 288), (144, 144), (72, 72))
    if image_size != 1008 or feature_sizes != expected_sizes:
        raise RuntimeError(
            "SAM3 tracker TensorRT builder supports the official 1008px tracker only; "
            f"got image_size={image_size}, backbone_feature_sizes={feature_sizes}"
        )
    if num_maskmem != _NUM_MASK_MEMORY_FRAMES:
        raise RuntimeError(
            "SAM3 tracker memory profile must match the checkpoint: expected "
            f"{_NUM_MASK_MEMORY_FRAMES}, got {num_maskmem}"
        )
    if max_cond_frames != _MAX_CONDITIONING_FRAMES:
        raise RuntimeError(
            "SAM3 tracker conditioning-memory profile must match the checkpoint: "
            f"expected {_MAX_CONDITIONING_FRAMES}, got {max_cond_frames}"
        )
    if max_pointers != _POINTER_RECENT_WINDOW:
        raise RuntimeError(
            "SAM3 tracker pointer profile must match the checkpoint: expected "
            f"{_POINTER_RECENT_WINDOW}, got {max_pointers}"
        )


def _read_model_config(model_dir: str) -> dict[str, Any]:
    config_path = Path(model_dir) / "config.json"
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read SAM3 configuration from {config_path}") from error
    tracker = raw.get("tracker_config")
    if not isinstance(tracker, dict):
        raise RuntimeError("SAM3 video tracker configuration is missing tracker_config")
    _validate_tracker_config(tracker)
    _validate_config_contract(
        raw,
        _ROOT_TRACKER_ARCHITECTURE_CONTRACT,
        prefix="",
    )
    return raw


def _validate_video_policy(model_dir: str) -> None:
    config_path = Path(model_dir) / "config.json"
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read SAM3 video policy from {config_path}") from error
    cadence = int(raw.get("recondition_every_nth_frame", SAM3_TRACKER_RECONDITION_CADENCE))
    if cadence != SAM3_TRACKER_RECONDITION_CADENCE:
        raise RuntimeError(
            "SAM3 tracker TensorRT profile supports the reviewed reconditioning "
            f"cadence {SAM3_TRACKER_RECONDITION_CADENCE}, got {cadence}"
        )


def _new_network(*, enable_tf32: bool, verbose: bool):
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(explicit_batch=True, strongly_typed=True)
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = 5
    if hasattr(config, "avg_timing_iterations"):
        config.avg_timing_iterations = 8
    if hasattr(config, "max_aux_streams"):
        config.max_aux_streams = 0
    if hasattr(trt.BuilderFlag, "STRICT_NANS"):
        config.set_flag(trt.BuilderFlag.STRICT_NANS)
    if hasattr(trt.BuilderFlag, "TF32"):
        if enable_tf32:
            config.set_flag(trt.BuilderFlag.TF32)
        else:
            config.clear_flag(trt.BuilderFlag.TF32)
    return trt, builder, network, config


def _input(network, name: str, dtype, shape: tuple[int, ...]):
    tensor = network.add_input(name, dtype, shape)
    if tensor is None:
        raise RuntimeError(f"Could not add SAM3 tracker input {name!r}")
    return tensor


def _reshape(network, tensor, shape: tuple[int, ...]):
    layer = network.add_shuffle(tensor)
    layer.reshape_dims = shape
    return layer.get_output(0)


def _mark(network, tensor, name: str) -> None:
    tensor.name = name
    network.mark_output(tensor)


def _add_step_profile(builder, config, network, *, batch_size: int) -> None:
    profile = builder.create_optimization_profile()
    profile_shapes_for = _step_profile_shapes if batch_size == 1 else _step_batch2_profile_shapes
    dynamic_names: set[str] = set()
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        if -1 not in tuple(int(dim) for dim in tensor.shape):
            continue
        shapes = profile_shapes_for(tensor.name)
        if shapes is None:
            raise RuntimeError(f"Unexpected dynamic SAM3 tracker input {tensor.name!r}")
        if profile.set_shape(tensor.name, *shapes) is False:
            raise RuntimeError(f"Could not set SAM3 tracker profile for {tensor.name!r}")
        dynamic_names.add(tensor.name)
    expected = {
        "memory_features",
        "memory_position",
        "memory_temporal_offsets",
        "object_pointers",
        "object_pointer_temporal_offsets",
    }
    if dynamic_names != expected:
        raise RuntimeError(
            "SAM3 tracker step graph has an unexpected dynamic-input set: "
            f"expected {sorted(expected)}, got {sorted(dynamic_names)}"
        )
    if config.add_optimization_profile(profile) < 0:
        raise RuntimeError("Could not add the SAM3 tracker optimization profile")


def _serialize(builder, network, config, *, kind: str, verbose: bool) -> bytes:
    if verbose:
        print(
            f"[trtmc build] Building direct TensorRT SAM3 tracker {kind} plan "
            f"from {network.num_layers} layers ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"TensorRT SAM3 tracker {kind} engine build failed")
    return bytes(plan)


def _build_init(weights: TrackerWeights, *, verbose: bool) -> bytes:
    from .tracker_decoder_builder import add_tracker_init_head

    trt, builder, network, config = _new_network(enable_tf32=False, verbose=verbose)
    feature_0 = _input(network, "tracker_feature_0", trt.float32, _FEATURE_SHAPES[0])
    feature_1 = _input(network, "tracker_feature_1", trt.float32, _FEATURE_SHAPES[1])
    feature_2 = _input(network, "tracker_feature_2", trt.float32, _FEATURE_SHAPES[2])
    detector_mask = _input(network, "detector_mask", trt.float32, _DETECTOR_MASK_SHAPE)
    head = add_tracker_init_head(network, feature_0, feature_1, feature_2, detector_mask, weights)
    pointer = _reshape(network, head.object_pointer, (1, 1, 256))
    score = _reshape(network, head.object_score_logits, (1, 1, 1))
    _mark(network, pointer, "object_pointer")
    _mark(network, score, "object_score_logits")
    return _serialize(builder, network, config, kind="init", verbose=verbose)


def build_sam3_tracker_engines(
    model_dir: str,
    *,
    step_plan_spec: TrackerStepPlanSpec | None = None,
    memory_plan_spec: TrackerMemoryPlanSpec | None = None,
    verbose: bool = False,
) -> dict[str, bytes]:
    """Build tracker init plus fixed step and memory AOTI wrapper plans."""

    from .tracker_memory_ffi_builder import (
        TrackerMemoryPlanSpec,
        build_sam3_tracker_memory_ffi_plans,
    )
    from .tracker_step_ffi_builder import (
        TrackerStepPlanSpec,
        build_sam3_tracker_step_ffi_plans,
    )

    resolved = Path(model_dir)
    if not (resolved / "config.json").is_file():
        raise RuntimeError(f"SAM3 tracker build requires a local HF model directory: {model_dir}")
    if step_plan_spec is None:
        raise RuntimeError(
            "SAM3 tracker build requires exported content-addressed B1/B2 split tracker-step "
            "pipelines; the native recurrent fallback is not supported"
        )
    if not isinstance(step_plan_spec, TrackerStepPlanSpec):
        raise TypeError("step_plan_spec must be a TrackerStepPlanSpec")
    if memory_plan_spec is None:
        raise RuntimeError(
            "SAM3 tracker build requires exported content-addressed soft/hard B1/B2 "
            "tracker-memory packages; the direct TensorRT memory fallback is not supported"
        )
    if not isinstance(memory_plan_spec, TrackerMemoryPlanSpec):
        raise TypeError("memory_plan_spec must be a TrackerMemoryPlanSpec")
    _read_model_config(str(resolved))
    _validate_video_policy(str(resolved))
    if verbose:
        print("[trtmc build] Loading SAM3 tracker safetensors ...", file=sys.stderr)
    weights = load_tracker_weights(resolved)
    step_plans = build_sam3_tracker_step_ffi_plans(step_plan_spec, verbose=verbose)
    memory_plans = build_sam3_tracker_memory_ffi_plans(memory_plan_spec, verbose=verbose)
    plans = {
        TRACKER_INIT_SECTION: _build_init(weights, verbose=verbose),
    }
    plans.update(step_plans)
    plans.update(memory_plans)
    return plans
