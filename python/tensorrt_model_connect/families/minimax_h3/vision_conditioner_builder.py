# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned Qwen3-VL vision conditioner for MiniMax-H3 FL2VA.

The public H3 FL2VA checkpoint stores the complete Qwen3-VL vision tower at
``text_encoder/model.visual``.  The Hugging Face image processor emits
normalized, merge-group-ordered flattened patches rather than a CHW image:

``pixel_values[num_patches, channels * temporal_patch * patch_h * patch_w]``.

For H3's fixed 768x1344 keyframe this is ``[4032, 1536]``.  A still image is
duplicated across the two-frame temporal patch by the processor; the builder
therefore consumes that temporal dimension exactly once and must not duplicate
it again.

The graph follows the official Qwen3-VL vision forward pass:

* flattened Conv3d patch projection;
* align-corners bilinear learned-position interpolation in processor merge
  order, plus 2-D rotary positions in that same order;
* 27 full-attention LayerNorm/GELU blocks;
* the main pre-shuffle-normalized patch merger; and
* all three post-shuffle-normalized DeepStack mergers, captured after blocks
  8, 16, and 24.

No implementation from another model family is imported.  TensorRT is loaded
lazily so the pure contract and interpolation helpers remain unit-testable on
hosts without TensorRT.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math
import sys
from typing import Any, Mapping

import ml_dtypes
import numpy as np


VISION_CONDITIONER_DEFAULT_WORKSPACE_BYTES = 96 << 30
_VISUAL_PREFIX = "model.visual"
_WORKFLOWS = ("fl2va", "ref2va")
_REF2VA_MIN_PATCHES = 48 * 48
_REF2VA_OPT_PATCHES = 48 * 84
_REF2VA_MAX_PATCHES = 65536


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"MiniMax-H3 {label} must be a mapping")
    return value


def _require_exact_field(config: Mapping[str, Any], name: str, expected: object) -> None:
    if name not in config:
        raise ValueError(f"MiniMax-H3 vision config is missing {name!r}")
    actual = config[name]
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"MiniMax-H3 vision config {name!r} must be {expected!r}, got {actual!r}")


@dataclass(frozen=True)
class MiniMaxH3VisionConditionerSpec:
    """Static processor and Qwen3-VL graph ABI for one keyframe."""

    workflow: str = "fl2va"
    image_height: int = 768
    image_width: int = 1344
    grid_t: int = 1
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    depth: int = 27
    in_channels: int = 3
    temporal_patch_size: int = 2
    patch_size: int = 16
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304
    out_hidden_size: int = 5120
    deepstack_visual_indexes: tuple[int, ...] = (8, 16, 24)
    layer_norm_eps: float = 1.0e-6
    rope_theta: float = 10000.0
    hidden_act: str = "gelu_pytorch_tanh"
    min_patches: int = 48 * 84
    opt_patches: int = 48 * 84
    max_patches: int = 48 * 84

    def __post_init__(self) -> None:
        integer_fields = (
            "image_height",
            "image_width",
            "grid_t",
            "hidden_size",
            "intermediate_size",
            "num_heads",
            "depth",
            "in_channels",
            "temporal_patch_size",
            "patch_size",
            "spatial_merge_size",
            "num_position_embeddings",
            "out_hidden_size",
            "min_patches",
            "opt_patches",
            "max_patches",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"MiniMax-H3 vision {name} must be a positive integer")
        if self.workflow not in _WORKFLOWS:
            raise ValueError(
                f"MiniMax-H3 vision workflow must be one of {_WORKFLOWS}, got {self.workflow!r}"
            )
        if not self.min_patches <= self.opt_patches <= self.max_patches:
            raise ValueError("MiniMax-H3 vision patches must satisfy min <= opt <= max")
        if any(
            patches % (self.spatial_merge_size**2)
            for patches in (self.min_patches, self.opt_patches, self.max_patches)
        ):
            raise ValueError("MiniMax-H3 vision patch profiles must contain complete merge groups")
        if self.grid_t != 1:
            raise ValueError("MiniMax-H3 FL2VA keyframe vision requires grid_t=1")
        if self.image_height % self.patch_size or self.image_width % self.patch_size:
            raise ValueError(
                "MiniMax-H3 keyframe dimensions must be divisible by vision patch_size"
            )
        if self.grid_h % self.spatial_merge_size or self.grid_w % self.spatial_merge_size:
            raise ValueError("MiniMax-H3 vision patch grid must be divisible by spatial_merge_size")
        if self.hidden_size % self.num_heads:
            raise ValueError("MiniMax-H3 vision hidden_size must be divisible by num_heads")
        if self.head_dim % 4:
            raise ValueError("MiniMax-H3 vision head_dim must be divisible by four for 2-D RoPE")
        position_side = math.isqrt(self.num_position_embeddings)
        if position_side * position_side != self.num_position_embeddings:
            raise ValueError("MiniMax-H3 vision num_position_embeddings must form a square grid")
        indexes = self.deepstack_visual_indexes
        if (
            not isinstance(indexes, tuple)
            or any(not isinstance(index, int) or isinstance(index, bool) for index in indexes)
            or tuple(sorted(set(indexes))) != indexes
            or any(index < 0 or index >= self.depth for index in indexes)
        ):
            raise ValueError(
                "MiniMax-H3 deepstack_visual_indexes must be unique, increasing block indexes"
            )
        if self.hidden_act != "gelu_pytorch_tanh":
            raise ValueError("MiniMax-H3 vision blocks require gelu_pytorch_tanh")
        if not math.isclose(self.layer_norm_eps, 1.0e-6, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("MiniMax-H3 Qwen3-VL LayerNorm epsilon must be 1e-6")
        if not math.isclose(self.rope_theta, 10000.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("MiniMax-H3 Qwen3-VL vision RoPE theta must be 10000")

    @property
    def grid_h(self) -> int:
        return self.image_height // self.patch_size

    @property
    def grid_w(self) -> int:
        return self.image_width // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_t * self.grid_h * self.grid_w

    @property
    def merge_unit(self) -> int:
        return self.spatial_merge_size**2

    @property
    def num_merged_tokens(self) -> int:
        return self.num_patches // self.merge_unit

    @property
    def patch_vector_size(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size

    @property
    def merged_hidden_size(self) -> int:
        return self.hidden_size * self.merge_unit

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def position_grid_side(self) -> int:
        return math.isqrt(self.num_position_embeddings)

    @property
    def pixel_values_shape(self) -> tuple[int, int]:
        return (self.num_patches, self.patch_vector_size)

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self.num_merged_tokens, self.out_hidden_size)

    @classmethod
    def for_workflow(cls, workflow: str, **overrides: Any) -> "MiniMaxH3VisionConditionerSpec":
        if workflow not in _WORKFLOWS:
            raise ValueError(
                f"MiniMax-H3 vision workflow must be one of {_WORKFLOWS}, got {workflow!r}"
            )
        profile = {"workflow": workflow}
        if workflow == "ref2va":
            profile.update(
                min_patches=_REF2VA_MIN_PATCHES,
                opt_patches=_REF2VA_OPT_PATCHES,
                max_patches=_REF2VA_MAX_PATCHES,
            )
        profile.update(overrides)
        return cls(**profile)

    @classmethod
    def from_checkpoint_config(
        cls,
        checkpoint_config: Mapping[str, Any],
        *,
        workflow: str = "fl2va",
    ) -> "MiniMaxH3VisionConditionerSpec":
        """Validate the exact public H3 Qwen3-VL config and return its ABI."""

        root = _require_mapping(checkpoint_config, "text-encoder config")
        if root.get("model_type") != "qwen3_vl":
            raise ValueError("MiniMax-H3 text encoder must have model_type='qwen3_vl'")
        if root.get("architectures") != ["Qwen3VLForConditionalGeneration"]:
            raise ValueError("MiniMax-H3 text encoder must use Qwen3VLForConditionalGeneration")
        text_config = _require_mapping(root.get("text_config"), "text_config")
        if text_config.get("hidden_size") != 5120:
            raise ValueError("MiniMax-H3 Qwen3-VL text hidden_size must be 5120")
        if text_config.get("dtype") != "bfloat16":
            raise ValueError("MiniMax-H3 Qwen3-VL checkpoint dtype must be bfloat16")

        vision = _require_mapping(root.get("vision_config"), "vision_config")
        expected = {
            "model_type": "qwen3_vl",
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_heads": 16,
            "depth": 27,
            "in_channels": 3,
            "temporal_patch_size": 2,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "num_position_embeddings": 2304,
            "out_hidden_size": 5120,
            "deepstack_visual_indexes": [8, 16, 24],
            "hidden_act": "gelu_pytorch_tanh",
        }
        for name, value in expected.items():
            _require_exact_field(vision, name, value)
        return cls.for_workflow(
            workflow,
            hidden_size=vision["hidden_size"],
            intermediate_size=vision["intermediate_size"],
            num_heads=vision["num_heads"],
            depth=vision["depth"],
            in_channels=vision["in_channels"],
            temporal_patch_size=vision["temporal_patch_size"],
            patch_size=vision["patch_size"],
            spatial_merge_size=vision["spatial_merge_size"],
            num_position_embeddings=vision["num_position_embeddings"],
            out_hidden_size=vision["out_hidden_size"],
            deepstack_visual_indexes=tuple(vision["deepstack_visual_indexes"]),
            hidden_act=vision["hidden_act"],
        )


def expected_weight_shapes(
    spec: MiniMaxH3VisionConditionerSpec | None = None,
) -> dict[str, tuple[int, ...]]:
    """Return the complete, exact ``model.visual`` checkpoint contract."""

    spec = spec or MiniMaxH3VisionConditionerSpec()
    hidden = spec.hidden_size
    intermediate = spec.intermediate_size
    merged = spec.merged_hidden_size
    shapes: dict[str, tuple[int, ...]] = {
        f"{_VISUAL_PREFIX}.patch_embed.proj.weight": (
            hidden,
            spec.in_channels,
            spec.temporal_patch_size,
            spec.patch_size,
            spec.patch_size,
        ),
        f"{_VISUAL_PREFIX}.patch_embed.proj.bias": (hidden,),
        f"{_VISUAL_PREFIX}.pos_embed.weight": (
            spec.num_position_embeddings,
            hidden,
        ),
    }
    for index in range(spec.depth):
        prefix = f"{_VISUAL_PREFIX}.blocks.{index}"
        shapes.update(
            {
                f"{prefix}.attn.qkv.weight": (3 * hidden, hidden),
                f"{prefix}.attn.qkv.bias": (3 * hidden,),
                f"{prefix}.attn.proj.weight": (hidden, hidden),
                f"{prefix}.attn.proj.bias": (hidden,),
                f"{prefix}.mlp.linear_fc1.weight": (intermediate, hidden),
                f"{prefix}.mlp.linear_fc1.bias": (intermediate,),
                f"{prefix}.mlp.linear_fc2.weight": (hidden, intermediate),
                f"{prefix}.mlp.linear_fc2.bias": (hidden,),
                f"{prefix}.norm1.weight": (hidden,),
                f"{prefix}.norm1.bias": (hidden,),
                f"{prefix}.norm2.weight": (hidden,),
                f"{prefix}.norm2.bias": (hidden,),
            }
        )

    def add_merger(prefix: str, *, postshuffle_norm: bool) -> None:
        norm_width = merged if postshuffle_norm else hidden
        shapes.update(
            {
                f"{prefix}.norm.weight": (norm_width,),
                f"{prefix}.norm.bias": (norm_width,),
                f"{prefix}.linear_fc1.weight": (merged, merged),
                f"{prefix}.linear_fc1.bias": (merged,),
                f"{prefix}.linear_fc2.weight": (spec.out_hidden_size, merged),
                f"{prefix}.linear_fc2.bias": (spec.out_hidden_size,),
            }
        )

    add_merger(f"{_VISUAL_PREFIX}.merger", postshuffle_norm=False)
    for index in range(len(spec.deepstack_visual_indexes)):
        add_merger(
            f"{_VISUAL_PREFIX}.deepstack_merger_list.{index}",
            postshuffle_norm=True,
        )
    return shapes


def checkpoint_keys() -> tuple[str, ...]:
    """Return every H3 ``model.visual`` tensor needed by this engine."""

    return tuple(expected_weight_shapes())


def validate_vision_weights(
    weights: Mapping[str, Any], spec: MiniMaxH3VisionConditionerSpec
) -> None:
    """Fail closed on missing, extra, malformed, or lossy vision tensors."""

    if not isinstance(weights, Mapping):
        raise ValueError("MiniMax-H3 vision weights must be a mapping")
    expected = expected_weight_shapes(spec)
    actual_visual = {name for name in weights if name.startswith(f"{_VISUAL_PREFIX}.")}
    missing = sorted(set(expected) - set(weights))
    unexpected = sorted(actual_visual - set(expected))
    if missing or unexpected:
        raise ValueError(
            "MiniMax-H3 model.visual checkpoint contract mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    allowed_dtypes = {np.dtype(np.float32), np.dtype(ml_dtypes.bfloat16)}
    for name, shape in expected.items():
        value = weights[name]
        actual_shape = tuple(int(dimension) for dimension in getattr(value, "shape", ()))
        if actual_shape != shape:
            raise ValueError(
                f"MiniMax-H3 vision tensor {name!r} must have shape {shape}, got {actual_shape}"
            )
        try:
            dtype = np.dtype(value.dtype)
        except (AttributeError, TypeError) as error:
            raise ValueError(
                f"MiniMax-H3 vision tensor {name!r} must expose a NumPy dtype"
            ) from error
        if dtype not in allowed_dtypes:
            raise ValueError(f"MiniMax-H3 vision tensor {name!r} must be BF16 or FP32, got {dtype}")


def _merge_group_coordinates(grid_h: int, grid_w: int, merge: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode one processor block's patch sequence into row/column IDs."""

    within = np.arange(grid_h * grid_w, dtype=np.int64)
    blocks_w = grid_w // merge
    in_col = within % merge
    in_row = (within // merge) % merge
    block_col = (within // (merge * merge)) % blocks_w
    block_row = within // (merge * merge * blocks_w)
    return block_row * merge + in_row, block_col * merge + in_col


def _processor_merge_group_coordinates(
    spec: MiniMaxH3VisionConditionerSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode the fixed FL2VA processor sequence into row/column IDs."""

    return _merge_group_coordinates(spec.grid_h, spec.grid_w, spec.spatial_merge_size)


def _interpolation_axis_taps_weights(
    indexes: np.ndarray, size: int, source_side: int
) -> tuple[np.ndarray, np.ndarray]:
    if size == 1:
        source = np.zeros_like(indexes, dtype=np.float32)
    else:
        # Pinned Qwen3-VL multiplies each FP32 index by `(side - 1)` first,
        # then divides by `(size - 1)`. Both operations publish FP32, and the
        # alternative division-first form differs by one ULP on H3 grids.
        source = (indexes.astype(np.float32) * np.float32(source_side - 1)) / np.float32(size - 1)
    floor = np.floor(source).astype(np.int64)
    taps = np.stack((floor, floor + 1), axis=1)
    taps = np.clip(taps, 0, source_side - 1)
    offsets = np.asarray((0.0, 1.0), dtype=np.float32)
    distance = np.abs(source[:, None] - floor[:, None].astype(np.float32) - offsets)
    return taps, np.maximum(np.float32(1.0) - distance, np.float32(0.0))


def _position_interpolation_indices_weights(
    spec: MiniMaxH3VisionConditionerSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Port HF align-corners bilinear position lookup for the fixed H3 grid."""

    return _position_interpolation_for_grid(spec.grid_h, spec.grid_w, spec)


def _position_interpolation_for_grid(
    grid_h: int,
    grid_w: int,
    spec: MiniMaxH3VisionConditionerSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Port HF learned-position interpolation for one runtime patch grid."""

    rows, columns = _merge_group_coordinates(grid_h, grid_w, spec.spatial_merge_size)
    side = spec.position_grid_side
    h_taps, h_weights = _interpolation_axis_taps_weights(rows, grid_h, side)
    w_taps, w_weights = _interpolation_axis_taps_weights(columns, grid_w, side)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 4)
    return np.ascontiguousarray(indices), np.ascontiguousarray(weights, dtype=np.float32)


def make_ref2va_position_bindings(
    grid_h: int,
    grid_w: int,
    spec: MiniMaxH3VisionConditionerSpec | None = None,
) -> dict[str, np.ndarray]:
    """Materialize the exact runtime position inputs for one Ref2VA block."""

    spec = spec or MiniMaxH3VisionConditionerSpec.for_workflow("ref2va")
    if spec.workflow != "ref2va":
        raise ValueError("Ref2VA position bindings require workflow='ref2va'")
    if (
        not isinstance(grid_h, int)
        or isinstance(grid_h, bool)
        or not isinstance(grid_w, int)
        or isinstance(grid_w, bool)
        or grid_h <= 0
        or grid_w <= 0
        or grid_h % spec.spatial_merge_size
        or grid_w % spec.spatial_merge_size
    ):
        raise ValueError("MiniMax-H3 Ref2VA patch grids must be positive merge-aligned integers")
    patches = grid_h * grid_w
    if not spec.min_patches <= patches <= spec.max_patches:
        raise ValueError(
            "MiniMax-H3 Ref2VA raw patch rows are outside the dynamic profile: "
            f"rows={patches}, range=[{spec.min_patches}, {spec.max_patches}]"
        )
    indices, weights = _position_interpolation_for_grid(grid_h, grid_w, spec)
    rows, columns = _merge_group_coordinates(grid_h, grid_w, spec.spatial_merge_size)
    return {
        "position_indices": np.ascontiguousarray(indices, dtype=np.int32),
        "position_weights": np.ascontiguousarray(weights, dtype=np.float32),
        "vision_position_ids": np.ascontiguousarray(
            np.stack((rows, columns), axis=1), dtype=np.int32
        ),
    }


def _is_legal_video_grid(grid_h: int, grid_w: int) -> bool:
    """Whether a grid can result from H3's area cap and multiple-32 rounding."""

    short, long = sorted((grid_h, grid_w))
    if short % 2 or long % 2 or not 0.25 <= grid_w / grid_h <= 4.0:
        return False
    # Before the 768x1344 area cap engages, the short edge remains exactly
    # 768 pixels (48 Qwen patches) and the long edge spans 48..84 patches.
    if short == 48 and 48 <= long <= 84:
        return True
    # With the cap active the ideal patch-grid area is 4032. A rounded grid is
    # legal when some aspect ratio in [1.75, 4] lands in both dimensions'
    # half-open nearest-even rounding intervals.
    area = 48 * 84
    lower = max(
        1.75,
        ((long - 1) ** 2) / area,
        area / ((short + 1) ** 2),
    )
    upper = min(
        4.0,
        ((long + 1) ** 2) / area,
        area / ((short - 1) ** 2),
    )
    return lower <= upper


def validate_ref2va_vision_bindings(
    *,
    modality: str,
    grid_h: int,
    grid_w: int,
    pixel_values: np.ndarray,
    position_indices: np.ndarray,
    position_weights: np.ndarray,
    vision_position_ids: np.ndarray,
    spec: MiniMaxH3VisionConditionerSpec | None = None,
) -> int:
    """Fail closed on one prepared Ref2VA image or temporal-pair block."""

    spec = spec or MiniMaxH3VisionConditionerSpec.for_workflow("ref2va")
    if (
        not isinstance(grid_h, int)
        or isinstance(grid_h, bool)
        or not isinstance(grid_w, int)
        or isinstance(grid_w, bool)
        or grid_h <= 0
        or grid_w <= 0
        or grid_h % spec.spatial_merge_size
        or grid_w % spec.spatial_merge_size
    ):
        raise ValueError("MiniMax-H3 Ref2VA patch grids must be positive merge-aligned integers")
    patches = grid_h * grid_w
    if not spec.min_patches <= patches <= spec.max_patches:
        raise ValueError(
            "MiniMax-H3 Ref2VA raw patch rows are outside the dynamic profile: "
            f"rows={patches}, range=[{spec.min_patches}, {spec.max_patches}]"
        )
    ratio = grid_w / grid_h
    if modality == "image":
        if min(grid_h, grid_w) != 128 or not 0.25 <= ratio <= 4.0 or patches > 65536:
            raise ValueError(
                "MiniMax-H3 Ref2VA images require a 2048-pixel short edge and 1:4..4:1 aspect"
            )
    elif modality == "video":
        if not _is_legal_video_grid(grid_h, grid_w) or patches > 4176:
            raise ValueError("MiniMax-H3 Ref2VA video pairs require a legal rounded 768p canvas")
    else:
        raise ValueError("MiniMax-H3 Ref2VA vision modality must be 'image' or 'video'")
    expected = make_ref2va_position_bindings(grid_h, grid_w, spec)

    if not isinstance(pixel_values, np.ndarray):
        raise ValueError("MiniMax-H3 Ref2VA pixel_values must be a NumPy array")
    allowed_pixel_dtypes = {np.dtype(np.float32), np.dtype(ml_dtypes.bfloat16)}
    if tuple(pixel_values.shape) != (patches, spec.patch_vector_size):
        raise ValueError("MiniMax-H3 Ref2VA pixel_values shape does not match its runtime grid")
    if pixel_values.dtype not in allowed_pixel_dtypes or not pixel_values.flags.c_contiguous:
        raise ValueError("MiniMax-H3 Ref2VA pixel_values must be contiguous FP32 or BF16")
    if not np.all(np.isfinite(pixel_values)):
        raise ValueError("MiniMax-H3 Ref2VA pixel_values must be finite")
    supplied = {
        "position_indices": position_indices,
        "position_weights": position_weights,
        "vision_position_ids": vision_position_ids,
    }
    for name, reference in expected.items():
        value = supplied[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != reference.dtype
            or tuple(value.shape) != tuple(reference.shape)
            or not value.flags.c_contiguous
            or not np.array_equal(value, reference)
        ):
            raise ValueError(f"MiniMax-H3 Ref2VA {name} does not match the runtime patch grid")
    return patches // spec.merge_unit


def _round_float32_to_bf16(values: np.ndarray) -> np.ndarray:
    """Round FP32 to BF16, retaining FP32 storage for NumPy arithmetic."""

    values = np.ascontiguousarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def _interpolated_position_embeddings(
    position_embeddings: np.ndarray,
    spec: MiniMaxH3VisionConditionerSpec,
) -> np.ndarray:
    """Match HF's BF16 table, FP32 interpolation, then one BF16 publication."""

    table = np.asarray(position_embeddings)
    expected = (spec.num_position_embeddings, spec.hidden_size)
    if tuple(table.shape) != expected:
        raise ValueError(
            f"MiniMax-H3 learned position table must have shape {expected}, got {table.shape}"
        )
    indices, weights = _position_interpolation_indices_weights(spec)
    table_bf16 = _round_float32_to_bf16(table)
    result = table_bf16[indices[:, 0]] * weights[:, 0, None]
    for tap in range(1, 4):
        result = result + table_bf16[indices[:, tap]] * weights[:, tap, None]
    return np.ascontiguousarray(result.astype(ml_dtypes.bfloat16))


def _vision_rope_cos_sin(
    spec: MiniMaxH3VisionConditionerSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Qwen3-VL's half-head cos/sin tables in processor patch order."""

    rows, columns = _processor_merge_group_coordinates(spec)
    rotary_dim = spec.head_dim // 2
    inverse_frequency = np.float32(1.0) / np.power(
        np.float32(spec.rope_theta),
        np.arange(0, rotary_dim, 2, dtype=np.float32) / np.float32(rotary_dim),
    )
    rotary = np.concatenate(
        (
            rows.astype(np.float32)[:, None] * inverse_frequency[None, :],
            columns.astype(np.float32)[:, None] * inverse_frequency[None, :],
        ),
        axis=1,
    )
    return (
        np.ascontiguousarray(np.cos(rotary).astype(np.float32)),
        np.ascontiguousarray(np.sin(rotary).astype(np.float32)),
    )


def _layer_norm(network, tensor, weight, bias, width: int, eps: float, trt, op):
    source_dtype = tensor.dtype
    value = op.cast(network, tensor, trt.float32)
    axis = 1 << (len(tuple(value.shape)) - 1)
    mean = network.add_reduce(value, trt.ReduceOperation.AVG, axis, True).get_output(0)
    centered = network.add_elementwise(value, mean, trt.ElementWiseOperation.SUB).get_output(0)
    square = network.add_elementwise(centered, centered, trt.ElementWiseOperation.PROD).get_output(
        0
    )
    variance = network.add_reduce(square, trt.ReduceOperation.AVG, axis, True).get_output(0)
    epsilon = op.constant(network, np.full((1,) * len(tuple(value.shape)), eps, dtype=np.float32))
    denominator = network.add_elementwise(
        variance, epsilon, trt.ElementWiseOperation.SUM
    ).get_output(0)
    denominator = network.add_unary(denominator, trt.UnaryOperation.SQRT).get_output(0)
    inverse = network.add_unary(denominator, trt.UnaryOperation.RECIP).get_output(0)
    normalized = network.add_elementwise(
        centered, inverse, trt.ElementWiseOperation.PROD
    ).get_output(0)
    gamma = op.weight_constant(network, np.asarray(weight).reshape(1, width))
    beta = op.weight_constant(network, np.asarray(bias).reshape(1, width))
    gamma = op.cast(network, gamma, trt.float32)
    beta = op.cast(network, beta, trt.float32)
    scaled = network.add_elementwise(normalized, gamma, trt.ElementWiseOperation.PROD).get_output(0)
    shifted = network.add_elementwise(scaled, beta, trt.ElementWiseOperation.SUM).get_output(0)
    return op.cast(network, shifted, source_dtype)


def _gelu_tanh(network, tensor, trt, op):
    """PyTorch ``gelu_pytorch_tanh`` with an explicit FP32 activation island."""

    source_dtype = tensor.dtype
    value = op.cast(network, tensor, trt.float32)
    scalar_shape = (1,) * len(tuple(value.shape))

    def constant(number: float):
        return op.constant(network, np.full(scalar_shape, number, dtype=np.float32))

    square = network.add_elementwise(value, value, trt.ElementWiseOperation.PROD).get_output(0)
    cube = network.add_elementwise(square, value, trt.ElementWiseOperation.PROD).get_output(0)
    cube = network.add_elementwise(
        cube, constant(0.044715), trt.ElementWiseOperation.PROD
    ).get_output(0)
    inner = network.add_elementwise(value, cube, trt.ElementWiseOperation.SUM).get_output(0)
    inner = network.add_elementwise(
        inner,
        constant(math.sqrt(2.0 / math.pi)),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    activated = network.add_activation(inner, trt.ActivationType.TANH).get_output(0)
    activated = network.add_elementwise(
        activated, constant(1.0), trt.ElementWiseOperation.SUM
    ).get_output(0)
    activated = network.add_elementwise(activated, value, trt.ElementWiseOperation.PROD).get_output(
        0
    )
    activated = network.add_elementwise(
        activated, constant(0.5), trt.ElementWiseOperation.PROD
    ).get_output(0)
    return op.cast(network, activated, source_dtype)


def _gelu_exact(network, tensor, trt, op):
    """Default ``torch.nn.GELU`` used by both Qwen3-VL patch mergers."""

    source_dtype = tensor.dtype
    value = op.cast(network, tensor, trt.float32)
    scalar_shape = (1,) * len(tuple(value.shape))
    inverse_sqrt_two = op.constant(
        network,
        np.full(scalar_shape, 1.0 / math.sqrt(2.0), dtype=np.float32),
    )
    scaled = network.add_elementwise(
        value, inverse_sqrt_two, trt.ElementWiseOperation.PROD
    ).get_output(0)
    erf = network.add_unary(scaled, trt.UnaryOperation.ERF).get_output(0)
    one = op.constant(network, np.ones(scalar_shape, dtype=np.float32))
    factor = network.add_elementwise(erf, one, trt.ElementWiseOperation.SUM).get_output(0)
    half = op.constant(network, np.full(scalar_shape, 0.5, dtype=np.float32))
    value = network.add_elementwise(value, half, trt.ElementWiseOperation.PROD).get_output(0)
    result = network.add_elementwise(value, factor, trt.ElementWiseOperation.PROD).get_output(0)
    return op.cast(network, result, source_dtype)


def _apply_vision_rope(network, tensor, cos_half, sin_half, spec, trt, op):
    """Apply Qwen3-VL rotate-half RoPE in FP32, then publish BF16."""

    rows = spec.num_patches
    heads = spec.num_heads
    head_dim = spec.head_dim
    source_dtype = tensor.dtype
    value = op.rows_to_heads(network, tensor, rows, heads, head_dim)
    value = op.cast(network, value, trt.float32)
    half = head_dim // 2
    first = network.add_slice(value, (0, 0, 0, 0), (1, heads, rows, half), (1, 1, 1, 1))
    second = network.add_slice(value, (0, 0, 0, half), (1, heads, rows, half), (1, 1, 1, 1))
    negative_second = network.add_unary(second.get_output(0), trt.UnaryOperation.NEG).get_output(0)
    rotated = network.add_concatenation((negative_second, first.get_output(0)))
    rotated.axis = 3

    def duplicated_table(table):
        constant = op.weight_constant(network, table)
        constant = op.cast(network, constant, trt.float32)
        reshape = network.add_shuffle(constant)
        reshape.reshape_dims = (1, 1, rows, half)
        duplicate = network.add_concatenation((reshape.get_output(0), reshape.get_output(0)))
        duplicate.axis = 3
        return duplicate.get_output(0)

    cosine = duplicated_table(cos_half)
    sine = duplicated_table(sin_half)
    left = network.add_elementwise(value, cosine, trt.ElementWiseOperation.PROD).get_output(0)
    right = network.add_elementwise(
        rotated.get_output(0), sine, trt.ElementWiseOperation.PROD
    ).get_output(0)
    result = network.add_elementwise(left, right, trt.ElementWiseOperation.SUM).get_output(0)
    result = op.cast(network, result, source_dtype)
    return op.heads_to_rows(network, result, rows, spec.hidden_size)


def _patch_embedding(network, pixel_values, weights, spec, trt, op):
    prefix = f"{_VISUAL_PREFIX}.patch_embed.proj"
    projection = np.asarray(weights[f"{prefix}.weight"]).reshape(
        spec.hidden_size, spec.patch_vector_size
    )
    return op.linear(
        network,
        pixel_values,
        projection,
        weights[f"{prefix}.bias"],
        compute_dtype=trt.bfloat16,
    )


def _vision_attention(network, hidden, weights, prefix, cos_half, sin_half, spec, trt, op):
    qkv = op.linear(
        network,
        hidden,
        weights[f"{prefix}.attn.qkv.weight"],
        weights[f"{prefix}.attn.qkv.bias"],
        compute_dtype=trt.bfloat16,
    )
    rows = spec.num_patches
    hidden_size = spec.hidden_size
    stride = (1, 1)
    q = network.add_slice(qkv, (0, 0), (rows, hidden_size), stride).get_output(0)
    k = network.add_slice(qkv, (0, hidden_size), (rows, hidden_size), stride).get_output(0)
    v = network.add_slice(qkv, (0, 2 * hidden_size), (rows, hidden_size), stride).get_output(0)
    q = _apply_vision_rope(network, q, cos_half, sin_half, spec, trt, op)
    k = _apply_vision_rope(network, k, cos_half, sin_half, spec, trt, op)
    q = op.rows_to_heads(network, q, rows, spec.num_heads, spec.head_dim)
    k = op.rows_to_heads(network, k, rows, spec.num_heads, spec.head_dim)
    v = op.rows_to_heads(network, v, rows, spec.num_heads, spec.head_dim)
    scale = op.constant(
        network,
        np.full((1, 1, 1, 1), 1.0 / math.sqrt(spec.head_dim), dtype=np.float32),
    )
    scale = op.cast(network, scale, q.dtype)
    q = network.add_elementwise(q, scale, trt.ElementWiseOperation.PROD).get_output(0)
    attention = network.add_attention(q, k, v, trt.AttentionNormalizationOp.SOFTMAX, False)
    if attention is None:
        raise RuntimeError(f"TensorRT failed to add MiniMax-H3 vision attention {prefix}")
    attention.name = f"{prefix}.attn.native_attention"
    attention.metadata = f"trtmc.native_op=IAttention;source={attention.name}"
    attention.get_output(0).name = f"{attention.name}.output"
    attention.decomposable = False
    context = op.cast(network, attention.get_output(0), trt.bfloat16)
    context = op.heads_to_rows(network, context, rows, hidden_size)
    return op.linear(
        network,
        context,
        weights[f"{prefix}.attn.proj.weight"],
        weights[f"{prefix}.attn.proj.bias"],
        compute_dtype=trt.bfloat16,
    )


def _vision_block(network, hidden, weights, index, cos_half, sin_half, spec, trt, op):
    prefix = f"{_VISUAL_PREFIX}.blocks.{index}"
    normalized = _layer_norm(
        network,
        hidden,
        weights[f"{prefix}.norm1.weight"],
        weights[f"{prefix}.norm1.bias"],
        spec.hidden_size,
        spec.layer_norm_eps,
        trt,
        op,
    )
    update = _vision_attention(
        network, normalized, weights, prefix, cos_half, sin_half, spec, trt, op
    )
    hidden = network.add_elementwise(hidden, update, trt.ElementWiseOperation.SUM).get_output(0)
    normalized = _layer_norm(
        network,
        hidden,
        weights[f"{prefix}.norm2.weight"],
        weights[f"{prefix}.norm2.bias"],
        spec.hidden_size,
        spec.layer_norm_eps,
        trt,
        op,
    )
    update = op.linear(
        network,
        normalized,
        weights[f"{prefix}.mlp.linear_fc1.weight"],
        weights[f"{prefix}.mlp.linear_fc1.bias"],
        compute_dtype=trt.bfloat16,
    )
    update = _gelu_tanh(network, update, trt, op)
    update = op.linear(
        network,
        update,
        weights[f"{prefix}.mlp.linear_fc2.weight"],
        weights[f"{prefix}.mlp.linear_fc2.bias"],
        compute_dtype=trt.bfloat16,
    )
    return network.add_elementwise(hidden, update, trt.ElementWiseOperation.SUM).get_output(0)


def _patch_merger(network, hidden, weights, prefix, *, postshuffle_norm: bool, spec, trt, op):
    merged_shape = (spec.num_merged_tokens, spec.merged_hidden_size)
    if postshuffle_norm:
        grouped = network.add_shuffle(hidden)
        grouped.reshape_dims = merged_shape
        hidden = _layer_norm(
            network,
            grouped.get_output(0),
            weights[f"{prefix}.norm.weight"],
            weights[f"{prefix}.norm.bias"],
            spec.merged_hidden_size,
            spec.layer_norm_eps,
            trt,
            op,
        )
    else:
        hidden = _layer_norm(
            network,
            hidden,
            weights[f"{prefix}.norm.weight"],
            weights[f"{prefix}.norm.bias"],
            spec.hidden_size,
            spec.layer_norm_eps,
            trt,
            op,
        )
        grouped = network.add_shuffle(hidden)
        grouped.reshape_dims = merged_shape
        hidden = grouped.get_output(0)
    hidden = op.linear(
        network,
        hidden,
        weights[f"{prefix}.linear_fc1.weight"],
        weights[f"{prefix}.linear_fc1.bias"],
        compute_dtype=trt.bfloat16,
    )
    hidden = _gelu_exact(network, hidden, trt, op)
    return op.linear(
        network,
        hidden,
        weights[f"{prefix}.linear_fc2.weight"],
        weights[f"{prefix}.linear_fc2.bias"],
        compute_dtype=trt.bfloat16,
    )


def _rows_to_heads_dynamic(network, tensor, heads: int, head_dim: int, trt):
    reshape = network.add_shuffle(tensor)
    reshape.reshape_dims = (-1, heads, head_dim)
    reshape.second_transpose = trt.Permutation([1, 0, 2])
    batch = network.add_shuffle(reshape.get_output(0))
    batch.reshape_dims = (1, heads, -1, head_dim)
    return batch.get_output(0)


def _heads_to_rows_dynamic(network, tensor, width: int, trt):
    reshape = network.add_shuffle(tensor)
    reshape.first_transpose = trt.Permutation([0, 2, 1, 3])
    reshape.reshape_dims = (-1, width)
    return reshape.get_output(0)


def _dynamic_column_slice(network, tensor, start: int, width: int, op):
    tensor_shape = network.add_shape(tensor).get_output(0)
    row_index = op.constant(network, np.asarray([0], np.int32), dtype=np.int32)
    rows = network.add_gather(tensor_shape, row_index, 0).get_output(0)
    column_width = op.constant(network, np.asarray([width], np.int64), dtype=np.int64)
    shape = network.add_concatenation((rows, column_width))
    shape.axis = 0
    layer = network.add_slice(tensor, (0, start), (1, width), (1, 1))
    layer.set_input(2, shape.get_output(0))
    return layer.get_output(0)


def _runtime_interpolated_positions(network, table, indices, weights, spec, trt, op):
    result = None
    for tap in range(4):
        tap_index = op.constant(network, np.asarray([tap], np.int32), dtype=np.int32)
        position_index = network.add_gather(indices, tap_index, 1).get_output(0)
        position_index = network.add_shuffle(position_index)
        position_index.reshape_dims = (-1,)
        position = network.add_gather(table, position_index.get_output(0), 0).get_output(0)
        position = op.cast(network, position, trt.float32)
        blend = network.add_gather(weights, tap_index, 1).get_output(0)
        term = network.add_elementwise(position, blend, trt.ElementWiseOperation.PROD).get_output(0)
        result = (
            term
            if result is None
            else network.add_elementwise(result, term, trt.ElementWiseOperation.SUM).get_output(0)
        )
    assert result is not None
    return op.cast(network, result, trt.bfloat16)


def _runtime_vision_rope_tables(network, position_ids, spec, trt, op):
    position = op.cast(network, position_ids, trt.float32)
    inverse = np.float32(1.0) / np.power(
        np.float32(spec.rope_theta),
        np.arange(0, spec.head_dim // 2, 2, dtype=np.float32) / np.float32(spec.head_dim // 2),
    )
    inverse = op.constant(network, inverse.reshape(1, -1))
    frequencies = []
    for axis in range(2):
        axis_index = op.constant(network, np.asarray([axis], np.int32), dtype=np.int32)
        coordinate = network.add_gather(position, axis_index, 1).get_output(0)
        frequencies.append(
            network.add_elementwise(coordinate, inverse, trt.ElementWiseOperation.PROD).get_output(
                0
            )
        )
    joined = network.add_concatenation(tuple(frequencies))
    joined.axis = 1
    cosine = network.add_unary(joined.get_output(0), trt.UnaryOperation.COS)
    sine = network.add_unary(joined.get_output(0), trt.UnaryOperation.SIN)
    return cosine.get_output(0), sine.get_output(0)


def _apply_vision_rope_dynamic(network, tensor, cosine, sine, heads, spec, trt, op):
    source_dtype = tensor.dtype
    value = op.cast(network, tensor, trt.float32)
    value = _rows_to_heads_dynamic(network, value, heads, spec.head_dim, trt)

    def shaped(table):
        table = op.cast(network, table, trt.float32)
        reshape = network.add_shuffle(table)
        reshape.reshape_dims = (1, -1, spec.head_dim // 2)
        return reshape.get_output(0)

    rotary = network.add_rotary_embedding(value, shaped(cosine), shaped(sine), False, spec.head_dim)
    if rotary is None:
        raise RuntimeError("TensorRT rejected MiniMax-H3 dynamic vision RoPE")
    result = _heads_to_rows_dynamic(network, rotary.get_output(0), heads * spec.head_dim, trt)
    return op.cast(network, result, source_dtype)


def _vision_attention_dynamic(network, hidden, weights, prefix, cosine, sine, spec, trt, op):
    qkv = op.linear(
        network,
        hidden,
        weights[f"{prefix}.attn.qkv.weight"],
        weights[f"{prefix}.attn.qkv.bias"],
        compute_dtype=trt.bfloat16,
    )
    q = _dynamic_column_slice(network, qkv, 0, spec.hidden_size, op)
    k = _dynamic_column_slice(network, qkv, spec.hidden_size, spec.hidden_size, op)
    v = _dynamic_column_slice(network, qkv, 2 * spec.hidden_size, spec.hidden_size, op)
    q = _apply_vision_rope_dynamic(network, q, cosine, sine, spec.num_heads, spec, trt, op)
    k = _apply_vision_rope_dynamic(network, k, cosine, sine, spec.num_heads, spec, trt, op)
    q = _rows_to_heads_dynamic(network, q, spec.num_heads, spec.head_dim, trt)
    k = _rows_to_heads_dynamic(network, k, spec.num_heads, spec.head_dim, trt)
    v = _rows_to_heads_dynamic(network, v, spec.num_heads, spec.head_dim, trt)
    scale = op.constant(
        network,
        np.full((1, 1, 1, 1), 1.0 / math.sqrt(spec.head_dim), np.float32),
    )
    scale = op.cast(network, scale, q.dtype)
    q = network.add_elementwise(q, scale, trt.ElementWiseOperation.PROD).get_output(0)
    attention = network.add_attention(q, k, v, trt.AttentionNormalizationOp.SOFTMAX, False)
    if attention is None:
        raise RuntimeError(f"TensorRT failed to add dynamic vision attention {prefix}")
    attention.name = f"{prefix}.attn.native_dynamic_attention"
    attention.metadata = f"trtmc.native_op=IAttention;source={attention.name}"
    attention.get_output(0).name = f"{attention.name}.output"
    attention.decomposable = False
    context = _heads_to_rows_dynamic(network, attention.get_output(0), spec.hidden_size, trt)
    return op.linear(
        network,
        context,
        weights[f"{prefix}.attn.proj.weight"],
        weights[f"{prefix}.attn.proj.bias"],
        compute_dtype=trt.bfloat16,
    )


def _vision_block_dynamic(network, hidden, weights, index, cosine, sine, spec, trt, op):
    prefix = f"{_VISUAL_PREFIX}.blocks.{index}"
    normalized = _layer_norm(
        network,
        hidden,
        weights[f"{prefix}.norm1.weight"],
        weights[f"{prefix}.norm1.bias"],
        spec.hidden_size,
        spec.layer_norm_eps,
        trt,
        op,
    )
    update = _vision_attention_dynamic(
        network, normalized, weights, prefix, cosine, sine, spec, trt, op
    )
    hidden = network.add_elementwise(hidden, update, trt.ElementWiseOperation.SUM).get_output(0)
    normalized = _layer_norm(
        network,
        hidden,
        weights[f"{prefix}.norm2.weight"],
        weights[f"{prefix}.norm2.bias"],
        spec.hidden_size,
        spec.layer_norm_eps,
        trt,
        op,
    )
    update = op.linear(
        network,
        normalized,
        weights[f"{prefix}.mlp.linear_fc1.weight"],
        weights[f"{prefix}.mlp.linear_fc1.bias"],
        compute_dtype=trt.bfloat16,
    )
    update = _gelu_tanh(network, update, trt, op)
    update = op.linear(
        network,
        update,
        weights[f"{prefix}.mlp.linear_fc2.weight"],
        weights[f"{prefix}.mlp.linear_fc2.bias"],
        compute_dtype=trt.bfloat16,
    )
    return network.add_elementwise(hidden, update, trt.ElementWiseOperation.SUM).get_output(0)


def _patch_merger_dynamic(network, hidden, weights, prefix, *, postshuffle_norm, spec, trt, op):
    if postshuffle_norm:
        grouped = network.add_shuffle(hidden)
        grouped.reshape_dims = (-1, spec.merged_hidden_size)
        hidden = _layer_norm(
            network,
            grouped.get_output(0),
            weights[f"{prefix}.norm.weight"],
            weights[f"{prefix}.norm.bias"],
            spec.merged_hidden_size,
            spec.layer_norm_eps,
            trt,
            op,
        )
    else:
        hidden = _layer_norm(
            network,
            hidden,
            weights[f"{prefix}.norm.weight"],
            weights[f"{prefix}.norm.bias"],
            spec.hidden_size,
            spec.layer_norm_eps,
            trt,
            op,
        )
        grouped = network.add_shuffle(hidden)
        grouped.reshape_dims = (-1, spec.merged_hidden_size)
        hidden = grouped.get_output(0)
    hidden = op.linear(
        network,
        hidden,
        weights[f"{prefix}.linear_fc1.weight"],
        weights[f"{prefix}.linear_fc1.bias"],
        compute_dtype=trt.bfloat16,
    )
    hidden = _gelu_exact(network, hidden, trt, op)
    return op.linear(
        network,
        hidden,
        weights[f"{prefix}.linear_fc2.weight"],
        weights[f"{prefix}.linear_fc2.bias"],
        compute_dtype=trt.bfloat16,
    )


def _resolve_pixel_dtype(pixel_dtype: str, trt):
    if pixel_dtype == "fp32":
        return trt.float32
    if pixel_dtype == "bf16":
        return trt.bfloat16
    raise ValueError(f"MiniMax-H3 vision pixel_dtype must be 'fp32' or 'bf16', got {pixel_dtype!r}")


def _assemble_vision_conditioner_graph(
    network,
    weights: Mapping[str, Any],
    spec: MiniMaxH3VisionConditionerSpec,
    *,
    pixel_dtype: str,
    trt,
    op,
) -> tuple[Any, ...]:
    """Assemble the graph; split out so ordering can be tested with a fake network."""

    binding_dtype = _resolve_pixel_dtype(pixel_dtype, trt)
    pixel_values = network.add_input("pixel_values", binding_dtype, spec.pixel_values_shape)
    if pixel_values is None:
        raise RuntimeError("TensorRT rejected the MiniMax-H3 pixel_values input")
    pixel_values = op.cast(network, pixel_values, trt.bfloat16)
    hidden = _patch_embedding(network, pixel_values, weights, spec, trt, op)

    positions = _interpolated_position_embeddings(
        weights[f"{_VISUAL_PREFIX}.pos_embed.weight"], spec
    )
    position_tensor = op.weight_constant(network, positions)
    position_tensor = op.cast(network, position_tensor, trt.bfloat16)
    hidden = network.add_elementwise(
        hidden, position_tensor, trt.ElementWiseOperation.SUM
    ).get_output(0)
    cos_half, sin_half = _vision_rope_cos_sin(spec)

    deepstack_hidden: dict[int, Any] = {}
    deepstack_indexes = set(spec.deepstack_visual_indexes)
    for index in range(spec.depth):
        hidden = _vision_block(network, hidden, weights, index, cos_half, sin_half, spec, trt, op)
        if index in deepstack_indexes:
            deepstack_hidden[index] = hidden

    outputs: list[Any] = []
    main = _patch_merger(
        network,
        hidden,
        weights,
        f"{_VISUAL_PREFIX}.merger",
        postshuffle_norm=False,
        spec=spec,
        trt=trt,
        op=op,
    )
    main.name = "image_features"
    network.mark_output(main)
    outputs.append(main)

    for merger_index, block_index in enumerate(spec.deepstack_visual_indexes):
        if block_index not in deepstack_hidden:
            raise RuntimeError(f"MiniMax-H3 vision did not capture DeepStack block {block_index}")
        output = _patch_merger(
            network,
            deepstack_hidden[block_index],
            weights,
            f"{_VISUAL_PREFIX}.deepstack_merger_list.{merger_index}",
            postshuffle_norm=True,
            spec=spec,
            trt=trt,
            op=op,
        )
        output.name = f"deepstack_features_{merger_index}"
        network.mark_output(output)
        outputs.append(output)
    return tuple(outputs)


def _set_patch_dimension(tensor, axis: int) -> None:
    setter = getattr(tensor, "set_dimension_name", None)
    if not callable(setter):
        raise RuntimeError("TensorRT does not support named dynamic dimensions")
    setter(axis, "vision_patch_rows")


def _declare_ref2va_inputs(network, spec, pixel_dtype, trt):
    inputs = {
        "pixel_values": network.add_input(
            "pixel_values",
            _resolve_pixel_dtype(pixel_dtype, trt),
            (-1, spec.patch_vector_size),
        ),
        "position_indices": network.add_input("position_indices", trt.int32, (-1, 4)),
        "position_weights": network.add_input("position_weights", trt.float32, (-1, 4)),
        "vision_position_ids": network.add_input("vision_position_ids", trt.int32, (-1, 2)),
    }
    rejected = sorted(name for name, tensor in inputs.items() if tensor is None)
    if rejected:
        raise RuntimeError(f"TensorRT rejected MiniMax-H3 Ref2VA vision inputs: {rejected}")
    for tensor in inputs.values():
        _set_patch_dimension(tensor, 0)
    return inputs


def _add_ref2va_profile(builder, config, spec):
    profile = builder.create_optimization_profile()
    shapes = {
        "pixel_values": (
            (spec.min_patches, spec.patch_vector_size),
            (spec.opt_patches, spec.patch_vector_size),
            (spec.max_patches, spec.patch_vector_size),
        ),
        "position_indices": (
            (spec.min_patches, 4),
            (spec.opt_patches, 4),
            (spec.max_patches, 4),
        ),
        "position_weights": (
            (spec.min_patches, 4),
            (spec.opt_patches, 4),
            (spec.max_patches, 4),
        ),
        "vision_position_ids": (
            (spec.min_patches, 2),
            (spec.opt_patches, 2),
            (spec.max_patches, 2),
        ),
    }
    for name, (minimum, optimum, maximum) in shapes.items():
        if profile.set_shape(name, minimum, optimum, maximum) is False:
            raise RuntimeError(f"TensorRT rejected MiniMax-H3 Ref2VA profile for {name}")
    if config.add_optimization_profile(profile) < 0:
        raise RuntimeError("TensorRT rejected the MiniMax-H3 Ref2VA vision profile")


def _assemble_ref2va_vision_graph(network, weights, spec, inputs, trt, op):
    pixel_values = op.cast(network, inputs["pixel_values"], trt.bfloat16)
    hidden = _patch_embedding(network, pixel_values, weights, spec, trt, op)
    position_table = op.weight_constant(network, weights[f"{_VISUAL_PREFIX}.pos_embed.weight"])
    position_table = op.cast(network, position_table, trt.bfloat16)
    positions = _runtime_interpolated_positions(
        network,
        position_table,
        inputs["position_indices"],
        inputs["position_weights"],
        spec,
        trt,
        op,
    )
    hidden = network.add_elementwise(hidden, positions, trt.ElementWiseOperation.SUM).get_output(0)
    cosine, sine = _runtime_vision_rope_tables(
        network, inputs["vision_position_ids"], spec, trt, op
    )
    deepstack_hidden = {}
    deepstack_indexes = set(spec.deepstack_visual_indexes)
    for index in range(spec.depth):
        hidden = _vision_block_dynamic(network, hidden, weights, index, cosine, sine, spec, trt, op)
        if index in deepstack_indexes:
            deepstack_hidden[index] = hidden

    outputs = []
    main = _patch_merger_dynamic(
        network,
        hidden,
        weights,
        f"{_VISUAL_PREFIX}.merger",
        postshuffle_norm=False,
        spec=spec,
        trt=trt,
        op=op,
    )
    main.name = "image_features"
    network.mark_output(main)
    outputs.append(main)
    for merger_index, block_index in enumerate(spec.deepstack_visual_indexes):
        output = _patch_merger_dynamic(
            network,
            deepstack_hidden[block_index],
            weights,
            f"{_VISUAL_PREFIX}.deepstack_merger_list.{merger_index}",
            postshuffle_norm=True,
            spec=spec,
            trt=trt,
            op=op,
        )
        output.name = f"deepstack_features_{merger_index}"
        network.mark_output(output)
        outputs.append(output)
    return tuple(outputs)


def build_vision_conditioner_engine(
    checkpoint_config: Mapping[str, Any],
    weights: Mapping[str, Any],
    *,
    workflow: str = "fl2va",
    pixel_dtype: str = "fp32",
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
) -> bytes:
    """Build the FL2VA fixed or Ref2VA dynamic Qwen3-VL vision engine.

    ``pixel_dtype`` selects the processor buffer ABI and accepts only ``fp32``
    or ``bf16``.  Both paths cast to checkpoint-native BF16 before the learned
    patch projection.  Outputs remain BF16, matching the official Qwen3-VL
    vision tower, and are named ``image_features`` followed by
    ``deepstack_features_0`` through ``deepstack_features_2``.
    """

    spec = MiniMaxH3VisionConditionerSpec.from_checkpoint_config(
        checkpoint_config, workflow=workflow
    )
    validate_vision_weights(weights, spec)

    from tensorrt_model_connect import trt_compat

    from . import graph_ops as op

    trt = trt_compat.get_trt()
    # Validate this before allocating a TensorRT builder, including on hosts
    # where the requested binding type is not supported by an older TRT.
    _resolve_pixel_dtype(pixel_dtype, trt)
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    op.configure_builder(config)
    op.configure_workspace(
        config,
        workspace_bytes,
        default_bytes=VISION_CONDITIONER_DEFAULT_WORKSPACE_BYTES,
    )
    ref2va_inputs = None
    if spec.workflow == "ref2va":
        ref2va_inputs = _declare_ref2va_inputs(network, spec, pixel_dtype, trt)
        _add_ref2va_profile(builder, config, spec)
    try:
        if ref2va_inputs is None:
            _assemble_vision_conditioner_graph(
                network,
                weights,
                spec,
                pixel_dtype=pixel_dtype,
                trt=trt,
                op=op,
            )
        else:
            _assemble_ref2va_vision_graph(network, weights, spec, ref2va_inputs, trt, op)
        op.validate_native_network(
            network, expected_attentions=spec.depth, label="vision conditioner"
        )
        print(
            "[minimax-h3] building native Qwen3-VL vision conditioner: "
            f"workflow={spec.workflow}, patches={spec.min_patches}..{spec.max_patches}, "
            f"pixel_dtype={pixel_dtype}, "
            f"deepstack={spec.deepstack_visual_indexes}",
            file=sys.stderr,
        )
        plan = builder.build_serialized_network(network, config)
    finally:
        op.release_weight_buffers(network)
        if consume_weights:
            weights.clear()
    if plan is None:
        raise RuntimeError("TensorRT failed to build MiniMax-H3 vision conditioner")
    del network, config, builder
    gc.collect()
    return bytes(plan)


__all__ = [
    "MiniMaxH3VisionConditionerSpec",
    "VISION_CONDITIONER_DEFAULT_WORKSPACE_BYTES",
    "build_vision_conditioner_engine",
    "checkpoint_keys",
    "expected_weight_shapes",
    "make_ref2va_position_bindings",
    "validate_ref2va_vision_bindings",
    "validate_vision_weights",
]
