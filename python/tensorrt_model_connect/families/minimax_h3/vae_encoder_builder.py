# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static TensorRT builder for the official MiniMax-H3 visual VAE encoder.

This reconstructs the encoder half of Diffusers ``AutoencoderKLMiniMaxH3``
at revision ``06e0f2a81caaa6eaf4381e25cef07b0819582160``.  One engine has a
fixed ``[B, 3, T, H, W]`` shape, but the shape is explicit at build time so a
caller can build separate keyframe and reference-video profiles.

The input is already ImageNet-normalized RGB in float32.  The output is the
48-channel float32 posterior *moments* tensor produced by ``quant_conv``.
This boundary is intentional: the released FL2VA/Ref2VA conditioning recipe
samples that posterior with an independent torch-compatible generator seeded
to 42, rounds the sample through float16 back to float32, and only then
applies per-channel mean/std normalization.  Those stochastic, request-level
steps do not belong in a static TensorRT encoder plan.

The graph preserves the released encoder's details:

* reflect padding in space and zero-only left padding in time;
* frame-isolated GroupNorm statistics;
* 16x spatial and 4x temporal compression;
* single-frame spatial encoding without temporal chunk padding;
* 17-frame chunks, repeated-last-frame padding, and three trailing latent
  frames dropped for multi-frame inputs; and
* the default 256-pixel spatial tiling with at least 64 pixels of overlap,
  including Diffusers' exact overlap distribution and linear stitching.

The checkpoint and all VAE math stay float32, as required by the upstream
``_keep_in_fp32_modules`` contract.  Conversion to the denoiser's lower
precision is a downstream boundary, after posterior sampling and
normalization; this builder does not silently move that cast earlier.
"""

from __future__ import annotations

import gc
import io
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tensorrt_model_connect import trt_compat

from .checkpoint import load_selected_component_state_dict
from .config import resolve_workspace_bytes


DIFFUSERS_ENCODER_REVISION = "06e0f2a81caaa6eaf4381e25cef07b0819582160"
VAE_ENCODER_DEFAULT_WORKSPACE_BYTES = 96 << 30

VAE_INPUT_CHANNELS = 3
VAE_LATENT_CHANNELS = 24
VAE_MOMENT_CHANNELS = 2 * VAE_LATENT_CHANNELS
VAE_SPATIAL_COMPRESSION = 16
VAE_TEMPORAL_COMPRESSION = 4
VAE_CLIP_LENGTH = 17
VAE_TOKEN_DROP = 3
VAE_TILE_SIZE = 256
VAE_TILE_MIN_OVERLAP = 64
VAE_REFERENCE_SPATIAL_ALIGNMENT = 32
VAE_ENCODER_TILE_FRAMES = (1, VAE_CLIP_LENGTH)

# Runtime stitching must mirror ``AutoencoderKLMiniMaxH3._stitch_tiles``.
# Keeping these details in the builder-owned contract avoids a second,
# approximately-equivalent tiler growing in the native runtime.
VAE_SPATIAL_STITCH_ORDER = ("height", "width")
VAE_SPATIAL_BLEND_PREVIOUS_WEIGHT = "1-index/overlap"
VAE_SPATIAL_BLEND_CURRENT_WEIGHT = "index/overlap"


@dataclass(frozen=True)
class VideoVaeEncoderConfig:
    """Validated fields that affect the released visual encoder graph."""

    in_channels: int
    latent_channels: int
    block_out_channels: tuple[int, ...]
    layers_per_block: int
    spatial_downsample_factors: tuple[int, ...]
    temporal_downsample_factors: tuple[int, ...]
    norm_num_groups: int
    norm_eps: float
    spatial_padding_mode: str
    clip_length: int
    token_drop: int
    latents_mean: tuple[float, ...]
    latents_std: tuple[float, ...]

    @property
    def spatial_compression_ratio(self) -> int:
        return math.prod(self.spatial_downsample_factors)

    @property
    def temporal_compression_ratio(self) -> int:
        return math.prod(self.temporal_downsample_factors)

    @property
    def moment_channels(self) -> int:
        return 2 * self.latent_channels


@dataclass(frozen=True)
class VideoVaeEncoderShape:
    """One fixed TensorRT visual-encoder profile."""

    batch_size: int
    num_frames: int
    height: int
    width: int

    @property
    def input_shape(self) -> tuple[int, int, int, int, int]:
        return (self.batch_size, VAE_INPUT_CHANNELS, self.num_frames, self.height, self.width)

    @property
    def latent_frames(self) -> int:
        return latent_frames_for(self.num_frames)

    @property
    def output_shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.batch_size,
            VAE_MOMENT_CHANNELS,
            self.latent_frames,
            self.height // VAE_SPATIAL_COMPRESSION,
            self.width // VAE_SPATIAL_COMPRESSION,
        )

    def validate(self) -> None:
        values = {
            "batch_size": self.batch_size,
            "num_frames": self.num_frames,
            "height": self.height,
            "width": self.width,
        }
        invalid = {
            name: value
            for name, value in values.items()
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0
        }
        if invalid:
            raise ValueError(
                f"MiniMax-H3 visual VAE dimensions must be positive integers: {invalid}"
            )
        for name, value in (("height", self.height), ("width", self.width)):
            if value < 32 or value % VAE_SPATIAL_COMPRESSION:
                raise ValueError(
                    f"MiniMax-H3 visual VAE {name} must be a multiple of "
                    f"{VAE_SPATIAL_COMPRESSION} and at least 32, got {value}"
                )


@dataclass(frozen=True)
class VideoVaeEncoderTileShape:
    """One reusable raw encoder tile plan used by Ref2VA.

    The released encoder has two genuinely different temporal paths: a still
    image is encoded at ``T=1``, while videos are split into ``T=17`` clips.
    Both use the released 256x256 spatial tile.  The video plan deliberately
    returns all five raw temporal moments; the runtime concatenates clip
    outputs and drops the final three tokens once, exactly like Diffusers.
    """

    num_frames: int

    @property
    def input_shape(self) -> tuple[int, int, int, int, int]:
        return (1, VAE_INPUT_CHANNELS, self.num_frames, VAE_TILE_SIZE, VAE_TILE_SIZE)

    @property
    def output_shape(self) -> tuple[int, int, int, int, int]:
        return (
            1,
            VAE_MOMENT_CHANNELS,
            raw_latent_frames_for(self.num_frames),
            VAE_TILE_SIZE // VAE_SPATIAL_COMPRESSION,
            VAE_TILE_SIZE // VAE_SPATIAL_COMPRESSION,
        )

    def validate(self) -> None:
        if self.num_frames not in VAE_ENCODER_TILE_FRAMES or isinstance(self.num_frames, bool):
            raise ValueError(
                "MiniMax-H3 visual VAE tile num_frames must be exactly one of "
                f"{VAE_ENCODER_TILE_FRAMES}, got {self.num_frames!r}"
            )


@dataclass(frozen=True)
class VideoVaeSpatialTile:
    """One axis of one spatial tile, in both pixel and latent units."""

    input_start: int
    input_length: int
    latent_start: int
    latent_length: int
    latent_blend_before: int
    latent_crop_after: int
    stitch_start: int
    stitch_length: int

    def to_metadata(self) -> dict[str, int]:
        return {
            "input_start": self.input_start,
            "input_length": self.input_length,
            "latent_start": self.latent_start,
            "latent_length": self.latent_length,
            "latent_blend_before": self.latent_blend_before,
            "latent_crop_after": self.latent_crop_after,
            "stitch_start": self.stitch_start,
            "stitch_length": self.stitch_length,
        }


@dataclass(frozen=True)
class VideoVaeSpatialTilePlan:
    """Canonical 2-D tile and stitch metadata for one reference canvas."""

    height: int
    width: int
    rows: tuple[VideoVaeSpatialTile, ...]
    columns: tuple[VideoVaeSpatialTile, ...]

    @property
    def latent_height(self) -> int:
        return self.height // VAE_SPATIAL_COMPRESSION

    @property
    def latent_width(self) -> int:
        return self.width // VAE_SPATIAL_COMPRESSION

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-safe contract suitable for a native bundle asset."""

        return {
            "schema_version": 1,
            "height": self.height,
            "width": self.width,
            "tile_size": VAE_TILE_SIZE,
            "minimum_overlap": VAE_TILE_MIN_OVERLAP,
            "input_alignment": VAE_REFERENCE_SPATIAL_ALIGNMENT,
            "spatial_compression": VAE_SPATIAL_COMPRESSION,
            "latent_height": self.latent_height,
            "latent_width": self.latent_width,
            "rows": [tile.to_metadata() for tile in self.rows],
            "columns": [tile.to_metadata() for tile in self.columns],
            "stitch": {
                "blend_order": list(VAE_SPATIAL_STITCH_ORDER),
                "predecessor": "unstitched_neighbor_tile",
                "index_range": "[0,overlap)",
                "previous_weight": VAE_SPATIAL_BLEND_PREVIOUS_WEIGHT,
                "current_weight": VAE_SPATIAL_BLEND_CURRENT_WEIGHT,
            },
        }


@dataclass(frozen=True)
class VideoVaeTemporalChunk:
    """One input slice for a static T=1 or T=17 tile plan."""

    input_start: int
    valid_input_frames: int
    repeated_tail_frames: int
    repeat_source_frame: int | None
    engine_num_frames: int
    raw_moment_start: int
    raw_moment_frames: int

    def to_metadata(self) -> dict[str, int | None]:
        return {
            "input_start": self.input_start,
            "valid_input_frames": self.valid_input_frames,
            "repeated_tail_frames": self.repeated_tail_frames,
            "repeat_source_frame": self.repeat_source_frame,
            "engine_num_frames": self.engine_num_frames,
            "raw_moment_start": self.raw_moment_start,
            "raw_moment_frames": self.raw_moment_frames,
        }


@dataclass(frozen=True)
class VideoVaeTemporalChunkPlan:
    """Canonical temporal split/pad/concatenate/token-drop metadata."""

    num_frames: int
    chunks: tuple[VideoVaeTemporalChunk, ...]
    raw_moment_frames: int
    token_drop: int
    output_moment_frames: int

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-safe contract suitable for a native bundle asset."""

        return {
            "schema_version": 1,
            "num_frames": self.num_frames,
            "clip_length": VAE_CLIP_LENGTH,
            "temporal_compression": VAE_TEMPORAL_COMPRESSION,
            "padding_mode": "repeat_last_frame",
            "concatenate_dimension": 2,
            "chunks": [chunk.to_metadata() for chunk in self.chunks],
            "raw_moment_frames": self.raw_moment_frames,
            "token_drop": self.token_drop,
            "token_drop_scope": "once_from_concatenated_tail",
            "output_moment_frames": self.output_moment_frames,
        }


_EXPECTED_ENCODER_ARCHITECTURE = {
    "in_channels": VAE_INPUT_CHANNELS,
    "latent_channels": VAE_LATENT_CHANNELS,
    "block_out_channels": (128, 256, 256, 512, 512, 1024),
    "layers_per_block": 2,
    "spatial_downsample_factors": (2, 2, 2, 2, 1, 1),
    "temporal_downsample_factors": (1, 2, 2, 1, 1, 1),
    "norm_num_groups": 32,
    "norm_eps": 1.0e-6,
    "spatial_padding_mode": "reflect",
    "clip_length": VAE_CLIP_LENGTH,
    "token_drop": VAE_TOKEN_DROP,
}


def _as_int_tuple(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise ValueError(f"MiniMax-H3 visual VAE {name} must be an integer array")
    return tuple(value)


def _finite_channel_values(value: object, *, name: str, positive: bool) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != VAE_LATENT_CHANNELS:
        raise ValueError(f"MiniMax-H3 visual VAE {name} must contain {VAE_LATENT_CHANNELS} values")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"MiniMax-H3 visual VAE {name} must contain finite numbers")
        try:
            converted = float(item)
        except (TypeError, ValueError) as error:
            raise ValueError(f"MiniMax-H3 visual VAE {name} must contain finite numbers") from error
        if not math.isfinite(converted) or (positive and converted <= 0.0):
            qualifier = "finite positive" if positive else "finite"
            raise ValueError(f"MiniMax-H3 visual VAE {name} must contain {qualifier} numbers")
        result.append(converted)
    return tuple(result)


def validate_video_vae_encoder_config(raw: object) -> VideoVaeEncoderConfig:
    """Require the exact encoder architecture published for MiniMax-H3."""

    if not isinstance(raw, dict):
        raise ValueError("MiniMax-H3 visual VAE config must be a JSON object")

    observed: dict[str, object] = {}
    tuple_fields = {
        "block_out_channels",
        "spatial_downsample_factors",
        "temporal_downsample_factors",
    }
    int_fields = {
        "in_channels",
        "latent_channels",
        "layers_per_block",
        "norm_num_groups",
        "clip_length",
        "token_drop",
    }
    for name in _EXPECTED_ENCODER_ARCHITECTURE:
        value = raw.get(name)
        if name in tuple_fields:
            value = _as_int_tuple(value, name=name)
        elif name in int_fields:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"MiniMax-H3 visual VAE {name} must be an integer")
        elif name == "norm_eps":
            if isinstance(value, bool):
                raise ValueError("MiniMax-H3 visual VAE norm_eps must be finite")
            try:
                value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError("MiniMax-H3 visual VAE norm_eps must be finite") from error
            if not math.isfinite(value):
                raise ValueError("MiniMax-H3 visual VAE norm_eps must be finite")
        elif not isinstance(value, str):
            raise ValueError(f"MiniMax-H3 visual VAE {name} must be a string")
        observed[name] = value

    mismatches = {
        name: (observed[name], expected)
        for name, expected in _EXPECTED_ENCODER_ARCHITECTURE.items()
        if observed[name] != expected
    }
    if mismatches:
        raise ValueError(f"Unsupported MiniMax-H3 visual VAE encoder architecture: {mismatches}")

    means = _finite_channel_values(raw.get("latents_mean"), name="latents_mean", positive=False)
    stds = _finite_channel_values(raw.get("latents_std"), name="latents_std", positive=True)
    return VideoVaeEncoderConfig(
        in_channels=int(observed["in_channels"]),
        latent_channels=int(observed["latent_channels"]),
        block_out_channels=tuple(observed["block_out_channels"]),
        layers_per_block=int(observed["layers_per_block"]),
        spatial_downsample_factors=tuple(observed["spatial_downsample_factors"]),
        temporal_downsample_factors=tuple(observed["temporal_downsample_factors"]),
        norm_num_groups=int(observed["norm_num_groups"]),
        norm_eps=float(observed["norm_eps"]),
        spatial_padding_mode=str(observed["spatial_padding_mode"]),
        clip_length=int(observed["clip_length"]),
        token_drop=int(observed["token_drop"]),
        latents_mean=means,
        latents_std=stds,
    )


def load_video_vae_encoder_config(vae_dir: str | Path) -> VideoVaeEncoderConfig:
    path = Path(vae_dir) / "config.json"
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read MiniMax-H3 visual VAE config: {path}") from error
    return validate_video_vae_encoder_config(raw)


def latent_frames_for(num_frames: int) -> int:
    """Return Diffusers' fixed-chunk encoder output length for ``num_frames``."""

    if not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames <= 0:
        raise ValueError("MiniMax-H3 visual VAE num_frames must be a positive integer")
    if num_frames == 1:
        return 1
    chunks = math.ceil(num_frames / VAE_CLIP_LENGTH)
    tokens_per_chunk = math.ceil(VAE_CLIP_LENGTH / VAE_TEMPORAL_COMPRESSION)
    return chunks * tokens_per_chunk - VAE_TOKEN_DROP


def raw_latent_frames_for(num_frames: int) -> int:
    """Return raw causal-encoder frames before request-level token dropping."""

    if not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames <= 0:
        raise ValueError("MiniMax-H3 visual VAE num_frames must be a positive integer")
    return math.ceil(num_frames / VAE_TEMPORAL_COMPRESSION)


def split_spatial_tiles(
    length: int,
    *,
    tile_size: int = VAE_TILE_SIZE,
    min_overlap: int = VAE_TILE_MIN_OVERLAP,
    compression_ratio: int = VAE_SPATIAL_COMPRESSION,
) -> tuple[list[int], list[int], list[int]]:
    """Mirror Diffusers' latent-aligned MiniMax-H3 tile placement."""

    values = {
        "length": length,
        "tile_size": tile_size,
        "min_overlap": min_overlap,
        "compression_ratio": compression_ratio,
    }
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values.values()):
        raise ValueError("MiniMax-H3 visual VAE tile geometry must contain integers")
    if length <= 0 or tile_size <= 0 or compression_ratio <= 0:
        raise ValueError("MiniMax-H3 visual VAE tile sizes and compression must be positive")
    if min_overlap < 0 or min_overlap >= tile_size:
        raise ValueError("MiniMax-H3 visual VAE tile overlap must be in [0, tile_size)")
    if tile_size % compression_ratio or min_overlap % compression_ratio:
        raise ValueError("MiniMax-H3 visual VAE tile geometry must be latent-aligned")
    if tile_size >= length:
        return [0], [length], []

    num_tiles = math.ceil(length / tile_size)
    while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
        num_tiles += 1

    overlaps = [min_overlap] * (num_tiles - 1)
    remaining = tile_size * num_tiles - sum(overlaps) - length
    if remaining % compression_ratio:
        raise ValueError(
            "MiniMax-H3 visual VAE input length is not compatible with latent-aligned tiling"
        )
    for index in range(remaining // compression_ratio):
        overlaps[index % (num_tiles - 1)] += compression_ratio

    starts = [0]
    for index in range(num_tiles - 1):
        starts.append(starts[-1] + tile_size - overlaps[index])
    return starts, [tile_size] * num_tiles, overlaps


def _spatial_axis_plan(length: int) -> tuple[VideoVaeSpatialTile, ...]:
    starts, lengths, overlaps = split_spatial_tiles(length)
    ratio = VAE_SPATIAL_COMPRESSION
    latent_overlaps = [overlap // ratio for overlap in overlaps]
    result = []
    for index, (start, tile_length) in enumerate(zip(starts, lengths)):
        latent_start = start // ratio
        latent_length = tile_length // ratio
        blend_before = latent_overlaps[index - 1] if index else 0
        crop_after = latent_overlaps[index] if index < len(latent_overlaps) else 0
        result.append(
            VideoVaeSpatialTile(
                input_start=start,
                input_length=tile_length,
                latent_start=latent_start,
                latent_length=latent_length,
                latent_blend_before=blend_before,
                latent_crop_after=crop_after,
                stitch_start=latent_start,
                stitch_length=latent_length - crop_after,
            )
        )
    return tuple(result)


def make_spatial_tile_plan(height: int, width: int) -> VideoVaeSpatialTilePlan:
    """Describe exact Diffusers tiling for one legal Ref2VA VAE canvas.

    Ref2VA preprocessing produces canvases aligned to 32 pixels and its legal
    image/video sizes are all at least one released 256-pixel tile per axis.
    Smaller inputs cannot be made equivalent by padding a static tile: reflect
    padding and frame-isolated GroupNorm would observe a different domain, so
    they are rejected instead of being routed through an approximate profile.
    """

    values = {"height": height, "width": width}
    invalid = {
        name: value
        for name, value in values.items()
        if not isinstance(value, int) or isinstance(value, bool) or value < VAE_TILE_SIZE
    }
    if invalid:
        raise ValueError(
            "MiniMax-H3 Ref2VA visual VAE canvas dimensions must be integers at least "
            f"{VAE_TILE_SIZE}: {invalid}"
        )
    misaligned = {
        name: value for name, value in values.items() if value % VAE_REFERENCE_SPATIAL_ALIGNMENT
    }
    if misaligned:
        raise ValueError(
            f"MiniMax-H3 Ref2VA visual VAE canvas dimensions must be 32-aligned: {misaligned}"
        )
    return VideoVaeSpatialTilePlan(
        height=height,
        width=width,
        rows=_spatial_axis_plan(height),
        columns=_spatial_axis_plan(width),
    )


def validate_spatial_tile_plan(
    plan: VideoVaeSpatialTilePlan,
) -> VideoVaeSpatialTilePlan:
    """Fail closed unless ``plan`` is the exact canonical stitch contract."""

    if not isinstance(plan, VideoVaeSpatialTilePlan):
        raise ValueError("MiniMax-H3 visual VAE spatial tile plan has the wrong type")
    expected = make_spatial_tile_plan(plan.height, plan.width)
    if plan != expected:
        raise ValueError("MiniMax-H3 visual VAE spatial tile plan does not match Diffusers")
    return plan


def spatial_tile_metadata(height: int, width: int) -> dict[str, object]:
    """Return canonical JSON-safe spatial tile metadata for native runtime use."""

    return make_spatial_tile_plan(height, width).to_metadata()


def validate_spatial_tile_metadata(
    metadata: object, *, height: int, width: int
) -> dict[str, object]:
    """Fail closed on altered or incomplete serialized stitch metadata."""

    expected = spatial_tile_metadata(height, width)
    if not isinstance(metadata, dict) or metadata != expected:
        raise ValueError("MiniMax-H3 visual VAE spatial tile metadata does not match Diffusers")
    return expected


def make_temporal_chunk_plan(num_frames: int) -> VideoVaeTemporalChunkPlan:
    """Describe exact Diffusers T=1/T=17 routing for a reference clip."""

    if not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames <= 0:
        raise ValueError("MiniMax-H3 visual VAE num_frames must be a positive integer")

    if num_frames == 1:
        chunk = VideoVaeTemporalChunk(
            input_start=0,
            valid_input_frames=1,
            repeated_tail_frames=0,
            repeat_source_frame=None,
            engine_num_frames=1,
            raw_moment_start=0,
            raw_moment_frames=1,
        )
        return VideoVaeTemporalChunkPlan(
            num_frames=1,
            chunks=(chunk,),
            raw_moment_frames=1,
            token_drop=0,
            output_moment_frames=1,
        )

    chunk_count = math.ceil(num_frames / VAE_CLIP_LENGTH)
    raw_frames_per_chunk = raw_latent_frames_for(VAE_CLIP_LENGTH)
    chunks = []
    for index in range(chunk_count):
        input_start = index * VAE_CLIP_LENGTH
        valid_frames = min(VAE_CLIP_LENGTH, num_frames - input_start)
        repeated_frames = VAE_CLIP_LENGTH - valid_frames
        chunks.append(
            VideoVaeTemporalChunk(
                input_start=input_start,
                valid_input_frames=valid_frames,
                repeated_tail_frames=repeated_frames,
                repeat_source_frame=num_frames - 1 if repeated_frames else None,
                engine_num_frames=VAE_CLIP_LENGTH,
                raw_moment_start=index * raw_frames_per_chunk,
                raw_moment_frames=raw_frames_per_chunk,
            )
        )
    raw_moment_frames = chunk_count * raw_frames_per_chunk
    return VideoVaeTemporalChunkPlan(
        num_frames=num_frames,
        chunks=tuple(chunks),
        raw_moment_frames=raw_moment_frames,
        token_drop=VAE_TOKEN_DROP,
        output_moment_frames=raw_moment_frames - VAE_TOKEN_DROP,
    )


def validate_temporal_chunk_plan(
    plan: VideoVaeTemporalChunkPlan,
) -> VideoVaeTemporalChunkPlan:
    """Fail closed unless ``plan`` is the exact canonical chunk contract."""

    if not isinstance(plan, VideoVaeTemporalChunkPlan):
        raise ValueError("MiniMax-H3 visual VAE temporal chunk plan has the wrong type")
    expected = make_temporal_chunk_plan(plan.num_frames)
    if plan != expected:
        raise ValueError("MiniMax-H3 visual VAE temporal chunk plan does not match Diffusers")
    return plan


def temporal_chunk_metadata(num_frames: int) -> dict[str, object]:
    """Return canonical JSON-safe temporal metadata for native runtime use."""

    return make_temporal_chunk_plan(num_frames).to_metadata()


def validate_temporal_chunk_metadata(metadata: object, *, num_frames: int) -> dict[str, object]:
    """Fail closed on altered or incomplete serialized chunk metadata."""

    expected = temporal_chunk_metadata(num_frames)
    if not isinstance(metadata, dict) or metadata != expected:
        raise ValueError("MiniMax-H3 visual VAE temporal chunk metadata does not match Diffusers")
    return expected


# Descriptive aliases for call sites that keep multiple model-specific tilers.
make_vae_encoder_spatial_tile_plan = make_spatial_tile_plan
validate_vae_encoder_spatial_tile_plan = validate_spatial_tile_plan
make_vae_encoder_temporal_chunk_plan = make_temporal_chunk_plan
validate_vae_encoder_temporal_chunk_plan = validate_temporal_chunk_plan


def checkpoint_keys(config: VideoVaeEncoderConfig | None = None) -> tuple[str, ...]:
    """Return only checkpoint tensors owned by the visual encoder and quant conv."""

    block_channels = (
        tuple(_EXPECTED_ENCODER_ARCHITECTURE["block_out_channels"])
        if config is None
        else config.block_out_channels
    )
    layers_per_block = (
        int(_EXPECTED_ENCODER_ARCHITECTURE["layers_per_block"])
        if config is None
        else config.layers_per_block
    )
    spatial_factors = (
        tuple(_EXPECTED_ENCODER_ARCHITECTURE["spatial_downsample_factors"])
        if config is None
        else config.spatial_downsample_factors
    )
    temporal_factors = (
        tuple(_EXPECTED_ENCODER_ARCHITECTURE["temporal_downsample_factors"])
        if config is None
        else config.temporal_downsample_factors
    )

    names = ["encoder.conv_in.weight", "encoder.conv_in.bias"]
    in_channels = block_channels[0]
    for block_index, out_channels in enumerate(block_channels):
        for layer_index in range(layers_per_block):
            prefix = f"encoder.down_blocks.{block_index}.resnets.{layer_index}"
            names.extend(
                f"{prefix}.{module}.{kind}"
                for module in ("norm1", "conv1", "norm2", "conv2")
                for kind in ("weight", "bias")
            )
            if layer_index == 0 and in_channels != out_channels:
                names.extend((f"{prefix}.conv_shortcut.weight", f"{prefix}.conv_shortcut.bias"))
            in_channels = out_channels
        if spatial_factors[block_index] * temporal_factors[block_index] > 1:
            prefix = f"encoder.down_blocks.{block_index}.downsamplers.0.conv"
            names.extend((f"{prefix}.weight", f"{prefix}.bias"))
    names.extend(
        (
            "encoder.norm_out.weight",
            "encoder.norm_out.bias",
            "encoder.conv_out.weight",
            "encoder.conv_out.bias",
            "quant_conv.weight",
            "quant_conv.bias",
        )
    )
    return tuple(names)


def _make_encoder_module(torch: Any, config: VideoVaeEncoderConfig, *, raw_tile: bool = False):
    """Reconstruct the pinned Diffusers encoder under exact checkpoint names.

    ``raw_tile`` exposes only ``quant_conv(encoder(tile))``.  It intentionally
    skips spatial orchestration and temporal token dropping so the same two
    static plans can serve every legal Ref2VA reference geometry.
    """

    nn = torch.nn
    functional = torch.nn.functional

    class MiniMaxH3VideoCausalConv3d(nn.Conv3d):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int | tuple[int, int, int],
            stride: int | tuple[int, int, int] = 1,
            spatial_padding: int = 0,
            temporal_padding: int = 0,
        ) -> None:
            super().__init__(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
            )
            self.spatial_padding = spatial_padding
            self.temporal_padding = temporal_padding

        def forward(self, hidden_states):
            if self.spatial_padding > 0:
                padding = self.spatial_padding
                hidden_states = functional.pad(
                    hidden_states,
                    (padding, padding, padding, padding, 0, 0),
                    mode=config.spatial_padding_mode,
                )
            if self.temporal_padding > 0:
                hidden_states = functional.pad(
                    hidden_states,
                    (0, 0, 0, 0, self.temporal_padding, 0),
                    mode="constant",
                )
            return functional.conv3d(
                hidden_states,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=0,
                dilation=self.dilation,
            )

    class MiniMaxH3VideoGroupNorm(nn.GroupNorm):
        def forward(self, hidden_states):
            batch_size, channels, frames, height, width = hidden_states.shape
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()
            hidden_states = hidden_states.view(batch_size * frames, channels, 1, height, width)
            hidden_states = super().forward(hidden_states)
            hidden_states = hidden_states.view(batch_size, frames, channels, height, width)
            return hidden_states.permute(0, 2, 1, 3, 4).contiguous()

    class MiniMaxH3VideoResnetBlock3d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.norm1 = MiniMaxH3VideoGroupNorm(
                config.norm_num_groups, in_channels, eps=config.norm_eps, affine=True
            )
            self.conv1 = MiniMaxH3VideoCausalConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                spatial_padding=1,
                temporal_padding=2,
            )
            self.norm2 = MiniMaxH3VideoGroupNorm(
                config.norm_num_groups, out_channels, eps=config.norm_eps, affine=True
            )
            self.conv2 = MiniMaxH3VideoCausalConv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                spatial_padding=1,
                temporal_padding=2,
            )
            self.conv_shortcut = None
            if in_channels != out_channels:
                self.conv_shortcut = MiniMaxH3VideoCausalConv3d(
                    in_channels, out_channels, kernel_size=1
                )

        def forward(self, hidden_states):
            residual = hidden_states
            hidden_states = self.conv1(functional.silu(self.norm1(hidden_states)))
            hidden_states = self.conv2(functional.silu(self.norm2(hidden_states)))
            if self.conv_shortcut is not None:
                residual = self.conv_shortcut(residual)
            return residual + hidden_states

    class MiniMaxH3VideoDownsample3d(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            temporal_stride: int,
            spatial_stride: int,
        ) -> None:
            super().__init__()
            self.spatial_stride = spatial_stride
            self.conv = MiniMaxH3VideoCausalConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=(temporal_stride, spatial_stride, spatial_stride),
                temporal_padding=2,
            )

        def forward(self, hidden_states):
            if self.spatial_stride == 2:
                hidden_states = functional.pad(
                    hidden_states, (0, 1, 0, 1, 0, 0), mode=config.spatial_padding_mode
                )
            return self.conv(hidden_states)

    class MiniMaxH3VideoDownBlock3d(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            temporal_factor: int,
            spatial_factor: int,
        ) -> None:
            super().__init__()
            self.resnets = nn.ModuleList(
                [
                    MiniMaxH3VideoResnetBlock3d(
                        in_channels if index == 0 else out_channels, out_channels
                    )
                    for index in range(config.layers_per_block)
                ]
            )
            self.downsamplers = None
            if temporal_factor * spatial_factor > 1:
                self.downsamplers = nn.ModuleList(
                    [
                        MiniMaxH3VideoDownsample3d(
                            out_channels,
                            out_channels,
                            temporal_stride=temporal_factor,
                            spatial_stride=spatial_factor,
                        )
                    ]
                )

        def forward(self, hidden_states):
            for resnet in self.resnets:
                hidden_states = resnet(hidden_states)
            if self.downsamplers is not None:
                for downsampler in self.downsamplers:
                    hidden_states = downsampler(hidden_states)
            return hidden_states

    class MiniMaxH3VideoEncoder3d(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv_in = MiniMaxH3VideoCausalConv3d(
                config.in_channels,
                config.block_out_channels[0],
                kernel_size=3,
                spatial_padding=1,
                temporal_padding=2,
            )
            block_inputs = (config.block_out_channels[0],) + config.block_out_channels[:-1]
            self.down_blocks = nn.ModuleList(
                [
                    MiniMaxH3VideoDownBlock3d(
                        block_inputs[index],
                        config.block_out_channels[index],
                        config.temporal_downsample_factors[index],
                        config.spatial_downsample_factors[index],
                    )
                    for index in range(len(config.block_out_channels))
                ]
            )
            self.norm_out = MiniMaxH3VideoGroupNorm(
                config.norm_num_groups,
                config.block_out_channels[-1],
                eps=config.norm_eps,
                affine=True,
            )
            self.conv_out = MiniMaxH3VideoCausalConv3d(
                config.block_out_channels[-1],
                config.moment_channels,
                kernel_size=3,
                spatial_padding=1,
                temporal_padding=2,
            )

        def forward(self, hidden_states):
            hidden_states = self.conv_in(hidden_states)
            for down_block in self.down_blocks:
                hidden_states = down_block(hidden_states)
            hidden_states = functional.silu(self.norm_out(hidden_states))
            return self.conv_out(hidden_states)

    class StaticVideoVaeEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = MiniMaxH3VideoEncoder3d()
            self.quant_conv = nn.Conv3d(
                config.moment_channels, config.moment_channels, kernel_size=1
            )

        @staticmethod
        def _blend(a, b, extent: int, dim: int):
            extent = min(a.shape[dim], b.shape[dim], extent)
            positions = torch.arange(extent, device=b.device, dtype=b.dtype)
            shape = [1] * a.ndim
            shape[dim] = extent
            weight_a = (1 - positions / extent).view(shape)
            weight_b = (positions / extent).view(shape)
            slice_a = [slice(None)] * a.ndim
            slice_a[dim] = slice(-extent, None)
            slice_b = [slice(None)] * b.ndim
            slice_b[dim] = slice(0, extent)
            blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b
            if extent == b.shape[dim]:
                return blended
            slice_rest = [slice(None)] * b.ndim
            slice_rest[dim] = slice(extent, None)
            return torch.cat([blended, b[tuple(slice_rest)]], dim=dim)

        def _stitch(self, tiles, height_overlaps, width_overlaps):
            rows = []
            for row_index, row in enumerate(tiles):
                result_row = []
                for column_index, tile in enumerate(row):
                    if row_index > 0:
                        tile = self._blend(
                            tiles[row_index - 1][column_index],
                            tile,
                            height_overlaps[row_index - 1],
                            dim=-2,
                        )
                    if column_index > 0:
                        tile = self._blend(
                            row[column_index - 1],
                            tile,
                            width_overlaps[column_index - 1],
                            dim=-1,
                        )
                    if row_index < len(tiles) - 1:
                        tile = tile[..., : -height_overlaps[row_index], :]
                    if column_index < len(row) - 1:
                        tile = tile[..., :, : -width_overlaps[column_index]]
                    result_row.append(tile)
                rows.append(torch.cat(result_row, dim=-1))
            return torch.cat(rows, dim=-2)

        def _encode_clip(self, normalized_rgb):
            height, width = normalized_rgb.shape[-2:]
            y_starts, y_lengths, y_overlaps = split_spatial_tiles(height)
            x_starts, x_lengths, x_overlaps = split_spatial_tiles(width)
            if len(y_starts) == len(x_starts) == 1:
                return self.quant_conv(self.encoder(normalized_rgb))

            rows = []
            for y_start, y_length in zip(y_starts, y_lengths):
                row = []
                for x_start, x_length in zip(x_starts, x_lengths):
                    tile = normalized_rgb[
                        ...,
                        y_start : y_start + y_length,
                        x_start : x_start + x_length,
                    ]
                    row.append(self.quant_conv(self.encoder(tile)))
                rows.append(row)

            ratio = config.spatial_compression_ratio
            latent_y_overlaps = [overlap // ratio for overlap in y_overlaps]
            latent_x_overlaps = [overlap // ratio for overlap in x_overlaps]
            return self._stitch(rows, latent_y_overlaps, latent_x_overlaps)

        def forward(self, normalized_rgb):
            # Upstream pins encoder and quant_conv weights to float32 and
            # casts incoming pixels to that dtype before any VAE operation.
            normalized_rgb = normalized_rgb.to(torch.float32)
            if raw_tile:
                return self.quant_conv(self.encoder(normalized_rgb))
            num_frames = normalized_rgb.shape[2]
            if num_frames == 1:
                return self._encode_clip(normalized_rgb)
            if num_frames % config.clip_length:
                padding = (-num_frames) % config.clip_length
                tail = normalized_rgb[:, :, -1:].repeat(1, 1, padding, 1, 1)
                normalized_rgb = torch.cat([normalized_rgb, tail], dim=2)
            moments = torch.cat(
                [
                    self._encode_clip(
                        normalized_rgb[
                            :,
                            :,
                            index * config.clip_length : (index + 1) * config.clip_length,
                        ]
                    )
                    for index in range(normalized_rgb.shape[2] // config.clip_length)
                ],
                dim=2,
            )
            if config.token_drop:
                moments = moments[:, :, : -config.token_drop]
            return moments

    return StaticVideoVaeEncoder().float().eval()


def _make_encoder_tile_module(torch: Any, config: VideoVaeEncoderConfig):
    """Reconstruct the reusable raw-tile graph under the same checkpoint keys."""

    return _make_encoder_module(torch, config, raw_tile=True)


def _load_encoder_weights(
    torch: Any,
    module: Any,
    vae_dir: Path,
    config: VideoVaeEncoderConfig,
) -> None:
    expected = tuple(module.state_dict())
    declared = checkpoint_keys(config)
    if set(expected) != set(declared):
        missing = sorted(set(declared) - set(expected))
        extra = sorted(set(expected) - set(declared))
        raise RuntimeError(
            f"MiniMax-H3 visual VAE reconstruction key mismatch: missing={missing}, extra={extra}"
        )
    state = load_selected_component_state_dict(vae_dir, declared)
    wrong_dtype = sorted(name for name, value in state.items() if value.dtype != torch.float32)
    if wrong_dtype:
        raise ValueError(
            f"MiniMax-H3 visual VAE encoder checkpoint tensors must be float32: {wrong_dtype}"
        )
    module.load_state_dict(state, strict=True)


def _export_encoder_onnx(
    vae_dir: Path,
    config: VideoVaeEncoderConfig,
    shape: VideoVaeEncoderShape,
    verbose: bool,
) -> bytes:
    import torch

    module = _make_encoder_module(torch, config)
    _load_encoder_weights(torch, module, vae_dir, config)
    dummy = torch.zeros(shape.input_shape, dtype=torch.float32)
    onnx_buffer = io.BytesIO()
    if verbose:
        print(
            "[trtmc build]   Exporting official MiniMax-H3 float32 visual VAE "
            f"encoder {shape.input_shape} -> {shape.output_shape} posterior moments ...",
            file=sys.stderr,
        )
    with torch.inference_mode():
        torch.onnx.export(
            module,
            dummy,
            onnx_buffer,
            opset_version=17,
            input_names=["normalized_rgb"],
            output_names=["posterior_moments"],
            dynamo=False,
        )
    payload = onnx_buffer.getvalue()
    del dummy, module
    gc.collect()
    return payload


def _export_encoder_tile_onnx(
    vae_dir: Path,
    config: VideoVaeEncoderConfig,
    shape: VideoVaeEncoderTileShape,
    verbose: bool,
) -> bytes:
    """Export raw encoder+quant_conv for one reusable static tile shape."""

    import torch

    module = _make_encoder_tile_module(torch, config)
    _load_encoder_weights(torch, module, vae_dir, config)
    dummy = torch.zeros(shape.input_shape, dtype=torch.float32)
    onnx_buffer = io.BytesIO()
    if verbose:
        print(
            "[trtmc build]   Exporting official MiniMax-H3 float32 visual VAE "
            f"tile encoder {shape.input_shape} -> {shape.output_shape} raw posterior moments ...",
            file=sys.stderr,
        )
    with torch.inference_mode():
        torch.onnx.export(
            module,
            dummy,
            onnx_buffer,
            opset_version=17,
            input_names=["normalized_rgb"],
            output_names=["posterior_moments"],
            dynamo=False,
        )
    payload = onnx_buffer.getvalue()
    del dummy, module
    gc.collect()
    return payload


def _build_serialized_engine(
    onnx_bytes: bytes,
    *,
    shape: VideoVaeEncoderShape | VideoVaeEncoderTileShape,
    verbose: bool,
    workspace_bytes: int | None,
) -> bytes:
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(explicit_batch=True, strongly_typed=True)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_bytes):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError(
            "MiniMax-H3 visual VAE encoder ONNX parsing failed:\n" + "\n".join(errors)
        )

    if network.num_inputs != 1 or network.num_outputs != 1:
        raise RuntimeError(
            "MiniMax-H3 visual VAE encoder ONNX must expose exactly one input and one output"
        )
    input_tensor = network.get_input(0)
    output_tensor = network.get_output(0)
    input_contract = (input_tensor.name, tuple(input_tensor.shape), input_tensor.dtype)
    output_contract = (output_tensor.name, tuple(output_tensor.shape), output_tensor.dtype)
    expected_input = ("normalized_rgb", shape.input_shape, trt.float32)
    expected_output = ("posterior_moments", shape.output_shape, trt.float32)
    if input_contract != expected_input or output_contract != expected_output:
        raise RuntimeError(
            "MiniMax-H3 visual VAE encoder ONNX contract mismatch: "
            f"input={input_contract}, output={output_contract}"
        )

    build_config = builder.create_builder_config()
    resolved_workspace = resolve_workspace_bytes(
        workspace_bytes, default_bytes=VAE_ENCODER_DEFAULT_WORKSPACE_BYTES
    )
    pool = trt.MemoryPoolType.WORKSPACE
    build_config.set_memory_pool_limit(pool, resolved_workspace)
    if int(build_config.get_memory_pool_limit(pool)) != resolved_workspace:
        raise RuntimeError(
            "TensorRT did not apply the requested MiniMax-H3 visual VAE encoder workspace limit"
        )
    if verbose:
        print(
            "[trtmc build]   Building MiniMax-H3 float32 visual VAE encoder "
            f"for {shape.input_shape} ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("TensorRT MiniMax-H3 visual VAE encoder build failed")
    return bytes(plan)


def build_vae_encoder_engine(
    vae_dir: str | Path,
    *,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    verbose: bool = False,
    workspace_bytes: int | None = None,
) -> bytes:
    """Build normalized RGB -> posterior moments for one explicit static shape.

    This does not sample or normalize the posterior.  In particular, callers
    implementing visual conditioning must still use the released seed-42
    posterior sample and FP16-rounding recipe before latent normalization.
    """

    shape = VideoVaeEncoderShape(batch_size, num_frames, height, width)
    shape.validate()
    root = Path(vae_dir)
    config = load_video_vae_encoder_config(root)
    onnx_bytes = _export_encoder_onnx(root, config, shape, verbose)
    try:
        return _build_serialized_engine(
            onnx_bytes,
            shape=shape,
            verbose=verbose,
            workspace_bytes=workspace_bytes,
        )
    finally:
        del onnx_bytes
        gc.collect()


def build_vae_encoder_tile_engine(
    vae_dir: str | Path,
    *,
    num_frames: int,
    verbose: bool = False,
    workspace_bytes: int | None = None,
) -> bytes:
    """Build one raw 256x256 Ref2VA tile plan for exactly T=1 or T=17.

    The returned plan performs no spatial stitching, temporal padding, or
    token dropping.  Callers must use :func:`make_spatial_tile_plan` and
    :func:`make_temporal_chunk_plan` to reproduce the released orchestration.
    """

    shape = VideoVaeEncoderTileShape(num_frames)
    shape.validate()
    root = Path(vae_dir)
    config = load_video_vae_encoder_config(root)
    onnx_bytes = _export_encoder_tile_onnx(root, config, shape, verbose)
    try:
        return _build_serialized_engine(
            onnx_bytes,
            shape=shape,
            verbose=verbose,
            workspace_bytes=workspace_bytes,
        )
    finally:
        del onnx_bytes
        gc.collect()


def build_vae_encoder_tile_engines(
    vae_dir: str | Path,
    *,
    verbose: bool = False,
    workspace_bytes: int | None = None,
) -> dict[int, bytes]:
    """Build the complete reusable Ref2VA tile-plan set, keyed by T."""

    return {
        num_frames: build_vae_encoder_tile_engine(
            vae_dir,
            num_frames=num_frames,
            verbose=verbose,
            workspace_bytes=workspace_bytes,
        )
        for num_frames in VAE_ENCODER_TILE_FRAMES
    }


# Descriptive alias for call sites that distinguish the visual and audio VAEs.
build_visual_vae_encoder_engine = build_vae_encoder_engine
build_ref2va_vae_encoder_tile_engines = build_vae_encoder_tile_engines
