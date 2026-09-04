# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated native TensorRT profiles for MiniMax-H3.

The fixed profile preserves the 124-frame, 1344x768 shape used by the public
Sol-Engine H3 benchmark.  The production media profile covers every released
5--15 second geometry, the public continuous 1:4--4:1 canvas resolver, and the
documented explicit 960x544 performance canvas in both orientations.
Structural row counts are explicit because prompt/media packing is part of
the engine ABI and must match the Hugging Face reference.
"""

from __future__ import annotations

from dataclasses import dataclass


TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES = 96 << 30
VISION_ENCODER_DEFAULT_WORKSPACE_BYTES = 32 << 30
ADALN_PRECOMPUTE_DEFAULT_WORKSPACE_BYTES = 64 << 30
DENOISER_DEFAULT_WORKSPACE_BYTES = 96 << 30
KEYFRAME_VAE_ENCODER_DEFAULT_WORKSPACE_BYTES = 32 << 30
VAE_TILE_DECODER_DEFAULT_WORKSPACE_BYTES = 96 << 30
AUDIO_VAE_DECODER_DEFAULT_WORKSPACE_BYTES = 96 << 30
AUDIO_LATENT_FRAMES_MIN = 207
AUDIO_LATENT_FRAMES_OPT = 207
AUDIO_LATENT_FRAMES_MAX = 575
VIDEO_NUM_FRAMES_MIN = 124
VIDEO_NUM_FRAMES_OPT = 124
VIDEO_NUM_FRAMES_MAX = 345
VIDEO_ROWS_MIN = 18_870
VIDEO_ROWS_OPT = 37_296
TARGET_VIDEO_ROWS_MAX = 106_488
VIDEO_ROWS_MAX = 108_576
CANVAS_MULTIPLE = 32
CANVAS_SHORT_EDGE = 768
CANVAS_MAX_PIXELS = 768 * 1344
CANVAS_MIN_ASPECT_RATIO = 0.25
CANVAS_MAX_ASPECT_RATIO = 4.0
# Extra explicit Diffusers performance canvas, stored as (height, width). The
# TensorRT runtime remains a finite allowlist: these two orientations are in
# addition to, not a replacement for, the 95 resolver-produced canvases.
NATIVE_EXPLICIT_CANVAS_SIZES = ((544, 960), (960, 544))
FASTH3_SCHEDULER_GRID_POINTS = 5
FASTH3_TRANSFORMER_FORWARDS = 4
FASTH3_GUIDANCE_SCALE = 1.0
FASTH3_VSA_TILE_SIZE = 64
FASTH3_VSA_VIDEO_KEEP_NUMERATOR = 1
FASTH3_VSA_VIDEO_KEEP_DENOMINATOR = 10
FASTH3_VSA_VIDEO_ROWS_PER_LATENT_FRAME = 24 * 42
FASTH3_VSA_VIDEO_TILE_ROWS = 4
FASTH3_VSA_VIDEO_TILE_HEIGHT = 4
FASTH3_VSA_VIDEO_TILE_WIDTH = 4
# Maximum across the released 768p canvas resolver (continuous 1:4..4:1
# aspect range, dimensions rounded to 32) at 345 frames, including one or two
# FL2VA conditioning frames in the same video-tile segment.
FASTH3_VSA_MAX_VIDEO_TILES = 2080
FASTH3_VSA_MIN_VIDEO_TILES = 360

# The RTX path builds each plan in a fresh process, so one conservative
# workspace and runtime budget cover every stage without coupling the public
# artifact to a particular workstation identity.
RTX_STAGED_WORKSPACE_BYTES = 16 << 30
RTX_WEIGHT_STREAMING_BUDGET_BYTES = 32 << 30
RTX_CUDA_MAJOR = 12
TRT_DEFAULT_WORKSPACE_POLICY = "trt_default_max"

DEFAULT_WORKSPACE_LIMIT_BYTES = {
    "text_encoder.plan": TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES,
    "vision_encoder.plan": VISION_ENCODER_DEFAULT_WORKSPACE_BYTES,
    "adaln_precompute.plan": ADALN_PRECOMPUTE_DEFAULT_WORKSPACE_BYTES,
    "denoiser.plan": DENOISER_DEFAULT_WORKSPACE_BYTES,
    "fl2va_keyframe_vae_encoder.plan": KEYFRAME_VAE_ENCODER_DEFAULT_WORKSPACE_BYTES,
    "vae_tile_decoder.plan": VAE_TILE_DECODER_DEFAULT_WORKSPACE_BYTES,
    "audio_vae_decoder.plan": AUDIO_VAE_DECODER_DEFAULT_WORKSPACE_BYTES,
}

FIRST_BLOCK_CACHE_DENOISER_PLAN_FILENAMES = (
    "denoiser_head.plan",
    "denoiser_tail.plan",
    "denoiser_finish.plan",
)
FASTH3_SEGMENTED_DENOISER_PLAN_FILENAMES = (
    "denoiser_entry.plan",
    *(f"denoiser_transition_{index:02d}.plan" for index in range(49)),
    "denoiser_finish.plan",
)


def native_plan_filenames(
    *, first_block_cache: bool, segmented_vsa: bool = False
) -> tuple[str, ...]:
    """Return the exact plan set selected by the native denoiser profile."""

    if not isinstance(first_block_cache, bool):
        raise ValueError("MiniMax-H3 first_block_cache must be a boolean")
    if not isinstance(segmented_vsa, bool):
        raise ValueError("MiniMax-H3 segmented_vsa must be a boolean")
    if first_block_cache and segmented_vsa:
        raise ValueError("MiniMax-H3 segmented VSA cannot be combined with FirstBlockCache")
    if segmented_vsa:
        denoiser_plans = FASTH3_SEGMENTED_DENOISER_PLAN_FILENAMES
    elif first_block_cache:
        denoiser_plans = FIRST_BLOCK_CACHE_DENOISER_PLAN_FILENAMES
    else:
        denoiser_plans = ("denoiser.plan",)
    return (
        "text_encoder.plan",
        "vision_encoder.plan",
        "adaln_precompute.plan",
        *denoiser_plans,
        "fl2va_keyframe_vae_encoder.plan",
        "vae_tile_decoder.plan",
        "audio_vae_decoder.plan",
    )


def default_workspace_limit_bytes(
    *, first_block_cache: bool, segmented_vsa: bool = False
) -> dict[str, int | str]:
    """Return per-plan tactic workspace limits for one denoiser layout."""

    return {
        filename: (
            TRT_DEFAULT_WORKSPACE_POLICY
            if first_block_cache
            and (filename.startswith("denoiser_") or filename.startswith("adaln_"))
            else
            DENOISER_DEFAULT_WORKSPACE_BYTES
            if filename.startswith("denoiser_") or filename == "denoiser.plan"
            else ADALN_PRECOMPUTE_DEFAULT_WORKSPACE_BYTES
            if filename.startswith("adaln_")
            else DEFAULT_WORKSPACE_LIMIT_BYTES[filename]
        )
        for filename in native_plan_filenames(
            first_block_cache=first_block_cache, segmented_vsa=segmented_vsa
        )
    }


def resolve_workspace_bytes(workspace_bytes: int | None, *, default_bytes: int) -> int:
    """Resolve a positive tactic-workspace limit without silently coercing values."""

    resolved = default_bytes if workspace_bytes is None else workspace_bytes
    if not isinstance(resolved, int) or isinstance(resolved, bool) or resolved <= 0:
        raise ValueError("MiniMax-H3 TensorRT workspace_bytes must be a positive integer")
    return resolved


@dataclass(frozen=True)
class MiniMaxH3Config:
    hidden_size: int = 5376
    num_layers: int = 50
    num_refiner_layers: int = 2
    num_heads: int = 56
    head_dim: int = 128
    ffn_dim: int = 14336
    video_in_channels: int = 24
    video_patch: tuple[int, int, int] = (1, 2, 2)
    audio_in_channels: int = 32
    text_dim: int = 5120
    timestep_input_dim: int = 256
    timestep_hidden_size: int = 5376
    timestep_embed_dim: int = 2688
    rope_freq_dim: int = 16
    norm_eps: float = 1.0e-5
    min_video_rows: int = 37296
    opt_video_rows: int = 37296
    video_rows: int = 37296
    min_audio_rows: int = 414
    opt_audio_rows: int = 414
    audio_rows: int = 414
    min_text_rows: int = 1
    opt_text_rows: int = 128
    text_rows: int = 537
    padded_sequence_length: int = 38247
    max_timestep_count: int = 4
    context_parallel_size: int = 1
    first_block_cache: bool = False
    min_vsa_video_tiles: int | None = None
    opt_vsa_video_tiles: int | None = None
    max_vsa_video_tiles: int | None = None

    @property
    def sequence_length(self) -> int:
        return self.video_rows + self.audio_rows + self.text_rows

    @property
    def min_sequence_length(self) -> int:
        return self.min_video_rows + self.min_audio_rows + self.min_text_rows

    @property
    def opt_sequence_length(self) -> int:
        return self.opt_video_rows + self.opt_audio_rows + self.opt_text_rows

    @property
    def video_row_profile(self) -> tuple[int, int, int]:
        return self.min_video_rows, self.opt_video_rows, self.video_rows

    @property
    def audio_row_profile(self) -> tuple[int, int, int]:
        return self.min_audio_rows, self.opt_audio_rows, self.audio_rows

    @property
    def packed_row_profile(self) -> tuple[int, int, int]:
        return self.min_sequence_length, self.opt_sequence_length, self.sequence_length

    @staticmethod
    def _ceil_div(value: int, divisor: int) -> int:
        return (value + divisor - 1) // divisor

    @property
    def vsa_prefix_tile_profile(self) -> tuple[int, int, int]:
        """Segment-pure ``text | audio`` tile counts for the dynamic ABI."""

        return tuple(
            self._ceil_div(text, FASTH3_VSA_TILE_SIZE) + self._ceil_div(audio, FASTH3_VSA_TILE_SIZE)
            for text, audio in zip(self.text_row_profile, self.audio_row_profile, strict=True)
        )

    @property
    def text_row_profile(self) -> tuple[int, int, int]:
        return self.min_text_rows, self.opt_text_rows, self.text_rows

    def _vsa_video_tiles(self, video_rows: int) -> int:
        if video_rows % FASTH3_VSA_VIDEO_ROWS_PER_LATENT_FRAME:
            raise ValueError("MiniMax-H3 VSA video rows do not encode whole latent frames")
        latent_frames = video_rows // FASTH3_VSA_VIDEO_ROWS_PER_LATENT_FRAME
        return (
            self._ceil_div(latent_frames, FASTH3_VSA_VIDEO_TILE_ROWS)
            * self._ceil_div(24, FASTH3_VSA_VIDEO_TILE_HEIGHT)
            * self._ceil_div(42, FASTH3_VSA_VIDEO_TILE_WIDTH)
        )

    @property
    def vsa_video_tile_profile(self) -> tuple[int, int, int]:
        explicit = (
            self.min_vsa_video_tiles,
            self.opt_vsa_video_tiles,
            self.max_vsa_video_tiles,
        )
        if any(value is not None for value in explicit):
            if any(value is None for value in explicit):
                raise ValueError("MiniMax-H3 VSA tile profile must specify min, opt, and max")
            return tuple(int(value) for value in explicit)
        return tuple(self._vsa_video_tiles(rows) for rows in self.video_row_profile)

    @property
    def vsa_video_tile_abi_profile(self) -> tuple[int, int, int]:
        active = self.vsa_video_tile_profile
        return active[0], active[1], max(active[2], FASTH3_VSA_MAX_VIDEO_TILES)

    @property
    def vsa_total_tile_profile(self) -> tuple[int, int, int]:
        return tuple(
            prefix + video
            for prefix, video in zip(
                self.vsa_prefix_tile_profile, self.vsa_video_tile_abi_profile, strict=True
            )
        )

    @property
    def padding_rows(self) -> int:
        return self.padded_sequence_length - self.sequence_length

    @property
    def video_patch_dim(self) -> int:
        pt, ph, pw = self.video_patch
        return self.video_in_channels * pt * ph * pw

    @property
    def attention_size(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def adaln_table_rows(self) -> int:
        return self.max_timestep_count * 3

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError("MiniMax-H3 hidden_size and num_layers must be positive")
        if self.context_parallel_size != 1:
            raise ValueError("MiniMax-H3 native runtime currently requires context_parallel_size=1")
        if self.attention_size <= self.hidden_size:
            raise ValueError("MiniMax-H3 attention width must exceed its residual width")
        if not 1 <= self.min_text_rows <= self.opt_text_rows <= self.text_rows:
            raise ValueError("MiniMax-H3 text rows must satisfy 1 <= min <= opt <= max")
        if not 0 < self.min_video_rows <= self.opt_video_rows <= self.video_rows:
            raise ValueError("MiniMax-H3 video rows must satisfy 0 < min <= opt <= max")
        if not 0 < self.min_audio_rows <= self.opt_audio_rows <= self.audio_rows:
            raise ValueError("MiniMax-H3 audio rows must satisfy 0 < min <= opt <= max")
        if any(
            value is not None
            for value in (
                self.min_vsa_video_tiles,
                self.opt_vsa_video_tiles,
                self.max_vsa_video_tiles,
            )
        ):
            vsa_tiles = self.vsa_video_tile_profile
            if not 0 < vsa_tiles[0] <= vsa_tiles[1] <= vsa_tiles[2]:
                raise ValueError("MiniMax-H3 VSA video tiles must satisfy 0 < min <= opt <= max")
            if vsa_tiles[2] > FASTH3_VSA_MAX_VIDEO_TILES:
                raise ValueError("MiniMax-H3 VSA video tile profile exceeds the native CUDA ABI")
        if any(rows % 2 for rows in self.audio_row_profile):
            raise ValueError("MiniMax-H3 audio_rows must contain two equal stereo channels")
        if self.sequence_length != self.padded_sequence_length:
            raise ValueError(
                "MiniMax-H3 requires no packed-sequence padding: "
                "padded_sequence_length must equal the maximum packed sequence"
            )
        if self.rope_freq_dim * 6 > self.head_dim:
            raise ValueError("MiniMax-H3 rotary channels exceed head_dim")
        if not isinstance(self.first_block_cache, bool):
            raise ValueError("MiniMax-H3 first_block_cache must be a boolean")


SOL_ENGINE_1344X768_124F = MiniMaxH3Config()

# The released local pipeline aligns requested frame counts to ``17 * n + 5``.
# At 24 fps its supported 5--15 second endpoints are therefore 124 and 345
# frames.  Video tokens use 1,008 rows per latent frame at 1344x768; audio is
# packed as two stereo row groups of 207 through 575 latent frames.
SOL_ENGINE_1344X768_124_TO_345F = MiniMaxH3Config(
    min_video_rows=VIDEO_ROWS_MIN,
    opt_video_rows=VIDEO_ROWS_OPT,
    video_rows=VIDEO_ROWS_MAX,
    audio_rows=1150,
    text_rows=2641,
    padded_sequence_length=112367,
    min_vsa_video_tiles=FASTH3_VSA_MIN_VIDEO_TILES,
    opt_vsa_video_tiles=660,
    max_vsa_video_tiles=FASTH3_VSA_MAX_VIDEO_TILES,
)
