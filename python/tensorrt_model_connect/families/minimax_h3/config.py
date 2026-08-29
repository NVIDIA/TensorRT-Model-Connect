# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated native TensorRT profiles for MiniMax-H3.

The default profile is the 124-frame, 1344x768 shape used by the public
Sol-Engine H3 benchmark. Structural row counts are explicit because prompt
packing is part of the engine ABI and must match the Hugging Face reference.
"""

from __future__ import annotations

from dataclasses import dataclass


TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES = 96 << 30
ADALN_PRECOMPUTE_DEFAULT_WORKSPACE_BYTES = 64 << 30
DENOISER_DEFAULT_WORKSPACE_BYTES = 96 << 30
VAE_TILE_DECODER_DEFAULT_WORKSPACE_BYTES = 96 << 30
AUDIO_VAE_DECODER_DEFAULT_WORKSPACE_BYTES = 32 << 30
AUDIO_VAE_ENCODER_DEFAULT_WORKSPACE_BYTES = 32 << 30
VISION_CONDITIONER_DEFAULT_WORKSPACE_BYTES = 96 << 30
VAE_ENCODER_DEFAULT_WORKSPACE_BYTES = 96 << 30

MINIMAX_H3_WORKFLOWS = ("t2va", "fl2va", "ref2va")

# The released 1344x768 FL2VA canvas is 48x84 in video-VAE latent space.
# Omni-DiT patchifies it by 2x2, so each first/last-frame anchor contributes
# exactly 24x42 = 1,008 live attention rows.  Keep the supported counts
# explicit: accepting an arbitrary number here would claim a model mode the
# released FL2VA checkpoint does not provide.
FL2VA_KEYFRAME_COUNTS = (0, 1, 2)
FL2VA_KEYFRAME_ROWS_1344X768 = 1008
FL2VA_TRANSFORMER_CHECKPOINT_SUBFOLDER = "transformer"

# Ref2VA model-card maxima, converted to the row domains consumed by the
# released transformer_ref checkpoint. Nine 2048-short-edge, 4:1 images use
# 9 * (2048 / 32) * (8192 / 32) = 147,456 video rows. Three video references
# sharing the documented 15-second limit can contribute at most 106 VAE
# latent frames after the released 17n+5 chunking. The canvas resolver rounds
# after applying its area cap; 576x1856 is therefore the true maximum at 1,044
# rows per frame, not the 1,008 rows of the default 768x1344 canvas, for
# 106 * 1,044 = 110,664 video rows. Reference audio can contain three standalone
# clips plus the soundtracks of three videos. Each clip is resampled separately
# and right-padded separately to the 800-sample audio-VAE hop. The nested
# resample/sample and hop ceilings reduce to `ceil(40 * duration)` per file.
# With three files per group, 15 seconds therefore needs at most 602 latents
# per channel, not the unpadded 600; both groups need 2,408 rows.
REF2VA_TRANSFORMER_CHECKPOINT_SUBFOLDER = "transformer_ref"
REF2VA_MAX_TEXT_ROWS = 262144
# The official Ref2VA input specification permits audio-only references, so
# the denoiser profile must accept target video rows with no condition video.
REF2VA_MIN_CONDITION_VIDEO_ROWS = 0
REF2VA_OPT_CONDITION_VIDEO_ROWS = 4096
REF2VA_MAX_IMAGE_CONDITION_VIDEO_ROWS = 147456
REF2VA_MAX_VIDEO_LATENT_FRAMES = 106
REF2VA_MAX_VIDEO_ROWS_PER_LATENT_FRAME = 1044
REF2VA_MAX_VIDEO_CONDITION_VIDEO_ROWS = (
    REF2VA_MAX_VIDEO_LATENT_FRAMES * REF2VA_MAX_VIDEO_ROWS_PER_LATENT_FRAME
)
REF2VA_MAX_CONDITION_VIDEO_ROWS = (
    REF2VA_MAX_IMAGE_CONDITION_VIDEO_ROWS + REF2VA_MAX_VIDEO_CONDITION_VIDEO_ROWS
)
REF2VA_MAX_STANDALONE_AUDIO_ROWS = 1204
REF2VA_MAX_VIDEO_SOUNDTRACK_ROWS = 1204
REF2VA_MAX_CONDITION_AUDIO_ROWS = (
    REF2VA_MAX_STANDALONE_AUDIO_ROWS + REF2VA_MAX_VIDEO_SOUNDTRACK_ROWS
)
MINIMAX_H3_NATIVE_PLUGIN_SECTION = "minimax_h3_native_plugin_so"
MINIMAX_H3_NATIVE_PLUGIN_FILENAME = "libtrtmc_minimax_h3_native_plugin.so"
MINIMAX_H3_NATIVE_PLUGIN_ABI = 1
MINIMAX_H3_NATIVE_PLUGIN_IDENTITY = "trtmc.minimax_h3.native_plugin:aten-ops:1"
REF2VA_VISION_PLAN_LAYOUT = "split-image-video-v1"
REF2VA_LANGUAGE_ATTENTION_IMPLEMENTATION = "tensorrt-bf16-iattention-v1"
REF2VA_LANGUAGE_ATTENTION_PRECISION = "bf16"
REF2VA_LANGUAGE_Q_PRE_SCALE_PRECISION = "bf16"
REF2VA_IMAGE_VISION_ATTENTION_IMPLEMENTATION = "aten-bf16-sdpa-v1"
REF2VA_IMAGE_VISION_ATTENTION_PRECISION = "bf16"
REF2VA_IMAGE_VISION_ATTENTION_SCALE = "fp64:0x1.e2b7dddfefa66p-4"
REF2VA_IMAGE_VISION_LINEAR_IMPLEMENTATION = "aten-bf16-linear-v1"
REF2VA_IMAGE_VISION_LINEAR_COUNT = 116
REF2VA_IMAGE_VISION_LAYER_NORM_IMPLEMENTATION = "aten-bf16-layer-norm-v1"
REF2VA_IMAGE_VISION_LAYER_NORM_COUNT = 58
REF2VA_IMAGE_VISION_PATCH_IMPLEMENTATION = "aten-bf16-conv3d-v1"
REF2VA_IMAGE_VISION_PATCH_PRECISION = "bf16"
REF2VA_IMAGE_VISION_PATCH_INPUT_SHAPE = (-1, 1536)
REF2VA_IMAGE_VISION_PATCH_WEIGHT_SHAPE = (1152, 3, 2, 16, 16)
REF2VA_IMAGE_VISION_PATCH_BIAS_SHAPE = (1152,)
REF2VA_IMAGE_VISION_PATCH_KERNEL = (2, 16, 16)
REF2VA_IMAGE_VISION_PATCH_STRIDE = (2, 16, 16)
REF2VA_IMAGE_VISION_PATCH_OUTPUT_SHAPE = (-1, 1152)
REF2VA_VIDEO_VISION_ATTENTION_IMPLEMENTATION = "tensorrt-fp16-iattention-v1"
REF2VA_VIDEO_VISION_ATTENTION_PRECISION = "fp16"
REF2VA_VIDEO_VISION_Q_PRE_SCALE_PRECISION = "fp16"
REF2VA_IMAGE_VISION_PATCH_PROFILE = (16384, 16384, 65536)
REF2VA_VIDEO_VISION_PATCH_PROFILE = (2304, 4032, 4176)

DEFAULT_WORKSPACE_LIMIT_BYTES = {
    "text_encoder.plan": TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES,
    "adaln_precompute.plan": ADALN_PRECOMPUTE_DEFAULT_WORKSPACE_BYTES,
    "denoiser.plan": DENOISER_DEFAULT_WORKSPACE_BYTES,
    "vae_tile_decoder.plan": VAE_TILE_DECODER_DEFAULT_WORKSPACE_BYTES,
    "audio_vae_decoder.plan": AUDIO_VAE_DECODER_DEFAULT_WORKSPACE_BYTES,
}

FL2VA_PLAN_FILENAMES = (
    "language_conditioner.plan",
    "vision_conditioner.plan",
    "vae_encoder_tile_t1.plan",
    "adaln_precompute.plan",
    "fl2va_denoiser.plan",
    "vae_tile_decoder.plan",
    "audio_vae_decoder.plan",
)

FL2VA_DEFAULT_WORKSPACE_LIMIT_BYTES = {
    "language_conditioner.plan": TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES,
    "vision_conditioner.plan": VISION_CONDITIONER_DEFAULT_WORKSPACE_BYTES,
    "vae_encoder_tile_t1.plan": VAE_ENCODER_DEFAULT_WORKSPACE_BYTES,
    "adaln_precompute.plan": ADALN_PRECOMPUTE_DEFAULT_WORKSPACE_BYTES,
    "fl2va_denoiser.plan": DENOISER_DEFAULT_WORKSPACE_BYTES,
    "vae_tile_decoder.plan": VAE_TILE_DECODER_DEFAULT_WORKSPACE_BYTES,
    "audio_vae_decoder.plan": AUDIO_VAE_DECODER_DEFAULT_WORKSPACE_BYTES,
}

REF2VA_PLAN_FILENAMES = (
    "language_conditioner.plan",
    "vision_conditioner_image.plan",
    "vision_conditioner_video.plan",
    "vae_encoder_tile_t1.plan",
    "vae_encoder_tile_t17.plan",
    "audio_vae_encoder.plan",
    "adaln_precompute.plan",
    "ref2va_denoiser.plan",
    "vae_tile_decoder.plan",
    "audio_vae_decoder.plan",
)

REF2VA_DEFAULT_WORKSPACE_LIMIT_BYTES = {
    "language_conditioner.plan": TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES,
    "vision_conditioner_image.plan": VISION_CONDITIONER_DEFAULT_WORKSPACE_BYTES,
    "vision_conditioner_video.plan": VISION_CONDITIONER_DEFAULT_WORKSPACE_BYTES,
    "vae_encoder_tile_t1.plan": VAE_ENCODER_DEFAULT_WORKSPACE_BYTES,
    "vae_encoder_tile_t17.plan": VAE_ENCODER_DEFAULT_WORKSPACE_BYTES,
    "audio_vae_encoder.plan": AUDIO_VAE_ENCODER_DEFAULT_WORKSPACE_BYTES,
    "adaln_precompute.plan": ADALN_PRECOMPUTE_DEFAULT_WORKSPACE_BYTES,
    "ref2va_denoiser.plan": DENOISER_DEFAULT_WORKSPACE_BYTES,
    "vae_tile_decoder.plan": VAE_TILE_DECODER_DEFAULT_WORKSPACE_BYTES,
    "audio_vae_decoder.plan": AUDIO_VAE_DECODER_DEFAULT_WORKSPACE_BYTES,
}

FL2VA_PROCESSOR_ASSET_SECTIONS = (
    "processor/preprocessor_config.json",
    "processor/video_preprocessor_config.json",
)

FIRST_BLOCK_CACHE_DENOISER_PLAN_FILENAMES = (
    "denoiser_head.plan",
    "denoiser_tail.plan",
    "denoiser_finish.plan",
)


def native_plan_filenames(*, first_block_cache: bool, workflow: str = "t2va") -> tuple[str, ...]:
    """Return the exact plan set selected by the native denoiser profile."""

    if not isinstance(first_block_cache, bool):
        raise ValueError("MiniMax-H3 first_block_cache must be a boolean")
    if workflow not in MINIMAX_H3_WORKFLOWS:
        raise ValueError(f"Unsupported MiniMax-H3 workflow: {workflow!r}")
    if workflow == "ref2va":
        if first_block_cache:
            raise ValueError("MiniMax-H3 Ref2VA does not support first_block_cache")
        return REF2VA_PLAN_FILENAMES
    if workflow == "fl2va":
        if first_block_cache:
            raise ValueError("MiniMax-H3 FL2VA does not support first_block_cache")
        return FL2VA_PLAN_FILENAMES
    denoiser_plans = (
        FIRST_BLOCK_CACHE_DENOISER_PLAN_FILENAMES if first_block_cache else ("denoiser.plan",)
    )
    return (
        "text_encoder.plan",
        "adaln_precompute.plan",
        *denoiser_plans,
        "vae_tile_decoder.plan",
        "audio_vae_decoder.plan",
    )


def default_workspace_limit_bytes(
    *, first_block_cache: bool, workflow: str = "t2va"
) -> dict[str, int]:
    """Return per-plan tactic workspace limits for one denoiser layout."""

    filenames = native_plan_filenames(
        first_block_cache=first_block_cache,
        workflow=workflow,
    )
    if workflow == "fl2va":
        return {filename: FL2VA_DEFAULT_WORKSPACE_LIMIT_BYTES[filename] for filename in filenames}
    if workflow == "ref2va":
        return {filename: REF2VA_DEFAULT_WORKSPACE_LIMIT_BYTES[filename] for filename in filenames}
    return {
        filename: (
            DENOISER_DEFAULT_WORKSPACE_BYTES
            if filename.startswith("denoiser_") or filename == "denoiser.plan"
            else DEFAULT_WORKSPACE_LIMIT_BYTES[filename]
        )
        for filename in filenames
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
    video_rows: int = 37296
    audio_rows: int = 414
    text_rows: int = 537
    min_text_rows: int = 1
    # Two 1344x768 Qwen vision blocks contribute 2 * 1,008 rows before
    # picture labels and prompt tokens.  A 4,096-row build envelope covers
    # both plus the current 537-row operating point with label/prompt headroom.
    max_text_rows: int = 4096
    ref2va_min_text_rows: int = 1
    ref2va_opt_text_rows: int = 8192
    ref2va_max_text_rows: int = REF2VA_MAX_TEXT_ROWS
    ref2va_min_condition_video_rows: int = REF2VA_MIN_CONDITION_VIDEO_ROWS
    ref2va_opt_condition_video_rows: int = REF2VA_OPT_CONDITION_VIDEO_ROWS
    ref2va_max_condition_video_rows: int = REF2VA_MAX_CONDITION_VIDEO_ROWS
    ref2va_min_condition_audio_rows: int = 0
    ref2va_opt_condition_audio_rows: int = 0
    ref2va_max_condition_audio_rows: int = REF2VA_MAX_CONDITION_AUDIO_ROWS
    padded_sequence_length: int = 38247
    max_timestep_count: int = 4
    context_parallel_size: int = 1
    first_block_cache: bool = False

    @property
    def sequence_length(self) -> int:
        return self.video_rows + self.audio_rows + self.text_rows

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

    def fl2va_condition_rows(self, keyframe_count: int) -> int:
        """Return the exact number of packed condition rows for one FL2VA mode."""

        if (
            not isinstance(keyframe_count, int)
            or isinstance(keyframe_count, bool)
            or keyframe_count not in FL2VA_KEYFRAME_COUNTS
        ):
            raise ValueError(
                f"MiniMax-H3 FL2VA keyframe_count must be 0, 1, or 2, got {keyframe_count!r}"
            )
        return keyframe_count * FL2VA_KEYFRAME_ROWS_1344X768

    def fl2va_video_input_rows(self, keyframe_count: int) -> int:
        """Rows projected by ``proj_in``: pinned conditions followed by target video."""

        return self.fl2va_condition_rows(keyframe_count) + self.video_rows

    def fl2va_sequence_length(self, text_rows: int, keyframe_count: int) -> int:
        """Live padless rows in ``text | conditions | audio | target video`` order."""

        if not isinstance(text_rows, int) or isinstance(text_rows, bool):
            raise ValueError("MiniMax-H3 FL2VA text_rows must be an integer")
        if not self.min_text_rows <= text_rows <= self.max_text_rows:
            raise ValueError(
                "MiniMax-H3 FL2VA text_rows is outside the configured dynamic range: "
                f"rows={text_rows}, range=[{self.min_text_rows}, {self.max_text_rows}]"
            )
        return (
            text_rows
            + self.fl2va_condition_rows(keyframe_count)
            + self.audio_rows
            + self.video_rows
        )

    def ref2va_sequence_length(
        self,
        text_rows: int,
        condition_video_rows: int,
        condition_audio_rows: int,
    ) -> int:
        """Live rows in one request-ordered, padless Ref2VA attention document."""

        values = {
            "text_rows": (
                text_rows,
                self.ref2va_min_text_rows,
                self.ref2va_max_text_rows,
            ),
            "condition_video_rows": (
                condition_video_rows,
                self.ref2va_min_condition_video_rows,
                self.ref2va_max_condition_video_rows,
            ),
            "condition_audio_rows": (
                condition_audio_rows,
                self.ref2va_min_condition_audio_rows,
                self.ref2va_max_condition_audio_rows,
            ),
        }
        for name, (value, minimum, maximum) in values.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"MiniMax-H3 Ref2VA {name} must be an integer")
            if not minimum <= value <= maximum:
                raise ValueError(
                    f"MiniMax-H3 Ref2VA {name} is outside the configured dynamic range: "
                    f"rows={value}, range=[{minimum}, {maximum}]"
                )
        return (
            text_rows
            + condition_video_rows
            + condition_audio_rows
            + self.audio_rows
            + self.video_rows
        )

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError("MiniMax-H3 hidden_size and num_layers must be positive")
        if self.context_parallel_size != 1:
            raise ValueError("MiniMax-H3 native runtime currently requires context_parallel_size=1")
        if self.attention_size <= self.hidden_size:
            raise ValueError("MiniMax-H3 attention width must exceed its residual width")
        if (
            not isinstance(self.min_text_rows, int)
            or isinstance(self.min_text_rows, bool)
            or not isinstance(self.max_text_rows, int)
            or isinstance(self.max_text_rows, bool)
            or not 1 <= self.min_text_rows <= self.text_rows <= self.max_text_rows
        ):
            raise ValueError(
                "MiniMax-H3 dynamic text rows must satisfy "
                "1 <= min_text_rows <= text_rows <= max_text_rows"
            )
        ref2va_ranges = (
            (
                "text",
                self.ref2va_min_text_rows,
                self.ref2va_opt_text_rows,
                self.ref2va_max_text_rows,
                1,
            ),
            (
                "condition video",
                self.ref2va_min_condition_video_rows,
                self.ref2va_opt_condition_video_rows,
                self.ref2va_max_condition_video_rows,
                0,
            ),
            (
                "condition audio",
                self.ref2va_min_condition_audio_rows,
                self.ref2va_opt_condition_audio_rows,
                self.ref2va_max_condition_audio_rows,
                0,
            ),
        )
        for name, minimum, optimum, maximum, floor in ref2va_ranges:
            if (
                any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in (minimum, optimum, maximum)
                )
                or not floor <= minimum <= optimum <= maximum
            ):
                raise ValueError(
                    "MiniMax-H3 Ref2VA dynamic rows must satisfy "
                    f"{floor} <= min <= opt <= max for {name} rows"
                )
        required_ref2va_maxima = {
            "text": (self.ref2va_max_text_rows, REF2VA_MAX_TEXT_ROWS),
            "condition video": (
                self.ref2va_max_condition_video_rows,
                REF2VA_MAX_CONDITION_VIDEO_ROWS,
            ),
            "condition audio": (
                self.ref2va_max_condition_audio_rows,
                REF2VA_MAX_CONDITION_AUDIO_ROWS,
            ),
        }
        narrowed = {
            name: (actual, required)
            for name, (actual, required) in required_ref2va_maxima.items()
            if actual < required
        }
        if narrowed:
            raise ValueError(
                "MiniMax-H3 Ref2VA profiles may not silently narrow the model-card maxima: "
                f"{narrowed}"
            )
        if self.sequence_length != self.padded_sequence_length:
            raise ValueError("MiniMax-H3 single-device profile requires no packed-sequence padding")
        if self.rope_freq_dim * 6 > self.head_dim:
            raise ValueError("MiniMax-H3 rotary channels exceed head_dim")
        if not isinstance(self.first_block_cache, bool):
            raise ValueError("MiniMax-H3 first_block_cache must be a boolean")


SOL_ENGINE_1344X768_124F = MiniMaxH3Config()
