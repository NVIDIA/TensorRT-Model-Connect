# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native-runtime contracts for the public MiniMax-H3 Ref2VA workflow.

This module contains no framework runtime.  It is the executable specification
used by the bundle builder and by the C++ port for request validation, media
geometry, Qwen3-VL presentation/MRoPE, packed Omni-DiT rows, and per-row noise
levels.  Context-IR and Regenerate-2K are intentionally outside this contract.

The Python is build/test tooling only.  A serialized ModelConnect bundle must
implement the same operations in C++/CUDA and TensorRT-RTX; importing this file
at inference time is neither required nor supported.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from .fl2va_contract import PlanAbi, TensorAbi, resolve_canvas_size, validate_canvas


ReferenceKind = Literal["image", "video", "audio"]
QwenModality = Literal["text", "image", "video"]

MAX_IMAGES = 9
MAX_VIDEOS = 3
MAX_AUDIOS = 3
MAX_REFERENCES = 12
MIN_REFERENCE_DURATION_SECONDS = 2.0
MAX_REFERENCE_DURATION_SECONDS = 15.0
MAX_TOTAL_VIDEO_DURATION_SECONDS = 15.0
MAX_TOTAL_AUDIO_DURATION_SECONDS = 15.0

TARGET_FPS = 24.0
AUDIO_SAMPLE_RATE = 32_000
AUDIO_HOP_LENGTH = 800
AUDIO_CHANNELS = 2
VIDEO_VAE_SPATIAL_COMPRESSION = 16
VIDEO_PATCH_HEIGHT = 2
VIDEO_PATCH_WIDTH = 2
VIDEO_FRAMES_PER_CHUNK = 17
VIDEO_LATENTS_PER_CHUNK = 5
REFERENCE_IMAGE_SHORT_EDGE = 2048
CANVAS_MULTIPLE = 32
QWEN_VISION_PATCH_SIZE = 16
QWEN_VISION_MERGE_SIZE = 2
QWEN_VISION_TEMPORAL_PATCH_SIZE = 2
QWEN_VIDEO_SAMPLE_FPS = 2.0
QWEN_MAX_POSITION_EMBEDDINGS = 262_144

QWEN_VISION_START_TOKEN_ID = 151_652
QWEN_VISION_END_TOKEN_ID = 151_653
QWEN_IMAGE_PAD_TOKEN_ID = 151_655
QWEN_VIDEO_PAD_TOKEN_ID = 151_656

H3_VIDEO_TAG = 0
H3_TEXT_TAG = 1
H3_AUDIO_TAG = 2
CONDITION_VIDEO_TIMESTEP = 0.999
CONDITION_AUDIO_TIMESTEP = 1.0

_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32.0


@dataclass(frozen=True)
class ReferenceSpec:
    """Decoded public reference metadata, in request order.

    ``duration_seconds`` belongs to video and explicit-audio files.  A video
    soundtrack remains attached to its video and therefore does not increment
    the explicit-audio count or the twelve-file count.
    """

    kind: ReferenceKind
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    has_audio: bool = False

    def validate(self) -> None:
        if self.kind not in ("image", "video", "audio"):
            raise ValueError(f"MiniMax-H3 reference kind is invalid: {self.kind!r}")
        if not isinstance(self.has_audio, bool):
            raise ValueError("MiniMax-H3 has_audio must be a boolean")
        if self.has_audio and self.kind != "video":
            raise ValueError("Only a MiniMax-H3 video reference can carry a soundtrack")

        if self.kind in ("image", "video"):
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in (self.width, self.height)
            ):
                raise ValueError("MiniMax-H3 visual references need positive integer geometry")
            assert self.width is not None and self.height is not None
            if self.width > 4 * self.height or self.height > 4 * self.width:
                raise ValueError(
                    "MiniMax-H3 visual references must be within the public 1:4..4:1 aspect range"
                )
        elif self.width is not None or self.height is not None:
            raise ValueError("MiniMax-H3 audio references cannot carry pixel geometry")

        if self.kind == "image":
            if self.duration_seconds is not None:
                raise ValueError("MiniMax-H3 image references do not have a duration")
            return
        duration = self.duration_seconds
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or not MIN_REFERENCE_DURATION_SECONDS <= duration <= MAX_REFERENCE_DURATION_SECONDS
        ):
            raise ValueError("MiniMax-H3 video/audio references must each be 2..15 seconds long")


@dataclass(frozen=True)
class ReferenceRequestSummary:
    image_count: int
    video_count: int
    audio_count: int
    total_video_seconds: float
    total_audio_seconds: float
    audio_bearing_count: int


def validate_reference_request(
    references: Sequence[ReferenceSpec],
) -> ReferenceRequestSummary:
    """Fail closed on every public H3-Base Ref2VA file/duration limit."""

    if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
        raise ValueError("MiniMax-H3 references must be an ordered sequence")
    if not references:
        raise ValueError("MiniMax-H3 Ref2VA needs at least one reference")
    if len(references) > MAX_REFERENCES:
        raise ValueError(
            f"MiniMax-H3 accepts at most {MAX_REFERENCES} reference files, got {len(references)}"
        )
    for reference in references:
        if not isinstance(reference, ReferenceSpec):
            raise ValueError("MiniMax-H3 references must use ReferenceSpec metadata")
        reference.validate()

    image_count = sum(reference.kind == "image" for reference in references)
    video_count = sum(reference.kind == "video" for reference in references)
    audio_count = sum(reference.kind == "audio" for reference in references)
    if image_count > MAX_IMAGES:
        raise ValueError(f"MiniMax-H3 accepts at most {MAX_IMAGES} image references")
    if video_count > MAX_VIDEOS:
        raise ValueError(f"MiniMax-H3 accepts at most {MAX_VIDEOS} video references")
    if audio_count > MAX_AUDIOS:
        raise ValueError(f"MiniMax-H3 accepts at most {MAX_AUDIOS} explicit audio references")
    total_video = math.fsum(
        float(reference.duration_seconds) for reference in references if reference.kind == "video"
    )
    total_audio = math.fsum(
        float(reference.duration_seconds) for reference in references if reference.kind == "audio"
    )
    if total_video > MAX_TOTAL_VIDEO_DURATION_SECONDS:
        raise ValueError("MiniMax-H3 reference-video duration total exceeds 15 seconds")
    if total_audio > MAX_TOTAL_AUDIO_DURATION_SECONDS:
        raise ValueError("MiniMax-H3 explicit-audio duration total exceeds 15 seconds")
    return ReferenceRequestSummary(
        image_count=image_count,
        video_count=video_count,
        audio_count=audio_count,
        total_video_seconds=total_video,
        total_audio_seconds=total_audio,
        audio_bearing_count=audio_count
        + sum(reference.kind == "video" and reference.has_audio for reference in references),
    )


def resolve_reference_image_size(width: int, height: int) -> tuple[int, int]:
    """Return ``(height, width)`` at short-edge 2048 with no area cap.

    The round is Python/IEEE round-half-to-even, matching the released setup
    block.  The 1:4 and 4:1 endpoints resolve to 2048x8192 and therefore to the
    Qwen image processor's exact 16,777,216-pixel ceiling.
    """

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (width, height)
    ):
        raise ValueError("MiniMax-H3 reference image geometry must use positive integers")
    if width > 4 * height or height > 4 * width:
        raise ValueError("MiniMax-H3 reference image must be within 1:4 and 4:1")
    scale = REFERENCE_IMAGE_SHORT_EDGE / min(width, height)
    target_height = max(CANVAS_MULTIPLE, round(height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    target_width = max(CANVAS_MULTIPLE, round(width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return target_height, target_width


def resolve_reference_video_size(width: int, height: int) -> tuple[int, int]:
    """Put a video reference on the public 768p canvas of its own aspect."""

    return resolve_canvas_size(width, height)


def video_resample_source_indices(
    source_frames: int,
    source_fps: float,
    *,
    target_frames: int,
    target_fps: float = TARGET_FPS,
) -> tuple[int, ...]:
    """Reproduce the reference's constant-frame-rate drop/duplicate pass.

    The returned source row for each output slot is applied before LANCZOS
    spatial resize and is truncated to the generated frame count.
    """

    if (
        isinstance(source_frames, bool)
        or not isinstance(source_frames, int)
        or source_frames <= 0
        or isinstance(target_frames, bool)
        or not isinstance(target_frames, int)
        or target_frames <= 0
    ):
        raise ValueError("MiniMax-H3 video frame counts must be positive integers")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for value in (source_fps, target_fps)
    ):
        raise ValueError("MiniMax-H3 video frame rates must be finite and positive")
    if float(source_fps) == float(target_fps):
        return tuple(range(min(source_frames, target_frames)))

    scale = float(target_fps) / float(source_fps)
    slots = np.floor(np.arange(source_frames, dtype=np.float64) * scale + 0.5).astype(np.int64)
    end = math.floor(source_frames * scale + 0.5)
    counts = np.diff(slots, append=np.int64(end))
    if np.any(counts < 0):
        raise RuntimeError("MiniMax-H3 constant-frame-rate slots are not monotonic")
    result = np.repeat(np.arange(source_frames, dtype=np.int64), counts)
    return tuple(int(value) for value in result[:target_frames])


def snap_reference_video_frames_down(num_frames: int) -> int:
    """Snap a normalized reference down to ``17*n+5`` before VAE encode."""

    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0:
        raise ValueError("MiniMax-H3 reference video needs a positive frame count")
    chunks = max(1, (num_frames - VIDEO_LATENTS_PER_CHUNK) // VIDEO_FRAMES_PER_CHUNK)
    return chunks * VIDEO_FRAMES_PER_CHUNK + VIDEO_LATENTS_PER_CHUNK


def reference_video_latent_frames(num_frames: int) -> int:
    aligned = snap_reference_video_frames_down(num_frames)
    return (
        (aligned - VIDEO_LATENTS_PER_CHUNK) // VIDEO_FRAMES_PER_CHUNK
    ) * VIDEO_LATENTS_PER_CHUNK + 2


@dataclass(frozen=True)
class ReferenceVideoEncodeSchedule:
    """Native clip schedule matching ``AutoencoderKLMiniMaxH3._encode``."""

    normalized_frames: int
    snapped_frames: int
    clip_count: int
    repeated_tail_frames: int
    raw_posterior_frames: int
    dropped_tail_latents: int
    output_latent_frames: int


def reference_video_encode_schedule(num_frames: int) -> ReferenceVideoEncodeSchedule:
    """Plan exact 17-frame invocations and the one global three-token drop.

    The final reference clip has five real frames and twelve copies of the
    final frame.  The VideoVAE produces five posterior frames per clip, then
    the released wrapper drops three frames once after concatenating *all*
    clips (never once per clip).
    """

    snapped = snap_reference_video_frames_down(num_frames)
    if snapped > num_frames:
        raise ValueError("MiniMax-H3 reference video is too short for the released 17*n+5 schedule")
    clip_count = (snapped + VIDEO_FRAMES_PER_CHUNK - 1) // VIDEO_FRAMES_PER_CHUNK
    repeated = clip_count * VIDEO_FRAMES_PER_CHUNK - snapped
    raw = clip_count * VIDEO_LATENTS_PER_CHUNK
    dropped = 3
    output = raw - dropped
    if repeated != VIDEO_FRAMES_PER_CHUNK - VIDEO_LATENTS_PER_CHUNK:
        raise RuntimeError("MiniMax-H3 reference video clip padding invariant failed")
    if output != reference_video_latent_frames(num_frames):
        raise RuntimeError("MiniMax-H3 reference video latent schedule mismatch")
    return ReferenceVideoEncodeSchedule(
        normalized_frames=num_frames,
        snapped_frames=snapped,
        clip_count=clip_count,
        repeated_tail_frames=repeated,
        raw_posterior_frames=raw,
        dropped_tail_latents=dropped,
        output_latent_frames=output,
    )


def audio_latent_frames(num_samples: int) -> int:
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("MiniMax-H3 reference audio needs a positive sample count")
    return (num_samples + AUDIO_HOP_LENGTH - 1) // AUDIO_HOP_LENGTH


def qwen_video_condition_sample(
    num_frames: int,
    *,
    fps: float = TARGET_FPS,
    sample_fps: float = QWEN_VIDEO_SAMPLE_FPS,
    temporal_patch: int = QWEN_VISION_TEMPORAL_PATCH_SIZE,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Return the sampled frame indices and one timestamp per merged pair."""

    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0:
        raise ValueError("MiniMax-H3 Qwen video input needs a positive frame count")
    if (
        not math.isfinite(fps)
        or not math.isfinite(sample_fps)
        or fps <= 0
        or sample_fps <= 0
        or sample_fps > fps
    ):
        raise ValueError("MiniMax-H3 Qwen video sample rates are invalid")
    if (
        isinstance(temporal_patch, bool)
        or not isinstance(temporal_patch, int)
        or temporal_patch <= 0
    ):
        raise ValueError("MiniMax-H3 Qwen temporal patch must be a positive integer")

    stride = fps / sample_fps
    indices: list[int] = []
    cursor = 0.0
    while round(cursor) < num_frames:
        index = round(cursor)
        if not indices or index > indices[-1]:
            indices.append(index)
        cursor += stride
    if len(indices) < temporal_patch:
        minimum = round((temporal_patch - 1) * stride) + 1
        raise ValueError(
            f"MiniMax-H3 Qwen reads a reference video only when it has at least {minimum} frames"
        )
    timestamps = [index / sample_fps for index in range(len(indices))]
    timestamps.extend([timestamps[-1]] * (-len(timestamps) % temporal_patch))
    blocks = tuple(
        (timestamps[index] + timestamps[index + temporal_patch - 1]) / 2.0
        for index in range(0, len(timestamps), temporal_patch)
    )
    return tuple(indices), blocks


def qwen_patch_grid(height: int, width: int) -> tuple[int, int]:
    validate_canvas(height, width)
    if height % QWEN_VISION_PATCH_SIZE or width % QWEN_VISION_PATCH_SIZE:
        raise ValueError("MiniMax-H3 Qwen vision input is not patch aligned")
    return height // QWEN_VISION_PATCH_SIZE, width // QWEN_VISION_PATCH_SIZE


def qwen_merged_rows(height: int, width: int) -> int:
    grid_h, grid_w = qwen_patch_grid(height, width)
    merge = QWEN_VISION_MERGE_SIZE
    if grid_h % merge or grid_w % merge:
        raise ValueError("MiniMax-H3 Qwen vision input is not merge aligned")
    return (grid_h // merge) * (grid_w // merge)


@dataclass(frozen=True)
class PresentationPiece:
    """One literal text run or one Qwen visual-pad run."""

    modality: QwenModality
    text: str = ""
    height: int = 0
    width: int = 0

    @property
    def vision_rows(self) -> int:
        return 0 if self.modality == "text" else qwen_merged_rows(self.height, self.width)


def ref2va_presentation_blueprint(
    prompt: str,
    references: Sequence[ReferenceSpec],
    *,
    normalized_visual_sizes: Sequence[tuple[int, int]],
    normalized_video_frames: Sequence[int],
) -> tuple[PresentationPiece, ...]:
    """Build the ordered labels/timestamps/vision blocks before tokenization."""

    if not isinstance(prompt, str):
        raise ValueError("MiniMax-H3 prompt must be one string")
    validate_reference_request(references)
    visual_sizes = iter(normalized_visual_sizes)
    video_frames = iter(normalized_video_frames)
    pieces: list[PresentationPiece] = []
    counts = {"image": 0, "video": 0, "audio": 0}
    for reference in references:
        if reference.has_audio or reference.kind == "audio":
            counts["audio"] += 1
            pieces.append(PresentationPiece("text", f"<Audio {counts['audio']}>: "))
        if reference.kind == "image":
            counts["image"] += 1
            height, width = next(visual_sizes)
            pieces.append(PresentationPiece("text", f"<Picture {counts['image']}>: "))
            pieces.append(PresentationPiece("image", height=height, width=width))
        elif reference.kind == "video":
            counts["video"] += 1
            height, width = next(visual_sizes)
            pieces.append(PresentationPiece("text", f"<Video {counts['video']}>: "))
            _indices, timestamps = qwen_video_condition_sample(next(video_frames))
            for timestamp in timestamps:
                pieces.append(PresentationPiece("text", f"<{timestamp:.1f} seconds>"))
                pieces.append(PresentationPiece("video", height=height, width=width))
    try:
        next(visual_sizes)
    except StopIteration:
        pass
    else:
        raise ValueError("MiniMax-H3 visual-size metadata has unused entries")
    try:
        next(video_frames)
    except StopIteration:
        pass
    else:
        raise ValueError("MiniMax-H3 video-frame metadata has unused entries")
    pieces.append(PresentationPiece("text", prompt))
    return tuple(pieces)


@dataclass(frozen=True)
class MaterializedPresentation:
    input_ids: tuple[int, ...]
    qwen_token_types: tuple[int, ...]
    h3_token_tags: tuple[int, ...]
    vision_row_indices: tuple[int, ...]
    mrope_position_ids: np.ndarray
    image_grids: tuple[tuple[int, int, int], ...]
    video_grids: tuple[tuple[int, int, int], ...]


def qwen_mrope_position_ids(
    token_types: Sequence[int],
    *,
    image_grids: Sequence[tuple[int, int, int]],
    video_grids: Sequence[tuple[int, int, int]],
) -> np.ndarray:
    """Mirror ``Qwen3VLModel.get_rope_index`` for a single unpadded request."""

    if not token_types:
        raise ValueError("MiniMax-H3 Qwen presentation cannot be empty")
    if any(value not in (0, 1, 2) for value in token_types):
        raise ValueError("MiniMax-H3 Qwen token types must be text/image/video")
    image_iter = iter(image_grids)
    expanded_videos: list[tuple[int, int, int]] = []
    for grid_t, grid_h, grid_w in video_grids:
        if grid_t <= 0:
            raise ValueError("MiniMax-H3 Qwen video grid must have a positive temporal axis")
        expanded_videos.extend((1, grid_h, grid_w) for _ in range(grid_t))
    video_iter = iter(expanded_videos)

    axes = ([], [], [])
    current_position = 0
    cursor = 0
    while cursor < len(token_types):
        modality = token_types[cursor]
        end = cursor + 1
        while end < len(token_types) and token_types[end] == modality:
            end += 1
        length = end - cursor
        if modality == 0:
            values = range(current_position, current_position + length)
            for axis in axes:
                axis.extend(values)
            current_position += length
        else:
            grid_t, grid_h, grid_w = next(image_iter if modality == 1 else video_iter)
            merge = QWEN_VISION_MERGE_SIZE
            if grid_h % merge or grid_w % merge:
                raise ValueError("MiniMax-H3 Qwen grid is not spatial-merge aligned")
            merged_h, merged_w = grid_h // merge, grid_w // merge
            expected = grid_t * merged_h * merged_w
            if expected != length:
                raise ValueError(
                    f"MiniMax-H3 Qwen visual run has {length} pads but grid requires {expected}"
                )
            for temporal in range(grid_t):
                for row in range(merged_h):
                    for column in range(merged_w):
                        axes[0].append(current_position + temporal)
                        axes[1].append(current_position + row)
                        axes[2].append(current_position + column)
            current_position += max(grid_h, grid_w) // merge
        cursor = end

    for iterator, label in ((image_iter, "image"), (video_iter, "video")):
        try:
            next(iterator)
        except StopIteration:
            continue
        raise ValueError(f"MiniMax-H3 Qwen {label} grid metadata has unused entries")
    return np.asarray(axes, dtype=np.int32)


def materialize_ref2va_presentation(
    pieces: Sequence[PresentationPiece],
    tokenize: Callable[[str], Sequence[int]],
) -> MaterializedPresentation:
    """Tokenize a blueprint while preserving the two distinct modality maps."""

    ids: list[int] = []
    qwen_types: list[int] = []
    h3_tags: list[int] = []
    vision_rows: list[int] = []
    image_grids: list[tuple[int, int, int]] = []
    video_grids: list[tuple[int, int, int]] = []
    for piece in pieces:
        if piece.modality == "text":
            token_ids = tuple(int(value) for value in tokenize(piece.text))
            ids.extend(token_ids)
            qwen_types.extend([0] * len(token_ids))
            h3_tags.extend([H3_TEXT_TAG] * len(token_ids))
            continue
        grid_h, grid_w = qwen_patch_grid(piece.height, piece.width)
        pad_count = piece.vision_rows
        pad_id = QWEN_IMAGE_PAD_TOKEN_ID if piece.modality == "image" else QWEN_VIDEO_PAD_TOKEN_ID
        qwen_type = 1 if piece.modality == "image" else 2
        ids.append(QWEN_VISION_START_TOKEN_ID)
        qwen_types.append(0)
        h3_tags.append(H3_VIDEO_TAG)
        start = len(ids)
        ids.extend([pad_id] * pad_count)
        qwen_types.extend([qwen_type] * pad_count)
        h3_tags.extend([H3_VIDEO_TAG] * pad_count)
        vision_rows.extend(range(start, start + pad_count))
        ids.append(QWEN_VISION_END_TOKEN_ID)
        qwen_types.append(0)
        h3_tags.append(H3_VIDEO_TAG)
        grid = (1, grid_h, grid_w)
        (image_grids if piece.modality == "image" else video_grids).append(grid)
    if len(ids) > QWEN_MAX_POSITION_EMBEDDINGS:
        raise ValueError(
            f"MiniMax-H3 Ref2VA presentation has {len(ids)} rows, Qwen limit is "
            f"{QWEN_MAX_POSITION_EMBEDDINGS}"
        )
    mrope = qwen_mrope_position_ids(
        qwen_types,
        image_grids=image_grids,
        video_grids=video_grids,
    )
    return MaterializedPresentation(
        input_ids=tuple(ids),
        qwen_token_types=tuple(qwen_types),
        h3_token_tags=tuple(h3_tags),
        vision_row_indices=tuple(vision_rows),
        mrope_position_ids=mrope,
        image_grids=tuple(image_grids),
        video_grids=tuple(video_grids),
    )


@dataclass(frozen=True)
class EncodedReferenceGeometry:
    """Geometry emitted by the native video/audio VAE encoders."""

    kind: ReferenceKind
    latent_frames: int = 0
    latent_height: int = 0
    latent_width: int = 0
    audio_latents: int = 0

    @property
    def has_audio(self) -> bool:
        return self.audio_latents > 0

    @property
    def video_rows(self) -> int:
        if self.kind == "audio":
            return 0
        return (
            self.latent_frames
            * (self.latent_height // VIDEO_PATCH_HEIGHT)
            * (self.latent_width // VIDEO_PATCH_WIDTH)
        )

    @property
    def audio_rows(self) -> int:
        return self.audio_latents * AUDIO_CHANNELS

    def validate(self) -> None:
        if self.kind not in ("image", "video", "audio"):
            raise ValueError("MiniMax-H3 encoded reference kind is invalid")
        if self.kind == "audio":
            if any((self.latent_frames, self.latent_height, self.latent_width)):
                raise ValueError("MiniMax-H3 audio geometry cannot contain video latents")
            if self.audio_latents <= 0:
                raise ValueError("MiniMax-H3 audio reference must contain audio latents")
            return
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (self.latent_frames, self.latent_height, self.latent_width)
        ):
            raise ValueError("MiniMax-H3 visual reference latent geometry must be positive")
        if self.latent_height % VIDEO_PATCH_HEIGHT or self.latent_width % VIDEO_PATCH_WIDTH:
            raise ValueError("MiniMax-H3 reference latents are not transformer-patch aligned")
        if self.kind == "image" and self.latent_frames != 1:
            raise ValueError("MiniMax-H3 image references encode to exactly one latent frame")
        if self.audio_latents < 0:
            raise ValueError("MiniMax-H3 reference audio latent count cannot be negative")


@dataclass(frozen=True)
class Ref2VAPackedLayout:
    position_ids: np.ndarray
    token_tags: np.ndarray
    video_indices: np.ndarray
    audio_indices: np.ndarray
    text_indices: np.ndarray
    num_condition_video_rows: int
    num_condition_audio_rows: int

    @property
    def sequence_length(self) -> int:
        return int(self.position_ids.shape[0])


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> np.ndarray:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE


def _frame_position_grid(latent_height: int, latent_width: int) -> tuple[np.ndarray, np.ndarray]:
    sqrt_area = np.sqrt(latent_height * latent_width)
    height = _spatial_position_grid(latent_height, VIDEO_PATCH_HEIGHT, sqrt_area)
    width = _spatial_position_grid(latent_width, VIDEO_PATCH_WIDTH, sqrt_area)
    grid_h, grid_w = np.meshgrid(height, width, indexing="ij")
    return np.stack((grid_h.reshape(-1), grid_w.reshape(-1)), axis=-1), width


def _temporal_position_grid(num_latent_frames: int, origin: float) -> np.ndarray:
    result = np.empty((num_latent_frames,), dtype=np.float64)
    cursor = float(origin)
    for index in range(num_latent_frames):
        result[index] = cursor
        cursor += (
            _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
        )
    return result


def _fill_audio_positions(
    positions: np.ndarray,
    rows: slice,
    num_audio_latents: int,
    rotary_time: float,
    width_grid: np.ndarray,
) -> None:
    if num_audio_latents == 0:
        return
    time = rotary_time + np.arange(num_audio_latents, dtype=np.float64)
    positions[rows, 0] = np.tile(time, AUDIO_CHANNELS)
    positions[rows, 2] = np.concatenate(
        (
            np.full((num_audio_latents,), float(width_grid[0]), dtype=np.float64),
            np.full((num_audio_latents,), float(width_grid[-1]), dtype=np.float64),
        )
    )


def build_ref2va_packed_layout(
    text_token_tags: Sequence[int],
    references: Sequence[EncodedReferenceGeometry],
    *,
    target_latent_frames: int,
    target_latent_height: int,
    target_latent_width: int,
    target_audio_latents: int,
) -> Ref2VAPackedLayout:
    """Build ``[text | ordered reference blocks | target audio | target video]``."""

    if not text_token_tags or any(
        tag not in (H3_VIDEO_TAG, H3_TEXT_TAG) for tag in text_token_tags
    ):
        raise ValueError("MiniMax-H3 Ref2VA text tags must contain text/vision rows")
    for reference in references:
        if not isinstance(reference, EncodedReferenceGeometry):
            raise ValueError("MiniMax-H3 encoded references need explicit geometry")
        reference.validate()
    integer_values = (
        target_latent_frames,
        target_latent_height,
        target_latent_width,
        target_audio_latents,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in integer_values
    ):
        raise ValueError("MiniMax-H3 target latent geometry must be positive")
    if target_latent_height % VIDEO_PATCH_HEIGHT or target_latent_width % VIDEO_PATCH_WIDTH:
        raise ValueError("MiniMax-H3 target latent geometry is not patch aligned")

    text_rows = len(text_token_tags)
    condition_video_rows = sum(reference.video_rows for reference in references)
    condition_audio_rows = sum(reference.audio_rows for reference in references)
    target_video_rows = (
        target_latent_frames
        * (target_latent_height // VIDEO_PATCH_HEIGHT)
        * (target_latent_width // VIDEO_PATCH_WIDTH)
    )
    target_audio_rows = target_audio_latents * AUDIO_CHANNELS
    sequence_length = (
        text_rows
        + condition_video_rows
        + condition_audio_rows
        + target_audio_rows
        + target_video_rows
    )
    positions = np.zeros((sequence_length, 3), dtype=np.float64)
    positions[:text_rows, 0] = np.arange(text_rows, dtype=np.float64)
    target_frame_grid, target_width_grid = _frame_position_grid(
        target_latent_height, target_latent_width
    )

    video_indices: list[np.ndarray] = []
    audio_indices: list[np.ndarray] = []
    cursor = text_rows
    rotary_time = float(text_rows)
    for reference in references:
        if reference.kind == "image":
            rows = slice(cursor, cursor + reference.video_rows)
            cursor = rows.stop
            video_indices.append(np.arange(rows.start, rows.stop, dtype=np.int32))
            frame_grid, _width = _frame_position_grid(
                reference.latent_height, reference.latent_width
            )
            positions[rows, 0] = rotary_time
            positions[rows, 1:] = frame_grid
            rotary_time += 1.0
        elif reference.kind == "audio":
            rows = slice(cursor, cursor + reference.audio_rows)
            cursor = rows.stop
            audio_indices.append(np.arange(rows.start, rows.stop, dtype=np.int32))
            _fill_audio_positions(
                positions,
                rows,
                reference.audio_latents,
                rotary_time,
                target_width_grid,
            )
            rotary_time += float(reference.audio_latents)
        else:
            audio_rows = slice(cursor, cursor + reference.audio_rows)
            video_rows = slice(audio_rows.stop, audio_rows.stop + reference.video_rows)
            cursor = video_rows.stop
            if reference.audio_rows:
                audio_indices.append(np.arange(audio_rows.start, audio_rows.stop, dtype=np.int32))
            video_indices.append(np.arange(video_rows.start, video_rows.stop, dtype=np.int32))
            frame_grid, width_grid = _frame_position_grid(
                reference.latent_height, reference.latent_width
            )
            _fill_audio_positions(
                positions,
                audio_rows,
                reference.audio_latents,
                rotary_time,
                width_grid,
            )
            frame_time = _temporal_position_grid(reference.latent_frames, rotary_time)
            positions[video_rows, 0] = np.repeat(frame_time, frame_grid.shape[0])
            positions[video_rows, 1:] = np.tile(frame_grid, (reference.latent_frames, 1))
            video_span = 0.0
            for index in range(reference.latent_frames):
                video_span += (
                    _ROPE_FRAME_RESCALE
                    * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
                )
            rotary_time += max(float(reference.audio_latents), video_span)

    target_audio_start = cursor
    target_video_start = target_audio_start + target_audio_rows
    _fill_audio_positions(
        positions,
        slice(target_audio_start, target_video_start),
        target_audio_latents,
        rotary_time,
        target_width_grid,
    )
    target_time = _temporal_position_grid(target_latent_frames, rotary_time)
    positions[target_video_start:, 0] = np.repeat(target_time, target_frame_grid.shape[0])
    positions[target_video_start:, 1:] = np.tile(target_frame_grid, (target_latent_frames, 1))
    video_indices.append(np.arange(target_video_start, sequence_length, dtype=np.int32))
    audio_indices.append(np.arange(target_audio_start, target_video_start, dtype=np.int32))

    video_index_array = np.concatenate(video_indices)
    audio_index_array = np.concatenate(audio_indices)
    text_index_array = np.arange(text_rows, dtype=np.int32)
    tags = np.empty((sequence_length,), dtype=np.int32)
    tags[text_index_array] = np.asarray(text_token_tags, dtype=np.int32)
    tags[video_index_array] = H3_VIDEO_TAG
    tags[audio_index_array] = H3_AUDIO_TAG
    if cursor != target_audio_start:
        raise RuntimeError("MiniMax-H3 Ref2VA reference row cursor mismatch")
    return Ref2VAPackedLayout(
        position_ids=positions,
        token_tags=tags,
        video_indices=video_index_array,
        audio_indices=audio_index_array,
        text_indices=text_index_array,
        num_condition_video_rows=condition_video_rows,
        num_condition_audio_rows=condition_audio_rows,
    )


def build_row_timestep_indices(
    layout: Ref2VAPackedLayout,
    *,
    video_timestep: float,
    audio_timestep: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted unique timesteps, inverse indices, and AdaLN row indices."""

    row_timesteps = np.full((layout.sequence_length,), video_timestep, dtype=np.float32)
    video_condition = layout.video_indices[: layout.num_condition_video_rows]
    audio_condition = layout.audio_indices[: layout.num_condition_audio_rows]
    row_timesteps[video_condition] = max(float(video_timestep), CONDITION_VIDEO_TIMESTEP)
    row_timesteps[layout.audio_indices[layout.num_condition_audio_rows :]] = audio_timestep
    row_timesteps[audio_condition] = CONDITION_AUDIO_TIMESTEP
    unique, inverse = np.unique(row_timesteps, return_inverse=True)
    adaln = inverse.astype(np.int32) * 3 + layout.token_tags
    return unique.astype(np.float32), inverse.astype(np.int32), adaln.astype(np.int32)


@dataclass(frozen=True)
class Ref2VATimestepTable:
    """Fixed four-row AdaLN input plus the number of live sorted values."""

    values: np.ndarray
    count: int


def pad_ref2va_timesteps(
    unique_timesteps: Sequence[float], *, capacity: int = 4
) -> Ref2VATimestepTable:
    """Pad sorted unique row timesteps to the static AdaLN plan ABI.

    Unused rows repeat the final live timestep.  They are never addressed by
    ``timestep_indices``/``adaln_indices``, but deterministic repetition avoids
    feeding unspecified values into the fixed four-row precompute engine.
    """

    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("MiniMax-H3 Ref2VA timestep-table capacity must be positive")
    values = np.asarray(tuple(unique_timesteps), dtype=np.float32)
    if values.ndim != 1 or not 1 <= values.size <= capacity:
        raise ValueError(f"MiniMax-H3 Ref2VA needs 1..{capacity} unique row timesteps")
    if not np.all(np.isfinite(values)) or np.any(values[1:] <= values[:-1]):
        raise ValueError("MiniMax-H3 Ref2VA row timesteps must be finite and strictly sorted")
    padded = np.full((capacity,), values[-1], dtype=np.float32)
    padded[: values.size] = values
    return Ref2VATimestepTable(values=padded, count=int(values.size))


def reference_rng_draw_order(references: Sequence[EncodedReferenceGeometry]) -> tuple[str, ...]:
    """Describe request-generator draws; posterior seed-42 draws are separate."""

    visual = tuple(
        f"condition_{index}_{reference.kind}"
        for index, reference in enumerate(references)
        if reference.kind in ("image", "video")
    )
    return (*visual, "target_video", "target_audio")


# Full documented envelope.  It is intentionally explicit: a workstation may
# select a smaller optimization profile, but must then advertise that capacity
# instead of claiming the complete public Ref2VA limits.
REF2VA_MAX_IMAGE_VIDEO_ROWS = 9 * 16_384
REF2VA_MAX_VIDEO_REFERENCE_LATENT_FRAMES = 106
REF2VA_MAX_VIDEO_ROWS_PER_LATENT = 1_044
REF2VA_MAX_REFERENCE_VIDEO_ROWS = (
    REF2VA_MAX_VIDEO_REFERENCE_LATENT_FRAMES * REF2VA_MAX_VIDEO_ROWS_PER_LATENT
)
REF2VA_MAX_TARGET_VIDEO_ROWS = 106_488
REF2VA_MAX_ALL_VIDEO_ROWS = (
    REF2VA_MAX_IMAGE_VIDEO_ROWS + REF2VA_MAX_REFERENCE_VIDEO_ROWS + REF2VA_MAX_TARGET_VIDEO_ROWS
)
# Three separately padded clips can add at most two 800-sample tail frames per
# 15-second modality group.  Video soundtracks and explicit audios are separate
# groups, and all latents are stereo/channel-major.
REF2VA_MAX_REFERENCE_AUDIO_ROWS = 2 * (602 + 602)
REF2VA_MAX_TARGET_AUDIO_ROWS = 1_150
REF2VA_MAX_ALL_AUDIO_ROWS = REF2VA_MAX_REFERENCE_AUDIO_ROWS + REF2VA_MAX_TARGET_AUDIO_ROWS
REF2VA_MAX_TEXT_ROWS = QWEN_MAX_POSITION_EMBEDDINGS
REF2VA_MAX_PACKED_ROWS = (
    REF2VA_MAX_TEXT_ROWS + REF2VA_MAX_ALL_VIDEO_ROWS + REF2VA_MAX_ALL_AUDIO_ROWS
)


@dataclass(frozen=True)
class Ref2VADenoiserProfile:
    # The live public workflow permits an audio-only reference, so the video
    # minimum is the 5.167-second target on the explicit 960x544 canvas with no
    # conditioning video rows. Keep the text profile at one rather than baking
    # a tokenizer-dependent lower bound into the engine; request validation
    # still requires the complete presentation.
    min_video_rows: int = 18_870
    opt_video_rows: int = 44_592
    max_video_rows: int = REF2VA_MAX_ALL_VIDEO_ROWS
    min_audio_rows: int = 414
    opt_audio_rows: int = 414
    max_audio_rows: int = REF2VA_MAX_ALL_AUDIO_ROWS
    min_text_rows: int = 1
    opt_text_rows: int = 7_433
    max_text_rows: int = REF2VA_MAX_TEXT_ROWS

    @property
    def min_packed_rows(self) -> int:
        return self.min_video_rows + self.min_audio_rows + self.min_text_rows

    @property
    def opt_packed_rows(self) -> int:
        return self.opt_video_rows + self.opt_audio_rows + self.opt_text_rows

    @property
    def max_packed_rows(self) -> int:
        return self.max_video_rows + self.max_audio_rows + self.max_text_rows

    def validate(self) -> None:
        for label, values in (
            ("video", (self.min_video_rows, self.opt_video_rows, self.max_video_rows)),
            ("audio", (self.min_audio_rows, self.opt_audio_rows, self.max_audio_rows)),
            ("text", (self.min_text_rows, self.opt_text_rows, self.max_text_rows)),
        ):
            if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
                raise ValueError(f"MiniMax-H3 Ref2VA {label} profile must use integers")
            if not 1 <= values[0] <= values[1] <= values[2]:
                raise ValueError(
                    f"MiniMax-H3 Ref2VA {label} profile must satisfy min <= opt <= max"
                )
        if self.max_text_rows > QWEN_MAX_POSITION_EMBEDDINGS:
            raise ValueError("MiniMax-H3 Ref2VA text profile exceeds Qwen context")


def ref2va_denoiser_abi(profile: Ref2VADenoiserProfile = Ref2VADenoiserProfile()) -> PlanAbi:
    """Scatter/gather ABI of the dedicated ``transformer_ref`` plan."""

    profile.validate()
    video_shape = (
        (profile.min_video_rows, 96),
        (profile.opt_video_rows, 96),
        (profile.max_video_rows, 96),
    )
    audio_shape = (
        (profile.min_audio_rows, 32),
        (profile.opt_audio_rows, 32),
        (profile.max_audio_rows, 32),
    )
    text_shape = (
        (profile.min_text_rows, 5120),
        (profile.opt_text_rows, 5120),
        (profile.max_text_rows, 5120),
    )
    packed_shape = (
        (profile.min_packed_rows,),
        (profile.opt_packed_rows,),
        (profile.max_packed_rows,),
    )

    def binding(name: str, dtype: str, shapes: tuple[tuple[int, ...], ...]) -> TensorAbi:
        return TensorAbi(name, dtype, shapes[0], shapes[1], shapes[2])

    return PlanAbi(
        filename="ref2va_denoiser.plan",
        inputs=(
            binding("video_hidden_states", "float32", video_shape),
            binding("audio_hidden_states", "float32", audio_shape),
            binding("encoder_hidden_states", "float32", text_shape),
            binding(
                "position_ids",
                "float32",
                tuple(shape + (3,) for shape in packed_shape),
            ),
            binding("video_indices", "int32", tuple((shape[0],) for shape in video_shape)),
            binding("audio_indices", "int32", tuple((shape[0],) for shape in audio_shape)),
            binding("text_indices", "int32", tuple((shape[0],) for shape in text_shape)),
            binding("adaln_indices", "int32", packed_shape),
            binding("timestep_indices", "int32", packed_shape),
            *(
                TensorAbi(
                    f"block_modulation_{index}",
                    "bfloat16",
                    (12, 6, 5376),
                    (12, 6, 5376),
                    (12, 6, 5376),
                )
                for index in range(50)
            ),
            TensorAbi("final_modulation", "bfloat16", (4, 2, 5376), (4, 2, 5376), (4, 2, 5376)),
        ),
        outputs=(
            binding("video_velocity", "float32", video_shape),
            binding("audio_velocity", "float32", audio_shape),
        ),
    )
