# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time contracts for MiniMax-H3 first/last-frame conditioning.

The runtime consumes the plans described here without importing this module.
Keeping the public FL2VA geometry, Qwen presentation, and row accounting in a
small module makes it possible to validate a bundle before any multi-gigabyte
engine is built.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


CANVAS_MULTIPLE = 32
CANVAS_SHORT_EDGE = 768
CANVAS_MAX_PIXELS = 768 * 1344
VAE_SPATIAL_COMPRESSION = 16
VAE_TILE_SIZE = 256
VAE_TILE_MIN_OVERLAP = 64
VAE_TILE_OPT_BATCH = 28
VAE_TILE_MAX_BATCH = 33
VIDEO_PATCH_SIZE = (1, 2, 2)
QWEN_VISION_PATCH_SIZE = 16
QWEN_VISION_TEMPORAL_PATCH_SIZE = 2
QWEN_VISION_MERGE_SIZE = 2
QWEN_VISION_PATCH_WIDTH = 3 * QWEN_VISION_TEMPORAL_PATCH_SIZE * QWEN_VISION_PATCH_SIZE**2
QWEN_VISION_HIDDEN_SIZE = 1152
QWEN_TEXT_HIDDEN_SIZE = 5120
QWEN_PROMPT_MAX_TOKENS = 537
QWEN_LABEL_TOKENS = 6
QWEN_VISION_BOUNDARY_TOKENS = 2
QWEN_VISION_START_TOKEN_ID = 151652
QWEN_VISION_END_TOKEN_ID = 151653
QWEN_IMAGE_PAD_TOKEN_ID = 151655
MIN_FRAMES = 124
MAX_FRAMES = 345
FRAMES_PER_CHUNK = 17
LATENTS_PER_CHUNK = 5
AUDIO_LATENTS_PER_SECOND = 40
FPS = 24
KEYFRAME_NOISE_AUG = 0.999
KEYFRAME_ENCODE_SEED = 42
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
POSTERIOR_LOGVAR_MIN = -30.0
POSTERIOR_LOGVAR_MAX = 20.0
QWEN_PIXEL_MEAN = (0.5, 0.5, 0.5)
QWEN_PIXEL_STD = (0.5, 0.5, 0.5)
LATENTS_MEAN = (
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264,
)
LATENTS_STD = (
    1.2223774194717407,
    1.2767263650894165,
    1.6831774711608887,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.9653137922286987,
    1.0569885969161987,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.44988900423049927,
    0.7197399735450745,
    0.6936293244361877,
    2.961095094680786,
    2.7694199085235596,
    3.0496184825897217,
    2.1088054180145264,
    3.276226282119751,
    3.1627357006073,
    2.2816812992095947,
    2.6127843856811523,
)


@dataclass(frozen=True)
class TensorAbi:
    """One TensorRT binding, including every optimization-profile shape."""

    name: str
    dtype: str
    min_shape: tuple[int, ...]
    opt_shape: tuple[int, ...]
    max_shape: tuple[int, ...]


@dataclass(frozen=True)
class PlanAbi:
    """The complete externally visible binding contract of one native plan."""

    filename: str
    inputs: tuple[TensorAbi, ...]
    outputs: tuple[TensorAbi, ...]


@dataclass(frozen=True)
class VisionEncoderProfile:
    """Patch-row profile for one Qwen3-VL image or temporal video block."""

    min_patches: int = 2040
    opt_patches: int = 4032
    max_patches: int = 4176

    def validate(self) -> None:
        values = (self.min_patches, self.opt_patches, self.max_patches)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("MiniMax-H3 vision profile dimensions must be integers")
        if not 4 <= self.min_patches <= self.opt_patches <= self.max_patches:
            raise ValueError("MiniMax-H3 vision profile must satisfy 4 <= min <= opt <= max")
        if any(value % (QWEN_VISION_MERGE_SIZE**2) for value in values):
            raise ValueError("MiniMax-H3 vision profile dimensions must be divisible by four")


@dataclass(frozen=True)
class MultimodalTextProfile:
    """Sequence profile shared by T2VA, FL2VA, and Ref2VA conditioning."""

    min_sequence_length: int = 1
    opt_sequence_length: int = 1144
    max_sequence_length: int = 2641
    # TensorRT profiles cannot bind a zero-length tensor.  Text-only T2VA uses
    # one ignored dummy visual row with ``vision_count == 0``.
    min_vision_rows: int = 1
    opt_vision_rows: int = 1008
    max_vision_rows: int = 2088

    def validate(self) -> None:
        values = (
            self.min_sequence_length,
            self.opt_sequence_length,
            self.max_sequence_length,
            self.min_vision_rows,
            self.opt_vision_rows,
            self.max_vision_rows,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("MiniMax-H3 text profile dimensions must be integers")
        if (
            not 1
            <= self.min_sequence_length
            <= self.opt_sequence_length
            <= self.max_sequence_length
        ):
            raise ValueError("MiniMax-H3 text profile must satisfy 1 <= min <= opt <= max")
        if not 1 <= self.min_vision_rows <= self.opt_vision_rows <= self.max_vision_rows:
            raise ValueError("MiniMax-H3 visual-row profile must satisfy 1 <= min <= opt <= max")
        if self.min_vision_rows > self.min_sequence_length:
            raise ValueError("MiniMax-H3 minimum visual rows cannot exceed minimum sequence rows")
        if self.opt_vision_rows > self.opt_sequence_length:
            raise ValueError("MiniMax-H3 optimum visual rows cannot exceed optimum sequence rows")
        if self.max_vision_rows > self.max_sequence_length:
            raise ValueError("MiniMax-H3 maximum visual rows cannot exceed maximum sequence rows")


@dataclass(frozen=True)
class TileAxis:
    starts: tuple[int, ...]
    lengths: tuple[int, ...]
    overlaps: tuple[int, ...]


@dataclass(frozen=True)
class KeyframeResize:
    """Integer geometry for one exact LANCZOS resize/crop operation."""

    mode: str
    resized_height: int
    resized_width: int
    crop_top: int = 0
    crop_left: int = 0


@dataclass(frozen=True)
class PackedRows:
    text: int
    condition_video: int
    target_audio: int
    target_video: int

    @property
    def total(self) -> int:
        return self.text + self.condition_video + self.target_audio + self.target_video


def resolve_canvas_size(
    aspect_width: float,
    aspect_height: float,
    *,
    multiple: int = CANVAS_MULTIPLE,
    short_edge: int = CANVAS_SHORT_EDGE,
    max_pixels: int = CANVAS_MAX_PIXELS,
) -> tuple[int, int]:
    """Reproduce the public canvas resolver and return ``(height, width)``."""

    values = (aspect_width, aspect_height, multiple, short_edge, max_pixels)
    if any(isinstance(value, bool) for value in values):
        raise ValueError("MiniMax-H3 canvas geometry cannot use booleans")
    if not all(isinstance(value, (int, float)) for value in (aspect_width, aspect_height)):
        raise ValueError("MiniMax-H3 aspect dimensions must be numeric")
    if not all(isinstance(value, int) for value in (multiple, short_edge, max_pixels)):
        raise ValueError("MiniMax-H3 canvas limits must be integers")
    if not math.isfinite(aspect_width) or not math.isfinite(aspect_height):
        raise ValueError("MiniMax-H3 aspect dimensions must be finite")
    if (
        aspect_width <= 0
        or aspect_height <= 0
        or multiple <= 0
        or short_edge <= 0
        or max_pixels <= 0
    ):
        raise ValueError("MiniMax-H3 canvas geometry must be positive")

    ratio = aspect_width / aspect_height
    if not 0.25 <= ratio <= 4.0:
        raise ValueError(
            f"MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got {aspect_width}:{aspect_height}"
        )
    if ratio >= 1.0:
        width, height = short_edge * ratio, float(short_edge)
    else:
        width, height = float(short_edge), short_edge / ratio
    area = width * height
    if area > max_pixels:
        scale = math.sqrt(max_pixels / area)
        width *= scale
        height *= scale
    return (
        max(multiple, round(height / multiple) * multiple),
        max(multiple, round(width / multiple) * multiple),
    )


def validate_canvas(height: int, width: int) -> None:
    if isinstance(height, bool) or isinstance(width, bool):
        raise ValueError("MiniMax-H3 canvas dimensions must be integers")
    if not isinstance(height, int) or not isinstance(width, int) or height <= 0 or width <= 0:
        raise ValueError(f"MiniMax-H3 canvas must be positive, got {height}x{width}")
    if height % CANVAS_MULTIPLE or width % CANVAS_MULTIPLE:
        raise ValueError(
            f"MiniMax-H3 canvas must be a multiple of {CANVAS_MULTIPLE}, got {height}x{width}"
        )
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"MiniMax-H3 canvas must be within 1:4 and 4:1, got {height}x{width}")


def resolve_keyframe_anchors(*, has_first: bool, has_last: bool) -> tuple[str, ...]:
    """Return the reference's packed keyframe order and endpoint anchors."""

    if not isinstance(has_first, bool) or not isinstance(has_last, bool):
        raise ValueError("MiniMax-H3 keyframe presence flags must be booleans")
    anchors = tuple(
        anchor for anchor, present in (("first", has_first), ("last", has_last)) if present
    )
    if not anchors:
        raise ValueError("MiniMax-H3 FL2VA needs a first frame, a last frame, or both")
    return anchors


def keyframe_resize_geometry(
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
    *,
    packed_index: int,
) -> KeyframeResize:
    """Resolve the exact stretch/cover-crop geometry for one packed keyframe.

    Packed index zero is always the geometry anchor, including a last-only
    request.  It is stretched to the canvas.  Only the second keyframe is a
    cover-cropped follower.
    """

    values = (source_height, source_width, target_height, target_width, packed_index)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("MiniMax-H3 resize geometry must use integer values")
    if min(source_height, source_width, target_height, target_width) <= 0:
        raise ValueError("MiniMax-H3 resize dimensions must be positive")
    if packed_index not in (0, 1):
        raise ValueError("MiniMax-H3 FL2VA has at most two packed keyframes")
    validate_canvas(target_height, target_width)
    if (source_height, source_width) == (target_height, target_width):
        return KeyframeResize("identity", target_height, target_width)
    if packed_index == 0:
        return KeyframeResize("stretch", target_height, target_width)

    scale = max(target_width / source_width, target_height / source_height)
    resized_width = max(target_width, round(source_width * scale))
    resized_height = max(target_height, round(source_height * scale))
    return KeyframeResize(
        "cover_crop",
        resized_height,
        resized_width,
        crop_top=max(0, (resized_height - target_height) // 2),
        crop_left=max(0, (resized_width - target_width) // 2),
    )


def split_tile_axis(
    length: int,
    tile_size: int = VAE_TILE_SIZE,
    min_overlap: int = VAE_TILE_MIN_OVERLAP,
    alignment: int = VAE_SPATIAL_COMPRESSION,
) -> TileAxis:
    """Reproduce ``AutoencoderKLMiniMaxH3._split_tiles`` exactly."""

    if any(isinstance(value, bool) for value in (length, tile_size, min_overlap, alignment)):
        raise ValueError("MiniMax-H3 tile geometry must use integer values")
    if not all(isinstance(value, int) for value in (length, tile_size, min_overlap, alignment)):
        raise ValueError("MiniMax-H3 tile geometry must use integer values")
    if length <= 0 or tile_size <= 0 or alignment <= 0 or not 0 <= min_overlap < tile_size:
        raise ValueError("invalid MiniMax-H3 tile geometry")
    if length % alignment:
        raise ValueError(f"MiniMax-H3 tile axis {length} is not latent-aligned to {alignment}")
    if tile_size >= length:
        return TileAxis((0,), (length,), ())

    num_tiles = math.ceil(length / tile_size)
    while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
        num_tiles += 1

    overlaps = [min_overlap] * (num_tiles - 1)
    remaining = tile_size * num_tiles - sum(overlaps) - length
    if remaining % alignment:
        raise ValueError("MiniMax-H3 tile slack is not latent-aligned")
    for index in range(remaining // alignment):
        overlaps[index % (num_tiles - 1)] += alignment

    starts = [0]
    for overlap in overlaps:
        starts.append(starts[-1] + tile_size - overlap)
    result = TileAxis(tuple(starts), (tile_size,) * num_tiles, tuple(overlaps))
    if result.starts[-1] + result.lengths[-1] != length:
        raise ValueError("MiniMax-H3 tile split does not cover the canvas exactly")
    return result


def latent_tile_axis(length: int) -> TileAxis:
    """Map an exact pixel-space tile split onto the VAE latent grid.

    Stitching blends each overlap of ``n`` latent cells at positions
    ``i = 0..n-1`` as ``previous * (1 - i/n) + current * (i/n)``.  Vertical
    blending is applied before horizontal blending, then every non-final tile
    drops its trailing overlap exactly as the reference ``_stitch_tiles`` does.
    """

    pixels = split_tile_axis(length)
    return TileAxis(
        tuple(start // VAE_SPATIAL_COMPRESSION for start in pixels.starts),
        tuple(size // VAE_SPATIAL_COMPRESSION for size in pixels.lengths),
        tuple(overlap // VAE_SPATIAL_COMPRESSION for overlap in pixels.overlaps),
    )


def keyframe_tile_count(height: int, width: int) -> int:
    validate_canvas(height, width)
    if height < VAE_TILE_SIZE or width < VAE_TILE_SIZE:
        raise ValueError(
            f"MiniMax-H3 keyframe plan requires both canvas axes to be at least "
            f"{VAE_TILE_SIZE}, got {height}x{width}"
        )
    count = len(split_tile_axis(height).starts) * len(split_tile_axis(width).starts)
    if count > VAE_TILE_MAX_BATCH:
        raise ValueError(
            f"MiniMax-H3 keyframe needs {count} VAE tiles, plan capacity is {VAE_TILE_MAX_BATCH}"
        )
    return count


def qwen_vision_patch_rows(height: int, width: int) -> int:
    validate_canvas(height, width)
    if height % QWEN_VISION_PATCH_SIZE or width % QWEN_VISION_PATCH_SIZE:
        raise ValueError("MiniMax-H3 Qwen vision canvas is not patch-aligned")
    return (height // QWEN_VISION_PATCH_SIZE) * (width // QWEN_VISION_PATCH_SIZE)


def qwen_vision_token_rows(height: int, width: int) -> int:
    patches = qwen_vision_patch_rows(height, width)
    merge_unit = QWEN_VISION_MERGE_SIZE**2
    if patches % merge_unit:
        raise ValueError("MiniMax-H3 Qwen vision patches are not merge-aligned")
    return patches // merge_unit


def fl2va_text_rows(prompt_tokens: int, keyframes: int, *, height: int, width: int) -> int:
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens < 0:
        raise ValueError("MiniMax-H3 prompt token count must be a non-negative integer")
    if keyframes not in (1, 2):
        raise ValueError(f"MiniMax-H3 FL2VA needs one or two keyframes, got {keyframes}")
    per_keyframe = (
        QWEN_LABEL_TOKENS + QWEN_VISION_BOUNDARY_TOKENS + qwen_vision_token_rows(height, width)
    )
    return prompt_tokens + keyframes * per_keyframe


def fl2va_mrope_position_ids(
    prompt_tokens: int,
    keyframes: int,
    *,
    height: int,
    width: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Build Qwen3-VL's exact temporal/height/width position IDs.

    Qwen marks only ``image_pad`` tokens as image modality.  Each label and
    ``vision_start`` therefore forms a text run before the image grid, while a
    preceding ``vision_end`` joins the next label when two images are present.
    This mirrors ``Qwen3VLModel.get_rope_index`` without requiring Transformers
    at bundle-build time.
    """

    expected_rows = fl2va_text_rows(prompt_tokens, keyframes, height=height, width=width)
    llm_height = height // (QWEN_VISION_PATCH_SIZE * QWEN_VISION_MERGE_SIZE)
    llm_width = width // (QWEN_VISION_PATCH_SIZE * QWEN_VISION_MERGE_SIZE)
    axes: tuple[list[int], list[int], list[int]] = ([], [], [])
    current_position = 0

    def append_text(length: int) -> None:
        nonlocal current_position
        positions = range(current_position, current_position + length)
        for axis in axes:
            axis.extend(positions)
        current_position += length

    for index in range(keyframes):
        # The second text run begins with the previous image's vision_end.
        append_text(QWEN_LABEL_TOKENS + 1 + int(index > 0))
        for row in range(llm_height):
            for column in range(llm_width):
                axes[0].append(current_position)
                axes[1].append(current_position + row)
                axes[2].append(current_position + column)
        current_position += max(llm_height, llm_width)
    append_text(1 + prompt_tokens)  # final vision_end followed by the prompt

    result = tuple(tuple(axis) for axis in axes)
    if any(len(axis) != expected_rows for axis in result):
        raise RuntimeError("MiniMax-H3 FL2VA MRoPE row accounting mismatch")
    return result


def align_num_frames(num_frames: int) -> int:
    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0:
        raise ValueError("MiniMax-H3 frame count must be a positive integer")
    aligned = num_frames + (LATENTS_PER_CHUNK - num_frames % FRAMES_PER_CHUNK) % FRAMES_PER_CHUNK
    if not MIN_FRAMES <= aligned <= MAX_FRAMES:
        raise ValueError(
            f"MiniMax-H3 aligned frame count must be {MIN_FRAMES}..{MAX_FRAMES}, got {aligned}"
        )
    return aligned


def video_latent_frames(num_frames: int) -> int:
    aligned = align_num_frames(num_frames)
    return ((aligned - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK) * LATENTS_PER_CHUNK + 2


def audio_latent_frames(num_frames: int) -> int:
    aligned = align_num_frames(num_frames)
    return int(round(aligned / FPS * AUDIO_LATENTS_PER_SECOND))


def packed_rows(
    *,
    prompt_tokens: int,
    keyframes: int,
    height: int,
    width: int,
    num_frames: int,
) -> PackedRows:
    text = fl2va_text_rows(prompt_tokens, keyframes, height=height, width=width)
    rows_per_frame = (height // VAE_SPATIAL_COMPRESSION // VIDEO_PATCH_SIZE[1]) * (
        width // VAE_SPATIAL_COMPRESSION // VIDEO_PATCH_SIZE[2]
    )
    return PackedRows(
        text=text,
        condition_video=keyframes * rows_per_frame,
        target_audio=2 * audio_latent_frames(num_frames),
        target_video=video_latent_frames(num_frames) * rows_per_frame,
    )


def last_keyframe_rotary_time(text_rows: int, num_frames: int) -> float:
    """The reference's NumPy-summed last-frame rotary anchor."""

    latent_frames = video_latent_frames(num_frames)
    spans = [5.0 / 3.0 * (1, 4, 4, 4, 4)[index % 5] for index in range(latent_frames)]
    # Python's sum is deliberately not used: NumPy pairwise summation is the
    # reference.  The native runtime implements the same reduction order.
    import numpy as np

    return float(text_rows) + float(np.asarray(spans, dtype=np.float64).sum()) - 5.0 / 3.0


def keyframe_vae_encoder_abi() -> PlanAbi:
    """Tile-plan ABI for the exact released keyframe posterior recipe.

    The caller supplies row-major tiles after ImageNet normalization.  It must
    stitch all 48 posterior-parameter channels before splitting mean/logvar,
    clamp logvar to ``[-30, 20]``, sample with a fresh CPU generator seeded 42,
    round the sample through FP16, and apply ``LATENTS_MEAN/LATENTS_STD``.  Each
    keyframe repeats that fresh-generator recipe independently.
    """

    return PlanAbi(
        filename="fl2va_keyframe_vae_encoder.plan",
        inputs=(
            TensorAbi(
                "pixel_tiles",
                "float32",
                (1, 3, 1, VAE_TILE_SIZE, VAE_TILE_SIZE),
                (VAE_TILE_OPT_BATCH, 3, 1, VAE_TILE_SIZE, VAE_TILE_SIZE),
                (VAE_TILE_MAX_BATCH, 3, 1, VAE_TILE_SIZE, VAE_TILE_SIZE),
            ),
        ),
        outputs=(
            TensorAbi(
                "posterior_parameter_tiles",
                "float32",
                (
                    1,
                    48,
                    1,
                    VAE_TILE_SIZE // VAE_SPATIAL_COMPRESSION,
                    VAE_TILE_SIZE // VAE_SPATIAL_COMPRESSION,
                ),
                (
                    VAE_TILE_OPT_BATCH,
                    48,
                    1,
                    VAE_TILE_SIZE // VAE_SPATIAL_COMPRESSION,
                    VAE_TILE_SIZE // VAE_SPATIAL_COMPRESSION,
                ),
                (
                    VAE_TILE_MAX_BATCH,
                    48,
                    1,
                    VAE_TILE_SIZE // VAE_SPATIAL_COMPRESSION,
                    VAE_TILE_SIZE // VAE_SPATIAL_COMPRESSION,
                ),
            ),
        ),
    )


def vision_encoder_abi(profile: VisionEncoderProfile = VisionEncoderProfile()) -> PlanAbi:
    """One-image/frame dynamic-aspect Qwen3-VL vision-tower plan.

    A reference video is evaluated one temporal patch block at a time.  This
    preserves Qwen3-VL's ``cu_seqlens`` boundary (vision attention never crosses
    temporal blocks) while allowing the same weights and engine to serve FL2VA
    images, Ref2VA images, and Ref2VA video frames.
    """

    profile.validate()
    min_patches = profile.min_patches
    max_patches = profile.max_patches
    opt_patches = profile.opt_patches
    bindings = (
        TensorAbi(
            "pixel_values",
            "float32",
            (min_patches, QWEN_VISION_PATCH_WIDTH),
            (opt_patches, QWEN_VISION_PATCH_WIDTH),
            (max_patches, QWEN_VISION_PATCH_WIDTH),
        ),
        TensorAbi("interp_indices", "int32", (min_patches, 4), (opt_patches, 4), (max_patches, 4)),
        TensorAbi(
            "interp_weights", "float32", (min_patches, 4), (opt_patches, 4), (max_patches, 4)
        ),
        TensorAbi(
            "vision_position_ids", "int32", (min_patches, 2), (opt_patches, 2), (max_patches, 2)
        ),
    )

    def output(name: str) -> TensorAbi:
        return TensorAbi(
            name,
            "float32",
            (min_patches // 4, QWEN_TEXT_HIDDEN_SIZE),
            (opt_patches // 4, QWEN_TEXT_HIDDEN_SIZE),
            (max_patches // 4, QWEN_TEXT_HIDDEN_SIZE),
        )

    return PlanAbi(
        filename="vision_encoder.plan",
        inputs=bindings,
        outputs=tuple(
            output(name) for name in ("vision_embeds", "deepstack_0", "deepstack_1", "deepstack_2")
        ),
    )


def text_encoder_abi(
    profile: MultimodalTextProfile = MultimodalTextProfile(),
) -> PlanAbi:
    """Unified Qwen3-VL language-stack ABI for all three public workflows.

    Visual features remain compact at the ABI and are scattered through
    ``vision_row_indices``.  Text-only requests bind one dummy row, set
    ``vision_count`` to zero, and clear ``vision_mask``.  Thus the plan owns one
    copy of language weights instead of shipping separate T2VA and FL2VA
    language plans.
    """

    profile.validate()
    min_rows = profile.min_sequence_length
    opt_rows = profile.opt_sequence_length
    max_rows = profile.max_sequence_length
    min_vision = profile.min_vision_rows
    opt_vision = profile.opt_vision_rows
    max_vision = profile.max_vision_rows

    def rows(name: str, dtype: str, width: int | None = None) -> TensorAbi:
        suffix = () if width is None else (width,)
        return TensorAbi(name, dtype, (min_rows, *suffix), (opt_rows, *suffix), (max_rows, *suffix))

    def visual_rows(name: str, dtype: str, width: int | None = None) -> TensorAbi:
        suffix = () if width is None else (width,)
        return TensorAbi(
            name,
            dtype,
            (min_vision, *suffix),
            (opt_vision, *suffix),
            (max_vision, *suffix),
        )

    inputs = (
        rows("input_ids", "int32"),
        TensorAbi("mrope_position_ids", "int32", (3, min_rows), (3, opt_rows), (3, max_rows)),
        rows("vision_mask", "float32", 1),
        TensorAbi("vision_count", "int32", (1,), (1,), (1,)),
        visual_rows("vision_row_indices", "int32"),
        visual_rows("vision_embeds", "float32", QWEN_TEXT_HIDDEN_SIZE),
        visual_rows("deepstack_0", "float32", QWEN_TEXT_HIDDEN_SIZE),
        visual_rows("deepstack_1", "float32", QWEN_TEXT_HIDDEN_SIZE),
        visual_rows("deepstack_2", "float32", QWEN_TEXT_HIDDEN_SIZE),
    )
    return PlanAbi(
        filename="text_encoder.plan",
        inputs=inputs,
        outputs=(rows("encoder_hidden_states", "float32", QWEN_TEXT_HIDDEN_SIZE),),
    )
