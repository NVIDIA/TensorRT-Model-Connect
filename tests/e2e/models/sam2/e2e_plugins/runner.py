# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute a caller-built SAM2 bundle through its public C ABI."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import stat
import struct
import zlib
from pathlib import Path

import numpy as np

from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput, StageSpec

_ABI_VERSION = 1
_FRAME_COUNT = 5
_HEIGHT = 1280
_WIDTH = 1088
_FRAME_PIXELS = _HEIGHT * _WIDTH
_RGB_BYTES = _FRAME_PIXELS * 3
_RUNTIME_LIBRARY = "libtrtmc_model_sam2.so"
_PUBLIC_CORE_VARIANT = "public_sam2_1_small_with_synthetic_bbox_v1"
_PLAN_SECTIONS = (
    "engine_plan",
    "sam2_prompt_engine_plan",
    "sam2_recurrent_h1_engine_plan",
    "sam2_recurrent_h2_engine_plan",
    "sam2_recurrent_h3_engine_plan",
    "sam2_recurrent_h4_engine_plan",
)

_RUN_MATERIALIZE_MASKS_HOST = 1
_MASK_MEMORY_HOST = 1

_RGB8_SHA256 = (
    "0bcadde0e5a6f8ba04f79c44f064c5b00d3cd1b250e2f2f3bbf10ef0630a9ce9",
    "0abfd57f9e3886a8c3068bf6bcc353b26d1e3a8a43819a80dfeb00f309b24ec3",
    "9166cc263c3edb262065fa3b98ee062cbf6d781dd656bae13def7f4141b7d025",
    "77525faadfc8a607e4e1556135887caaddd0b64d7cd677fcf47c38ecf9e25a4f",
    "cb0801b490ba13dfb6d36aeef06b049ff67ff11864ef62ccd858a0096d97c6af",
)
_GOLDEN_MANIFEST_SHA256 = "c25251ee27da05afd75adc3c6869cbc2944b80c05c5d6e703b6ebbbba697a4f0"
_PACKED_MASK_SHA256 = "1c7830b37739e409fbb8dab2b81c31c63b3379e6c10ae9e6b4ca2cc48a656094"
_UNPACK_LSB = tuple(bytes((value >> bit) & 1 for bit in range(8)) for value in range(256))
_MASK_TO_GRAYSCALE = bytes.maketrans(b"\x01", b"\xff")


class _RunResult(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint64),
        ("abi_version", ctypes.c_uint32),
        ("mask_memory_kind", ctypes.c_uint32),
        ("mask_device_ordinal", ctypes.c_int32),
        ("label", ctypes.c_int32),
        ("detector_score", ctypes.c_float),
        ("prompt_box_xyxy", ctypes.c_float * 4),
        ("masks", ctypes.c_void_p * _FRAME_COUNT),
        ("reserved_u64", ctypes.c_uint64 * 4),
    ]


if (ctypes.sizeof(_RunResult), _RunResult.masks.offset, _RunResult.reserved_u64.offset) != (
    120,
    48,
    88,
):
    raise RuntimeError("SAM2 Python runner does not match the public C result layout")


def _regular(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RuntimeError(f"{label} is missing: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} must be a regular non-symlink: {path}")
    return path


def _directory(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RuntimeError(f"{label} is missing: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"{label} must be a directory non-symlink: {path}")
    return path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path: Path, label: str) -> bytes:
    return _regular(path, label).read_bytes()


def _load_frames(fixture_dir: Path) -> list[ctypes.Array]:
    frames = []
    for index, expected_hash in enumerate(_RGB8_SHA256):
        data = _read(fixture_dir / "rgb8" / f"{index:06d}.rgb8", "SAM2 RGB8 frame")
        if len(data) != _RGB_BYTES:
            raise RuntimeError("SAM2 RGB8 frame must contain exactly 1280x1088x3 bytes")
        if _sha256(data) != expected_hash:
            raise RuntimeError(f"SAM2 RGB8 frame SHA-256 mismatch for frame {index}")
        frames.append((ctypes.c_uint8 * len(data)).from_buffer_copy(data))
    return frames


def _synthetic_frames() -> list[ctypes.Array]:
    """Generate five distinct RGB8 fixtures without repository artifacts."""

    rows = np.arange(_HEIGHT, dtype=np.uint16)[:, None]
    columns = np.arange(_WIDTH, dtype=np.uint16)[None, :]
    frames = []
    for index in range(_FRAME_COUNT):
        rgb = np.empty((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
        rgb[..., 0] = (columns + 17 * index) & 0xFF
        rgb[..., 1] = (rows + 31 * index) & 0xFF
        rgb[..., 2] = ((columns // 8) ^ (rows // 8) ^ (53 * index)) & 0xFF
        top = 300 + 20 * index
        left = 250 + 30 * index
        rgb[top : top + 500, left : left + 400] = (235, 45 + 20 * index, 25)
        frames.append((ctypes.c_uint8 * rgb.size).from_buffer_copy(rgb))
    return frames


def _png(width: int, height: int, channels: int, pixels: bytes) -> bytes:
    """Encode deterministic report-only RGB or grayscale pixels as PNG."""

    if channels not in (1, 3) or len(pixels) != width * height * channels:
        raise RuntimeError("SAM2 report image has an invalid shape")

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    row_bytes = width * channels
    scanlines = b"".join(
        b"\0" + pixels[offset : offset + row_bytes] for offset in range(0, len(pixels), row_bytes)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 0 if channels == 1 else 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


def _write_report_media(
    case: E2ECase, ctx: RunContext, frames: list[ctypes.Array], masks: bytes
) -> tuple[str, str]:
    """Persist the actual first input and mask for the consolidated report."""

    if not ctx.artifacts_dir or len(masks) < _FRAME_PIXELS:
        raise RuntimeError("SAM2 report artifacts are unavailable")
    output_dir = Path(ctx.artifacts_dir) / case.name
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "input_frame.png"
    mask_path = output_dir / "segmentation_mask.png"
    input_path.write_bytes(_png(_WIDTH, _HEIGHT, 3, bytes(frames[0])))
    mask_path.write_bytes(
        _png(_WIDTH, _HEIGHT, 1, masks[:_FRAME_PIXELS].translate(_MASK_TO_GRAYSCALE))
    )
    return str(input_path), str(mask_path)


def _l0_bundle_contract(bundle: Path) -> dict[str, object]:
    """Bind the smoke receipt to the exact six-plan synthetic-public bundle."""

    with bundle.open("rb") as stream:
        if stream.read(8) != b"BUNDLE\x01\x00":
            raise RuntimeError("SAM2 L0 bundle has an invalid magic header")
        encoded_length = stream.read(8)
        if len(encoded_length) != 8:
            raise RuntimeError("SAM2 L0 bundle has a truncated header length")
        header_length = struct.unpack("<Q", encoded_length)[0]
        try:
            header = json.loads(stream.read(header_length))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("SAM2 L0 bundle header is invalid") from error
        sections = header.get("sections")
        expected = (*_PLAN_SECTIONS, "config.json")
        if not isinstance(sections, dict) or tuple(sections) != expected:
            raise RuntimeError("SAM2 L0 bundle does not contain exactly six plans and config.json")
        config_metadata = sections["config.json"]
        if not isinstance(config_metadata, dict):
            raise RuntimeError("SAM2 L0 bundle config metadata is invalid")
        config_offset = config_metadata.get("offset")
        config_size = config_metadata.get("size")
        if not isinstance(config_offset, int) or not isinstance(config_size, int):
            raise RuntimeError("SAM2 L0 bundle config location is invalid")
        section_base = stream.tell()
        stream.seek(section_base + config_offset)
        try:
            config = json.loads(stream.read(config_size))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("SAM2 L0 bundle config is invalid") from error
    if config.get("sam2_checkpoint_variant") != _PUBLIC_CORE_VARIANT:
        raise RuntimeError("SAM2 L0 bundle did not bind the public-core checkpoint variant")
    return {
        "plan_sections": list(_PLAN_SECTIONS),
        "checkpoint_variant": _PUBLIC_CORE_VARIANT,
    }


def _finite_float(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} contains a non-number")
    result = ctypes.c_float(value).value
    if not math.isfinite(result):
        raise RuntimeError(f"{label} contains a non-finite value")
    return result


def _load_golden(fixture_dir: Path) -> tuple[tuple[tuple[float, ...], float, int], bytes, str]:
    golden_dir = _directory(fixture_dir / "golden", "SAM2 golden root")
    manifest_bytes = _read(golden_dir / "manifest.json", "SAM2 golden manifest")
    manifest_hash = _sha256(manifest_bytes)
    if manifest_hash != _GOLDEN_MANIFEST_SHA256:
        raise RuntimeError("SAM2 golden manifest hash mismatch")
    try:
        bbox = json.loads(manifest_bytes)["frame_zero_bbox"]
        coordinates = tuple(
            _finite_float(value, "original box") for value in bbox["original_image_xyxy"]
        )
        score = _finite_float(bbox["score"], "SAM2 golden bbox score")
        label = bbox["label"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("SAM2 golden manifest shape mismatch") from error
    if len(coordinates) != 4:
        raise RuntimeError("original box has the wrong shape")
    if not isinstance(label, int) or isinstance(label, bool) or not 0 <= score <= 1 or label != 1:
        raise RuntimeError("SAM2 golden bbox metadata mismatch")

    packed = _read(golden_dir / "masks.bitpack", "SAM2 golden packed mask")
    mask_elements = _FRAME_COUNT * _FRAME_PIXELS
    if len(packed) != (mask_elements + 7) // 8 or _sha256(packed) != _PACKED_MASK_SHA256:
        raise RuntimeError("SAM2 golden packed mask mismatch")
    logical = b"".join(_UNPACK_LSB[value] for value in packed)[:mask_elements]
    return (coordinates, score, label), logical, manifest_hash


def _load_library(path: Path):
    return ctypes.CDLL(str(path))


def _runtime_library(plugin_dir: Path) -> Path:
    candidates = (
        plugin_dir / _RUNTIME_LIBRARY,
        plugin_dir / "sam2" / _RUNTIME_LIBRARY,
    )
    matches = [path for path in candidates if path.is_file() and not path.is_symlink()]
    if len(matches) != 1:
        raise RuntimeError(
            f"SAM2 runtime library must resolve exactly once below {plugin_dir}: {matches}"
        )
    return matches[0]


def _declare(library, name: str, result, *arguments) -> None:
    function = getattr(library, name)
    function.argtypes = list(arguments)
    function.restype = result


def _configure_library(library) -> None:
    pointer = ctypes.c_void_p
    signatures = (
        ("trtmc_sam2_video_abi_version", ctypes.c_uint32, ()),
        ("trtmc_sam2_video_last_error", ctypes.c_char_p, ()),
        (
            "trtmc_sam2_video_create_from_bundle_v1",
            pointer,
            (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p),
        ),
        ("trtmc_sam2_video_session_destroy", None, (pointer,)),
        (
            "trtmc_sam2_video_run_rgb8_v1",
            ctypes.c_int32,
            (
                pointer,
                *(ctypes.POINTER(ctypes.c_uint8),) * _FRAME_COUNT,
                ctypes.c_uint32,
                ctypes.POINTER(_RunResult),
                ctypes.c_uint64,
            ),
        ),
    )
    for name, result, arguments in signatures:
        _declare(library, name, result, *arguments)


def _diagnostic(library) -> str:
    value = library.trtmc_sam2_video_last_error()
    return value.decode(errors="replace") if value else "no diagnostic"


def _require_status(library, status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"{operation} failed with status {status}: {_diagnostic(library)}")


def _run_runtime(
    library,
    bundle: Path,
    plugin_dir: Path,
    backend_dir: Path,
    frames,
    *,
    repetitions: int = 1,
):
    _configure_library(library)
    if library.trtmc_sam2_video_abi_version() != _ABI_VERSION:
        raise RuntimeError("runtime does not expose SAM2 ABI version 1")
    session = library.trtmc_sam2_video_create_from_bundle_v1(
        os.fsencode(bundle), os.fsencode(plugin_dir), os.fsencode(backend_dir)
    )
    if not session:
        raise RuntimeError(f"session creation failed: {_diagnostic(library)}")
    try:
        outputs = []
        for _ in range(repetitions):
            result = _RunResult()
            _require_status(
                library,
                library.trtmc_sam2_video_run_rgb8_v1(
                    session,
                    *frames,
                    _RUN_MATERIALIZE_MASKS_HOST,
                    ctypes.byref(result),
                    ctypes.sizeof(result),
                ),
                "run RGB8 video",
            )
            coordinates = tuple(float(value) for value in result.prompt_box_xyxy)
            if (
                result.struct_size != ctypes.sizeof(result)
                or result.abi_version != _ABI_VERSION
                or result.mask_memory_kind != _MASK_MEMORY_HOST
                or result.mask_device_ordinal != -1
                or not math.isfinite(result.detector_score)
                or not 0.0 <= result.detector_score <= 1.0
                or result.label < 0
                or coordinates[0] > coordinates[2]
                or coordinates[1] > coordinates[3]
                or not all(math.isfinite(value) for value in coordinates)
                or not all(result.masks)
                or any(result.reserved_u64)
            ):
                raise RuntimeError("SAM2 aggregate result violated the public ABI contract")
            masks = b"".join(ctypes.string_at(pointer, _FRAME_PIXELS) for pointer in result.masks)
            outputs.append(((coordinates, float(result.detector_score), int(result.label)), masks))
        return outputs
    finally:
        library.trtmc_sam2_video_session_destroy(session)


def _mask_accuracy(candidate: bytes, reference: bytes) -> tuple[list[float], float, float]:
    if len(candidate) != len(reference) or len(candidate) != _FRAME_COUNT * _FRAME_PIXELS:
        raise RuntimeError("SAM2 candidate mask has the wrong size")
    frame_iou = []
    global_intersection = 0
    global_union = 0
    for index in range(_FRAME_COUNT):
        begin = index * _FRAME_PIXELS
        left = candidate[begin : begin + _FRAME_PIXELS]
        right = reference[begin : begin + _FRAME_PIXELS]
        if not set(left) <= {0, 1}:
            raise RuntimeError("SAM2 candidate mask is not binary")
        intersection = sum(a & b for a, b in zip(left, right, strict=True))
        union = sum(left) + sum(right) - intersection
        frame_iou.append(1.0 if union == 0 else intersection / union)
        global_intersection += intersection
        global_union += union
    return (
        frame_iou,
        sum(frame_iou) / _FRAME_COUNT,
        1.0 if global_union == 0 else global_intersection / global_union,
    )


def _bbox_accuracy(candidate, reference) -> tuple[float, float, float, bool]:
    left, left_score, left_label = candidate
    right, right_score, right_label = reference
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return (
        intersection / union if union > 0 else 0.0,
        max(abs(a - b) for a, b in zip(left, right, strict=True)),
        abs(left_score - right_score),
        left_label == right_label,
    )


class Sam2PublicRunner:
    @property
    def strategy_name(self) -> str:
        return "prompted_segmentation"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if stage.name != "five_frame_tracking":
            raise RuntimeError(f"unsupported SAM2 E2E stage: {stage.name}")
        bundle = _regular(Path(ctx.engine_dir) / case.bundle, "caller-built SAM2 bundle")
        if not ctx.model_plugin_dir or not ctx.binary_path:
            raise RuntimeError("SAM2 plugin and backend paths are required")
        plugin_dir = _directory(Path(ctx.model_plugin_dir), "SAM2 plugin directory")
        backend_dir = _directory(Path(ctx.binary_path).parent, "SAM2 backend directory")
        library_path = _runtime_library(plugin_dir)

        bundle_sha256 = _sha256_file(bundle)
        if case.reference_backend == "invariant_only":
            if case.inputs.get("fixture_kind") != "deterministic_synthetic_rgb8":
                raise RuntimeError("SAM2 L0 requires its deterministic synthetic RGB8 fixture")
            bundle_contract = _l0_bundle_contract(bundle)
            frames = _synthetic_frames()
            first, second = _run_runtime(
                _load_library(library_path),
                bundle,
                plugin_dir,
                backend_dir,
                frames,
                repetitions=2,
            )
            if _sha256_file(bundle) != bundle_sha256:
                raise RuntimeError("caller-built SAM2 bundle changed during public C ABI execution")
            track, masks = first
            input_image_path, viz_path = _write_report_media(case, ctx, frames, masks)
            mask_sums = [
                sum(masks[index * _FRAME_PIXELS : (index + 1) * _FRAME_PIXELS])
                for index in range(_FRAME_COUNT)
            ]
            return StageOutput(
                stage_name=stage.name,
                data={
                    "schema_version": 2,
                    "bundle_sha256": bundle_sha256,
                    "input_image_path": input_image_path,
                    "viz_path": viz_path,
                    "runtime_invariants": {
                        **bundle_contract,
                        "same_session_repeat_exact": first == second,
                        "bbox_xyxy": list(track[0]),
                        "detector_score": track[1],
                        "label": track[2],
                        "binary_masks": set(masks) <= {0, 1},
                        "mask_foreground_pixels": mask_sums,
                        "mask_sha256": [
                            _sha256(masks[index * _FRAME_PIXELS : (index + 1) * _FRAME_PIXELS])
                            for index in range(_FRAME_COUNT)
                        ],
                    },
                },
                metadata={"runtime_library": str(library_path)},
            )

        fixture_dir = _directory(
            Path(ctx.engine_dir) / str(case.inputs["fixture_dir"]), "SAM2 fixture directory"
        )
        frames = _load_frames(fixture_dir)
        golden_bbox, golden_masks, manifest_hash = _load_golden(fixture_dir)
        ((track, masks),) = _run_runtime(
            _load_library(library_path), bundle, plugin_dir, backend_dir, frames
        )
        if _sha256_file(bundle) != bundle_sha256:
            raise RuntimeError("caller-built SAM2 bundle changed during public C ABI execution")
        input_image_path, viz_path = _write_report_media(case, ctx, frames, masks)
        frame_iou, macro_iou, global_iou = _mask_accuracy(masks, golden_masks)
        bbox_iou, coordinate_error, score_error, label_exact = _bbox_accuracy(track, golden_bbox)
        return StageOutput(
            stage_name=stage.name,
            data={
                "schema_version": 1,
                "bundle_sha256": bundle_sha256,
                "input_image_path": input_image_path,
                "viz_path": viz_path,
                "golden_manifest_sha256": manifest_hash,
                "accuracy": {
                    "frame_mask_iou": frame_iou,
                    "macro_mask_iou": macro_iou,
                    "global_mask_iou": global_iou,
                    "bbox_iou": bbox_iou,
                    "bbox_max_coordinate_error": coordinate_error,
                    "bbox_score_error": score_error,
                    "label_exact": label_exact,
                },
            },
            metadata={"runtime_library": str(library_path)},
        )


runner = Sam2PublicRunner()
