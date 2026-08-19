# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run fixed five-frame SAM2-HOI tracking through its model-owned C ABI."""

from __future__ import annotations

import ctypes
import hashlib
import os
import stat
import time
from pathlib import Path

from .. import case_artifact_dir
from .._schema import (
    FRAME_COUNT,
    load_npz_arrays,
    normalize_runtime_json,
    resolve_project_path,
    structured_summary,
    validate_dimensions,
)
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec


_ABI_VERSION = 1
_RUNTIME_LIBRARY = "libtrtmc_model_sam2_hoi.so"


class _RunResult(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint64),
        ("abi_version", ctypes.c_uint32),
        ("produced_frame_count", ctypes.c_int32),
        ("reserved_u64", ctypes.c_uint64 * 6),
    ]


if (
    ctypes.sizeof(_RunResult),
    _RunResult.produced_frame_count.offset,
    _RunResult.reserved_u64.offset,
) != (64, 12, 16):
    raise RuntimeError("SAM2 HOI Python runner does not match the public C result layout")


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


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_library(path: Path):
    return ctypes.CDLL(str(path))


def _runtime_library(plugin_dir: Path) -> Path:
    candidates = (
        plugin_dir / _RUNTIME_LIBRARY,
        plugin_dir / "sam2_hoi" / _RUNTIME_LIBRARY,
    )
    matches = [path for path in candidates if path.is_file() and not path.is_symlink()]
    if len(matches) != 1:
        raise RuntimeError(
            f"SAM2 HOI runtime library must resolve exactly once below {plugin_dir}: {matches}"
        )
    return matches[0]


def _declare(library, name: str, result, *arguments) -> None:
    function = getattr(library, name)
    function.argtypes = list(arguments)
    function.restype = result


def _configure_library(library) -> None:
    pointer = ctypes.c_void_p
    signatures = (
        ("trtmc_sam2_hoi_video_abi_version", ctypes.c_uint32, ()),
        ("trtmc_sam2_hoi_video_last_error", ctypes.c_char_p, ()),
        (
            "trtmc_sam2_hoi_video_create_from_bundle_v1",
            pointer,
            (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p),
        ),
        ("trtmc_sam2_hoi_video_session_destroy", None, (pointer,)),
        (
            "trtmc_sam2_hoi_video_run_jpeg_files_v1",
            ctypes.c_int32,
            (
                pointer,
                *(ctypes.c_char_p,) * FRAME_COUNT,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.POINTER(_RunResult),
                ctypes.c_uint64,
            ),
        ),
    )
    for name, result, arguments in signatures:
        _declare(library, name, result, *arguments)


def _diagnostic(library) -> str:
    value = library.trtmc_sam2_hoi_video_last_error()
    return value.decode(errors="replace") if value else "no diagnostic"


def _run_runtime(
    library,
    bundle: Path,
    plugin_dir: Path,
    backend_dir: Path,
    frame_paths: tuple[Path, ...],
    output_json: Path,
    output_masks_dir: Path,
) -> _RunResult:
    _configure_library(library)
    if library.trtmc_sam2_hoi_video_abi_version() != _ABI_VERSION:
        raise RuntimeError("runtime does not expose SAM2 HOI ABI version 1")
    session = library.trtmc_sam2_hoi_video_create_from_bundle_v1(
        os.fsencode(bundle), os.fsencode(plugin_dir), os.fsencode(backend_dir)
    )
    if not session:
        raise RuntimeError(f"SAM2 HOI session creation failed: {_diagnostic(library)}")
    try:
        result = _RunResult()
        status = library.trtmc_sam2_hoi_video_run_jpeg_files_v1(
            session,
            *(os.fsencode(path) for path in frame_paths),
            os.fsencode(output_json),
            os.fsencode(output_masks_dir),
            ctypes.byref(result),
            ctypes.sizeof(result),
        )
        if status != 0:
            raise RuntimeError(
                f"SAM2 HOI five-frame C ABI failed with status {status}: {_diagnostic(library)}"
            )
        if (
            result.struct_size != ctypes.sizeof(result)
            or result.abi_version != _ABI_VERSION
            or result.produced_frame_count != FRAME_COUNT
            or any(result.reserved_u64)
        ):
            raise RuntimeError("SAM2 HOI result violated the public C ABI contract")
        return result
    finally:
        library.trtmc_sam2_hoi_video_session_destroy(session)


class HoiVideoTrackingRunner:
    """Run the family-owned fixed five-frame C ABI and normalize its output."""

    @property
    def strategy_name(self) -> str:
        return "hoi_video_tracking"

    @staticmethod
    def _bundle_path(case: E2ECase, ctx: RunContext) -> Path:
        bundle = Path(case.bundle or f"{case.name}.bundle")
        return bundle if bundle.is_absolute() else Path(ctx.engine_dir) / bundle

    @staticmethod
    def _frame_paths(case: E2ECase) -> tuple[Path, ...]:
        value = case.inputs.get("frames_dir")
        if not isinstance(value, str) or not value:
            raise ValueError("SAM2 HOI E2E input frames_dir is required")
        frames_dir = _directory(resolve_project_path(value), "SAM2 HOI frame directory")
        expected = tuple(frames_dir / f"{index:06d}.jpg" for index in range(FRAME_COUNT))
        for index, path in enumerate(expected):
            _regular(path, f"SAM2 HOI JPEG frame {index}")
        extras = sorted(path.name for path in frames_dir.glob("*.jpg") if path not in expected)
        if extras:
            raise ValueError(f"SAM2 HOI frames_dir contains unexpected JPEGs: {extras}")
        return expected

    def run_stage(
        self,
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
    ) -> StageOutput:
        if stage.name != "full_tracking":
            raise ValueError(f"Unsupported SAM2 HOI E2E stage: {stage.name}")
        bundle = _regular(self._bundle_path(case, ctx), "caller-built SAM2 HOI bundle")
        if not ctx.model_plugin_dir or not ctx.binary_path:
            raise RuntimeError("SAM2 HOI plugin and backend paths are required")
        plugin_dir = _directory(Path(ctx.model_plugin_dir), "SAM2 HOI plugin directory")
        backend_dir = _directory(Path(ctx.binary_path).parent, "SAM2 HOI backend directory")
        library_path = _runtime_library(plugin_dir)
        frame_paths = self._frame_paths(case)
        artifact_dir = case_artifact_dir(ctx.artifacts_dir, case.name)
        output_json = artifact_dir / "trt_tracking.json"
        output_masks_dir = artifact_dir / "trt_masks"
        bundle_sha256 = _sha256_file(bundle)

        started = time.monotonic()
        result = _run_runtime(
            _load_library(library_path),
            bundle,
            plugin_dir,
            backend_dir,
            frame_paths,
            output_json,
            output_masks_dir,
        )
        elapsed = time.monotonic() - started
        if _sha256_file(bundle) != bundle_sha256:
            raise RuntimeError("caller-built SAM2 HOI bundle changed during public C ABI execution")

        output_npz = output_json.with_name("trt_tracking.npz")
        normalize_runtime_json(output_json, output_npz)
        arrays = load_npz_arrays(output_npz)
        validate_dimensions(
            arrays,
            height=int(case.inputs.get("expected_height", 1280)),
            width=int(case.inputs.get("expected_width", 1088)),
        )
        return StageOutput(
            stage_name=stage.name,
            data={
                "schema_version": 1,
                "frame_count": result.produced_frame_count,
                "frames": structured_summary(arrays),
                "output_path": str(output_npz),
                "output_npz": str(output_npz),
                "bundle_sha256": bundle_sha256,
                "returncode": 0,
            },
            timing_s=elapsed,
            metadata={
                "runtime_library": str(library_path),
                "runtime_entrypoint": "trtmc_sam2_hoi_video_run_jpeg_files_v1",
            },
        )


plugin = HoiVideoTrackingRunner()
