#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run model-first TRTMC reference validation for Dev and QA."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import platform
import secrets
import shlex
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
for import_root in (REPO_ROOT, PYTHON_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tensorrt_model_connect.python_profiles import (  # noqa: E402
    DEFAULT_PROFILE,
    normalize_execution_profiles,
    profile_env_var,
    resolve_profile_python,
)
from tools import task_eval, trtmc_disagreements  # noqa: E402


DEFAULT_CATALOG = REPO_ROOT / "tests" / "validation" / "model_workloads.yaml"
DEFAULT_SUITES = REPO_ROOT / "tests" / "task_eval" / "validation_suites.yaml"
DEFAULT_MODELS = REPO_ROOT / "tests" / "e2e" / "models"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "trtmc-validate"
DEFAULT_ENGINE_DIR = DEFAULT_OUTPUT / "engines"
DEFAULT_REFERENCE_CACHE = DEFAULT_OUTPUT / "references"
COMMON_REFERENCE_PROFILE = "reference_common"
NOT_COMPARED_DIRECTORY = "not-compared"
DISAGREEMENT_ARTIFACT_NAME = "disagreements.jsonl"
MAX_REPORT_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_REPORT_JSON_DEPTH = 256
MAX_VALIDATION_RESULT_JSON_DEPTH = MAX_REPORT_JSON_DEPTH - 2
MAX_COMMAND_LOG_DISCOVERY_ENTRIES = 4096
MAX_COMMAND_LOG_FILES = 128
MAX_COMMAND_LOG_TOTAL_BYTES = 64 * 1024 * 1024
MAX_COMMAND_LOG_DEPTH = 64
MAX_REPORT_RESULTS = 1000
MAX_REPORT_DISCOVERY_ENTRIES = 4096
MAX_REPORT_RESULT_BYTES = 64 * 1024 * 1024
MAX_REPORT_DISAGREEMENT_BYTES = 64 * 1024 * 1024
MAX_REPORT_DISAGREEMENT_RECORDS = 1000
MAX_REPORT_DISAGREEMENT_SOURCE_BYTES = 64 * 1024 * 1024
MAX_REPORT_MEDIA_FILES = trtmc_disagreements.MAX_CASE_MEDIA_FILES
MAX_REPORT_MEDIA_BYTES = trtmc_disagreements.MAX_CASE_MEDIA_BYTES
MAX_TRANSACTION_TREE_ENTRIES = 100_000
MAX_TRANSACTION_TREE_DEPTH = 2048
LEGACY_E2E_REASON = (
    "E2E execution does not compare aligned reference and TRTMC outputs."
)


class ValidationError(RuntimeError):
    """The requested validation cannot be resolved or executed."""


@dataclass(frozen=True)
class Binding:
    model: str
    workload: str | None
    not_compared_reason: str = ""
    reference_cache_identity: str = ""

    @property
    def runnable(self) -> bool:
        return self.workload is not None


def _required_workload(binding: Binding) -> str:
    if binding.workload is None:
        raise ValidationError(
            f"model {binding.model} has no reference-consistency workload"
        )
    return binding.workload


def _case_directory(output: Path, binding: Binding) -> Path:
    return output / binding.model / (
        binding.workload if binding.workload is not None else NOT_COMPARED_DIRECTORY
    )


def _ensure_real_directory(path: Path, *, description: str) -> None:
    """Create a directory tree without traversing a symlink component."""
    absolute = path.absolute()
    descriptor = os.open("/", _secure_directory_flags())
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(
                    component,
                    _secure_directory_flags(),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o777, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(
                    component,
                    _secure_directory_flags(),
                    dir_fd=descriptor,
                )
            os.close(descriptor)
            descriptor = child
    except (OSError, ValueError) as exc:
        raise ValidationError(
            f"cannot securely create or inspect {description} {path}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)


def _prepare_case_directory(output: Path, binding: Binding) -> Path:
    """Return a case directory whose model/workload components are not links."""
    components = (
        binding.model,
        binding.workload if binding.workload is not None else NOT_COMPARED_DIRECTORY,
    )
    if any(
        not component
        or component in {".", ".."}
        or Path(component).name != component
        for component in components
    ):
        raise ValidationError(
            f"unsafe validation output path for {binding.model}/{binding.workload}"
        )
    _ensure_real_directory(output, description="validation output")
    model_dir = output / binding.model
    _ensure_real_directory(model_dir, description="validation model output")
    case_dir = _case_directory(output, binding)
    _ensure_real_directory(case_dir, description="validation case output")
    return case_dir


def _secure_directory_flags() -> int:
    missing = [
        name
        for name in ("O_DIRECTORY", "O_NOFOLLOW")
        if not hasattr(os, name)
    ]
    if missing:
        raise ValidationError(
            "secure validation artifacts require " + ", ".join(missing)
        )
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(
        os,
        "O_CLOEXEC",
        0,
    )


def _open_real_directory(path: Path) -> int:
    absolute = path.absolute()
    descriptor: int | None = None
    try:
        descriptor = os.open("/", _secure_directory_flags())
        for component in absolute.parts[1:]:
            child = os.open(
                component,
                _secure_directory_flags(),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
    except (OSError, ValueError) as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise ValidationError(
            f"cannot securely open validation artifact directory {path}: {exc}"
        ) from exc
    assert descriptor is not None
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValidationError(
                f"validation artifact directory must be real: {path}"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _validate_atomic_destination(directory_fd: int, name: str, path: Path) -> None:
    try:
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValidationError(f"cannot inspect validation artifact {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValidationError(f"validation artifact must be a regular file: {path}")


def _create_atomic_artifact(
    directory_fd: int,
    path: Path,
) -> tuple[int, str]:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    for attempt in range(100):
        temporary_name = (
            f".{path.name}.{os.getpid()}.{time.time_ns()}.{attempt}.tmp"
        )
        try:
            descriptor = os.open(
                temporary_name,
                flags,
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError:
            continue
        except OSError as exc:
            raise ValidationError(
                f"cannot create temporary validation artifact for {path}: {exc}"
            ) from exc
        return descriptor, temporary_name
    raise ValidationError(f"cannot reserve temporary validation artifact for {path}")


def _verify_visible_artifact_parent(path: Path, directory_fd: int) -> None:
    visible_fd: int | None = None
    try:
        visible_fd = _open_real_directory(path.parent)
        visible = os.fstat(visible_fd)
        opened = os.fstat(directory_fd)
    except (OSError, ValidationError) as exc:
        raise ValidationError(
            "validation artifact parent changed while writing "
            f"{path}: {exc}"
        ) from exc
    finally:
        if visible_fd is not None:
            os.close(visible_fd)
    if (
        not stat.S_ISDIR(visible.st_mode)
        or visible.st_dev != opened.st_dev
        or visible.st_ino != opened.st_ino
    ):
        raise ValidationError(
            f"validation artifact parent changed while writing {path}"
        )


def _verify_visible_regular_artifact(
    path: Path,
    directory_fd: int,
    opened_metadata: os.stat_result,
    *,
    operation: str,
    check_ctime: bool = True,
) -> None:
    try:
        visible = os.stat(
            path.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise ValidationError(
            f"validation artifact changed while {operation} {path}: {exc}"
        ) from exc
    if (
        not stat.S_ISREG(visible.st_mode)
        or visible.st_dev != opened_metadata.st_dev
        or visible.st_ino != opened_metadata.st_ino
        or visible.st_nlink != opened_metadata.st_nlink
        or visible.st_size != opened_metadata.st_size
        or visible.st_mtime_ns != opened_metadata.st_mtime_ns
        or (
            check_ctime
            and visible.st_ctime_ns != opened_metadata.st_ctime_ns
        )
    ):
        raise ValidationError(
            f"validation artifact changed while {operation} {path}"
        )


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Replace a regular artifact through an anchored, non-symlink directory."""
    _ensure_real_directory(path.parent, description="validation artifact parent")
    directory_fd = _open_real_directory(path.parent)
    temporary_name = ""
    descriptor: int | None = None
    temporary_metadata: os.stat_result | None = None
    try:
        _validate_atomic_destination(directory_fd, path.name, path)
        descriptor, temporary_name = _create_atomic_artifact(directory_fd, path)
        with os.fdopen(descriptor, "wb") as output_file:
            descriptor = None
            output_file.write(data)
            output_file.flush()
            os.fsync(output_file.fileno())
            temporary_metadata = os.fstat(output_file.fileno())
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_name = ""
        assert temporary_metadata is not None
        _verify_visible_regular_artifact(
            path,
            directory_fd,
            temporary_metadata,
            operation="writing",
            check_ctime=False,
        )
        _verify_visible_artifact_parent(path, directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except OSError:
                pass
        os.close(directory_fd)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _regular_artifact_metadata(
    path: Path,
    *,
    missing_ok: bool = False,
) -> os.stat_result | None:
    """Return regular artifact metadata through an anchored parent."""
    directory_fd = _open_real_directory(path.parent)
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                path.name,
                (
                    os.O_RDONLY
                    | os.O_NONBLOCK
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0)
                ),
                dir_fd=directory_fd,
            )
        except FileNotFoundError:
            if missing_ok:
                return None
            raise
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValidationError(
                f"validation artifact must be a regular file: {path}"
            )
        _verify_visible_regular_artifact(
            path,
            directory_fd,
            metadata,
            operation="recording",
        )
        _verify_visible_artifact_parent(path, directory_fd)
        return metadata
    except OSError as exc:
        raise ValidationError(
            f"cannot securely inspect validation artifact {path}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)


def _regular_artifact_identity(
    path: Path,
    *,
    missing_ok: bool = False,
) -> tuple[int, int] | None:
    metadata = _regular_artifact_metadata(
        path,
        missing_ok=missing_ok,
    )
    if metadata is None:
        return None
    return metadata.st_dev, metadata.st_ino


def _atomic_copy_regular_file(
    source: Path,
    destination: Path,
    *,
    missing_ok: bool = False,
    require_single_link: bool = False,
    maximum_bytes: int | None = None,
) -> int | None:
    """Copy one regular artifact without following source or destination links."""
    _ensure_real_directory(
        destination.parent,
        description="validation artifact parent",
    )
    source_directory_fd: int | None = None
    destination_directory_fd: int | None = None
    source_fd: int | None = None
    temporary_fd: int | None = None
    temporary_name = ""
    temporary_metadata: os.stat_result | None = None
    try:
        source_directory_fd = _open_real_directory(source.parent)
        destination_directory_fd = _open_real_directory(
            destination.parent
        )
        source_flags = (
            os.O_RDONLY
            | os.O_NONBLOCK
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            source_fd = os.open(
                source.name,
                source_flags,
                dir_fd=source_directory_fd,
            )
        except FileNotFoundError:
            if missing_ok:
                return None
            raise
        source_metadata = os.fstat(source_fd)
        if not stat.S_ISREG(source_metadata.st_mode):
            raise ValidationError(
                f"validation artifact source must be a regular file: {source}"
            )
        if require_single_link and source_metadata.st_nlink != 1:
            raise ValidationError(
                f"validation artifact source must not be a hard link: {source}"
            )
        if (
            maximum_bytes is not None
            and source_metadata.st_size > maximum_bytes
        ):
            raise ValidationError(
                f"validation artifact source exceeds {maximum_bytes} bytes: "
                f"{source}"
            )
        _validate_atomic_destination(
            destination_directory_fd,
            destination.name,
            destination,
        )
        temporary_fd, temporary_name = _create_atomic_artifact(
            destination_directory_fd,
            destination,
        )
        with os.fdopen(source_fd, "rb") as source_file:
            source_fd = None
            with os.fdopen(temporary_fd, "wb") as destination_file:
                temporary_fd = None
                copied_bytes = 0
                while chunk := source_file.read(1024 * 1024):
                    copied_bytes += len(chunk)
                    if (
                        maximum_bytes is not None
                        and copied_bytes > maximum_bytes
                    ):
                        raise ValidationError(
                            "validation artifact source exceeds "
                            f"{maximum_bytes} bytes while copying: {source}"
                        )
                    destination_file.write(chunk)
                destination_file.flush()
                os.fsync(destination_file.fileno())
                os.fchmod(
                    destination_file.fileno(),
                    stat.S_IMODE(source_metadata.st_mode),
                )
                temporary_metadata = os.fstat(destination_file.fileno())
        _verify_visible_regular_artifact(
            source,
            source_directory_fd,
            source_metadata,
            operation="copying",
        )
        _verify_visible_artifact_parent(source, source_directory_fd)
        os.replace(
            temporary_name,
            destination.name,
            src_dir_fd=destination_directory_fd,
            dst_dir_fd=destination_directory_fd,
        )
        temporary_name = ""
        assert temporary_metadata is not None
        _verify_visible_regular_artifact(
            destination,
            destination_directory_fd,
            temporary_metadata,
            operation="copying",
            check_ctime=False,
        )
        _verify_visible_artifact_parent(destination, destination_directory_fd)
        return copied_bytes
    except OSError as exc:
        raise ValidationError(
            f"cannot securely copy validation artifact {source} to "
            f"{destination}: {exc}"
        ) from exc
    finally:
        for descriptor in (source_fd, temporary_fd):
            if descriptor is not None:
                os.close(descriptor)
        if temporary_name:
            try:
                assert destination_directory_fd is not None
                os.unlink(temporary_name, dir_fd=destination_directory_fd)
            except OSError:
                pass
        if destination_directory_fd is not None:
            os.close(destination_directory_fd)
        if source_directory_fd is not None:
            os.close(source_directory_fd)


def _lexical_case_artifact(
    path: Path,
    case_dir: Path,
) -> tuple[Path, Path]:
    """Resolve a path lexically and require it to remain below one case."""
    case_absolute = Path(os.path.abspath(os.fspath(case_dir)))
    candidate = path if path.is_absolute() else case_absolute / path
    candidate_absolute = Path(os.path.abspath(os.fspath(candidate)))
    try:
        relative = candidate_absolute.relative_to(case_absolute)
    except ValueError as exc:
        raise ValidationError(
            f"validation artifact is outside its case directory: {path}"
        ) from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValidationError(f"invalid validation case artifact path: {path}")
    return candidate_absolute, relative


def _read_case_text_artifact(
    path: Path,
    *,
    case_dir: Path,
    missing_ok: bool = False,
    errors: str = "strict",
    maximum_bytes: int = MAX_REPORT_ARTIFACT_BYTES,
) -> str | None:
    """Read a bounded regular file through a held case-directory descriptor."""
    absolute, relative = _lexical_case_artifact(path, case_dir)
    case_fd = _open_real_directory(Path(os.path.abspath(os.fspath(case_dir))))
    directory_fd = case_fd
    descriptor: int | None = None
    try:
        for component in relative.parts[:-1]:
            child = os.open(
                component,
                _secure_directory_flags(),
                dir_fd=directory_fd,
            )
            if directory_fd != case_fd:
                os.close(directory_fd)
            directory_fd = child
        flags = (
            os.O_RDONLY
            | os.O_NONBLOCK
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(
                relative.name,
                flags,
                dir_fd=directory_fd,
            )
        except FileNotFoundError:
            if missing_ok:
                return None
            raise
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValidationError(
                f"validation report artifact must be a regular file: {absolute}"
            )
        if metadata.st_nlink != 1:
            raise ValidationError(
                f"validation report artifact must not be a hard link: {absolute}"
            )
        with os.fdopen(descriptor, "rb") as input_file:
            descriptor = None
            payload = input_file.read(maximum_bytes + 1)
        if len(payload) > maximum_bytes:
            raise ValidationError(
                f"validation report artifact exceeds {maximum_bytes} bytes: "
                f"{absolute}"
            )
        _verify_visible_regular_artifact(
            absolute,
            directory_fd,
            metadata,
            operation="reading",
        )
        _verify_visible_artifact_parent(absolute, directory_fd)
        try:
            return payload.decode("utf-8", errors=errors)
        except UnicodeError as exc:
            raise ValidationError(
                f"validation report artifact is not UTF-8: {absolute}: {exc}"
            ) from exc
    except FileNotFoundError as exc:
        if missing_ok:
            return None
        raise ValidationError(
            f"validation report artifact does not exist: {absolute}"
        ) from exc
    except (OSError, ValueError) as exc:
        raise ValidationError(
            f"cannot securely read validation report artifact {absolute}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_fd != case_fd:
            os.close(directory_fd)
        os.close(case_fd)


def _validate_case_media_artifact(
    path: Path,
    *,
    case_dir: Path,
) -> None:
    """Require one generated media path to be a bounded case-owned file."""
    absolute, relative = _lexical_case_artifact(path, case_dir)
    case_fd = _open_real_directory(Path(os.path.abspath(os.fspath(case_dir))))
    directory_fd = case_fd
    descriptor: int | None = None
    try:
        for component in relative.parts[:-1]:
            child = os.open(
                component,
                _secure_directory_flags(),
                dir_fd=directory_fd,
            )
            if directory_fd != case_fd:
                os.close(directory_fd)
            directory_fd = child
        descriptor = os.open(
            relative.name,
            (
                os.O_RDONLY
                | os.O_NONBLOCK
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0)
            ),
            dir_fd=directory_fd,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValidationError(
                f"validation report media must be a single-link regular "
                f"file: {absolute}"
            )
        if metadata.st_size > trtmc_disagreements.MAX_MEDIA_FILE_BYTES:
            raise ValidationError(
                "validation report media exceeds "
                f"{trtmc_disagreements.MAX_MEDIA_FILE_BYTES} bytes: "
                f"{absolute}"
            )
        _verify_visible_regular_artifact(
            absolute,
            directory_fd,
            metadata,
            operation="validating",
        )
        _verify_visible_artifact_parent(absolute, directory_fd)
    except (OSError, ValueError) as exc:
        raise ValidationError(
            f"cannot securely validate report media {absolute}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_fd != case_fd:
            os.close(directory_fd)
        os.close(case_fd)


def _validated_case_work_dir(
    result: Mapping[str, Any],
    case_dir: Path,
) -> Path | None:
    raw_result = result.get("raw_result", {})
    work_dir = raw_result.get("work_dir") if isinstance(raw_result, Mapping) else None
    if not work_dir:
        return None
    try:
        candidate, _relative = _lexical_case_artifact(
            Path(str(work_dir)),
            case_dir,
        )
    except ValidationError:
        return None
    try:
        descriptor = _open_real_directory(candidate)
    except ValidationError as exc:
        try:
            candidate.lstat()
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            raise exc from None
        raise
    os.close(descriptor)
    return candidate


def _copy_disagreement_media(
    source: Path,
    destination: Path,
    require_single_link: bool,
    maximum_bytes: int,
) -> int:
    copied = _atomic_copy_regular_file(
        source,
        destination,
        require_single_link=require_single_link,
        maximum_bytes=maximum_bytes,
    )
    assert copied is not None
    return copied


def _scan_disagreement_media(
    root: Path,
    maximum_entries: int,
) -> list[Path]:
    descriptor = _open_real_directory(root)
    try:
        return trtmc_disagreements._scan_frame_artifacts_from_fd(
            root,
            descriptor,
            maximum_entries,
        )
    finally:
        os.close(descriptor)


def _build_disagreement_artifact(
    *,
    work_dir: Path,
    case_dir: Path,
    read_artifact: Callable[[Path], str | None],
    staging_root: Path | None = None,
    media_budget: dict[str, int] | None = None,
) -> dict[str, Any]:
    def staged_path(path: Path) -> Path:
        if staging_root is None:
            return path
        _absolute, relative = _lexical_case_artifact(path, case_dir)
        return staging_root / relative

    def write_artifact(path: Path, text: str) -> None:
        _atomic_write_text(staged_path(path), text)

    def copy_artifact(
        source: Path,
        destination: Path,
        require_single_link: bool,
        maximum_bytes: int,
    ) -> int:
        return _copy_disagreement_media(
            source,
            staged_path(destination),
            require_single_link,
            maximum_bytes,
        )

    try:
        return trtmc_disagreements.build_disagreement_artifact(
            work_dir=work_dir,
            case_dir=case_dir,
            write_artifact=write_artifact,
            copy_artifact=copy_artifact,
            read_artifact=read_artifact,
            scan_artifacts=_scan_disagreement_media,
            media_budget=media_budget,
        )
    except (TypeError, UnicodeError, ValueError, RecursionError) as exc:
        raise ValidationError(
            f"invalid disagreement evidence in {work_dir}: {exc}"
        ) from exc


def _directory_identity(descriptor: int) -> tuple[int, int]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValidationError(
            "validation transaction descriptor is not a directory"
        )
    return metadata.st_dev, metadata.st_ino


@dataclass
class _CaseArtifactStage:
    case_dir: Path
    name: str
    case_fd: int | None
    stage_fd: int | None
    case_identity: tuple[int, int]
    stage_identity: tuple[int, int]
    closed: bool = False

    @property
    def path(self) -> Path:
        return self.case_dir / self.name


def _create_case_artifact_stage(
    case_dir: Path,
) -> _CaseArtifactStage:
    _ensure_real_directory(
        case_dir,
        description="validation case output",
    )
    case_fd = _open_real_directory(case_dir)
    try:
        for _attempt in range(100):
            name = f".report-stage-{secrets.token_hex(12)}"
            try:
                os.mkdir(name, 0o700, dir_fd=case_fd)
            except FileExistsError:
                continue
            try:
                stage_fd = os.open(
                    name,
                    _secure_directory_flags(),
                    dir_fd=case_fd,
                )
            except BaseException:
                os.rmdir(name, dir_fd=case_fd)
                raise
            return _CaseArtifactStage(
                case_dir=case_dir,
                name=name,
                case_fd=case_fd,
                stage_fd=stage_fd,
                case_identity=_directory_identity(case_fd),
                stage_identity=_directory_identity(stage_fd),
            )
    except BaseException:
        os.close(case_fd)
        raise
    os.close(case_fd)
    raise ValidationError(
        f"cannot reserve report stage below {case_dir}"
    )


def _verify_case_artifact_stage(stage: _CaseArtifactStage) -> None:
    if stage.closed:
        raise ValidationError(
            f"validation report stage is already closed: {stage.path}"
        )
    _acquire_case_artifact_stage(stage)
    assert stage.case_fd is not None
    assert stage.stage_fd is not None
    if _directory_identity(stage.case_fd) != stage.case_identity:
        raise ValidationError(
            f"validation case directory changed during report staging: "
            f"{stage.case_dir}"
        )
    if _directory_identity(stage.stage_fd) != stage.stage_identity:
        raise ValidationError(
            f"validation report stage changed during report staging: "
            f"{stage.path}"
        )
    visible_case_fd: int | None = None
    visible_stage_fd: int | None = None
    try:
        visible_case_fd = _open_real_directory(stage.case_dir)
        if _directory_identity(visible_case_fd) != stage.case_identity:
            raise ValidationError(
                "validation case directory was replaced during report "
                f"staging: {stage.case_dir}"
            )
        visible_stage_fd = os.open(
            stage.name,
            _secure_directory_flags(),
            dir_fd=visible_case_fd,
        )
        if _directory_identity(visible_stage_fd) != stage.stage_identity:
            raise ValidationError(
                "validation report stage was replaced during report "
                f"staging: {stage.path}"
            )
        held_visible = os.stat(
            stage.name,
            dir_fd=stage.case_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(held_visible.st_mode)
            or (held_visible.st_dev, held_visible.st_ino)
            != stage.stage_identity
        ):
            raise ValidationError(
                "validation report stage changed below its case directory: "
                f"{stage.path}"
            )
    except (OSError, ValueError) as exc:
        raise ValidationError(
            f"cannot verify validation report stage {stage.path}: {exc}"
        ) from exc
    finally:
        if visible_stage_fd is not None:
            os.close(visible_stage_fd)
        if visible_case_fd is not None:
            os.close(visible_case_fd)


def _acquire_case_artifact_stage(
    stage: _CaseArtifactStage,
) -> None:
    if stage.closed:
        raise ValidationError(
            f"validation report stage is already closed: {stage.path}"
        )
    if stage.case_fd is not None or stage.stage_fd is not None:
        if stage.case_fd is None or stage.stage_fd is None:
            raise ValidationError(
                f"validation report stage has inconsistent descriptors: "
                f"{stage.path}"
            )
        return
    case_fd: int | None = None
    stage_fd: int | None = None
    try:
        case_fd = _open_real_directory(stage.case_dir)
        if _directory_identity(case_fd) != stage.case_identity:
            raise ValidationError(
                "validation case directory was replaced during report "
                f"staging: {stage.case_dir}"
            )
        stage_fd = os.open(
            stage.name,
            _secure_directory_flags(),
            dir_fd=case_fd,
        )
        if _directory_identity(stage_fd) != stage.stage_identity:
            raise ValidationError(
                "validation report stage was replaced during report "
                f"staging: {stage.path}"
            )
        stage.case_fd = case_fd
        stage.stage_fd = stage_fd
    except BaseException:
        if stage_fd is not None:
            os.close(stage_fd)
        if case_fd is not None:
            os.close(case_fd)
        raise


def _release_case_artifact_stage(
    stage: _CaseArtifactStage,
) -> None:
    if stage.stage_fd is not None:
        os.close(stage.stage_fd)
        stage.stage_fd = None
    if stage.case_fd is not None:
        os.close(stage.case_fd)
        stage.case_fd = None


def _remove_directory_tree_at(
    parent_fd: int,
    name: str,
) -> None:
    root_fd = os.open(
        name,
        _secure_directory_flags(),
        dir_fd=parent_fd,
    )
    try:
        pending: list[tuple[str, ...]] = [()]
        entries_to_remove: list[tuple[tuple[str, ...], bool]] = []
        visited = 0
        while pending:
            relative = pending.pop()
            if len(relative) > MAX_TRANSACTION_TREE_DEPTH:
                raise ValidationError(
                    "validation transaction cleanup nesting exceeds "
                    f"{MAX_TRANSACTION_TREE_DEPTH}"
                )
            directory_fd = _open_relative_directory_at(
                root_fd,
                relative,
            )
            try:
                with os.scandir(directory_fd) as entries:
                    for entry in entries:
                        visited += 1
                        if visited > MAX_TRANSACTION_TREE_ENTRIES:
                            raise ValidationError(
                                "validation transaction cleanup exceeds "
                                f"{MAX_TRANSACTION_TREE_ENTRIES} entries"
                            )
                        child = (*relative, entry.name)
                        is_directory = entry.is_dir(
                            follow_symlinks=False
                        )
                        entries_to_remove.append(
                            (child, is_directory)
                        )
                        if is_directory:
                            pending.append(child)
            finally:
                os.close(directory_fd)
        for relative, is_directory in reversed(entries_to_remove):
            directory_fd = _open_relative_directory_at(
                root_fd,
                relative[:-1],
            )
            try:
                if is_directory:
                    os.rmdir(relative[-1], dir_fd=directory_fd)
                else:
                    os.unlink(relative[-1], dir_fd=directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        os.close(root_fd)
    os.rmdir(name, dir_fd=parent_fd)


def _open_relative_directory_at(
    root_fd: int,
    components: Sequence[str],
) -> int:
    descriptor = os.dup(root_fd)
    try:
        for component in components:
            child = os.open(
                component,
                _secure_directory_flags(),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _cleanup_case_artifact_stage(
    stage: _CaseArtifactStage,
    *,
    anchored_case_fd: int | None = None,
) -> None:
    if stage.closed:
        return
    if (
        not stage.name.startswith(".report-stage-")
        or not stage.case_dir.name
    ):
        raise ValidationError(
            f"refusing to remove unexpected report stage {stage.path}"
        )
    try:
        if anchored_case_fd is None:
            try:
                _acquire_case_artifact_stage(stage)
            except ValidationError:
                return
            assert stage.case_fd is not None
            case_fd = stage.case_fd
        else:
            if _directory_identity(
                anchored_case_fd
            ) != stage.case_identity:
                return
            case_fd = anchored_case_fd
        try:
            visible = os.stat(
                stage.name,
                dir_fd=case_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        if (
            not stat.S_ISDIR(visible.st_mode)
            or (visible.st_dev, visible.st_ino)
            != stage.stage_identity
        ):
            return
        _remove_directory_tree_at(case_fd, stage.name)
    finally:
        stage.closed = True
        _release_case_artifact_stage(stage)


@dataclass
class _CaseDirectoryUpdate:
    stage: _CaseArtifactStage
    next_name: str
    next_metadata: os.stat_result
    original_metadata: os.stat_result | None
    backup_name: str = ""
    backed_up: bool = False
    installed: bool = False
    anchor_fd: int | None = None


def _prepare_case_directory_update(
    stage: _CaseArtifactStage,
) -> _CaseDirectoryUpdate:
    _verify_case_artifact_stage(stage)
    assert stage.case_fd is not None
    assert stage.stage_fd is not None
    try:
        staged_repro = os.stat(
            "repro",
            dir_fd=stage.stage_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        os.mkdir("repro", 0o700, dir_fd=stage.stage_fd)
        staged_repro = os.stat(
            "repro",
            dir_fd=stage.stage_fd,
            follow_symlinks=False,
        )
    if not stat.S_ISDIR(staged_repro.st_mode):
        raise ValidationError(
            f"validation staged reproduction must be a directory: "
            f"{stage.path / 'repro'}"
        )
    try:
        original = os.stat(
            "repro",
            dir_fd=stage.case_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        original = None
    if original is not None and not stat.S_ISDIR(original.st_mode):
        raise ValidationError(
            "validation reproduction path must be a directory: "
            f"{stage.case_dir / 'repro'}"
        )
    next_name = f".repro-next.{secrets.token_hex(12)}"
    os.mkdir(next_name, 0o700, dir_fd=stage.case_fd)
    try:
        os.rename(
            "repro",
            next_name,
            src_dir_fd=stage.stage_fd,
            dst_dir_fd=stage.case_fd,
        )
    except BaseException:
        os.rmdir(next_name, dir_fd=stage.case_fd)
        raise
    update = _CaseDirectoryUpdate(
        stage=stage,
        next_name=next_name,
        next_metadata=os.stat(
            next_name,
            dir_fd=stage.case_fd,
            follow_symlinks=False,
        ),
        original_metadata=original,
    )
    _release_case_artifact_stage(stage)
    return update


def _verify_case_directory_target(
    update: _CaseDirectoryUpdate,
) -> None:
    stage = update.stage
    if update.anchor_fd is not None:
        case_fd = update.anchor_fd
    else:
        _acquire_case_artifact_stage(stage)
        assert stage.case_fd is not None
        case_fd = stage.case_fd
    visible_case_fd: int | None = None
    try:
        visible_case_fd = _open_real_directory(stage.case_dir)
        if _directory_identity(visible_case_fd) != stage.case_identity:
            raise ValidationError(
                "validation case directory was replaced before report "
                f"publication: {stage.case_dir}"
            )
    finally:
        if visible_case_fd is not None:
            os.close(visible_case_fd)
    try:
        staged = os.stat(
            update.next_name,
            dir_fd=case_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError as exc:
        raise ValidationError(
            "validation staged reproduction disappeared before report "
            f"publication: {stage.case_dir / update.next_name}"
        ) from exc
    if (
        not stat.S_ISDIR(staged.st_mode)
        or staged.st_dev != update.next_metadata.st_dev
        or staged.st_ino != update.next_metadata.st_ino
        or staged.st_ctime_ns != update.next_metadata.st_ctime_ns
        or staged.st_mtime_ns != update.next_metadata.st_mtime_ns
    ):
        raise ValidationError(
            "validation staged reproduction changed before report "
            f"publication: {stage.case_dir / update.next_name}"
        )
    try:
        current = os.stat(
            "repro",
            dir_fd=case_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        current = None
    expected = update.original_metadata
    if expected is None:
        if current is not None:
            raise ValidationError(
                "validation reproduction changed before report publication: "
                f"{stage.case_dir / 'repro'}"
            )
        return
    if (
        current is None
        or not stat.S_ISDIR(current.st_mode)
        or current.st_dev != expected.st_dev
        or current.st_ino != expected.st_ino
        or current.st_ctime_ns != expected.st_ctime_ns
        or current.st_mtime_ns != expected.st_mtime_ns
    ):
        raise ValidationError(
            "validation reproduction changed before report publication: "
            f"{stage.case_dir / 'repro'}"
        )


def _commit_case_directory_update(
    update: _CaseDirectoryUpdate,
) -> None:
    try:
        _verify_case_directory_target(update)
        case_fd = (
            update.anchor_fd
            if update.anchor_fd is not None
            else update.stage.case_fd
        )
        assert case_fd is not None
        if update.original_metadata is not None:
            update.backup_name = (
                f".repro-previous.{secrets.token_hex(12)}"
            )
            os.mkdir(update.backup_name, 0o700, dir_fd=case_fd)
            os.rename(
                "repro",
                update.backup_name,
                src_dir_fd=case_fd,
                dst_dir_fd=case_fd,
            )
            update.backed_up = True
        os.rename(
            update.next_name,
            "repro",
            src_dir_fd=case_fd,
            dst_dir_fd=case_fd,
        )
        update.installed = True
        installed = os.stat(
            "repro",
            dir_fd=case_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(installed.st_mode)
            or installed.st_dev != update.next_metadata.st_dev
            or installed.st_ino != update.next_metadata.st_ino
            or installed.st_mtime_ns
            != update.next_metadata.st_mtime_ns
        ):
            raise ValidationError(
                "validation staged reproduction changed during report "
                f"publication: {update.stage.case_dir / 'repro'}"
            )
    finally:
        if update.anchor_fd is None:
            _release_case_artifact_stage(update.stage)


def _rollback_case_directory_update(
    update: _CaseDirectoryUpdate,
) -> None:
    if update.anchor_fd is None:
        _acquire_case_artifact_stage(update.stage)
    try:
        case_fd = (
            update.anchor_fd
            if update.anchor_fd is not None
            else update.stage.case_fd
        )
        assert case_fd is not None
        if update.installed:
            os.rename(
                "repro",
                update.next_name,
                src_dir_fd=case_fd,
                dst_dir_fd=case_fd,
            )
            update.installed = False
        if update.backed_up:
            os.rename(
                update.backup_name,
                "repro",
                src_dir_fd=case_fd,
                dst_dir_fd=case_fd,
            )
            update.backed_up = False
    finally:
        if update.anchor_fd is None:
            _release_case_artifact_stage(update.stage)


def _finalize_case_directory_update(
    update: _CaseDirectoryUpdate,
) -> None:
    if update.anchor_fd is None:
        try:
            _acquire_case_artifact_stage(update.stage)
        except ValidationError:
            return
    try:
        case_fd = (
            update.anchor_fd
            if update.anchor_fd is not None
            else update.stage.case_fd
        )
        assert case_fd is not None
        for name in (update.backup_name, update.next_name):
            if not name:
                continue
            try:
                _remove_directory_tree_at(case_fd, name)
            except (FileNotFoundError, OSError, ValidationError):
                pass
        try:
            with os.scandir(case_fd) as entries:
                visited = 0
                for entry in entries:
                    visited += 1
                    if visited > MAX_TRANSACTION_TREE_ENTRIES:
                        break
                    if (
                        not entry.name.startswith(".repro-next.")
                        or not entry.is_dir(follow_symlinks=False)
                    ):
                        continue
                    metadata = os.stat(
                        entry.name,
                        dir_fd=case_fd,
                        follow_symlinks=False,
                    )
                    if (
                        metadata.st_dev
                        == update.next_metadata.st_dev
                        and metadata.st_ino
                        == update.next_metadata.st_ino
                    ):
                        _remove_directory_tree_at(
                            case_fd,
                            entry.name,
                        )
                        break
        except (FileNotFoundError, OSError, ValidationError):
            pass
    finally:
        if update.anchor_fd is None:
            _release_case_artifact_stage(update.stage)


@dataclass
class _FileUpdate:
    path: Path
    parent_fd: int | None
    parent_identity: tuple[int, int]
    original_metadata: os.stat_result | None
    next_name: str
    next_metadata: os.stat_result
    backup_name: str = ""
    backed_up: bool = False
    installed: bool = False
    closed: bool = False
    anchor_fd: int | None = None


def _prepare_file_update(path: Path, payload: bytes) -> _FileUpdate:
    _ensure_real_directory(
        path.parent,
        description="validation transaction parent",
    )
    parent_fd = _open_real_directory(path.parent)
    descriptor: int | None = None
    next_name = ""
    try:
        try:
            original = os.stat(
                path.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            original = None
        if original is not None and (
            not stat.S_ISREG(original.st_mode)
            or original.st_nlink != 1
        ):
            raise ValidationError(
                f"validation transaction target must be a single-link "
                f"regular file: {path}"
            )
        descriptor, next_name = _create_atomic_artifact(
            parent_fd,
            path,
        )
        with os.fdopen(descriptor, "wb") as output_file:
            descriptor = None
            output_file.write(payload)
            output_file.flush()
            os.fsync(output_file.fileno())
            next_metadata = os.fstat(output_file.fileno())
        update = _FileUpdate(
            path=path,
            parent_fd=parent_fd,
            parent_identity=_directory_identity(parent_fd),
            original_metadata=original,
            next_name=next_name,
            next_metadata=next_metadata,
        )
        os.close(parent_fd)
        update.parent_fd = None
        return update
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if next_name:
            try:
                os.unlink(next_name, dir_fd=parent_fd)
            except OSError:
                pass
        os.close(parent_fd)
        raise


def _verify_file_update_target(update: _FileUpdate) -> None:
    _acquire_file_update(update)
    assert update.parent_fd is not None
    _verify_visible_artifact_parent(update.path, update.parent_fd)
    try:
        current = os.stat(
            update.path.name,
            dir_fd=update.parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        current = None
    expected = update.original_metadata
    if expected is None:
        if current is not None:
            raise ValidationError(
                f"validation transaction target changed: {update.path}"
            )
        return
    if (
        current is None
        or not stat.S_ISREG(current.st_mode)
        or current.st_dev != expected.st_dev
        or current.st_ino != expected.st_ino
        or current.st_nlink != expected.st_nlink
        or current.st_size != expected.st_size
        or current.st_ctime_ns != expected.st_ctime_ns
        or current.st_mtime_ns != expected.st_mtime_ns
    ):
        raise ValidationError(
            f"validation transaction target changed: {update.path}"
        )


def _acquire_file_update(update: _FileUpdate) -> None:
    if update.closed:
        raise ValidationError(
            f"validation transaction is already closed: {update.path}"
        )
    if update.parent_fd is not None:
        return
    if update.anchor_fd is not None:
        update.parent_fd = update.anchor_fd
        return
    parent_fd = _open_real_directory(update.path.parent)
    try:
        if _directory_identity(parent_fd) != update.parent_identity:
            raise ValidationError(
                "validation transaction parent was replaced: "
                f"{update.path.parent}"
            )
    except BaseException:
        os.close(parent_fd)
        raise
    update.parent_fd = parent_fd


def _release_file_update(update: _FileUpdate) -> None:
    if update.parent_fd is not None:
        if update.parent_fd != update.anchor_fd:
            os.close(update.parent_fd)
        update.parent_fd = None


def _commit_file_update(update: _FileUpdate) -> None:
    try:
        _verify_file_update_target(update)
        assert update.parent_fd is not None
        visible_next = os.stat(
            update.next_name,
            dir_fd=update.parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(visible_next.st_mode)
            or visible_next.st_dev != update.next_metadata.st_dev
            or visible_next.st_ino != update.next_metadata.st_ino
            or visible_next.st_nlink != update.next_metadata.st_nlink
            or visible_next.st_size != update.next_metadata.st_size
            or visible_next.st_ctime_ns
            != update.next_metadata.st_ctime_ns
            or visible_next.st_mtime_ns
            != update.next_metadata.st_mtime_ns
        ):
            raise ValidationError(
                "validation transaction staged file changed: "
                f"{update.path}"
            )
        if update.original_metadata is not None:
            descriptor, update.backup_name = _create_atomic_artifact(
                update.parent_fd,
                update.path.with_name(
                    f".{update.path.name}.previous"
                ),
            )
            os.close(descriptor)
            os.replace(
                update.path.name,
                update.backup_name,
                src_dir_fd=update.parent_fd,
                dst_dir_fd=update.parent_fd,
            )
            update.backed_up = True
        os.replace(
            update.next_name,
            update.path.name,
            src_dir_fd=update.parent_fd,
            dst_dir_fd=update.parent_fd,
        )
        update.next_name = ""
        update.installed = True
        installed = os.stat(
            update.path.name,
            dir_fd=update.parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(installed.st_mode)
            or installed.st_dev != update.next_metadata.st_dev
            or installed.st_ino != update.next_metadata.st_ino
            or installed.st_nlink != update.next_metadata.st_nlink
            or installed.st_size != update.next_metadata.st_size
            or installed.st_mtime_ns
            != update.next_metadata.st_mtime_ns
        ):
            raise ValidationError(
                "validation staged file changed during publication: "
                f"{update.path}"
            )
    finally:
        _release_file_update(update)


def _rollback_file_update(update: _FileUpdate) -> None:
    _acquire_file_update(update)
    try:
        assert update.parent_fd is not None
        if update.backed_up:
            os.replace(
                update.backup_name,
                update.path.name,
                src_dir_fd=update.parent_fd,
                dst_dir_fd=update.parent_fd,
            )
            update.backup_name = ""
            update.backed_up = False
            update.installed = False
        elif update.installed:
            os.unlink(
                update.path.name,
                dir_fd=update.parent_fd,
            )
            update.installed = False
    finally:
        _release_file_update(update)


def _finalize_file_update(update: _FileUpdate) -> None:
    if update.closed:
        return
    try:
        _acquire_file_update(update)
        assert update.parent_fd is not None
        for name in (update.backup_name, update.next_name):
            if not name:
                continue
            try:
                os.unlink(name, dir_fd=update.parent_fd)
            except OSError:
                pass
    except ValidationError:
        pass
    finally:
        update.closed = True
        _release_file_update(update)


def _read_stage_artifact(
    stage: _CaseArtifactStage,
    name: str,
    *,
    maximum_bytes: int,
) -> bytes:
    _verify_case_artifact_stage(stage)
    assert stage.stage_fd is not None
    descriptor = os.open(
        name,
        (
            os.O_RDONLY
            | os.O_NONBLOCK
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        ),
        dir_fd=stage.stage_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > maximum_bytes
        ):
            raise ValidationError(
                f"invalid staged validation artifact: {stage.path / name}"
            )
        chunks = []
        consumed = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            consumed += len(chunk)
            if consumed > maximum_bytes:
                raise ValidationError(
                    f"staged validation artifact exceeds {maximum_bytes} "
                    f"bytes: {stage.path / name}"
                )
            chunks.append(chunk)
        current = os.fstat(descriptor)
        if (
            current.st_dev != metadata.st_dev
            or current.st_ino != metadata.st_ino
            or current.st_size != metadata.st_size
            or current.st_ctime_ns != metadata.st_ctime_ns
            or current.st_mtime_ns != metadata.st_mtime_ns
        ):
            raise ValidationError(
                f"staged validation artifact changed while reading: "
                f"{stage.path / name}"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)
        _release_case_artifact_stage(stage)


def _publish_case_artifact_stage(
    case_dir: Path,
    stage: _CaseArtifactStage,
) -> None:
    if case_dir != stage.case_dir:
        raise ValidationError(
            f"validation report stage belongs to a different case: "
            f"{stage.path}"
        )
    artifact_payload = _read_stage_artifact(
        stage,
        DISAGREEMENT_ARTIFACT_NAME,
        maximum_bytes=MAX_REPORT_ARTIFACT_BYTES,
    )
    directory_update = _prepare_case_directory_update(stage)
    file_update = _prepare_file_update(
        case_dir / DISAGREEMENT_ARTIFACT_NAME,
        artifact_payload,
    )
    updates: list[tuple[str, Any]] = [
        ("directory", directory_update),
        ("file", file_update),
    ]
    try:
        _commit_case_directory_update(directory_update)
        _commit_file_update(file_update)
    except BaseException:
        for kind, update in reversed(updates):
            try:
                if kind == "directory":
                    _rollback_case_directory_update(update)
                else:
                    _rollback_file_update(update)
            except OSError:
                pass
        raise
    finally:
        _finalize_case_directory_update(directory_update)
        _finalize_file_update(file_update)


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


def _read_json_artifact(
    path: Path,
    *,
    missing_ok: bool = False,
) -> Any:
    """Read one regular JSON artifact through an anchored parent directory."""
    try:
        directory_fd = _open_real_directory(path.parent)
    except ValidationError:
        if missing_ok and not path.parent.exists():
            return None
        raise
    descriptor: int | None = None
    try:
        flags = (
            os.O_RDONLY
            | os.O_NONBLOCK
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(path.name, flags, dir_fd=directory_fd)
        except FileNotFoundError:
            if missing_ok:
                return None
            raise
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValidationError(
                f"validation JSON artifact must be a regular file: {path}"
            )
        if metadata.st_nlink != 1:
            raise ValidationError(
                f"validation JSON artifact must not be a hard link: {path}"
            )
        if metadata.st_size > MAX_REPORT_ARTIFACT_BYTES:
            raise ValidationError(
                "validation JSON artifact exceeds "
                f"{MAX_REPORT_ARTIFACT_BYTES} bytes: {path}"
            )
        with os.fdopen(descriptor, "rb") as input_file:
            descriptor = None
            try:
                payload = input_file.read(MAX_REPORT_ARTIFACT_BYTES + 1)
                if len(payload) > MAX_REPORT_ARTIFACT_BYTES:
                    raise ValidationError(
                        "validation JSON artifact exceeds "
                        f"{MAX_REPORT_ARTIFACT_BYTES} bytes: {path}"
                    )
                loaded = json.loads(
                    payload.decode("utf-8"),
                    parse_constant=_reject_nonstandard_json_constant,
                )
            except (UnicodeError, ValueError, RecursionError) as exc:
                raise ValidationError(
                    f"invalid validation JSON artifact {path}: {exc}"
                ) from exc
        _validate_report_json_depth(loaded, path=path)
        _verify_visible_regular_artifact(
            path,
            directory_fd,
            metadata,
            operation="reading",
        )
        _verify_visible_artifact_parent(path, directory_fd)
        return loaded
    except (OSError, ValueError) as exc:
        raise ValidationError(
            f"cannot securely read validation JSON artifact {path}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)


def _atomic_write_json(
    path: Path,
    value: Any,
    *,
    maximum_depth: int = MAX_REPORT_JSON_DEPTH,
) -> None:
    payload = _json_artifact_payload(
        path,
        value,
        maximum_depth=maximum_depth,
    )
    _atomic_write_bytes(path, payload)


def _json_artifact_payload(
    path: Path,
    value: Any,
    *,
    maximum_depth: int = MAX_REPORT_JSON_DEPTH,
) -> bytes:
    _validate_report_json_depth(
        value,
        path=path,
        maximum_depth=maximum_depth,
    )
    try:
        rendered = json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValidationError(
            f"invalid validation JSON artifact for {path}: {exc}"
        ) from exc
    payload = rendered.encode("utf-8")
    if len(payload) > MAX_REPORT_ARTIFACT_BYTES:
        raise ValidationError(
            "validation JSON artifact exceeds "
            f"{MAX_REPORT_ARTIFACT_BYTES} bytes: {path}"
        )
    return payload


def _atomic_write_validation_result(path: Path, value: Any) -> None:
    _atomic_write_json(
        path,
        value,
        maximum_depth=MAX_VALIDATION_RESULT_JSON_DEPTH,
    )


@dataclass(frozen=True)
class EnvironmentSelection:
    base_python: str
    names_and_paths: tuple[tuple[str, str], ...]
    overrides: Mapping[str, str]


@dataclass(frozen=True)
class ReferenceSource:
    name: str
    repository: str
    revision: str
    relative_checkout: Path
    entrypoint: Path


@dataclass(frozen=True)
class ReferenceSourceSelection:
    environment: Mapping[str, str]
    elf_reference_repo: Path | None = None


ELF_SOURCE = ReferenceSource(
    name="ELF",
    repository="https://github.com/lillian039/ELF.git",
    revision="b29d8833609e9ab7f67cd9da39435ac5cea04837",
    relative_checkout=Path("elf/reference/ELF-b29d8833609e"),
    entrypoint=Path("src"),
)
SANA_WM_SOURCE = ReferenceSource(
    name="SANA-WM",
    repository="https://github.com/NVlabs/Sana.git",
    revision="59629fdf790850797cb657bad014fce432bd713d",
    relative_checkout=Path("sana_wm/reference/Sana-59629fdf7908"),
    entrypoint=Path("inference_video_scripts/wm/inference_sana_wm.py"),
)


def _validate_model_spec(path: Path, name: Any, spec: Any) -> None:
    if not isinstance(name, str) or not isinstance(spec, dict):
        raise ValidationError(f"{path}: invalid model binding {name!r}")
    not_compared_reason = spec.get("not_compared_reason")
    if not_compared_reason is not None:
        if not isinstance(not_compared_reason, str) or not not_compared_reason.strip():
            raise ValidationError(
                f"{path}: {name}.not_compared_reason must be a non-empty string"
            )
        if "default" in spec or "workloads" in spec:
            raise ValidationError(
                f"{path}: {name} cannot declare workloads while marked not compared"
            )
        return
    workloads = spec.get("workloads")
    default = spec.get("default")
    valid_workloads = (
        isinstance(workloads, list)
        and bool(workloads)
        and all(isinstance(item, str) and item for item in workloads)
    )
    if not valid_workloads:
        raise ValidationError(f"{path}: {name}.workloads must contain names")
    if "e2e" in workloads:
        raise ValidationError(
            f"{path}: {name}.workloads cannot use e2e; reference consistency "
            "requires aligned reference and TRTMC outputs"
        )
    if default not in workloads:
        raise ValidationError(f"{path}: {name}.default must be one of {name}.workloads")
    reference_cache_identity = spec.get("reference_cache_identity")
    if reference_cache_identity is not None and (
        not isinstance(reference_cache_identity, str)
        or not reference_cache_identity.strip()
    ):
        raise ValidationError(
            f"{path}: {name}.reference_cache_identity must be a non-empty string"
        )


def _validate_sample_limits(path: Path, raw: Mapping[str, Any]) -> None:
    sample_limits = raw.get("sample_limits")
    if not isinstance(sample_limits, dict) or not sample_limits:
        raise ValidationError(f"{path}: sample_limits must be a non-empty mapping")
    for workload, limit in sample_limits.items():
        if not isinstance(workload, str) or not workload:
            raise ValidationError(f"{path}: invalid sample-limit workload {workload!r}")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValidationError(f"{path}: sample_limits.{workload} must be a positive integer")


def load_catalog(path: Path = DEFAULT_CATALOG) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValidationError(f"{path}: expected version: 1")
    models = raw.get("models")
    if not isinstance(models, dict) or not models:
        raise ValidationError(f"{path}: models must be a non-empty mapping")
    _validate_sample_limits(path, raw)
    for name, spec in models.items():
        _validate_model_spec(path, name, spec)
    return raw


def ready_model_names(models_root: Path = DEFAULT_MODELS) -> tuple[str, ...]:
    models = task_eval.load_manifest_records(models_root)
    return tuple(
        sorted(
            str(model["name"])
            for model in models
            if not model["requires_multi_device"]
            and not model.get("skip")
            and model.get("ci_tier") != "l0_only"
        )
    )


def audit_catalog(
    catalog: Mapping[str, Any],
    *,
    ready_models: Iterable[str],
    suite_names: Iterable[str],
) -> None:
    models = catalog["models"]
    ready = set(ready_models)
    configured = set(models)
    missing = sorted(ready - configured)
    stale = sorted(configured - ready)
    if missing or stale:
        details = []
        if missing:
            details.append(f"missing ready models: {', '.join(missing)}")
        if stale:
            details.append(f"non-ready or unknown models: {', '.join(stale)}")
        raise ValidationError("; ".join(details))

    known_workloads = set(suite_names)
    unknown = sorted(
        {
            workload
            for spec in models.values()
            for workload in spec.get("workloads", [])
            if workload not in known_workloads
        }
    )
    if unknown:
        raise ValidationError(f"unknown workloads: {', '.join(unknown)}")

    declared_sampled = {
        workload
        for spec in models.values()
        for workload in spec.get("workloads", [])
    }
    configured_sampled = set(catalog["sample_limits"])
    missing_limits = sorted(declared_sampled - configured_sampled)
    stale_limits = sorted(configured_sampled - declared_sampled)
    if missing_limits or stale_limits:
        details = []
        if missing_limits:
            details.append(f"missing sample limits: {', '.join(missing_limits)}")
        if stale_limits:
            details.append(f"unused sample limits: {', '.join(stale_limits)}")
        raise ValidationError("; ".join(details))


def audit_workload_compatibility(
    catalog: Mapping[str, Any],
    *,
    suites: Mapping[str, dict[str, Any]],
    task_models: Mapping[str, dict[str, Any]],
) -> None:
    incompatible = []
    reference_cache_contracts: dict[str, set[tuple[str, ...]]] = {}
    for model_name, spec in catalog["models"].items():
        for workload in spec.get("workloads", []):
            matched, reason = task_eval.suite_match_reason(
                suites[workload],
                task_models[model_name],
            )
            if not matched:
                incompatible.append(f"{model_name}/{workload}: {reason}")
            reference_cache_identity = str(
                spec.get("reference_cache_identity", "") or ""
            )
            if reference_cache_identity:
                model = task_models[model_name]
                contract = (
                    str(model.get("hf_id", "") or ""),
                    str(model.get("hf_revision", "") or ""),
                    str(model.get("family", "") or ""),
                    str(model.get("reference_backend", "") or ""),
                    str(model.get("reference_family", "") or ""),
                    workload,
                )
                reference_cache_contracts.setdefault(
                    reference_cache_identity,
                    set(),
                ).add(contract)
    for identity, contracts in sorted(reference_cache_contracts.items()):
        if len(contracts) > 1:
            incompatible.append(
                f"reference cache identity {identity!r} spans "
                "different reference contracts"
            )
    if incompatible:
        raise ValidationError("incompatible model/workload bindings: " + "; ".join(incompatible))


def resolve_binding(
    catalog: Mapping[str, Any],
    model: str,
    workload: str | None = None,
) -> Binding:
    models = catalog["models"]
    if model not in models:
        raise ValidationError(f"unknown or unsupported model: {model}")
    spec = models[model]
    not_compared_reason = str(spec.get("not_compared_reason", "") or "")
    if not_compared_reason:
        if workload:
            raise ValidationError(
                f"model {model} has no reference-consistency workloads: "
                f"{not_compared_reason}"
            )
        return Binding(
            model=model,
            workload=None,
            not_compared_reason=not_compared_reason,
        )
    selected = workload or spec["default"]
    if selected not in spec["workloads"]:
        available = ", ".join(spec["workloads"])
        raise ValidationError(
            f"model {model} does not declare workload {selected}; available: {available}"
        )
    return Binding(
        model=model,
        workload=selected,
        reference_cache_identity=str(
            spec.get("reference_cache_identity", "") or ""
        ),
    )


def resolve_sample_limit(
    catalog: Mapping[str, Any],
    binding: Binding,
    explicit_limit: int | None,
) -> int:
    if explicit_limit is not None and explicit_limit < 0:
        raise ValidationError("--limit must be zero or greater")
    if not binding.runnable:
        return 0
    if explicit_limit is not None:
        return explicit_limit
    assert binding.workload is not None
    return int(catalog["sample_limits"][binding.workload])


def _task_eval_models(models_root: Path) -> dict[str, dict[str, Any]]:
    return {str(model["name"]): model for model in task_eval.load_manifest_records(models_root)}


def _declared_profile(
    *,
    family: str,
    runtime_strategy: str,
    reference_backend: str,
    execution_profiles: Mapping[str, str] | None,
) -> str:
    profiles = normalize_execution_profiles(
        execution_profiles,
        family=family,
        runtime_strategy=runtime_strategy,
        reference_backend=reference_backend,
    )
    profile = profiles["reference"]
    if profile == DEFAULT_PROFILE:
        return COMMON_REFERENCE_PROFILE
    return profile


def _binding_profiles(
    binding: Binding,
    *,
    task_models: Mapping[str, dict[str, Any]],
) -> tuple[str, ...]:
    if not binding.runnable:
        raise ValidationError(
            f"model {binding.model} has no reference-consistency workload"
        )
    model = task_models[binding.model]
    profile = _declared_profile(
        family=str(model.get("family", "") or ""),
        runtime_strategy=str(model.get("runtime_strategy", "") or ""),
        reference_backend=str(model.get("reference_backend", "") or ""),
        execution_profiles=model.get("execution_profiles"),
    )
    return (
        (COMMON_REFERENCE_PROFILE,)
        if profile == COMMON_REFERENCE_PROFILE
        else (COMMON_REFERENCE_PROFILE, profile)
    )


def ensure_environments(
    profile_names: Iterable[str],
    base_python: str,
) -> EnvironmentSelection:
    names_and_paths = []
    overrides = {}
    selected_base = base_python

    def announce_create(name: str) -> None:
        print(f"Creating reference environment: {name}", flush=True)

    for name in profile_names:
        path = resolve_profile_python(
            name,
            base_python,
            on_create=announce_create,
        )
        print(f"Using reference environment: {path}", flush=True)
        names_and_paths.append((name, path))
        if name == COMMON_REFERENCE_PROFILE:
            selected_base = path
        elif name != DEFAULT_PROFILE:
            overrides[profile_env_var(name)] = path
    return EnvironmentSelection(
        base_python=selected_base,
        names_and_paths=tuple(names_and_paths),
        overrides=overrides,
    )


def _ensure_reference_source(source: ReferenceSource, cache_root: Path) -> Path:
    checkout = cache_root / source.relative_checkout
    entrypoint = checkout / source.entrypoint
    if entrypoint.exists():
        print(f"Using reference source: {checkout}", flush=True)
        return checkout
    if checkout.exists():
        raise ValidationError(f"Incomplete cached {source.name} reference: {checkout}")

    checkout.parent.mkdir(parents=True, exist_ok=True)
    print(f"Creating reference source: {source.name}", flush=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{checkout.name}-",
            dir=checkout.parent,
        ) as temporary:
            staged = Path(temporary) / "checkout"
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    source.repository,
                    str(staged),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(staged),
                    "checkout",
                    "--detach",
                    source.revision,
                ],
                check=True,
            )
            if not (staged / source.entrypoint).exists():
                raise ValidationError(
                    f"Pinned {source.name} checkout is missing {source.entrypoint}"
                )
            staged.rename(checkout)
    except subprocess.CalledProcessError as exc:
        raise ValidationError(
            f"Could not prepare pinned {source.name} reference "
            f"{source.revision}: git exited with code {exc.returncode}"
        ) from exc
    print(f"Using reference source: {checkout}", flush=True)
    return checkout


def ensure_reference_sources(
    family: str,
    cache_root: Path,
) -> ReferenceSourceSelection:
    environment = {"TRTMC_STORAGE_ROOT": str(cache_root)}
    if family == "elf_flow":
        checkout = _ensure_reference_source(ELF_SOURCE, cache_root)
        return ReferenceSourceSelection(
            environment=environment,
            elf_reference_repo=checkout,
        )
    if family == "sana_wm":
        checkout = _ensure_reference_source(SANA_WM_SOURCE, cache_root)
        environment["SANA_WM_SCRIPT"] = str(checkout / SANA_WM_SOURCE.entrypoint)
    return ReferenceSourceSelection(environment=environment)


def _dataset_path(suite: Mapping[str, Any], dataset_root: Path | None) -> Path:
    raw = str(suite.get("dataset", {}).get("default_path", "") or "")
    if not raw:
        raise ValidationError(f"workload {suite.get('id')} has no default dataset path")
    path = Path(raw)
    if dataset_root is None:
        return path
    try:
        relative = path.relative_to("/mnt/data")
    except ValueError:
        relative = Path(path.name)
    return dataset_root / relative


def _run_subprocess(command: Sequence[str], log_path: Path, env: Mapping[str, str]) -> int:
    _ensure_real_directory(
        log_path.parent,
        description="validation worker log parent",
    )
    directory_fd = _open_real_directory(log_path.parent)
    descriptor: int | None = None
    temporary_name = ""
    final_metadata: os.stat_result | None = None
    completed: subprocess.CompletedProcess[str] | None = None
    execution_error: BaseException | None = None
    try:
        _validate_atomic_destination(directory_fd, log_path.name, log_path)
        descriptor, temporary_name = _create_atomic_artifact(
            directory_fd,
            log_path,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            descriptor = None
            output.write(f"$ {shlex.join(command)}\n")
            output.flush()
            try:
                completed = subprocess.run(
                    list(command),
                    check=False,
                    text=True,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    env=dict(env),
                )
            except BaseException as exc:
                execution_error = exc
                output.write(
                    "\nValidation command could not complete: "
                    f"{type(exc).__name__}: {exc}\n"
                )
            output.flush()
            os.fsync(output.fileno())
            final_metadata = os.fstat(output.fileno())
        os.replace(
            temporary_name,
            log_path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_name = ""
        assert final_metadata is not None
        _verify_visible_regular_artifact(
            log_path,
            directory_fd,
            final_metadata,
            operation="publishing worker log",
            check_ctime=False,
        )
        _verify_visible_artifact_parent(log_path, directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except OSError:
                pass
        os.close(directory_fd)
    if execution_error is not None:
        raise execution_error
    assert completed is not None
    return completed.returncode


def _source_environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    roots = [str(PYTHON_ROOT), str(REPO_ROOT)]
    if existing:
        roots.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    return environment


def _comparison_command(
    binding: Binding,
    *,
    case_dir: Path,
    dataset: Path,
    arguments: argparse.Namespace,
    reference_python: str,
    reference_sources: ReferenceSourceSelection | None = None,
) -> list[str]:
    work_root = case_dir / "validation"
    workload = _required_workload(binding)
    command = [
        reference_python,
        str(REPO_ROOT / "tools" / "trtmc_compare.py"),
        "--suite",
        workload,
        "--dataset",
        str(dataset),
        "--model",
        binding.model,
        "--work-root",
        str(work_root),
        "--engine-dir",
        str(arguments.engine_dir),
        "--trtmc-binary",
        str(arguments.trtmc_binary),
        "--benchmark-binary",
        str(arguments.benchmark_binary),
        "--hf-python",
        reference_python,
        "--reference-cache-dir",
        str(arguments.reference_cache_dir),
        "--replace-bundle-on-build",
        "--single-device-only",
        "--include-waived",
        "--fail-fast",
    ]
    if arguments.limit:
        command.extend(["--limit", str(arguments.limit)])
    if binding.reference_cache_identity:
        command.extend(
            [
                "--reference-cache-identity",
                binding.reference_cache_identity,
            ]
        )
    if arguments.force_hf:
        command.append("--force-hf")
    if arguments.force_build:
        command.append("--force-build")
    if arguments.no_build:
        command.append("--require-prebuilt-bundles")
    if arguments.local_files_only:
        command.append("--local-files-only")
    if arguments.backend_dir:
        command.extend(["--backend-dir", str(arguments.backend_dir)])
    if arguments.model_plugin_dir:
        command.extend(["--model-plugin-dir", str(arguments.model_plugin_dir)])
    if arguments.cuda_visible_devices:
        command.extend(["--cuda-visible-devices", arguments.cuda_visible_devices])
    if reference_sources and reference_sources.elf_reference_repo:
        command.extend(
            [
                "--elf-reference-repo",
                str(reference_sources.elf_reference_repo),
            ]
        )
    return command


MAX_REPRO_COMMANDS_PER_BACKEND = 3
_FAILED_SAMPLE_STATUSES = {"disagreement", "fail", "failed", "mismatch"}
_FAILED_SAMPLE_FIELDS = (
    "agreement_match",
    "exact_match",
    "passed",
    "top1_agreement",
    "transcript_exact",
)
_SAMPLE_ID_FIELDS = ("sample_id", "case_id", "id", "name")


def _command_record_from_log_line(
    line: str,
    *,
    strict_json: bool = False,
) -> tuple[str, str]:
    if line.startswith("$ ") and not strict_json:
        rendered = line[2:].strip()
        if "\x00" in rendered:
            raise ValidationError(
                "validation command log contains a NUL character"
            )
        return rendered, ""
    try:
        data = json.loads(
            line,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except json.JSONDecodeError as exc:
        if strict_json:
            raise ValidationError(
                f"invalid native validation command record: {exc}"
            ) from exc
        return "", ""
    except (ValueError, RecursionError) as exc:
        raise ValidationError(
            f"invalid validation command log record: {exc}"
        ) from exc
    _validate_report_json_depth(data, path=Path("<command-log-record>"))
    if not isinstance(data, dict):
        if strict_json:
            raise ValidationError(
                "native validation command records must be JSON objects"
            )
        return "", ""
    command = data.get("command")
    if strict_json:
        sample_value = data.get("sample_id")
        if (
            not isinstance(sample_value, str)
            or not sample_value.strip()
            or "\x00" in sample_value
        ):
            raise ValidationError(
                "native validation command records require an exact "
                "non-empty, NUL-free sample_id string"
            )
        sample_id = sample_value
    else:
        sample_id = next(
            (
                str(data[name])
                for name in _SAMPLE_ID_FIELDS
                if data.get(name) is not None
            ),
            "",
        )
    if isinstance(command, list) and command:
        if any(
            not isinstance(token, str) or "\x00" in token
            for token in command
        ):
            raise ValidationError(
                "validation command list tokens must be NUL-free strings"
            )
        if strict_json and not command[0].strip():
            raise ValidationError(
                "native validation command executable must be a "
                "non-empty, non-whitespace string"
            )
        return shlex.join(command), sample_id
    if isinstance(command, str):
        rendered = command.strip()
        if "\x00" in command:
            raise ValidationError(
                "validation command contains a NUL character"
            )
        if rendered:
            return rendered, sample_id
    if strict_json:
        raise ValidationError(
            "native validation command records must contain a non-empty "
            "command string or list"
        )
    return "", ""


def _command_from_log_line(line: str) -> str:
    return _command_record_from_log_line(line)[0]


def _sample_id(record: Mapping[str, Any]) -> str:
    return next(
        (str(record[name]) for name in _SAMPLE_ID_FIELDS if record.get(name) is not None),
        "",
    )


def _explicit_disagreement_id(data: Mapping[str, Any]) -> str:
    disagreements = data.get("disagreements", [])
    if not isinstance(disagreements, list):
        return ""
    for item in disagreements:
        if isinstance(item, dict) and _sample_id(item):
            return _sample_id(item)
    return ""


def _record_is_disagreement(record: Mapping[str, Any]) -> bool:
    status = str(record.get("status", "") or "").lower()
    if status in _FAILED_SAMPLE_STATUSES:
        return True
    return any(record.get(name) is False for name in _FAILED_SAMPLE_FIELDS)


def _failed_sample_id(data: Any) -> str:
    if isinstance(data, list):
        for item in data:
            failed = _failed_sample_id(item)
            if failed:
                return failed
        return ""
    if not isinstance(data, dict):
        return ""
    if _record_is_disagreement(data) and _sample_id(data):
        return _sample_id(data)
    for value in data.values():
        failed = _failed_sample_id(value)
        if failed:
            return failed
    return ""


def _command_artifact_text(
    path: Path,
    read_artifact: Callable[[Path, str, int], str | None] | None,
    *,
    errors: str = "strict",
    maximum_bytes: int = MAX_REPORT_ARTIFACT_BYTES,
) -> str | None:
    if read_artifact is not None:
        return read_artifact(path, errors, maximum_bytes)
    try:
        with path.open("rb") as artifact:
            payload = artifact.read(maximum_bytes + 1)
    except FileNotFoundError:
        return None
    if len(payload) > maximum_bytes:
        raise ValidationError(
            f"validation command artifact exceeds {maximum_bytes} bytes: "
            f"{path}"
        )
    try:
        return payload.decode("utf-8", errors=errors)
    except UnicodeError as exc:
        raise ValidationError(
            f"validation command artifact is not UTF-8: {path}: {exc}"
        ) from exc


def _parse_command_json(text: str, *, path: Path) -> Any:
    try:
        loaded = json.loads(
            text,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValidationError(
            f"invalid validation command artifact {path}: {exc}"
        ) from exc
    _validate_report_json_depth(loaded, path=path)
    return loaded


def _first_disagreement_id(
    work_dir: Path,
    read_artifact: Callable[[Path, str, int], str | None] | None = None,
) -> str:
    for name in ("summary.json", "eval_result.json"):
        path = work_dir / name
        text = _command_artifact_text(path, read_artifact)
        if text is None:
            continue
        data = _parse_command_json(text, path=path)
        if isinstance(data, dict) and isinstance(data.get("disagreements"), list):
            return _explicit_disagreement_id(data)
        failed = _failed_sample_id(data)
        if failed:
            return failed
    return ""


def _prepared_sample_ids(
    work_dir: Path,
    read_artifact: Callable[[Path, str, int], str | None] | None = None,
) -> list[str]:
    prompts = work_dir / "prompts.jsonl"
    text = _command_artifact_text(prompts, read_artifact)
    if text is None:
        return []
    sample_ids = []
    for index, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        record = _parse_command_json(line, path=prompts)
        if isinstance(record, dict):
            sample_ids.append(_sample_id(record) or f"sample-{index}")
    return sample_ids


def _sample_ids_match(candidate: str, target: str) -> bool:
    return bool(
        candidate
        and target
        and (
            candidate == target
            or candidate.startswith(f"{target}:")
            or target.startswith(f"{candidate}:")
        )
    )


def _summarize_command_log(
    path: Path,
    *,
    sample_ids: Sequence[str],
    target_sample_id: str,
    read_artifact: Callable[[Path, str, int], str | None] | None = None,
    maximum_bytes: int = MAX_COMMAND_LOG_TOTAL_BYTES,
) -> tuple[int, str, int]:
    count = 0
    first = ""
    selected = ""
    strict_json = path.name.endswith("_native_commands.jsonl")
    text = _command_artifact_text(
        path,
        read_artifact,
        errors="strict" if strict_json else "replace",
        maximum_bytes=maximum_bytes,
    )
    if text is None:
        return count, first, 0
    consumed_bytes = min(
        maximum_bytes,
        len(text.encode("utf-8")),
    )
    for line in text.splitlines():
        if strict_json and not line.strip():
            continue
        command, logged_sample_id = _command_record_from_log_line(
            line,
            strict_json=strict_json,
        )
        if not command:
            continue
        indexed_sample_id = sample_ids[count] if count < len(sample_ids) else ""
        command_sample_id = logged_sample_id or indexed_sample_id
        count += 1
        first = first or command
        if _sample_ids_match(command_sample_id, target_sample_id):
            selected = command
    return count, selected or first, consumed_bytes


def _command_log_kind(
    path: Path,
    *,
    has_native_reference: bool,
    has_native_reference_commands: bool,
    has_native_trtmc: bool,
) -> str | None:
    if has_native_reference and path.name == "hf_run.log":
        return None
    if has_native_reference_commands and path.name == "hf_native_run.log":
        return None
    if has_native_trtmc and path.name == "trtfb_run.log":
        return None
    return "hf" if "hf" in path.name.lower() else "trtmc"


def _relocate_cached_reference_command(command: str, work_dir: Path) -> str:
    """Point a cached native reference command at the current materialized run."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return command
    original_work_dir = None
    for flag in ("--manifest", "--prompts", "--answers"):
        if flag in tokens:
            index = tokens.index(flag)
            if index + 1 < len(tokens):
                candidate = Path(tokens[index + 1])
                if candidate.is_absolute():
                    original_work_dir = candidate.parent
                    break
    if original_work_dir is None:
        return command
    original_prefix = str(original_work_dir)
    current_prefix = str(work_dir.resolve())
    relocated = [
        (
            current_prefix + token[len(original_prefix) :]
            if token == original_prefix or token.startswith(original_prefix + os.sep)
            else token
        )
        for token in tokens
    ]
    return shlex.join(relocated)


def _collect_command_logs(
    root: Path,
    *,
    log_paths: Sequence[Path],
    sample_ids: Sequence[str],
    representative_id: str,
    read_artifact: Callable[[Path, str, int], str | None] | None = None,
) -> tuple[dict[str, list[str]], dict[str, int], dict[str, list[str]]]:
    commands: dict[str, list[str]] = {"hf": [], "trtmc": []}
    counts = {"hf": 0, "trtmc": 0}
    logs: dict[str, list[str]] = {"hf": [], "trtmc": []}
    if len(log_paths) > MAX_COMMAND_LOG_FILES:
        raise ValidationError(
            "validation command log count exceeds "
            f"{MAX_COMMAND_LOG_FILES}: {root}"
        )
    remaining_bytes = MAX_COMMAND_LOG_TOTAL_BYTES
    has_native_reference = any(
        path.name in {"hf_native_run.log", "hf_native_commands.jsonl"}
        for path in log_paths
    )
    has_native_reference_commands = any(
        path.name == "hf_native_commands.jsonl" for path in log_paths
    )
    has_native_trtmc = any(
        path.name == "trtfb_native_commands.jsonl" for path in log_paths
    )
    for path in log_paths:
        kind = _command_log_kind(
            path,
            has_native_reference=has_native_reference,
            has_native_reference_commands=has_native_reference_commands,
            has_native_trtmc=has_native_trtmc,
        )
        if kind is None:
            continue
        indexed_sample_ids = sample_ids if path.name == "trtfb_run.log" else ()
        count, representative, consumed_bytes = _summarize_command_log(
            path,
            sample_ids=indexed_sample_ids,
            target_sample_id=representative_id,
            read_artifact=read_artifact,
            maximum_bytes=remaining_bytes,
        )
        remaining_bytes -= consumed_bytes
        if path.name == "hf_native_run.log":
            representative = _relocate_cached_reference_command(
                representative,
                root,
            )
        counts[kind] += count
        if count:
            logs[kind].append(str(path.relative_to(root)))
        _append_unique(commands, kind, representative)
    return commands, counts, logs


def _commands_from_logs(
    root: Path,
    *,
    read_artifact: Callable[[Path, str, int], str | None] | None = None,
    log_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    sample_ids = _prepared_sample_ids(root, read_artifact)
    disagreement_id = _first_disagreement_id(root, read_artifact)
    representative_id = disagreement_id or (sample_ids[0] if sample_ids else "")
    selected_log_paths = (
        sorted(log_paths)
        if log_paths is not None
        else _secure_command_log_paths(root, missing_ok=True)
    )
    commands, counts, logs = _collect_command_logs(
        root,
        log_paths=selected_log_paths,
        sample_ids=sample_ids,
        representative_id=representative_id,
        read_artifact=read_artifact,
    )
    for kind in commands:
        commands[kind] = commands[kind][:MAX_REPRO_COMMANDS_PER_BACKEND]
    return {
        **commands,
        "command_count": counts,
        "commands_shown": {kind: len(values) for kind, values in commands.items()},
        "command_logs": logs,
        "representative": {
            "sample_id": representative_id,
            "reason": "first_disagreement" if disagreement_id else "first_input",
        },
        "prepared_input_count": len(sample_ids),
    }


def _secure_command_log_paths(
    root: Path,
    *,
    missing_ok: bool = False,
) -> list[Path]:
    """Discover command logs without following links below the work directory."""
    try:
        root_fd = _open_real_directory(root)
    except ValidationError:
        if missing_ok:
            try:
                root.lstat()
            except FileNotFoundError:
                return []
        raise
    discovered: list[Path] = []
    visited = 0
    total_bytes = 0

    def is_command_log(name: str) -> bool:
        return (
            name.endswith(".log")
            or name.endswith("_native_commands.jsonl")
        )

    def visit(
        directory_fd: int,
        relative_directory: Path,
        depth: int,
    ) -> None:
        nonlocal visited, total_bytes
        if depth > MAX_COMMAND_LOG_DEPTH:
            raise ValidationError(
                "validation command artifact directory nesting exceeds "
                f"{MAX_COMMAND_LOG_DEPTH}: {root}"
            )
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                visited += 1
                if visited > MAX_COMMAND_LOG_DISCOVERY_ENTRIES:
                    raise ValidationError(
                        "validation command artifact scan exceeds "
                        f"{MAX_COMMAND_LOG_DISCOVERY_ENTRIES} entries: "
                        f"{root}"
                    )
                if entry.is_dir(follow_symlinks=False):
                    child_fd = os.open(
                        entry.name,
                        _secure_directory_flags(),
                        dir_fd=directory_fd,
                    )
                    try:
                        visit(
                            child_fd,
                            relative_directory / entry.name,
                            depth + 1,
                        )
                    finally:
                        os.close(child_fd)
                    continue
                if not is_command_log(entry.name):
                    continue
                metadata = os.stat(
                    entry.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValidationError(
                        "validation command artifact must be a regular file: "
                        f"{root / relative_directory / entry.name}"
                    )
                discovered.append(
                    root / relative_directory / entry.name
                )
                if len(discovered) > MAX_COMMAND_LOG_FILES:
                    raise ValidationError(
                        "validation command log count exceeds "
                        f"{MAX_COMMAND_LOG_FILES}: {root}"
                    )
                total_bytes += metadata.st_size
                if total_bytes > MAX_COMMAND_LOG_TOTAL_BYTES:
                    raise ValidationError(
                        "validation command logs exceed "
                        f"{MAX_COMMAND_LOG_TOTAL_BYTES} bytes: {root}"
                    )

    try:
        visit(root_fd, Path(), 0)
    except OSError as exc:
        raise ValidationError(
            f"cannot securely discover validation command artifacts in "
            f"{root}: {exc}"
        ) from exc
    finally:
        os.close(root_fd)
    return sorted(discovered)


def _append_unique(commands: dict[str, list[str]], kind: str, command: str) -> None:
    if command and command not in commands[kind]:
        commands[kind].append(command)


_PRIMARY_COMPARISON_METRICS = (
    "sample_agreement_rate",
    "prediction_agreement_rate",
    "vector_pass_rate",
    "top1_agreement",
    "backend_pixel_agreement",
    "mean_pairwise_ordering_agreement",
    "token_prefix_agreement",
    "token_agreement_rate",
    "exact_match_rate",
)
_PRIMARY_METRIC_BY_MODE = {
    "asr_transcript": "prediction_agreement_rate",
    "continuation": "token_prefix_agreement",
    "diffusion_image_clip_parity": "overall_pass_rate",
    "diffusion_text_parity": "token_agreement_rate",
    "encoder_embedding_parity": "vector_pass_rate",
    "image_classification_parity": "top1_agreement",
    "ocrbench_v2": "prediction_agreement_rate",
    "reranking_parity": "mean_pairwise_ordering_agreement",
    "semantic_segmentation_parity": "backend_pixel_agreement",
    "time_series_parity": "sample_agreement_rate",
}
_COMPARISON_METRICS = (
    *_PRIMARY_COMPARISON_METRICS,
    "overall_pass_rate",
    "passed_count",
    "valid_count",
    "skipped_count",
    "token_id_prefix_agreement",
    "normalized_transcript_exact_agreement_rate",
    "correctness_agreement_rate",
    "divergence_rate",
    "divergent_count",
    "hf_accuracy",
    "trtfb_accuracy",
    "accuracy_delta_trtfb_minus_hf",
    "accuracy_drop_from_hf",
    "hf_top1_accuracy",
    "trtfb_top1_accuracy",
    "top1_accuracy_drop_from_hf",
    "hf_mean_iou",
    "trtfb_mean_iou",
    "backend_mean_iou",
    "mean_iou_drop_from_hf",
    "mean_vector_cosine",
    "min_vector_cosine",
    "mean_pair_cosine_abs_delta",
    "max_pair_cosine_abs_delta",
    "mean_relative_l2",
    "max_relative_l2",
    "max_absolute_error",
)
_EXECUTION_ERROR_FIELDS = ("error", "exception", "traceback", "failure_class")


def _is_comparison_gate_failure(raw_result: Mapping[str, Any]) -> bool:
    failures = raw_result.get("gate_failures")
    return (
        raw_result.get("error_type") == "BenchmarkGateError"
        and isinstance(failures, list)
        and bool(failures)
    )


def _raw_comparison(result: Mapping[str, Any]) -> dict[str, Any]:
    raw_result = result.get("raw_result")
    if isinstance(raw_result, dict) and raw_result:
        return dict(raw_result)
    status = str(result.get("status", "") or "")
    return {"status": status} if status else {}


def _execution_details(
    result: Mapping[str, Any],
    raw_result: Mapping[str, Any],
) -> dict[str, Any]:
    comparison_gate_failure = _is_comparison_gate_failure(raw_result)
    has_error = any(
        raw_result.get(name)
        for name in _EXECUTION_ERROR_FIELDS
        if name != "error" or not comparison_gate_failure
    )
    completed = bool(raw_result) and not has_error
    return {
        "status": "completed" if completed else "error",
        "exit_code": result.get("returncode"),
    }


def _comparison_metrics(raw_result: Mapping[str, Any]) -> dict[str, Any]:
    metrics = {
        name: raw_result[name]
        for name in _COMPARISON_METRICS
        if raw_result.get(name) is not None
    }
    nested = raw_result.get("metrics", {})
    if isinstance(nested, Mapping):
        for name, summary in nested.items():
            if not isinstance(summary, Mapping):
                continue
            mean = summary.get("mean")
            if isinstance(mean, (int, float)) and not isinstance(mean, bool):
                metrics[str(name)] = mean
    return metrics


def _primary_metric(
    mode: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any] | None:
    preferred = _PRIMARY_METRIC_BY_MODE.get(mode)
    if preferred in metrics:
        return {"name": preferred, "value": metrics[preferred]}
    for name in _PRIMARY_COMPARISON_METRICS:
        if name in metrics:
            return {"name": name, "value": metrics[name]}
    return None


def _comparison_details(
    raw_result: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    raw_status = str(raw_result.get("status", "") or "")
    status_by_raw = {
        "pass": "agreement",
        "passed": "agreement",
        "fail": "disagreement",
        "failed": "disagreement",
        "skip": "not_run",
        "skipped": "not_run",
    }
    status = (
        status_by_raw.get(raw_status, "not_run")
        if execution.get("status") == "completed"
        else "not_run"
    )
    metrics = _comparison_metrics(raw_result)
    failures = raw_result.get("gate_failures", [])
    mode = str(raw_result.get("mode", "") or "")
    return {
        "status": status,
        "mode": mode,
        "primary_metric": _primary_metric(mode, metrics),
        "metrics": metrics,
        "failures": failures if isinstance(failures, list) else [],
    }


def _validation_details(
    execution: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, str]:
    if execution.get("status") != "completed":
        return {"status": "failed"}
    status_by_comparison = {
        "agreement": "passed",
        "disagreement": "failed",
        "not_run": "skipped",
    }
    return {"status": status_by_comparison[str(comparison["status"])]}


def _string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValidationError(
            f"validation result {field} must be a list of non-empty strings"
        )
    if any(
        not isinstance(item, str)
        or not item.strip()
        or "\x00" in item
        for item in value
    ):
        raise ValidationError(
            f"validation result {field} must contain only non-empty strings"
        )
    return list(value)


def _normalized_command_count(
    reproduce: Mapping[str, Any],
    kind: str,
    commands: Sequence[str],
) -> int:
    counts = reproduce.get("command_count")
    if counts is None:
        return len(commands)
    if not isinstance(counts, Mapping):
        raise ValidationError(
            "validation result reproduce.command_count must be an object"
        )
    if kind not in counts:
        return len(commands)
    configured = counts[kind]
    if (
        type(configured) is not int
        or configured < len(commands)
    ):
        raise ValidationError(
            "validation result reproduce.command_count."
            f"{kind} must be a non-negative integer at least as large as "
            f"the {kind} command list"
        )
    return configured


def _normalized_command_logs(reproduce: Mapping[str, Any], kind: str) -> list[str]:
    logs = reproduce.get("command_logs")
    if logs is None:
        return []
    if not isinstance(logs, Mapping):
        raise ValidationError(
            "validation result reproduce.command_logs must be an object"
        )
    return _string_list(
        logs.get(kind, []),
        field=f"reproduce.command_logs.{kind}",
    )


def _normalize_reproduction(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(
            "validation result reproduce must be an object"
        )
    reproduce = dict(value)
    all_commands = {
        kind: _string_list(
            reproduce.get(kind, []),
            field=f"reproduce.{kind}",
        )
        for kind in ("hf", "trtmc")
    }
    commands = {
        kind: values[:MAX_REPRO_COMMANDS_PER_BACKEND] for kind, values in all_commands.items()
    }
    dataset = reproduce.get("dataset", {})
    if not isinstance(dataset, Mapping):
        raise ValidationError(
            "validation result reproduce.dataset must be an object"
        )
    dataset = dict(dataset)
    command = dataset.get("command", "")
    if (
        not isinstance(command, str)
        or "\x00" in command
        or (command != "" and not command.strip())
    ):
        raise ValidationError(
            "validation result reproduce.dataset.command must be empty or "
            "a non-whitespace, NUL-free string"
        )
    for field in ("sample_limit", "prepared_input_count"):
        if field not in dataset:
            continue
        configured = dataset[field]
        if type(configured) is not int or configured < 0:
            raise ValidationError(
                f"validation result reproduce.dataset.{field} must be a "
                "non-negative integer"
            )
    representative = reproduce.get("representative", {})
    if not isinstance(representative, Mapping):
        raise ValidationError(
            "validation result reproduce.representative must be an object"
        )
    representative = dict(representative)
    for field in ("sample_id", "reason"):
        if field not in representative:
            continue
        representative_value = representative[field]
        if (
            not isinstance(representative_value, str)
            or "\x00" in representative_value
            or (
                representative_value != ""
                and not representative_value.strip()
            )
        ):
            raise ValidationError(
                "validation result reproduce.representative."
                f"{field} must be empty or a non-whitespace, NUL-free string"
            )
    return {
        "dataset": dataset,
        **commands,
        "command_count": {
            kind: _normalized_command_count(reproduce, kind, all_commands[kind])
            for kind in commands
        },
        "commands_shown": {kind: len(values) for kind, values in commands.items()},
        "command_logs": {kind: _normalized_command_logs(reproduce, kind) for kind in commands},
        "representative": representative,
    }


def _add_dataset_reproduction(
    reproduce: Mapping[str, Any],
    command: str,
    sample_limit: int = 0,
) -> dict[str, Any]:
    result = dict(reproduce)
    prepared_input_count = int(result.pop("prepared_input_count", 0) or 0)
    result["dataset"] = {
        "command": command,
        "sample_limit": sample_limit,
        "prepared_input_count": prepared_input_count,
    }
    return result


def _normalize_status_details(
    value: Any,
    *,
    fallback: Mapping[str, Any],
    allowed_statuses: set[str],
    field: str,
) -> dict[str, Any]:
    details = dict(fallback)
    if isinstance(value, Mapping):
        details.update(value)
    elif value is not None:
        raise ValidationError(
            f"validation result {field} details must be an object"
        )
    status = str(details.get("status", "") or "")
    if status not in allowed_statuses:
        raise ValidationError(
            f"validation result {field}.status must be one of "
            f"{sorted(allowed_statuses)}, got {status!r}"
        )
    details["status"] = status
    return details


def _normalize_execution_result(
    value: Any,
    *,
    fallback: Mapping[str, Any],
) -> dict[str, Any]:
    execution = _normalize_status_details(
        value,
        fallback=fallback,
        allowed_statuses={"completed", "error", "not_run"},
        field="execution",
    )
    if "attempt_count" in execution:
        attempt_count = execution["attempt_count"]
        if type(attempt_count) is not int or attempt_count < 1:
            raise ValidationError(
                "validation result execution.attempt_count must be a "
                "positive integer"
            )
    attempts = execution.get("attempts")
    if attempts is not None and not isinstance(attempts, list):
        raise ValidationError(
            "validation result execution.attempts must be a list"
        )
    if (
        isinstance(attempts, list)
        and "attempt_count" in execution
        and execution["attempt_count"] != len(attempts)
    ):
        raise ValidationError(
            "validation result execution.attempt_count must equal the "
            "number of execution.attempts"
        )
    return execution


def _normalize_comparison_result(
    value: Any,
    *,
    fallback: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = _normalize_status_details(
        value,
        fallback=fallback,
        allowed_statuses={"agreement", "disagreement", "not_run"},
        field="comparison",
    )
    metrics = comparison.get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise ValidationError(
            "validation result comparison.metrics must be an object"
        )
    comparison["metrics"] = dict(metrics)
    failures = comparison.get("failures", [])
    if not isinstance(failures, list):
        raise ValidationError(
            "validation result comparison.failures must be a list"
        )
    primary = comparison.get("primary_metric")
    if primary is not None and not isinstance(primary, Mapping):
        raise ValidationError(
            "validation result comparison.primary_metric must be an object "
            "or null"
        )
    if isinstance(primary, Mapping):
        primary_name = primary.get("name")
        if not isinstance(primary_name, str) or not primary_name:
            raise ValidationError(
                "validation result comparison.primary_metric.name must be "
                "a non-empty string"
            )
        if primary_name not in comparison["metrics"]:
            raise ValidationError(
                "validation result comparison.primary_metric.name must name "
                "an entry in comparison.metrics"
            )
        primary_value = primary.get("value")
        metric_value = comparison["metrics"][primary_name]
        if (
            type(primary_value) is not type(metric_value)
            or primary_value != metric_value
        ):
            raise ValidationError(
                "validation result comparison.primary_metric.value must "
                "exactly match comparison.metrics at primary_metric.name"
            )
        comparison["primary_metric"] = dict(primary)
    comparison["mode"] = str(comparison.get("mode", "") or "")
    return comparison


def _normalize_validation_result(
    value: Any,
    *,
    fallback: Mapping[str, Any],
) -> dict[str, Any]:
    return _normalize_status_details(
        value,
        fallback=fallback,
        allowed_statuses={
            "passed",
            "failed",
            "skipped",
            "not_compared",
        },
        field="validation",
    )


def _normalize_result(result: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    if "schema_version" in normalized:
        schema_version = normalized["schema_version"]
        if (
            not isinstance(schema_version, str)
            or schema_version
            not in {
                "trtmc.validation-result/v1",
                "trtmc.validation-result/v2",
            }
        ):
            raise ValidationError(
                "validation result schema_version must be one of "
                "trtmc.validation-result/v1 or "
                "trtmc.validation-result/v2"
            )
    if "not_compared_reason" in normalized and not isinstance(
        normalized["not_compared_reason"],
        str,
    ):
        raise ValidationError(
            "validation result not_compared_reason must be a string"
        )
    if normalized.get("executor") == "e2e" or isinstance(
        normalized.get("raw_results"),
        list,
    ):
        normalized.update(
            {
                "workload": None,
                "execution": {"status": "not_run", "exit_code": None},
                "comparison": {
                    "status": "not_run",
                    "mode": "",
                    "primary_metric": None,
                    "metrics": {},
                    "failures": [],
                },
                "validation": {"status": "not_compared"},
                "not_compared_reason": LEGACY_E2E_REASON,
            }
        )
    raw_result = _raw_comparison(normalized)
    execution = _normalize_execution_result(
        normalized.get("execution"),
        fallback=_execution_details(normalized, raw_result),
    )
    comparison = _normalize_comparison_result(
        normalized.get("comparison"),
        fallback=_comparison_details(raw_result, execution),
    )
    validation = _normalize_validation_result(
        normalized.get("validation"),
        fallback=_validation_details(execution, comparison),
    )
    reference_environment = normalized.get("reference_environment", [])
    if reference_environment is None:
        reference_environment = []
    if not isinstance(reference_environment, list) or any(
        not isinstance(item, Mapping)
        for item in reference_environment
    ):
        raise ValidationError(
            "validation result reference_environment must be a list of "
            "objects"
        )
    precision_contract = normalized.get("precision_contract")
    if not isinstance(precision_contract, dict):
        candidate = raw_result.get("precision_contract")
        precision_contract = dict(candidate) if isinstance(candidate, dict) else {}
    normalized.update(
        {
            "schema_version": "trtmc.validation-result/v2",
            "execution": execution,
            "comparison": comparison,
            "validation": validation,
            "reproduce": _normalize_reproduction(
                normalized.get("reproduce", {})
            ),
            "reference_environment": [
                dict(item) for item in reference_environment
            ],
        }
    )
    if precision_contract:
        normalized["precision_contract"] = precision_contract
    else:
        normalized.pop("precision_contract", None)
    normalized.pop("returncode", None)
    normalized.pop("status", None)
    _validate_result_status_consistency(normalized)
    return normalized


def _validate_result_status_consistency(
    result: Mapping[str, Any],
) -> None:
    execution = result["execution"]
    comparison = result["comparison"]
    validation = result["validation"]
    execution_status = execution["status"]
    comparison_status = comparison["status"]
    validation_status = validation["status"]
    if result.get("not_compared_reason"):
        expected = ("not_run", "not_run", "not_compared")
    else:
        expected_comparison = (
            comparison_status
            if execution_status == "completed"
            else "not_run"
        )
        expected_validation = (
            "failed"
            if execution_status != "completed"
            else {
                "agreement": "passed",
                "disagreement": "failed",
                "not_run": "skipped",
            }[comparison_status]
        )
        expected = (
            execution_status,
            expected_comparison,
            expected_validation,
        )
    actual = (
        execution_status,
        comparison_status,
        validation_status,
    )
    if actual != expected:
        raise ValidationError(
            "validation result execution/comparison/validation statuses "
            f"are inconsistent: got {actual}, expected {expected}"
        )


def _not_compared_result(binding: Binding) -> dict[str, Any]:
    return _normalize_result(
        {
            "schema_version": "trtmc.validation-result/v2",
            "model": binding.model,
            "workload": None,
            "executor": "not_compared",
            "execution": {"status": "not_run", "exit_code": None},
            "comparison": {
                "status": "not_run",
                "mode": "",
                "primary_metric": None,
                "metrics": {},
                "failures": [],
            },
            "validation": {"status": "not_compared"},
            "not_compared_reason": binding.not_compared_reason,
            "reference_environment": [],
            "reproduce": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


def _write_not_compared_case(binding: Binding, output: Path) -> tuple[dict[str, Any], Path]:
    case_dir = _prepare_case_directory(output, binding)
    result = _not_compared_result(binding)
    comparison = case_dir / "comparison.json"
    _atomic_write_validation_result(comparison, result)
    return result, comparison


def _comparison_result(
    binding: Binding,
    *,
    case_dir: Path,
    returncode: int,
    reference_environment: EnvironmentSelection,
    dataset_command: str,
    sample_limit: int = 0,
) -> dict[str, Any]:
    workload = _required_workload(binding)
    summary_path = case_dir / "validation" / workload / "eval_summary.json"
    raw_result: dict[str, Any] = {}
    summary = _read_json_artifact(summary_path, missing_ok=True)
    if summary is not None:
        if not isinstance(summary, Mapping):
            raise ValidationError(
                f"comparison summary must contain an object: {summary_path}"
            )
        candidates = summary.get("results", [])
        if not isinstance(candidates, list):
            raise ValidationError(
                f"comparison summary results must be a list: {summary_path}"
            )
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            if candidate.get("model") == binding.model:
                raw_result = dict(candidate)
                break
        if not raw_result:
            raw_result = next(
                (dict(candidate) for candidate in candidates if isinstance(candidate, Mapping)),
                {},
            )
    if not raw_result:
        raw_result = {
            "status": "failed",
            "error_type": "ComparisonProcessError",
            "error": (
                f"comparison exited with code {returncode} without writing "
                f"a model result to {summary_path}"
            ),
        }
    status = str(raw_result.get("status", "") or "")
    if status not in {"passed", "failed", "skipped"}:
        status = "passed" if returncode == 0 else "failed"
    work_dir = case_dir / "validation" / workload / binding.model

    def read_command_artifact(
        path: Path,
        errors: str,
        maximum_bytes: int,
    ) -> str | None:
        return _read_case_text_artifact(
            path,
            case_dir=case_dir,
            missing_ok=True,
            errors=errors,
            maximum_bytes=maximum_bytes,
        )

    disagreements = {
        "count": 0,
        "path": DISAGREEMENT_ARTIFACT_NAME,
        "inline_limit": (
            trtmc_disagreements.INLINE_DISAGREEMENT_LIMIT
        ),
        "reference_vanilla_available": False,
        "trtmc_vanilla_available": False,
    }
    return _normalize_result(
        {
            "schema_version": "trtmc.validation-result/v2",
            "model": binding.model,
            "workload": binding.workload,
            "executor": "trtmc_compare",
            "status": status,
            "returncode": returncode,
            "reference_environment": [
                {"name": name, "python": path}
                for name, path in reference_environment.names_and_paths
            ],
            "reproduce": _add_dataset_reproduction(
                _commands_from_logs(
                    work_dir,
                    read_artifact=read_command_artifact,
                    log_paths=_secure_command_log_paths(
                        work_dir,
                        missing_ok=True,
                    ),
                ),
                dataset_command,
                sample_limit,
            ),
            "raw_result": raw_result,
            "raw_result_path": str(summary_path),
            "disagreements": disagreements,
            "execution_log": str(case_dir / "execution.log"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


def run_binding(
    binding: Binding,
    *,
    arguments: argparse.Namespace,
    task_models: Mapping[str, dict[str, Any]],
    suites: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    workload = _required_workload(binding)
    case_dir = _prepare_case_directory(Path(arguments.output), binding)
    profiles = _binding_profiles(
        binding,
        task_models=task_models,
    )
    environment = ensure_environments(profiles, str(arguments.hf_python))
    reference_sources = ensure_reference_sources(
        str(task_models[binding.model].get("family", "") or ""),
        Path(arguments.reference_cache_dir),
    )
    process_env = _source_environment()
    process_env.update(environment.overrides)
    process_env.update(reference_sources.environment)
    dataset_command = shlex.join([sys.executable, *sys.argv])

    suite = suites[workload]
    dataset = (
        Path(arguments.dataset)
        if arguments.dataset
        else _dataset_path(suite, arguments.dataset_root)
    )
    command = _comparison_command(
        binding,
        case_dir=case_dir,
        dataset=dataset,
        arguments=arguments,
        reference_python=environment.base_python,
        reference_sources=reference_sources,
    )
    returncode = _run_subprocess(command, case_dir / "execution.log", process_env)
    result = _comparison_result(
        binding,
        case_dir=case_dir,
        returncode=returncode,
        reference_environment=environment,
        dataset_command=dataset_command,
        sample_limit=int(arguments.limit or 0),
    )

    comparison = case_dir / "comparison.json"
    _atomic_write_validation_result(comparison, result)
    return result


def _source_revision() -> str:
    configured = os.environ.get("TRTMC_VALIDATION_SOURCE_REVISION", "").strip()
    if configured:
        return configured
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def write_run_metadata(output: Path) -> Path:
    metadata = {
        "schema_version": "trtmc.validation-run/v1",
        "source_revision": _source_revision(),
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "command": shlex.join(sys.argv),
        "started_at": _utc_now().isoformat(),
        "finished_at": None,
        "duration_seconds": None,
        "status": "running",
    }
    path = output / "run.json"
    _atomic_write_json(path, metadata)
    return path


def finalize_run_metadata(
    output: Path,
    *,
    error: str = "",
) -> Path:
    path = output / "run.json"
    loaded = _read_json_artifact(path)
    if not isinstance(loaded, Mapping):
        raise ValidationError(f"validation run metadata must be an object: {path}")
    metadata = dict(loaded)
    finished_at = _utc_now()
    metadata["finished_at"] = finished_at.isoformat()
    metadata["duration_seconds"] = _elapsed_seconds(
        metadata.get("started_at"),
        finished_at,
    )
    metadata["status"] = "failed" if error else "completed"
    if error:
        metadata["error"] = error
    else:
        metadata.pop("error", None)
    _atomic_write_json(path, metadata)
    return path


def _report_provenance(run: Mapping[str, Any]) -> str:
    fields = (
        ("source", run.get("source_revision")),
        ("host", run.get("hostname")),
        ("CUDA_VISIBLE_DEVICES", run.get("cuda_visible_devices")),
    )
    return " · ".join(f"{name}={value}" for name, value in fields if value)


def _validate_run_metadata(
    run: Mapping[str, Any],
    *,
    path: Path,
) -> str:
    has_schema = "schema_version" in run
    has_status = "status" in run
    if has_schema != has_status:
        raise ValidationError(
            "validation run metadata must provide schema_version and status "
            f"together: {path}"
        )
    if has_schema and run.get(
        "schema_version"
    ) != "trtmc.validation-run/v1":
        raise ValidationError(
            f"validation run metadata has an unsupported schema_version: {path}"
        )
    status = run.get("status") if has_status else None
    if has_status and (
        not isinstance(status, str)
        or status not in {"running", "completed", "failed"}
    ):
        raise ValidationError(
            f"validation run metadata has an invalid status: {path}"
        )
    started_at = run.get("started_at")
    if not isinstance(started_at, str) or not started_at:
        raise ValidationError(
            f"validation run metadata must include started_at: {path}"
        )
    started_timestamp: datetime | None = None
    if has_schema:
        try:
            started_timestamp = datetime.fromisoformat(
                started_at[:-1] + "+00:00"
                if started_at.endswith("Z")
                else started_at
            )
        except ValueError as exc:
            raise ValidationError(
                f"validation run metadata started_at is not ISO-8601: "
                f"{path}"
            ) from exc
        if started_timestamp.tzinfo is None:
            raise ValidationError(
                "validation run metadata started_at must include a "
                f"timezone: {path}"
            )
    finished_at = run.get("finished_at")
    duration = run.get("duration_seconds")
    error = run.get("error", "")
    if not has_status:
        if finished_at is None and duration is None and not error:
            status = "running"
        elif error:
            status = "failed"
        else:
            status = "completed"
    assert isinstance(status, str)
    if status == "running":
        if finished_at is not None or duration is not None or error:
            raise ValidationError(
                "running validation metadata cannot be finalized or carry "
                f"an error: {path}"
            )
        return status
    if not isinstance(finished_at, str) or not finished_at:
        raise ValidationError(
            f"finalized validation metadata must include finished_at: {path}"
        )
    if has_schema:
        try:
            finished_timestamp = datetime.fromisoformat(
                finished_at[:-1] + "+00:00"
                if finished_at.endswith("Z")
                else finished_at
            )
        except ValueError as exc:
            raise ValidationError(
                f"validation run metadata finished_at is not ISO-8601: "
                f"{path}"
            ) from exc
        if finished_timestamp.tzinfo is None:
            raise ValidationError(
                "validation run metadata finished_at must include a "
                f"timezone: {path}"
            )
        assert started_timestamp is not None
        if finished_timestamp < started_timestamp:
            raise ValidationError(
                "validation run metadata finished_at precedes started_at: "
                f"{path}"
            )
    if (
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(duration)
        or duration < 0
    ):
        raise ValidationError(
            "finalized validation metadata must include a finite, "
            f"non-negative duration_seconds: {path}"
        )
    if status == "failed":
        if not isinstance(error, str) or not error:
            raise ValidationError(
                f"failed validation metadata must include an error: {path}"
            )
    elif error:
        raise ValidationError(
            f"completed validation metadata cannot carry an error: {path}"
        )
    return status


def _elapsed_seconds(
    started_at: Any,
    finished_at: datetime,
) -> float | None:
    if not isinstance(started_at, str) or not started_at:
        return None
    try:
        started = datetime.fromisoformat(started_at)
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return round(max(0.0, (finished_at - started).total_seconds()), 3)


def _format_duration(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
        return ""
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s"


def _merge_commands_from_result_logs(
    result: dict[str, Any],
    case_dir: Path,
) -> None:
    root = _validated_case_work_dir(result, case_dir)
    if root is None:
        return

    def read_artifact(
        path: Path,
        errors: str,
        maximum_bytes: int,
    ) -> str | None:
        return _read_case_text_artifact(
            path,
            case_dir=case_dir,
            missing_ok=True,
            errors=errors,
            maximum_bytes=maximum_bytes,
        )

    discovered = _commands_from_logs(
        root,
        read_artifact=read_artifact,
        log_paths=_secure_command_log_paths(root),
    )
    reproduce = result.get("reproduce", {})
    if not isinstance(reproduce, dict):
        reproduce = {}
    for kind in ("hf", "trtmc"):
        existing = _string_list(
            reproduce.get(kind, []),
            field=f"reproduce.{kind}",
        )
        existing_counts = reproduce.get("command_count", {})
        existing_count = (
            existing_counts.get(kind, len(existing))
            if isinstance(existing_counts, Mapping)
            else len(existing)
        )
        extra = [
            command
            for command in existing
            if command not in discovered[kind]
        ]
        discovered[kind] = (
            extra + discovered[kind]
        )[:MAX_REPRO_COMMANDS_PER_BACKEND]
        discovered["command_count"][kind] = max(
            existing_count,
            discovered["command_count"][kind] + len(extra),
        )
    discovered["dataset"] = reproduce.get("dataset", {})
    representative = discovered.get("representative", {})
    if (
        not isinstance(representative, Mapping)
        or not representative.get("sample_id")
    ):
        discovered["representative"] = reproduce.get(
            "representative",
            {},
        )
    result["reproduce"] = _normalize_reproduction(discovered)


def _refresh_disagreement_artifact(
    result: dict[str, Any],
    case_dir: Path,
    *,
    staging_root: Path | None = None,
    source_budget: list[int] | None = None,
    media_budget: dict[str, int] | None = None,
) -> bool:
    work_dir = _validated_case_work_dir(result, case_dir)
    if work_dir is None:
        return False

    def read_artifact(path: Path) -> str | None:
        maximum_bytes = (
            min(MAX_REPORT_ARTIFACT_BYTES, source_budget[0])
            if source_budget is not None
            else MAX_REPORT_ARTIFACT_BYTES
        )
        text = _read_case_text_artifact(
            path,
            case_dir=case_dir,
            missing_ok=True,
            maximum_bytes=maximum_bytes,
        )
        if text is not None and source_budget is not None:
            consumed = len(text.encode("utf-8"))
            source_budget[0] -= consumed
            if source_budget[0] < 0:
                raise ValidationError(
                    "validation report disagreement source artifacts "
                    f"exceed {MAX_REPORT_DISAGREEMENT_SOURCE_BYTES} bytes"
                )
        return text

    result["disagreements"] = _build_disagreement_artifact(
        work_dir=work_dir,
        case_dir=case_dir,
        read_artifact=read_artifact,
        staging_root=staging_root,
        media_budget=media_budget,
    )
    return True


def _result_commands(result: Mapping[str, Any], kind: str) -> list[str]:
    reproduce = result.get("reproduce", {})
    if not isinstance(reproduce, dict):
        return []
    commands = reproduce.get(kind, [])
    if not isinstance(commands, list):
        return []
    return [str(command) for command in commands if str(command).strip()]


def _render_command_group(
    label: str,
    commands: Sequence[str],
    *,
    total: int | None = None,
    logs: Sequence[str] = (),
) -> str:
    if not commands:
        body = '<span class="unavailable">Not reached; see comparison.json.</span>'
    else:
        shell = "\n".join(f"$ {command}" for command in commands)
        body = f"<pre><code>{html.escape(shell)}</code></pre>"
    command_total = len(commands) if total is None else total
    if command_total > len(commands):
        locations = ", ".join(logs) or "comparison.json"
        body += (
            f'<div class="detail">Showing {len(commands)} of {command_total} commands. '
            f"Full command log: {html.escape(locations)}.</div>"
        )
    return f"<h4>{html.escape(label)}</h4>{body}"


def _reproduction_count(result: Mapping[str, Any], kind: str) -> int:
    reproduce = result.get("reproduce", {})
    counts = reproduce.get("command_count", {}) if isinstance(reproduce, dict) else {}
    commands = _result_commands(result, kind)
    try:
        return (
            max(int(counts.get(kind)), len(commands)) if isinstance(counts, dict) else len(commands)
        )
    except (TypeError, ValueError):
        return len(commands)


def _reproduction_logs(result: Mapping[str, Any], kind: str) -> list[str]:
    reproduce = result.get("reproduce", {})
    logs = reproduce.get("command_logs", {}) if isinstance(reproduce, dict) else {}
    return (
        _string_list(
            logs.get(kind, []),
            field=f"reproduce.command_logs.{kind}",
        )
        if isinstance(logs, dict)
        else []
    )


def _dataset_reproduction(result: Mapping[str, Any]) -> tuple[str, int, int]:
    reproduce = result.get("reproduce", {})
    dataset = reproduce.get("dataset", {}) if isinstance(reproduce, dict) else {}
    if not isinstance(dataset, dict):
        return "", 0, 0
    command = str(dataset.get("command", "") or "")
    try:
        sample_limit = int(dataset.get("sample_limit", 0) or 0)
        prepared = int(dataset.get("prepared_input_count", 0) or 0)
    except (TypeError, ValueError):
        sample_limit = 0
        prepared = 0
    return command, sample_limit, prepared


def _representative_note(result: Mapping[str, Any]) -> str:
    reproduce = result.get("reproduce", {})
    representative = reproduce.get("representative", {}) if isinstance(reproduce, dict) else {}
    if not isinstance(representative, dict):
        return ""
    sample_id = str(representative.get("sample_id", "") or "")
    if not sample_id:
        return ""
    reason = str(representative.get("reason", "") or "").replace("_", " ")
    return (
        '<div class="detail">Representative sample: '
        f"{html.escape(sample_id)}"
        f" ({html.escape(reason)}).</div>"
    )


def _json_preview(value: Any, *, max_characters: int = 2000) -> str:
    rendered = json.dumps(value, indent=2, ensure_ascii=False)
    if len(rendered) > max_characters:
        rendered = rendered[:max_characters] + "\n... see disagreements.jsonl"
    return f"<pre><code>{html.escape(rendered)}</code></pre>"


def _render_vanilla_command(label: str, command: str) -> str:
    if command:
        body = f"<pre><code>{html.escape('$ ' + command)}</code></pre>"
    else:
        body = (
            '<span class="unavailable">Native single-sample command unavailable '
            "for this backend.</span>"
        )
    return f"<h5>{html.escape(label)}</h5>{body}"


def _render_failure_media(
    record: Mapping[str, Any],
    *,
    asset_base: Path,
    case_dir: Path,
) -> str:
    artifacts = record.get("artifacts", {})
    media = artifacts.get("media", []) if isinstance(artifacts, dict) else []
    if not isinstance(media, list):
        return ""
    rendered = []
    for item in media:
        if not isinstance(item, dict):
            continue
        label = html.escape(str(item.get("label", "artifact")))
        relative_path = str(item.get("path", "") or "")
        if not relative_path:
            continue
        media_path = Path(relative_path)
        sample_id = str(record.get("sample_id", "") or "")
        expected_directory = (
            trtmc_disagreements._sample_directory_name(sample_id)
        )
        if (
            media_path.is_absolute()
            or relative_path != media_path.as_posix()
            or len(media_path.parts) != 4
            or media_path.parts[:1] != ("repro",)
            or media_path.parts[1] != expected_directory
            or media_path.parts[2] != "media"
            or len(media_path.name) < 4
            or not media_path.name[:2].isdigit()
            or media_path.name[2] != "-"
            or any(
                not (
                    character.isascii()
                    and (
                        character.isalnum()
                        or character in "._-"
                    )
                )
                for character in media_path.name
            )
            or trtmc_disagreements._media_kind(media_path)
            != str(item.get("kind", ""))
        ):
            continue
        try:
            _validate_case_media_artifact(
                case_dir / media_path,
                case_dir=case_dir,
            )
        except ValidationError:
            continue
        href = html.escape(str(asset_base / relative_path))
        body = _failure_media_tag(str(item.get("kind", "")), href, label)
        if not body:
            continue
        rendered.append(f"<figure><figcaption>{label}</figcaption>{body}</figure>")
    if not rendered:
        return ""
    return '<h5>Failure media</h5><div class="failure-media">' + "".join(rendered) + "</div>"


def _failure_media_tag(kind: str, href: str, label: str) -> str:
    if kind == "image":
        return f'<a href="{href}"><img src="{href}" alt="{label}" loading="lazy"></a>'
    if kind == "audio":
        return f'<audio controls preload="metadata" src="{href}"></audio>'
    if kind == "video":
        return f'<video controls preload="metadata" src="{href}"></video>'
    return ""


def _render_disagreement_record(
    record: Mapping[str, Any],
    *,
    asset_base: Path,
    case_dir: Path,
) -> str:
    sample_id = str(record.get("sample_id", "") or "unknown sample")
    reason = str(record.get("reason", "") or "comparison mismatch").replace("_", " ")
    reproduce = record.get("reproduce", {})
    reproduce = reproduce if isinstance(reproduce, dict) else {}
    return (
        '<details class="sample-difference">'
        f"<summary>{html.escape(sample_id)} · {html.escape(reason)}</summary>"
        '<div class="difference-grid">'
        f"<section><h5>Input</h5>{_json_preview(record.get('input', {}))}</section>"
        f"<section><h5>Reference result</h5>{_json_preview(record.get('reference_result', {}))}</section>"
        f"<section><h5>TRTMC result</h5>{_json_preview(record.get('trtmc_result', {}))}</section>"
        f"<section><h5>Comparison</h5>{_json_preview(record.get('comparison', {}))}</section>"
        "</div>"
        f"{_render_failure_media(record, asset_base=asset_base, case_dir=case_dir)}"
        f"{_render_vanilla_command('Reference vanilla command', str(reproduce.get('reference', '') or ''))}"
        f"{_render_vanilla_command('TRTMC vanilla command', str(reproduce.get('trtmc', '') or ''))}"
        "</details>"
    )


def _render_disagreements(
    result: Mapping[str, Any],
    *,
    case_dir: Path,
    artifact_href: str,
    artifact_text: str | None = None,
) -> str:
    metadata = result.get("disagreements", {})
    if not isinstance(metadata, dict):
        return ""
    if "count" not in metadata:
        return ""
    count = metadata["count"]
    limit = metadata.get(
        "inline_limit",
        trtmc_disagreements.INLINE_DISAGREEMENT_LIMIT,
    )
    if (
        type(count) is not int
        or count < 0
        or type(limit) is not int
        or limit < 0
    ):
        raise ValidationError(
            "invalid validation disagreement metadata"
        )
    artifact_name = str(
        metadata.get("path", DISAGREEMENT_ARTIFACT_NAME)
    )
    if artifact_name != DISAGREEMENT_ARTIFACT_NAME:
        raise ValidationError(
            "validation disagreement artifact path must be "
            f"{DISAGREEMENT_ARTIFACT_NAME}"
        )
    limit = min(
        max(limit, 0),
        trtmc_disagreements.INLINE_DISAGREEMENT_LIMIT,
    )
    artifact_path = case_dir / DISAGREEMENT_ARTIFACT_NAME
    try:
        preview = trtmc_disagreements.load_disagreement_preview(
            artifact_path,
            limit=limit,
            expected_count=count,
            read_artifact=lambda path: _read_case_text_artifact(
                path,
                case_dir=case_dir,
                missing_ok=True,
            )
            if artifact_text is None
            else artifact_text,
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValidationError(
            f"invalid validation disagreement artifact {artifact_path}: {exc}"
        ) from exc
    if count == 0:
        return ""
    comparison = result.get("comparison", {})
    failed = isinstance(comparison, dict) and comparison.get("status") == "disagreement"
    noun = "failed samples" if failed else "sample differences"
    asset_base = Path(artifact_href).parent
    records = "".join(
        _render_disagreement_record(
            record,
            asset_base=asset_base,
            case_dir=case_dir,
        )
        for record in preview
    )
    more = ""
    if count > len(preview):
        more = (
            f'<div class="detail">Showing {len(preview)} of {count}. '
            f'<a href="{html.escape(artifact_href)}">View all in disagreements.jsonl</a>.</div>'
        )
    return (
        f'<details class="failure-details"><summary>{count} {noun} · '
        "results and vanilla commands</summary>"
        f"{records}{more}</details>"
    )


def _render_reproduction(
    result: Mapping[str, Any],
    *,
    case_dir: Path,
    artifact_href: str,
    artifact_text: str | None = None,
) -> str:
    not_compared_reason = str(result.get("not_compared_reason", "") or "")
    if not_compared_reason:
        return (
            '<span class="unavailable">'
            f"{html.escape(not_compared_reason)}"
            "</span>"
        )
    reference_commands = _result_commands(result, "hf")
    trtmc_commands = _result_commands(result, "trtmc")
    dataset_command, sample_limit, _ = _dataset_reproduction(result)
    reference_total = _reproduction_count(result, "hf")
    trtmc_total = _reproduction_count(result, "trtmc")
    if sample_limit:
        sample_label = "sample" if sample_limit == 1 else "samples"
        dataset_label = f"Dataset slice ({sample_limit} {sample_label})"
    else:
        dataset_label = "Full dataset"
    summary = (
        f"Dataset · Reference {len(reference_commands)}/{reference_total} · "
        f"TRTMC {len(trtmc_commands)}/{trtmc_total}"
    )
    return (
        f"{_render_disagreements(result, case_dir=case_dir, artifact_href=artifact_href, artifact_text=artifact_text)}"
        f"<details><summary>{summary}</summary>"
        '<div class="commands">'
        f"{_render_command_group(dataset_label, [dataset_command] if dataset_command else [])}"
        f"{_render_command_group('Reference sample', reference_commands, total=reference_total, logs=_reproduction_logs(result, 'hf'))}"
        f"{_render_command_group('TRTMC sample', trtmc_commands, total=trtmc_total, logs=_reproduction_logs(result, 'trtmc'))}"
        f"{_representative_note(result)}"
        "</div></details>"
    )


def _reference_result_status(result: Mapping[str, Any]) -> str:
    raw_result = result.get("raw_result", {})
    if not isinstance(raw_result, dict):
        return ""
    return str(raw_result.get("hf_cache_status") or raw_result.get("hf_reference_status") or "")


def _signal(status: str, labels: Mapping[str, str]) -> str:
    label = labels.get(status, status.replace("_", " ").title())
    safe_status = status if status.replace("_", "").isalnum() else "unknown"
    return (
        f'<span class="signal signal-{safe_status}">'
        '<span class="signal-light"></span>'
        f"{html.escape(label)}</span>"
    )


def _render_execution(result: Mapping[str, Any]) -> str:
    execution = result.get("execution", {})
    status = str(execution.get("status", "error")) if isinstance(execution, dict) else "error"
    rendered = _signal(
        status,
        {"completed": "Completed", "error": "Error", "not_run": "Not run"},
    )
    attempts = int(execution.get("attempt_count", 1)) if isinstance(execution, dict) else 1
    if attempts > 1:
        outcome = "recovered" if status == "completed" else "failed"
        rendered += f'<div class="detail">{outcome} after {attempts} attempts</div>'
    return rendered


def _render_reference(result: Mapping[str, Any]) -> str:
    if result.get("not_compared_reason"):
        return _signal("not_run", {"not_run": "Not configured"})
    status = _reference_result_status(result)
    display_status = {
        "reused": "cached",
        "adopted": "cached",
        "generated": "completed",
        "ran": "completed",
    }.get(status, "not_run")
    label = {
        "reused": "Reused",
        "adopted": "Adopted",
        "generated": "Generated",
        "ran": "Generated",
    }.get(status, "Not recorded")
    environments = ", ".join(
        str(item.get("name", ""))
        for item in result.get("reference_environment", [])
        if isinstance(item, dict)
    )
    detail = f'<div class="detail">{html.escape(environments)}</div>' if environments else ""
    return _signal(display_status, {display_status: label}) + detail


def _render_comparison(result: Mapping[str, Any]) -> str:
    comparison = result.get("comparison", {})
    if not isinstance(comparison, dict):
        return _signal("not_run", {"not_run": "Not compared"})
    status = str(comparison.get("status", "not_run"))
    signal = _signal(
        status,
        {
            "agreement": "Agreement",
            "disagreement": "Disagreement",
            "not_run": "Not compared",
        },
    )
    mode = str(comparison.get("mode", "") or "")
    not_compared_reason = str(result.get("not_compared_reason", "") or "")
    details = [value for value in (mode, not_compared_reason) if value]
    contract = result.get("precision_contract", {})
    if isinstance(contract, Mapping) and contract:
        base = str(contract.get("trtmc_base_precision", "") or "").upper()
        quantization = str(
            contract.get("trtmc_quantization", "") or ""
        ).upper()
        reference = str(
            contract.get("reference_precision", "") or ""
        ).upper()
        candidate = (
            f"{quantization} ({base} base)"
            if quantization and quantization != "NONE"
            else base
        )
        if candidate and reference:
            details.append(f"TRTMC {candidate} vs HF {reference}")
        comparison_kind = str(contract.get("comparison", "") or "")
        if comparison_kind == "quantized_vs_unquantized_reference":
            details.append("Quantized candidate vs unquantized reference")
        elif comparison_kind == "aligned":
            details.append("Aligned precision")
        elif comparison_kind == "reference_defined":
            details.append("Reference-defined precision")
    return signal + "".join(
        f'<div class="detail">{html.escape(detail)}</div>'
        for detail in details
    )


def _format_metric_value(name: str, value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if not isinstance(value, float):
        return str(value)
    is_ratio = any(
        token in name
        for token in ("accuracy", "agreement", "pass_rate", "exact_match", "divergence_rate")
    )
    if is_ratio:
        return f"{value * 100:.2f}%"
    if value and abs(value) < 0.001:
        return f"{value:.3e}"
    return f"{value:.6f}"


def _render_metrics(result: Mapping[str, Any]) -> str:
    if result.get("not_compared_reason"):
        return '<span class="unavailable">Not compared</span>'
    comparison = result.get("comparison", {})
    metrics = comparison.get("metrics", {}) if isinstance(comparison, dict) else {}
    if not isinstance(metrics, dict) or not metrics:
        return '<span class="unavailable">No metrics</span>'
    primary = comparison.get("primary_metric")
    primary_name = primary.get("name") if isinstance(primary, dict) else None
    ordered = ([primary_name] if primary_name in metrics else []) + [
        name for name in metrics if name != primary_name
    ]
    visible = ordered[:5]
    rows = [
        (
            f'<div class="metric{" primary" if name == primary_name else ""}">'
            f"<span>{html.escape(str(name))}</span>"
            f"<strong>{html.escape(_format_metric_value(str(name), metrics[name]))}</strong>"
            "</div>"
        )
        for name in visible
    ]
    remaining = len(ordered) - len(visible)
    if remaining:
        rows.append(f'<div class="detail">+{remaining} more in comparison.json</div>')
    return "".join(rows)


def _render_validation(result: Mapping[str, Any]) -> str:
    validation = result.get("validation", {})
    status = str(validation.get("status", "failed")) if isinstance(validation, dict) else "failed"
    return _signal(
        status,
        {
            "passed": "Pass",
            "failed": "Fail",
            "skipped": "Skipped",
            "not_compared": "Not compared",
        },
    )


def _render_samples(result: Mapping[str, Any]) -> str:
    if result.get("not_compared_reason"):
        return "—"
    _command, sample_limit, _ = _dataset_reproduction(result)
    if sample_limit:
        return str(sample_limit)
    return "Full"


def _discover_report_result_paths(output: Path) -> list[Path]:
    """Find two-level comparison results without traversing directory links."""
    _ensure_real_directory(output, description="validation output")
    result_paths: list[Path] = []
    output_fd = _open_real_directory(output)
    visited = 0
    try:
        with os.scandir(output_fd) as model_entries:
            for model_entry in model_entries:
                visited += 1
                if visited > MAX_REPORT_DISCOVERY_ENTRIES:
                    raise ValidationError(
                        "validation report discovery exceeds "
                        f"{MAX_REPORT_DISCOVERY_ENTRIES} entries: {output}"
                    )
                if model_entry.is_symlink():
                    raise ValidationError(
                        "validation result directory must not be a symlink: "
                        f"{output / model_entry.name}"
                    )
                if not model_entry.is_dir(follow_symlinks=False):
                    continue
                model_fd = os.open(
                    model_entry.name,
                    _secure_directory_flags(),
                    dir_fd=output_fd,
                )
                try:
                    with os.scandir(model_fd) as workload_entries:
                        for workload_entry in workload_entries:
                            visited += 1
                            if visited > MAX_REPORT_DISCOVERY_ENTRIES:
                                raise ValidationError(
                                    "validation report discovery exceeds "
                                    f"{MAX_REPORT_DISCOVERY_ENTRIES} "
                                    f"entries: {output}"
                                )
                            workload_dir = (
                                output
                                / model_entry.name
                                / workload_entry.name
                            )
                            if workload_entry.is_symlink():
                                raise ValidationError(
                                    "validation result directory must not "
                                    f"be a symlink: {workload_dir}"
                                )
                            if not workload_entry.is_dir(
                                follow_symlinks=False
                            ):
                                continue
                            workload_fd = os.open(
                                workload_entry.name,
                                _secure_directory_flags(),
                                dir_fd=model_fd,
                            )
                            try:
                                try:
                                    metadata = os.stat(
                                        "comparison.json",
                                        dir_fd=workload_fd,
                                        follow_symlinks=False,
                                    )
                                except FileNotFoundError:
                                    continue
                                candidate = (
                                    workload_dir / "comparison.json"
                                )
                                if not stat.S_ISREG(metadata.st_mode):
                                    raise ValidationError(
                                        "validation result must be a "
                                        f"regular file: {candidate}"
                                    )
                                result_paths.append(candidate)
                                if len(result_paths) > MAX_REPORT_RESULTS:
                                    raise ValidationError(
                                        "validation report result count "
                                        f"exceeds {MAX_REPORT_RESULTS}: "
                                        f"{output}"
                                    )
                            finally:
                                os.close(workload_fd)
                finally:
                    os.close(model_fd)
    except OSError as exc:
        raise ValidationError(
            f"cannot securely discover validation results in {output}: "
            f"{exc}"
        ) from exc
    finally:
        os.close(output_fd)
    return sorted(result_paths)


def _validate_report_result_path(output: Path, path: Path) -> None:
    """Require an exact output/model/workload/comparison.json real path."""
    try:
        relative = path.relative_to(output)
    except ValueError as exc:
        raise ValidationError(
            f"validation result is outside the output root: {path}"
        ) from exc
    if len(relative.parts) != 3 or relative.name != "comparison.json":
        raise ValidationError(f"invalid validation result path: {path}")
    for directory in (output, path.parent.parent, path.parent):
        try:
            metadata = directory.lstat()
        except OSError as exc:
            raise ValidationError(
                f"cannot inspect validation result directory {directory}: {exc}"
            ) from exc
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValidationError(
                f"validation result directory must not be a symlink: {directory}"
            )
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValidationError(f"cannot inspect validation result {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValidationError(f"validation result must be a regular file: {path}")


def _read_report_result(
    output: Path,
    path: Path,
    *,
    include_size: bool = False,
) -> Any:
    _validate_report_result_path(output, path)
    relative = path.relative_to(output)
    missing = [
        name
        for name in ("O_NONBLOCK", "O_NOFOLLOW")
        if not hasattr(os, name)
    ]
    if missing:
        raise ValidationError(
            "secure validation result reads require " + ", ".join(missing)
        )
    directory_flags = _secure_directory_flags()
    result_flags = (
        os.O_RDONLY
        | os.O_NONBLOCK
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    root_descriptor: int | None = None
    model_descriptor: int | None = None
    workload_descriptor: int | None = None
    descriptor: int | None = None
    try:
        root_descriptor = _open_real_directory(output)
        model_descriptor = os.open(
            relative.parts[0],
            directory_flags,
            dir_fd=root_descriptor,
        )
        if not stat.S_ISDIR(os.fstat(model_descriptor).st_mode):
            raise ValidationError(
                f"validation model result must be a real directory: {path.parent.parent}"
            )
        workload_descriptor = os.open(
            relative.parts[1],
            directory_flags,
            dir_fd=model_descriptor,
        )
        if not stat.S_ISDIR(os.fstat(workload_descriptor).st_mode):
            raise ValidationError(
                f"validation case result must be a real directory: {path.parent}"
            )
        descriptor = os.open(
            relative.name,
            result_flags,
            dir_fd=workload_descriptor,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValidationError(
                f"validation result must be a regular file: {path}"
            )
        if metadata.st_nlink != 1:
            raise ValidationError(
                f"validation result must not be a hard link: {path}"
            )
        if metadata.st_size > MAX_REPORT_ARTIFACT_BYTES:
            raise ValidationError(
                "validation result exceeds "
                f"{MAX_REPORT_ARTIFACT_BYTES} bytes: {path}"
            )
        with os.fdopen(descriptor, "rb") as result_file:
            descriptor = None
            try:
                payload = result_file.read(MAX_REPORT_ARTIFACT_BYTES + 1)
                if len(payload) > MAX_REPORT_ARTIFACT_BYTES:
                    raise ValidationError(
                        "validation result exceeds "
                        f"{MAX_REPORT_ARTIFACT_BYTES} bytes: {path}"
                    )
                loaded = json.loads(
                    payload.decode("utf-8"),
                    parse_constant=_reject_nonstandard_json_constant,
                )
            except (UnicodeError, ValueError, RecursionError) as exc:
                raise ValidationError(
                    f"invalid validation result JSON in {path}: {exc}"
                ) from exc
        _verify_visible_regular_artifact(
            path,
            workload_descriptor,
            metadata,
            operation="reading",
        )
        _verify_visible_artifact_parent(path, workload_descriptor)
        return (loaded, len(payload)) if include_size else loaded
    except (OSError, ValueError) as exc:
        raise ValidationError(f"cannot open validation result {path}: {exc}") from exc
    finally:
        for open_descriptor in (
            descriptor,
            workload_descriptor,
            model_descriptor,
            root_descriptor,
        ):
            if open_descriptor is not None:
                os.close(open_descriptor)


def _validate_report_json_depth(
    value: Any,
    *,
    path: Path,
    maximum_depth: int = 256,
) -> None:
    pending = [(value, 0)]
    while pending:
        current, depth = pending.pop()
        if depth > maximum_depth:
            raise ValidationError(
                f"invalid validation result JSON in {path}: "
                f"nesting exceeds {maximum_depth}"
            )
        if isinstance(current, float) and not math.isfinite(current):
            raise ValidationError(
                f"invalid validation result JSON in {path}: "
                "non-finite number"
            )
        if isinstance(current, str):
            try:
                current.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValidationError(
                    f"invalid validation result JSON in {path}: "
                    "string is not valid UTF-8"
                ) from exc
        if isinstance(current, Mapping):
            for key, item in current.items():
                if isinstance(key, str):
                    try:
                        key.encode("utf-8")
                    except UnicodeEncodeError as exc:
                        raise ValidationError(
                            f"invalid validation result JSON in {path}: "
                            "object key is not valid UTF-8"
                        ) from exc
                pending.append((item, depth + 1))
        elif isinstance(current, (list, tuple)):
            pending.extend((item, depth + 1) for item in current)


def _validate_disagreement_metadata(
    result: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    metadata = result.get("disagreements")
    if metadata is None:
        return
    if not isinstance(metadata, Mapping):
        raise ValidationError(
            f"validation disagreement metadata must be an object: {path}"
        )
    if "count" not in metadata:
        raise ValidationError(
            f"validation disagreement metadata must include count: {path}"
        )
    count = metadata["count"]
    if type(count) is not int or count < 0:
        raise ValidationError(
            "validation disagreement metadata count must be a "
            f"non-negative integer: {path}"
        )
    if count > trtmc_disagreements.MAX_DISAGREEMENT_RECORDS:
        raise ValidationError(
            "validation disagreement metadata count exceeds "
            f"{trtmc_disagreements.MAX_DISAGREEMENT_RECORDS}: {path}"
        )
    if "inline_limit" in metadata:
        inline_limit = metadata["inline_limit"]
        if type(inline_limit) is not int or inline_limit < 0:
            raise ValidationError(
                "validation disagreement metadata inline_limit must be a "
                f"non-negative integer: {path}"
            )
    artifact_name = str(
        metadata.get("path", DISAGREEMENT_ARTIFACT_NAME)
    )
    if artifact_name != DISAGREEMENT_ARTIFACT_NAME:
        raise ValidationError(
            "validation disagreement artifact path must be "
            f"{DISAGREEMENT_ARTIFACT_NAME}: {path}"
        )
    comparison = result.get("comparison")
    if isinstance(comparison, Mapping):
        comparison_status = comparison.get("status")
        if (
            isinstance(comparison_status, str)
            and comparison_status in {"agreement", "not_run"}
            and count
        ):
            raise ValidationError(
                "validation result cannot report disagreement evidence "
                f"when comparison.status is {comparison_status}: {path}"
            )


def _validate_result_identity(
    result: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    expected_model = path.parent.parent.name
    expected_workload = path.parent.name
    model = result.get("model")
    if not isinstance(model, str) or not model or model != expected_model:
        raise ValidationError(
            f"validation result model does not match its path: {path}"
        )
    workload = result.get("workload")
    reason_value = result.get("not_compared_reason", "")
    if not isinstance(reason_value, str):
        raise ValidationError(
            "validation result not_compared_reason must be a string: "
            f"{path}"
        )
    not_compared_reason = reason_value
    legacy_e2e = (
        result.get("executor") == "e2e"
        and expected_workload == "e2e"
        and workload is None
        and not_compared_reason == LEGACY_E2E_REASON
    )
    if (
        not_compared_reason
        and expected_workload != NOT_COMPARED_DIRECTORY
        and not legacy_e2e
    ):
        raise ValidationError(
            "only not-compared or exact legacy e2e result paths may set "
            f"not_compared_reason: {path}"
        )
    if expected_workload == NOT_COMPARED_DIRECTORY:
        if workload is not None or not not_compared_reason:
            raise ValidationError(
                "not-compared validation result must have a null workload "
                "and a non-empty not_compared_reason: "
                f"{path}"
            )
        expected_statuses = ("not_run", "not_run", "not_compared")
        actual_statuses = (
            result.get("execution", {}).get("status")
            if isinstance(result.get("execution"), Mapping)
            else None,
            result.get("comparison", {}).get("status")
            if isinstance(result.get("comparison"), Mapping)
            else None,
            result.get("validation", {}).get("status")
            if isinstance(result.get("validation"), Mapping)
            else None,
        )
        if actual_statuses != expected_statuses:
            raise ValidationError(
                "not-compared validation result must use canonical "
                "not_run/not_run/not_compared statuses: "
                f"{path}"
            )
    elif legacy_e2e:
        return
    elif (
        not isinstance(workload, str)
        or not workload
        or workload != expected_workload
    ):
        raise ValidationError(
            f"validation result workload does not match its path: {path}"
        )


def _normalize_result_files(
    output: Path,
    result_paths: Sequence[Path],
) -> tuple[
    list[dict[str, Any]],
    dict[Path, _CaseArtifactStage],
]:
    if len(result_paths) > MAX_REPORT_RESULTS:
        raise ValidationError(
            "validation report result count exceeds "
            f"{MAX_REPORT_RESULTS}"
        )
    results = []
    stages: dict[Path, _CaseArtifactStage] = {}
    aggregate_bytes = 0
    for path in result_paths:
        _validate_report_result_path(output, path)
        loaded, payload_bytes = _read_report_result(
            output,
            path,
            include_size=True,
        )
        aggregate_bytes += payload_bytes
        if aggregate_bytes > MAX_REPORT_RESULT_BYTES:
            raise ValidationError(
                "validation report comparison inputs exceed "
                f"{MAX_REPORT_RESULT_BYTES} bytes"
            )
        _validate_report_json_depth(loaded, path=path)
        if not isinstance(loaded, Mapping):
            raise ValidationError(
                f"validation result JSON must be an object: {path}"
            )
        _validate_disagreement_metadata(loaded, path=path)
        _validate_result_identity(loaded, path=path)
        result = _normalize_result(loaded)
        _validate_report_json_depth(
            result,
            path=path,
            maximum_depth=MAX_VALIDATION_RESULT_JSON_DEPTH,
        )
        _validate_disagreement_metadata(result, path=path)
        _validate_result_identity(result, path=path)
        results.append(result)
    try:
        disagreement_source_budget = [
            MAX_REPORT_DISAGREEMENT_SOURCE_BYTES
        ]
        disagreement_media_budget = {
            "files": MAX_REPORT_MEDIA_FILES,
            "bytes": MAX_REPORT_MEDIA_BYTES,
        }
        for path, result in zip(result_paths, results, strict=True):
            _merge_commands_from_result_logs(result, path.parent)
            work_dir = _validated_case_work_dir(
                result,
                path.parent,
            )
            metadata = result.get("disagreements")
            existing_count = (
                metadata.get("count")
                if isinstance(metadata, Mapping)
                else 0
            )
            stage: _CaseArtifactStage | None = None
            should_clear = (
                work_dir is None
                and isinstance(metadata, Mapping)
                and existing_count == 0
            )
            if should_clear:
                artifact_path = (
                    path.parent / DISAGREEMENT_ARTIFACT_NAME
                )
                try:
                    trtmc_disagreements.load_disagreement_preview(
                        artifact_path,
                        limit=0,
                        expected_count=0,
                        read_artifact=lambda artifact: (
                            _read_case_text_artifact(
                                artifact,
                                case_dir=path.parent,
                                missing_ok=True,
                            )
                        ),
                    )
                except (
                    UnicodeError,
                    ValueError,
                    RecursionError,
                ) as exc:
                    raise ValidationError(
                        "invalid validation disagreement artifact "
                        f"{artifact_path}: {exc}"
                    ) from exc
            if work_dir is not None or should_clear:
                stage = _create_case_artifact_stage(path.parent)
                stages[path] = stage
            if work_dir is not None:
                _refresh_disagreement_artifact(
                    result,
                    path.parent,
                    staging_root=(
                        stage.path if stage is not None else None
                    ),
                    source_budget=disagreement_source_budget,
                    media_budget=disagreement_media_budget,
                )
            elif stage is not None:
                _atomic_write_text(
                    stage.path / DISAGREEMENT_ARTIFACT_NAME,
                    "",
                )
                result["disagreements"] = {
                    "count": 0,
                    "path": DISAGREEMENT_ARTIFACT_NAME,
                    "inline_limit": (
                        trtmc_disagreements
                        .INLINE_DISAGREEMENT_LIMIT
                    ),
                    "reference_vanilla_available": False,
                    "trtmc_vanilla_available": False,
                }
            _validate_disagreement_metadata(result, path=path)
            _validate_result_identity(result, path=path)
            _json_artifact_payload(
                path,
                result,
                maximum_depth=MAX_VALIDATION_RESULT_JSON_DEPTH,
            )
            if stage is not None:
                _release_case_artifact_stage(stage)
    except BaseException:
        for stage in stages.values():
            try:
                _cleanup_case_artifact_stage(stage)
            except (OSError, ValidationError):
                pass
        raise
    return results, stages


def _deduplicate_results(
    result_paths: Sequence[Path],
    results: Sequence[dict[str, Any]],
) -> tuple[list[Path], list[dict[str, Any]]]:
    selected: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for path, result in zip(result_paths, results, strict=True):
        key = (
            str(result.get("model", "")),
            str(result.get("workload") or ""),
        )
        current = selected.get(key)
        if current is None or path.parent.name == NOT_COMPARED_DIRECTORY:
            selected[key] = (path, result)
    ordered = sorted(selected.values(), key=lambda item: str(item[0]))
    return (
        [path for path, _result in ordered],
        [result for _path, result in ordered],
    )


def _report_counts(
    results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, int], int]:
    validation_counts = {
        name: sum(result["validation"]["status"] == name for result in results)
        for name in ("passed", "failed", "skipped", "not_compared")
    }
    comparison_counts = {
        name: sum(result["comparison"]["status"] == name for result in results)
        for name in ("agreement", "disagreement", "not_run")
    }
    execution_errors = sum(result["execution"]["status"] == "error" for result in results)
    return validation_counts, comparison_counts, execution_errors


def _traffic_light_counts(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts = {"green": 0, "yellow": 0, "red": 0, "white": 0}
    for result in results:
        validation_status = str(result["validation"]["status"])
        comparison_status = str(result["comparison"]["status"])
        if validation_status == "skipped":
            counts["yellow"] += 1
        elif comparison_status == "agreement":
            counts["green"] += 1
        elif comparison_status == "disagreement":
            counts["red"] += 1
        else:
            counts["white"] += 1
    return counts


def _report_rows(
    output: Path,
    results: Sequence[Mapping[str, Any]],
    result_paths: Sequence[Path],
    *,
    artifact_roots: Mapping[Path, Path] | None = None,
    artifact_texts: Mapping[Path, str] | None = None,
) -> str:
    rows = []
    for result, path in zip(results, result_paths, strict=True):
        relative = path.relative_to(output)
        artifact_relative = relative.parent / DISAGREEMENT_ARTIFACT_NAME
        artifact_root = (
            artifact_roots.get(path, path.parent)
            if artifact_roots is not None
            else path.parent
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(result.get('model', '')))}</td>"
            f"<td>{html.escape(str(result.get('workload') or '—'))}</td>"
            f"<td>{_render_samples(result)}</td>"
            f"<td>{_render_execution(result)}</td>"
            f"<td>{_render_reference(result)}</td>"
            f"<td>{_render_comparison(result)}</td>"
            f"<td>{_render_metrics(result)}</td>"
            f"<td>{_render_validation(result)}</td>"
            f"<td>{_render_reproduction(result, case_dir=artifact_root, artifact_href=str(artifact_relative), artifact_text=artifact_texts.get(path) if artifact_texts is not None else None)}</td>"
            f'<td><a href="{html.escape(str(relative))}">comparison.json</a></td>'
            "</tr>"
        )
    return "".join(rows)


def _report_document(
    report: Mapping[str, Any],
    *,
    rows: str,
    comparison_counts: Mapping[str, int],
    execution_errors: int,
    traffic_light_counts: Mapping[str, int],
) -> str:
    run = report.get("run", {})
    provenance = _report_provenance(run)
    run_error = (
        str(run.get("error", "") or "")
        if isinstance(run, Mapping)
        else ""
    )
    run_failure = (
        '<div class="run-failure"><strong>Run failure:</strong> '
        f"{html.escape(run_error)}</div>"
        if run_error
        else ""
    )
    duration = _format_duration(report["summary"].get("duration_seconds"))
    duration_summary = f" · {html.escape(duration)} total duration" if duration else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>TRTMC Reference Consistency Report</title>
<style>
body {{ font: 14px system-ui, sans-serif; margin: 32px; color: #202124; }}
h1 {{ margin-bottom: 4px; }}
.purpose {{ color: #5f6368; margin-bottom: 8px; }}
.traffic-summary {{ font-size: 20px; font-weight: 650; margin: 14px 0 8px; }}
.summary {{ color: #5f6368; margin-bottom: 24px; }}
.run-failure {{ color: #b3261e; background: #fce8e6; border-radius: 4px;
                margin: 12px 0 20px; padding: 10px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #dadce0; padding: 8px 10px; text-align: left; }}
th {{ background: #f8f9fa; }}
details {{ min-width: 210px; }}
summary {{ cursor: pointer; color: #185abc; }}
.commands {{ min-width: min(760px, 70vw); padding: 4px 0; }}
.commands h4 {{ margin: 12px 0 4px; }}
.failure-details {{ min-width: min(900px, 75vw); margin-bottom: 10px; }}
.sample-difference {{ margin: 8px 0; padding: 8px; border: 1px solid #dadce0;
                      border-radius: 4px; }}
.sample-difference h5 {{ margin: 10px 0 4px; }}
.difference-grid {{ display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr));
                    gap: 10px; }}
.difference-grid pre {{ max-height: 260px; overflow: auto; }}
.failure-media {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start; }}
.failure-media figure {{ margin: 0; max-width: 360px; }}
.failure-media figcaption {{ color: #5f6368; margin-bottom: 4px; }}
.failure-media img, .failure-media video {{ display: block; max-width: 360px;
                                           max-height: 280px; }}
.failure-media audio {{ width: min(360px, 70vw); }}
pre {{ margin: 0; padding: 10px; white-space: pre-wrap; overflow-wrap: anywhere;
       background: #f8f9fa; border: 1px solid #dadce0; border-radius: 4px; }}
.unavailable {{ color: #5f6368; }}
.signal {{ display: inline-flex; align-items: center; gap: 7px; font-weight: 650;
           white-space: nowrap; }}
.signal-light {{ width: 10px; height: 10px; border-radius: 50%;
                 background: #80868b; box-shadow: 0 0 0 3px #eef0f1; }}
.signal-completed, .signal-agreement, .signal-passed {{ color: #137333; }}
.signal-completed .signal-light, .signal-agreement .signal-light,
.signal-passed .signal-light {{ background: #1e8e3e; box-shadow: 0 0 0 3px #e6f4ea; }}
.signal-error, .signal-disagreement, .signal-failed {{ color: #b3261e; }}
.signal-error .signal-light, .signal-disagreement .signal-light,
.signal-failed .signal-light {{ background: #d93025; box-shadow: 0 0 0 3px #fce8e6; }}
.signal-skipped {{ color: #8a4f00; }}
.signal-skipped .signal-light {{ background: #f9ab00; box-shadow: 0 0 0 3px #fef7e0; }}
.signal-cached {{ color: #185abc; }}
.signal-cached .signal-light {{ background: #1a73e8; box-shadow: 0 0 0 3px #e8f0fe; }}
.signal-not_run, .signal-not_compared {{ color: #5f6368; }}
.detail {{ color: #5f6368; font-size: 12px; margin-top: 4px; }}
.metric {{ display: flex; justify-content: space-between; gap: 14px;
           font-variant-numeric: tabular-nums; font-size: 12px; }}
.metric span {{ color: #5f6368; }}
.metric.primary {{ font-size: 13px; }}
.metric.primary span, .metric.primary strong {{ color: #202124; }}
</style></head><body>
<h1>TRTMC Reference Consistency Report</h1>
<div class="purpose">Accuracy and output agreement against the model reference.</div>
<div class="traffic-summary" title="Agreement · Skipped · Disagreement · Not compared">
🟢 {traffic_light_counts["green"]} &nbsp; 🟡 {traffic_light_counts["yellow"]} &nbsp;
🔴 {traffic_light_counts["red"]} &nbsp; ⚪ {traffic_light_counts["white"]}
</div>
<div class="summary">{report["summary"]["cases"]} cases ·
{comparison_counts["agreement"]} agreements ·
{comparison_counts["disagreement"]} disagreements ·
{comparison_counts["not_run"]} not compared ·
{execution_errors} execution errors ·
{report["summary"]["selected_samples"]} samples{duration_summary}<br>
{html.escape(provenance)}</div>
{run_failure}
<table><thead><tr><th>Model</th><th>Workload</th><th>Samples</th><th>Execution</th>
<th>Reference</th><th>Comparison</th><th>Agreement metrics</th>
<th>Validation</th><th>Vanilla reproduction</th><th>Result</th></tr></thead>
<tbody>{rows}</tbody></table>
</body></html>
"""


def _preflight_report_disagreements(
    paths: Sequence[Path],
    results: Sequence[Mapping[str, Any]],
    stages: Mapping[Path, _CaseArtifactStage],
) -> tuple[dict[Path, bytes], dict[Path, str]]:
    staged_payloads: dict[Path, bytes] = {}
    artifact_texts: dict[Path, str] = {}
    remaining_bytes = MAX_REPORT_DISAGREEMENT_BYTES
    record_count = 0
    for path, result in zip(paths, results, strict=True):
        metadata = result.get("disagreements")
        if metadata is None:
            continue
        if not isinstance(metadata, Mapping):
            raise ValidationError(
                f"validation disagreement metadata must be an object: "
                f"{path}"
            )
        count = metadata.get("count")
        if type(count) is not int or count < 0:
            raise ValidationError(
                f"invalid validation disagreement count: {path}"
            )
        if count > trtmc_disagreements.MAX_DISAGREEMENT_RECORDS:
            raise ValidationError(
                "validation disagreement count exceeds "
                f"{trtmc_disagreements.MAX_DISAGREEMENT_RECORDS}: {path}"
            )
        record_count += count
        if record_count > MAX_REPORT_DISAGREEMENT_RECORDS:
            raise ValidationError(
                "validation report disagreement count exceeds "
                f"{MAX_REPORT_DISAGREEMENT_RECORDS}"
            )
        stage = stages.get(path)
        if stage is not None:
            payload = _read_stage_artifact(
                stage,
                DISAGREEMENT_ARTIFACT_NAME,
                maximum_bytes=remaining_bytes,
            )
            staged_payloads[path] = payload
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValidationError(
                    f"invalid UTF-8 disagreement artifact: "
                    f"{stage.path / DISAGREEMENT_ARTIFACT_NAME}"
                ) from exc
        else:
            text = _read_case_text_artifact(
                path.parent / DISAGREEMENT_ARTIFACT_NAME,
                case_dir=path.parent,
                missing_ok=True,
                maximum_bytes=remaining_bytes,
            )
            text = "" if text is None else text
            payload = text.encode("utf-8")
        remaining_bytes -= len(payload)
        if remaining_bytes < 0:
            raise ValidationError(
                "validation report disagreement artifacts exceed "
                f"{MAX_REPORT_DISAGREEMENT_BYTES} bytes"
            )
        artifact_texts[path] = text
    return staged_payloads, artifact_texts


def _transaction_parent_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _open_report_transaction_anchors(
    entries: Sequence[tuple[str, Any]],
) -> dict[Path, int]:
    identities: dict[Path, tuple[int, int]] = {}
    for kind, update in entries:
        if kind == "directory":
            parent = _transaction_parent_path(
                update.stage.case_dir
            )
            identity = update.stage.case_identity
        else:
            parent = _transaction_parent_path(update.path.parent)
            identity = update.parent_identity
        previous = identities.setdefault(parent, identity)
        if previous != identity:
            raise ValidationError(
                "validation transaction has conflicting parent identities: "
                f"{parent}"
            )
    anchors: dict[Path, int] = {}
    try:
        for parent, identity in sorted(
            identities.items(),
            key=lambda item: str(item[0]),
        ):
            descriptor = _open_real_directory(parent)
            if _directory_identity(descriptor) != identity:
                os.close(descriptor)
                raise ValidationError(
                    "validation transaction parent was replaced before "
                    f"commit: {parent}"
                )
            anchors[parent] = descriptor
    except BaseException:
        for descriptor in anchors.values():
            os.close(descriptor)
        raise
    for kind, update in entries:
        parent = _transaction_parent_path(
            update.stage.case_dir
            if kind == "directory"
            else update.path.parent
        )
        update.anchor_fd = anchors[parent]
    return anchors


def write_report(
    output: Path,
    *,
    result_paths: Sequence[Path] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    selected_paths = (
        _discover_report_result_paths(output)
        if result_paths is None
        else sorted(dict.fromkeys(result_paths))
    )
    run_path = output / "run.json"
    run_metadata = _read_json_artifact(run_path, missing_ok=True)
    run_status: str | None = None
    if run_metadata is not None:
        if not isinstance(run_metadata, Mapping):
            raise ValidationError(
                f"validation run metadata must be an object: {run_path}"
            )
        _validate_report_json_depth(
            run_metadata,
            path=run_path,
            maximum_depth=MAX_REPORT_JSON_DEPTH - 1,
        )
        run_status = _validate_run_metadata(
            run_metadata,
            path=run_path,
        )
    json_path = output / "report.json"
    html_path = output / "report.html"
    all_results, stages = _normalize_result_files(
        output,
        selected_paths,
    )
    all_paths = selected_paths
    transaction_entries: list[tuple[str, Any]] = []
    transaction_anchors: dict[Path, int] = {}
    try:
        staged_payloads, artifact_texts = (
            _preflight_report_disagreements(
                all_paths,
                all_results,
                stages,
            )
        )
        selected_paths, results = _deduplicate_results(
            all_paths,
            all_results,
        )
        validation_counts, comparison_counts, execution_errors = (
            _report_counts(results)
        )
        traffic_light_counts = _traffic_light_counts(results)
        sample_limits = [
            _dataset_reproduction(result)[1]
            for result in results
        ]
        generated_at = _utc_now()
        report = {
            "schema_version": "trtmc.validation-report/v2",
            "generated_at": generated_at.isoformat(),
            "validation_status": (
                "failed"
                if (
                    run_status == "failed"
                    or validation_counts["failed"]
                    or (not results and run_status != "running")
                )
                else "incomplete"
                if (
                    run_status == "running"
                    or validation_counts["not_compared"]
                )
                else "passed"
            ),
            "summary": {
                "cases": len(results),
                "execution_completed": sum(
                    result["execution"]["status"] == "completed"
                    for result in results
                ),
                "execution_errors": execution_errors,
                "agreements": comparison_counts["agreement"],
                "disagreements": comparison_counts["disagreement"],
                "not_compared": comparison_counts["not_run"],
                "validation_passed": validation_counts["passed"],
                "validation_failed": validation_counts["failed"],
                "validation_skipped": validation_counts["skipped"],
                "selected_samples": sum(sample_limits),
            },
            "results": results,
        }
        if run_metadata is not None:
            report["run"] = dict(run_metadata)
            duration_seconds = report["run"].get("duration_seconds")
            if duration_seconds is None:
                duration_seconds = _elapsed_seconds(
                    report["run"].get("started_at"),
                    generated_at,
                )
            if duration_seconds is not None:
                report["summary"]["duration_seconds"] = (
                    duration_seconds
                )
        document = _report_document(
            report,
            rows=_report_rows(
                output,
                results,
                selected_paths,
                artifact_roots={
                    path: stage.path
                    for path, stage in stages.items()
                },
                artifact_texts=artifact_texts,
            ),
            comparison_counts=comparison_counts,
            execution_errors=execution_errors,
            traffic_light_counts=traffic_light_counts,
        )
        report_payload = _json_artifact_payload(
            json_path,
            report,
        )
        document_payload = document.encode("utf-8")
        if len(document_payload) > MAX_REPORT_ARTIFACT_BYTES:
            raise ValidationError(
                "validation HTML report exceeds "
                f"{MAX_REPORT_ARTIFACT_BYTES} bytes: {html_path}"
            )
        for path in all_paths:
            stage = stages.get(path)
            if stage is not None:
                directory_update = (
                    _prepare_case_directory_update(stage)
                )
                transaction_entries.append(
                    ("directory", directory_update)
                )
                artifact_update = _prepare_file_update(
                    path.parent / DISAGREEMENT_ARTIFACT_NAME,
                    staged_payloads[path],
                )
                transaction_entries.append(
                    ("file", artifact_update)
                )
        for path, result in zip(
            all_paths,
            all_results,
            strict=True,
        ):
            transaction_entries.append(
                (
                    "file",
                    _prepare_file_update(
                        path,
                        _json_artifact_payload(
                            path,
                            result,
                            maximum_depth=(
                                MAX_VALIDATION_RESULT_JSON_DEPTH
                            ),
                        ),
                    ),
                )
            )
        transaction_entries.extend(
            [
                (
                    "file",
                    _prepare_file_update(json_path, report_payload),
                ),
                (
                    "file",
                    _prepare_file_update(
                        html_path,
                        document_payload,
                    ),
                ),
            ]
        )
        transaction_anchors = _open_report_transaction_anchors(
            transaction_entries
        )
        try:
            for kind, update in transaction_entries:
                if kind == "directory":
                    _commit_case_directory_update(update)
                else:
                    _commit_file_update(update)
        except BaseException:
            for kind, update in reversed(transaction_entries):
                try:
                    if kind == "directory":
                        _rollback_case_directory_update(update)
                    else:
                        _rollback_file_update(update)
                except (OSError, ValidationError):
                    pass
            raise
        return json_path, html_path, report
    finally:
        for kind, update in transaction_entries:
            if kind == "directory":
                _finalize_case_directory_update(update)
            else:
                _finalize_file_update(update)
        for stage in stages.values():
            try:
                anchored_case_fd = transaction_anchors.get(
                    _transaction_parent_path(stage.case_dir)
                )
                _cleanup_case_artifact_stage(
                    stage,
                    anchored_case_fd=anchored_case_fd,
                )
            except (OSError, ValidationError):
                pass
        for descriptor in transaction_anchors.values():
            os.close(descriptor)


def _print_result(result: Mapping[str, Any], comparison: Path, report: Path) -> None:
    not_compared_reason = str(result.get("not_compared_reason", "") or "")
    if not_compared_reason:
        print()
        print(f"Compare result: {comparison}")
        print(f"Report:         {report}")
        return
    reproduce = result.get("reproduce", {})
    hf_commands = reproduce.get("hf", []) if isinstance(reproduce, dict) else []
    trtmc_commands = reproduce.get("trtmc", []) if isinstance(reproduce, dict) else []
    dataset_command, _, _ = _dataset_reproduction(result)
    print()
    print("Reproduce dataset run:")
    print(f"  {dataset_command}" if dataset_command else "  unavailable; see comparison result")
    print()
    print("Reproduce representative HF:")
    if hf_commands:
        for command in hf_commands:
            print(f"  {command}")
    else:
        print("  unavailable; see comparison result")
    print()
    print("Reproduce representative TRTMC:")
    if trtmc_commands:
        for command in trtmc_commands:
            print(f"  {command}")
    else:
        print("  unavailable; see comparison result")
    print()
    print(f"Compare result: {comparison}")
    print(f"Report:         {report}")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate TRTMC against model reference implementations."
    )
    parser.add_argument("model", nargs="?")
    parser.add_argument("workload", nargs="?")
    parser.add_argument(
        "--all",
        action="store_true",
        help="run every validation-eligible ready single-device non-l0-only model",
    )
    parser.add_argument(
        "--on-model-failure",
        choices=("continue", "stop"),
        default="continue",
        help="continue after a failed model or stop after recording it",
    )
    parser.add_argument(
        "--model-attempts",
        type=_positive_int,
        default=2,
        help="maximum worker attempts for execution errors; comparisons are not retried",
    )
    parser.add_argument(
        "--model-retry-delay-seconds",
        type=_nonnegative_float,
        default=5.0,
        help="delay before retrying a model worker after an execution error",
    )
    parser.add_argument(
        "--model-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--list", action="store_true", help="list model-first workloads")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--suites", type=Path, default=DEFAULT_SUITES)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--engine-dir", type=Path, default=DEFAULT_ENGINE_DIR)
    parser.add_argument(
        "--reference-cache-dir",
        type=Path,
        default=DEFAULT_REFERENCE_CACHE,
    )
    parser.add_argument("--trtmc-binary", type=Path, default=REPO_ROOT / "build" / "trtmc")
    parser.add_argument(
        "--benchmark-binary",
        type=Path,
        default=REPO_ROOT / "build" / "trtmc_dataset_benchmark",
    )
    parser.add_argument("--hf-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--backend-dir", type=Path)
    parser.add_argument("--model-plugin-dir", type=Path)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=("override the workload sample limit; use 0 for the complete dataset"),
    )
    parser.add_argument("--force-hf", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_validation_inputs(
    arguments: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    tuple[str, ...],
    dict[str, dict[str, Any]],
]:
    catalog = load_catalog(arguments.catalog)
    suites_list = task_eval.load_suites(arguments.suites)
    suites = {suite["id"]: suite for suite in suites_list}
    ready = ready_model_names(arguments.models_dir)
    task_models = _task_eval_models(arguments.models_dir)
    audit_catalog(catalog, ready_models=ready, suite_names=suites)
    audit_workload_compatibility(
        catalog,
        suites=suites,
        task_models=task_models,
    )
    return catalog, suites, ready, task_models


def _select_bindings(
    arguments: argparse.Namespace,
    catalog: Mapping[str, Any],
    ready_models: Iterable[str],
) -> list[Binding]:
    if arguments.all:
        if arguments.model or arguments.workload or arguments.dataset:
            raise ValidationError("--all cannot be combined with MODEL, WORKLOAD, or --dataset")
        return [resolve_binding(catalog, model) for model in ready_models]
    if not arguments.model:
        raise ValidationError("provide MODEL [WORKLOAD], --all, or --list")
    return [resolve_binding(catalog, arguments.model, arguments.workload)]


def _print_bindings(
    bindings: Iterable[Binding],
    *,
    catalog: Mapping[str, Any],
    explicit_limit: int | None,
) -> None:
    print(
        json.dumps(
            [
                (
                    {
                        "model": binding.model,
                        "workload": binding.workload,
                        "sample_limit": resolve_sample_limit(
                            catalog,
                            binding,
                            explicit_limit,
                        ),
                    }
                    if binding.runnable
                    else {
                        "model": binding.model,
                        "workload": None,
                        "sample_limit": 0,
                        "status": "not_compared",
                        "reason": binding.not_compared_reason,
                    }
                )
                for binding in bindings
            ],
            indent=2,
        )
    )


def _prepare_run_directories(arguments: argparse.Namespace) -> None:
    arguments.output.mkdir(parents=True, exist_ok=True)
    arguments.engine_dir.mkdir(parents=True, exist_ok=True)
    arguments.reference_cache_dir.mkdir(parents=True, exist_ok=True)


def _worker_command(
    binding: Binding,
    arguments: argparse.Namespace,
) -> list[str]:
    workload = _required_workload(binding)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        binding.model,
        workload,
        "--model-worker",
    ]
    for option, value in (
        ("--catalog", arguments.catalog),
        ("--suites", arguments.suites),
        ("--models-dir", arguments.models_dir),
        ("--output", arguments.output),
        ("--engine-dir", arguments.engine_dir),
        ("--reference-cache-dir", arguments.reference_cache_dir),
        ("--trtmc-binary", arguments.trtmc_binary),
        ("--benchmark-binary", arguments.benchmark_binary),
        ("--hf-python", arguments.hf_python),
        ("--model-attempts", arguments.model_attempts),
        ("--model-retry-delay-seconds", arguments.model_retry_delay_seconds),
    ):
        command.extend([option, str(value)])
    for option, value in (
        ("--dataset-root", arguments.dataset_root),
        ("--backend-dir", arguments.backend_dir),
        ("--model-plugin-dir", arguments.model_plugin_dir),
        ("--cuda-visible-devices", arguments.cuda_visible_devices),
    ):
        if value:
            command.extend([option, str(value)])
    if arguments.limit is not None:
        command.extend(["--limit", str(arguments.limit)])
    for option, enabled in (
        ("--force-hf", arguments.force_hf),
        ("--force-build", arguments.force_build),
        ("--no-build", arguments.no_build),
        ("--local-files-only", arguments.local_files_only),
    ):
        if enabled:
            command.append(option)
    return command


def _worker_error_result(
    binding: Binding,
    *,
    command: Sequence[str],
    returncode: int,
    worker_log: Path,
    sample_limit: int,
    error: str,
) -> dict[str, Any]:
    return _normalize_result(
        {
            "schema_version": "trtmc.validation-result/v2",
            "model": binding.model,
            "workload": binding.workload,
            "executor": "model_worker",
            "status": "failed",
            "returncode": returncode,
            "reference_environment": [],
            "reproduce": {
                "dataset": {
                    "command": shlex.join(command),
                    "sample_limit": sample_limit,
                    "prepared_input_count": 0,
                },
                "hf": [],
                "trtmc": [],
            },
            "raw_result": {
                "status": "failed",
                "error_type": "WorkerProcessError",
                "error": error,
            },
            "raw_result_path": "",
            "disagreements": {
                "count": 0,
                "path": "disagreements.jsonl",
                "inline_limit": trtmc_disagreements.INLINE_DISAGREEMENT_LIMIT,
                "reference_vanilla_available": False,
                "trtmc_vanilla_available": False,
            },
            "execution_log": str(worker_log),
            "worker_log": str(worker_log),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


def _run_supervised_binding(
    binding: Binding,
    *,
    arguments: argparse.Namespace,
    catalog: Mapping[str, Any],
    attempt: int = 1,
) -> dict[str, Any]:
    case_dir = _prepare_case_directory(arguments.output, binding)
    comparison_path = case_dir / "comparison.json"
    previous_comparison_identity = _regular_artifact_identity(
        comparison_path,
        missing_ok=True,
    )
    worker_log = case_dir / (
        "worker.log" if attempt == 1 else f"worker.attempt-{attempt}.log"
    )
    command = _worker_command(binding, arguments)
    launch_error = ""
    try:
        returncode = _run_subprocess(command, worker_log, _source_environment())
    except OSError as exc:
        returncode = 127
        launch_error = f"could not start model worker: {exc}"
    try:
        if launch_error:
            raise ValidationError(launch_error)
        try:
            loaded = _read_report_result(arguments.output, comparison_path)
        except ValidationError as exc:
            raise ValidationError(
                f"worker exited with code {returncode} without a valid "
                f"comparison.json: {exc}"
            ) from exc
        current_comparison_identity = _regular_artifact_identity(
            comparison_path,
        )
        if (
            previous_comparison_identity is not None
            and current_comparison_identity
            == previous_comparison_identity
        ):
            raise ValidationError(
                "worker did not replace its stale comparison.json"
            )
        _validate_report_json_depth(loaded, path=comparison_path)
        if not isinstance(loaded, Mapping):
            raise ValidationError("worker comparison.json must contain an object")
        result = _normalize_result(loaded)
        if result.get("model") != binding.model or result.get("workload") != binding.workload:
            raise ValidationError("worker wrote comparison.json for a different binding")
    except (OSError, ValueError, ValidationError) as exc:
        result = _worker_error_result(
            binding,
            command=command,
            returncode=returncode,
            worker_log=worker_log,
            sample_limit=resolve_sample_limit(catalog, binding, arguments.limit),
            error=str(exc),
        )
        _atomic_write_text(case_dir / "disagreements.jsonl", "")
    else:
        result["worker_log"] = str(worker_log)
        dataset = result.get("reproduce", {}).get("dataset", {})
        if isinstance(dataset, dict):
            dataset["command"] = shlex.join(token for token in command if token != "--model-worker")
    _atomic_write_validation_result(comparison_path, result)
    return result


def _archive_failed_attempt(case_dir: Path, attempt: int) -> dict[str, str]:
    archived = {}
    for name in ("comparison.json", "execution.log", "disagreements.jsonl"):
        source = case_dir / name
        path = case_dir / f"{source.stem}.attempt-{attempt}{source.suffix}"
        copied = _atomic_copy_regular_file(
            source,
            path,
            missing_ok=True,
            require_single_link=True,
        )
        if copied is None:
            continue
        archived[name] = str(path)
    return archived


def _attempt_record(
    result: Mapping[str, Any],
    *,
    attempt: int,
    archived: Mapping[str, str],
) -> dict[str, Any]:
    execution = result.get("execution", {})
    validation = result.get("validation", {})
    raw_result = result.get("raw_result", {})
    execution_log = archived.get(
        "execution.log",
        str(result.get("execution_log", "") or ""),
    )
    return {
        "attempt": attempt,
        "execution_status": (
            str(execution.get("status", "")) if isinstance(execution, Mapping) else ""
        ),
        "validation_status": (
            str(validation.get("status", "")) if isinstance(validation, Mapping) else ""
        ),
        "worker_log": str(result.get("worker_log", "") or ""),
        "execution_log": execution_log,
        "comparison_result": archived.get("comparison.json", ""),
        "error_type": (
            str(raw_result.get("error_type", ""))
            if isinstance(raw_result, Mapping)
            else ""
        ),
        "error": (
            str(raw_result.get("error", ""))
            if isinstance(raw_result, Mapping)
            else ""
        ),
    }


def _run_supervised_binding_with_retries(
    binding: Binding,
    *,
    arguments: argparse.Namespace,
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    case_dir = _case_directory(arguments.output, binding)
    attempts = []
    result: dict[str, Any] = {}
    for attempt in range(1, arguments.model_attempts + 1):
        result = _run_supervised_binding(
            binding,
            arguments=arguments,
            catalog=catalog,
            attempt=attempt,
        )
        execution = result.get("execution", {})
        execution_error = (
            isinstance(execution, Mapping)
            and execution.get("status") == "error"
        )
        archived = (
            _archive_failed_attempt(case_dir, attempt)
            if execution_error and attempt < arguments.model_attempts
            else {}
        )
        attempts.append(
            _attempt_record(
                result,
                attempt=attempt,
                archived=archived,
            )
        )
        if not execution_error or attempt == arguments.model_attempts:
            break
        print(
            f"Retrying worker after execution error: {binding.model} "
            f"(attempt {attempt + 1}/{arguments.model_attempts})",
            flush=True,
        )
        if arguments.model_retry_delay_seconds:
            time.sleep(arguments.model_retry_delay_seconds)
    execution = dict(result.get("execution", {}))
    execution.update(
        {
            "attempt_count": len(attempts),
            "max_attempts": arguments.model_attempts,
            "retry_count": max(0, len(attempts) - 1),
            "attempts": attempts,
        }
    )
    result["execution"] = execution
    comparison_path = case_dir / "comparison.json"
    _atomic_write_validation_result(comparison_path, result)
    return result


def _run_all_bindings(
    bindings: Iterable[Binding],
    *,
    arguments: argparse.Namespace,
    catalog: Mapping[str, Any],
) -> int:
    _prepare_run_directories(arguments)
    write_run_metadata(arguments.output)
    failed = False
    not_compared = False
    current_result_paths: list[Path] = []
    try:
        write_report(arguments.output, result_paths=[])
        for binding in bindings:
            if not binding.runnable:
                print(
                    f"\nNot compared: {binding.model} / "
                    f"{binding.not_compared_reason}",
                    flush=True,
                )
                result, comparison = _write_not_compared_case(
                    binding,
                    arguments.output,
                )
                current_result_paths.append(comparison)
                not_compared = True
                _, report_path, _ = write_report(
                    arguments.output,
                    result_paths=current_result_paths,
                )
                _print_result(result, comparison, report_path)
                continue
            sample_limit = resolve_sample_limit(
                catalog,
                binding,
                arguments.limit,
            )
            sample_note = (
                "full dataset"
                if sample_limit == 0
                else f"{sample_limit} samples"
            )
            print(
                f"\nStarting worker: {binding.model} / "
                f"{binding.workload} / {sample_note}",
                flush=True,
            )
            result = _run_supervised_binding_with_retries(
                binding,
                arguments=arguments,
                catalog=catalog,
            )
            comparison = (
                _case_directory(arguments.output, binding)
                / "comparison.json"
            )
            current_result_paths.append(comparison)
            _, report_path, _ = write_report(
                arguments.output,
                result_paths=current_result_paths,
            )
            _print_result(result, comparison, report_path)
            model_failed = result["validation"]["status"] == "failed"
            failed = failed or model_failed
            if model_failed and arguments.on_model_failure == "stop":
                print(
                    f"Stopping after failed model: {binding.model}",
                    flush=True,
                )
                break
    except BaseException as exc:
        try:
            finalize_run_metadata(
                arguments.output,
                error=f"{type(exc).__name__}: {exc}",
            )
            write_report(
                arguments.output,
                result_paths=current_result_paths,
            )
        except BaseException as reporting_exc:
            note = (
                "Additionally failed to finalize the validation report: "
                f"{reporting_exc}"
            )
            if hasattr(exc, "add_note"):
                exc.add_note(note)
            else:
                print(note, file=sys.stderr)
        raise
    finalize_run_metadata(arguments.output)
    write_report(
        arguments.output,
        result_paths=current_result_paths,
    )
    if failed:
        return 1
    return 2 if not_compared and not arguments.all else 0


def _run_bindings(
    bindings: Iterable[Binding],
    *,
    arguments: argparse.Namespace,
    catalog: Mapping[str, Any],
    task_models: Mapping[str, dict[str, Any]],
    suites: Mapping[str, dict[str, Any]],
) -> int:
    _prepare_run_directories(arguments)
    if not arguments.model_worker:
        write_run_metadata(arguments.output)
    failed = False
    not_compared = False
    for binding in bindings:
        if not binding.runnable:
            print(
                f"\nNot compared: {binding.model} / "
                f"{binding.not_compared_reason}",
                flush=True,
            )
            result, comparison = _write_not_compared_case(
                binding,
                arguments.output,
            )
            not_compared = True
            if not arguments.model_worker:
                _, report_path, _ = write_report(arguments.output)
                _print_result(result, comparison, report_path)
            continue
        binding_arguments = copy.copy(arguments)
        binding_arguments.limit = resolve_sample_limit(
            catalog,
            binding,
            arguments.limit,
        )
        sample_note = (
            "full dataset"
            if binding_arguments.limit == 0
            else f"{binding_arguments.limit} samples"
        )
        print(
            f"\n{binding.model} / {binding.workload} / {sample_note}",
            flush=True,
        )
        result = run_binding(
            binding,
            arguments=binding_arguments,
            task_models=task_models,
            suites=suites,
        )
        if not arguments.model_worker:
            _, report_path, _ = write_report(arguments.output)
            comparison = _case_directory(arguments.output, binding) / "comparison.json"
            _print_result(result, comparison, report_path)
        failed = failed or result["validation"]["status"] == "failed"
    if failed:
        return 1
    return 2 if not_compared and not arguments.all else 0


def _main(arguments: argparse.Namespace) -> int:
    catalog, suites, ready, task_models = _load_validation_inputs(arguments)
    if arguments.list:
        for name, spec in catalog["models"].items():
            not_compared_reason = str(spec.get("not_compared_reason", "") or "")
            if not_compared_reason:
                print(f"{name}: not compared ({not_compared_reason})")
                continue
            workloads = []
            for workload in spec["workloads"]:
                limit = catalog["sample_limits"][workload]
                workloads.append(f"{workload} ({limit} samples)")
            print(f"{name}: {', '.join(workloads)}")
        return 0
    bindings = _select_bindings(arguments, catalog, ready)
    if arguments.dry_run:
        _print_bindings(
            bindings,
            catalog=catalog,
            explicit_limit=arguments.limit,
        )
        return 0
    if not arguments.model_worker:
        return _run_all_bindings(
            bindings,
            arguments=arguments,
            catalog=catalog,
        )
    return _run_bindings(
        bindings,
        arguments=arguments,
        catalog=catalog,
        task_models=task_models,
        suites=suites,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        return _main(parser.parse_args(argv))
    except (OSError, ValueError, ValidationError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
