#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run model-first TRTMC reference validation for Dev and QA."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import fcntl
import functools
import hashlib
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
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

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
LEGACY_E2E_REASON = "E2E execution does not compare aligned reference and TRTMC outputs."


class ValidationError(RuntimeError):
    """The requested validation cannot be resolved or executed."""


_RENAME_NOREPLACE = 1
_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC_RENAMEAT2 = getattr(_LIBC, "renameat2", None)
if _LIBC_RENAMEAT2 is not None:
    _LIBC_RENAMEAT2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    _LIBC_RENAMEAT2.restype = ctypes.c_int


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
        raise ValidationError(f"model {binding.model} has no reference-consistency workload")
    return binding.workload


def _case_directory(output: Path, binding: Binding) -> Path:
    return (
        output
        / binding.model
        / (binding.workload if binding.workload is not None else NOT_COMPARED_DIRECTORY)
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
        not component or component in {".", ".."} or Path(component).name != component
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
    missing = [name for name in ("O_DIRECTORY", "O_NOFOLLOW") if not hasattr(os, name)]
    if missing:
        raise ValidationError("secure validation artifacts require " + ", ".join(missing))
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(
            os,
            "O_CLOEXEC",
            0,
        )
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
            raise ValidationError(f"validation artifact directory must be real: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


@contextmanager
def _validation_output_publication_lock(
    output: Path,
) -> Iterator[None]:
    """Serialize report generation for one validation output."""
    _ensure_real_directory(output, description="validation output")
    descriptor = _open_real_directory(output)
    locked = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            locked = True
        except OSError as exc:
            raise ValidationError(
                f"cannot lock validation output for publication {output}: {exc}"
            ) from exc
        visible_descriptor = _open_real_directory(output)
        try:
            locked_metadata = os.fstat(descriptor)
            visible_metadata = os.fstat(visible_descriptor)
            if (
                locked_metadata.st_dev,
                locked_metadata.st_ino,
            ) != (
                visible_metadata.st_dev,
                visible_metadata.st_ino,
            ):
                raise ValidationError(
                    f"validation output changed while acquiring its publication lock: {output}"
                )
        finally:
            os.close(visible_descriptor)
        yield
    finally:
        active_error = sys.exc_info()[1]
        cleanup_errors: list[str] = []
        try:
            if locked:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError as exc:
                    cleanup_errors.append(f"unlock failed: {exc}")
        finally:
            try:
                os.close(descriptor)
            except OSError as exc:
                cleanup_errors.append(f"descriptor close failed: {exc}")
        if cleanup_errors:
            cleanup_message = (
                "validation output publication lock cleanup incomplete: "
                + " | ".join(cleanup_errors)
            )
            if active_error is not None:
                if hasattr(active_error, "add_note"):
                    active_error.add_note(cleanup_message)
                else:
                    print(cleanup_message, file=sys.stderr)
            else:
                raise ValidationError(cleanup_message)


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
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    for attempt in range(100):
        temporary_name = f".{path.name}.{os.getpid()}.{time.time_ns()}.{attempt}.tmp"
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


def _rename_noreplace_at(
    source_fd: int,
    source_name: str,
    destination_fd: int,
    destination_name: str,
) -> None:
    """Atomically rename without replacing a concurrently-created target."""
    if _LIBC_RENAMEAT2 is None:
        raise ValidationError("validation transactions require renameat2(RENAME_NOREPLACE)")
    ctypes.set_errno(0)
    result = _LIBC_RENAMEAT2(
        source_fd,
        os.fsencode(source_name),
        destination_fd,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.ENOSYS, errno.EOPNOTSUPP}:
        raise ValidationError(
            "validation transaction filesystem does not support renameat2(RENAME_NOREPLACE)"
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        destination_name,
    )


def _rename_to_unique_noreplace(
    source_fd: int,
    source_name: str,
    destination_fd: int,
    *,
    prefix: str,
    expected: os.stat_result,
) -> tuple[str, BaseException | None]:
    for _ in range(100):
        destination_name = f"{prefix}.{secrets.token_hex(12)}"
        try:
            deferred_error = _rename_expected_noreplace_at(
                source_fd,
                source_name,
                destination_fd,
                destination_name,
                expected=expected,
            )
        except FileExistsError:
            continue
        except ValidationError as exc:
            if _stat_at(destination_fd, destination_name) is not None:
                return destination_name, exc
            raise
        return destination_name, deferred_error
    raise ValidationError(f"cannot reserve validation transaction recovery name for {source_name}")


def _metadata_identity(
    metadata: os.stat_result,
) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _same_file_version(
    current: os.stat_result,
    expected: os.stat_result,
) -> bool:
    return (
        stat.S_ISREG(current.st_mode)
        and _metadata_identity(current) == _metadata_identity(expected)
        and current.st_nlink == expected.st_nlink
        and current.st_size == expected.st_size
        and current.st_ctime_ns == expected.st_ctime_ns
        and current.st_mtime_ns == expected.st_mtime_ns
    )


def _same_renamed_file_version(
    current: os.stat_result,
    expected: os.stat_result,
) -> bool:
    """Match one file across a rename, which is allowed to change ctime."""
    return (
        stat.S_ISREG(current.st_mode)
        and _metadata_identity(current) == _metadata_identity(expected)
        and current.st_nlink == expected.st_nlink
        and current.st_size == expected.st_size
        and current.st_mtime_ns == expected.st_mtime_ns
    )


def _same_directory_version(
    current: os.stat_result,
    expected: os.stat_result,
) -> bool:
    return (
        stat.S_ISDIR(current.st_mode)
        and _metadata_identity(current) == _metadata_identity(expected)
        and current.st_ctime_ns == expected.st_ctime_ns
        and current.st_mtime_ns == expected.st_mtime_ns
    )


def _stat_at(
    directory_fd: int,
    name: str,
) -> os.stat_result | None:
    try:
        return os.stat(
            name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None


def _rename_expected_noreplace_at(
    source_fd: int,
    source_name: str,
    destination_fd: int,
    destination_name: str,
    *,
    expected: os.stat_result,
) -> BaseException | None:
    """Rename one expected inode and reconcile success followed by an error."""
    deferred_error: BaseException | None = None
    try:
        _rename_noreplace_at(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )
    except BaseException as exc:
        deferred_error = exc
    source = _stat_at(source_fd, source_name)
    destination = _stat_at(destination_fd, destination_name)
    source_matches = source is not None and _metadata_identity(source) == _metadata_identity(
        expected
    )
    destination_matches = destination is not None and _metadata_identity(
        destination
    ) == _metadata_identity(expected)
    if destination_matches and not source_matches:
        return deferred_error
    if deferred_error is not None and source_matches and not destination_matches:
        raise deferred_error
    raise ValidationError(
        "validation transaction rename outcome is ambiguous for "
        f"{source_name} -> {destination_name}"
    ) from deferred_error


def _regular_file_digest_at(
    directory_fd: int,
    name: str,
    *,
    maximum_bytes: int,
) -> tuple[os.stat_result, bytes]:
    descriptor = os.open(
        name,
        (os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)),
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > maximum_bytes
        ):
            raise ValidationError(f"invalid validation transaction file: {name}")
        digest = hashlib.sha256()
        consumed = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            consumed += len(chunk)
            if consumed > maximum_bytes:
                raise ValidationError(
                    f"validation transaction file exceeds {maximum_bytes} bytes: {name}"
                )
            digest.update(chunk)
        after = os.fstat(descriptor)
        visible = _stat_at(directory_fd, name)
        if (
            not _same_file_version(after, before)
            or visible is None
            or not _same_file_version(visible, after)
        ):
            raise ValidationError(f"validation transaction file changed while hashing: {name}")
        return after, digest.digest()
    finally:
        os.close(descriptor)


def _verify_visible_artifact_parent(path: Path, directory_fd: int) -> None:
    visible_fd: int | None = None
    try:
        visible_fd = _open_real_directory(path.parent)
        visible = os.fstat(visible_fd)
        opened = os.fstat(directory_fd)
    except (OSError, ValidationError) as exc:
        raise ValidationError(
            f"validation artifact parent changed while writing {path}: {exc}"
        ) from exc
    finally:
        if visible_fd is not None:
            os.close(visible_fd)
    if (
        not stat.S_ISDIR(visible.st_mode)
        or visible.st_dev != opened.st_dev
        or visible.st_ino != opened.st_ino
    ):
        raise ValidationError(f"validation artifact parent changed while writing {path}")


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
        or (check_ctime and visible.st_ctime_ns != opened_metadata.st_ctime_ns)
    ):
        raise ValidationError(f"validation artifact changed while {operation} {path}")


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
                (os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)),
                dir_fd=directory_fd,
            )
        except FileNotFoundError:
            if missing_ok:
                return None
            raise
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValidationError(f"validation artifact must be a regular file: {path}")
        _verify_visible_regular_artifact(
            path,
            directory_fd,
            metadata,
            operation="recording",
        )
        _verify_visible_artifact_parent(path, directory_fd)
        return metadata
    except OSError as exc:
        raise ValidationError(f"cannot securely inspect validation artifact {path}: {exc}") from exc
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


@dataclass(frozen=True)
class _ArtifactVersion:
    metadata: os.stat_result | None
    digest: bytes | None


def _same_artifact_version(
    current: _ArtifactVersion,
    expected: _ArtifactVersion,
) -> bool:
    if current.metadata is None or expected.metadata is None:
        return (
            current.metadata is None
            and current.digest is None
            and expected.metadata is None
            and expected.digest is None
        )
    return current.digest == expected.digest and _same_file_version(
        current.metadata, expected.metadata
    )


def _regular_artifact_version(
    path: Path,
    *,
    missing_ok: bool = False,
) -> _ArtifactVersion:
    """Capture one exact regular-file identity and content digest."""
    directory_fd = _open_real_directory(path.parent)
    try:
        try:
            metadata, digest = _regular_file_digest_at(
                directory_fd,
                path.name,
                maximum_bytes=MAX_REPORT_ARTIFACT_BYTES,
            )
        except FileNotFoundError:
            if missing_ok:
                return _ArtifactVersion(None, None)
            raise
        _verify_visible_artifact_parent(path, directory_fd)
        return _ArtifactVersion(metadata, digest)
    except (OSError, ValidationError) as exc:
        raise ValidationError(f"cannot securely version validation artifact {path}: {exc}") from exc
    finally:
        os.close(directory_fd)


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
        destination_directory_fd = _open_real_directory(destination.parent)
        source_flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
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
            raise ValidationError(f"validation artifact source must be a regular file: {source}")
        if require_single_link and source_metadata.st_nlink != 1:
            raise ValidationError(f"validation artifact source must not be a hard link: {source}")
        if maximum_bytes is not None and source_metadata.st_size > maximum_bytes:
            raise ValidationError(
                f"validation artifact source exceeds {maximum_bytes} bytes: {source}"
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
                    if maximum_bytes is not None and copied_bytes > maximum_bytes:
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
            f"cannot securely copy validation artifact {source} to {destination}: {exc}"
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
        raise ValidationError(f"validation artifact is outside its case directory: {path}") from exc
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
        flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
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
            raise ValidationError(f"validation report artifact must be a regular file: {absolute}")
        if metadata.st_nlink != 1:
            raise ValidationError(f"validation report artifact must not be a hard link: {absolute}")
        with os.fdopen(descriptor, "rb") as input_file:
            descriptor = None
            payload = input_file.read(maximum_bytes + 1)
        if len(payload) > maximum_bytes:
            raise ValidationError(
                f"validation report artifact exceeds {maximum_bytes} bytes: {absolute}"
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
        raise ValidationError(f"validation report artifact does not exist: {absolute}") from exc
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
            (os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)),
            dir_fd=directory_fd,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValidationError(
                f"validation report media must be a single-link regular file: {absolute}"
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
        raise ValidationError(f"cannot securely validate report media {absolute}: {exc}") from exc
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
        raise ValidationError(f"invalid disagreement evidence in {work_dir}: {exc}") from exc


def _directory_identity(descriptor: int) -> tuple[int, int]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValidationError("validation transaction descriptor is not a directory")
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
    raise ValidationError(f"cannot reserve report stage below {case_dir}")


def _verify_case_artifact_stage(stage: _CaseArtifactStage) -> None:
    if stage.closed:
        raise ValidationError(f"validation report stage is already closed: {stage.path}")
    _acquire_case_artifact_stage(stage)
    assert stage.case_fd is not None
    assert stage.stage_fd is not None
    if _directory_identity(stage.case_fd) != stage.case_identity:
        raise ValidationError(
            f"validation case directory changed during report staging: {stage.case_dir}"
        )
    if _directory_identity(stage.stage_fd) != stage.stage_identity:
        raise ValidationError(
            f"validation report stage changed during report staging: {stage.path}"
        )
    visible_case_fd: int | None = None
    visible_stage_fd: int | None = None
    try:
        visible_case_fd = _open_real_directory(stage.case_dir)
        if _directory_identity(visible_case_fd) != stage.case_identity:
            raise ValidationError(
                f"validation case directory was replaced during report staging: {stage.case_dir}"
            )
        visible_stage_fd = os.open(
            stage.name,
            _secure_directory_flags(),
            dir_fd=visible_case_fd,
        )
        if _directory_identity(visible_stage_fd) != stage.stage_identity:
            raise ValidationError(
                f"validation report stage was replaced during report staging: {stage.path}"
            )
        held_visible = os.stat(
            stage.name,
            dir_fd=stage.case_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(held_visible.st_mode)
            or (held_visible.st_dev, held_visible.st_ino) != stage.stage_identity
        ):
            raise ValidationError(
                f"validation report stage changed below its case directory: {stage.path}"
            )
    except (OSError, ValueError) as exc:
        raise ValidationError(f"cannot verify validation report stage {stage.path}: {exc}") from exc
    finally:
        if visible_stage_fd is not None:
            os.close(visible_stage_fd)
        if visible_case_fd is not None:
            os.close(visible_case_fd)


def _acquire_case_artifact_stage(
    stage: _CaseArtifactStage,
) -> None:
    if stage.closed:
        raise ValidationError(f"validation report stage is already closed: {stage.path}")
    if stage.case_fd is not None or stage.stage_fd is not None:
        if stage.case_fd is None or stage.stage_fd is None:
            raise ValidationError(
                f"validation report stage has inconsistent descriptors: {stage.path}"
            )
        return
    case_fd: int | None = None
    stage_fd: int | None = None
    try:
        case_fd = _open_real_directory(stage.case_dir)
        if _directory_identity(case_fd) != stage.case_identity:
            raise ValidationError(
                f"validation case directory was replaced during report staging: {stage.case_dir}"
            )
        stage_fd = os.open(
            stage.name,
            _secure_directory_flags(),
            dir_fd=case_fd,
        )
        if _directory_identity(stage_fd) != stage.stage_identity:
            raise ValidationError(
                f"validation report stage was replaced during report staging: {stage.path}"
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
    root_metadata = os.fstat(root_fd)
    try:
        pending: list[tuple[str, ...]] = [()]
        entries_to_remove: list[tuple[tuple[str, ...], os.stat_result]] = []
        visited = 0
        while pending:
            relative = pending.pop()
            if len(relative) > MAX_TRANSACTION_TREE_DEPTH:
                raise ValidationError(
                    f"validation transaction cleanup nesting exceeds {MAX_TRANSACTION_TREE_DEPTH}"
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
                        metadata = entry.stat(follow_symlinks=False)
                        entries_to_remove.append((child, metadata))
                        if stat.S_ISDIR(metadata.st_mode):
                            pending.append(child)
            finally:
                os.close(directory_fd)
        for relative, expected in reversed(entries_to_remove):
            directory_fd = _open_relative_directory_at(
                root_fd,
                relative[:-1],
            )
            try:
                current = _stat_at(directory_fd, relative[-1])
                matches = current is not None and (
                    (
                        stat.S_ISDIR(current.st_mode)
                        and _metadata_identity(current) == _metadata_identity(expected)
                    )
                    if stat.S_ISDIR(expected.st_mode)
                    else _same_file_version(current, expected)
                )
                if not matches:
                    raise ValidationError(
                        f"validation transaction cleanup entry changed: {relative[-1]}"
                    )
                if stat.S_ISDIR(expected.st_mode):
                    os.rmdir(relative[-1], dir_fd=directory_fd)
                else:
                    os.unlink(relative[-1], dir_fd=directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        os.close(root_fd)
    visible_root = _stat_at(parent_fd, name)
    if (
        visible_root is None
        or not stat.S_ISDIR(visible_root.st_mode)
        or _metadata_identity(visible_root) != _metadata_identity(root_metadata)
    ):
        raise ValidationError(f"validation transaction cleanup root changed: {name}")
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


def _directory_tree_fingerprint_at(
    parent_fd: int,
    name: str,
) -> tuple[os.stat_result, bytes]:
    root_fd = os.open(
        name,
        _secure_directory_flags(),
        dir_fd=parent_fd,
    )
    try:
        root_before = os.fstat(root_fd)
        digest = hashlib.sha256()
        pending: list[tuple[str, ...]] = [()]
        visited = 0
        total_bytes = 0
        while pending:
            relative = pending.pop()
            if len(relative) > MAX_TRANSACTION_TREE_DEPTH:
                raise ValidationError(
                    "validation transaction fingerprint nesting exceeds "
                    f"{MAX_TRANSACTION_TREE_DEPTH}"
                )
            directory_fd = _open_relative_directory_at(
                root_fd,
                relative,
            )
            try:
                before = os.fstat(directory_fd)
                with os.scandir(directory_fd) as entries:
                    names = sorted(
                        (entry.name for entry in entries),
                        key=os.fsencode,
                    )
                encoded_relative = b"/".join(os.fsencode(component) for component in relative)
                digest.update(b"D")
                digest.update(len(encoded_relative).to_bytes(8, "big"))
                digest.update(encoded_relative)
                digest.update(stat.S_IMODE(before.st_mode).to_bytes(4, "big"))
                for child_name in names:
                    visited += 1
                    if visited > MAX_TRANSACTION_TREE_ENTRIES:
                        raise ValidationError(
                            "validation transaction fingerprint exceeds "
                            f"{MAX_TRANSACTION_TREE_ENTRIES} entries"
                        )
                    child = (*relative, child_name)
                    child_metadata = os.stat(
                        child_name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    encoded_child = b"/".join(os.fsencode(component) for component in child)
                    if stat.S_ISDIR(child_metadata.st_mode):
                        child_fd = os.open(
                            child_name,
                            _secure_directory_flags(),
                            dir_fd=directory_fd,
                        )
                        try:
                            opened = os.fstat(child_fd)
                            if _metadata_identity(opened) != _metadata_identity(child_metadata):
                                raise ValidationError(
                                    "validation transaction directory "
                                    f"changed while fingerprinting: "
                                    f"{child_name}"
                                )
                        finally:
                            os.close(child_fd)
                        pending.append(child)
                        continue
                    if not stat.S_ISREG(child_metadata.st_mode):
                        raise ValidationError(
                            "validation transaction tree contains a "
                            f"non-regular entry: {child_name}"
                        )
                    remaining_bytes = MAX_REPORT_MEDIA_BYTES - total_bytes
                    file_metadata, file_digest = _regular_file_digest_at(
                        directory_fd,
                        child_name,
                        maximum_bytes=remaining_bytes,
                    )
                    if _metadata_identity(file_metadata) != _metadata_identity(child_metadata):
                        raise ValidationError(
                            "validation transaction file changed while "
                            f"fingerprinting: {child_name}"
                        )
                    total_bytes += file_metadata.st_size
                    digest.update(b"F")
                    digest.update(len(encoded_child).to_bytes(8, "big"))
                    digest.update(encoded_child)
                    digest.update(
                        stat.S_IMODE(file_metadata.st_mode).to_bytes(
                            4,
                            "big",
                        )
                    )
                    digest.update(file_metadata.st_size.to_bytes(8, "big"))
                    digest.update(file_digest)
                after = os.fstat(directory_fd)
                if not _same_directory_version(after, before):
                    raise ValidationError(
                        f"validation transaction directory changed while fingerprinting: {name}"
                    )
            finally:
                os.close(directory_fd)
        root_after = os.fstat(root_fd)
        visible = _stat_at(parent_fd, name)
        if (
            not _same_directory_version(root_after, root_before)
            or visible is None
            or not _same_directory_version(visible, root_after)
        ):
            raise ValidationError(
                f"validation transaction tree changed while fingerprinting: {name}"
            )
        return root_after, digest.digest()
    finally:
        os.close(root_fd)


def _cleanup_case_artifact_stage(
    stage: _CaseArtifactStage,
    *,
    anchored_case_fd: int | None = None,
) -> None:
    if stage.closed:
        return
    if not stage.name.startswith(".report-stage-") or not stage.case_dir.name:
        raise ValidationError(f"refusing to remove unexpected report stage {stage.path}")
    try:
        if anchored_case_fd is None:
            try:
                _acquire_case_artifact_stage(stage)
            except ValidationError:
                return
            assert stage.case_fd is not None
            case_fd = stage.case_fd
        else:
            if _directory_identity(anchored_case_fd) != stage.case_identity:
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
            or (visible.st_dev, visible.st_ino) != stage.stage_identity
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
    next_fingerprint: bytes
    original_metadata: os.stat_result | None
    original_fingerprint: bytes | None
    backup_name: str = ""
    backup_metadata: os.stat_result | None = None
    backup_fingerprint: bytes | None = None
    installed_metadata: os.stat_result | None = None
    backed_up: bool = False
    installed: bool = False
    preserve_backup: bool = False
    preserve_next: bool = False
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
            f"validation staged reproduction must be a directory: {stage.path / 'repro'}"
        )
    staged_repro, staged_fingerprint = _directory_tree_fingerprint_at(stage.stage_fd, "repro")
    try:
        original, original_fingerprint = _directory_tree_fingerprint_at(stage.case_fd, "repro")
    except FileNotFoundError:
        original = None
        original_fingerprint = None
    if original is not None and not stat.S_ISDIR(original.st_mode):
        raise ValidationError(
            f"validation reproduction path must be a directory: {stage.case_dir / 'repro'}"
        )
    next_name, deferred_error = _rename_to_unique_noreplace(
        stage.stage_fd,
        "repro",
        stage.case_fd,
        prefix=".repro-next",
        expected=staged_repro,
    )
    try:
        next_metadata, next_fingerprint = _directory_tree_fingerprint_at(
            stage.case_fd,
            next_name,
        )
        if next_fingerprint != staged_fingerprint:
            raise ValidationError(
                "validation staged reproduction changed while moving into "
                f"transaction state: {stage.case_dir / next_name}"
            )
        if deferred_error is not None:
            raise deferred_error
    except BaseException as exc:
        recovery_error: BaseException | None = None
        try:
            recovery_error = _rename_expected_noreplace_at(
                stage.case_fd,
                next_name,
                stage.stage_fd,
                "repro",
                expected=staged_repro,
            )
        except BaseException as rollback_exc:
            recovery_error = rollback_exc
        if recovery_error is not None:
            note = (
                "validation staged reproduction recovery incomplete; "
                f"preserved path {stage.case_dir / next_name}: "
                f"{recovery_error}"
            )
            if hasattr(exc, "add_note"):
                exc.add_note(note)
            else:
                print(note, file=sys.stderr)
        raise
    update = _CaseDirectoryUpdate(
        stage=stage,
        next_name=next_name,
        next_metadata=next_metadata,
        next_fingerprint=next_fingerprint,
        original_metadata=original,
        original_fingerprint=original_fingerprint,
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
        staged, staged_fingerprint = _directory_tree_fingerprint_at(
            case_fd,
            update.next_name,
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
        or staged_fingerprint != update.next_fingerprint
    ):
        raise ValidationError(
            "validation staged reproduction changed before report "
            f"publication: {stage.case_dir / update.next_name}"
        )
    try:
        current, current_fingerprint = _directory_tree_fingerprint_at(case_fd, "repro")
    except FileNotFoundError:
        current = None
        current_fingerprint = None
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
        or current_fingerprint != update.original_fingerprint
    ):
        raise ValidationError(
            f"validation reproduction changed before report publication: {stage.case_dir / 'repro'}"
        )


def _commit_case_directory_update(
    update: _CaseDirectoryUpdate,
) -> None:
    try:
        _verify_case_directory_target(update)
        case_fd = update.anchor_fd if update.anchor_fd is not None else update.stage.case_fd
        assert case_fd is not None
        if update.original_metadata is not None:
            (
                update.backup_name,
                deferred_error,
            ) = _rename_to_unique_noreplace(
                case_fd,
                "repro",
                case_fd,
                prefix=".repro-previous",
                expected=update.original_metadata,
            )
            update.backed_up = True
            (
                update.backup_metadata,
                update.backup_fingerprint,
            ) = _directory_tree_fingerprint_at(
                case_fd,
                update.backup_name,
            )
            if (
                update.backup_metadata is None
                or not stat.S_ISDIR(update.backup_metadata.st_mode)
                or _metadata_identity(update.backup_metadata)
                != _metadata_identity(update.original_metadata)
                or update.backup_metadata.st_mtime_ns != update.original_metadata.st_mtime_ns
                or update.backup_fingerprint != update.original_fingerprint
            ):
                raise ValidationError(
                    "validation reproduction changed while moving it to "
                    f"transaction backup: "
                    f"{update.stage.case_dir / 'repro'}"
                )
            if deferred_error is not None:
                raise deferred_error
        deferred_error = _rename_expected_noreplace_at(
            case_fd,
            update.next_name,
            case_fd,
            "repro",
            expected=update.next_metadata,
        )
        update.installed = True
        update.installed_metadata = update.next_metadata
        if deferred_error is not None:
            raise deferred_error
        installed, installed_fingerprint = _directory_tree_fingerprint_at(case_fd, "repro")
        if (
            installed is None
            or not stat.S_ISDIR(installed.st_mode)
            or installed.st_dev != update.next_metadata.st_dev
            or installed.st_ino != update.next_metadata.st_ino
            or installed.st_mtime_ns != update.next_metadata.st_mtime_ns
            or installed_fingerprint != update.next_fingerprint
        ):
            raise ValidationError(
                "validation staged reproduction changed during report "
                f"publication: {update.stage.case_dir / 'repro'}"
            )
        update.installed_metadata = installed
        _verify_committed_case_directory_update(update)
    finally:
        if update.anchor_fd is None:
            _release_case_artifact_stage(update.stage)


def _verify_committed_case_directory_update(
    update: _CaseDirectoryUpdate,
) -> None:
    if not update.installed or update.installed_metadata is None:
        raise ValidationError(
            "validation reproduction transaction is not installed: "
            f"{update.stage.case_dir / 'repro'}"
        )
    if update.anchor_fd is None:
        _acquire_case_artifact_stage(update.stage)
    visible_case_fd: int | None = None
    try:
        visible_case_fd = _open_real_directory(update.stage.case_dir)
        if _directory_identity(visible_case_fd) != update.stage.case_identity:
            raise ValidationError(
                "validation case directory changed after report "
                f"publication: {update.stage.case_dir}"
            )
        try:
            visible, visible_fingerprint = _directory_tree_fingerprint_at(
                visible_case_fd,
                "repro",
            )
        except FileNotFoundError:
            visible = None
            visible_fingerprint = None
        if (
            visible is None
            or not _same_directory_version(
                visible,
                update.installed_metadata,
            )
            or visible_fingerprint != update.next_fingerprint
        ):
            raise ValidationError(
                "validation reproduction is not visible after report "
                f"publication: {update.stage.case_dir / 'repro'}"
            )
    finally:
        if visible_case_fd is not None:
            os.close(visible_case_fd)
        if update.anchor_fd is None:
            _release_case_artifact_stage(update.stage)


def _rollback_case_directory_update(
    update: _CaseDirectoryUpdate,
) -> None:
    if update.anchor_fd is None:
        _acquire_case_artifact_stage(update.stage)
    conflicts: list[str] = []
    try:
        case_fd = update.anchor_fd if update.anchor_fd is not None else update.stage.case_fd
        assert case_fd is not None
        if update.installed:
            current = _stat_at(case_fd, "repro")
            expected = update.installed_metadata or update.next_metadata
            if (
                current is not None
                and stat.S_ISDIR(current.st_mode)
                and _metadata_identity(current) == _metadata_identity(expected)
            ):
                moved_name, deferred_error = _rename_to_unique_noreplace(
                    case_fd,
                    "repro",
                    case_fd,
                    prefix=".repro-rollback",
                    expected=expected,
                )
                moved, moved_fingerprint = _directory_tree_fingerprint_at(
                    case_fd,
                    moved_name,
                )
                update.next_name = moved_name
                if (
                    not stat.S_ISDIR(moved.st_mode)
                    or _metadata_identity(moved) != _metadata_identity(expected)
                    or moved_fingerprint != update.next_fingerprint
                ):
                    update.preserve_next = True
                    conflicts.append(
                        "transaction reproduction changed while moving to "
                        f"recovery path "
                        f"{update.stage.case_dir / moved_name}"
                    )
                else:
                    update.next_metadata = moved
                    if not _same_directory_version(current, expected) or deferred_error is not None:
                        update.preserve_next = True
                        conflicts.append(
                            "transaction reproduction was modified; "
                            f"preserved at "
                            f"{update.stage.case_dir / moved_name}"
                        )
            else:
                conflicts.append(
                    f"concurrent reproduction left untouched at {update.stage.case_dir / 'repro'}"
                )
            update.installed = False
        if update.backed_up:
            try:
                backup, backup_fingerprint = _directory_tree_fingerprint_at(
                    case_fd,
                    update.backup_name,
                )
            except FileNotFoundError:
                backup = None
                backup_fingerprint = None
            expected_backup = update.backup_metadata
            if (
                backup is None
                or expected_backup is None
                or not _same_directory_version(
                    backup,
                    expected_backup,
                )
                or backup_fingerprint != update.backup_fingerprint
            ):
                update.preserve_backup = True
                conflicts.append(
                    "reproduction backup changed; preserved recovery path "
                    f"{update.stage.case_dir / update.backup_name}"
                )
            elif _stat_at(case_fd, "repro") is not None:
                update.preserve_backup = True
                conflicts.append(
                    "concurrent reproduction prevented rollback; original "
                    f"preserved at "
                    f"{update.stage.case_dir / update.backup_name}"
                )
            else:
                try:
                    deferred_error = _rename_expected_noreplace_at(
                        case_fd,
                        update.backup_name,
                        case_fd,
                        "repro",
                        expected=backup,
                    )
                except FileExistsError:
                    update.preserve_backup = True
                    conflicts.append(
                        "concurrent reproduction prevented rollback; "
                        f"original preserved at "
                        f"{update.stage.case_dir / update.backup_name}"
                    )
                else:
                    (
                        restored,
                        restored_fingerprint,
                    ) = _directory_tree_fingerprint_at(
                        case_fd,
                        "repro",
                    )
                    if (
                        _metadata_identity(restored) != _metadata_identity(backup)
                        or restored_fingerprint != update.backup_fingerprint
                    ):
                        conflicts.append(
                            "reproduction changed while restoring rollback "
                            f"target {update.stage.case_dir / 'repro'}"
                        )
                    update.backed_up = False
                    if deferred_error is not None:
                        conflicts.append(
                            "reproduction rollback rename reported an "
                            "error after the original was restored"
                        )
        if conflicts:
            raise ValidationError(
                "validation reproduction rollback incomplete: " + "; ".join(conflicts)
            )
    finally:
        if update.anchor_fd is None:
            _release_case_artifact_stage(update.stage)


def _finalize_case_directory_update(
    update: _CaseDirectoryUpdate,
) -> list[str]:
    errors: list[str] = []
    if update.anchor_fd is None:
        try:
            _acquire_case_artifact_stage(update.stage)
        except ValidationError as exc:
            return [str(exc)]
    try:
        case_fd = update.anchor_fd if update.anchor_fd is not None else update.stage.case_fd
        assert case_fd is not None
        cleanup = (
            (
                update.backup_name,
                update.backup_metadata,
                update.preserve_backup,
            ),
            (
                update.next_name,
                update.next_metadata,
                update.preserve_next,
            ),
        )
        for name, expected, preserve in cleanup:
            if not name or expected is None or preserve:
                continue
            try:
                current, current_fingerprint = _directory_tree_fingerprint_at(case_fd, name)
            except FileNotFoundError:
                continue
            expected_fingerprint = (
                update.backup_fingerprint if name == update.backup_name else update.next_fingerprint
            )
            if (
                not _same_directory_version(current, expected)
                or current_fingerprint != expected_fingerprint
            ):
                errors.append(
                    "validation transaction preserved changed directory "
                    f"recovery path {update.stage.case_dir / name}"
                )
                continue
            try:
                quarantine_name, deferred_error = _rename_to_unique_noreplace(
                    case_fd,
                    name,
                    case_fd,
                    prefix=".repro-cleanup",
                    expected=current,
                )
            except (OSError, ValidationError) as exc:
                errors.append(
                    "validation transaction could not quarantine "
                    f"{update.stage.case_dir / name}: {exc}"
                )
                continue
            try:
                quarantined, quarantined_fingerprint = _directory_tree_fingerprint_at(
                    case_fd,
                    quarantine_name,
                )
            except (OSError, ValidationError) as exc:
                errors.append(
                    "validation transaction preserved unverifiable "
                    f"directory recovery path "
                    f"{update.stage.case_dir / quarantine_name}: {exc}"
                )
                continue
            if (
                deferred_error is not None
                or _metadata_identity(quarantined) != _metadata_identity(expected)
                or quarantined_fingerprint != expected_fingerprint
            ):
                errors.append(
                    "validation transaction preserved directory recovery "
                    f"path {update.stage.case_dir / quarantine_name}"
                )
                continue
            try:
                _remove_directory_tree_at(
                    case_fd,
                    quarantine_name,
                )
            except (FileNotFoundError, OSError, ValidationError) as exc:
                errors.append(
                    "validation transaction cleanup incomplete; preserved "
                    f"or changed path "
                    f"{update.stage.case_dir / quarantine_name}: {exc}"
                )
    finally:
        if update.anchor_fd is None:
            _release_case_artifact_stage(update.stage)
    return errors


@dataclass
class _FileRemoval:
    path: Path
    parent_fd: int | None
    parent_identity: tuple[int, int]
    original_metadata: os.stat_result
    original_digest: bytes
    backup_name: str = ""
    backup_metadata: os.stat_result | None = None
    backup_digest: bytes | None = None
    removed: bool = False
    preserve_backup: bool = False
    closed: bool = False


def _prepare_file_removal(path: Path) -> _FileRemoval | None:
    _ensure_real_directory(
        path.parent,
        description="validation transaction parent",
    )
    parent_fd = _open_real_directory(path.parent)
    try:
        try:
            original, original_digest = _regular_file_digest_at(
                parent_fd,
                path.name,
                maximum_bytes=MAX_REPORT_ARTIFACT_BYTES,
            )
        except FileNotFoundError:
            return None
        return _FileRemoval(
            path=path,
            parent_fd=None,
            parent_identity=_directory_identity(parent_fd),
            original_metadata=original,
            original_digest=original_digest,
        )
    finally:
        os.close(parent_fd)


def _acquire_file_removal(removal: _FileRemoval) -> None:
    if removal.closed:
        raise ValidationError(f"validation removal transaction is already closed: {removal.path}")
    if removal.parent_fd is not None:
        return
    parent_fd = _open_real_directory(removal.path.parent)
    try:
        if _directory_identity(parent_fd) != removal.parent_identity:
            raise ValidationError(
                f"validation removal transaction parent was replaced: {removal.path.parent}"
            )
    except BaseException:
        os.close(parent_fd)
        raise
    removal.parent_fd = parent_fd


def _release_file_removal(removal: _FileRemoval) -> None:
    if removal.parent_fd is not None:
        os.close(removal.parent_fd)
        removal.parent_fd = None


def _verify_file_removal_target(removal: _FileRemoval) -> None:
    _acquire_file_removal(removal)
    assert removal.parent_fd is not None
    _verify_visible_artifact_parent(removal.path, removal.parent_fd)
    try:
        current, current_digest = _regular_file_digest_at(
            removal.parent_fd,
            removal.path.name,
            maximum_bytes=MAX_REPORT_ARTIFACT_BYTES,
        )
    except FileNotFoundError as exc:
        raise ValidationError(f"validation removal target changed: {removal.path}") from exc
    if (
        not _same_file_version(current, removal.original_metadata)
        or current_digest != removal.original_digest
    ):
        raise ValidationError(f"validation removal target changed: {removal.path}")


def _commit_file_removal(removal: _FileRemoval) -> None:
    try:
        _verify_file_removal_target(removal)
        assert removal.parent_fd is not None
        removal.backup_name, deferred_error = _rename_to_unique_noreplace(
            removal.parent_fd,
            removal.path.name,
            removal.parent_fd,
            prefix=f".{removal.path.name}.stale",
            expected=removal.original_metadata,
        )
        removal.removed = True
        (
            removal.backup_metadata,
            removal.backup_digest,
        ) = _regular_file_digest_at(
            removal.parent_fd,
            removal.backup_name,
            maximum_bytes=MAX_REPORT_ARTIFACT_BYTES,
        )
        if (
            not _same_renamed_file_version(
                removal.backup_metadata,
                removal.original_metadata,
            )
            or removal.backup_digest != removal.original_digest
        ):
            raise ValidationError(
                f"validation removal target changed while moving to recovery: {removal.path}"
            )
        if deferred_error is not None:
            raise deferred_error
        _verify_committed_file_removal(removal)
    finally:
        _release_file_removal(removal)


def _verify_committed_file_removal(removal: _FileRemoval) -> None:
    if (
        not removal.removed
        or not removal.backup_name
        or removal.backup_metadata is None
        or removal.backup_digest is None
    ):
        raise ValidationError(f"validation removal transaction is not installed: {removal.path}")
    _acquire_file_removal(removal)
    assert removal.parent_fd is not None
    _verify_visible_artifact_parent(removal.path, removal.parent_fd)
    if _stat_at(removal.parent_fd, removal.path.name) is not None:
        raise ValidationError(
            f"validation removal target is still visible after publication: {removal.path}"
        )
    backup, backup_digest = _regular_file_digest_at(
        removal.parent_fd,
        removal.backup_name,
        maximum_bytes=MAX_REPORT_ARTIFACT_BYTES,
    )
    if (
        not _same_file_version(backup, removal.backup_metadata)
        or backup_digest != removal.backup_digest
    ):
        raise ValidationError(
            "validation removal recovery file changed after publication: "
            f"{removal.path.parent / removal.backup_name}"
        )


def _rollback_file_removal(removal: _FileRemoval) -> None:
    _acquire_file_removal(removal)
    conflicts: list[str] = []
    try:
        assert removal.parent_fd is not None
        if not removal.removed:
            return
        if _stat_at(removal.parent_fd, removal.path.name) is not None:
            removal.preserve_backup = True
            conflicts.append(
                f"concurrent file prevented rollback at {removal.path}; "
                f"original preserved at "
                f"{removal.path.parent / removal.backup_name}"
            )
        else:
            try:
                backup, backup_digest = _regular_file_digest_at(
                    removal.parent_fd,
                    removal.backup_name,
                    maximum_bytes=MAX_REPORT_ARTIFACT_BYTES,
                )
            except FileNotFoundError:
                backup = None
                backup_digest = None
            if (
                backup is None
                or removal.backup_metadata is None
                or not _same_file_version(
                    backup,
                    removal.backup_metadata,
                )
                or backup_digest != removal.backup_digest
            ):
                removal.preserve_backup = True
                conflicts.append(
                    "validation removal backup changed; preserved recovery "
                    f"path {removal.path.parent / removal.backup_name}"
                )
            else:
                try:
                    deferred_error = _rename_expected_noreplace_at(
                        removal.parent_fd,
                        removal.backup_name,
                        removal.parent_fd,
                        removal.path.name,
                        expected=backup,
                    )
                except FileExistsError:
                    removal.preserve_backup = True
                    conflicts.append(
                        "concurrent file prevented removal rollback; original "
                        f"preserved at "
                        f"{removal.path.parent / removal.backup_name}"
                    )
                else:
                    restored, restored_digest = _regular_file_digest_at(
                        removal.parent_fd,
                        removal.path.name,
                        maximum_bytes=MAX_REPORT_ARTIFACT_BYTES,
                    )
                    if (
                        not _same_renamed_file_version(
                            restored,
                            removal.original_metadata,
                        )
                        or restored_digest != removal.original_digest
                    ):
                        conflicts.append(
                            f"validation removal target changed while restoring {removal.path}"
                        )
                    removal.removed = False
                    if deferred_error is not None:
                        conflicts.append(
                            "removal rollback rename reported an error after "
                            "the original was restored"
                        )
        if conflicts:
            raise ValidationError(
                "validation file removal rollback incomplete: " + "; ".join(conflicts)
            )
    finally:
        _release_file_removal(removal)


def _finalize_file_removal(removal: _FileRemoval) -> list[str]:
    if removal.closed:
        return []
    errors: list[str] = []
    try:
        _acquire_file_removal(removal)
        assert removal.parent_fd is not None
        if removal.removed and not removal.preserve_backup:
            try:
                backup, backup_digest = _regular_file_digest_at(
                    removal.parent_fd,
                    removal.backup_name,
                    maximum_bytes=MAX_REPORT_ARTIFACT_BYTES,
                )
            except (FileNotFoundError, OSError, ValidationError) as exc:
                errors.append(
                    "validation removal cleanup preserved recovery path "
                    f"{removal.path.parent / removal.backup_name}: {exc}"
                )
            else:
                if (
                    removal.backup_metadata is None
                    or not _same_file_version(
                        backup,
                        removal.backup_metadata,
                    )
                    or backup_digest != removal.backup_digest
                ):
                    errors.append(
                        "validation removal cleanup preserved changed recovery "
                        f"path {removal.path.parent / removal.backup_name}"
                    )
                else:
                    try:
                        os.unlink(
                            removal.backup_name,
                            dir_fd=removal.parent_fd,
                        )
                    except OSError as exc:
                        errors.append(
                            "validation removal cleanup incomplete at "
                            f"{removal.path.parent / removal.backup_name}: "
                            f"{exc}"
                        )
                    else:
                        removal.removed = False
    except (OSError, ValidationError) as exc:
        errors.append(str(exc))
    finally:
        removal.closed = True
        _release_file_removal(removal)
    return errors


@dataclass
class _FileUpdate:
    path: Path
    parent_fd: int | None
    parent_identity: tuple[int, int]
    original_metadata: os.stat_result | None
    original_digest: bytes | None
    next_name: str
    next_metadata: os.stat_result
    payload_digest: bytes
    backup_name: str = ""
    backup_metadata: os.stat_result | None = None
    backup_digest: bytes | None = None
    installed_metadata: os.stat_result | None = None
    backed_up: bool = False
    installed: bool = False
    preserve_backup: bool = False
    preserve_next: bool = False
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
        if original is not None and (not stat.S_ISREG(original.st_mode) or original.st_nlink != 1):
            raise ValidationError(
                f"validation transaction target must be a single-link regular file: {path}"
            )
        if original is None:
            original_digest = None
        else:
            original, original_digest = _regular_file_digest_at(
                parent_fd,
                path.name,
                maximum_bytes=MAX_REPORT_ARTIFACT_BYTES,
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
            original_digest=original_digest,
            next_name=next_name,
            next_metadata=next_metadata,
            payload_digest=hashlib.sha256(payload).digest(),
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
    staged, staged_digest = _regular_file_digest_at(
        update.parent_fd,
        update.next_name,
        maximum_bytes=MAX_REPORT_ARTIFACT_BYTES,
    )
    if (
        not _same_file_version(staged, update.next_metadata)
        or staged_digest != update.payload_digest
    ):
        raise ValidationError(f"validation transaction staged file changed: {update.path}")
    try:
        current, current_digest = _regular_file_digest_at(
            update.parent_fd,
            update.path.name,
            maximum_bytes=MAX_REPORT_ARTIFACT_BYTES,
        )
    except FileNotFoundError:
        current = None
        current_digest = None
    expected = update.original_metadata
    if expected is None:
        if current is not None:
            raise ValidationError(f"validation transaction target changed: {update.path}")
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
        or current_digest != update.original_digest
    ):
        raise ValidationError(f"validation transaction target changed: {update.path}")


def _acquire_file_update(update: _FileUpdate) -> None:
    if update.closed:
        raise ValidationError(f"validation transaction is already closed: {update.path}")
    if update.parent_fd is not None:
        return
    if update.anchor_fd is not None:
        update.parent_fd = update.anchor_fd
        return
    parent_fd = _open_real_directory(update.path.parent)
    try:
        if _directory_identity(parent_fd) != update.parent_identity:
            raise ValidationError(
                f"validation transaction parent was replaced: {update.path.parent}"
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
        visible_next, visible_next_digest = _regular_file_digest_at(
            update.parent_fd,
            update.next_name,
            maximum_bytes=MAX_REPORT_ARTIFACT_BYTES,
        )
        if (
            not stat.S_ISREG(visible_next.st_mode)
            or visible_next.st_dev != update.next_metadata.st_dev
            or visible_next.st_ino != update.next_metadata.st_ino
            or visible_next.st_nlink != update.next_metadata.st_nlink
            or visible_next.st_size != update.next_metadata.st_size
            or visible_next.st_ctime_ns != update.next_metadata.st_ctime_ns
            or visible_next.st_mtime_ns != update.next_metadata.st_mtime_ns
            or visible_next_digest != update.payload_digest
        ):
            raise ValidationError(f"validation transaction staged file changed: {update.path}")
        if update.original_metadata is not None:
            (
                update.backup_name,
                deferred_error,
            ) = _rename_to_unique_noreplace(
                update.parent_fd,
                update.path.name,
                update.parent_fd,
                prefix=f".{update.path.name}.previous",
                expected=update.original_metadata,
            )
            update.backed_up = True
            (
                update.backup_metadata,
                update.backup_digest,
            ) = _regular_file_digest_at(
                update.parent_fd,
                update.backup_name,
                maximum_bytes=MAX_REPORT_ARTIFACT_BYTES,
            )
            if (
                update.backup_metadata is None
                or not stat.S_ISREG(update.backup_metadata.st_mode)
                or update.backup_metadata.st_nlink != 1
                or _metadata_identity(update.backup_metadata)
                != _metadata_identity(update.original_metadata)
                or update.backup_metadata.st_size != update.original_metadata.st_size
                or update.backup_metadata.st_mtime_ns != update.original_metadata.st_mtime_ns
                or update.backup_digest != update.original_digest
            ):
                raise ValidationError(
                    f"validation transaction target changed while moving to backup: {update.path}"
                )
            if deferred_error is not None:
                raise deferred_error
        deferred_error = _rename_expected_noreplace_at(
            update.parent_fd,
            update.next_name,
            update.parent_fd,
            update.path.name,
            expected=update.next_metadata,
        )
        update.installed = True
        update.installed_metadata = update.next_metadata
        if deferred_error is not None:
            raise deferred_error
        installed, installed_digest = _regular_file_digest_at(
            update.parent_fd,
            update.path.name,
            maximum_bytes=MAX_REPORT_ARTIFACT_BYTES,
        )
        if (
            installed is None
            or not stat.S_ISREG(installed.st_mode)
            or installed.st_dev != update.next_metadata.st_dev
            or installed.st_ino != update.next_metadata.st_ino
            or installed.st_nlink != update.next_metadata.st_nlink
            or installed.st_size != update.next_metadata.st_size
            or installed.st_mtime_ns != update.next_metadata.st_mtime_ns
            or installed_digest != update.payload_digest
        ):
            raise ValidationError(
                f"validation staged file changed during publication: {update.path}"
            )
        update.installed_metadata = installed
        _verify_committed_file_update(update)
    finally:
        _release_file_update(update)


def _verify_committed_file_update(update: _FileUpdate) -> None:
    if not update.installed or update.installed_metadata is None:
        raise ValidationError(f"validation transaction is not installed: {update.path}")
    visible_parent_fd: int | None = None
    try:
        visible_parent_fd = _open_real_directory(update.path.parent)
        if _directory_identity(visible_parent_fd) != update.parent_identity:
            raise ValidationError(
                f"validation transaction parent changed after publication: {update.path.parent}"
            )
        try:
            visible, visible_digest = _regular_file_digest_at(
                visible_parent_fd,
                update.path.name,
                maximum_bytes=MAX_REPORT_ARTIFACT_BYTES,
            )
        except FileNotFoundError:
            visible = None
            visible_digest = None
        if (
            visible is None
            or not _same_file_version(
                visible,
                update.installed_metadata,
            )
            or visible_digest != update.payload_digest
        ):
            raise ValidationError(
                f"validation transaction target is not visible after publication: {update.path}"
            )
    finally:
        if visible_parent_fd is not None:
            os.close(visible_parent_fd)


def _rollback_file_update(update: _FileUpdate) -> None:
    _acquire_file_update(update)
    conflicts: list[str] = []
    try:
        assert update.parent_fd is not None
        if update.installed:
            current = _stat_at(update.parent_fd, update.path.name)
            expected = update.installed_metadata or update.next_metadata
            if (
                current is not None
                and stat.S_ISREG(current.st_mode)
                and _metadata_identity(current) == _metadata_identity(expected)
            ):
                moved_name, deferred_error = _rename_to_unique_noreplace(
                    update.parent_fd,
                    update.path.name,
                    update.parent_fd,
                    prefix=f".{update.path.name}.rollback",
                    expected=expected,
                )
                moved, moved_digest = _regular_file_digest_at(
                    update.parent_fd,
                    moved_name,
                    maximum_bytes=MAX_REPORT_ARTIFACT_BYTES,
                )
                update.next_name = moved_name
                if (
                    not stat.S_ISREG(moved.st_mode)
                    or _metadata_identity(moved) != _metadata_identity(expected)
                    or moved_digest != update.payload_digest
                ):
                    update.preserve_next = True
                    conflicts.append(
                        "transaction file changed while moving to recovery "
                        f"path {update.path.parent / moved_name}"
                    )
                else:
                    update.next_metadata = moved
                    if not _same_file_version(current, expected) or deferred_error is not None:
                        update.preserve_next = True
                        conflicts.append(
                            "transaction file was modified; preserved at "
                            f"{update.path.parent / moved_name}"
                        )
            else:
                conflicts.append(f"concurrent file left untouched at {update.path}")
            update.installed = False
        if update.backed_up:
            try:
                backup, backup_digest = _regular_file_digest_at(
                    update.parent_fd,
                    update.backup_name,
                    maximum_bytes=MAX_REPORT_ARTIFACT_BYTES,
                )
            except FileNotFoundError:
                backup = None
                backup_digest = None
            expected_backup = update.backup_metadata
            if (
                backup is None
                or expected_backup is None
                or not _same_file_version(
                    backup,
                    expected_backup,
                )
                or backup_digest != update.backup_digest
            ):
                update.preserve_backup = True
                conflicts.append(
                    "transaction backup changed; preserved recovery path "
                    f"{update.path.parent / update.backup_name}"
                )
            elif _stat_at(update.parent_fd, update.path.name) is not None:
                update.preserve_backup = True
                conflicts.append(
                    f"concurrent file prevented rollback at {update.path}; "
                    f"original preserved at "
                    f"{update.path.parent / update.backup_name}"
                )
            else:
                try:
                    deferred_error = _rename_expected_noreplace_at(
                        update.parent_fd,
                        update.backup_name,
                        update.parent_fd,
                        update.path.name,
                        expected=backup,
                    )
                except FileExistsError:
                    update.preserve_backup = True
                    conflicts.append(
                        "concurrent file prevented rollback; original "
                        f"preserved at "
                        f"{update.path.parent / update.backup_name}"
                    )
                else:
                    restored, restored_digest = _regular_file_digest_at(
                        update.parent_fd,
                        update.path.name,
                        maximum_bytes=MAX_REPORT_ARTIFACT_BYTES,
                    )
                    if (
                        _metadata_identity(restored) != _metadata_identity(backup)
                        or restored_digest != update.backup_digest
                    ):
                        conflicts.append(
                            f"transaction target changed while restoring {update.path}"
                        )
                    update.backed_up = False
                    if deferred_error is not None:
                        conflicts.append(
                            "file rollback rename reported an error after the original was restored"
                        )
        if conflicts:
            raise ValidationError("validation file rollback incomplete: " + "; ".join(conflicts))
    finally:
        _release_file_update(update)


def _finalize_file_update(update: _FileUpdate) -> list[str]:
    if update.closed:
        return []
    errors: list[str] = []
    try:
        _acquire_file_update(update)
        assert update.parent_fd is not None
        cleanup = (
            (
                update.backup_name,
                update.backup_metadata,
                update.preserve_backup,
            ),
            (
                update.next_name,
                update.next_metadata,
                update.preserve_next,
            ),
        )
        for name, expected, preserve in cleanup:
            if not name or expected is None or preserve:
                continue
            try:
                current, current_digest = _regular_file_digest_at(
                    update.parent_fd,
                    name,
                    maximum_bytes=MAX_REPORT_ARTIFACT_BYTES,
                )
            except FileNotFoundError:
                continue
            expected_digest = (
                update.backup_digest if name == update.backup_name else update.payload_digest
            )
            if not _same_file_version(current, expected) or current_digest != expected_digest:
                errors.append(
                    "validation transaction preserved changed file "
                    f"recovery path {update.path.parent / name}"
                )
                continue
            try:
                quarantine_name, deferred_error = _rename_to_unique_noreplace(
                    update.parent_fd,
                    name,
                    update.parent_fd,
                    prefix=f".{update.path.name}.cleanup",
                    expected=current,
                )
            except (OSError, ValidationError) as exc:
                errors.append(
                    "validation transaction could not quarantine "
                    f"{update.path.parent / name}: {exc}"
                )
                continue
            try:
                quarantined, quarantined_digest = _regular_file_digest_at(
                    update.parent_fd,
                    quarantine_name,
                    maximum_bytes=MAX_REPORT_ARTIFACT_BYTES,
                )
            except (OSError, ValidationError) as exc:
                errors.append(
                    "validation transaction preserved unverifiable file "
                    f"recovery path "
                    f"{update.path.parent / quarantine_name}: {exc}"
                )
                continue
            if (
                deferred_error is not None
                or _metadata_identity(quarantined) != _metadata_identity(expected)
                or quarantined_digest != expected_digest
            ):
                errors.append(
                    "validation transaction preserved file recovery path "
                    f"{update.path.parent / quarantine_name}"
                )
                continue
            try:
                os.unlink(
                    quarantine_name,
                    dir_fd=update.parent_fd,
                )
            except OSError as exc:
                errors.append(
                    "validation transaction cleanup incomplete at "
                    f"{update.path.parent / quarantine_name}: {exc}"
                )
    except (OSError, ValidationError) as exc:
        errors.append(str(exc))
    finally:
        update.closed = True
        _release_file_update(update)
    return errors


def _commit_versioned_file_update(
    path: Path,
    payload: bytes,
    *,
    expected_target: _ArtifactVersion,
    before_commit: Callable[[], None] | None = None,
    after_commit: Callable[[], None] | None = None,
    description: str,
) -> _ArtifactVersion:
    """CAS-publish one file and roll it back if a surrounding check fails."""
    update: _FileUpdate | None = None
    try:
        update = _prepare_file_update(path, payload)
        prepared_target = _ArtifactVersion(
            update.original_metadata,
            update.original_digest,
        )
        if not _same_artifact_version(prepared_target, expected_target):
            raise ValidationError(f"{description} target changed before publication: {path}")
        try:
            if before_commit is not None:
                before_commit()
            _commit_file_update(update)
            if after_commit is not None:
                after_commit()
            _verify_committed_file_update(update)
        except BaseException as exc:
            try:
                _rollback_file_update(update)
            except (OSError, ValidationError) as rollback_exc:
                raise ValidationError(
                    f"{description} failed and rollback was incomplete: {rollback_exc}"
                ) from exc
            raise
        assert update.installed_metadata is not None
        return _ArtifactVersion(
            update.installed_metadata,
            update.payload_digest,
        )
    finally:
        active_error = sys.exc_info()[1]
        cleanup_errors = _finalize_file_update(update) if update is not None else []
        if cleanup_errors:
            cleanup_message = f"{description} cleanup incomplete: " + " | ".join(cleanup_errors)
            if active_error is not None:
                if hasattr(active_error, "add_note"):
                    active_error.add_note(cleanup_message)
                else:
                    print(cleanup_message, file=sys.stderr)
            else:
                raise ValidationError(cleanup_message)


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
        (os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)),
        dir_fd=stage.stage_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > maximum_bytes
        ):
            raise ValidationError(f"invalid staged validation artifact: {stage.path / name}")
        chunks = []
        consumed = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            consumed += len(chunk)
            if consumed > maximum_bytes:
                raise ValidationError(
                    f"staged validation artifact exceeds {maximum_bytes} bytes: {stage.path / name}"
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
                f"staged validation artifact changed while reading: {stage.path / name}"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)
        _release_case_artifact_stage(stage)


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


def _read_json_artifact(
    path: Path,
    *,
    missing_ok: bool = False,
    include_version: bool = False,
) -> Any:
    """Read one regular JSON artifact through an anchored parent directory."""
    try:
        directory_fd = _open_real_directory(path.parent)
    except ValidationError:
        if missing_ok and not path.parent.exists():
            if include_version:
                return None, _ArtifactVersion(None, None)
            return None
        raise
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(path.name, flags, dir_fd=directory_fd)
        except FileNotFoundError:
            if missing_ok:
                if include_version:
                    return None, _ArtifactVersion(None, None)
                return None
            raise
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValidationError(f"validation JSON artifact must be a regular file: {path}")
        if metadata.st_nlink != 1:
            raise ValidationError(f"validation JSON artifact must not be a hard link: {path}")
        if metadata.st_size > MAX_REPORT_ARTIFACT_BYTES:
            raise ValidationError(
                f"validation JSON artifact exceeds {MAX_REPORT_ARTIFACT_BYTES} bytes: {path}"
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
                after = os.fstat(input_file.fileno())
            except (UnicodeError, ValueError, RecursionError) as exc:
                raise ValidationError(f"invalid validation JSON artifact {path}: {exc}") from exc
        if not _same_file_version(after, metadata):
            raise ValidationError(f"validation JSON artifact changed while reading: {path}")
        _validate_report_json_depth(loaded, path=path)
        _verify_visible_regular_artifact(
            path,
            directory_fd,
            after,
            operation="reading",
        )
        _verify_visible_artifact_parent(path, directory_fd)
        if include_version:
            return loaded, _ArtifactVersion(
                after,
                hashlib.sha256(payload).digest(),
            )
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
        raise ValidationError(f"invalid validation JSON artifact for {path}: {exc}") from exc
    payload = rendered.encode("utf-8")
    if len(payload) > MAX_REPORT_ARTIFACT_BYTES:
        raise ValidationError(
            f"validation JSON artifact exceeds {MAX_REPORT_ARTIFACT_BYTES} bytes: {path}"
        )
    return payload


def _require_validation_run_id(output: Path, expected_run_id: str) -> None:
    if _current_run_id(output) != expected_run_id:
        raise ValidationError(
            "validation run changed while a comparison result was being published"
        )


def _publish_validation_result(
    output: Path,
    path: Path,
    value: Mapping[str, Any],
    *,
    expected_run_id: str,
    expected_target: _ArtifactVersion,
) -> _ArtifactVersion:
    """Publish canonical comparison evidence with run and target CAS checks."""
    result_run_id = value.get("run_id", "")
    if not isinstance(result_run_id, str):
        raise ValidationError("validation result run_id must be a string before publication")
    if result_run_id != expected_run_id:
        raise ValidationError("validation result does not belong to the expected validation run")
    payload = _json_artifact_payload(
        path,
        value,
        maximum_depth=MAX_VALIDATION_RESULT_JSON_DEPTH,
    )
    with _validation_output_publication_lock(output):
        revalidate_run = functools.partial(
            _require_validation_run_id,
            output,
            expected_run_id,
        )
        return _commit_versioned_file_update(
            path,
            payload,
            expected_target=expected_target,
            before_commit=revalidate_run,
            after_commit=revalidate_run,
            description="validation comparison publication",
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
            raise ValidationError(f"{path}: {name}.not_compared_reason must be a non-empty string")
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
        not isinstance(reference_cache_identity, str) or not reference_cache_identity.strip()
    ):
        raise ValidationError(f"{path}: {name}.reference_cache_identity must be a non-empty string")


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
        workload for spec in models.values() for workload in spec.get("workloads", [])
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
            reference_cache_identity = str(spec.get("reference_cache_identity", "") or "")
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
                f"reference cache identity {identity!r} spans different reference contracts"
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
                f"model {model} has no reference-consistency workloads: {not_compared_reason}"
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
        reference_cache_identity=str(spec.get("reference_cache_identity", "") or ""),
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


def _authoritative_gates_by_binding(
    catalog: Mapping[str, Any],
    *,
    suites: Mapping[str, dict[str, Any]],
    task_models: Mapping[str, dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    gates_by_binding: dict[tuple[str, str], dict[str, Any]] = {}
    for model_name, specification in catalog["models"].items():
        for workload in specification.get("workloads", []):
            resolved = task_eval.resolve_suite_for_model(
                suites[workload],
                task_models[model_name],
            )
            gates_by_binding[(model_name, workload)] = _validated_gate_configuration(
                resolved.get("gates", {}),
                field=f"resolved gates for {model_name}/{workload}",
            )
    return gates_by_binding


@functools.lru_cache(maxsize=1)
def _default_authoritative_gates_by_binding() -> dict[tuple[str, str], dict[str, Any]]:
    catalog = load_catalog()
    suites = {str(suite["id"]): suite for suite in task_eval.load_suites()}
    return _authoritative_gates_by_binding(
        catalog,
        suites=suites,
        task_models=_task_eval_models(DEFAULT_MODELS),
    )


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
        raise ValidationError(f"model {binding.model} has no reference-consistency workload")
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
                    f"\nValidation command could not complete: {type(exc).__name__}: {exc}\n"
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
            raise ValidationError("validation command log contains a NUL character")
        return rendered, ""
    try:
        data = json.loads(
            line,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except json.JSONDecodeError as exc:
        if strict_json:
            raise ValidationError(f"invalid native validation command record: {exc}") from exc
        return "", ""
    except (ValueError, RecursionError) as exc:
        raise ValidationError(f"invalid validation command log record: {exc}") from exc
    _validate_report_json_depth(data, path=Path("<command-log-record>"))
    if not isinstance(data, dict):
        if strict_json:
            raise ValidationError("native validation command records must be JSON objects")
        return "", ""
    command = data.get("command")
    if strict_json:
        sample_value = data.get("sample_id")
        if not isinstance(sample_value, str) or not sample_value.strip() or "\x00" in sample_value:
            raise ValidationError(
                "native validation command records require an exact "
                "non-empty, NUL-free sample_id string"
            )
        sample_id = sample_value
    else:
        sample_id = next(
            (str(data[name]) for name in _SAMPLE_ID_FIELDS if data.get(name) is not None),
            "",
        )
    if isinstance(command, list) and command:
        if any(not isinstance(token, str) or "\x00" in token for token in command):
            raise ValidationError("validation command list tokens must be NUL-free strings")
        if strict_json and not command[0].strip():
            raise ValidationError(
                "native validation command executable must be a non-empty, non-whitespace string"
            )
        return shlex.join(command), sample_id
    if isinstance(command, str):
        rendered = command.strip()
        if "\x00" in command:
            raise ValidationError("validation command contains a NUL character")
        if rendered:
            return rendered, sample_id
    if strict_json:
        raise ValidationError(
            "native validation command records must contain a non-empty command string or list"
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
        raise ValidationError(f"validation command artifact exceeds {maximum_bytes} bytes: {path}")
    try:
        return payload.decode("utf-8", errors=errors)
    except UnicodeError as exc:
        raise ValidationError(f"validation command artifact is not UTF-8: {path}: {exc}") from exc


def _parse_command_json(text: str, *, path: Path) -> Any:
    try:
        loaded = json.loads(
            text,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValidationError(f"invalid validation command artifact {path}: {exc}") from exc
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
            f"validation command log count exceeds {MAX_COMMAND_LOG_FILES}: {root}"
        )
    remaining_bytes = MAX_COMMAND_LOG_TOTAL_BYTES
    has_native_reference = any(
        path.name in {"hf_native_run.log", "hf_native_commands.jsonl"} for path in log_paths
    )
    has_native_reference_commands = any(
        path.name == "hf_native_commands.jsonl" for path in log_paths
    )
    has_native_trtmc = any(path.name == "trtfb_native_commands.jsonl" for path in log_paths)
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
        return name.endswith(".log") or name.endswith("_native_commands.jsonl")

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
                discovered.append(root / relative_directory / entry.name)
                if len(discovered) > MAX_COMMAND_LOG_FILES:
                    raise ValidationError(
                        f"validation command log count exceeds {MAX_COMMAND_LOG_FILES}: {root}"
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
            f"cannot securely discover validation command artifacts in {root}: {exc}"
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
    "mean_backend_mask_iou",
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
    "mcq": "prediction_agreement_rate",
    "ocrbench_v2": "prediction_agreement_rate",
    "prompted_segmentation_parity": "mean_backend_mask_iou",
    "reranking_parity": "mean_pairwise_ordering_agreement",
    "semantic_segmentation_parity": "backend_pixel_agreement",
    "time_series_parity": "sample_agreement_rate",
    "tts_intelligibility": "prediction_agreement_rate",
}
_VALID_COUNT_REQUIRED_MODES = set(_PRIMARY_METRIC_BY_MODE) - {"continuation"}
_REQUIRED_PASS_METRICS_BY_MODE = {
    "asr_transcript": (
        "accuracy_drop_from_hf",
        "normalized_transcript_exact_agreement_rate",
        "correctness_agreement_rate",
    ),
    "image_classification_parity": ("top1_accuracy_drop_from_hf",),
    "mcq": ("accuracy_drop_from_hf",),
    "ocrbench_v2": (
        "accuracy_drop_from_hf",
        "correctness_agreement_rate",
    ),
    "prompted_segmentation_parity": (
        "hf_mean_ground_truth_iou",
        "trtfb_mean_ground_truth_iou",
        "ground_truth_iou_drop_from_hf",
    ),
    "semantic_segmentation_parity": (
        "backend_mean_iou",
        "mean_iou_drop_from_hf",
    ),
    "time_series_parity": (
        "mean_relative_l2",
        "max_relative_l2",
        "max_absolute_error",
    ),
    "tts_intelligibility": (
        "pass_rate_drop_from_hf",
        "correctness_agreement_rate",
    ),
}
_PASSED_COMPLETE_COUNT_FIELD_BY_MODE = {
    "asr_transcript": "total_count",
    "continuation": "count",
    "diffusion_image_clip_parity": "total_count",
    "diffusion_text_parity": "sample_count",
    "encoder_embedding_parity": "sample_count",
    "image_classification_parity": "sample_count",
    "mcq": "total_count",
    "ocrbench_v2": "total_count",
    "prompted_segmentation_parity": "sample_count",
    "reranking_parity": "sample_count",
    "semantic_segmentation_parity": "sample_count",
    "time_series_parity": "sample_count",
    "tts_intelligibility": "total_count",
}
_PASSED_COUNT_RATE_FIELD_BY_MODE = {
    "diffusion_image_clip_parity": "overall_pass_rate",
    "encoder_embedding_parity": "vector_pass_rate",
    "reranking_parity": "sample_pass_rate",
    "time_series_parity": "sample_agreement_rate",
}
_GATE_METRIC_ALIASES = {
    "backend_mask_iou": "mean_backend_mask_iou",
    "correctness_agreement": "correctness_agreement_rate",
    "prediction_agreement": "prediction_agreement_rate",
}
_COMPARISON_METRICS = (
    *_PRIMARY_COMPARISON_METRICS,
    "overall_pass_rate",
    "count",
    "complete_count",
    "excluded_count",
    "exact_count",
    "passed_count",
    "sample_count",
    "valid_count",
    "skipped_count",
    "total_count",
    "initial_latents_match_rate",
    "token_id_prefix_agreement",
    "normalized_transcript_exact_agreement_rate",
    "correctness_agreement_rate",
    "divergence_rate",
    "divergent_count",
    "hf_accuracy",
    "trtfb_accuracy",
    "accuracy_delta_trtfb_minus_hf",
    "accuracy_drop_from_hf",
    "pass_rate_drop_from_hf",
    "hf_top1_accuracy",
    "trtfb_top1_accuracy",
    "top1_accuracy_drop_from_hf",
    "hf_mean_iou",
    "trtfb_mean_iou",
    "backend_mean_iou",
    "mean_backend_mask_iou",
    "hf_mean_ground_truth_iou",
    "trtfb_mean_ground_truth_iou",
    "mean_iou_drop_from_hf",
    "ground_truth_iou_drop_from_hf",
    "mean_vector_cosine",
    "min_vector_cosine",
    "mean_pair_cosine_abs_delta",
    "max_pair_cosine_abs_delta",
    "mean_relative_l2",
    "max_relative_l2",
    "max_absolute_error",
)
_COUNT_COMPARISON_METRICS = {
    "count",
    "exact_count",
    "passed_count",
    "sample_count",
    "valid_count",
    "skipped_count",
    "total_count",
    "divergent_count",
}
_UNIT_INTERVAL_COMPARISON_METRICS = {
    *_PRIMARY_COMPARISON_METRICS,
    "overall_pass_rate",
    "token_id_prefix_agreement",
    "normalized_transcript_exact_agreement_rate",
    "correctness_agreement_rate",
    "divergence_rate",
    "hf_accuracy",
    "trtfb_accuracy",
    "hf_top1_accuracy",
    "trtfb_top1_accuracy",
    "hf_mean_iou",
    "trtfb_mean_iou",
    "backend_mean_iou",
    "mean_backend_mask_iou",
    "hf_mean_ground_truth_iou",
    "trtfb_mean_ground_truth_iou",
}
_SIGNED_UNIT_COMPARISON_METRICS = {
    "accuracy_delta_trtfb_minus_hf",
    "accuracy_drop_from_hf",
    "pass_rate_drop_from_hf",
    "top1_accuracy_drop_from_hf",
    "mean_iou_drop_from_hf",
    "ground_truth_iou_drop_from_hf",
    "mean_vector_cosine",
    "min_vector_cosine",
}
_NONNEGATIVE_COMPARISON_METRICS = {
    "mean_pair_cosine_abs_delta",
    "max_pair_cosine_abs_delta",
    "mean_relative_l2",
    "max_relative_l2",
    "max_absolute_error",
}
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
    if isinstance(raw_result, Mapping) and raw_result:
        normalized = dict(raw_result)
        if (
            result.get("schema_version") != "trtmc.validation-result/v2"
            and not normalized.get("status")
            and result.get("status")
        ):
            normalized["status"] = result["status"]
        return normalized
    if result.get("schema_version") == "trtmc.validation-result/v2":
        return {}
    status = str(result.get("status", "") or "")
    return {"status": status} if status else {}


def _execution_details(
    result: Mapping[str, Any],
    raw_result: Mapping[str, Any],
) -> dict[str, Any]:
    exit_code = result.get("returncode")
    raw_status = str(raw_result.get("status", "") or "")
    error_type = str(raw_result.get("error_type", "") or "")
    comparison_gate_failure = _is_comparison_gate_failure(raw_result)
    gate_failures = raw_result.get("gate_failures")
    invalid_gate_status = (
        isinstance(gate_failures, list)
        and bool(gate_failures)
        and raw_status not in {"fail", "failed"}
    )
    has_error = (
        bool(error_type and not comparison_gate_failure)
        or any(
            raw_result.get(name)
            for name in _EXECUTION_ERROR_FIELDS
            if name != "error" or not comparison_gate_failure
        )
        or invalid_gate_status
    )
    compatible_exit = exit_code == 0 or (
        exit_code == 1 and raw_status in {"fail", "failed"} and not has_error
    )
    completed = bool(raw_result) and not has_error and compatible_exit
    return {
        "status": "completed" if completed else "error",
        "exit_code": exit_code,
    }


def _comparison_metrics(raw_result: Mapping[str, Any]) -> dict[str, Any]:
    metrics = {
        name: raw_result[name] for name in _COMPARISON_METRICS if raw_result.get(name) is not None
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
        raise ValidationError(f"validation result {field} must be a list of non-empty strings")
    if any(not isinstance(item, str) or not item.strip() or "\x00" in item for item in value):
        raise ValidationError(f"validation result {field} must contain only non-empty strings")
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
        raise ValidationError("validation result reproduce.command_count must be an object")
    if kind not in counts:
        return len(commands)
    configured = counts[kind]
    if type(configured) is not int or configured < len(commands):
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
        raise ValidationError("validation result reproduce.command_logs must be an object")
    return _string_list(
        logs.get(kind, []),
        field=f"reproduce.command_logs.{kind}",
    )


def _normalize_reproduction(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError("validation result reproduce must be an object")
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
        raise ValidationError("validation result reproduce.dataset must be an object")
    dataset = dict(dataset)
    command = dataset.get("command", "")
    if not isinstance(command, str) or "\x00" in command or (command != "" and not command.strip()):
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
                f"validation result reproduce.dataset.{field} must be a non-negative integer"
            )
    representative = reproduce.get("representative", {})
    if not isinstance(representative, Mapping):
        raise ValidationError("validation result reproduce.representative must be an object")
    representative = dict(representative)
    for field in ("sample_id", "reason"):
        if field not in representative:
            continue
        representative_value = representative[field]
        if (
            not isinstance(representative_value, str)
            or "\x00" in representative_value
            or (representative_value != "" and not representative_value.strip())
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
        raise ValidationError(f"validation result {field} details must be an object")
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
    exit_code = execution.get("exit_code")
    if exit_code is not None and type(exit_code) is not int:
        raise ValidationError("validation result execution.exit_code must be an integer or null")
    if "attempt_count" in execution:
        attempt_count = execution["attempt_count"]
        if type(attempt_count) is not int or attempt_count < 1:
            raise ValidationError(
                "validation result execution.attempt_count must be a positive integer"
            )
    retry_fields = {
        "attempt_count",
        "max_attempts",
        "retry_count",
        "attempts",
    }
    present_retry_fields = retry_fields.intersection(execution)
    if present_retry_fields and present_retry_fields != retry_fields:
        raise ValidationError(
            "validation result execution retry evidence must include "
            "attempt_count, max_attempts, retry_count, and attempts"
        )
    attempts = execution.get("attempts")
    if "attempts" in execution and not isinstance(attempts, list):
        raise ValidationError("validation result execution.attempts must be a list")
    if (
        isinstance(attempts, list)
        and "attempt_count" in execution
        and execution["attempt_count"] != len(attempts)
    ):
        raise ValidationError(
            "validation result execution.attempt_count must equal the number of execution.attempts"
        )
    if present_retry_fields == retry_fields:
        attempt_count = execution["attempt_count"]
        max_attempts = execution["max_attempts"]
        retry_count = execution["retry_count"]
        assert isinstance(attempts, list)
        if type(max_attempts) is not int or max_attempts < attempt_count:
            raise ValidationError(
                "validation result execution.max_attempts must be an "
                "integer at least as large as attempt_count"
            )
        if type(retry_count) is not int or retry_count != attempt_count - 1:
            raise ValidationError(
                "validation result execution.retry_count must equal attempt_count minus one"
            )
        for expected_attempt, record in enumerate(attempts, start=1):
            if not isinstance(record, Mapping):
                raise ValidationError(
                    "validation result execution.attempts entries must be objects"
                )
            if type(record.get("attempt")) is not int or record["attempt"] != expected_attempt:
                raise ValidationError(
                    "validation result execution.attempts must use "
                    "integer, contiguous one-based attempt numbers"
                )
            execution_status = record.get("execution_status")
            if not isinstance(execution_status, str) or execution_status not in {
                "completed",
                "error",
                "not_run",
            }:
                raise ValidationError(
                    "validation result execution.attempts execution_status is invalid"
                )
            validation_status = record.get("validation_status")
            if not isinstance(validation_status, str) or validation_status not in {
                "passed",
                "failed",
                "skipped",
                "not_compared",
            }:
                raise ValidationError(
                    "validation result execution.attempts validation_status is invalid"
                )
            for field in (
                "worker_log",
                "execution_log",
                "comparison_result",
                "error_type",
                "error",
            ):
                if not isinstance(record.get(field, ""), str):
                    raise ValidationError(
                        f"validation result execution.attempts {field} must be a string"
                    )
            error_type = str(record.get("error_type", ""))
            error = str(record.get("error", ""))
            if expected_attempt < attempt_count:
                if (
                    execution_status != "error"
                    or validation_status != "failed"
                    or not (error_type or error)
                ):
                    raise ValidationError(
                        "validation result non-final retry attempts must "
                        "record an evidenced execution error and failed "
                        "validation"
                    )
            elif execution_status == "completed":
                if validation_status not in {"passed", "failed"}:
                    raise ValidationError(
                        "validation result completed final attempt must pass or fail validation"
                    )
                if error_type not in {"", "BenchmarkGateError"} or (
                    validation_status == "passed" and (error_type or error)
                ):
                    raise ValidationError(
                        "validation result completed final attempt has incompatible error evidence"
                    )
            elif execution_status == "error":
                if validation_status != "failed" or not (error_type or error):
                    raise ValidationError(
                        "validation result errored final attempt must "
                        "record evidence and failed validation"
                    )
            elif validation_status not in {
                "skipped",
                "not_compared",
            }:
                raise ValidationError(
                    "validation result not-run final attempt must be skipped or not compared"
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
        raise ValidationError("validation result comparison.metrics must be an object")
    comparison["metrics"] = dict(metrics)
    failures = comparison.get("failures", [])
    if not isinstance(failures, list):
        raise ValidationError("validation result comparison.failures must be a list")
    primary = comparison.get("primary_metric")
    if primary is not None and not isinstance(primary, Mapping):
        raise ValidationError(
            "validation result comparison.primary_metric must be an object or null"
        )
    if isinstance(primary, Mapping):
        primary_name = primary.get("name")
        if not isinstance(primary_name, str) or not primary_name:
            raise ValidationError(
                "validation result comparison.primary_metric.name must be a non-empty string"
            )
        if primary_name not in comparison["metrics"]:
            raise ValidationError(
                "validation result comparison.primary_metric.name must name "
                "an entry in comparison.metrics"
            )
        primary_value = primary.get("value")
        metric_value = comparison["metrics"][primary_name]
        if type(primary_value) is not type(metric_value) or primary_value != metric_value:
            raise ValidationError(
                "validation result comparison.primary_metric.value must "
                "exactly match comparison.metrics at primary_metric.name"
            )
        comparison["primary_metric"] = dict(primary)
    mode = comparison.get("mode", "")
    if not isinstance(mode, str):
        raise ValidationError("validation result comparison.mode must be a string")
    comparison["mode"] = mode
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


def _validated_gate_configuration(
    value: Any,
    *,
    field: str,
) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"validation result {field} must be an object")
    gates: dict[str, int | float] = {}
    for gate, required in value.items():
        if not isinstance(gate, str) or not gate:
            raise ValidationError(f"validation result {field} names must be non-empty strings")
        finite = False
        if isinstance(required, (int, float)) and not isinstance(required, bool):
            try:
                finite = math.isfinite(required)
            except OverflowError:
                finite = False
        if not finite:
            raise ValidationError(f"validation result {field}.{gate} must be a finite number")
        gates[gate] = required
    return gates


def _validate_passed_complete_count_evidence(
    raw_result: Mapping[str, Any],
    *,
    mode: str,
    dataset_evidence: Mapping[str, Any] | None = None,
) -> None:
    count_field = _PASSED_COMPLETE_COUNT_FIELD_BY_MODE.get(mode)
    if count_field is None:
        return
    evidence_label = "diffusion" if mode == "diffusion_image_clip_parity" else mode
    valid_count = raw_result.get("valid_count")
    canonical_complete_count = raw_result.get("complete_count")
    complete_count = raw_result.get(count_field)
    if (
        type(valid_count) is not int
        or valid_count <= 0
        or type(canonical_complete_count) is not int
        or canonical_complete_count <= 0
        or type(complete_count) is not int
        or complete_count <= 0
    ):
        raise ValidationError(
            f"passed {evidence_label} comparison must include positive integer "
            f"valid_count, complete_count, and "
            f"{count_field} evidence"
        )
    if canonical_complete_count != complete_count:
        raise ValidationError(
            f"passed {evidence_label} comparison requires complete_count to equal {count_field}"
        )
    skipped_count = raw_result.get("skipped_count")
    excluded_count = raw_result.get("excluded_count")
    if type(skipped_count) is not int or type(excluded_count) is not int:
        raise ValidationError(
            f"passed {evidence_label} comparison must include integer "
            "skipped_count and excluded_count evidence"
        )
    if skipped_count != 0 or excluded_count != 0:
        raise ValidationError(
            f"passed {evidence_label} comparison requires zero skipped and excluded samples"
        )
    if valid_count + skipped_count + excluded_count != canonical_complete_count:
        raise ValidationError(
            f"passed {evidence_label} comparison requires valid_count + "
            "skipped_count + excluded_count to equal complete_count"
        )
    rate_field = _PASSED_COUNT_RATE_FIELD_BY_MODE.get(mode)
    if rate_field is not None:
        passed_count = raw_result.get("passed_count")
        rate = raw_result.get(rate_field)
        if (
            type(passed_count) is not int
            or not 0 <= passed_count <= valid_count
            or not isinstance(rate, (int, float))
            or isinstance(rate, bool)
            or not math.isclose(
                float(rate),
                passed_count / valid_count if valid_count else 0.0,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValidationError(
                f"passed {evidence_label} comparison requires consistent "
                f"passed_count and {rate_field}"
            )
    alternate_count_field = "sample_count" if count_field == "total_count" else "total_count"
    if alternate_count_field in raw_result and raw_result[alternate_count_field] != complete_count:
        raise ValidationError(
            f"passed {evidence_label} comparison has conflicting "
            f"{count_field} and "
            f"{alternate_count_field} evidence"
        )
    if dataset_evidence is None or not dataset_evidence:
        return
    sample_limit = dataset_evidence.get("sample_limit")
    prepared_count = dataset_evidence.get("prepared_input_count")
    if (
        type(sample_limit) is not int
        or sample_limit < 0
        or type(prepared_count) is not int
        or prepared_count <= 0
    ):
        raise ValidationError(
            f"passed {evidence_label} comparison requires non-negative "
            "sample_limit and positive prepared_input_count evidence"
        )
    if prepared_count != canonical_complete_count:
        raise ValidationError(
            f"passed {evidence_label} comparison requires "
            "prepared_input_count to equal complete_count"
        )
    if sample_limit > 0 and sample_limit != canonical_complete_count:
        raise ValidationError(
            f"passed {evidence_label} comparison did not complete the "
            f"requested sample_limit ({sample_limit})"
        )
    for raw_field, expected in (
        ("requested_sample_limit", sample_limit),
        ("prepared_input_count", prepared_count),
    ):
        if raw_field in raw_result and raw_result[raw_field] != expected:
            raise ValidationError(
                f"passed {evidence_label} comparison raw_result."
                f"{raw_field} conflicts with reproduce.dataset evidence"
            )


def _validate_raw_metric_relationships(
    raw_result: Mapping[str, Any],
    *,
    expected_gates: Mapping[str, Any] | None = None,
    dataset_evidence: Mapping[str, Any] | None = None,
) -> None:
    raw_status = str(raw_result.get("status", "") or "")
    passed = raw_status in {"pass", "passed"}
    mode = str(raw_result.get("mode", "") or "")
    expected_primary = _PRIMARY_METRIC_BY_MODE.get(mode)
    if passed and expected_primary is not None:
        primary_value = raw_result.get(expected_primary)
        if (
            not isinstance(primary_value, (int, float))
            or isinstance(primary_value, bool)
            or not math.isfinite(primary_value)
        ):
            raise ValidationError(
                "passed validation result for mode "
                f"{mode!r} must include finite raw_result."
                f"{expected_primary}"
            )
    if passed:
        required = _REQUIRED_PASS_METRICS_BY_MODE.get(mode, ())
        missing = [
            name
            for name in required
            if (
                not isinstance(raw_result.get(name), (int, float))
                or isinstance(raw_result.get(name), bool)
                or not math.isfinite(raw_result[name])
            )
        ]
        if missing:
            raise ValidationError(
                f"passed {mode or '<missing>'} comparison is missing raw "
                "metric evidence: " + ", ".join(missing)
            )
    if passed:
        _validate_passed_complete_count_evidence(
            raw_result,
            mode=mode,
            dataset_evidence=dataset_evidence,
        )
    supported_mode = mode in _PRIMARY_METRIC_BY_MODE
    if passed and supported_mode and mode != "continuation":
        gates = raw_result.get("gates")
        if not isinstance(gates, Mapping) or not gates:
            raise ValidationError(
                f"passed {mode} comparison must include its non-empty raw gate configuration"
            )
    else:
        gates = raw_result.get("gates", {})
        if gates is None:
            gates = {}
    if passed and supported_mode:
        actual_gates = _validated_gate_configuration(
            gates,
            field="raw_result.gates",
        )
        if expected_gates is not None:
            authoritative_gates = _validated_gate_configuration(
                expected_gates,
                field="expected_gates",
            )
            if actual_gates != authoritative_gates:
                unknown = sorted(set(actual_gates) - set(authoritative_gates))
                missing = sorted(set(authoritative_gates) - set(actual_gates))
                changed = sorted(
                    gate
                    for gate in set(actual_gates).intersection(authoritative_gates)
                    if actual_gates[gate] != authoritative_gates[gate]
                )
                details = []
                if unknown:
                    details.append("unknown gates: " + ", ".join(unknown))
                if missing:
                    details.append("missing gates: " + ", ".join(missing))
                if changed:
                    details.append("changed thresholds: " + ", ".join(changed))
                raise ValidationError(
                    f"passed {mode} raw_result.gates must exactly match "
                    "the authoritative workload configuration"
                    + (": " + "; ".join(details) if details else "")
                )
    if passed and supported_mode and gates:
        assert isinstance(gates, Mapping)
        violations: list[str] = []
        for gate, required in gates.items():
            if gate.startswith("min_"):
                metric = gate[len("min_") :]
                operator = "min"
            elif gate.startswith("max_"):
                metric = gate[len("max_") :]
                operator = "max"
            else:
                metric = gate
                operator = "min"
            metric = _GATE_METRIC_ALIASES.get(metric, metric)
            if gate == "require_matching_initial_latents":
                metric = "initial_latents_match_rate"
            actual = raw_result.get(metric)
            if actual is None:
                actual = raw_result.get(gate)
            if actual is None:
                nested = raw_result.get("metrics")
                summary = nested.get(metric) if isinstance(nested, Mapping) else None
                if isinstance(summary, Mapping):
                    actual = summary.get("min" if operator == "min" else "max")
            if (
                not isinstance(actual, (int, float))
                or isinstance(actual, bool)
                or not math.isfinite(actual)
            ):
                violations.append(f"{gate} metric {metric} is unavailable")
            elif (operator == "min" and actual < required) or (
                operator == "max" and actual > required
            ):
                violations.append(f"{gate} actual={actual} required={required}")
        if violations:
            raise ValidationError(
                f"passed {mode} comparison violates raw_result.gates: " + "; ".join(violations)
            )

    if mode == "diffusion_image_clip_parity" and passed:
        required = (
            "overall_pass_rate",
            "passed_count",
            "valid_count",
            "skipped_count",
        )
        missing = [name for name in required if name not in raw_result]
        if missing:
            raise ValidationError(
                "passed diffusion comparison is missing raw count evidence: " + ", ".join(missing)
            )
        if (
            raw_result["valid_count"] <= 0
            or raw_result["passed_count"] != raw_result["valid_count"]
            or raw_result["skipped_count"] != 0
        ):
            raise ValidationError(
                "passed diffusion comparison requires a non-empty valid "
                "set, every valid sample passed, and zero skipped samples"
            )

    if mode != "continuation":
        return
    continuation_fields = (
        "count",
        "exact_count",
        "divergent_count",
        "exact_match_rate",
        "divergence_rate",
    )
    if passed:
        missing = [name for name in continuation_fields if name not in raw_result]
        if missing:
            raise ValidationError(
                "passed continuation comparison is missing raw sample "
                "evidence: " + ", ".join(missing)
            )
        if raw_result["count"] <= 0:
            raise ValidationError("passed continuation comparison requires at least one sample")
    if all(name in raw_result for name in ("count", "exact_count", "divergent_count")):
        count = raw_result["count"]
        exact_count = raw_result["exact_count"]
        divergent_count = raw_result["divergent_count"]
        if exact_count + divergent_count != count:
            raise ValidationError("continuation exact_count plus divergent_count must equal count")
        if "exact_match_rate" in raw_result:
            expected_exact_rate = exact_count / count if count else 0.0
            if not math.isclose(
                raw_result["exact_match_rate"],
                expected_exact_rate,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValidationError("continuation exact_match_rate conflicts with sample counts")
        if "divergence_rate" in raw_result:
            expected_divergence_rate = divergent_count / count if count else 0.0
            if not math.isclose(
                raw_result["divergence_rate"],
                expected_divergence_rate,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValidationError("continuation divergence_rate conflicts with sample counts")
    divergent_count = raw_result.get("divergent_count")
    divergence_rate = raw_result.get("divergence_rate")
    exact_match_rate = raw_result.get("exact_match_rate")
    if (
        type(divergent_count) is int
        and divergent_count > 0
        and (divergence_rate == 0 or exact_match_rate == 1)
    ):
        raise ValidationError("continuation divergence evidence conflicts with exact-match metrics")
    if passed:
        evaluation_policy = raw_result.get("evaluation_policy")
        if evaluation_policy != "threshold_gated":
            raise ValidationError(
                "diagnostic-only continuation evidence cannot be published "
                "as passed reference consistency"
            )
        if not isinstance(gates, Mapping) or not gates:
            raise ValidationError(
                "passed threshold-gated continuation comparison must include "
                "its non-empty raw gate configuration"
            )


def _normalize_result(
    result: Mapping[str, Any],
    *,
    expected_gates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = dict(result)
    if "run_id" in normalized and (
        not isinstance(normalized["run_id"], str)
        or not normalized["run_id"]
        or "\x00" in normalized["run_id"]
    ):
        raise ValidationError("validation result run_id must be a non-empty, NUL-free string")
    normalized_reproduction = _normalize_reproduction(normalized.get("reproduce", {}))
    if "status" in normalized and not isinstance(
        normalized["status"],
        str,
    ):
        raise ValidationError("validation result legacy status must be a string")
    if (
        "returncode" in normalized
        and normalized["returncode"] is not None
        and type(normalized["returncode"]) is not int
    ):
        raise ValidationError("validation result returncode must be an integer or null")
    if "schema_version" in normalized:
        schema_version = normalized["schema_version"]
        if not isinstance(schema_version, str) or schema_version not in {
            "trtmc.validation-result/v1",
            "trtmc.validation-result/v2",
        }:
            raise ValidationError(
                "validation result schema_version must be one of "
                "trtmc.validation-result/v1 or "
                "trtmc.validation-result/v2"
            )
    declared_raw_result = normalized.get("raw_result")
    if (
        "raw_result" in normalized
        and declared_raw_result is not None
        and not isinstance(declared_raw_result, Mapping)
    ):
        raise ValidationError("validation result raw_result must be an object or null")
    if (
        isinstance(declared_raw_result, Mapping)
        and "status" in declared_raw_result
        and isinstance(declared_raw_result["status"], str)
        and isinstance(normalized.get("status"), str)
    ):
        aliases = {
            "pass": "passed",
            "fail": "failed",
            "skip": "skipped",
        }
        raw_declared_status = aliases.get(
            declared_raw_result["status"],
            declared_raw_result["status"],
        )
        outer_status = aliases.get(
            normalized["status"],
            normalized["status"],
        )
        if raw_declared_status != outer_status:
            raise ValidationError(
                "validation result legacy status conflicts with raw_result.status"
            )
    if normalized.get("schema_version") == "trtmc.validation-result/v2":
        v2_without_comparison_evidence = (
            bool(normalized.get("not_compared_reason"))
            or normalized.get("executor") == "e2e"
            or isinstance(
                normalized.get("raw_results"),
                list,
            )
        )
        if not v2_without_comparison_evidence and (
            not isinstance(declared_raw_result, Mapping)
            or not declared_raw_result
            or not isinstance(declared_raw_result.get("status"), str)
            or not declared_raw_result["status"]
        ):
            raise ValidationError(
                "validation result v2 runnable result must include a "
                "non-empty raw_result with an explicit status"
            )
    if "not_compared_reason" in normalized and not isinstance(
        normalized["not_compared_reason"],
        str,
    ):
        raise ValidationError("validation result not_compared_reason must be a string")
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
    without_comparison_evidence = (
        bool(normalized.get("not_compared_reason"))
        or normalized.get("executor") == "e2e"
        or isinstance(
            normalized.get("raw_results"),
            list,
        )
    )
    if not raw_result and not without_comparison_evidence:
        raise ValidationError(
            "validation runnable result must include raw comparison "
            "evidence or a legacy outer status"
        )
    raw_status_value = raw_result.get("status")
    if "status" in raw_result and not isinstance(raw_status_value, str):
        raise ValidationError("validation result raw_result.status must be a string")
    if raw_result and raw_status_value not in {
        "pass",
        "passed",
        "fail",
        "failed",
        "skip",
        "skipped",
    }:
        raise ValidationError(
            "validation result raw_result.status must be one of pass, "
            "passed, fail, failed, skip, or skipped"
        )
    for field in ("model", "suite", "workload"):
        if field in raw_result and not isinstance(
            raw_result[field],
            str,
        ):
            raise ValidationError(f"validation result raw_result.{field} must be a string")
    if "mode" in raw_result and not isinstance(
        raw_result["mode"],
        str,
    ):
        raise ValidationError("validation result raw_result.mode must be a string")
    for field in (
        "error_type",
        "error",
        "exception",
        "traceback",
        "failure_class",
    ):
        if field in raw_result and not isinstance(
            raw_result[field],
            str,
        ):
            raise ValidationError(f"validation result raw_result.{field} must be a string")
    if "gate_failures" in raw_result and not isinstance(raw_result["gate_failures"], list):
        raise ValidationError("validation result raw_result.gate_failures must be a list")
    if "gates" in raw_result and not isinstance(raw_result["gates"], Mapping):
        raise ValidationError("validation result raw_result.gates must be an object")
    if "metrics" in raw_result and not isinstance(raw_result["metrics"], Mapping):
        raise ValidationError("validation result raw_result.metrics must be an object")
    for field in _COMPARISON_METRICS:
        if field not in raw_result:
            continue
        metric = raw_result[field]
        if metric is None and raw_status_value in {"fail", "failed"}:
            continue
        if (
            not isinstance(metric, (int, float))
            or isinstance(metric, bool)
            or not math.isfinite(metric)
        ):
            raise ValidationError(f"validation result raw_result.{field} must be a finite number")
        if field in _COUNT_COMPARISON_METRICS and (type(metric) is not int or metric < 0):
            raise ValidationError(
                f"validation result raw_result.{field} must be a non-negative integer"
            )
        if field in _UNIT_INTERVAL_COMPARISON_METRICS and not 0.0 <= metric <= 1.0:
            raise ValidationError(f"validation result raw_result.{field} must be in [0, 1]")
        if field in _SIGNED_UNIT_COMPARISON_METRICS and not -1.0 <= metric <= 1.0:
            raise ValidationError(f"validation result raw_result.{field} must be in [-1, 1]")
        if field in _NONNEGATIVE_COMPARISON_METRICS and metric < 0:
            raise ValidationError(f"validation result raw_result.{field} must be non-negative")
    nested_metrics = raw_result.get("metrics")
    if isinstance(nested_metrics, Mapping):
        for name, summary in nested_metrics.items():
            if not isinstance(name, str) or not name:
                raise ValidationError(
                    "validation result raw_result.metrics names must be non-empty strings"
                )
            if not isinstance(summary, Mapping):
                raise ValidationError(
                    f"validation result raw_result.metrics.{name} must be an object"
                )
            mean = summary.get("mean")
            if (
                not isinstance(mean, (int, float))
                or isinstance(mean, bool)
                or not math.isfinite(mean)
            ):
                raise ValidationError(
                    f"validation result raw_result.metrics.{name}.mean must be a finite number"
                )
            minimum = summary.get("min", mean)
            maximum = summary.get("max", mean)
            for field, value in (
                ("min", minimum),
                ("max", maximum),
            ):
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                ):
                    raise ValidationError(
                        "validation result raw_result.metrics."
                        f"{name}.{field} must be a finite number"
                    )
            if not minimum <= mean <= maximum:
                raise ValidationError(
                    f"validation result raw_result.metrics.{name} must satisfy min <= mean <= max"
                )
            if "count" in summary and (type(summary["count"]) is not int or summary["count"] < 0):
                raise ValidationError(
                    "validation result raw_result.metrics."
                    f"{name}.count must be a non-negative integer"
                )
            for field in ("gated_count", "passed_count"):
                if field in summary and (type(summary[field]) is not int or summary[field] < 0):
                    raise ValidationError(
                        "validation result raw_result.metrics."
                        f"{name}.{field} must be a non-negative integer"
                    )
            count = summary.get("count")
            gated_count = summary.get("gated_count")
            nested_passed_count = summary.get("passed_count")
            if type(count) is int and type(gated_count) is int and gated_count > count:
                raise ValidationError(
                    f"validation result raw_result.metrics.{name}.gated_count cannot exceed count"
                )
            if (
                type(gated_count) is int
                and type(nested_passed_count) is int
                and nested_passed_count > gated_count
            ):
                raise ValidationError(
                    "validation result raw_result.metrics."
                    f"{name}.passed_count cannot exceed gated_count"
                )
            if (
                raw_status_value in {"pass", "passed"}
                and type(gated_count) is int
                and gated_count > 0
                and type(nested_passed_count) is int
                and nested_passed_count != gated_count
            ):
                raise ValidationError(
                    "passed validation result raw_result.metrics."
                    f"{name} must pass every gated sample"
                )
            if name in raw_result and raw_result[name] != mean:
                raise ValidationError(
                    f"validation result raw metric conflicts with nested mean for {name}"
                )
    passed_count = raw_result.get("passed_count")
    valid_count = raw_result.get("valid_count")
    if type(passed_count) is int and type(valid_count) is int and passed_count > valid_count:
        raise ValidationError("validation result raw_result.passed_count cannot exceed valid_count")
    overall_pass_rate = raw_result.get("overall_pass_rate")
    if (
        type(passed_count) is int
        and type(valid_count) is int
        and isinstance(overall_pass_rate, (int, float))
        and not isinstance(overall_pass_rate, bool)
    ):
        expected_rate = passed_count / valid_count if valid_count else 0.0
        if not math.isclose(
            overall_pass_rate,
            expected_rate,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValidationError(
                "validation result raw_result.overall_pass_rate "
                "conflicts with passed_count and valid_count"
            )
    for reference_name, candidate_name, delta_name, factor in (
        (
            "hf_accuracy",
            "trtfb_accuracy",
            "accuracy_delta_trtfb_minus_hf",
            1,
        ),
        (
            "hf_accuracy",
            "trtfb_accuracy",
            "accuracy_drop_from_hf",
            -1,
        ),
        (
            "hf_accuracy",
            "trtfb_accuracy",
            "pass_rate_drop_from_hf",
            -1,
        ),
        (
            "hf_top1_accuracy",
            "trtfb_top1_accuracy",
            "top1_accuracy_drop_from_hf",
            -1,
        ),
        (
            "hf_mean_iou",
            "trtfb_mean_iou",
            "mean_iou_drop_from_hf",
            -1,
        ),
        (
            "hf_mean_ground_truth_iou",
            "trtfb_mean_ground_truth_iou",
            "ground_truth_iou_drop_from_hf",
            -1,
        ),
    ):
        if all(
            name in raw_result
            for name in (
                reference_name,
                candidate_name,
                delta_name,
            )
        ):
            expected_delta = factor * (raw_result[candidate_name] - raw_result[reference_name])
            if not math.isclose(
                raw_result[delta_name],
                expected_delta,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValidationError(
                    "validation result raw_result."
                    f"{delta_name} conflicts with {reference_name} and "
                    f"{candidate_name}"
                )
    _validate_raw_metric_relationships(
        raw_result,
        expected_gates=expected_gates,
        dataset_evidence=normalized_reproduction.get("dataset", {}),
    )
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
    _validate_result_evidence_consistency(
        normalized,
        raw_result=raw_result,
        execution=execution,
        comparison=comparison,
        validation=validation,
    )
    reference_environment = normalized.get("reference_environment", [])
    if reference_environment is None:
        reference_environment = []
    if not isinstance(reference_environment, list) or any(
        not isinstance(item, Mapping) for item in reference_environment
    ):
        raise ValidationError("validation result reference_environment must be a list of objects")
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
            "reproduce": normalized_reproduction,
            "reference_environment": [dict(item) for item in reference_environment],
        }
    )
    if precision_contract:
        normalized["precision_contract"] = precision_contract
    else:
        normalized.pop("precision_contract", None)
    if raw_result:
        normalized["raw_result"] = dict(raw_result)
    normalized.pop("returncode", None)
    normalized.pop("status", None)
    _validate_result_status_consistency(normalized)
    return normalized


def _validate_result_evidence_consistency(
    result: Mapping[str, Any],
    *,
    raw_result: Mapping[str, Any],
    execution: Mapping[str, Any],
    comparison: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> None:
    not_compared_reason = str(result.get("not_compared_reason", "") or "")
    if not_compared_reason:
        if raw_result and result.get("executor") != "e2e":
            raise ValidationError(
                "validation result not_compared_reason cannot override raw comparison evidence"
            )
        return
    if not raw_result:
        return
    expected_model = result.get("model")
    expected_workload = result.get("workload")
    for field, expected in (
        ("model", expected_model),
        ("suite", expected_workload),
        ("workload", expected_workload),
    ):
        if field in raw_result and raw_result[field] != expected:
            raise ValidationError(
                f"validation result raw_result.{field} conflicts with the canonical binding"
            )
    legacy_exit_code = result.get("returncode")
    if legacy_exit_code is not None and type(legacy_exit_code) is not int:
        raise ValidationError("validation result returncode must be an integer or null")
    canonical_exit_code = execution.get("exit_code")
    if (
        "returncode" in result
        and isinstance(result.get("execution"), Mapping)
        and "exit_code" in result["execution"]
        and legacy_exit_code != canonical_exit_code
    ):
        raise ValidationError("validation result returncode conflicts with execution.exit_code")
    exit_code = canonical_exit_code if canonical_exit_code is not None else legacy_exit_code
    expected_execution = _execution_details(
        {"returncode": exit_code},
        raw_result,
    )
    if expected_execution["status"] == "completed":
        if raw_result.get("model") != result.get("model") or raw_result.get("suite") != result.get(
            "workload"
        ):
            raise ValidationError(
                "completed validation result must include exact raw "
                "model and suite binding evidence"
            )
    expected_comparison = _comparison_details(
        raw_result,
        expected_execution,
    )
    expected_validation = _validation_details(
        expected_execution,
        expected_comparison,
    )
    if expected_validation["status"] == "passed":
        mode = raw_result.get("mode")
        if not isinstance(mode, str) or mode not in _PRIMARY_METRIC_BY_MODE:
            raise ValidationError("passed validation result must name a supported raw_result.mode")
        primary_metric = _PRIMARY_METRIC_BY_MODE[mode]
        primary_value = raw_result.get(primary_metric)
        if (
            not isinstance(primary_value, (int, float))
            or isinstance(primary_value, bool)
            or not math.isfinite(primary_value)
        ):
            raise ValidationError(
                "passed validation result for mode "
                f"{mode!r} must include finite raw_result."
                f"{primary_metric}"
            )
        if mode in _VALID_COUNT_REQUIRED_MODES and (
            type(raw_result.get("valid_count")) is not int or raw_result["valid_count"] <= 0
        ):
            raise ValidationError(
                "passed validation result for mode "
                f"{mode!r} requires a positive raw_result.valid_count"
            )
    attempts = execution.get("attempts")
    if isinstance(attempts, list) and attempts:
        final_attempt = attempts[-1]
        if final_attempt.get("execution_status") != execution.get("status") or final_attempt.get(
            "validation_status"
        ) != validation.get("status"):
            raise ValidationError(
                "validation result final retry attempt must match the "
                "final execution and validation statuses"
            )
        for field in ("error_type", "error"):
            if final_attempt.get(field, "") != raw_result.get(field, ""):
                raise ValidationError(
                    f"validation result final retry attempt {field} must match raw_result.{field}"
                )
    evidence_fields = (
        "mode",
        "primary_metric",
        "metrics",
        "failures",
    )
    mismatched_fields = [
        field
        for field in evidence_fields
        if comparison.get(field) != expected_comparison.get(field)
    ]
    if mismatched_fields:
        raise ValidationError(
            "validation result canonical comparison evidence conflicts "
            "with raw evidence for fields " + ", ".join(mismatched_fields)
        )
    actual = (
        execution["status"],
        comparison["status"],
        validation["status"],
    )
    expected = (
        expected_execution["status"],
        expected_comparison["status"],
        expected_validation["status"],
    )
    if actual != expected:
        raise ValidationError(
            "validation result canonical statuses conflict with raw "
            f"evidence: got {actual}, expected {expected}"
        )


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
        expected_comparison = comparison_status if execution_status == "completed" else "not_run"
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


def _not_compared_result(
    binding: Binding,
    *,
    run_id: str = "",
) -> dict[str, Any]:
    payload = {
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
    if run_id:
        payload["run_id"] = run_id
    return _normalize_result(payload)


def _write_not_compared_case(
    binding: Binding,
    output: Path,
    *,
    run_id: str = "",
) -> tuple[dict[str, Any], Path]:
    case_dir = _prepare_case_directory(output, binding)
    comparison = case_dir / "comparison.json"
    expected_target = _regular_artifact_version(
        comparison,
        missing_ok=True,
    )
    expected_run_id = run_id or _current_run_id(output)
    result = _not_compared_result(
        binding,
        run_id=expected_run_id,
    )
    _publish_validation_result(
        output,
        comparison,
        result,
        expected_run_id=expected_run_id,
        expected_target=expected_target,
    )
    return result, comparison


def _comparison_result(
    binding: Binding,
    *,
    case_dir: Path,
    returncode: int,
    reference_environment: EnvironmentSelection,
    dataset_command: str,
    sample_limit: int = 0,
    expected_gates: Mapping[str, Any] | None = None,
    run_id: str = "",
) -> dict[str, Any]:
    workload = _required_workload(binding)
    summary_path = _comparison_summary_path(binding, case_dir)

    def comparison_process_error(message: str) -> dict[str, Any]:
        return {
            "status": "failed",
            "error_type": "ComparisonProcessError",
            "error": message,
        }

    raw_result: dict[str, Any] = {}
    try:
        summary = _read_json_artifact(summary_path, missing_ok=True)
    except ValidationError as exc:
        raw_result = comparison_process_error(
            f"comparison wrote an invalid summary to {summary_path}: {exc}"
        )
        summary = None
    if summary is not None:
        if not isinstance(summary, Mapping):
            raw_result = comparison_process_error(
                f"comparison summary must contain an object: {summary_path}"
            )
        else:
            candidates = summary.get("results", [])
            if not isinstance(candidates, list):
                raw_result = comparison_process_error(
                    f"comparison summary results must be a list: {summary_path}"
                )
            elif (
                len(candidates) == 1
                and isinstance(candidates[0], Mapping)
                and candidates[0].get("model") == binding.model
            ):
                raw_result = dict(candidates[0])
            elif candidates:
                raw_result = comparison_process_error(
                    "comparison must write exactly one result for requested "
                    f"model {binding.model!r} to {summary_path}"
                )
    if not raw_result:
        raw_result = comparison_process_error(
            f"comparison exited with code {returncode} without writing "
            f"a model result to {summary_path}"
        )
    raw_status = str(raw_result.get("status", "") or "")
    if raw_status not in {
        "pass",
        "passed",
        "fail",
        "failed",
    }:
        raw_result = comparison_process_error(
            f"comparison wrote invalid status "
            f"{raw_status or '<missing>'!r} for requested model "
            f"{binding.model!r}"
        )
        raw_status = "failed"
    if returncode not in {0, 1} or (returncode == 1 and raw_status not in {"fail", "failed"}):
        raw_result = comparison_process_error(
            f"comparison exited with code {returncode} while reporting "
            f"status {raw_status or '<missing>'!r} for requested model "
            f"{binding.model!r}"
        )
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
        "inline_limit": (trtmc_disagreements.INLINE_DISAGREEMENT_LIMIT),
        "reference_vanilla_available": False,
        "trtmc_vanilla_available": False,
    }
    result_payload = {
        "schema_version": "trtmc.validation-result/v2",
        "model": binding.model,
        "workload": binding.workload,
        "executor": "trtmc_compare",
        "status": status,
        "returncode": returncode,
        "reference_environment": [
            {"name": name, "python": path} for name, path in reference_environment.names_and_paths
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
    if run_id:
        result_payload["run_id"] = run_id
    try:
        return _normalize_result(
            result_payload,
            expected_gates=expected_gates,
        )
    except ValidationError as exc:
        if raw_result.get("error_type") == "ComparisonProcessError":
            raise
        result_payload["status"] = "failed"
        result_payload["raw_result"] = comparison_process_error(
            f"comparison wrote malformed model evidence for {binding.model!r}: {exc}"
        )
        return _normalize_result(
            result_payload,
            expected_gates=expected_gates,
        )


def _comparison_summary_path(binding: Binding, case_dir: Path) -> Path:
    return case_dir / "validation" / _required_workload(binding) / "eval_summary.json"


def run_binding(
    binding: Binding,
    *,
    arguments: argparse.Namespace,
    task_models: Mapping[str, dict[str, Any]],
    suites: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    workload = _required_workload(binding)
    run_id = _validated_result_run_id(arguments)
    case_dir = _prepare_case_directory(Path(arguments.output), binding)
    comparison = case_dir / "comparison.json"
    expected_target = _regular_artifact_version(
        comparison,
        missing_ok=True,
    )
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
    dataset_command = shlex.join(_public_worker_command([sys.executable, *sys.argv]))

    suite = suites[workload]
    resolved_suite = task_eval.resolve_suite_for_model(
        suite,
        task_models[binding.model],
    )
    expected_gates = resolved_suite.get("gates", {})
    if not isinstance(expected_gates, Mapping):
        raise ValidationError(f"resolved gates for {binding.model}/{workload} must be an object")
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
    summary_path = _comparison_summary_path(binding, case_dir)
    _ensure_real_directory(
        summary_path.parent,
        description="comparison summary parent",
    )
    _atomic_write_json(summary_path, {"results": []})
    returncode = _run_subprocess(command, case_dir / "execution.log", process_env)
    if _validated_result_run_id(arguments) != run_id:
        raise ValidationError("validation run changed while this model worker was running")
    result = _comparison_result(
        binding,
        case_dir=case_dir,
        returncode=returncode,
        reference_environment=environment,
        dataset_command=dataset_command,
        sample_limit=int(arguments.limit or 0),
        expected_gates=expected_gates,
        run_id=run_id,
    )

    _publish_validation_result(
        Path(arguments.output),
        comparison,
        result,
        expected_run_id=run_id,
        expected_target=expected_target,
    )
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
    with _validation_output_publication_lock(output):
        metadata = {
            "schema_version": "trtmc.validation-run/v1",
            "run_id": secrets.token_hex(16),
            "source_revision": _source_revision(),
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES",
                "",
            ),
            "command": shlex.join(sys.argv),
            "started_at": _utc_now().isoformat(),
            "finished_at": None,
            "duration_seconds": None,
            "status": "running",
        }
        path = output / "run.json"
        run_update: _FileUpdate | None = None
        removals: list[_FileRemoval] = []
        try:
            run_update = _prepare_file_update(
                path,
                _json_artifact_payload(path, metadata),
            )
            for report_name in ("report.json", "report.html"):
                removal = _prepare_file_removal(output / report_name)
                if removal is not None:
                    removals.append(removal)
            try:
                for removal in removals:
                    _commit_file_removal(removal)
                _commit_file_update(run_update)
                _verify_committed_file_update(run_update)
                for removal in removals:
                    _verify_committed_file_removal(removal)
            except BaseException as exc:
                rollback_errors: list[str] = []
                try:
                    _rollback_file_update(run_update)
                except (OSError, ValidationError) as rollback_exc:
                    rollback_errors.append(f"{type(rollback_exc).__name__}: {rollback_exc}")
                for removal in reversed(removals):
                    try:
                        _rollback_file_removal(removal)
                    except (OSError, ValidationError) as rollback_exc:
                        rollback_errors.append(f"{type(rollback_exc).__name__}: {rollback_exc}")
                if rollback_errors:
                    raise ValidationError(
                        "validation run metadata transaction failed and "
                        "rollback was incomplete: " + " | ".join(rollback_errors)
                    ) from exc
                raise
            return path
        finally:
            active_error = sys.exc_info()[1]
            cleanup_errors: list[str] = []
            if run_update is not None:
                cleanup_errors.extend(_finalize_file_update(run_update))
            for removal in removals:
                cleanup_errors.extend(_finalize_file_removal(removal))
            if cleanup_errors:
                cleanup_message = (
                    "validation run metadata transaction cleanup incomplete: "
                    + " | ".join(cleanup_errors)
                )
                if active_error is not None:
                    if hasattr(active_error, "add_note"):
                        active_error.add_note(cleanup_message)
                    else:
                        print(cleanup_message, file=sys.stderr)
                else:
                    raise ValidationError(cleanup_message)


def finalize_run_metadata(
    output: Path,
    *,
    error: str = "",
    expected_run_id: str = "",
) -> Path:
    with _validation_output_publication_lock(output):
        path = output / "run.json"
        loaded, expected_target = _read_json_artifact(
            path,
            include_version=True,
        )
        if not isinstance(loaded, Mapping):
            raise ValidationError(f"validation run metadata must be an object: {path}")
        metadata = dict(loaded)
        if expected_run_id and metadata.get("run_id") != expected_run_id:
            raise ValidationError("validation run changed before its metadata could be finalized")
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
        _commit_versioned_file_update(
            path,
            _json_artifact_payload(path, metadata),
            expected_target=expected_target,
            description="validation run metadata finalization",
        )
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
            f"validation run metadata must provide schema_version and status together: {path}"
        )
    if has_schema and run.get("schema_version") != "trtmc.validation-run/v1":
        raise ValidationError(f"validation run metadata has an unsupported schema_version: {path}")
    run_id = run.get("run_id")
    if run_id is not None and (not isinstance(run_id, str) or not run_id or "\x00" in run_id):
        raise ValidationError(f"validation run metadata has an invalid run_id: {path}")
    status = run.get("status") if has_status else None
    if has_status and (
        not isinstance(status, str) or status not in {"running", "completed", "failed"}
    ):
        raise ValidationError(f"validation run metadata has an invalid status: {path}")
    started_at = run.get("started_at")
    if not isinstance(started_at, str) or not started_at:
        raise ValidationError(f"validation run metadata must include started_at: {path}")
    started_timestamp: datetime | None = None
    if has_schema:
        try:
            started_timestamp = datetime.fromisoformat(
                started_at[:-1] + "+00:00" if started_at.endswith("Z") else started_at
            )
        except ValueError as exc:
            raise ValidationError(
                f"validation run metadata started_at is not ISO-8601: {path}"
            ) from exc
        if started_timestamp.tzinfo is None:
            raise ValidationError(
                f"validation run metadata started_at must include a timezone: {path}"
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
                f"running validation metadata cannot be finalized or carry an error: {path}"
            )
        return status
    if not isinstance(finished_at, str) or not finished_at:
        raise ValidationError(f"finalized validation metadata must include finished_at: {path}")
    if has_schema:
        try:
            finished_timestamp = datetime.fromisoformat(
                finished_at[:-1] + "+00:00" if finished_at.endswith("Z") else finished_at
            )
        except ValueError as exc:
            raise ValidationError(
                f"validation run metadata finished_at is not ISO-8601: {path}"
            ) from exc
        if finished_timestamp.tzinfo is None:
            raise ValidationError(
                f"validation run metadata finished_at must include a timezone: {path}"
            )
        assert started_timestamp is not None
        if finished_timestamp < started_timestamp:
            raise ValidationError(
                f"validation run metadata finished_at precedes started_at: {path}"
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
            raise ValidationError(f"failed validation metadata must include an error: {path}")
    elif error:
        raise ValidationError(f"completed validation metadata cannot carry an error: {path}")
    return status


def _current_run_id(output: Path) -> str:
    path = output / "run.json"
    loaded = _read_json_artifact(path, missing_ok=True)
    if loaded is None:
        return ""
    if not isinstance(loaded, Mapping):
        raise ValidationError(f"validation run metadata must be an object: {path}")
    _validate_run_metadata(loaded, path=path)
    run_id = loaded.get("run_id", "")
    return str(run_id) if isinstance(run_id, str) else ""


def _validated_result_run_id(arguments: argparse.Namespace) -> str:
    output = Path(arguments.output)
    current_run_id = _current_run_id(output)
    captured_run_id = str(getattr(arguments, "_validation_run_id", "") or "")
    if arguments.model_worker:
        captured_run_id = str(arguments.worker_run_id or "")
        if not captured_run_id or "\x00" in captured_run_id:
            raise ValidationError("model worker requires a valid parent validation run ID")
    if captured_run_id and current_run_id != captured_run_id:
        raise ValidationError("validation run changed while this model worker was running")
    return captured_run_id or current_run_id


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
        extra = [command for command in existing if command not in discovered[kind]]
        discovered[kind] = (extra + discovered[kind])[:MAX_REPRO_COMMANDS_PER_BACKEND]
        discovered["command_count"][kind] = max(
            existing_count,
            discovered["command_count"][kind] + len(extra),
        )
    discovered["dataset"] = reproduce.get("dataset", {})
    representative = discovered.get("representative", {})
    if not isinstance(representative, Mapping) or not representative.get("sample_id"):
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


def _actual_complete_count(result: Mapping[str, Any]) -> int:
    raw_result = result.get("raw_result", {})
    if not isinstance(raw_result, Mapping):
        return 0
    complete_count = raw_result.get("complete_count")
    return complete_count if type(complete_count) is int else 0


def _validate_report_run_binding(
    run: Mapping[str, Any] | None,
    results: Sequence[Mapping[str, Any]],
    paths: Sequence[Path],
) -> bool:
    if run is None:
        return False
    run_id = run.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        return False
    for result, path in zip(results, paths, strict=True):
        if result.get("run_id") != run_id:
            raise ValidationError(
                f"validation result belongs to a different run than run.json: {path}"
            )
        validation = result.get("validation", {})
        if not isinstance(validation, Mapping) or validation.get("status") != "passed":
            continue
        _command, requested, prepared = _dataset_reproduction(result)
        actual = _actual_complete_count(result)
        if (
            prepared <= 0
            or actual <= 0
            or prepared != actual
            or (requested > 0 and requested != actual)
        ):
            raise ValidationError(
                "passed validation result lacks complete requested, "
                f"prepared, and actual sample evidence: {path}"
            )
    return True


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
        expected_directory = trtmc_disagreements._sample_directory_name(sample_id)
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
                not (character.isascii() and (character.isalnum() or character in "._-"))
                for character in media_path.name
            )
            or trtmc_disagreements._media_kind(media_path) != str(item.get("kind", ""))
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
    if type(count) is not int or count < 0 or type(limit) is not int or limit < 0:
        raise ValidationError("invalid validation disagreement metadata")
    artifact_name = str(metadata.get("path", DISAGREEMENT_ARTIFACT_NAME))
    if artifact_name != DISAGREEMENT_ARTIFACT_NAME:
        raise ValidationError(
            f"validation disagreement artifact path must be {DISAGREEMENT_ARTIFACT_NAME}"
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
            read_artifact=lambda path: (
                _read_case_text_artifact(
                    path,
                    case_dir=case_dir,
                    missing_ok=True,
                )
                if artifact_text is None
                else artifact_text
            ),
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
        return f'<span class="unavailable">{html.escape(not_compared_reason)}</span>'
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
        quantization = str(contract.get("trtmc_quantization", "") or "").upper()
        reference = str(contract.get("reference_precision", "") or "").upper()
        candidate = (
            f"{quantization} ({base} base)" if quantization and quantization != "NONE" else base
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
        f'<div class="detail">{html.escape(detail)}</div>' for detail in details
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
    _command, sample_limit, prepared = _dataset_reproduction(result)
    actual = _actual_complete_count(result)
    requested = str(sample_limit) if sample_limit else "full dataset"
    return (
        f"{actual}"
        '<div class="detail">'
        f"requested {html.escape(requested)} · prepared {prepared}"
        "</div>"
    )


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
                            workload_dir = output / model_entry.name / workload_entry.name
                            if workload_entry.is_symlink():
                                raise ValidationError(
                                    "validation result directory must not "
                                    f"be a symlink: {workload_dir}"
                                )
                            if not workload_entry.is_dir(follow_symlinks=False):
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
                                candidate = workload_dir / "comparison.json"
                                if not stat.S_ISREG(metadata.st_mode):
                                    raise ValidationError(
                                        f"validation result must be a regular file: {candidate}"
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
            f"cannot securely discover validation results in {output}: {exc}"
        ) from exc
    finally:
        os.close(output_fd)
    return sorted(result_paths)


def _validate_report_result_path(output: Path, path: Path) -> None:
    """Require an exact output/model/workload/comparison.json real path."""
    try:
        relative = path.relative_to(output)
    except ValueError as exc:
        raise ValidationError(f"validation result is outside the output root: {path}") from exc
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
            raise ValidationError(f"validation result directory must not be a symlink: {directory}")
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
    include_identity: bool = False,
    include_version: bool = False,
) -> Any:
    _validate_report_result_path(output, path)
    relative = path.relative_to(output)
    missing = [name for name in ("O_NONBLOCK", "O_NOFOLLOW") if not hasattr(os, name)]
    if missing:
        raise ValidationError("secure validation result reads require " + ", ".join(missing))
    directory_flags = _secure_directory_flags()
    result_flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
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
            raise ValidationError(f"validation case result must be a real directory: {path.parent}")
        descriptor = os.open(
            relative.name,
            result_flags,
            dir_fd=workload_descriptor,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValidationError(f"validation result must be a regular file: {path}")
        if metadata.st_nlink != 1:
            raise ValidationError(f"validation result must not be a hard link: {path}")
        if metadata.st_size > MAX_REPORT_ARTIFACT_BYTES:
            raise ValidationError(
                f"validation result exceeds {MAX_REPORT_ARTIFACT_BYTES} bytes: {path}"
            )
        with os.fdopen(descriptor, "rb") as result_file:
            descriptor = None
            try:
                payload = result_file.read(MAX_REPORT_ARTIFACT_BYTES + 1)
                if len(payload) > MAX_REPORT_ARTIFACT_BYTES:
                    raise ValidationError(
                        f"validation result exceeds {MAX_REPORT_ARTIFACT_BYTES} bytes: {path}"
                    )
                loaded = json.loads(
                    payload.decode("utf-8"),
                    parse_constant=_reject_nonstandard_json_constant,
                )
                after = os.fstat(result_file.fileno())
            except (UnicodeError, ValueError, RecursionError) as exc:
                raise ValidationError(f"invalid validation result JSON in {path}: {exc}") from exc
        if not _same_file_version(after, metadata):
            raise ValidationError(f"validation result changed while reading: {path}")
        _verify_visible_regular_artifact(
            path,
            workload_descriptor,
            after,
            operation="reading",
        )
        _verify_visible_artifact_parent(path, workload_descriptor)
        extras: list[Any] = []
        if include_size:
            extras.append(len(payload))
        if include_identity:
            extras.append(_metadata_identity(after))
        if include_version:
            extras.append(
                _ArtifactVersion(
                    after,
                    hashlib.sha256(payload).digest(),
                )
            )
        if extras:
            return loaded, *extras
        return loaded
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
                f"invalid validation result JSON in {path}: nesting exceeds {maximum_depth}"
            )
        if isinstance(current, float) and not math.isfinite(current):
            raise ValidationError(f"invalid validation result JSON in {path}: non-finite number")
        if isinstance(current, str):
            try:
                current.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValidationError(
                    f"invalid validation result JSON in {path}: string is not valid UTF-8"
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
        raise ValidationError(f"validation disagreement metadata must be an object: {path}")
    if "count" not in metadata:
        raise ValidationError(f"validation disagreement metadata must include count: {path}")
    count = metadata["count"]
    if type(count) is not int or count < 0:
        raise ValidationError(
            f"validation disagreement metadata count must be a non-negative integer: {path}"
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
    artifact_name = str(metadata.get("path", DISAGREEMENT_ARTIFACT_NAME))
    if artifact_name != DISAGREEMENT_ARTIFACT_NAME:
        raise ValidationError(
            f"validation disagreement artifact path must be {DISAGREEMENT_ARTIFACT_NAME}: {path}"
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
        raise ValidationError(f"validation result model does not match its path: {path}")
    workload = result.get("workload")
    reason_value = result.get("not_compared_reason", "")
    if not isinstance(reason_value, str):
        raise ValidationError(f"validation result not_compared_reason must be a string: {path}")
    not_compared_reason = reason_value
    legacy_e2e = (
        result.get("executor") == "e2e"
        and expected_workload == "e2e"
        and workload is None
        and not_compared_reason == LEGACY_E2E_REASON
    )
    if not_compared_reason and expected_workload != NOT_COMPARED_DIRECTORY and not legacy_e2e:
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
    elif not isinstance(workload, str) or not workload or workload != expected_workload:
        raise ValidationError(f"validation result workload does not match its path: {path}")


def _normalize_result_files(
    output: Path,
    result_paths: Sequence[Path],
    *,
    expected_gates_by_binding: (Mapping[tuple[str, str], Mapping[str, Any]] | None) = None,
) -> tuple[
    list[dict[str, Any]],
    dict[Path, _CaseArtifactStage],
]:
    if len(result_paths) > MAX_REPORT_RESULTS:
        raise ValidationError(f"validation report result count exceeds {MAX_REPORT_RESULTS}")
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
                f"validation report comparison inputs exceed {MAX_REPORT_RESULT_BYTES} bytes"
            )
        _validate_report_json_depth(loaded, path=path)
        if not isinstance(loaded, Mapping):
            raise ValidationError(f"validation result JSON must be an object: {path}")
        _validate_disagreement_metadata(loaded, path=path)
        _validate_result_identity(loaded, path=path)
        expected_gates = None
        if loaded.get("workload") is not None:
            key = (loaded.get("model"), loaded.get("workload"))
            gate_context = expected_gates_by_binding
            explicit_context = gate_context is not None
            if gate_context is None:
                gate_context = _default_authoritative_gates_by_binding()
            if key in gate_context:
                expected_gates = gate_context[key]
            elif explicit_context:
                raise ValidationError(
                    "validation report has no authoritative gate "
                    f"configuration for {key[0]}/{key[1]}: {path}"
                )
            else:
                raw_result = loaded.get("raw_result")
                raw_mode = raw_result.get("mode") if isinstance(raw_result, Mapping) else None
                raw_status = raw_result.get("status") if isinstance(raw_result, Mapping) else None
                if raw_status in {"pass", "passed"} and raw_mode in _PRIMARY_METRIC_BY_MODE:
                    raise ValidationError(
                        "validation report has no authoritative gate "
                        f"configuration for passed {key[0]}/{key[1]}: "
                        f"{path}"
                    )
        result = _normalize_result(
            loaded,
            expected_gates=expected_gates,
        )
        _validate_report_json_depth(
            result,
            path=path,
            maximum_depth=MAX_VALIDATION_RESULT_JSON_DEPTH,
        )
        _validate_disagreement_metadata(result, path=path)
        _validate_result_identity(result, path=path)
        results.append(result)
    try:
        disagreement_source_budget = [MAX_REPORT_DISAGREEMENT_SOURCE_BYTES]
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
            existing_count = metadata.get("count") if isinstance(metadata, Mapping) else 0
            stage: _CaseArtifactStage | None = None
            should_clear = (
                work_dir is None and isinstance(metadata, Mapping) and existing_count == 0
            )
            if should_clear:
                artifact_path = path.parent / DISAGREEMENT_ARTIFACT_NAME
                try:
                    trtmc_disagreements.load_disagreement_preview(
                        artifact_path,
                        limit=0,
                        expected_count=0,
                        read_artifact=lambda artifact: _read_case_text_artifact(
                            artifact,
                            case_dir=path.parent,
                            missing_ok=True,
                        ),
                    )
                except (
                    UnicodeError,
                    ValueError,
                    RecursionError,
                ) as exc:
                    raise ValidationError(
                        f"invalid validation disagreement artifact {artifact_path}: {exc}"
                    ) from exc
            if work_dir is not None or should_clear:
                stage = _create_case_artifact_stage(path.parent)
                stages[path] = stage
            if work_dir is not None:
                _refresh_disagreement_artifact(
                    result,
                    path.parent,
                    staging_root=(stage.path if stage is not None else None),
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
                    "inline_limit": (trtmc_disagreements.INLINE_DISAGREEMENT_LIMIT),
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
            artifact_roots.get(path, path.parent) if artifact_roots is not None else path.parent
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
    run_error = str(run.get("error", "") or "") if isinstance(run, Mapping) else ""
    run_failure = (
        f'<div class="run-failure"><strong>Run failure:</strong> {html.escape(run_error)}</div>'
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
{report["summary"]["actual_samples"]} completed samples ·
{report["summary"]["prepared_samples"]} prepared ·
{report["summary"]["requested_samples"]} explicitly requested
{duration_summary}<br>
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
            raise ValidationError(f"validation disagreement metadata must be an object: {path}")
        count = metadata.get("count")
        if type(count) is not int or count < 0:
            raise ValidationError(f"invalid validation disagreement count: {path}")
        if count > trtmc_disagreements.MAX_DISAGREEMENT_RECORDS:
            raise ValidationError(
                "validation disagreement count exceeds "
                f"{trtmc_disagreements.MAX_DISAGREEMENT_RECORDS}: {path}"
            )
        record_count += count
        if record_count > MAX_REPORT_DISAGREEMENT_RECORDS:
            raise ValidationError(
                f"validation report disagreement count exceeds {MAX_REPORT_DISAGREEMENT_RECORDS}"
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
            parent = _transaction_parent_path(update.stage.case_dir)
            identity = update.stage.case_identity
        else:
            parent = _transaction_parent_path(update.path.parent)
            identity = update.parent_identity
        previous = identities.setdefault(parent, identity)
        if previous != identity:
            raise ValidationError(
                f"validation transaction has conflicting parent identities: {parent}"
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
                    f"validation transaction parent was replaced before commit: {parent}"
                )
            anchors[parent] = descriptor
    except BaseException:
        for descriptor in anchors.values():
            os.close(descriptor)
        raise
    for kind, update in entries:
        parent = _transaction_parent_path(
            update.stage.case_dir if kind == "directory" else update.path.parent
        )
        update.anchor_fd = anchors[parent]
    return anchors


def _verify_report_transaction_visibility(
    entries: Sequence[tuple[str, Any]],
) -> None:
    for kind, update in entries:
        if kind == "directory":
            _verify_committed_case_directory_update(update)
        else:
            _verify_committed_file_update(update)


def _rollback_report_transaction(
    entries: Sequence[tuple[str, Any]],
) -> list[str]:
    errors: list[str] = []
    for kind, update in reversed(entries):
        try:
            if kind == "directory":
                _rollback_case_directory_update(update)
            else:
                _rollback_file_update(update)
        except (OSError, ValidationError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return errors


def write_report(
    output: Path,
    *,
    result_paths: Sequence[Path] | None = None,
    expected_gates_by_binding: (Mapping[tuple[str, str], Mapping[str, Any]] | None) = None,
) -> tuple[Path, Path, dict[str, Any]]:
    with _validation_output_publication_lock(output):
        report_kwargs: dict[str, Any] = {}
        if expected_gates_by_binding is not None:
            report_kwargs["expected_gates_by_binding"] = expected_gates_by_binding
        return _write_report_locked(
            output,
            result_paths=result_paths,
            **report_kwargs,
        )


def _revalidate_report_run_version(
    run_path: Path,
    expected_version: _ArtifactVersion,
) -> None:
    current_version = _regular_artifact_version(
        run_path,
        missing_ok=True,
    )
    if not _same_artifact_version(current_version, expected_version):
        raise ValidationError("validation run changed while the report was being prepared")


def _write_report_locked(
    output: Path,
    *,
    result_paths: Sequence[Path] | None = None,
    expected_gates_by_binding: (Mapping[tuple[str, str], Mapping[str, Any]] | None) = None,
) -> tuple[Path, Path, dict[str, Any]]:
    selected_paths = (
        _discover_report_result_paths(output)
        if result_paths is None
        else sorted(dict.fromkeys(result_paths))
    )
    run_path = output / "run.json"
    run_metadata, run_version = _read_json_artifact(
        run_path,
        missing_ok=True,
        include_version=True,
    )
    run_status: str | None = None
    if run_metadata is not None:
        if not isinstance(run_metadata, Mapping):
            raise ValidationError(f"validation run metadata must be an object: {run_path}")
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
        expected_gates_by_binding=expected_gates_by_binding,
    )
    all_paths = selected_paths
    provenance_bound = _validate_report_run_binding(
        run_metadata if isinstance(run_metadata, Mapping) else None,
        all_results,
        all_paths,
    )
    transaction_entries: list[tuple[str, Any]] = []
    transaction_anchors: dict[Path, int] = {}
    try:
        staged_payloads, artifact_texts = _preflight_report_disagreements(
            all_paths,
            all_results,
            stages,
        )
        selected_paths, results = _deduplicate_results(
            all_paths,
            all_results,
        )
        validation_counts, comparison_counts, execution_errors = _report_counts(results)
        traffic_light_counts = _traffic_light_counts(results)
        sample_evidence = [
            (
                _dataset_reproduction(result)[1],
                _dataset_reproduction(result)[2],
                _actual_complete_count(result),
            )
            for result in results
        ]
        requested_samples = sum(item[0] for item in sample_evidence)
        prepared_samples = sum(item[1] for item in sample_evidence)
        actual_samples = sum(item[2] for item in sample_evidence)
        generated_at = _utc_now()
        report = {
            "schema_version": "trtmc.validation-report/v2",
            "generated_at": generated_at.isoformat(),
            "validation_status": (
                "failed"
                if (
                    run_status == "failed"
                    or validation_counts["failed"]
                    or (not results and run_status == "completed")
                )
                else "incomplete"
                if (
                    run_status != "completed"
                    or not provenance_bound
                    or validation_counts["not_compared"]
                    or validation_counts["skipped"]
                )
                else "passed"
            ),
            "summary": {
                "cases": len(results),
                "execution_completed": sum(
                    result["execution"]["status"] == "completed" for result in results
                ),
                "execution_errors": execution_errors,
                "agreements": comparison_counts["agreement"],
                "disagreements": comparison_counts["disagreement"],
                "not_compared": comparison_counts["not_run"],
                "validation_passed": validation_counts["passed"],
                "validation_failed": validation_counts["failed"],
                "validation_skipped": validation_counts["skipped"],
                "requested_samples": requested_samples,
                "prepared_samples": prepared_samples,
                "actual_samples": actual_samples,
                "full_dataset_cases": sum(
                    requested == 0 for requested, _prepared, _actual in sample_evidence
                ),
                "selected_samples": actual_samples,
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
                report["summary"]["duration_seconds"] = duration_seconds
        document = _report_document(
            report,
            rows=_report_rows(
                output,
                results,
                selected_paths,
                artifact_roots={path: stage.path for path, stage in stages.items()},
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
                f"validation HTML report exceeds {MAX_REPORT_ARTIFACT_BYTES} bytes: {html_path}"
            )
        for path in all_paths:
            stage = stages.get(path)
            if stage is not None:
                directory_update = _prepare_case_directory_update(stage)
                transaction_entries.append(("directory", directory_update))
                artifact_update = _prepare_file_update(
                    path.parent / DISAGREEMENT_ARTIFACT_NAME,
                    staged_payloads[path],
                )
                transaction_entries.append(("file", artifact_update))
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
                            maximum_depth=(MAX_VALIDATION_RESULT_JSON_DEPTH),
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
        transaction_anchors = _open_report_transaction_anchors(transaction_entries)
        try:
            _revalidate_report_run_version(run_path, run_version)
            for kind, update in transaction_entries:
                if kind == "directory":
                    _commit_case_directory_update(update)
                else:
                    _commit_file_update(update)
            _revalidate_report_run_version(run_path, run_version)
            _verify_report_transaction_visibility(transaction_entries)
        except BaseException as exc:
            rollback_errors = _rollback_report_transaction(transaction_entries)
            if rollback_errors:
                raise ValidationError(
                    "validation report transaction failed and rollback "
                    "was incomplete: " + " | ".join(rollback_errors)
                ) from exc
            raise
        return json_path, html_path, report
    finally:
        cleanup_errors: list[str] = []
        try:
            for kind, update in transaction_entries:
                try:
                    if kind == "directory":
                        cleanup_errors.extend(_finalize_case_directory_update(update))
                    else:
                        cleanup_errors.extend(_finalize_file_update(update))
                except (OSError, ValidationError) as exc:
                    cleanup_errors.append(
                        f"validation transaction cleanup failed: {type(exc).__name__}: {exc}"
                    )
            for stage in stages.values():
                try:
                    anchored_case_fd = transaction_anchors.get(
                        _transaction_parent_path(stage.case_dir)
                    )
                    _cleanup_case_artifact_stage(
                        stage,
                        anchored_case_fd=anchored_case_fd,
                    )
                except (OSError, ValidationError) as exc:
                    cleanup_errors.append(
                        f"validation report stage cleanup incomplete: {stage.path}: {exc}"
                    )
        finally:
            for descriptor in transaction_anchors.values():
                try:
                    os.close(descriptor)
                except OSError as exc:
                    cleanup_errors.append(f"validation transaction anchor close failed: {exc}")
        if cleanup_errors:
            cleanup_message = "validation report cleanup incomplete: " + " | ".join(cleanup_errors)
            active_error = sys.exc_info()[1]
            if active_error is not None:
                if hasattr(active_error, "add_note"):
                    active_error.add_note(cleanup_message)
                else:
                    print(cleanup_message, file=sys.stderr)
            else:
                raise ValidationError(cleanup_message)


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
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
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
    parser.add_argument(
        "--worker-run-id",
        default="",
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
    *,
    run_id: str = "",
) -> list[str]:
    workload = _required_workload(binding)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        binding.model,
        workload,
        "--model-worker",
    ]
    if run_id:
        command.extend(["--worker-run-id", run_id])
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


def _public_worker_command(command: Sequence[str]) -> list[str]:
    public_command = []
    index = 0
    while index < len(command):
        token = command[index]
        if token.startswith("--worker-run-id="):
            index += 1
            continue
        if token == "--worker-run-id":
            index += 1
            if index < len(command) and not command[index].startswith("-"):
                index += 1
            continue
        if token == "--model-worker":
            index += 1
            continue
        public_command.append(token)
        index += 1
    return public_command


def _worker_error_result(
    binding: Binding,
    *,
    command: Sequence[str],
    returncode: int,
    worker_log: Path,
    sample_limit: int,
    error: str,
    run_id: str = "",
) -> dict[str, Any]:
    payload = {
        "schema_version": "trtmc.validation-result/v2",
        "model": binding.model,
        "workload": binding.workload,
        "executor": "model_worker",
        "status": "failed",
        "returncode": returncode,
        "reference_environment": [],
        "reproduce": {
            "dataset": {
                "command": shlex.join(_public_worker_command(command)),
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
    if run_id:
        payload["run_id"] = run_id
    return _normalize_result(payload)


def _run_supervised_binding(
    binding: Binding,
    *,
    arguments: argparse.Namespace,
    catalog: Mapping[str, Any],
    attempt: int = 1,
    expected_gates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    case_dir = _prepare_case_directory(arguments.output, binding)
    current_run_id = str(getattr(arguments, "_validation_run_id", "") or "")
    if not current_run_id:
        current_run_id = _current_run_id(Path(arguments.output))
    comparison_path = case_dir / "comparison.json"
    previous_comparison_version = _regular_artifact_version(
        comparison_path,
        missing_ok=True,
    )
    publication_target = previous_comparison_version
    worker_log = case_dir / ("worker.log" if attempt == 1 else f"worker.attempt-{attempt}.log")
    command = _worker_command(
        binding,
        arguments,
        run_id=current_run_id,
    )
    launch_error = ""
    try:
        returncode = _run_subprocess(command, worker_log, _source_environment())
    except OSError as exc:
        returncode = 127
        launch_error = f"could not start model worker: {exc}"
    try:
        if launch_error:
            raise ValidationError(launch_error)
        if current_run_id and _current_run_id(Path(arguments.output)) != current_run_id:
            raise ValidationError("validation run changed while the model worker was running")
        try:
            loaded, read_version = _read_report_result(
                arguments.output,
                comparison_path,
                include_version=True,
            )
        except ValidationError as exc:
            raise ValidationError(
                f"worker exited with code {returncode} without a valid comparison.json: {exc}"
            ) from exc
        publication_target = read_version
        current_comparison_version = _regular_artifact_version(
            comparison_path,
        )
        if not _same_artifact_version(
            current_comparison_version,
            read_version,
        ):
            raise ValidationError("worker comparison.json changed after it was read")
        if previous_comparison_version.metadata is not None and _same_artifact_version(
            read_version,
            previous_comparison_version,
        ):
            raise ValidationError("worker did not replace its stale comparison.json")
        _validate_report_json_depth(loaded, path=comparison_path)
        if not isinstance(loaded, Mapping):
            raise ValidationError("worker comparison.json must contain an object")
        raw_evidence = loaded.get("raw_result")
        if not isinstance(raw_evidence, Mapping) or not raw_evidence:
            raise ValidationError(
                "runnable binding worker comparison.json must include non-empty raw_result evidence"
            )
        raw_status = raw_evidence.get("status")
        if not isinstance(raw_status, str) or raw_status not in {
            "pass",
            "passed",
            "fail",
            "failed",
        }:
            raise ValidationError(
                "runnable worker raw_result must include an explicit comparison status"
            )
        result = _normalize_result(
            loaded,
            expected_gates=expected_gates,
        )
        if current_run_id and result.get("run_id") != current_run_id:
            raise ValidationError("worker comparison.json belongs to a different validation run")
        if result.get("model") != binding.model or result.get("workload") != binding.workload:
            raise ValidationError("worker wrote comparison.json for a different binding")
        execution_completed = result["execution"]["status"] == "completed"
        raw_model = raw_evidence.get("model")
        raw_suite = raw_evidence.get("suite")
        raw_workload = raw_evidence.get("workload")
        if (
            (execution_completed and (raw_model != binding.model or raw_suite != binding.workload))
            or (raw_model is not None and raw_model != binding.model)
            or (raw_suite is not None and raw_suite != binding.workload)
            or (raw_workload is not None and raw_workload != binding.workload)
        ):
            raise ValidationError("worker raw_result evidence belongs to a different binding")
        if result["validation"]["status"] in {
            "not_compared",
            "skipped",
        }:
            raise ValidationError("worker did not complete a comparison for a runnable binding")
        if type(result["execution"].get("exit_code")) is not int:
            raise ValidationError(
                "runnable worker result must include an integer execution.exit_code"
            )
        expected_returncode = 1 if result["validation"]["status"] == "failed" else 0
        if returncode != expected_returncode:
            raise ValidationError(
                f"worker exited with code {returncode}, but its result "
                f"requires exit code {expected_returncode}"
            )
    except (OSError, ValueError, ValidationError) as exc:
        result = _worker_error_result(
            binding,
            command=command,
            returncode=returncode,
            worker_log=worker_log,
            sample_limit=resolve_sample_limit(catalog, binding, arguments.limit),
            error=str(exc),
            run_id=current_run_id,
        )
        _atomic_write_text(case_dir / "disagreements.jsonl", "")
    else:
        result["worker_log"] = str(worker_log)
        dataset = result.get("reproduce", {}).get("dataset", {})
        if isinstance(dataset, dict):
            dataset["command"] = shlex.join(_public_worker_command(command))
    _publish_validation_result(
        Path(arguments.output),
        comparison_path,
        result,
        expected_run_id=current_run_id,
        expected_target=publication_target,
    )
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
            str(raw_result.get("error_type", "")) if isinstance(raw_result, Mapping) else ""
        ),
        "error": (str(raw_result.get("error", "")) if isinstance(raw_result, Mapping) else ""),
    }


def _run_supervised_binding_with_retries(
    binding: Binding,
    *,
    arguments: argparse.Namespace,
    catalog: Mapping[str, Any],
    expected_gates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    case_dir = _case_directory(arguments.output, binding)
    comparison_path = case_dir / "comparison.json"
    expected_run_id = str(getattr(arguments, "_validation_run_id", "") or "")
    if not expected_run_id:
        expected_run_id = _current_run_id(Path(arguments.output))
    attempts = []
    result: dict[str, Any] = {}
    publication_target: _ArtifactVersion | None = None
    for attempt in range(1, arguments.model_attempts + 1):
        worker_kwargs: dict[str, Any] = {}
        if expected_gates is not None:
            worker_kwargs["expected_gates"] = expected_gates
        result = _run_supervised_binding(
            binding,
            arguments=arguments,
            catalog=catalog,
            attempt=attempt,
            **worker_kwargs,
        )
        published_result, publication_target = _read_report_result(
            Path(arguments.output),
            comparison_path,
            include_version=True,
        )
        if published_result != result:
            raise ValidationError("supervised comparison result changed before retry bookkeeping")
        execution = result.get("execution", {})
        execution_error = isinstance(execution, Mapping) and execution.get("status") == "error"
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
    if publication_target is None:
        raise ValidationError("validation worker did not produce a result")
    _publish_validation_result(
        Path(arguments.output),
        comparison_path,
        result,
        expected_run_id=expected_run_id,
        expected_target=publication_target,
    )
    return result


def _run_all_bindings(
    bindings: Iterable[Binding],
    *,
    arguments: argparse.Namespace,
    catalog: Mapping[str, Any],
    expected_gates_by_binding: (Mapping[tuple[str, str], Mapping[str, Any]] | None) = None,
) -> int:
    _prepare_run_directories(arguments)
    write_run_metadata(arguments.output)
    validation_run_id = _current_run_id(arguments.output)
    if not validation_run_id:
        raise ValidationError("validation run metadata did not provide a run ID")
    arguments._validation_run_id = validation_run_id
    failed = False
    not_compared = False
    current_result_paths: list[Path] = []
    report_kwargs: dict[str, Any] = {}
    if expected_gates_by_binding is not None:
        report_kwargs["expected_gates_by_binding"] = expected_gates_by_binding
    try:
        write_report(
            arguments.output,
            result_paths=[],
            **report_kwargs,
        )
        for binding in bindings:
            if not binding.runnable:
                print(
                    f"\nNot compared: {binding.model} / {binding.not_compared_reason}",
                    flush=True,
                )
                result, comparison = _write_not_compared_case(
                    binding,
                    arguments.output,
                    run_id=validation_run_id,
                )
                current_result_paths.append(comparison)
                not_compared = True
                _, report_path, _ = write_report(
                    arguments.output,
                    result_paths=current_result_paths,
                    **report_kwargs,
                )
                _print_result(result, comparison, report_path)
                continue
            sample_limit = resolve_sample_limit(
                catalog,
                binding,
                arguments.limit,
            )
            sample_note = "full dataset" if sample_limit == 0 else f"{sample_limit} samples"
            print(
                f"\nStarting worker: {binding.model} / {binding.workload} / {sample_note}",
                flush=True,
            )
            worker_kwargs: dict[str, Any] = {}
            if expected_gates_by_binding is not None:
                key = (binding.model, _required_workload(binding))
                if key not in expected_gates_by_binding:
                    raise ValidationError(
                        f"no authoritative gate configuration for {key[0]}/{key[1]}"
                    )
                worker_kwargs["expected_gates"] = expected_gates_by_binding[key]
            result = _run_supervised_binding_with_retries(
                binding,
                arguments=arguments,
                catalog=catalog,
                **worker_kwargs,
            )
            comparison = _case_directory(arguments.output, binding) / "comparison.json"
            current_result_paths.append(comparison)
            _, report_path, _ = write_report(
                arguments.output,
                result_paths=current_result_paths,
                **report_kwargs,
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
                expected_run_id=validation_run_id,
            )
            write_report(
                arguments.output,
                result_paths=current_result_paths,
                **report_kwargs,
            )
        except BaseException as reporting_exc:
            note = f"Additionally failed to finalize the validation report: {reporting_exc}"
            if hasattr(exc, "add_note"):
                exc.add_note(note)
            else:
                print(note, file=sys.stderr)
        raise
    finalize_run_metadata(
        arguments.output,
        expected_run_id=validation_run_id,
    )
    write_report(
        arguments.output,
        result_paths=current_result_paths,
        **report_kwargs,
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
    expected_gates_by_binding: (Mapping[tuple[str, str], Mapping[str, Any]] | None) = None,
) -> int:
    _prepare_run_directories(arguments)
    if not arguments.model_worker:
        write_run_metadata(arguments.output)
        validation_run_id = _current_run_id(arguments.output)
        if not validation_run_id:
            raise ValidationError("validation run metadata did not provide a run ID")
        arguments._validation_run_id = validation_run_id
    failed = False
    not_compared = False
    report_kwargs: dict[str, Any] = {}
    if expected_gates_by_binding is not None:
        report_kwargs["expected_gates_by_binding"] = expected_gates_by_binding
    for binding in bindings:
        if not binding.runnable:
            print(
                f"\nNot compared: {binding.model} / {binding.not_compared_reason}",
                flush=True,
            )
            result, comparison = _write_not_compared_case(
                binding,
                arguments.output,
            )
            not_compared = True
            if not arguments.model_worker:
                _, report_path, _ = write_report(
                    arguments.output,
                    **report_kwargs,
                )
                _print_result(result, comparison, report_path)
            continue
        binding_arguments = copy.copy(arguments)
        binding_arguments.limit = resolve_sample_limit(
            catalog,
            binding,
            arguments.limit,
        )
        sample_note = (
            "full dataset" if binding_arguments.limit == 0 else f"{binding_arguments.limit} samples"
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
            _, report_path, _ = write_report(
                arguments.output,
                **report_kwargs,
            )
            comparison = _case_directory(arguments.output, binding) / "comparison.json"
            _print_result(result, comparison, report_path)
        failed = failed or result["validation"]["status"] == "failed"
    if failed:
        return 1
    return 2 if not_compared and not arguments.all else 0


def _main(arguments: argparse.Namespace) -> int:
    if arguments.worker_run_id and not arguments.model_worker:
        raise ValidationError("--worker-run-id is only valid for an internal model worker")
    if arguments.model_worker and not arguments.worker_run_id:
        raise ValidationError("internal model worker is missing its parent validation run ID")
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
    expected_gates_by_binding = _authoritative_gates_by_binding(
        catalog,
        suites=suites,
        task_models=task_models,
    )
    if not arguments.model_worker:
        return _run_all_bindings(
            bindings,
            arguments=arguments,
            catalog=catalog,
            expected_gates_by_binding=expected_gates_by_binding,
        )
    return _run_bindings(
        bindings,
        arguments=arguments,
        catalog=catalog,
        task_models=task_models,
        suites=suites,
        expected_gates_by_binding=expected_gates_by_binding,
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
