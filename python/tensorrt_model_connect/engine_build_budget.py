# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed CI budget for full TensorRT bundle builds."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
import re
import signal
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar


_GUARD_DIR_ENV = "TRTMC_ENGINE_BUILD_GUARD_DIR"
_IDENTITY_ENV = "TRTMC_ENGINE_BUILD_IDENTITY"
_REVISION_ENV = "TRTMC_ENGINE_BUILD_REVISION"
_COMMAND_ENV = "TRTMC_ENGINE_BUILD_COMMAND_JSON"
_RECOVERY_ATTEMPT_ENV = "TRTMC_ENGINE_BUILD_RECOVERY_ATTEMPT"
_RECOVERY_SIGNAL_ENV = "TRTMC_ENGINE_BUILD_RECOVERY_SIGNAL"
_RECOVERABLE_SIGNALS = frozenset({signal.SIGSEGV})

P = ParamSpec("P")
R = TypeVar("R")


def _ledger_path(guard_dir: Path, identity: str) -> Path:
    safe_identity = re.sub(r"[^A-Za-z0-9_.-]+", "-", identity).strip("-.") or "model"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return guard_dir / f"{safe_identity}-{digest}.json"


def _jsonable_arguments(arguments: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(arguments, sort_keys=True, default=str))


def _stable_model_dir_argument(model_dir: object) -> object:
    """Return a stable identity for a verified synthetic NeMo model directory.

    Family-owned NeMo adapters stage a synthetic config beside one archive
    symlink.  The staging directory is intentionally temporary, so a fresh
    recovery process receives a different path for the same archive.  Only
    canonicalize that path when both independent declarations resolve to the
    same real archive; every malformed or ordinary model directory retains its
    exact original identity and therefore fails closed if it changes.
    """
    try:
        staged_dir = Path(str(model_dir))
        config_path = staged_dir / "config.json"
        if (
            staged_dir.is_symlink()
            or not staged_dir.is_dir()
            or config_path.is_symlink()
            or not config_path.is_file()
        ):
            return model_dir
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            return model_dir
        raw_archive = config.get("_nemo_archive_path")
        if not isinstance(raw_archive, str) or not raw_archive.strip():
            return model_dir
        declared_archive = Path(raw_archive)
        if declared_archive.suffix != ".nemo":
            return model_dir
        staged_archive = staged_dir / declared_archive.name
        if {entry.name for entry in staged_dir.iterdir()} != {
            "config.json",
            declared_archive.name,
        }:
            return model_dir
        if not staged_archive.is_symlink():
            return model_dir
        declared_target = declared_archive.resolve(strict=True)
        staged_target = staged_archive.resolve(strict=True)
        if declared_target != staged_target or not declared_target.is_file():
            return model_dir
        archive_stat = declared_target.stat()
        stable_config = dict(config)
        stable_config["_nemo_archive_path"] = {
            "device": archive_stat.st_dev,
            "inode": archive_stat.st_ino,
            "modified_ns": archive_stat.st_mtime_ns,
            "path": str(declared_target),
            "size_bytes": archive_stat.st_size,
            "status_changed_ns": archive_stat.st_ctime_ns,
        }
        return {
            "kind": "verified_nemo_staging",
            "config": stable_config,
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return model_dir


@contextmanager
def _locked_claim(claim_path: Path):
    import tensorrt_model_connect.utils.fcntl_shim as fcntl

    lock_path = claim_path.with_suffix(".lock")
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _recovery_request() -> tuple[int, int] | None:
    raw_attempt = os.environ.get(_RECOVERY_ATTEMPT_ENV, "").strip()
    raw_signal = os.environ.get(_RECOVERY_SIGNAL_ENV, "").strip()
    if not raw_attempt and not raw_signal:
        return None
    if not raw_attempt.isdigit() or not raw_signal.isdigit():
        raise RuntimeError(
            f"{_RECOVERY_ATTEMPT_ENV} and {_RECOVERY_SIGNAL_ENV} must both be integers"
        )
    attempt = int(raw_attempt)
    signal_number = int(raw_signal)
    if attempt != 2:
        raise RuntimeError(f"{_RECOVERY_ATTEMPT_ENV} must be 2")
    if signal_number not in _RECOVERABLE_SIGNALS:
        raise RuntimeError(
            f"{_RECOVERY_SIGNAL_ENV}={signal_number} is not a recoverable native signal"
        )
    return attempt, signal_number


def _pid_is_alive(pid: object) -> bool:
    if type(pid) is not int or pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _utc_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        return None
    return timestamp


def _recover_interrupted_claim(
    claim_path: Path,
    *,
    identity: str,
    arguments_sha256: str,
    output_path: Path,
    build_timing_path: str,
    source_revision: str,
    command: object,
    recovery: tuple[int, int],
) -> tuple[Path, float]:
    attempt, signal_number = recovery
    try:
        payload = json.loads(claim_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot recover unreadable full bundle build ledger for {identity!r}: {claim_path}"
        ) from exc
    previous_attempt = attempt - 1
    expected = {
        "schema_version": 1,
        "identity": identity,
        "status": "started",
        "invocation_count": 1,
        "attempt_count": previous_attempt,
        "source_revision": source_revision,
        "bundle_path": str(output_path),
        "build_timing_path": build_timing_path,
        "arguments_sha256": arguments_sha256,
        "command": command,
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    for field in ("schema_version", "invocation_count", "attempt_count"):
        if type(payload.get(field)) is not int:
            mismatches.append(field)
    recoveries = payload.get("recovery_attempts", [])
    if not isinstance(recoveries, list) or len(recoveries) != previous_attempt - 1:
        mismatches.append("recovery_attempts")
    previous_pid = payload.get("builder_pid")
    current_pid = os.getpid()
    if type(previous_pid) is not int or previous_pid < 1 or previous_pid == current_pid:
        mismatches.append("builder_pid")
    elif _pid_is_alive(previous_pid):
        mismatches.append("builder_pid")
    previous_started_at = _utc_timestamp(payload.get("started_at"))
    recovered_at_time = datetime.now(timezone.utc)
    if previous_started_at is None or previous_started_at > recovered_at_time:
        mismatches.append("started_at")
    if output_path.exists():
        mismatches.append("bundle_path_exists")
    if mismatches:
        raise RuntimeError(
            f"cannot recover full bundle build for {identity!r}; "
            f"ledger mismatch: {', '.join(sorted(set(mismatches)))}"
        )

    recovered_at = recovered_at_time.isoformat()
    recoveries.append(
        {
            "attempt": previous_attempt,
            "returncode": -signal_number,
            "signal": signal_number,
            "builder_pid": previous_pid,
            "started_at": payload.get("started_at", ""),
            "recovered_at": recovered_at,
        }
    )
    payload.update(
        {
            "status": "started",
            "attempt_count": attempt,
            "builder_pid": current_pid,
            "started_at": recovered_at,
            "recovery_attempts": recoveries,
        }
    )
    temporary = claim_path.with_suffix(f".json.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, claim_path)
    return claim_path, time.monotonic()


def _claim_build(
    *,
    arguments: dict[str, object],
    output_path: Path,
) -> tuple[Path | None, float]:
    raw_guard_dir = os.environ.get(_GUARD_DIR_ENV, "").strip()
    if not raw_guard_dir:
        return None, time.monotonic()

    identity = os.environ.get(_IDENTITY_ENV, "").strip()
    if not identity:
        raise RuntimeError(f"{_IDENTITY_ENV} is required when {_GUARD_DIR_ENV} is enabled")

    guard_dir = Path(raw_guard_dir)
    guard_dir.mkdir(parents=True, exist_ok=True)
    claim_path = _ledger_path(guard_dir, identity)
    recovery = _recovery_request()
    normalized_arguments = _jsonable_arguments(arguments)
    if "model_dir" in normalized_arguments:
        normalized_arguments["model_dir"] = _stable_model_dir_argument(
            normalized_arguments["model_dir"]
        )
    encoded_arguments = json.dumps(
        normalized_arguments, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    arguments_sha256 = hashlib.sha256(encoded_arguments).hexdigest()
    raw_command = os.environ.get(_COMMAND_ENV, "")
    try:
        command = json.loads(raw_command) if raw_command else []
    except json.JSONDecodeError:
        command = [raw_command]
    build_timing_path = str(arguments.get("build_timing_path") or "")
    source_revision = os.environ.get(_REVISION_ENV, "")
    with _locked_claim(claim_path):
        if recovery is not None:
            if not claim_path.is_file():
                raise RuntimeError(
                    f"cannot recover full bundle build for {identity!r}; "
                    f"ledger is missing: {claim_path}"
                )
            return _recover_interrupted_claim(
                claim_path,
                identity=identity,
                arguments_sha256=arguments_sha256,
                output_path=output_path,
                build_timing_path=build_timing_path,
                source_revision=source_revision,
                command=command,
                recovery=recovery,
            )

        payload = {
            "schema_version": 1,
            "identity": identity,
            "status": "started",
            "invocation_count": 1,
            "attempt_count": 1,
            "builder_pid": os.getpid(),
            "source_revision": source_revision,
            "bundle_path": str(output_path),
            "build_timing_path": build_timing_path,
            "arguments_sha256": arguments_sha256,
            "command": command,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "recovery_attempts": [],
        }
        try:
            fd = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise RuntimeError(
                f"full TensorRT bundle build budget already consumed for {identity!r}: {claim_path}"
            ) from exc
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as claim_file:
                json.dump(payload, claim_file, indent=2)
                claim_file.write("\n")
        except BaseException:
            claim_path.unlink(missing_ok=True)
            raise
    return claim_path, time.monotonic()


def _finish_build(
    claim_path: Path | None,
    started: float,
    *,
    output_path: Path,
    status: str,
    error: str = "",
) -> None:
    if claim_path is None:
        return
    payload = json.loads(claim_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "status": status,
            "returncode": 0 if status == "passed" else 1,
            "elapsed_s": time.monotonic() - started,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "bundle_exists": output_path.is_file(),
            "bundle_size_bytes": output_path.stat().st_size if output_path.is_file() else 0,
        }
    )
    if error:
        payload["error"] = error
    temporary = claim_path.with_suffix(f".json.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, claim_path)


def enforce_single_full_bundle_build(func: Callable[P, R]) -> Callable[P, R]:
    """Allow one guarded logical build, with one verified SIGSEGV recovery."""
    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        output_path = Path(str(bound.arguments["output_path"]))
        claim_path: Path | None = None
        started = time.monotonic()
        try:
            claim_path, started = _claim_build(
                arguments=dict(bound.arguments),
                output_path=output_path,
            )
            result = func(*args, **kwargs)
            if claim_path is not None and not output_path.is_file():
                raise RuntimeError(f"guarded full bundle build did not produce {output_path}")
        except BaseException as exc:
            _finish_build(
                claim_path,
                started,
                output_path=output_path,
                status="failed",
                error=str(exc),
            )
            raise
        _finish_build(
            claim_path,
            started,
            output_path=output_path,
            status="passed",
        )
        return result

    return wrapped
