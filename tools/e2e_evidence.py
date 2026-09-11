# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded evidence I/O for family-owned pytest checks; no model judgments."""

from __future__ import annotations

import contextvars
import importlib.metadata
import json
import math
import mimetypes
import os
import platform
import re
import shlex
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

_ACTIVE: contextvars.ContextVar[Evidence | None] = contextvars.ContextVar(
    "e2e_evidence", default=None
)
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_REVISION = re.compile(r"[0-9a-f]{40}")
_FILE_LIMIT = 32 * 1024 * 1024
_TOTAL_LIMIT = 128 * 1024 * 1024
_PREVIEW_LIMIT = 64
_CHECK_LIMIT = 2000
_JSON_LIMIT = 8 * 1024 * 1024
_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".wav",
        ".flac",
        ".mp3",
        ".ogg",
        ".mp4",
        ".webm",
        ".npy",
        ".npz",
        ".f32",
        ".i32",
        ".u8",
        ".json",
        ".jsonl",
        ".txt",
        ".log",
        ".csv",
        ".cif",
        ".yaml",
        ".a3m",
        ".b2rq",
        ".raw",
        ".ppm",
    }
)


def _environment() -> dict[str, str]:
    result = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
    }
    for package in ("tensorrt", "torch", "transformers", "diffusers", "numpy", "pytest"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if probe.returncode == 0:
            result["gpu_and_driver"] = probe.stdout.strip()[:2048]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return result


def _revision(repository: Path) -> str:
    supplied = os.environ.get("TRTMC_E2E_SOURCE_REVISION", "")
    if supplied:
        if not _REVISION.fullmatch(supplied):
            raise ValueError("TRTMC_E2E_SOURCE_REVISION must be one exact source commit")
        return supplied
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        value = result.stdout.strip()
        return value if result.returncode == 0 and _REVISION.fullmatch(value) else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def _bounded_preview(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        return "[additional nesting omitted]"
    if isinstance(value, str):
        return value if len(value) <= 2048 else value[:2048] + " [truncated]"
    if isinstance(value, list):
        items = [_bounded_preview(item, depth + 1) for item in value[:32]]
        return items + ([{"omitted_items": len(value) - 32}] if len(value) > 32 else [])
    if isinstance(value, dict):
        items = list(value.items())
        result = {str(key)[:256]: _bounded_preview(item, depth + 1) for key, item in items[:32]}
        if len(items) > 32:
            result["omitted_fields"] = len(items) - 32
        return result
    return value


class Evidence:
    """Snapshot values chosen by a family and the outcomes reported by pytest."""

    def __init__(
        self,
        directory: Path,
        *,
        family: str,
        case: str,
        source_revision: str,
        roots: tuple[Path, ...] = (),
        nodeid: str = "",
    ) -> None:
        if not _SAFE_NAME.fullmatch(family) or not _SAFE_NAME.fullmatch(case):
            raise ValueError("evidence family and case must be safe path components")
        self.directory = directory
        self.roots = roots
        self.request: Any = None
        self._prepared = False
        self.bytes_written = 0
        self.stage = "setup"
        self.started = time.monotonic()
        self._files: list[dict[str, Any]] = []
        self.data: dict[str, Any] = {
            "schema_version": 1,
            "family": family,
            "case": case,
            "source_revision": source_revision,
            "nodeid": nodeid,
            "status": "running",
            "failure_stage": None,
            "observations": [],
            "checks": [],
            "timing": [],
            "artifacts": [],
            "issues": [],
        }

    def _prepare(self) -> None:
        if self._prepared:
            return
        if (
            self.directory.parent.parent.is_symlink()
            or self.directory.parent.is_symlink()
            or self.directory.is_symlink()
        ):
            raise ValueError("case evidence destination must not be a symlink")
        if self.directory.exists():
            shutil.rmtree(self.directory)
        self._prepared = True

    def _allowed(self, path: Path) -> bool:
        roots = list(self.roots)
        if self.request is not None:
            temporary = self.request.node.funcargs.get("tmp_path")
            if isinstance(temporary, Path):
                roots.append(temporary)
        return not path.is_symlink() and any(
            path.resolve().is_relative_to(root.resolve()) for root in roots
        )

    def _file(self, path: Path, role: str) -> dict[str, Any]:
        if not path.is_file() or not self._allowed(path):
            return {"path": str(path), "available": False}
        size = path.stat().st_size
        if path.suffix.lower() not in _EXTENSIONS:
            return {"path": str(path), "size_bytes": size, "available": False}
        if size > _FILE_LIMIT or self.bytes_written + size > _TOTAL_LIMIT:
            self.data["issues"].append(
                f"Evidence file exceeds the report bound: {path.name} ({size} bytes)"
            )
            return {"path": str(path), "size_bytes": size, "omitted": "size limit"}
        filename = f"{len(self._files):04d}-{path.name}"
        destination = self.directory / "artifacts" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        self.bytes_written += size
        record = {
            "artifact": f"artifacts/{filename}",
            "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            "size_bytes": size,
        }
        self._files.append(record)
        self.data["artifacts"].append(
            {
                "path": record["artifact"],
                "role": role,
                "label": path.name,
                "media_type": record["media_type"],
                "size_bytes": size,
            }
        )
        return dict(record)

    def _array(self, value: Any, role: str) -> dict[str, Any]:
        import numpy as np

        array = np.asarray(value)
        result: dict[str, Any] = {"shape": list(array.shape), "dtype": str(array.dtype)}
        if array.dtype.hasobject:
            result["omitted"] = "object arrays are not serialized"
            self.data["issues"].append("An object array could not be recorded")
            return result
        flat = array.reshape(-1)
        result["preview"] = self.serialize(flat[:_PREVIEW_LIMIT].tolist(), role)
        if flat.size <= _PREVIEW_LIMIT:
            result["values"] = self.serialize(array.tolist(), role)
            return result
        if (
            array.nbytes + 1024 > _FILE_LIMIT
            or self.bytes_written + array.nbytes + 1024 > _TOTAL_LIMIT
        ):
            result["omitted"] = "size limit"
            self.data["issues"].append(f"Array exceeds evidence bound: {array.shape}")
            return result
        filename = f"array-{len(self.data['artifacts']):04d}.npy"
        path = self.directory / "artifacts" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array, allow_pickle=False)
        size = path.stat().st_size
        self.bytes_written += size
        result["artifact"] = f"artifacts/{filename}"
        self.data["artifacts"].append(
            {
                "path": result["artifact"],
                "role": role,
                "label": filename,
                "media_type": "application/octet-stream",
                "size_bytes": size,
            }
        )
        return result

    def serialize(self, value: Any, role: str, depth: int = 0) -> Any:
        if depth > 16:
            self.data["issues"].append("Evidence nesting exceeded the report bound")
            return {"omitted": "nesting limit"}
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else str(value)
        if isinstance(value, Path):
            return self._file(value, role) if value.is_file() else str(value)
        if isinstance(value, str):
            if len(value) < 4096 and "\n" not in value:
                try:
                    path = Path(value)
                    if (
                        path.suffix.lower() in _EXTENSIONS
                        and path.is_file()
                        and self._allowed(path)
                    ):
                        return self._file(path, role)
                except OSError:
                    pass
            if len(value) > 65536:
                self.data["issues"].append("A text observation was truncated")
                return value[:65536] + "\n[truncated]"
            return value
        if isinstance(value, dict):
            return {str(key): self.serialize(item, role, depth + 1) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            if len(value) > 1024:
                try:
                    import numpy as np

                    array = np.asarray(value)
                    if array.dtype.kind in "biufc":
                        return self._array(array, role)
                except (ImportError, TypeError, ValueError):
                    pass
                self.data["issues"].append("A sequence was truncated to the report bound")
                return [self.serialize(item, role, depth + 1) for item in value[:1024]]
            return [self.serialize(item, role, depth + 1) for item in value]
        if type(value).__module__.startswith("numpy"):
            return (
                self._array(value, role)
                if hasattr(value, "shape") and value.shape
                else self.serialize(value.item(), role, depth + 1)
            )
        if type(value).__module__.startswith("torch") and hasattr(value, "detach"):
            return self._array(value.detach().float().cpu().numpy(), role)
        self.data["issues"].append(f"Unsupported evidence type: {type(value).__name__}")
        return {"unavailable_type": type(value).__name__}

    def record(self, name: str, value: Any) -> None:
        self._prepare()
        snapshot = self.serialize(value, name)
        self.data["observations"].append({"name": name, "value": snapshot})
        if name not in {
            "schema_version",
            "family",
            "case",
            "source_revision",
            "nodeid",
            "status",
            "failure",
            "failure_stage",
            "duration_seconds",
            "evidence_status",
            "environment",
            "repro",
            "workflow_run_attempt",
            "checks",
            "timing",
            "artifacts",
            "observations",
            "issues",
        }:
            previous = self.data.get(name)
            self.data[name] = (
                {**previous, **snapshot}
                if name in {"inputs", "checkpoint", "thresholds"}
                and isinstance(previous, dict)
                and isinstance(snapshot, dict)
                else snapshot
            )

    def finish(self, status: str, *, failure: str = "") -> None:
        self._prepare()
        self.data["status"] = status
        if failure:
            self.data["failure"] = {"message": failure[:32768]}
            self.data["failure_stage"] = self.data["failure_stage"] or self.stage
        self.data["duration_seconds"] = time.monotonic() - self.started
        self.data["evidence_status"] = "partial" if self.data["issues"] else "recorded"
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.directory / ".evidence.tmp"

        def encoded() -> bytes:
            return (
                json.dumps(self.data, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
            ).encode("utf-8")

        payload = encoded()
        if len(payload) > _JSON_LIMIT:
            self.data["issues"].append("Detailed evidence was truncated to the JSON size bound")
            self.data["evidence_status"] = "partial"
            failed = [check for check in self.data["checks"] if check.get("status") == "failed"]
            passed = [check for check in self.data["checks"] if check.get("status") != "failed"]
            self.data["checks"] = failed + passed[-32:]
            self.data["observations"] = self.data["observations"][-8:]
            self.data = _bounded_preview(self.data)
            payload = encoded()
            if len(payload) > _JSON_LIMIT:
                self.data = {
                    "schema_version": 1,
                    "family": self.data["family"],
                    "case": self.data["case"],
                    "source_revision": self.data["source_revision"],
                    "status": status,
                    "workflow_run_attempt": self.data.get("workflow_run_attempt"),
                    "evidence_status": "partial",
                    "failure_stage": self.data.get("failure_stage"),
                    "failure": {"message": failure[:512]},
                    "checks": [
                        {
                            "status": "failed",
                            "expression": str(check.get("expression", ""))[:128],
                            "explanation": str(check.get("explanation", ""))[:256],
                        }
                        for check in failed[:1]
                    ],
                    "issues": [
                        "Detailed evidence exceeded the JSON size bound; raw artifacts remain in the testcase directory"
                    ],
                }
                payload = encoded()
            if len(payload) > _JSON_LIMIT:
                raise ValueError("evidence identity exceeds the JSON size bound")
        temporary.write_bytes(payload)
        temporary.replace(self.directory / "evidence.json")
        from tools.e2e_report import render_case

        (self.directory / "report.html").write_text(
            render_case(self.data, self.directory), encoding="utf-8"
        )


def evidence_enabled() -> bool:
    return _ACTIVE.get() is not None


def record_evidence(name: str, value: Any) -> Any:
    """Record a family-selected value without replacing or interpreting it."""
    recorder = _ACTIVE.get()
    if recorder is not None:
        try:
            recorder.record(name, value)
        except Exception as error:
            recorder.data["issues"].append(
                f"Could not record {name}: {type(error).__name__}: {error}"
            )
    return value


@contextmanager
def evidence_stage(name: str):
    """Record stage duration and failure location, preserving every exception."""
    recorder = _ACTIVE.get()
    if recorder is None:
        yield
        return
    previous, started = recorder.stage, time.monotonic()
    recorder.stage = name
    status = "passed"
    try:
        yield
    except BaseException as error:
        status = "failed"
        recorder.data["failure_stage"] = name
        if isinstance(error, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
            record_evidence(
                "subprocess_failure",
                {
                    "command": error.cmd,
                    "stdout": error.stdout.decode("utf-8", errors="replace")
                    if isinstance(error.stdout, bytes)
                    else error.stdout,
                    "stderr": error.stderr.decode("utf-8", errors="replace")
                    if isinstance(error.stderr, bytes)
                    else error.stderr,
                },
            )
        raise
    finally:
        recorder.data["timing"].append(
            {"stage": name, "seconds": time.monotonic() - started, "status": status}
        )
        recorder.stage = previous


@pytest.fixture(autouse=True)
def _capture_e2e_evidence(request):
    root_value = os.environ.get("TRTMC_E2E_ARTIFACT_DIR")
    name = getattr(request.node, "originalname", "") or ""
    parameters = getattr(getattr(request.node, "callspec", None), "params", {})
    if not root_value or not re.fullmatch(r"test_.*e2e", name) or "case_name" not in parameters:
        yield
        return
    repository = Path(str(request.config.rootpath)).resolve()
    path = Path(str(request.node.path)).resolve()
    try:
        relative = path.relative_to(repository / "families")
    except ValueError:
        yield
        return
    family, case = relative.parts[0], str(parameters["case_name"])
    requested = {
        value.strip()
        for raw in request.config.getoption("--e2e-testcase", default=[]) or []
        for value in str(raw).split(",")
        if value.strip()
    }
    if requested and case not in requested:
        # Selection and execution remain the family's responsibility. Only
        # the requested testcase may create or replace its evidence here.
        yield
        return
    if not _SAFE_NAME.fullmatch(family) or not _SAFE_NAME.fullmatch(case):
        raise ValueError("evidence family and case must be safe path components")
    evidence_root = Path(root_value) / "evidence"
    directory = evidence_root / family / case
    if evidence_root.is_symlink() or directory.parent.is_symlink():
        raise ValueError("evidence root must not be a symlink")
    recorder = Evidence(
        directory,
        family=family,
        case=case,
        source_revision=_revision(repository),
        roots=(repository / "families" / family / "tests",),
        nodeid=request.node.nodeid,
    )
    recorder.request = request
    recorder.data["environment"] = _environment()
    attempt = os.environ.get("TRTMC_E2E_WORKFLOW_ATTEMPT", "")
    if attempt.isdecimal() and int(attempt) > 0:
        recorder.data["workflow_run_attempt"] = int(attempt)
    recorder.data["repro"] = {
        "command": shlex.join(
            ["python", "-m", "pytest", request.node.nodeid, "--e2e-testcase", case, "-q"]
        ),
        "requirements": "Use the matching checkpoint, family dependencies, TRTMC_BINARY and TRTMC_RUNTIME_ROOT.",
    }
    request.node._trtmc_evidence = recorder
    token = _ACTIVE.set(recorder)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def _report_evidence_error(item, report, recorder: Evidence, operation: str, error: Exception):
    message = f"Could not {operation} evidence: {type(error).__name__}: {error}"
    recorder.data["issues"].append(message)
    recorder.data["evidence_status"] = "partial"
    report.sections.append(("evidence", message))
    report.user_properties.append(("trtmc_evidence_error", message))
    terminal = item.config.pluginmanager.get_plugin("terminalreporter")
    if terminal is not None:
        try:
            terminal.write_line(f"[evidence] {item.nodeid}: {message}", yellow=True)
        except Exception:
            # The report section and JUnit property retain the diagnostic even
            # when a terminal sink is unavailable. Keep the test outcome intact.
            pass


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    setattr(item, f"_trtmc_{report.when}_report", report)
    recorder = getattr(item, "_trtmc_evidence", None)
    if (
        recorder is not None
        and report.failed
        and call.excinfo is not None
        and call.excinfo.errisinstance(AssertionError)
    ):
        recorder.data["checks"].append(
            {
                "expression": str(call.excinfo.value)[:4096],
                "explanation": str(report.longrepr)[:32768],
                "status": "failed",
            }
        )
    if recorder is not None and report.when == "teardown":
        reports = [
            getattr(item, f"_trtmc_{phase}_report", None) for phase in ("setup", "call", "teardown")
        ]
        failures = [value for value in reports if value is not None and value.failed]
        skipped = any(value is not None and value.skipped for value in reports)
        status = (
            "error"
            if any(value.when != "call" for value in failures)
            else "failed"
            if failures
            else "skipped"
            if skipped
            else "error"
            if getattr(item, "_trtmc_call_report", None) is None
            else "passed"
        )
        if status == "skipped" and not recorder.data["observations"]:
            return
        # Logs alone must not prepare or replace a skipped testcase directory.
        # Pytest repeats earlier phase sections; keep each named stream once.
        sections = {
            name: content
            for value in reports
            if value is not None
            for name, content in value.sections
        }
        if sections:
            try:
                recorder.record(
                    "captured_output",
                    [
                        {
                            "stream": name,
                            "text": content[-12000:],
                            "truncated": len(content) > 12000,
                        }
                        for name, content in sections.items()
                    ],
                )
            except Exception as error:
                _report_evidence_error(item, report, recorder, "record captured output for", error)
        if failures and recorder.data["failure_stage"] is None:
            recorder.data["failure_stage"] = failures[0].when
        try:
            recorder.finish(status, failure="\n".join(str(value.longrepr) for value in failures))
        except Exception as error:
            _report_evidence_error(item, report, recorder, "write", error)


def pytest_assertion_pass(item, lineno, orig, expl):
    recorder = _ACTIVE.get()
    if recorder is None:
        return
    if len(recorder.data["checks"]) < _CHECK_LIMIT:
        recorder.data["checks"].append(
            {
                "line": lineno,
                "expression": orig[:4096],
                "explanation": expl[:8192],
                "status": "passed",
            }
        )
    elif "Assertion detail limit reached" not in recorder.data["issues"]:
        recorder.data["issues"].append("Assertion detail limit reached")
