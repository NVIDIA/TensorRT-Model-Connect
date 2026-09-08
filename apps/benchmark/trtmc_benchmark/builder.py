# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build missing benchmark bundles through the public build command."""

from __future__ import annotations

import os
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tensorrt_model_connect.build_cli import _resolve_model
import tensorrt_model_connect

from .types import BenchmarkError, ModelDescriptor, ResolvedCase


@dataclass(frozen=True)
class BundlePreparation:
    model: str
    status: str
    bundle: Path
    model_dir: Path | None = None
    build_time_s: float | None = None
    command: tuple[str, ...] = ()
    stdout_log: Path | None = None
    stderr_log: Path | None = None
    build_identity: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "status": self.status,
            "bundle": str(self.bundle),
            "model_dir": str(self.model_dir) if self.model_dir else None,
            "build_time_s": self.build_time_s,
            "command": list(self.command),
            "stdout_log": str(self.stdout_log) if self.stdout_log else None,
            "stderr_log": str(self.stderr_log) if self.stderr_log else None,
            "build_identity": self.build_identity,
            "included_in_performance_metrics": False,
        }


@dataclass(frozen=True)
class _BuildPlan:
    model: ModelDescriptor
    model_dir: Path
    bundle: Path
    command: tuple[str, ...]
    timeout_s: int
    identity: str | None


class BundleBuilder:
    """A managed bundle cache with a benchmark-owned build identity receipt."""

    def __init__(
        self,
        cache_root: Path | None = None,
        *,
        model_dirs: Mapping[str, Path] | None = None,
    ) -> None:
        self.cache_root = (cache_root or default_bundle_cache()).expanduser().resolve()
        self.model_dirs = {
            name: path.expanduser().resolve() for name, path in (model_dirs or {}).items()
        }

    def provisional_path(self, model: ModelDescriptor) -> Path:
        return self.cache_root / model.name / model.bundle_name

    def prepare(
        self,
        cases: Iterable[ResolvedCase],
        *,
        allow_build: bool,
        rebuild: bool,
        dry_run: bool,
    ) -> tuple[tuple[ResolvedCase, ...], tuple[BundlePreparation, ...]]:
        resolved = tuple(cases)
        groups: dict[tuple[Path, Path], list[ResolvedCase]] = {}
        for case in resolved:
            key = (case.model.manifest_path, case.bundle_path.expanduser().resolve())
            groups.setdefault(key, []).append(case)

        replacements: dict[tuple[Path, Path], Path] = {}
        records: list[BundlePreparation] = []
        for key, grouped in groups.items():
            path, record = self._prepare_group(
                grouped,
                key[1],
                allow_build=allow_build,
                rebuild=rebuild,
                dry_run=dry_run,
            )
            replacements[key] = path
            records.append(record)

        updated = tuple(
            case.with_values(
                bundle_path=replacements[
                    (case.model.manifest_path, case.bundle_path.expanduser().resolve())
                ]
            )
            for case in resolved
        )
        return updated, tuple(records)

    def _prepare_group(
        self,
        cases: Sequence[ResolvedCase],
        requested: Path,
        *,
        allow_build: bool,
        rebuild: bool,
        dry_run: bool,
    ) -> tuple[Path, BundlePreparation]:
        model = cases[0].model
        managed = _is_relative_to(requested, self.cache_root)
        if requested.is_file() and not managed and not rebuild:
            return requested, BundlePreparation(model.name, "reused", requested)
        if requested.is_file() and not managed:
            raise BenchmarkError(
                f"--rebuild cannot overwrite explicit bundle {requested}; "
                "omit --bundle to rebuild the managed cache"
            )
        if not managed:
            raise BenchmarkError(f"explicit bundle does not exist: {requested}")
        if not requested.is_file() and not allow_build:
            raise BenchmarkError(f"bundle for {model.name} is unavailable and --no-build was set")

        plan = self._plan(model, cases)
        if not rebuild and _matches_receipt(plan):
            return plan.bundle, BundlePreparation(
                model.name, "reused", plan.bundle, build_identity=plan.identity
            )
        if not allow_build:
            raise BenchmarkError(
                f"managed bundle for {model.name} has no matching immutable build identity; "
                "remove --no-build to rebuild, or provide an explicit --bundle"
            )
        if dry_run:
            return plan.bundle, BundlePreparation(
                model.name,
                "would_build",
                plan.bundle,
                model_dir=plan.model_dir,
                command=plan.command,
            )
        return plan.bundle, self._build(plan)

    def _plan(self, model: ModelDescriptor, cases: Sequence[ResolvedCase]) -> _BuildPlan:
        explicit = (
            self.model_dirs.get(model.name)
            or self.model_dirs.get(model.family)
            or self.model_dirs.get("")
        )
        if explicit is not None and not explicit.is_dir():
            raise BenchmarkError(f"model directory does not exist: {explicit}")
        if explicit is None and not model.hf_id:
            raise BenchmarkError(f"{model.name} has no hf_id; pass --model-dir")
        try:
            model_dir = _resolve_model(
                str(explicit) if explicit is not None else model.hf_id,
                None if explicit is not None else model.hf_revision or None,
            ).resolve()
        except Exception as error:
            raise BenchmarkError(
                f"cannot materialize checkpoint {model.hf_id!r} for {model.name}: {error}"
            ) from error
        bundle = self.provisional_path(model)
        command = _build_command(model, model_dir, bundle, cases)
        timeout = int(os.environ.get("TRTMC_BENCH_BUILD_TIMEOUT_S", "3600"))
        if timeout <= 0:
            raise BenchmarkError("TRTMC_BENCH_BUILD_TIMEOUT_S must be positive")
        identity = _build_identity(model, model_dir, command) if explicit is None else None
        return _BuildPlan(model, model_dir, bundle, command, timeout, identity)

    def _build(self, plan: _BuildPlan) -> BundlePreparation:
        plan.bundle.parent.mkdir(parents=True, exist_ok=True)
        stdout_log = plan.bundle.parent / "build.stdout.log"
        stderr_log = plan.bundle.parent / "build.stderr.log"
        descriptor, raw_temporary = tempfile.mkstemp(
            prefix=".trtmc-bench-", suffix=".bundle", dir=plan.bundle.parent
        )
        os.close(descriptor)
        temporary = Path(raw_temporary)
        temporary.unlink()
        command = list(plan.command)
        command[command.index("-o") + 1] = str(temporary)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=plan.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            _write(stdout_log, _text(error.stdout))
            _write(stderr_log, _text(error.stderr) or "build timed out\n")
            temporary.unlink(missing_ok=True)
            raise BenchmarkError(
                f"bundle build for {plan.model.name} timed out; see {stderr_log}",
                stage="build",
                domain="benchmark",
                code="bundle_build_timeout",
                artifacts=(("stdout", stdout_log), ("stderr", stderr_log)),
            ) from error
        elapsed = time.monotonic() - started
        _write(stdout_log, completed.stdout)
        _write(stderr_log, completed.stderr)
        if completed.returncode != 0 or not temporary.is_file():
            temporary.unlink(missing_ok=True)
            raise BenchmarkError(
                f"bundle build for {plan.model.name} failed with exit code "
                f"{completed.returncode}; see {stderr_log}",
                stage="build",
                domain="benchmark",
                code="bundle_build_failed",
                artifacts=(("stdout", stdout_log), ("stderr", stderr_log)),
            )
        os.replace(temporary, plan.bundle)
        _write_receipt(plan)
        return BundlePreparation(
            plan.model.name,
            "built",
            plan.bundle,
            model_dir=plan.model_dir,
            build_time_s=elapsed,
            command=plan.command,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            build_identity=plan.identity,
        )


def _source_digest(model: ModelDescriptor) -> str:
    specification = importlib.util.find_spec(f"families.{model.family}")
    if specification is None or not specification.origin:
        raise BenchmarkError(f"cannot identify builder sources for {model.family}")
    roots = (Path(tensorrt_model_connect.__file__).parent, Path(specification.origin).parent)
    digest = hashlib.sha256()
    for index, root in enumerate(roots):
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(root)
            if "tests" in relative.parts or "__pycache__" in relative.parts:
                continue
            digest.update(f"{index}/{relative.as_posix()}\0".encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _build_identity(model: ModelDescriptor, checkpoint: Path, command: Sequence[str]) -> str | None:
    # Hugging Face snapshot directories are immutable revisions. Arbitrary local
    # checkpoints are deliberately rebuilt instead of trusting file timestamps.
    if checkpoint.parent.name != "snapshots" or not re.fullmatch("[0-9a-f]{40}", checkpoint.name):
        return None
    versions = {}
    for package in ("tensorrt", "tensorrt-cu13", "torch", "numpy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    identity = {
        "manifest": hashlib.sha256(model.manifest_path.read_bytes()).hexdigest(),
        "checkpoint_revision": checkpoint.name,
        "command": list(command),
        "source": _source_digest(model),
        "python": sys.version,
        "packages": versions,
        "build_environment": {
            key: value for key, value in os.environ.items() if key.startswith("TRTMC_")
        },
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _receipt_path(bundle: Path) -> Path:
    return bundle.with_suffix(bundle.suffix + ".benchmark.json")


def _matches_receipt(plan: _BuildPlan) -> bool:
    # File identity/change metadata catches replacements and in-place writes,
    # including copies that preserve mtime. This is cache invalidation, not a
    # cryptographic guarantee about the serialized bundle's contents.
    if plan.identity is None or not plan.bundle.is_file():
        return False
    try:
        receipt = json.loads(_receipt_path(plan.bundle).read_text(encoding="utf-8"))
        stat = plan.bundle.stat()
        return receipt == {
            "schema_version": "trtmc.benchmark-build/v1",
            "identity": plan.identity,
            "bundle_size": stat.st_size,
            "bundle_mtime_ns": stat.st_mtime_ns,
            "bundle_ctime_ns": stat.st_ctime_ns,
            "bundle_device": stat.st_dev,
            "bundle_inode": stat.st_ino,
        }
    except (OSError, ValueError):
        return False


def _write_receipt(plan: _BuildPlan) -> None:
    path = _receipt_path(plan.bundle)
    if plan.identity is None:
        path.unlink(missing_ok=True)
        return
    stat = plan.bundle.stat()
    receipt = {
        "schema_version": "trtmc.benchmark-build/v1",
        "identity": plan.identity,
        "bundle_size": stat.st_size,
        "bundle_mtime_ns": stat.st_mtime_ns,
        "bundle_ctime_ns": stat.st_ctime_ns,
        "bundle_device": stat.st_dev,
        "bundle_inode": stat.st_ino,
    }
    # The receipt contains only digests and bundle metadata, not source paths or
    # environment values. It is published after the successfully built bundle.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(receipt, stream, sort_keys=True)
    os.replace(stream.name, path)


def default_bundle_cache() -> Path:
    configured = os.environ.get("TRTMC_BENCH_CACHE_DIR")
    if configured:
        return Path(configured)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "trtmc/bench/bundles"


def _build_command(
    model: ModelDescriptor,
    model_dir: Path,
    bundle: Path,
    cases: Sequence[ResolvedCase],
) -> tuple[str, ...]:
    settings = model.build_settings
    command = [
        sys.executable,
        "-m",
        "tensorrt_model_connect",
        "build",
        str(model_dir),
        "-o",
        str(bundle),
        "--task",
        model.task,
        "--precision",
        model.precision,
    ]
    flags = (
        ("max_sequence_length", "--max-sequence-length"),
        ("image_height", "--image-height"),
        ("image_width", "--image-width"),
        ("video_num_frames", "--video-num-frames"),
        ("max_batch_size", "--max-batch-size"),
        ("tensor_parallel_size", "--tensor-parallel-size"),
        ("context_parallel_size", "--context-parallel-size"),
        ("quantization", "--quantization"),
        ("backend", "--backend"),
    )
    for name, flag in flags:
        value = settings.get(name)
        if value is not None:
            command.extend((flag, str(value)))
    for layer in settings.get("fp32_layers", ()):
        command.extend(("--fp32-layer", str(int(layer))))
    if settings.get("dynamic_kv_cache", False):
        command.append("--dynamic-kv-cache")

    image_cases = [case for case in cases if case.operation == "generate_image"]
    if image_cases:
        heights = [int(case.request.get("height", 0)) for case in image_cases]
        widths = [int(case.request.get("width", 0)) for case in image_cases]
        batches = [int(case.request.get("batch_size", 1)) for case in image_cases]
        _replace_value(command, "--image-height", max(heights, default=0))
        _replace_value(command, "--image-width", max(widths, default=0))
        _replace_value(command, "--max-batch-size", max(batches, default=1))
    return tuple(command)


def _replace_value(command: list[str], flag: str, value: int) -> None:
    if value <= 0:
        return
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend((flag, str(value)))


def _write(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
