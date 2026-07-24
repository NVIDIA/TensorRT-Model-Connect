#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run and source-bind the frozen dynamic-memory test manifest.

The implementation plan names both the full CTest baseline and the focused
``dynamic_memory`` CTest/pytest manifests.  This producer owns those exact
commands so a review receipt cannot silently omit a suite, reuse old output,
or claim a dirty tree as the frozen release candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET


SCHEMA = "trtmc.dynamic-memory-test-manifest/v2"
PYTEST_PATHS = (
    "tests/builder",
    "tests/tools",
    "tests/e2e/test_native_dynamic_memory_graph.py",
)
BUILD_ARTIFACTS = {
    "trtmc": Path("trtmc"),
    "benchmark_worker": Path("trtmc_benchmark_worker"),
    "core": Path("libtrtmc_core.so"),
    "trt_backend": Path("libtrtmc_backend_trt.so"),
    "runtime_kv_plugin": Path("libtrtmc_trt_plugins.so"),
    "model_qwen": Path("models/qwen/libtrtmc_model_qwen.so"),
    "model_llama": Path("models/llama/libtrtmc_model_llama.so"),
    "qualify": Path("trtmc_dynamic_memory_qualify"),
    "nvrtc_optional_output_regression": Path(
        "trtmc_nvrtc_optional_output_regression"
    ),
    "surfaces": Path("trtmc_dynamic_memory_surfaces"),
}


class ManifestError(RuntimeError):
    """The test manifest was incomplete, failed, or changed source."""


def _pytest_environment_overrides(build_dir: Path) -> dict[str, str]:
    """Bind every pytest command to the plugin from this exact build tree."""

    return {
        "TRTMC_BENCH_WORKER": str(build_dir / "trtmc_benchmark_worker"),
        "TRTMC_TRT_PLUGIN_LIBRARY": str(
            build_dir / "libtrtmc_trt_plugins.so"
        ),
    }


def _load_boundary_module() -> Any:
    path = Path(__file__).with_name("qualify_native_dynamic_memory.py")
    spec = importlib.util.spec_from_file_location("_trtmc_dynamic_memory_manifest_boundary", path)
    if spec is None or spec.loader is None:
        raise ManifestError(f"cannot load source-state helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _open_file_identity(
    path: Path,
    *,
    artifact_key: str,
    relative_path: Path,
) -> dict[str, Any]:
    """Hash one open file descriptor and reject identity changes during read."""

    try:
        canonical = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise ManifestError(
            f"build artifact {artifact_key!r} cannot be resolved: {path}: {error}"
        ) from error
    try:
        with canonical.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
                raise ManifestError(
                    f"build artifact {artifact_key!r} is not a non-empty regular file: {canonical}"
                )
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise ManifestError(
            f"build artifact {artifact_key!r} cannot be read: {canonical}: {error}"
        ) from error
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise ManifestError(
            f"build artifact {artifact_key!r} changed while it was hashed: {canonical}"
        )
    return {
        "artifact_key": artifact_key,
        "relative_path": relative_path.as_posix(),
        "path": str(canonical),
        "st_dev": before.st_dev,
        "st_ino": before.st_ino,
        "size_bytes": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "mode": stat.S_IMODE(before.st_mode),
        "sha256": digest.hexdigest(),
    }


def _cmake_cache_identity(repo_root: Path, build_dir: Path) -> dict[str, Any]:
    cache = build_dir / "CMakeCache.txt"
    identity = _open_file_identity(
        cache,
        artifact_key="cmake_cache",
        relative_path=Path("CMakeCache.txt"),
    )
    try:
        lines = cache.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as error:
        raise ManifestError(f"cannot read CMake cache {cache}: {error}") from error
    prefixes = ("CMAKE_HOME_DIRECTORY:INTERNAL=", "CMAKE_HOME_DIRECTORY:STATIC=")
    homes = [line.split("=", maxsplit=1)[1] for line in lines if line.startswith(prefixes)]
    if len(homes) != 1:
        raise ManifestError("CMakeCache.txt must contain exactly one CMAKE_HOME_DIRECTORY")
    try:
        configured_source = Path(homes[0]).expanduser().resolve(strict=True)
    except OSError as error:
        raise ManifestError(
            f"CMakeCache.txt source directory cannot be resolved: {homes[0]}: {error}"
        ) from error
    if configured_source != repo_root:
        raise ManifestError(
            "CMake build directory was configured from a different source tree: "
            f"expected {repo_root}, got {configured_source}"
        )
    identity["configured_source"] = str(configured_source)
    return identity


def _build_artifact_identities(build_dir: Path) -> dict[str, dict[str, Any]]:
    cache = build_dir / "CMakeCache.txt"
    try:
        cache_lines = cache.read_text(
            encoding="utf-8", errors="strict"
        ).splitlines()
    except (OSError, UnicodeError) as error:
        raise ManifestError(
            f"cannot read TensorRT ABI from CMake cache: {error}"
        ) from error
    abi_prefix = "TRTMC_TRT_BACKEND_ABI:STRING="
    abi_values = [
        line.removeprefix(abi_prefix)
        for line in cache_lines
        if line.startswith(abi_prefix)
    ]
    if (
        len(abi_values) != 1
        or re.fullmatch(r"[0-9]+_[0-9]+", abi_values[0]) is None
    ):
        raise ManifestError(
            "CMakeCache.txt must select exactly one TensorRT backend ABI"
        )
    active_backend = Path(
        f"libtrtmc_backend_trt_{abi_values[0]}.so"
    )
    generic_backend = build_dir / BUILD_ARTIFACTS["trt_backend"]
    active_backend_path = build_dir / active_backend
    try:
        generic_canonical = generic_backend.resolve(strict=True)
        active_canonical = active_backend_path.resolve(strict=True)
        generic_stat = generic_backend.stat()
        active_stat = active_backend_path.stat()
    except OSError as error:
        raise ManifestError(
            f"cannot resolve active TensorRT backend alias: {error}"
        ) from error
    if (
        generic_canonical != active_canonical
        or generic_stat.st_dev != active_stat.st_dev
        or generic_stat.st_ino != active_stat.st_ino
    ):
        raise ManifestError(
            "active TensorRT backend alias is an independent artifact; "
            "the versioned runtime path must resolve to the generic backend"
        )
    artifact_paths = dict(BUILD_ARTIFACTS)
    artifact_paths["trt_backend"] = active_backend
    identities = {
        key: _open_file_identity(
            build_dir / relative,
            artifact_key=key,
            relative_path=relative,
        )
        for key, relative in artifact_paths.items()
    }
    inodes: dict[tuple[int, int], str] = {}
    for key, identity in identities.items():
        inode = (identity["st_dev"], identity["st_ino"])
        previous = inodes.get(inode)
        if previous is not None:
            raise ManifestError(
                f"distinct build artifacts resolve to one inode: {previous!r} and {key!r}"
            )
        inodes[inode] = key
    return identities


def load_and_validate_build_manifest(path: Path) -> dict[str, Any]:
    """Reopen a passed v2 manifest and every binary/log it source-binds."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"build manifest is unreadable: {path}: {error}") from error
    required_fields = {
        "schema_version",
        "repo_root",
        "build_dir",
        "python",
        "source_state_pre",
        "commands",
        "passed",
        "source_state_post",
        "source_state_unchanged",
        "cmake_cache",
        "build_artifacts",
        "build_artifacts_sha256",
        "clean_build_command_sha256",
    }
    if not isinstance(value, dict) or set(value) != required_fields:
        raise ManifestError("build manifest does not have the exact v2 field set")
    if value["schema_version"] != SCHEMA or value["passed"] is not True:
        raise ManifestError("build manifest is not a passed v2 report")
    if value["source_state_unchanged"] is not True:
        raise ManifestError("build manifest source state was not stable")
    source_pre = value["source_state_pre"]
    source_post = value["source_state_post"]
    if not isinstance(source_pre, dict) or not isinstance(source_post, dict):
        raise ManifestError("build manifest source-state evidence is malformed")
    if (
        source_pre.get("exact_head_gate_satisfied") is not True
        or source_post.get("exact_head_gate_satisfied") is not True
        or source_pre.get("git_head") != source_post.get("git_head")
        or source_pre.get("source_state_sha256") != source_post.get("source_state_sha256")
    ):
        raise ManifestError("build manifest is not bound to one clean exact HEAD")

    try:
        repo_root = Path(value["repo_root"]).expanduser().resolve(strict=True)
        build_dir = Path(value["build_dir"]).expanduser().resolve(strict=True)
    except (TypeError, OSError) as error:
        raise ManifestError(f"build manifest roots cannot be resolved: {error}") from error
    cmake_cache = _cmake_cache_identity(repo_root, build_dir)
    if value["cmake_cache"] != cmake_cache:
        raise ManifestError("build manifest CMake cache identity changed")

    artifacts = value["build_artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(BUILD_ARTIFACTS):
        raise ManifestError("build manifest has an incomplete artifact set")
    reopened_artifacts = _build_artifact_identities(build_dir)
    if artifacts != reopened_artifacts:
        raise ManifestError("build manifest artifact identity changed")
    if value["build_artifacts_sha256"] != _canonical_sha(reopened_artifacts):
        raise ManifestError("build manifest artifact aggregate hash mismatch")

    commands = value["commands"]
    try:
        python = Path(str(value["python"]))
        if (
            not python.is_absolute()
            or not python.is_file()
            or not os.access(python, os.X_OK)
        ):
            raise OSError("path is not an absolute executable file")
    except (TypeError, OSError) as error:
        raise ManifestError(
            f"build manifest Python executable cannot be resolved: {error}"
        ) from error
    expected_commands = _commands(build_dir, python)
    if (
        not isinstance(commands, list)
        or len(commands) != len(expected_commands)
    ):
        raise ManifestError(
            "build manifest must contain the complete ordered command set"
        )
    output_dir = path.resolve().parent
    dynamic_pytest_manifest: list[str] = []
    base_fields = {
        "label",
        "argv",
        "cwd",
        "environment_overrides",
        "returncode",
        "started_ns",
        "finished_ns",
        "stdout",
        "stdout_sha256",
        "stderr",
        "stderr_sha256",
        "manifest_entries",
        "manifest_count",
        "passed",
    }
    for index, ((expected_label, expected_argv), command) in enumerate(
        zip(expected_commands, commands, strict=True)
    ):
        if not isinstance(command, dict):
            raise ManifestError(
                f"build manifest command {index} is not an object"
            )
        expected_pytest_entries: list[str] | None = None
        if expected_label == "pytest_dynamic_memory":
            expected_pytest_entries = list(dynamic_pytest_manifest)
        elif expected_label == "pytest_graph_e2e":
            expected_pytest_entries = [
                entry
                for entry in dynamic_pytest_manifest
                if entry.startswith(
                    "tests/e2e/test_native_dynamic_memory_graph.py::"
                )
            ]
        expected_fields = set(base_fields)
        executed_argv = list(expected_argv)
        if expected_pytest_entries is not None:
            expected_fields.add("pytest_junit")
            executed_argv.append(
                f"--junitxml={output_dir / f'{expected_label}.junit.xml'}"
            )
        if set(command) != expected_fields:
            raise ManifestError(
                f"build manifest command {expected_label!r} has invalid fields"
            )
        expected_environment = (
            _pytest_environment_overrides(build_dir)
            if expected_label.startswith("pytest_")
            else {}
        )
        if (
            command.get("label") != expected_label
            or command.get("argv") != executed_argv
            or command.get("cwd") != str(repo_root)
            or command.get("environment_overrides")
            != expected_environment
            or command.get("returncode") != 0
            or command.get("passed") is not True
        ):
            raise ManifestError(
                f"build manifest command {expected_label!r} contract changed"
            )
        started_ns = command.get("started_ns")
        finished_ns = command.get("finished_ns")
        if (
            isinstance(started_ns, bool)
            or not isinstance(started_ns, int)
            or started_ns <= 0
            or isinstance(finished_ns, bool)
            or not isinstance(finished_ns, int)
            or finished_ns < started_ns
        ):
            raise ManifestError(
                f"build manifest command {expected_label!r} timestamps are invalid"
            )
        stdout_text = ""
        for stream_name in ("stdout", "stderr"):
            expected_stream = (
                output_dir / f"{expected_label}.{stream_name}.log"
            )
            stream_path = command.get(stream_name)
            stream_sha = command.get(f"{stream_name}_sha256")
            if (
                stream_path != str(expected_stream)
                or not isinstance(stream_sha, str)
                or _sha256(expected_stream) != stream_sha
            ):
                raise ManifestError(
                    f"build manifest command {expected_label!r} "
                    f"{stream_name} evidence changed"
                )
            if stream_name == "stdout":
                stdout_text = expected_stream.read_text(encoding="utf-8")
        entries = _manifest_entries(expected_label, stdout_text)
        if (
            command.get("manifest_entries") != entries
            or command.get("manifest_count") != len(entries)
        ):
            raise ManifestError(
                f"build manifest command {expected_label!r} manifest changed"
            )
        if expected_label == "pytest_manifest_dynamic_memory":
            dynamic_pytest_manifest = list(entries)
        if expected_pytest_entries is not None:
            junit = command.get("pytest_junit")
            junit_path = output_dir / f"{expected_label}.junit.xml"
            if (
                not isinstance(junit, dict)
                or set(junit) != {"path", "sha256", "outcomes"}
                or junit.get("path") != str(junit_path)
                or junit.get("sha256") != _sha256(junit_path)
            ):
                raise ManifestError(
                    f"build manifest command {expected_label!r} JUnit changed"
                )
            reopened_outcomes = _pytest_junit_outcomes(
                junit_path,
                repo_root=repo_root,
                expected_entries=expected_pytest_entries,
            )
            if (
                junit.get("outcomes") != reopened_outcomes
                or reopened_outcomes["passed"] is not True
            ):
                raise ManifestError(
                    f"build manifest command {expected_label!r} "
                    "JUnit outcomes changed"
                )
    build_command = commands[0]
    if value["clean_build_command_sha256"] != _canonical_sha(
        build_command
    ):
        raise ManifestError("build manifest clean-build receipt hash mismatch")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _commands(build_dir: Path, python: Path) -> list[tuple[str, list[str]]]:
    pytest_selection = [*PYTEST_PATHS]
    return [
        (
            "build",
            [
                "cmake",
                "--build",
                str(build_dir),
                "--clean-first",
                "-j",
            ],
        ),
        (
            "build_cpp_tests_and_qualifiers",
            [
                "cmake",
                "--build",
                str(build_dir),
                "-j",
                "--target",
                "trtmc_cpp_tests",
                "trtmc_dynamic_memory_qualify",
                "trtmc_dynamic_memory_surfaces",
                "trtmc_benchmark_worker",
            ],
        ),
        (
            "ctest_manifest_all",
            ["ctest", "--test-dir", str(build_dir), "-N"],
        ),
        (
            "ctest_all",
            [
                "ctest",
                "--test-dir",
                str(build_dir),
                "--output-on-failure",
            ],
        ),
        (
            "ctest_manifest_dynamic_memory",
            [
                "ctest",
                "--test-dir",
                str(build_dir),
                "-N",
                "-L",
                "dynamic_memory",
            ],
        ),
        (
            "ctest_dynamic_memory",
            [
                "ctest",
                "--test-dir",
                str(build_dir),
                "-L",
                "dynamic_memory",
                "--output-on-failure",
            ],
        ),
        (
            "pytest_manifest_dynamic_memory",
            [
                str(python),
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-m",
                "dynamic_memory",
                *pytest_selection,
            ],
        ),
        (
            "pytest_dynamic_memory",
            [
                str(python),
                "-m",
                "pytest",
                "-q",
                "-m",
                "dynamic_memory",
                *pytest_selection,
            ],
        ),
        (
            "pytest_graph_e2e",
            [
                str(python),
                "-m",
                "pytest",
                "-q",
                "tests/e2e/test_native_dynamic_memory_graph.py",
            ],
        ),
    ]


def _manifest_entries(label: str, stdout: str) -> list[str]:
    if label.startswith("ctest_manifest_"):
        return re.findall(r"Test\s+#\d+:\s+(\S+)", stdout)
    if label == "pytest_manifest_dynamic_memory":
        return sorted(
            line.strip()
            for line in stdout.splitlines()
            if "::" in line and not line.lstrip().startswith("=")
        )
    return []


def _junit_node_id(
    testcase: ET.Element,
    *,
    repo_root: Path,
) -> str | None:
    classname = testcase.get("classname", "")
    name = testcase.get("name", "")
    if not classname or not name:
        return None
    parts = classname.split(".")
    module_length = 0
    module_path: Path | None = None
    for length in range(len(parts), 0, -1):
        candidate = repo_root.joinpath(*parts[:length]).with_suffix(".py")
        if candidate.is_file():
            module_length = length
            module_path = candidate
            break
    if module_path is None:
        return None
    components = [
        module_path.relative_to(repo_root).as_posix(),
        *parts[module_length:],
        name,
    ]
    return "::".join(components)


def _pytest_junit_outcomes(
    path: Path,
    *,
    repo_root: Path,
    expected_entries: Sequence[str],
) -> dict[str, Any]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise ManifestError(f"pytest JUnit evidence is invalid: {path}: {error}") from error
    expected = set(expected_entries)
    if not expected:
        raise ManifestError("pytest qualification manifest is empty")
    selected: dict[str, str] = {}
    collection_skips: list[dict[str, str]] = []
    unexpected: list[str] = []
    for testcase in root.iter("testcase"):
        node_id = _junit_node_id(testcase, repo_root=repo_root)
        status = "passed"
        if testcase.find("failure") is not None:
            status = "failed"
        elif testcase.find("error") is not None:
            status = "error"
        elif testcase.find("skipped") is not None:
            status = "skipped"
        if node_id is None:
            skipped = testcase.find("skipped")
            if skipped is not None:
                collection_skips.append(
                    {
                        "classname": testcase.get("classname", ""),
                        "name": testcase.get("name", ""),
                        "message": skipped.get("message", ""),
                    }
                )
                continue
            unexpected.append(f"{testcase.get('classname', '')}::{testcase.get('name', '')}")
            continue
        if node_id not in expected:
            unexpected.append(node_id)
            continue
        if node_id in selected:
            raise ManifestError(f"pytest JUnit repeats selected test {node_id!r}")
        selected[node_id] = status
    missing = sorted(expected - selected.keys())
    selected_skips = sorted(node_id for node_id, status in selected.items() if status == "skipped")
    selected_failures = sorted(
        node_id for node_id, status in selected.items() if status in {"failed", "error"}
    )
    passed = not missing and not unexpected and not selected_skips and not selected_failures
    return {
        "expected_count": len(expected),
        "observed_count": len(selected),
        "passed_count": sum(status == "passed" for status in selected.values()),
        "selected_skips": selected_skips,
        "selected_failures": selected_failures,
        "missing_selected_tests": missing,
        "unexpected_tests": sorted(unexpected),
        "unselected_collection_skips": collection_skips,
        "passed": passed,
    }


def _run_one(
    label: str,
    argv: Sequence[str],
    *,
    repo_root: Path,
    output_dir: Path,
    environment_overrides: Mapping[str, str] | None = None,
    expected_pytest_entries: Sequence[str] | None = None,
) -> dict[str, Any]:
    stdout_path = output_dir / f"{label}.stdout.log"
    stderr_path = output_dir / f"{label}.stderr.log"
    executed_argv = list(argv)
    junit_path: Path | None = None
    if expected_pytest_entries is not None:
        junit_path = output_dir / f"{label}.junit.xml"
        executed_argv.append(f"--junitxml={junit_path}")
    started_ns = time.time_ns()
    environment = os.environ.copy()
    environment.update(environment_overrides or {})
    completed = subprocess.run(
        executed_argv,
        cwd=repo_root,
        check=False,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    finished_ns = time.time_ns()
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    entries = _manifest_entries(label, completed.stdout)
    pytest_outcomes = (
        _pytest_junit_outcomes(
            junit_path,
            repo_root=repo_root,
            expected_entries=expected_pytest_entries,
        )
        if junit_path is not None
        else None
    )
    passed = completed.returncode == 0 and (pytest_outcomes is None or pytest_outcomes["passed"])
    result = {
        "label": label,
        "argv": executed_argv,
        "cwd": str(repo_root),
        "environment_overrides": dict(environment_overrides or {}),
        "returncode": completed.returncode,
        "started_ns": started_ns,
        "finished_ns": finished_ns,
        "stdout": str(stdout_path),
        "stdout_sha256": _sha256(stdout_path),
        "stderr": str(stderr_path),
        "stderr_sha256": _sha256(stderr_path),
        "manifest_entries": entries,
        "manifest_count": len(entries),
        "passed": passed,
    }
    if junit_path is not None:
        result["pytest_junit"] = {
            "path": str(junit_path),
            "sha256": _sha256(junit_path),
            "outcomes": pytest_outcomes,
        }
    return result


def capture(
    *,
    repo_root: Path,
    build_dir: Path,
    python: Path,
    output_dir: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    build_dir = build_dir.resolve()
    # Preserve a virtual environment's interpreter symlink. Resolving
    # ``/opt/venv/bin/python`` to the system executable bypasses the adjacent
    # pyvenv.cfg and silently drops the venv site-packages.
    python = python.absolute()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    boundary = _load_boundary_module()
    source_pre = boundary.source_state_provenance(
        repo_root,
        Path(__file__).resolve(),
        output_dir,
        label="pre",
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "repo_root": str(repo_root),
        "build_dir": str(build_dir),
        "python": str(python),
        "source_state_pre": source_pre,
        "commands": [],
        "passed": False,
    }
    report_path = output_dir / "test-manifest-report.json"
    if not source_pre["exact_head_gate_satisfied"]:
        _write_json(report_path, report)
        raise ManifestError("test-manifest qualification requires a clean exact HEAD")
    try:
        _cmake_cache_identity(repo_root, build_dir)
    except ManifestError:
        _write_json(report_path, report)
        raise

    failed_label: str | None = None
    dynamic_pytest_manifest: list[str] = []
    for label, argv in _commands(build_dir, python):
        environment_overrides = (
            _pytest_environment_overrides(build_dir)
            if label.startswith("pytest_")
            else None
        )
        expected_pytest_entries = None
        if label == "pytest_dynamic_memory":
            expected_pytest_entries = dynamic_pytest_manifest
        elif label == "pytest_graph_e2e":
            expected_pytest_entries = [
                entry
                for entry in dynamic_pytest_manifest
                if entry.startswith("tests/e2e/test_native_dynamic_memory_graph.py::")
            ]
        result = _run_one(
            label,
            argv,
            repo_root=repo_root,
            output_dir=output_dir,
            environment_overrides=environment_overrides,
            expected_pytest_entries=expected_pytest_entries,
        )
        if label == "pytest_manifest_dynamic_memory":
            dynamic_pytest_manifest = list(result["manifest_entries"])
        report["commands"].append(result)
        _write_json(report_path, report)
        if not result["passed"]:
            failed_label = label
            break

    source_post = boundary.source_state_provenance(
        repo_root,
        Path(__file__).resolve(),
        output_dir,
        label="post",
    )
    unchanged = source_pre["source_state_sha256"] == source_post["source_state_sha256"]
    report["source_state_post"] = source_post
    report["source_state_unchanged"] = unchanged
    artifact_error: ManifestError | None = None
    if (
        failed_label is None
        and unchanged
        and source_post["exact_head_gate_satisfied"]
        and all(command["passed"] for command in report["commands"])
    ):
        build_commands = [command for command in report["commands"] if command["label"] == "build"]
        if len(build_commands) != 1:
            artifact_error = ManifestError("test manifest requires exactly one clean build command")
        else:
            try:
                report["cmake_cache"] = _cmake_cache_identity(
                    repo_root,
                    build_dir,
                )
                report["build_artifacts"] = _build_artifact_identities(build_dir)
                report["build_artifacts_sha256"] = _canonical_sha(report["build_artifacts"])
                report["clean_build_command_sha256"] = _canonical_sha(build_commands[0])
            except ManifestError as error:
                artifact_error = error
    report["passed"] = bool(
        failed_label is None
        and unchanged
        and source_post["exact_head_gate_satisfied"]
        and all(command["passed"] for command in report["commands"])
        and artifact_error is None
        and set(report.get("build_artifacts", {})) == set(BUILD_ARTIFACTS)
    )
    _write_json(report_path, report)
    if failed_label is not None:
        failed = report["commands"][-1]
        raise ManifestError(f"test-manifest command failed: {failed_label}; see {failed['stderr']}")
    if not report["passed"]:
        if artifact_error is not None:
            raise artifact_error
        raise ManifestError("source state changed while executing the test manifest")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = capture(
        repo_root=args.repo_root,
        build_dir=args.build_dir,
        python=args.python,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "report": str(args.output_dir.resolve() / "test-manifest-report.json"),
                "commands": len(report["commands"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ManifestError as error:
        print(f"capture_dynamic_memory_test_manifest: {error}", file=sys.stderr)
        raise SystemExit(1)
