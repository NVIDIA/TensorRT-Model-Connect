# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT / "tools" / "capture_dynamic_memory_test_manifest.py"
)
SPEC = importlib.util.spec_from_file_location(
    "capture_dynamic_memory_test_manifest", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
capture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capture
SPEC.loader.exec_module(capture)

pytestmark = [pytest.mark.unit, pytest.mark.dynamic_memory]


def test_commands_build_excluded_cpp_tests_before_ctest() -> None:
    commands = capture._commands(
        Path("/repo/build-dynkv"),
        Path("/opt/venv/bin/python"),
    )
    labels = [label for label, _ in commands]
    assert labels[:4] == [
        "build",
        "build_cpp_tests_and_qualifiers",
        "ctest_manifest_all",
        "ctest_all",
    ]
    build_argv = commands[1][1]
    assert build_argv[:4] == [
        "cmake",
        "--build",
        "/repo/build-dynkv",
        "-j",
    ]
    assert build_argv[4:] == [
        "--target",
        "trtmc_cpp_tests",
        "trtmc_dynamic_memory_qualify",
        "trtmc_dynamic_memory_surfaces",
        "trtmc_benchmark_worker",
    ]


def test_capture_preserves_virtual_environment_python_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    build_dir = repo / "build-dynkv"
    build_dir.mkdir()
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    python_link = venv_bin / "python"
    python_link.symlink_to(Path(sys.executable))
    observed: list[Path] = []

    def commands(_build_dir: Path, python: Path) -> list[tuple[str, list[str]]]:
        observed.append(python)
        return []

    monkeypatch.setattr(capture, "_commands", commands)
    report = capture.capture(
        repo_root=repo,
        build_dir=build_dir,
        python=python_link,
        output_dir=repo / "artifacts" / "manifest",
    )

    assert report["passed"]
    assert observed == [python_link.absolute()]
    assert observed[0].is_symlink()


def _init_repo(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=path,
        check=True,
    )
    (path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=path, check=True)


def test_capture_records_exact_manifests_and_source_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    build_dir = repo / "build-dynkv"
    build_dir.mkdir()
    monkeypatch.setattr(
        capture,
        "_commands",
        lambda *_: [
            (
                "ctest_manifest_all",
                [
                    sys.executable,
                    "-c",
                    "print('  Test #1: one\\n  Test #2: two')",
                ],
            ),
            (
                "pytest_manifest_dynamic_memory",
                [
                    sys.executable,
                    "-c",
                    "print('tests/example.py::test_one')",
                ],
            ),
        ],
    )

    output_dir = repo / "artifacts" / "manifest"
    report = capture.capture(
        repo_root=repo,
        build_dir=build_dir,
        python=Path(sys.executable),
        output_dir=output_dir,
    )

    assert report["passed"] is True
    assert report["source_state_unchanged"] is True
    assert report["source_state_pre"]["exact_head_gate_satisfied"] is True
    assert report["commands"][0]["manifest_entries"] == ["one", "two"]
    assert report["commands"][0]["manifest_count"] == 2
    assert report["commands"][1]["manifest_entries"] == [
        "tests/example.py::test_one"
    ]
    assert report["commands"][0]["environment_overrides"] == {}
    assert report["commands"][1]["environment_overrides"] == {
        "TRTMC_BENCH_WORKER": str(
            (build_dir / "trtmc_benchmark_worker").resolve()
        )
    }
    persisted = json.loads(
        (output_dir / "test-manifest-report.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted["passed"] is True


def test_capture_rejects_dirty_source_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    monkeypatch.setattr(
        capture,
        "_commands",
        lambda *_: pytest.fail("dirty source must fail before commands"),
    )

    output_dir = repo / "artifacts" / "manifest"
    with pytest.raises(capture.ManifestError, match="clean exact HEAD"):
        capture.capture(
            repo_root=repo,
            build_dir=repo / "build-dynkv",
            python=Path(sys.executable),
            output_dir=output_dir,
        )
    report = json.loads(
        (output_dir / "test-manifest-report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["passed"] is False
    assert report["source_state_pre"]["exact_head_gate_satisfied"] is False


def test_capture_persists_failure_and_post_source_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    monkeypatch.setattr(
        capture,
        "_commands",
        lambda *_: [
            (
                "ctest_all",
                [sys.executable, "-c", "raise SystemExit(7)"],
            )
        ],
    )

    output_dir = repo / "artifacts" / "manifest"
    with pytest.raises(capture.ManifestError, match="ctest_all"):
        capture.capture(
            repo_root=repo,
            build_dir=repo / "build-dynkv",
            python=Path(sys.executable),
            output_dir=output_dir,
        )
    report = json.loads(
        (output_dir / "test-manifest-report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["passed"] is False
    assert report["commands"][0]["returncode"] == 7
    assert report["source_state_unchanged"] is True
