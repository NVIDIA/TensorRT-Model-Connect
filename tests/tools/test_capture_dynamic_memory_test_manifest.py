# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from tests.tools.dynamic_memory_manifest_fixture import (
    complete_command_receipts,
    seed_manifest_test_modules,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "capture_dynamic_memory_test_manifest.py"
SPEC = importlib.util.spec_from_file_location("capture_dynamic_memory_test_manifest", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
capture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capture
SPEC.loader.exec_module(capture)

pytestmark = [pytest.mark.unit, pytest.mark.dynamic_memory]


def test_clean_first_keeps_configure_time_registration_sources() -> None:
    """Configure-time sources must not be registered as cleanable build outputs."""

    forbidden_by_file = {
        "cmake/trtmc_pipeline_plugins.cmake": (
            'set_source_files_properties("${TRTMC_MODEL_PLUGIN_INDEX_SOURCE}" '
            "PROPERTIES GENERATED TRUE)",
            'set_source_files_properties("${_trtmc_generated_model_reg}" '
            "PROPERTIES GENERATED TRUE)",
        ),
        "cmake/trtmc_registration_manifest.cmake": (
            'set_source_files_properties("${generated_source}" '
            "PROPERTIES GENERATED TRUE)",
        ),
    }
    for relative_path, forbidden_entries in forbidden_by_file.items():
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        for forbidden in forbidden_entries:
            assert forbidden not in text


def test_commands_build_excluded_cpp_tests_before_ctest() -> None:
    commands = capture._commands(
        Path("/repo/build-dynkv"),
        Path("/opt/venv/bin/python"),
    )
    labels = [label for label, _ in commands]
    assert labels == [
        "build",
        "build_cpp_tests_and_qualifiers",
        "ctest_manifest_all",
        "ctest_all",
        "ctest_manifest_dynamic_memory",
        "ctest_dynamic_memory",
        "pytest_manifest_dynamic_memory",
        "pytest_dynamic_memory",
        "pytest_graph_e2e",
    ]
    assert tuple(labels) == capture.FIXED_COMMAND_LABELS
    assert commands[0][1] == [
        "cmake",
        "--build",
        "/repo/build-dynkv",
        "--clean-first",
        "-j",
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


def test_nvrtc_regression_probe_is_dependency_and_manifest_artifact() -> None:
    # The command contract stays at nine entries.  Building the already-listed
    # main qualifier must transitively rebuild the regression executable, and
    # the manifest then binds that exact binary by identity and SHA.
    commands = capture._commands(
        Path("/repo/build-dynkv"),
        Path("/opt/venv/bin/python"),
    )
    assert len(commands) == 9
    assert capture.BUILD_ARTIFACTS[
        "nvrtc_optional_output_regression"
    ] == Path("trtmc_nvrtc_optional_output_regression")

    cmake = (REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "if(TARGET trtmc_nvrtc_optional_output_regression)" in cmake
    dependency_block = cmake.split(
        "if(TARGET trtmc_nvrtc_optional_output_regression)", 1
    )[1].split("endif()", 1)[0]
    assert "add_dependencies(" in dependency_block
    assert "trtmc_dynamic_memory_qualify" in dependency_block
    assert "trtmc_nvrtc_optional_output_regression" in dependency_block


def test_capture_preserves_virtual_environment_python_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    build_dir = repo / "build-dynkv"
    _seed_build_tree(repo, build_dir)
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    python_link = venv_bin / "python"
    python_link.symlink_to(Path(sys.executable))
    observed: list[Path] = []

    def commands(_build_dir: Path, python: Path) -> list[tuple[str, list[str]]]:
        observed.append(python)
        return [
            ("build", [sys.executable, "-c", "pass"]),
            (
                "build_cpp_tests_and_qualifiers",
                [sys.executable, "-c", "pass"],
            ),
        ]

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
    (path / ".gitignore").write_text(
        "build-dynkv/\nartifacts/\n",
        encoding="utf-8",
    )
    seed_manifest_test_modules(path)
    subprocess.run(
        ["git", "add", "-A"],
        cwd=path,
        check=True,
    )
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=path, check=True)


def _seed_build_tree(repo: Path, build_dir: Path) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "CMakeCache.txt").write_text(
        (
            f"CMAKE_HOME_DIRECTORY:INTERNAL={repo.resolve()}\n"
            "TRTMC_TRT_BACKEND_ABI:STRING=11_2\n"
        ),
        encoding="utf-8",
    )
    for key, relative in capture.BUILD_ARTIFACTS.items():
        path = build_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{key}-artifact\n".encode())
        path.chmod(0o755)
    active_backend = build_dir / "libtrtmc_backend_trt_11_2.so"
    active_backend.symlink_to("libtrtmc_backend_trt.so")


def test_capture_records_exact_manifests_and_source_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    build_dir = repo / "build-dynkv"
    _seed_build_tree(repo, build_dir)
    monkeypatch.setattr(
        capture,
        "_commands",
        lambda *_: [
            ("build", [sys.executable, "-c", "pass"]),
            (
                "build_cpp_tests_and_qualifiers",
                [sys.executable, "-c", "pass"],
            ),
            (
                "ctest_manifest_all",
                [
                    sys.executable,
                    "-c",
                    "print('  Test #1: one\\n  Test #2: two')",
                ],
            ),
            (
                "ctest_manifest_dynamic_memory",
                [
                    sys.executable,
                    "-c",
                    "print('  Test #1: one')",
                ],
            ),
            (
                "pytest_manifest_dynamic_memory",
                [
                    sys.executable,
                    "-c",
                    (
                        "print("
                        "'tests/builder/test_dynamic_memory_qualification.py::test_one"
                        "\\ntests/tools/test_capture_dynamic_memory_test_manifest.py::test_tool"
                        "\\ntests/e2e/test_native_dynamic_memory_graph.py::test_graph'"
                        ")"
                    ),
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
    assert report["commands"][2]["manifest_entries"] == ["one", "two"]
    assert report["commands"][2]["manifest_count"] == 2
    assert report["commands"][4]["manifest_entries"] == [
        "tests/builder/test_dynamic_memory_qualification.py::test_one",
        "tests/e2e/test_native_dynamic_memory_graph.py::test_graph",
        "tests/tools/test_capture_dynamic_memory_test_manifest.py::test_tool",
    ]
    assert report["commands"][0]["environment_overrides"] == {}
    assert report["commands"][4]["environment_overrides"] == {
        "TRTMC_BENCH_WORKER": str(
            (build_dir / "trtmc_benchmark_worker").resolve()
        ),
        "TRTMC_TRT_PLUGIN_LIBRARY": str(
            (build_dir / "libtrtmc_trt_plugins.so").resolve()
        ),
    }
    assert set(report["build_artifacts"]) == set(capture.BUILD_ARTIFACTS)
    assert (
        report["build_artifacts"]["runtime_kv_plugin"]["relative_path"] == "libtrtmc_trt_plugins.so"
    )
    assert (
        report["build_artifacts"]["nvrtc_optional_output_regression"][
            "relative_path"
        ]
        == "trtmc_nvrtc_optional_output_regression"
    )
    assert len(report["build_artifacts_sha256"]) == 64
    assert len(report["clean_build_command_sha256"]) == 64
    assert report["cmake_cache"]["configured_source"] == str(repo.resolve())
    persisted = json.loads((output_dir / "test-manifest-report.json").read_text(encoding="utf-8"))
    assert persisted["passed"] is True


def test_capture_rejects_dirty_source_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _seed_build_tree(repo, repo / "build-dynkv")
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
    report = json.loads((output_dir / "test-manifest-report.json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["source_state_pre"]["exact_head_gate_satisfied"] is False


def test_capture_persists_failure_and_post_source_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _seed_build_tree(repo, repo / "build-dynkv")
    monkeypatch.setattr(
        capture,
        "_commands",
        lambda *_: [
            ("build", [sys.executable, "-c", "pass"]),
            (
                "ctest_all",
                [sys.executable, "-c", "raise SystemExit(7)"],
            ),
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
    report = json.loads((output_dir / "test-manifest-report.json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["commands"][1]["returncode"] == 7
    assert report["source_state_unchanged"] is True


def test_capture_rejects_cmake_cache_from_another_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    build_dir = repo / "build-dynkv"
    _seed_build_tree(repo, build_dir)
    wrong_source = tmp_path / "wrong-source"
    wrong_source.mkdir()
    (build_dir / "CMakeCache.txt").write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={wrong_source.resolve()}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        capture,
        "_commands",
        lambda *_: pytest.fail("wrong-source cache must fail before commands"),
    )

    with pytest.raises(capture.ManifestError, match="different source tree"):
        capture.capture(
            repo_root=repo,
            build_dir=build_dir,
            python=Path(sys.executable),
            output_dir=repo / "artifacts" / "manifest",
        )


def test_capture_rejects_missing_exact_build_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    build_dir = repo / "build-dynkv"
    _seed_build_tree(repo, build_dir)
    (build_dir / capture.BUILD_ARTIFACTS["runtime_kv_plugin"]).unlink()
    monkeypatch.setattr(
        capture,
        "_commands",
        lambda *_: [
            ("build", [sys.executable, "-c", "pass"]),
            (
                "build_cpp_tests_and_qualifiers",
                [sys.executable, "-c", "pass"],
            ),
        ],
    )

    with pytest.raises(capture.ManifestError, match="runtime_kv_plugin"):
        capture.capture(
            repo_root=repo,
            build_dir=build_dir,
            python=Path(sys.executable),
            output_dir=repo / "artifacts" / "manifest",
        )


def test_capture_rejects_empty_collected_ctest_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    build_dir = repo / "build-dynkv"
    _seed_build_tree(repo, build_dir)
    monkeypatch.setattr(
        capture,
        "_commands",
        lambda *_: [
            ("build", [sys.executable, "-c", "pass"]),
            (
                "build_cpp_tests_and_qualifiers",
                [sys.executable, "-c", "pass"],
            ),
            (
                "ctest_manifest_all",
                [sys.executable, "-c", "pass"],
            ),
        ],
    )

    with pytest.raises(capture.ManifestError, match="empty test manifest"):
        capture.capture(
            repo_root=repo,
            build_dir=build_dir,
            python=Path(sys.executable),
            output_dir=repo / "artifacts" / "manifest",
        )


def test_capture_rejects_build_artifact_changed_during_test_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    build_dir = repo / "build-dynkv"
    _seed_build_tree(repo, build_dir)
    core = build_dir / capture.BUILD_ARTIFACTS["core"]
    mutation = (
        "from pathlib import Path; "
        f"Path({str(core)!r}).write_bytes(b'changed-after-build'); "
        "print('  Test #1: one')"
    )
    monkeypatch.setattr(
        capture,
        "_commands",
        lambda *_: [
            ("build", [sys.executable, "-c", "pass"]),
            (
                "build_cpp_tests_and_qualifiers",
                [sys.executable, "-c", "pass"],
            ),
            (
                "ctest_manifest_all",
                [sys.executable, "-c", mutation],
            ),
        ],
    )

    with pytest.raises(
        capture.ManifestError,
        match="build artifacts changed after the explicit qualifier build",
    ):
        capture.capture(
            repo_root=repo,
            build_dir=build_dir,
            python=Path(sys.executable),
            output_dir=repo / "artifacts" / "manifest",
        )


def test_build_artifacts_reject_duplicate_inode(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    for key, relative in capture.BUILD_ARTIFACTS.items():
        path = build_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(key.encode())
    (build_dir / "CMakeCache.txt").write_text(
        (
            f"CMAKE_HOME_DIRECTORY:INTERNAL={tmp_path.resolve()}\n"
            "TRTMC_TRT_BACKEND_ABI:STRING=11_2\n"
        ),
        encoding="utf-8",
    )
    (build_dir / "libtrtmc_backend_trt_11_2.so").symlink_to(
        "libtrtmc_backend_trt.so"
    )
    qwen = build_dir / capture.BUILD_ARTIFACTS["model_qwen"]
    llama = build_dir / capture.BUILD_ARTIFACTS["model_llama"]
    llama.unlink()
    llama.hardlink_to(qwen)

    with pytest.raises(capture.ManifestError, match="one inode"):
        capture._build_artifact_identities(build_dir)


def test_open_file_identity_rejects_in_place_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact.so"
    artifact.write_bytes(b"before")
    real_fstat = capture.os.fstat
    calls = 0

    def mutating_fstat(fd: int):
        nonlocal calls
        result = real_fstat(fd)
        calls += 1
        if calls == 1:
            artifact.write_bytes(b"after-and-larger")
        return result

    monkeypatch.setattr(capture.os, "fstat", mutating_fstat)
    with pytest.raises(capture.ManifestError, match="changed while it was hashed"):
        capture._open_file_identity(
            artifact,
            artifact_key="runtime_kv_plugin",
            relative_path=Path("artifact.so"),
        )


def _captured_validatable_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    _init_repo(repo)
    build_dir = repo / "build-dynkv"
    _seed_build_tree(repo, build_dir)
    real_commands = capture._commands
    monkeypatch.setattr(
        capture,
        "_commands",
        lambda *_: [
            ("build", [sys.executable, "-c", "pass"]),
            (
                "build_cpp_tests_and_qualifiers",
                [sys.executable, "-c", "pass"],
            ),
        ],
    )
    output_dir = repo / "artifacts" / "manifest"
    capture.capture(
        repo_root=repo,
        build_dir=build_dir,
        python=Path(sys.executable),
        output_dir=output_dir,
    )
    monkeypatch.setattr(capture, "_commands", real_commands)
    report_path = output_dir / "test-manifest-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["commands"] = complete_command_receipts(
        capture,
        repo_root=repo,
        build_dir=build_dir,
        output_dir=output_dir,
        python=Path(report["python"]),
    )
    report["clean_build_command_sha256"] = capture._canonical_sha(report["commands"][0])
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path, build_dir


def test_manifest_validator_reopens_exact_artifacts_and_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _ = _captured_validatable_manifest(tmp_path, monkeypatch)

    report = capture.load_and_validate_build_manifest(report_path)

    assert report["passed"] is True
    assert set(report["build_artifacts"]) == set(capture.BUILD_ARTIFACTS)


def test_manifest_validator_rejects_artifact_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, build_dir = _captured_validatable_manifest(
        tmp_path,
        monkeypatch,
    )
    (build_dir / capture.BUILD_ARTIFACTS["core"]).write_bytes(b"tampered")

    with pytest.raises(capture.ManifestError, match="artifact identity changed"):
        capture.load_and_validate_build_manifest(report_path)


def test_manifest_rejects_independent_versioned_trt_backend_copy(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    build_dir = repo / "build-dynkv"
    _seed_build_tree(repo, build_dir)
    active = build_dir / "libtrtmc_backend_trt_11_2.so"
    active.unlink()
    active.write_bytes((build_dir / "libtrtmc_backend_trt.so").read_bytes())

    with pytest.raises(capture.ManifestError, match="must be a symlink"):
        capture._build_artifact_identities(build_dir)


def test_manifest_rejects_versioned_trt_backend_hardlink(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    build_dir = repo / "build-dynkv"
    _seed_build_tree(repo, build_dir)
    active = build_dir / "libtrtmc_backend_trt_11_2.so"
    active.unlink()
    active.hardlink_to(build_dir / "libtrtmc_backend_trt.so")

    with pytest.raises(capture.ManifestError, match="must be a symlink"):
        capture._build_artifact_identities(build_dir)


def test_manifest_rejects_artifact_symlink_outside_build(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    build_dir = repo / "build-dynkv"
    _seed_build_tree(repo, build_dir)
    outside = tmp_path / "outside-model.so"
    outside.write_bytes(b"outside")
    model = build_dir / capture.BUILD_ARTIFACTS["model_qwen"]
    model.unlink()
    model.symlink_to(outside)

    with pytest.raises(capture.ManifestError, match="escapes the build directory"):
        capture._build_artifact_identities(build_dir)


def test_manifest_rejects_missing_or_ambiguous_trt_backend_abi(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    build_dir = repo / "build-dynkv"
    _seed_build_tree(repo, build_dir)
    cache = build_dir / "CMakeCache.txt"
    cache.write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={repo.resolve()}\n",
        encoding="utf-8",
    )

    with pytest.raises(capture.ManifestError, match="exactly one"):
        capture._build_artifact_identities(build_dir)


def test_manifest_validator_rejects_extra_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _ = _captured_validatable_manifest(tmp_path, monkeypatch)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["version_smuggling"] = True
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(capture.ManifestError, match="exact v2 field set"):
        capture.load_and_validate_build_manifest(report_path)


def test_manifest_validator_rejects_duplicate_json_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _ = _captured_validatable_manifest(tmp_path, monkeypatch)
    payload = report_path.read_text(encoding="utf-8").rstrip()
    report_path.write_text(
        payload[:-1] + ', "passed": true}\n',
        encoding="utf-8",
    )

    with pytest.raises(capture.ManifestError, match="duplicate JSON key 'passed'"):
        capture.load_and_validate_build_manifest(report_path)


def test_manifest_validator_rejects_wrong_schema_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _ = _captured_validatable_manifest(tmp_path, monkeypatch)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["schema_version"] = "trtmc.dynamic-memory-test-manifest/v1"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(capture.ManifestError, match="not a passed v2"):
        capture.load_and_validate_build_manifest(report_path)


def test_manifest_validator_rejects_clean_build_argv_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _ = _captured_validatable_manifest(tmp_path, monkeypatch)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["commands"][0]["argv"].append("--not-clean")
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(capture.ManifestError, match="contract changed"):
        capture.load_and_validate_build_manifest(report_path)


def test_manifest_validator_rejects_omitted_required_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _ = _captured_validatable_manifest(tmp_path, monkeypatch)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    del report["commands"][4]
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(capture.ManifestError, match="complete ordered"):
        capture.load_and_validate_build_manifest(report_path)


def test_manifest_validator_rejects_reordered_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _ = _captured_validatable_manifest(tmp_path, monkeypatch)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["commands"][3], report["commands"][4] = (
        report["commands"][4],
        report["commands"][3],
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(capture.ManifestError, match="contract changed"):
        capture.load_and_validate_build_manifest(report_path)


def test_manifest_validator_rejects_overlapping_command_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _ = _captured_validatable_manifest(tmp_path, monkeypatch)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    previous = report["commands"][0]
    current = report["commands"][1]
    current["started_ns"] = previous["finished_ns"] - 1
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(capture.ManifestError, match="ordered replay"):
        capture.load_and_validate_build_manifest(report_path)


def _replace_command_stdout(
    report: dict,
    *,
    label: str,
    text: str,
) -> None:
    command = next(
        item for item in report["commands"] if item["label"] == label
    )
    stdout = Path(command["stdout"])
    stdout.write_text(text, encoding="utf-8")
    command["stdout_sha256"] = capture._sha256(stdout)
    command["manifest_entries"] = capture._manifest_entries(label, text)
    command["manifest_count"] = len(command["manifest_entries"])


def test_manifest_validator_rejects_focused_ctest_outside_full_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _ = _captured_validatable_manifest(tmp_path, monkeypatch)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _replace_command_stdout(
        report,
        label="ctest_manifest_dynamic_memory",
        text="  Test #1: absent_from_full_manifest\n",
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(capture.ManifestError, match="not a subset"):
        capture.load_and_validate_build_manifest(report_path)


def test_manifest_validator_rejects_duplicate_pytest_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _ = _captured_validatable_manifest(tmp_path, monkeypatch)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    duplicate = (
        "tests/e2e/test_native_dynamic_memory_graph.py::test_graph_fixture"
    )
    _replace_command_stdout(
        report,
        label="pytest_manifest_dynamic_memory",
        text=f"{duplicate}\n{duplicate}\n",
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(capture.ManifestError, match="duplicate test entries"):
        capture.load_and_validate_build_manifest(report_path)


def test_manifest_validator_rejects_junit_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _ = _captured_validatable_manifest(tmp_path, monkeypatch)
    junit = report_path.parent / "pytest_dynamic_memory.junit.xml"
    junit.write_text("<testsuites />", encoding="utf-8")

    with pytest.raises(capture.ManifestError, match="JUnit changed"):
        capture.load_and_validate_build_manifest(report_path)


def _write_junit(path: Path, testcases: str) -> None:
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<testsuites><testsuite name="pytest">{testcases}</testsuite></testsuites>',
        encoding="utf-8",
    )


def test_pytest_junit_rejects_selected_skip_but_records_collection_skip(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    selected = repo / "tests" / "selected.py"
    selected.parent.mkdir(parents=True)
    selected.write_text("", encoding="utf-8")
    junit = tmp_path / "pytest.xml"
    _write_junit(
        junit,
        (
            '<testcase classname="" name="tests.unrelated">'
            '<skipped message="collection skipped" /></testcase>'
            '<testcase classname="tests.selected" name="test_required">'
            '<skipped message="runtime unavailable" /></testcase>'
        ),
    )

    outcomes = capture._pytest_junit_outcomes(
        junit,
        repo_root=repo,
        expected_entries=["tests/selected.py::test_required"],
    )

    assert outcomes["passed"] is False
    assert outcomes["selected_skips"] == ["tests/selected.py::test_required"]
    assert outcomes["unselected_collection_skips"] == [
        {
            "classname": "",
            "name": "tests.unrelated",
            "message": "collection skipped",
        }
    ]


def test_pytest_junit_requires_every_selected_manifest_entry(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    selected = repo / "tests" / "selected.py"
    selected.parent.mkdir(parents=True)
    selected.write_text("", encoding="utf-8")
    junit = tmp_path / "pytest.xml"
    _write_junit(
        junit,
        '<testcase classname="tests.selected" name="test_one" />',
    )

    outcomes = capture._pytest_junit_outcomes(
        junit,
        repo_root=repo,
        expected_entries=[
            "tests/selected.py::test_one",
            "tests/selected.py::TestGroup::test_two[param]",
        ],
    )

    assert outcomes["passed"] is False
    assert outcomes["missing_selected_tests"] == ["tests/selected.py::TestGroup::test_two[param]"]


def test_junit_node_id_preserves_class_and_parameter_name(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    selected = repo / "tests" / "selected.py"
    selected.parent.mkdir(parents=True)
    selected.write_text("", encoding="utf-8")
    testcase = ET.fromstring(
        '<testcase classname="tests.selected.TestGroup" name="test_value[param]" />'
    )

    assert capture._junit_node_id(testcase, repo_root=repo) == (
        "tests/selected.py::TestGroup::test_value[param]"
    )
