# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for installed-wheel model-smoke promotion evidence."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import subprocess
import uuid
import zipfile
from pathlib import Path
from typing import Any

import pytest

from tools.ci.context import CiContext
from tools.ci.package import (
    MEMORY_RECEIPT_PREFIX,
    WHEEL_BUILD_STATE,
    WHEEL_MODEL_SMOKE_RECEIPT,
    WheelPackageManager,
)
from tools.ci.process import CiError, ObservedProcessResult


MODEL_ID = "Qwen/Qwen3-0.6B"
BUNDLE_NAME = "qwen3-0.6b.trtfb"
TRTMC_BYTES = b"\x7fELFfake-trtmc"
PLUGIN_BYTES = b"\x7fELFfake-runtime-kv-plugin"


def _memory_receipt(*, request_complete: bool) -> dict[str, Any]:
    boundaries = ["after_runtime_kv_allocation"]
    if request_complete:
        boundaries.append("after_successful_request_completion")
    return {
        "receipt_schema_version": 3,
        "contract_version": 1,
        "policy": "auto",
        "policy_fraction": 0.9,
        "requested_kv_bytes": 0,
        "request_context_limit": 0,
        "model_context_limit": 32,
        "prefill_chunk_limit": 8,
        "runtime_kv_capacity_tokens": 16,
        "effective_request_limit": 16,
        "kv_bytes_per_token": 64,
        "kv_reserved_bytes": 1024,
        "kv_committed_bytes": 1024,
        "kv_allocation_id": 7,
        "capacity_decision_free_bytes": 4096,
        "capacity_decision_total_bytes": 16384,
        "capacity_decision_device_used_bytes": 12288,
        "settled_free_bytes": 3072,
        "settled_total_bytes": 16384,
        "settled_device_used_bytes": 13312,
        "settled_snapshot_unavailable_reason": None,
        "final_free_bytes": 4096,
        "final_total_bytes": 16384,
        "final_device_used_bytes": 12288,
        "backend_owned_cache_input_bytes": 0,
        "backend_owned_cache_output_bytes": 0,
        "peak_device_sample_boundaries": boundaries,
    }


def _memory_stderr(
    load: dict[str, Any] | None = None,
    completion: dict[str, Any] | None = None,
) -> str:
    values = [
        load or _memory_receipt(request_complete=False),
        completion or _memory_receipt(request_complete=True),
    ]
    return "".join(
        f"{MEMORY_RECEIPT_PREFIX}{json.dumps(value, sort_keys=True)}\n" for value in values
    )


def _source_snapshot() -> dict[str, object]:
    snapshot: dict[str, object] = {
        "git_head": "a" * 40,
        "git_tree": "b" * 40,
        "status": "",
        "clean": True,
    }
    snapshot["source_state_sha256"] = WheelPackageManager._canonical_sha256(
        {
            "git_head": snapshot["git_head"],
            "git_tree": snapshot["git_tree"],
            "status": snapshot["status"],
        }
    )
    return snapshot


def _write_fake_wheel(path: Path) -> tuple[str, str]:
    script_member = "fake-1.0.data/scripts/trtmc"
    plugin_member = "tensorrt_model_connect/bin/libtrtmc_trt_plugins.so"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(script_member, TRTMC_BYTES)
        archive.writestr(plugin_member, PLUGIN_BYTES)
    return script_member, plugin_member


def _process_receipt(name: str, pid: int, argv: list[str], cwd: Path) -> dict[str, object]:
    started = dt.datetime(2026, 1, 1, 0, 0, pid % 60, tzinfo=dt.UTC)
    return {
        "execution_id": f"{name}-{pid}",
        "pid": pid,
        "argv": argv,
        "cwd": str(cwd.resolve()),
        "started_at_utc": started.isoformat(),
        "finished_at_utc": (started + dt.timedelta(seconds=1)).isoformat(),
        "duration_ms": 1000,
        "timeout_seconds": 60,
        "returncode": 0,
    }


def _make_downloaded_artifact(root: Path) -> tuple[Path, dict[str, object]]:
    receipt_root = root / ".ci/wheel-model-smoke"
    artifact_dir = receipt_root / "artifacts"
    artifact_dir.mkdir(parents=True)
    bundle = artifact_dir / BUNDLE_NAME
    bundle.write_bytes(b"qualified-bundle")
    bundle_identity = WheelPackageManager._file_identity(bundle)

    wheel = root / "dist/fake-1.0-py312-none-manylinux_2_39_aarch64.whl"
    script_member, plugin_member = _write_fake_wheel(wheel)
    source = _source_snapshot()
    build_state = {
        "github_run_id": "123",
        "github_run_attempt": "1",
        "source_pre_json": WheelPackageManager._canonical_json(source),
        "source_post_json": WheelPackageManager._canonical_json(source),
        "py312_wheel_path": wheel.relative_to(root).as_posix(),
        "py312_wheel_sha256": WheelPackageManager._sha256(wheel),
        "py312_wheel_size_bytes": str(wheel.stat().st_size),
    }

    stderr = _memory_stderr()
    log_contents = {
        "build_stdout": "built\n",
        "build_stderr": "",
        "inspect_stdout": "engines\n",
        "inspect_stderr": "",
        "run_stdout": "Paris\n",
        "run_stderr": stderr,
    }
    logs: dict[str, dict[str, object]] = {}
    for name, content in log_contents.items():
        path = receipt_root / f"{name}.log"
        path.write_text(content, encoding="utf-8")
        logs[name] = WheelPackageManager._file_receipt(path, relative_to=root)

    archived_at_runtime = Path("/workspace/source") / bundle.relative_to(root)
    venv_trtmc = "/tmp/smoke/venv/bin/trtmc"
    processes = {
        "build": _process_receipt(
            "build", 101, [venv_trtmc, "build", MODEL_ID], Path("/tmp/smoke")
        ),
        "inspect": _process_receipt(
            "inspect",
            102,
            [venv_trtmc, "inspect", "--list-engines", str(archived_at_runtime)],
            Path("/tmp/smoke"),
        ),
        "run": _process_receipt(
            "run",
            103,
            [
                venv_trtmc,
                "run",
                str(archived_at_runtime),
                "--prompt",
                "hello",
                "--max-new-tokens",
                "1",
            ],
            Path("/tmp/smoke"),
        ),
    }
    memory = WheelPackageManager._parse_and_validate_memory_receipts(stderr)
    receipt: dict[str, object] = {
        "schema_version": 2,
        "producer": "test",
        "source": {
            "wheel_build_pre": source,
            "wheel_build_post": source,
            "smoke_pre": source,
            "smoke_post": source,
            "unchanged": True,
        },
        "model_id": MODEL_ID,
        "build_user_argv": ["trtmc", "build", MODEL_ID],
        "separate_processes": True,
        "processes": processes,
        "wheel": {
            "artifact": WheelPackageManager._file_receipt(wheel, relative_to=root),
            "build_state": build_state,
            "installed_members": {
                "trtmc": {
                    "wheel_member": script_member,
                    "member_sha256": hashlib.sha256(TRTMC_BYTES).hexdigest(),
                    "installed_path": venv_trtmc,
                    "installed_sha256": hashlib.sha256(TRTMC_BYTES).hexdigest(),
                    "matches": True,
                },
                "runtime_kv_plugin": {
                    "wheel_member": plugin_member,
                    "member_sha256": hashlib.sha256(PLUGIN_BYTES).hexdigest(),
                    "installed_path": "/tmp/smoke/plugin.so",
                    "installed_sha256": hashlib.sha256(PLUGIN_BYTES).hexdigest(),
                    "matches": True,
                },
            },
            "isolated_import_path": "/tmp/smoke/venv/site-packages/package/__init__.py",
            "isolated_import_under_venv": True,
        },
        "memory": memory,
        "artifacts": {
            "bundle": {
                "artifact": WheelPackageManager._file_receipt(bundle, relative_to=root),
                "after_build": bundle_identity,
                "after_copy": bundle_identity,
                "after_inspect": bundle_identity,
                "after_run": bundle_identity,
                "unchanged": True,
            },
            "logs": logs,
        },
    }
    receipt_path = root / ".ci" / WHEEL_MODEL_SMOKE_RECEIPT
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt_path, receipt


def test_memory_parser_requires_the_final_completion_receipt() -> None:
    parsed = WheelPackageManager._parse_and_validate_memory_receipts(_memory_stderr())

    assert parsed["receipt_count"] == 2
    assert parsed["request_completion"]["policy"] == "auto"
    assert (
        "after_successful_request_completion"
        in parsed["request_completion"]["peak_device_sample_boundaries"]
    )


def test_memory_parser_rejects_a_load_only_receipt() -> None:
    stderr = f"{MEMORY_RECEIPT_PREFIX}{json.dumps(_memory_receipt(request_complete=False))}\n"
    with pytest.raises(CiError, match="both load and request-completion"):
        WheelPackageManager._parse_and_validate_memory_receipts(stderr)


def test_memory_parser_rejects_malformed_json() -> None:
    with pytest.raises(CiError, match="invalid JSON"):
        WheelPackageManager._parse_and_validate_memory_receipts(
            f"{MEMORY_RECEIPT_PREFIX}{{broken\n"
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("policy", "fraction", "policy must be"),
        ("policy_fraction", 0.8, "policy_fraction must be 0.9"),
        ("kv_reserved_bytes", 512, "byte accounting is inconsistent"),
        ("backend_owned_cache_input_bytes", 1, "backend_owned_cache_input_bytes must be"),
        ("settled_free_bytes", None, "settled_free_bytes must be a positive integer"),
        (
            "settled_snapshot_unavailable_reason",
            "sampling failed",
            "missing its settled snapshot",
        ),
        (
            "final_free_bytes",
            1,
            "capacity-decision compatibility alias",
        ),
    ],
)
def test_memory_parser_rejects_invalid_completion_invariants(
    field: str, value: object, message: str
) -> None:
    completion = _memory_receipt(request_complete=True)
    completion[field] = value

    with pytest.raises(CiError, match=message):
        WheelPackageManager._parse_and_validate_memory_receipts(
            _memory_stderr(completion=completion)
        )


def test_downloaded_artifact_verifier_recomputes_all_evidence(
    tmp_path: Path,
) -> None:
    _, expected = _make_downloaded_artifact(tmp_path)

    actual = WheelPackageManager.verify_model_smoke_artifact(tmp_path)

    assert actual == expected


@pytest.mark.parametrize(
    "relative_path",
    [
        ".ci/wheel-model-smoke/artifacts/qwen3-0.6b.trtfb",
        ".ci/wheel-model-smoke/run_stderr.log",
        "dist/fake-1.0-py312-none-manylinux_2_39_aarch64.whl",
    ],
)
def test_downloaded_artifact_verifier_rejects_tampering(tmp_path: Path, relative_path: str) -> None:
    _make_downloaded_artifact(tmp_path)
    with (tmp_path / relative_path).open("ab") as stream:
        stream.write(b"tamper")

    with pytest.raises(CiError, match="hash or size"):
        WheelPackageManager.verify_model_smoke_artifact(tmp_path)


def _configure_fake_model_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutate_stage: str | None = None,
) -> tuple[WheelPackageManager, list[list[str]], Path]:
    run_id = f"unit-{uuid.uuid4().hex}"
    smoke_root = Path(f"/tmp/trtmc-wheel-model-smoke-{run_id}")
    context = CiContext(
        tmp_path,
        {
            "GITHUB_RUN_ID": run_id,
            "GITHUB_RUN_ATTEMPT": "1",
            "TRTMC_PACKAGE_WHEEL_ARCH": "manylinux_2_39_aarch64",
        },
    )
    manager = WheelPackageManager(context)

    config_dir = tmp_path / "tests/e2e/models/qwen"
    config_dir.mkdir(parents=True)
    (config_dir / "package_smoke.json").write_text(
        json.dumps(
            {
                "default": True,
                "name": "qwen3-0.6b",
                "model_id": MODEL_ID,
                "bundle": BUNDLE_NAME,
                "timing_cache": "timing.cache",
                "max_new_tokens": 1,
                "optimization_level": 1,
                "build_timeout": "1m",
                "run_timeout": "1m",
                "prompt": "hello",
                "run_args": ["--greedy"],
            }
        ),
        encoding="utf-8",
    )
    wheel = tmp_path / "dist/fake-1.0-py312-none-manylinux_2_39_aarch64.whl"
    _write_fake_wheel(wheel)

    venv = smoke_root / "venv"
    site_packages = venv / "lib/python3.12/site-packages"
    imported_package = site_packages / "tensorrt_model_connect/__init__.py"

    def fake_source_output(command: list[object], **_: object) -> str:
        arguments = [str(item) for item in command]
        if arguments == ["git", "rev-parse", "HEAD"]:
            return "a" * 40
        if arguments == ["git", "rev-parse", "HEAD^{tree}"]:
            return "b" * 40
        if arguments == [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ]:
            return ""
        if "-I" in arguments:
            return str(imported_package)
        raise AssertionError(f"unexpected output command: {arguments}")

    monkeypatch.setattr(context, "output", fake_source_output)
    source = manager._source_identity()
    context.write_state(
        WHEEL_BUILD_STATE,
        {
            "wheel_tag": "py312",
            "conan_out_dir": str(tmp_path / "conan"),
            "cmake_build_dir": str(tmp_path / "cmake"),
            "trt_include_dir": "/trt/include",
            "trt_library": "/trt/lib.so",
            "cuda_include_dir": "/cuda/include",
            "cudart_library": "/cuda/lib.so",
            "github_run_id": run_id,
            "github_run_attempt": "1",
            "source_pre_json": manager._canonical_json(source),
            "source_post_json": manager._canonical_json(source),
            "py312_wheel_path": wheel.relative_to(tmp_path).as_posix(),
            "py312_wheel_sha256": manager._sha256(wheel),
            "py312_wheel_size_bytes": str(wheel.stat().st_size),
        },
    )

    def fake_create_venv(_path: Path, _wheel: Path) -> None:
        (venv / "bin").mkdir(parents=True)
        (venv / "bin/python").write_bytes(b"python")
        (venv / "bin/trtmc").write_bytes(TRTMC_BYTES)
        imported_package.parent.mkdir(parents=True)
        imported_package.write_text("", encoding="utf-8")
        (imported_package.parent / "bin").mkdir()
        (imported_package.parent / "bin/libtrtmc_trt_plugins.so").write_bytes(PLUGIN_BYTES)

    monkeypatch.setattr(manager, "_create_venv", fake_create_venv)

    def fake_run(command: list[object], **_: object) -> subprocess.CompletedProcess[str]:
        arguments = [str(item) for item in command]
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    monkeypatch.setattr(context, "run", fake_run)
    observed_commands: list[list[str]] = []

    def fake_observed(
        command: list[object],
        *,
        limit: str | None = None,
        cwd: Path | None = None,
        **_: object,
    ) -> ObservedProcessResult:
        arguments = [str(item) for item in command]
        observed_commands.append(arguments)
        assert cwd is not None
        stderr = ""
        if arguments[1] == "build":
            (cwd / BUNDLE_NAME).write_bytes(b"qualified-bundle")
        elif arguments[1] == mutate_stage:
            Path(arguments[-1] if mutate_stage == "inspect" else arguments[2]).write_bytes(
                b"mutated-bundle"
            )
        if arguments[1] == "run":
            stderr = _memory_stderr()
        index = len(observed_commands)
        completed = subprocess.CompletedProcess(
            arguments, 0, stdout=f"{arguments[1]}\n", stderr=stderr
        )
        return ObservedProcessResult(
            completed=completed,
            execution_id=f"execution-{index}",
            pid=1000 + index,
            cwd=cwd.resolve(),
            started_at_utc=f"2026-01-01T00:00:0{index}+00:00",
            finished_at_utc=f"2026-01-01T00:00:1{index}+00:00",
            duration_ms=10,
            timeout_seconds=CiContext._duration_seconds(limit),
        )

    monkeypatch.setattr(context, "run_observed", fake_observed)
    return manager, observed_commands, smoke_root


def test_model_smoke_records_exact_no_flag_build_and_three_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, commands, smoke_root = _configure_fake_model_smoke(tmp_path, monkeypatch)
    try:
        manager.model_smoke()
    finally:
        shutil.rmtree(smoke_root, ignore_errors=True)

    receipt = json.loads((tmp_path / ".ci" / WHEEL_MODEL_SMOKE_RECEIPT).read_text(encoding="utf-8"))
    assert commands[0][1:] == ["build", MODEL_ID]
    assert [command[1] for command in commands] == ["build", "inspect", "run"]
    assert receipt["build_user_argv"] == ["trtmc", "build", MODEL_ID]
    assert receipt["separate_processes"] is True
    assert {entry["pid"] for entry in receipt["processes"].values()} == {
        1001,
        1002,
        1003,
    }
    assert (tmp_path / receipt["artifacts"]["bundle"]["artifact"]["path"]).is_file()


@pytest.mark.parametrize("stage", ["inspect", "run"])
def test_model_smoke_rejects_bundle_mutation_between_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    manager, _, smoke_root = _configure_fake_model_smoke(tmp_path, monkeypatch, mutate_stage=stage)
    try:
        with pytest.raises(CiError, match=f"bundle {stage} changed"):
            manager.model_smoke()
    finally:
        shutil.rmtree(smoke_root, ignore_errors=True)


def test_model_smoke_rejects_stale_wheel_before_spawning_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, commands, smoke_root = _configure_fake_model_smoke(tmp_path, monkeypatch)
    state_path = tmp_path / ".ci" / WHEEL_BUILD_STATE
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["py312_wheel_sha256"] = "0" * 64
    state_path.write_text(json.dumps(state), encoding="utf-8")
    try:
        with pytest.raises(CiError, match="same-run build provenance"):
            manager.model_smoke()
    finally:
        shutil.rmtree(smoke_root, ignore_errors=True)

    assert commands == []
