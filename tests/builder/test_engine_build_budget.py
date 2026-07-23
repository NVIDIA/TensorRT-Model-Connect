# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fail-closed CI budget around full bundle construction."""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tensorrt_model_connect.engine_build_budget import (
    enforce_single_full_bundle_build,
)


def _guarded_builder(calls: list[str]):
    @enforce_single_full_bundle_build
    def build_bundle(
        model_dir: str,
        output_path: str,
        *,
        build_timing_path: str | None = None,
    ) -> None:
        calls.append(model_dir)
        Path(output_path).write_bytes(b"bundle")
        if build_timing_path:
            Path(build_timing_path).write_text("{}\n", encoding="utf-8")

    return build_bundle


def _leave_started_build_ledger(
    model_dir: str,
    output_path: str,
    timing_path: str,
) -> None:
    @enforce_single_full_bundle_build
    def build_bundle(
        model_dir: str,
        output_path: str,
        *,
        build_timing_path: str | None = None,
    ) -> None:
        del model_dir, output_path, build_timing_path
        os._exit(91)

    build_bundle(
        model_dir,
        output_path,
        build_timing_path=timing_path,
    )


def _enable_guard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    identity: str = "unit-model-config",
) -> Path:
    guard_dir = tmp_path / "engine-builds"
    monkeypatch.setenv("TRTMC_ENGINE_BUILD_GUARD_DIR", str(guard_dir))
    monkeypatch.setenv("TRTMC_ENGINE_BUILD_IDENTITY", identity)
    monkeypatch.setenv("TRTMC_ENGINE_BUILD_REVISION", "abc123")
    monkeypatch.setenv(
        "TRTMC_ENGINE_BUILD_COMMAND_JSON",
        json.dumps(["python", "-m", "tensorrt_model_connect", "build"]),
    )
    return guard_dir


def test_guard_allows_one_full_bundle_build_and_rejects_the_second(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    guard_dir = _enable_guard(monkeypatch, tmp_path)
    calls: list[str] = []
    build_bundle = _guarded_builder(calls)
    bundle_path = tmp_path / "unit.trtfb"
    timing_path = tmp_path / "build_timing.json"

    build_bundle("/models/unit", str(bundle_path), build_timing_path=str(timing_path))
    with pytest.raises(RuntimeError, match="build budget already consumed"):
        build_bundle("/models/unit", str(bundle_path), build_timing_path=str(timing_path))

    assert calls == ["/models/unit"]
    records = list(guard_dir.glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["identity"] == "unit-model-config"
    assert record["status"] == "passed"
    assert record["invocation_count"] == 1
    assert record["attempt_count"] == 1
    assert record["recovery_attempts"] == []
    assert record["returncode"] == 0
    assert record["source_revision"] == "abc123"
    assert record["bundle_path"] == str(bundle_path)
    assert record["build_timing_path"] == str(timing_path)


def test_guard_rejects_concurrent_duplicate_builds_atomically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    guard_dir = _enable_guard(monkeypatch, tmp_path)
    calls: list[str] = []
    build_bundle = _guarded_builder(calls)
    barrier = threading.Barrier(3)

    def invoke(index: int) -> str:
        barrier.wait()
        try:
            build_bundle(
                f"/models/unit-{index}",
                str(tmp_path / f"unit-{index}.trtfb"),
                build_timing_path=str(tmp_path / f"timing-{index}.json"),
            )
        except RuntimeError as exc:
            return str(exc)
        return "passed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(invoke, index) for index in range(2)]
        barrier.wait()
        outcomes = [future.result() for future in futures]

    assert outcomes.count("passed") == 1
    assert sum("build budget already consumed" in item for item in outcomes) == 1
    assert len(calls) == 1
    assert len(list(guard_dir.glob("*.json"))) == 1


def test_guard_recovers_one_abandoned_sigsegv_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    guard_dir = _enable_guard(monkeypatch, tmp_path)
    bundle_path = tmp_path / "unit.trtfb"
    timing_path = tmp_path / "build_timing.json"
    process = multiprocessing.Process(
        target=_leave_started_build_ledger,
        args=("/models/unit", str(bundle_path), str(timing_path)),
    )
    process.start()
    process.join(timeout=10)

    assert process.exitcode == 91
    monkeypatch.setenv("TRTMC_ENGINE_BUILD_RECOVERY_ATTEMPT", "2")
    monkeypatch.setenv("TRTMC_ENGINE_BUILD_RECOVERY_SIGNAL", str(signal.SIGSEGV))
    calls: list[str] = []
    _guarded_builder(calls)(
        "/models/unit",
        str(bundle_path),
        build_timing_path=str(timing_path),
    )

    assert calls == ["/models/unit"]
    records = list(guard_dir.glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["status"] == "passed"
    assert record["invocation_count"] == 1
    assert record["attempt_count"] == 2
    assert record["returncode"] == 0
    assert [
        (attempt["attempt"], attempt["returncode"], attempt["signal"])
        for attempt in record["recovery_attempts"]
    ] == [(1, -signal.SIGSEGV, signal.SIGSEGV)]


@pytest.mark.parametrize("field", ["invocation_count", "attempt_count"])
def test_guard_recovery_rejects_boolean_ledger_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
) -> None:
    guard_dir = _enable_guard(monkeypatch, tmp_path)
    bundle_path = tmp_path / "unit.trtfb"
    timing_path = tmp_path / "build_timing.json"
    process = multiprocessing.Process(
        target=_leave_started_build_ledger,
        args=("/models/unit", str(bundle_path), str(timing_path)),
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 91
    record_path = next(guard_dir.glob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record[field] = True
    record_path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setenv("TRTMC_ENGINE_BUILD_RECOVERY_ATTEMPT", "2")
    monkeypatch.setenv("TRTMC_ENGINE_BUILD_RECOVERY_SIGNAL", str(signal.SIGSEGV))
    calls: list[str] = []

    with pytest.raises(RuntimeError, match=field):
        _guarded_builder(calls)(
            "/models/unit",
            str(bundle_path),
            build_timing_path=str(timing_path),
        )

    assert calls == []


def test_guard_allows_only_one_concurrent_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    guard_dir = _enable_guard(monkeypatch, tmp_path)
    bundle_path = tmp_path / "unit.trtfb"
    timing_path = tmp_path / "build_timing.json"
    process = multiprocessing.Process(
        target=_leave_started_build_ledger,
        args=("/models/unit", str(bundle_path), str(timing_path)),
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 91

    monkeypatch.setenv("TRTMC_ENGINE_BUILD_RECOVERY_ATTEMPT", "2")
    monkeypatch.setenv("TRTMC_ENGINE_BUILD_RECOVERY_SIGNAL", str(signal.SIGSEGV))
    calls: list[str] = []
    build_bundle = _guarded_builder(calls)
    barrier = threading.Barrier(3)

    def recover() -> str:
        barrier.wait()
        try:
            build_bundle(
                "/models/unit",
                str(bundle_path),
                build_timing_path=str(timing_path),
            )
        except RuntimeError as exc:
            return str(exc)
        return "passed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(recover) for _ in range(2)]
        barrier.wait()
        outcomes = [future.result() for future in futures]

    assert outcomes.count("passed") == 1
    assert len(calls) == 1
    records = list(guard_dir.glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["status"] == "passed"
    assert record["attempt_count"] == 2


def test_guard_fails_closed_without_manifest_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    guard_dir = _enable_guard(monkeypatch, tmp_path)
    monkeypatch.delenv("TRTMC_ENGINE_BUILD_IDENTITY")
    calls: list[str] = []
    build_bundle = _guarded_builder(calls)

    with pytest.raises(RuntimeError, match="TRTMC_ENGINE_BUILD_IDENTITY is required"):
        build_bundle("/models/unit", str(tmp_path / "unit.trtfb"))

    assert calls == []
    assert not guard_dir.exists()
