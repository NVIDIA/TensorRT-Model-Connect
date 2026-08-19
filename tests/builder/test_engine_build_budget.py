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
        precision: str = "fp32",
    ) -> None:
        del precision
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
        precision: str = "fp32",
    ) -> None:
        del model_dir, output_path, build_timing_path, precision
        os._exit(91)

    build_bundle(
        model_dir,
        output_path,
        build_timing_path=timing_path,
    )


def _start_abandoned_build(
    model_dir: str,
    bundle_path: Path,
    timing_path: Path,
) -> None:
    process = multiprocessing.Process(
        target=_leave_started_build_ledger,
        args=(model_dir, str(bundle_path), str(timing_path)),
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 91


def _request_sigsegv_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRTMC_ENGINE_BUILD_RECOVERY_ATTEMPT", "2")
    monkeypatch.setenv("TRTMC_ENGINE_BUILD_RECOVERY_SIGNAL", str(signal.SIGSEGV))


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


def _stage_nemo_archive(
    tmp_path: Path,
    name: str,
    archive_path: Path,
    *,
    staged_target: Path | None = None,
    config_overrides: dict[str, object] | None = None,
) -> Path:
    staged_dir = tmp_path / name
    staged_dir.mkdir()
    config = {
        "model_type": "nemotron_speech_streaming",
        "hidden_size": 1024,
        "_nemo_archive_path": str(archive_path),
    }
    config.update(config_overrides or {})
    (staged_dir / "config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    (staged_dir / archive_path.name).symlink_to((staged_target or archive_path).resolve())
    return staged_dir


def test_guard_allows_one_full_bundle_build_and_rejects_the_second(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    guard_dir = _enable_guard(monkeypatch, tmp_path)
    calls: list[str] = []
    build_bundle = _guarded_builder(calls)
    bundle_path = tmp_path / "unit.bundle"
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


def test_guard_reads_build_timing_from_opaque_model_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    guard_dir = _enable_guard(monkeypatch, tmp_path)
    bundle_path = tmp_path / "unit.bundle"
    timing_path = tmp_path / "build_timing.json"

    @enforce_single_full_bundle_build
    def build(model_id_or_path: str, output_path: str, **options: object) -> None:
        del model_id_or_path
        Path(output_path).write_bytes(b"bundle")
        Path(str(options["build_timing_path"])).write_text("{}\n", encoding="utf-8")

    build("unit/model", str(bundle_path), build_timing_path=str(timing_path))

    record = json.loads(next(guard_dir.glob("*.json")).read_text(encoding="utf-8"))
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
                str(tmp_path / f"unit-{index}.bundle"),
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
    bundle_path = tmp_path / "unit.bundle"
    timing_path = tmp_path / "build_timing.json"
    _start_abandoned_build("/models/unit", bundle_path, timing_path)
    _request_sigsegv_recovery(monkeypatch)
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


def test_guard_recovers_same_nemo_archive_from_new_staging_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    guard_dir = _enable_guard(monkeypatch, tmp_path)
    archive_path = tmp_path / "model.nemo"
    archive_path.write_bytes(b"nemo archive")
    first_parent = _stage_nemo_archive(tmp_path, "staging-first-parent", archive_path)
    second_parent = _stage_nemo_archive(tmp_path, "staging-second-parent", archive_path)
    first_staging = _stage_nemo_archive(tmp_path, "staging-first", first_parent / archive_path.name)
    second_staging = _stage_nemo_archive(
        tmp_path, "staging-second", second_parent / archive_path.name
    )
    bundle_path = tmp_path / "unit.bundle"
    timing_path = tmp_path / "build_timing.json"
    _start_abandoned_build(str(first_staging), bundle_path, timing_path)
    _request_sigsegv_recovery(monkeypatch)
    calls: list[str] = []
    _guarded_builder(calls)(
        str(second_staging),
        str(bundle_path),
        build_timing_path=str(timing_path),
    )

    assert calls == [str(second_staging)]
    record = json.loads(next(guard_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert record["status"] == "passed"
    assert record["attempt_count"] == 2
    assert [
        (attempt["attempt"], attempt["returncode"], attempt["signal"])
        for attempt in record["recovery_attempts"]
    ] == [(1, -signal.SIGSEGV, signal.SIGSEGV)]


@pytest.mark.parametrize(
    "model_dir_kind",
    ["changed_config", "different_archive", "extra_file", "ordinary", "mismatch"],
)
def test_guard_recovery_rejects_a_different_model_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_dir_kind: str,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    first_archive = tmp_path / "first" / "model.nemo"
    second_archive = tmp_path / "second" / "model.nemo"
    first_archive.parent.mkdir()
    second_archive.parent.mkdir()
    first_archive.write_bytes(b"first archive")
    second_archive.write_bytes(b"second archive")
    if model_dir_kind == "ordinary":
        first_model_dir = tmp_path / "ordinary-first"
        second_model_dir = tmp_path / "ordinary-second"
        first_model_dir.mkdir()
        second_model_dir.mkdir()
    else:
        first_model_dir = _stage_nemo_archive(tmp_path, "staging-first", first_archive)
        second_model_dir = _stage_nemo_archive(
            tmp_path,
            "staging-second",
            second_archive if model_dir_kind == "different_archive" else first_archive,
            staged_target=(second_archive if model_dir_kind == "mismatch" else None),
            config_overrides=(
                {"hidden_size": 2048} if model_dir_kind == "changed_config" else None
            ),
        )
        if model_dir_kind == "extra_file":
            (second_model_dir / "tokenizer.json").write_text(
                '{"version": "changed"}\n',
                encoding="utf-8",
            )
    bundle_path = tmp_path / "unit.bundle"
    timing_path = tmp_path / "build_timing.json"
    _start_abandoned_build(str(first_model_dir), bundle_path, timing_path)
    _request_sigsegv_recovery(monkeypatch)
    calls: list[str] = []
    with pytest.raises(RuntimeError, match="arguments_sha256"):
        _guarded_builder(calls)(
            str(second_model_dir),
            str(bundle_path),
            build_timing_path=str(timing_path),
        )

    assert calls == []


def test_guard_recovery_still_rejects_changed_build_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    archive_path = tmp_path / "model.nemo"
    archive_path.write_bytes(b"nemo archive")
    first_staging = _stage_nemo_archive(tmp_path, "staging-first", archive_path)
    second_staging = _stage_nemo_archive(tmp_path, "staging-second", archive_path)
    bundle_path = tmp_path / "unit.bundle"
    timing_path = tmp_path / "build_timing.json"
    _start_abandoned_build(str(first_staging), bundle_path, timing_path)
    _request_sigsegv_recovery(monkeypatch)
    calls: list[str] = []
    with pytest.raises(RuntimeError, match="arguments_sha256"):
        _guarded_builder(calls)(
            str(second_staging),
            str(bundle_path),
            build_timing_path=str(timing_path),
            precision="fp16",
        )

    assert calls == []


def test_guard_recovery_rejects_same_path_archive_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    archive_path = tmp_path / "model.nemo"
    archive_path.write_bytes(b"first archive")
    first_staging = _stage_nemo_archive(tmp_path, "staging-first", archive_path)
    bundle_path = tmp_path / "unit.bundle"
    timing_path = tmp_path / "build_timing.json"
    _start_abandoned_build(str(first_staging), bundle_path, timing_path)
    previous_stat = archive_path.stat()
    archive_path.write_bytes(b"other archive")
    os.utime(
        archive_path,
        ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns + 1_000_000_000),
    )
    second_staging = _stage_nemo_archive(tmp_path, "staging-second", archive_path)
    _request_sigsegv_recovery(monkeypatch)
    calls: list[str] = []
    with pytest.raises(RuntimeError, match="arguments_sha256"):
        _guarded_builder(calls)(
            str(second_staging),
            str(bundle_path),
            build_timing_path=str(timing_path),
        )

    assert calls == []


def test_guard_recovery_does_not_canonicalize_non_nemo_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    archive_path = tmp_path / "model.bin"
    archive_path.write_bytes(b"not a NeMo archive")
    first_staging = _stage_nemo_archive(tmp_path, "staging-first", archive_path)
    second_staging = _stage_nemo_archive(tmp_path, "staging-second", archive_path)
    bundle_path = tmp_path / "unit.bundle"
    timing_path = tmp_path / "build_timing.json"
    _start_abandoned_build(str(first_staging), bundle_path, timing_path)
    _request_sigsegv_recovery(monkeypatch)
    calls: list[str] = []
    with pytest.raises(RuntimeError, match="arguments_sha256"):
        _guarded_builder(calls)(
            str(second_staging),
            str(bundle_path),
            build_timing_path=str(timing_path),
        )

    assert calls == []


@pytest.mark.parametrize("field", ["invocation_count", "attempt_count"])
def test_guard_recovery_rejects_boolean_ledger_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
) -> None:
    guard_dir = _enable_guard(monkeypatch, tmp_path)
    bundle_path = tmp_path / "unit.bundle"
    timing_path = tmp_path / "build_timing.json"
    _start_abandoned_build("/models/unit", bundle_path, timing_path)
    record_path = next(guard_dir.glob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record[field] = True
    record_path.write_text(json.dumps(record), encoding="utf-8")
    _request_sigsegv_recovery(monkeypatch)
    calls: list[str] = []

    with pytest.raises(RuntimeError, match=field):
        _guarded_builder(calls)(
            "/models/unit",
            str(bundle_path),
            build_timing_path=str(timing_path),
        )

    assert calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("schema_version", 1.0),
        ("builder_pid", True),
        ("builder_pid", 0),
        ("started_at", "not-a-timestamp"),
        ("started_at", "9999-01-01T00:00:00+00:00"),
    ],
)
def test_guard_recovery_rejects_malformed_process_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    guard_dir = _enable_guard(monkeypatch, tmp_path)
    bundle_path = tmp_path / "unit.bundle"
    timing_path = tmp_path / "build_timing.json"
    _start_abandoned_build("/models/unit", bundle_path, timing_path)
    record_path = next(guard_dir.glob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record[field] = value
    record_path.write_text(json.dumps(record), encoding="utf-8")
    _request_sigsegv_recovery(monkeypatch)
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
    bundle_path = tmp_path / "unit.bundle"
    timing_path = tmp_path / "build_timing.json"
    _start_abandoned_build("/models/unit", bundle_path, timing_path)
    _request_sigsegv_recovery(monkeypatch)
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
        build_bundle("/models/unit", str(tmp_path / "unit.bundle"))

    assert calls == []
    assert not guard_dir.exists()
