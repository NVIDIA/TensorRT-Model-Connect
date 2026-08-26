# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tensorrt_model_connect.serve import worker as worker_module
from tensorrt_model_connect.serve.errors import (
    WorkerCrashedError,
    WorkerProtocolError,
    WorkerRemoteError,
    WorkerRequestTooLargeError,
    WorkerSaturatedError,
    WorkerStartupError,
    WorkerTimeoutError,
)
from tensorrt_model_connect.serve.worker import WorkerGroup, WorkerLoadOptions, WorkerProcess


FAKE_TRTMC = Path(__file__).with_name("fake_serve_worker.py")


def make_worker(
    tmp_path: Path,
    name: str = "chat",
    *,
    startup_timeout: float = 1.0,
    request_timeout: float = 1.0,
    max_request_line_bytes: int = 16 * 1024 * 1024,
    load_options: WorkerLoadOptions | None = None,
) -> WorkerProcess:
    bundle = tmp_path / f"{name}.bundle"
    bundle.write_bytes(b"fixture")
    return WorkerProcess(
        name=name,
        bundle=bundle,
        trtmc_binary=FAKE_TRTMC,
        startup_timeout=startup_timeout,
        request_timeout=request_timeout,
        max_request_line_bytes=max_request_line_bytes,
        load_options=load_options,
    )


def wait_for_stderr(worker: WorkerProcess, text: str, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(text in line for line in worker.stderr_tail):
            return True
        time.sleep(0.005)
    return any(text in line for line in worker.stderr_tail)


def test_worker_ready_request_ids_metadata_stderr_and_cleanup(tmp_path: Path) -> None:
    worker = make_worker(tmp_path)
    worker.start()

    assert worker.ready
    assert worker.pid is not None
    assert worker.ready_payload["protocol_version"] == 2
    assert worker.ready_payload["model_id"] == "chat"
    assert worker.ready_payload["pipeline_type"] == "chat"
    assert wait_for_stderr(worker, "fake worker ready")

    worker.close()
    assert worker.state == "closed"
    assert not worker.ready


def test_worker_serializes_concurrent_requests(tmp_path: Path) -> None:
    worker = make_worker(tmp_path)
    worker.start()
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(worker.request, "generate", {"prompt": f"p{index}"})
                for index in range(12)
            ]
        assert {future.result()["text"] for future in futures} == {
            f"generated:p{index}" for index in range(12)
        }
        assert worker.ready
    finally:
        worker.close()


def test_worker_session_excludes_other_requests_without_poisoning_worker(
    tmp_path: Path,
) -> None:
    worker = make_worker(tmp_path, "stream-asr", request_timeout=0.1)
    worker.start()
    session = worker.acquire_session()
    try:
        session.request(
            "stream_start",
            {"config": {}},
        )
        with pytest.raises(WorkerTimeoutError, match="remained busy"):
            worker.request("generate", {"prompt": "busy"}, timeout=0.05)
        assert worker.ready
        session.request("stream_reset")
    finally:
        session.close()
        worker.close()


@pytest.mark.parametrize(
    ("mode", "message"),
    [("bad-ready", "invalid JSONL"), ("legacy-ready", "string request id")],
)
def test_worker_startup_rejects_invalid_ready_shape(
    tmp_path: Path, mode: str, message: str
) -> None:
    worker = make_worker(tmp_path, mode, startup_timeout=0.5)
    with pytest.raises(WorkerStartupError, match=message):
        worker.start()
    assert worker.state == "failed"
    worker.close()


def test_worker_startup_timeout_terminates_process_without_exposing_stderr(
    tmp_path: Path,
) -> None:
    worker = make_worker(tmp_path, "startup-timeout", startup_timeout=1.0)
    with pytest.raises(WorkerStartupError, match="did not become ready") as failure:
        worker.start()
    assert "startup detail" not in str(failure.value)
    assert str(tmp_path) not in str(failure.value)
    assert wait_for_stderr(worker, "startup detail")
    assert worker.state == "failed"
    assert not worker.ready
    worker.close()


def test_worker_binary_startup_error_does_not_expose_absolute_path(tmp_path: Path) -> None:
    bundle = tmp_path / "model.bundle"
    bundle.write_bytes(b"fixture")
    missing_binary = tmp_path / "private" / "missing-trtmc"
    worker = WorkerProcess(
        name="chat",
        bundle=bundle,
        trtmc_binary=missing_binary,
        startup_timeout=0.1,
        request_timeout=0.1,
    )
    with pytest.raises(WorkerStartupError, match="failed to launch worker") as failure:
        worker.start()
    assert str(tmp_path) not in str(failure.value)
    assert str(missing_binary) not in str(failure.value)
    worker.close()


def test_worker_structured_remote_error(tmp_path: Path) -> None:
    worker = make_worker(tmp_path)
    worker.start()
    try:
        with pytest.raises(WorkerRemoteError, match="unsupported op") as failure:
            worker.request("does-not-exist")
        assert failure.value.details["type"] == "invalid_request_error"
        assert worker.ready
    finally:
        worker.close()


def test_worker_rejects_reserved_payload_fields_without_losing_sync(tmp_path: Path) -> None:
    worker = make_worker(tmp_path)
    worker.start()
    try:
        with pytest.raises(ValueError, match="cannot override 'id'"):
            worker.request("generate", {"id": "caller-controlled"})
        assert worker.request("generate", {"prompt": "healthy"})["text"] == ("generated:healthy")
        assert worker.ready
    finally:
        worker.close()


def test_worker_rejects_oversized_serialized_line_without_poisoning_worker(
    tmp_path: Path,
) -> None:
    worker = make_worker(tmp_path, max_request_line_bytes=256)
    worker.start()
    try:
        with pytest.raises(WorkerRequestTooLargeError, match="maximum is 256 bytes"):
            worker.request("generate", {"prompt": '"' * 300})
        assert worker.request("generate", {"prompt": "healthy"})["text"] == ("generated:healthy")
        assert worker.ready
    finally:
        worker.close()


def test_worker_environment_is_an_explicit_runtime_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    for name in (
        "ACCESS_TOKEN",
        "AUTHORIZATION",
        "AWS_SECRET_ACCESS_KEY",
        "COOKIE",
        "GITHUB_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "NVIDIA_API_KEY",
        "TRTMC_SERVE_TOKEN",
        "UNLISTED_ENVIRONMENT",
    ):
        monkeypatch.setenv(name, f"secret-{name.lower()}")
    worker = make_worker(tmp_path)
    worker.start()
    try:
        assert worker.ready_payload["serve_token_present"] is False
        assert worker.ready_payload["secret_environment_present"] is False
        assert worker.ready_payload["allowed_cuda_visible_devices"] == "7"
    finally:
        worker.close()


def test_worker_environment_adds_windows_process_basics_only_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windows_values = {
        "COMSPEC": r"C:\Windows\System32\cmd.exe",
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        "SYSTEMROOT": r"C:\Windows",
        "TEMP": r"C:\Temp",
        "TMP": r"C:\Temp",
        "USERPROFILE": r"C:\Users\trtmc",
    }
    for name, value in windows_values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("TRTMC_SERVE_TOKEN", "must-stay-private")

    with monkeypatch.context() as windows:
        windows.setattr(worker_module.os, "name", "nt")
        environment = worker_module._worker_environment()

    assert {name: environment[name] for name in windows_values} == windows_values
    assert "TRTMC_SERVE_TOKEN" not in environment
    if worker_module.os.name != "nt":
        posix_environment = worker_module._worker_environment()
        assert windows_values.keys().isdisjoint(posix_environment)


def test_worker_forwards_native_load_options_verbatim(tmp_path: Path) -> None:
    options = WorkerLoadOptions(
        backend_dirs=("/opt/backend-a", "/opt/backend-b"),
        model_plugin_dirs=("/opt/model-plugin",),
        runtime_cache="/tmp/runtime.cache",
        kernel_bindings="/tmp/kernels.json",
        config="/tmp/config.json",
        set_values=("runtime.a=1", "runtime.b=true"),
        cuda_graphs=True,
    )
    worker = make_worker(tmp_path, load_options=options)
    worker.start()
    try:
        assert worker.ready_payload["worker_args"] == [
            "--backend-dir",
            "/opt/backend-a",
            "--backend-dir",
            "/opt/backend-b",
            "--model-plugin-dir",
            "/opt/model-plugin",
            "--runtime-cache",
            "/tmp/runtime.cache",
            "--kernel-bindings",
            "/tmp/kernels.json",
            "--config",
            "/tmp/config.json",
            "--set",
            "runtime.a=1",
            "--set",
            "runtime.b=true",
            "--cuda-graphs",
        ]
    finally:
        worker.close()


def test_worker_group_scales_across_replicas(tmp_path: Path) -> None:
    workers = [
        make_worker(tmp_path, f"saturation-chat-{replica}", request_timeout=1)
        for replica in range(2)
    ]
    for worker in workers:
        worker.start()
    group = WorkerGroup("chat", workers)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    lambda prompt: _group_request(group, "generate", {"prompt": prompt}),
                    ("first", "second"),
                )
            )
        assert {result["text"] for result in results} == {
            "generated:first",
            "generated:second",
        }
    finally:
        group.close()


def test_worker_group_rejects_when_all_replicas_are_busy(tmp_path: Path) -> None:
    workers = [make_worker(tmp_path, f"chat-{replica}") for replica in range(2)]
    for worker in workers:
        worker.start()
    group = WorkerGroup("chat", workers)
    first = group.acquire_session()
    second = group.acquire_session()
    try:
        with pytest.raises(WorkerSaturatedError, match="all 2 replicas"):
            group.acquire_session()
    finally:
        first.close()
        second.close()
        group.close()


def test_worker_group_drops_failed_lane_and_keeps_healthy_lane(tmp_path: Path) -> None:
    crashed = make_worker(tmp_path, "crash-request-chat")
    healthy = make_worker(tmp_path, "healthy-chat")
    crashed.start()
    healthy.start()
    group = WorkerGroup("chat", (crashed, healthy))
    try:
        with pytest.raises(WorkerCrashedError):
            _group_request(group, "generate", {"prompt": "crash"})

        assert _group_request(group, "generate", {"prompt": "still-running"})["text"] == (
            "generated:still-running"
        )
        assert group.ready
        status = group.status()
        assert set(status) == {
            "state",
            "ready",
            "replicas",
            "ready_replicas",
            "idle_replicas",
            "busy",
        }
        assert status["ready_replicas"] == 1
    finally:
        group.close()


def _group_request(group: WorkerGroup, operation: str, payload: dict[str, str]) -> object:
    with group.acquire_session() as session:
        return session.request(operation, payload)


def test_worker_crash_keeps_stderr_out_of_exception_and_status(tmp_path: Path) -> None:
    worker = make_worker(tmp_path, "crash-chat")
    worker.start()
    with pytest.raises(WorkerCrashedError) as failure:
        worker.request("generate", {"prompt": "boom"})
    assert "intentional fake crash" not in str(failure.value)
    assert "worker-secret" not in str(failure.value)
    assert str(tmp_path) not in str(failure.value)
    assert wait_for_stderr(worker, "intentional fake crash")
    assert worker.state == "failed"
    assert not worker.ready
    worker.close()


def test_worker_request_timeout_terminates_desynchronized_worker(tmp_path: Path) -> None:
    worker = make_worker(tmp_path, "slow-chat", request_timeout=0.1)
    worker.start()
    with pytest.raises(WorkerTimeoutError, match="timed out"):
        worker.request("generate", {"prompt": "wait"})
    assert worker.state == "failed"
    assert not worker.ready
    worker.close()


def test_worker_unknown_response_id_is_protocol_failure(tmp_path: Path) -> None:
    worker = make_worker(tmp_path, "bad-response-chat")
    worker.start()
    with pytest.raises(WorkerProtocolError, match="unknown id"):
        worker.request("generate", {"prompt": "bad"})
    assert worker.state == "failed"
    worker.close()


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("non-bool-ok-chat", "boolean ok"),
        ("missing-result-chat", "missing result"),
    ],
)
def test_worker_rejects_malformed_v2_response(tmp_path: Path, mode: str, message: str) -> None:
    worker = make_worker(tmp_path, mode)
    worker.start()
    with pytest.raises(WorkerProtocolError, match=message):
        worker.request("generate", {"prompt": "bad"})
    assert worker.state == "failed"
    worker.close()
