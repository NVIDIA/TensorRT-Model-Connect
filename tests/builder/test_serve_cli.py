# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import websockets

from tensorrt_model_connect.serve.cli import (
    _RedactAccessToken,
    bind_socket,
    build_parser,
    is_loopback_host,
    main as serve_main,
    parse_model_assignment,
    parse_replica_assignment,
    resolve_trtmc_binary,
    validate_bind_policy,
)


FAKE_TRTMC = Path(__file__).with_name("fake_serve_worker.py")
REPOSITORY = Path(__file__).resolve().parents[2]


def test_model_assignment_and_bind_policy(tmp_path: Path) -> None:
    bundle = tmp_path / "model=revision.bundle"
    bundle.write_bytes(b"fixture")
    spec = parse_model_assignment(f"asr={bundle}", kind="transcription")
    assert spec.name == "asr"
    assert spec.bundle == bundle
    assert spec.kind == "transcription"

    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("::1")
    assert is_loopback_host("localhost")
    assert not is_loopback_host("0.0.0.0")
    with pytest.raises(ValueError, match="loopback-only"):
        validate_bind_policy("0.0.0.0")
    validate_bind_policy("127.0.0.1")


def test_prebound_port_zero_returns_actual_port() -> None:
    listener = bind_socket("127.0.0.1", 0)
    try:
        assert listener.getsockname()[0] == "127.0.0.1"
        assert listener.getsockname()[1] > 0
        assert listener.getblocking() is False
    finally:
        listener.close()


def test_model_replicas_are_explicit_and_default_to_one() -> None:
    assert build_parser().parse_args([]).model_replicas == []
    assert parse_replica_assignment("chat=3") == ("chat", 3)
    for value in ("chat=0", "chat=-1", "chat=many", "chat", "=2"):
        with pytest.raises(ValueError, match="model-replicas|positive integer"):
            parse_replica_assignment(value)


def test_cli_log_level_is_a_fixed_non_debug_allowlist() -> None:
    parser = build_parser()
    for level in ("critical", "error", "warning", "info"):
        assert parser.parse_args(["--log-level", level]).log_level == level
    for level in ("debug", "trace", "notset"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--log-level", level])


def test_log_filter_recursively_redacts_transport_credentials() -> None:
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        {
            "headers": [
                ("Authorization", "Bearer authorization-secret"),
                (b"cookie", b"session=cookie-secret"),
            ],
            "url": "ws://localhost/v1/realtime?access_token=query-secret&intent=transcription",
            "nested": {"ACCESS_TOKEN": "mapping-secret", "safe": "visible"},
        },
        (
            b"authorization: Bearer bytes-secret",
            ["cookie: sid=sequence-secret", {"access-token": "hyphen-secret"}],
        ),
        None,
    )
    assert _RedactAccessToken().filter(record)
    rendered = repr((record.msg, record.args))
    for secret in (
        "authorization-secret",
        "cookie-secret",
        "query-secret",
        "mapping-secret",
        "bytes-secret",
        "sequence-secret",
        "hyphen-secret",
    ):
        assert secret not in rendered
    assert rendered.count("<redacted>") >= 7
    assert "visible" in rendered


def test_cli_startup_errors_do_not_expose_bundle_or_binary_paths(tmp_path: Path) -> None:
    missing_bundle = tmp_path / "private" / "missing.bundle"
    with pytest.raises(ValueError, match="does not exist or is not a file") as bundle_failure:
        parse_model_assignment(f"chat={missing_bundle}", kind="chat")
    assert str(tmp_path) not in str(bundle_failure.value)
    assert str(missing_bundle) not in str(bundle_failure.value)

    missing_binary = tmp_path / "private" / "missing-trtmc"
    with pytest.raises(ValueError, match="does not exist or is not a file") as binary_failure:
        resolve_trtmc_binary(str(missing_binary))
    assert str(tmp_path) not in str(binary_failure.value)
    assert str(missing_binary) not in str(binary_failure.value)


def test_cli_rejects_duplicate_and_unknown_replica_assignments(tmp_path: Path) -> None:
    bundle = tmp_path / "chat.bundle"
    bundle.write_bytes(b"chat")
    common = [
        "--chat-model",
        f"chat={bundle}",
        "--trtmc-binary",
        str(FAKE_TRTMC),
    ]
    with pytest.raises(SystemExit):
        serve_main([*common, "--model-replicas", "missing=2"])
    with pytest.raises(SystemExit):
        serve_main(
            [
                *common,
                "--model-replicas",
                "chat=2",
                "--model-replicas",
                "chat=3",
            ]
        )


def test_cli_port_zero_emits_single_machine_readable_ready_record(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "asr.bundle"
    bundle.write_bytes(b"fixture")
    environment = dict(os.environ)
    python_path = str(REPOSITORY / "python")
    environment["PYTHONPATH"] = (
        python_path
        if not environment.get("PYTHONPATH")
        else f"{python_path}{os.pathsep}{environment['PYTHONPATH']}"
    )
    environment["TRTMC_SERVE_TOKEN"] = "environment-token"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tensorrt_model_connect.serve.cli",
            "--transcription-model",
            f"asr={bundle}",
            "--trtmc-binary",
            str(FAKE_TRTMC),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--require-streaming-transcription",
            "asr",
            "--model-replicas",
            "asr=2",
            "--backend-dir",
            "/opt/backend",
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
        ],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    records: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(target=lambda: records.put(process.stdout.readline()), daemon=True)
    reader.start()
    try:
        line = records.get(timeout=10)
        ready = json.loads(line)
        assert ready["event"] == "ready"
        assert ready["host"] == "127.0.0.1"
        assert ready["port"] > 0
        assert "pid" not in ready
        assert ready["models"] == ["asr"]
        worker_pids = _wait_for_child_pids(process.pid, expected=2)

        with pytest.raises(urllib.error.HTTPError) as unauthorized:
            urllib.request.urlopen(f"http://127.0.0.1:{ready['port']}/v1/models", timeout=3)
        assert unauthorized.value.code == 401

        request = urllib.request.Request(
            f"http://127.0.0.1:{ready['port']}/v1/models",
            headers={"Authorization": "Bearer environment-token"},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            model = json.load(response)["data"][0]
            assert model["id"] == "asr"
            assert model["metadata"] == {
                "default_max_new_tokens": 64,
                "max_cache_length": 2048,
                "streaming_transcription": True,
            }
        with urllib.request.urlopen(
            f"http://127.0.0.1:{ready['port']}/healthz", timeout=3
        ) as response:
            assert json.load(response) == {"status": "ok"}
        with pytest.raises(urllib.error.HTTPError) as anonymous_ready:
            urllib.request.urlopen(f"http://127.0.0.1:{ready['port']}/readyz", timeout=3)
        assert anonymous_ready.value.code == 401
        ready_request = urllib.request.Request(
            f"http://127.0.0.1:{ready['port']}/readyz",
            headers={"Authorization": "Bearer environment-token"},
        )
        with urllib.request.urlopen(ready_request, timeout=3) as response:
            worker_status = json.load(response)["models"]["asr"]
            assert set(worker_status) == {
                "ready",
                "replicas",
                "ready_replicas",
                "idle_replicas",
                "busy",
            }
            assert worker_status["replicas"] == 2
            assert worker_status["ready_replicas"] == 2

        async def verify_websocket() -> None:
            uri = (
                f"ws://127.0.0.1:{ready['port']}/v1/realtime"
                "?intent=transcription&access_token=environment-token"
            )
            async with websockets.connect(uri, origin="http://localhost:4173") as websocket:
                assert json.loads(await websocket.recv())["type"] == "session.created"
                await websocket.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(b"\x01\x00").decode(),
                        }
                    )
                )
                delta = json.loads(await websocket.recv())
                assert delta["type"] == ("conversation.item.input_audio_transcription.delta")
                assert delta["transcript"] == "1 samples"
                await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
                completed = json.loads(await websocket.recv())
                assert completed["type"] == (
                    "conversation.item.input_audio_transcription.completed"
                )

        asyncio.run(verify_websocket())

        process.terminate()
        process.wait(timeout=5)
        assert process.stdout.read() == ""
        stderr = process.stderr.read()
        assert "environment-token" not in stderr
        assert "access_token=<redacted>" in stderr
        for worker_pid in worker_pids:
            _assert_pid_disappears(worker_pid)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_parent_liveness_stdin_eof_gracefully_stops_server_and_worker(
    tmp_path: Path,
) -> None:
    process, _ready = _start_test_server(tmp_path, "parent-eof", parent_liveness=True)
    assert process.stdin is not None
    try:
        worker_pid = _wait_for_child_pids(process.pid, expected=1)[0]
        process.stdin.close()
        assert process.wait(timeout=5) == 0
        _assert_pid_disappears(worker_pid)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_server_sigkill_does_not_leave_native_worker(tmp_path: Path) -> None:
    process, _ready = _start_test_server(tmp_path, "sigkill", parent_liveness=False)
    try:
        worker_pid = _wait_for_child_pids(process.pid, expected=1)[0]
        process.kill()
        process.wait(timeout=5)
        _assert_pid_disappears(worker_pid)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def _start_test_server(
    tmp_path: Path, name: str, *, parent_liveness: bool
) -> tuple[subprocess.Popen[str], dict[str, object]]:
    bundle = tmp_path / f"{name}-asr.bundle"
    bundle.write_bytes(b"fixture")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY / "python")
    environment["TRTMC_SERVE_TOKEN"] = "environment-token"
    command = [
        sys.executable,
        "-m",
        "tensorrt_model_connect.serve.cli",
        "--transcription-model",
        f"asr={bundle}",
        "--trtmc-binary",
        str(FAKE_TRTMC),
        "--port",
        "0",
    ]
    if parent_liveness:
        command.append("--parent-liveness-stdin")
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY,
        env=environment,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    records: queue.Queue[str] = queue.Queue()
    threading.Thread(target=lambda: records.put(process.stdout.readline()), daemon=True).start()
    try:
        line = records.get(timeout=10)
    except queue.Empty:
        process.kill()
        process.wait(timeout=5)
        stderr = process.stderr.read() if process.stderr is not None else ""
        raise AssertionError(f"server did not emit ready: {stderr}") from None
    if not line:
        stderr = process.stderr.read() if process.stderr is not None else ""
        raise AssertionError(f"server exited before ready: {stderr}")
    return process, json.loads(line)


def _wait_for_child_pids(parent_pid: int, *, expected: int) -> list[int]:
    task_root = Path(f"/proc/{parent_pid}/task")
    deadline = time.monotonic() + 3
    children: list[int] = []
    while time.monotonic() < deadline:
        try:
            children = sorted(
                {
                    int(value)
                    for children_file in task_root.glob("*/children")
                    for value in children_file.read_text().split()
                }
            )
        except FileNotFoundError:
            children = []
        if len(children) == expected:
            return children
        time.sleep(0.02)
    raise AssertionError(
        f"server {parent_pid} has {len(children)} direct children; expected {expected}"
    )


def _assert_pid_disappears(pid: int) -> None:
    deadline = time.monotonic() + 3
    while _pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _pid_exists(pid), f"native worker {pid} survived parent termination"


def _pid_exists(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", maxsplit=1)[1].split()[0]
        if state == "Z":
            return False
    except (FileNotFoundError, IndexError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
