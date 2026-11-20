# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU process lifecycle checks for the Qwen-owned engine workers."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("tensorrt")

from families.qwen import build_worker, model  # noqa: E402
from families.qwen.tests.test_weightless_policy import _config  # noqa: E402


_FAKE_WORKER = '''
import argparse
import json
import os
from pathlib import Path
import sys
import time
from families.qwen.build_worker import _watch_parent

parser = argparse.ArgumentParser()
for name in ("model-dir", "config", "role", "max-sequence-length", "cuda-device", "plan", "metadata"):
    parser.add_argument("--" + name, required=True)
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--watch-parent", action="store_true")
args = parser.parse_args()
with _watch_parent(args.watch_parent):
    assert Path(args.model_dir).is_dir()
    configuration = json.loads(Path(args.config).read_text())
    assert configuration["raw"]["_resolved_build_precision"] == "bf16"
    records = Path(os.environ["QWEN_WORKER_TEST_RECORDS"])
    (records / (args.role + ".json")).write_text(json.dumps({
        "pid": os.getpid(), "cuda_device": int(args.cuda_device), "role": args.role,
        "plan": args.plan,
    }))
    mode = os.environ.get("QWEN_WORKER_TEST_MODE", "success")
    print("[TRT] worker diagnostic " + json.dumps({"role": args.role, "fake": True}), flush=True)
    if mode == "fail_" + args.role:
        print("intentional " + args.role + " worker failure", flush=True)
        sys.exit(7)
    if mode == "hang" or mode.startswith("fail_"):
        time.sleep(60)
    native = {"native_kv_cache": mode != "invalid_native", "native_kv_contract_version": 1}
    if mode == "metadata_mismatch" and args.role == "decode":
        native["unexpected"] = True
    plan = args.role.encode() if mode != "empty_plan" else b""
    Path(args.plan).write_bytes(plan)
    Path(args.metadata).write_text(json.dumps({
        "role": args.role, "plan_bytes": len(plan) + (mode == "wrong_size"),
        "native_kv_cache_metadata": native,
    }))
'''


@pytest.fixture
def worker_harness(monkeypatch, tmp_path):
    fake = tmp_path / "fake_worker.py"
    fake.write_text(_FAKE_WORKER)
    private = tmp_path / "private"
    private.mkdir()
    records = tmp_path / "records"
    records.mkdir()
    checkpoint = tmp_path / "model with spaces $()"
    checkpoint.mkdir()
    sentinel = tmp_path / "unrelated"
    sentinel.write_text("keep")
    monkeypatch.setenv("QWEN_WORKER_TEST_RECORDS", str(records))
    monkeypatch.setenv("QWEN_WORKER_TEST_MODE", "success")
    # A broken system temp location must not affect output-local staging.
    monkeypatch.setattr(model.tempfile, "tempdir", str(tmp_path / "missing-system-tmp"))
    monkeypatch.setattr(model, "_current_cuda_device", lambda: 3)
    processes, commands = [], []
    state = SimpleNamespace(fail_launch=False)

    def popen(command, **kwargs):
        assert isinstance(command, list)
        assert command[:6] == model._worker_command()
        assert "shell" not in kwargs
        if state.fail_launch and processes:
            raise OSError("intentional second worker launch failure")
        commands.append(command)
        roots = json.dumps([*json.loads(command[4]), str(fake.parent)])
        process = subprocess.Popen([*command[:4], roots, fake.stem, *command[6:]], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(model, "subprocess", SimpleNamespace(
        Popen=popen, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired))
    config = _config()
    config.raw["_resolved_build_precision"] = "bf16"
    return SimpleNamespace(
        fake=fake, private=private, records=records, checkpoint=checkpoint,
        sentinel=sentinel, processes=processes, commands=commands, state=state, config=config,
    )


def _assert_joined_and_clean(harness):
    assert harness.processes
    assert all(process.poll() is not None for process in harness.processes)
    assert all(process.stdin is not None and process.stdin.closed for process in harness.processes)
    assert not list(harness.private.iterdir())
    assert harness.sentinel.read_text() == "keep"


def test_isolated_worker_ignores_checkpoint_python_modules(tmp_path):
    # Both Python startup hooks and the worker module itself must resolve outside
    # an untrusted current directory, even if the caller has it on PYTHONPATH.
    package = tmp_path / "families" / "qwen"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").touch()
    (package / "__init__.py").touch()
    for path in (tmp_path / "sitecustomize.py", package / "build_worker.py"):
        path.write_text("from pathlib import Path\nPath('shadow-imported').touch()\n")
    result = subprocess.run(
        [*model._worker_command(), "--help"], cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(tmp_path)), capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "--cuda-device" in result.stdout
    assert not (tmp_path / "shadow-imported").exists()


def test_workers_preserve_device_roles_and_metadata(worker_harness, capsys):
    harness = worker_harness
    result = model._build_weightless_native_engines(
        harness.checkpoint, harness.config, 16384, output_dir=harness.private, verbose=True)
    assert result == (b"prefill", b"decode")
    assert harness.config.raw["_native_kv_cache_metadata"] == {
        "native_kv_cache": True, "native_kv_contract_version": 1}
    rows = [json.loads((harness.records / f"{role}.json").read_text()) for role in ("prefill", "decode")]
    assert all(row["cuda_device"] == 3 for row in rows)
    assert rows[0]["pid"] != rows[1]["pid"]
    assert all("--verbose" in command for command in harness.commands)
    assert capsys.readouterr().err.count("[TRT] worker diagnostic ") == 2
    _assert_joined_and_clean(harness)


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_failed_worker_stops_and_joins_its_peer(worker_harness, monkeypatch, role):
    harness = worker_harness
    monkeypatch.setenv("QWEN_WORKER_TEST_MODE", f"fail_{role}")
    with pytest.raises(RuntimeError, match=f"intentional {role} worker failure"):
        model._build_weightless_native_engines(harness.checkpoint, harness.config, 16384, output_dir=harness.private, verbose=False)
    assert len(harness.processes) == 2
    statuses = {command[command.index("--role") + 1]: process.returncode
                for command, process in zip(harness.commands, harness.processes, strict=True)}
    assert statuses[role] == 7
    peer = "decode" if role == "prefill" else "prefill"
    assert statuses[peer] is not None and statuses[peer] != 0
    _assert_joined_and_clean(harness)


@pytest.mark.parametrize("mode,reason", [
    ("invalid_native", "invalid native KV metadata"),
    ("metadata_mismatch", "different native KV metadata"),
    ("wrong_size", "incomplete plan"), ("empty_plan", "incomplete plan"),
])
def test_invalid_worker_artifacts_are_rejected(worker_harness, monkeypatch, mode, reason):
    harness = worker_harness
    monkeypatch.setenv("QWEN_WORKER_TEST_MODE", mode)
    with pytest.raises(RuntimeError, match=reason):
        model._build_weightless_native_engines(harness.checkpoint, harness.config, 16384, output_dir=harness.private, verbose=False)
    assert "_native_kv_cache_metadata" not in harness.config.raw
    _assert_joined_and_clean(harness)


def test_second_launch_failure_joins_first_worker(worker_harness):
    harness = worker_harness
    harness.state.fail_launch = True
    with pytest.raises(RuntimeError, match="intentional second worker launch failure"):
        model._build_weightless_native_engines(harness.checkpoint, harness.config, 16384, output_dir=harness.private, verbose=False)
    assert len(harness.processes) == 1
    _assert_joined_and_clean(harness)


def test_cancellation_joins_workers_and_preserves_interrupt(worker_harness, monkeypatch):
    harness = worker_harness
    monkeypatch.setenv("QWEN_WORKER_TEST_MODE", "hang")

    def interrupt(delay):
        raise KeyboardInterrupt("intentional cancellation")

    monkeypatch.setattr(model, "time", SimpleNamespace(sleep=interrupt))
    with pytest.raises(KeyboardInterrupt, match="intentional cancellation"):
        model._build_weightless_native_engines(harness.checkpoint, harness.config, 16384, output_dir=harness.private, verbose=False)
    assert len(harness.processes) == 2
    _assert_joined_and_clean(harness)


def _stdin_parent_source(harness):
    return f'''
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from families.qwen import model
from families.qwen.config import ModelConfig
model._current_cuda_device = lambda: 3
model.tempfile.tempdir = {str(harness.private / "missing-system-tmp")!r}
def popen(command, **kwargs):
    roots = json.dumps([*json.loads(command[4]), {str(harness.fake.parent)!r}])
    return subprocess.Popen([*command[:4], roots, {harness.fake.stem!r}, *command[6:]], **kwargs)
model.subprocess = SimpleNamespace(Popen=popen, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT,
                                   TimeoutExpired=subprocess.TimeoutExpired)
config = ModelConfig(**json.loads({json.dumps(asdict(harness.config))!r}))
prefill, decode = model._build_weightless_native_engines(
    Path({str(harness.checkpoint)!r}), config, 16384, output_dir=Path({str(harness.private)!r}), verbose=False)
print(json.dumps([prefill.decode(), decode.decode(), config.raw["_native_kv_cache_metadata"]]))
'''


def test_invocation_from_stdin_needs_no_main_guard(worker_harness):
    harness = worker_harness
    source = _stdin_parent_source(harness)
    result = subprocess.run([sys.executable, "-"], input=source, text=True,
                            capture_output=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["prefill", "decode", {
        "native_kv_cache": True, "native_kv_contract_version": 1}]
    assert result.stderr.count("[TRT] worker diagnostic ") == 2
    assert not list(harness.private.iterdir())
    assert harness.sentinel.read_text() == "keep"


def _process_is_running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # An orphan may remain a zombie until the container's PID 1 reaps it.
    # That process has exited and cannot retain an active CUDA context.
    stat = Path(f"/proc/{pid}/stat")
    try:
        state = stat.read_text().rsplit(") ", 1)[1].split()[0]
    except FileNotFoundError:
        return not Path("/proc").is_dir()
    return state not in {"Z", "X"}


@pytest.mark.skipif(os.name != "posix", reason="this check terminates the caller with POSIX SIGTERM")
def test_parent_sigterm_stops_both_worker_processes(worker_harness, monkeypatch):
    harness = worker_harness
    monkeypatch.setenv("QWEN_WORKER_TEST_MODE", "hang")
    parent = subprocess.Popen(
        [sys.executable, "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
    )
    pids = []
    try:
        parent.stdin.write(_stdin_parent_source(harness).encode())
        parent.stdin.close()
        parent.stdin = None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                records = [json.loads((harness.records / f"{role}.json").read_text())
                           for role in ("prefill", "decode")]
            except (FileNotFoundError, json.JSONDecodeError):
                if parent.poll() is not None:
                    break
                time.sleep(0.02)
            else:
                pids = [record["pid"] for record in records]
                break
        assert len(pids) == 2, parent.stderr.read().decode() if parent.poll() is not None else "workers did not start"
        assert all(_process_is_running(pid) for pid in pids)
        parent.terminate()
        assert parent.wait(timeout=5) == -signal.SIGTERM
        deadline = time.monotonic() + 10
        while any(_process_is_running(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not any(_process_is_running(pid) for pid in pids)
        assert harness.sentinel.read_text() == "keep"
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait()
        for pid in pids:
            if _process_is_running(pid):
                os.kill(pid, signal.SIGTERM)
        parent.stdout.close()
        parent.stderr.close()


def test_worker_selects_device_before_loading_and_building(monkeypatch, tmp_path):
    from cuda.bindings import runtime as cudart

    calls = []
    monkeypatch.setattr(cudart, "cudaSetDevice", lambda device: calls.append(("device", device)) or (cudart.cudaError_t.cudaSuccess,))
    config = _config()
    config.raw["_resolved_build_precision"] = "bf16"
    config_path, plan_path, metadata_path = (tmp_path / name for name in ("config.json", "engine.plan", "metadata.json"))
    config_path.write_text(json.dumps(asdict(config)))
    weights = object()

    def load(self, directory, config, *, precision):
        calls.append(("load", directory, precision))
        return weights

    def build(self, config, supplied, capacity, **kwargs):
        assert supplied is weights and capacity == 16384
        from families.qwen.weightless import _SESSION

        assert _SESSION.get() is not None
        assert kwargs == {"precision": "bf16", "verbose": False}
        calls.append(("build", config.raw["_decoder_engine_role"]))
        config.raw["_native_kv_cache_metadata"] = {"native_kv_cache": True, "native_kv_contract_version": 1}
        return b"complete plan"

    monkeypatch.setattr(model._QwenModel, "load_weights", load)
    monkeypatch.setattr(model._QwenModel, "build_engine", build)
    monkeypatch.setattr(sys, "argv", ["worker", "--model-dir", str(tmp_path), "--config", str(config_path),
                                    "--role", "decode", "--max-sequence-length", "16384", "--cuda-device", "3",
                                    "--plan", str(plan_path), "--metadata", str(metadata_path)])
    assert build_worker.main() == 0
    assert calls == [("device", 3), ("load", str(tmp_path), "bf16"), ("build", "decode")]
    assert plan_path.read_bytes() == b"complete plan"
    assert json.loads(metadata_path.read_text()) == {
        "role": "decode", "plan_bytes": len(b"complete plan"),
        "native_kv_cache_metadata": {"native_kv_cache": True, "native_kv_contract_version": 1},
    }
