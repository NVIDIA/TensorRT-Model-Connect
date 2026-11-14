# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real server executable with existing CPU-only SDK fixture DSOs."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def write_bundle(path: Path, task: str, family: str = "api_fixture") -> None:
    header = json.dumps({
        "format": 1, "family": family, "task": task, "backend": "fake",
        "sections": {"engine.plan": {"offset": 0, "length": 4}},
    }).encode("utf-8")
    path.write_bytes(b"BUNDLE\x01\x00" + struct.pack("<Q", len(header)) + header + b"PLAN")


def invoke(binary: Path, runtime: Path, bundle: Path, requests: list[dict]):
    completed = subprocess.run(
        [str(binary), "_serve-worker", str(bundle), "--runtime-root", str(runtime),
         "--kv-cache-size", "7"],
        input="".join(json.dumps(request) + "\n" for request in requests),
        capture_output=True, text=True, timeout=30, check=False,
    )
    records = [json.loads(line) for line in completed.stdout.splitlines()]
    return completed, records



def check_python_transport(binary: Path, runtime: Path, bundle: Path) -> None:
    # Exercise the real Python -> JSONL -> public SDK -> fixture DSO path.
    # Importing the transport does not require optional HTTP dependencies.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    from trtmc_server.worker import WorkerGroup, WorkerLoadOptions, WorkerProcess
    from trtmc_server.errors import WorkerCrashedError, WorkerProtocolError
    write_bundle(bundle, "text_continuation", "stream_fixture")
    worker = WorkerProcess(name="sdk-fixture", bundle=bundle, trtmc_binary=binary,
        startup_timeout=5, request_timeout=5, load_options=WorkerLoadOptions(runtime_root=str(runtime)))
    worker.start()
    group = WorkerGroup("sdk-fixture", [worker])
    try:
        assert worker.ready_payload["protocol_version"] == 2
        assert "streaming_text_continuation" in worker.ready_payload["capabilities"]
        with group.acquire_session() as session:
            stream = session.stream({"prompt": "hello", "config": {"suffix": "!"}})
            assert stream.next() == {"text_delta": "hello!", "token_count": 1}
            assert stream.next() is None
            final = stream.future.result()
            assert final["text"] == "hello!" and final["completion_tokens"] == 1
            assert final["model_call_ms"] >= 0
            assert final["timing_scope"] == "public_task_stream_wall_including_backpressure"
        with group.acquire_session() as session:
            assert session.request("generate", {"prompt": "next"})["text"] == "sync"
            # A malformed result must make a lane unavailable before asynchronous
            # termination, even if its native request has already completed.
            retirement = session.retire(WorkerProtocolError("bad result"))
            assert not worker.ready
            retirement.result(timeout=5)
            assert worker._process.poll() is not None
        try:
            group.acquire_session()
            raise AssertionError("retired lane was reused")
        except WorkerCrashedError:
            pass
    finally:
        group.close()



def check_parent_death(binary: Path, runtime: Path, bundle: Path) -> None:
    if not sys.platform.startswith("linux"):
        return
    import ctypes
    import select
    # Adopt and reap the worker after its frontend dies; never leave a zombie
    # behind in the validation host's PID namespace.
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    code = """import json,os,subprocess,sys
worker=subprocess.Popen(sys.argv[1:],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,start_new_session=True)
ready=json.loads(worker.stdout.readline())
worker.stdin.write(json.dumps({'id':'wait','op':'generate_stream','prompt':'hello','config':{'wait_for_cancel':True}})+'\\n')
worker.stdin.flush()
print(worker.pid,flush=True)
sys.stdin.readline()
os._exit(0)
"""
    parent = None
    pid = None
    try:
        parent = subprocess.Popen([sys.executable, "-c", code, str(binary), "_serve-worker",
            str(bundle), "--runtime-root", str(runtime)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert select.select([parent.stdout], [], [], 5)[0], "frontend did not report worker readiness"
        pid = int(parent.stdout.readline())
        parent.stdin.write("exit\n")
        parent.stdin.flush()
        assert parent.wait(timeout=5) == 0
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            exited, status = os.waitpid(pid, os.WNOHANG)
            if exited:
                assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == 15
                pid = None
                return
            time.sleep(.01)
        raise AssertionError("native generation outlived its frontend")
    finally:
        if parent is not None:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)
            for stream in (parent.stdin, parent.stdout, parent.stderr):
                if stream is not None:
                    stream.close()
        if pid is not None:
            try:
                os.kill(pid, 9)
                os.waitpid(pid, 0)
            except ProcessLookupError:
                pass
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="trtmc-server-sdk-") as temporary:
        bundle = Path(temporary) / "model.bundle"
        write_bundle(bundle, "text_continuation")
        requests = [
            {"id": "first", "op": "generate", "prompt": "hello"},
            {"id": "bad", "op": "generate", "prompt": "hello", "config": {"unknown": 1}},
            {"id": "last", "op": "generate", "prompt": "hi", "config": {"suffix": "?"}},
            {"id": "stop", "op": "shutdown"},
        ]
        completed, records = invoke(args.binary, args.runtime_root, bundle, requests)
        assert completed.returncode == 0, completed.stderr
        assert len(records) == 5
        assert records[0] == {
            "event": "ready", "protocol_version": 2,
            "capabilities": ["text_generation"], "default_max_new_tokens": 4,
        }
        assert records[1]["id"] == "first" and records[1]["ok"] is True
        timing = records[1]["result"].pop("model_call_ms")
        assert isinstance(timing, (int, float)) and timing >= 0
        assert records[1]["result"].pop("timing_scope") == "public_task_call_wall"
        assert records[1]["result"] == {
            "text": "hello!|eos", "completion_tokens": 2,
            "setup_ms": 7.0, "prefill_ms": 0.75, "decode_ms": 4.0,
        }
        assert records[2]["ok"] is False
        assert records[2]["error"]["type"] == "invalid_request_error"
        assert records[3]["id"] == "last" and records[3]["result"]["text"] == "hi?|eos"
        assert records[4]["result"]["status"] == "shutting_down"

        for task, expected in (
            ("conditional_text_generation", "conditional:hello!"),
            ("text_translation", "translation:fixed-src->en:hello!"),
        ):
            write_bundle(bundle, task, "text_fixture")
            completed, records = invoke(args.binary, args.runtime_root, bundle, [requests[0]])
            assert completed.returncode == 0, completed.stderr
            assert records[0]["default_max_new_tokens"] == 128
            assert records[1]["result"]["text"] == expected

        write_bundle(bundle, "must_not_run")
        completed, records = invoke(args.binary, args.runtime_root, bundle, requests)
        assert completed.returncode == 1
        assert len(records) == 2
        assert records[1]["id"] == "first" and records[1]["ok"] is False
        assert records[1]["error"] == {
            "type": "runtime_error", "message": "native worker operation failed",
        }
        assert "the fixture execution must not be reached" not in completed.stdout

        check_python_transport(args.binary, args.runtime_root, bundle)
        check_parent_death(args.binary, args.runtime_root, bundle)

        write_bundle(bundle, "disabled")
        completed, records = invoke(args.binary, args.runtime_root, bundle, requests)
        assert completed.returncode != 0 and not records


if __name__ == "__main__":
    main()
