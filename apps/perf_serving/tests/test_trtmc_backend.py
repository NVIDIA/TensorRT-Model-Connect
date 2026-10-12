# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import stat
import sys
import textwrap

import pytest

from trtmc_perf_serving.backends.base import BackendError, BackendUnavailable
from trtmc_perf_serving.backends.trtmc import TrtmcWorkerBackend

# Emulates `trtmc_benchmark_worker --serve SESSION.json`.
FAKE_WORKER = textwrap.dedent("""\
    #!{python}
    import json, os, sys
    session = json.load(open(sys.argv[2]))
    if session["bundle"] == "missing.bundle":
        print(json.dumps({{"event": "failed", "error": "cannot open bundle"}}), flush=True)
        sys.exit(1)
    print("stray library output", file=sys.stderr)
    print(json.dumps({{"event": "ready", "operation": session["operation"], "task": "t", "selected_task": "t",
                      "timing_scope": "public_task_call_wall", "asset_loading_included": False,
                      "load_ms": 1.5}}), flush=True)
    for line in sys.stdin:
        message = json.loads(line)
        if message.get("type") == "shutdown":
            print(json.dumps({{"id": message["id"], "ok": True}}), flush=True)
            break
        request = message["request"]
        if request.get("crash"):
            sys.exit(3)
        if request.get("invalid"):
            print(json.dumps({{"id": message["id"], "ok": False, "error": "invalid prompt"}}), flush=True)
            continue
        print(json.dumps({{"id": message["id"], "ok": True, "observation": {{
            "runtime_e2e_wall_ms": 7.25, "echo": request, "artifact_base": message["artifact_base"],
            "pid": os.getpid()}}}}),
            flush=True)
""")


@pytest.fixture
def worker(tmp_path):
    path = tmp_path / "fake_worker"
    path.write_text(FAKE_WORKER.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def session(bundle="model.bundle"):
    return {"schema_version": 2, "bundle": bundle, "operation": "classify", "request": {}}


def test_invoke_returns_worker_wall_time_and_observation(tmp_path, worker):
    backend = TrtmcWorkerBackend(worker=worker, session=session(), scratch=tmp_path)
    try:
        assert backend.describe()["load_ms"] == 1.5
        result = backend.invoke({"image_path": "/x.png"}, tmp_path / "out")
        assert result.model_call_ms == 7.25
        assert result.observation["echo"] == {"image_path": "/x.png"}
        assert "runtime_e2e_wall_ms" not in result.observation
    finally:
        backend.close()


def test_request_error_keeps_worker_available(tmp_path, worker):
    backend = TrtmcWorkerBackend(worker=worker, session=session(), scratch=tmp_path)
    try:
        with pytest.raises(BackendError, match="invalid prompt"):
            backend.invoke({"invalid": True}, tmp_path / "out")
        assert backend.invoke({}, tmp_path / "out").model_call_ms == 7.25
    finally:
        backend.close()


def test_worker_crash_retires_backend(tmp_path, worker):
    backend = TrtmcWorkerBackend(worker=worker, session=session(), scratch=tmp_path)
    try:
        with pytest.raises(BackendUnavailable):
            backend.invoke({"crash": True}, tmp_path / "out")
        with pytest.raises(BackendUnavailable):
            backend.invoke({}, tmp_path / "out")
    finally:
        backend.close()


def test_load_failure_is_reported(tmp_path, worker):
    with pytest.raises(BackendUnavailable, match="cannot open bundle"):
        TrtmcWorkerBackend(worker=worker, session=session("missing.bundle"), scratch=tmp_path)


def test_isolate_requests_uses_a_fresh_worker_per_request(tmp_path, worker):
    backend = TrtmcWorkerBackend(worker=worker, session=session(), scratch=tmp_path, isolate_requests=True)
    try:
        pids = [backend.invoke({}, tmp_path / "out").observation["pid"] for _ in range(3)]
        assert len(set(pids)) == 3
        assert backend.describe()["isolate_requests"] is True
    finally:
        backend.close()
