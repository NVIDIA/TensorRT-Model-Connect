# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real subprocess tests for bounded streaming, admission and cancellation."""
import threading
import time

import pytest

from trtmc_server.errors import WorkerProtocolError, WorkerSaturatedError, WorkerTimeoutError, WorkerCrashedError
from trtmc_server.registry import ModelRegistry
from trtmc_server.worker import WorkerGroup, WorkerLoadOptions, WorkerProcess

SCRIPT = '''#!/usr/bin/env python3
import json, sys, time
from pathlib import Path
print(json.dumps({"event":"ready", "protocol_version":2, "capabilities":["text_generation","streaming_text_continuation"]}), flush=True)
for line in sys.stdin:
 r=json.loads(line); i=r["id"]; op=r["op"]
 if op=="shutdown":
  print(json.dumps({"id":i,"ok":True,"result":{"status":"shutting_down"}}),flush=True); break
 if r.get("prompt")=="crash": sys.exit(1)
 if r.get("prompt")=="hang": time.sleep(30)
 if r.get("prompt")=="wrong-id": i="wrong"
 if op=="generate_stream":
  for n in range(r.get("count", 1)):
   print(json.dumps({"id":i,"ok":True,"event":"delta","result":{"text_delta":"中","token_count":1}}),flush=True)
  release=r.get("release")
  if release:
   while not Path(release).exists(): time.sleep(.005)
  print(json.dumps({"id":i,"ok":True,"event":"complete","result":{"text":"中"*r.get("count",1),"completion_tokens":r.get("count",1)}}),flush=True)
 else:
  print(json.dumps({"id":i,"ok":True,"result":{"text":"ok","completion_tokens":1}}),flush=True)
'''


def worker(tmp_path, timeout=3):
    binary = tmp_path / "worker"
    binary.write_text(SCRIPT)
    binary.chmod(0o755)
    result = WorkerProcess(name="replica", bundle=tmp_path / "unused.bundle", trtmc_binary=binary,
                           startup_timeout=3, request_timeout=timeout, load_options=WorkerLoadOptions())
    result.start()
    return result


def test_first_delta_precedes_completion_and_lane_stays_busy(tmp_path):
    native = worker(tmp_path)
    group = WorkerGroup("test", [native])
    session = group.acquire_session()
    release = tmp_path / "release"
    stream = session.stream({"prompt": "hello", "release": str(release)})
    try:
        assert stream.next() == {"text_delta": "中", "token_count": 1}
        assert not stream.future.done()
        with pytest.raises(WorkerSaturatedError):
            group.acquire_session()
        release.touch()
        assert stream.next() is None
        assert stream.future.result()["text"] == "中"
        session.close()
        with group.acquire_session() as second:
            assert second.request("generate", {"prompt": "next"})["text"] == "ok"
    finally:
        group.close()


def test_completion_wakes_consumer_without_polling(tmp_path, monkeypatch):
    native = worker(tmp_path)
    session = native.acquire_session()
    release = tmp_path / "release"
    stream = session.stream({"prompt": "hello", "release": str(release)})
    waiting, completed = threading.Event(), threading.Event()
    received = []
    consumer = None
    try:
        assert stream.next()["text_delta"] == "中"
        original_get = stream.events.get

        def get_without_polling(timeout=None):
            waiting.set()
            return original_get()  # Completion must notify, independently of a polling timeout.

        monkeypatch.setattr(stream.events, "get", get_without_polling)

        def consume():
            received.append(stream.next())
            completed.set()

        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()
        assert waiting.wait(3)
        assert not completed.is_set()
        release.touch()
        assert completed.wait(3), "native completion must wake the blocked stream consumer"
        assert received == [None]
        assert stream.future.result()["text"] == "中"
    finally:
        if consumer is not None and consumer.is_alive():
            stream.events.put(None)  # Also release the consumer if the assertion failed.
            consumer.join(3)
        session.close()
        native.close()


def test_disconnect_replaces_worker_only_after_exit(tmp_path):
    native = worker(tmp_path)
    group = WorkerGroup("test", [native])
    session = group.acquire_session()
    stream = session.stream({"prompt": "hello", "release": str(tmp_path / "never")})
    try:
        assert stream.next()["text_delta"] == "中"
        stream.cancel()
        assert stream.future.done()
        assert not native.ready
        assert native._process.poll() is not None
        session.close()
        wait_for_replacement(group)
        with group.acquire_session() as replacement:
            assert replacement._worker.pid != native.pid
            assert replacement.request("generate", {"prompt": "next"})["text"] == "ok"
        assert group.status()["idle_replicas"] == 1
    finally:
        group.close()


def test_slow_consumer_is_bounded_and_cancel_unblocks_producer(tmp_path):
    native = worker(tmp_path)
    session = native.acquire_session()
    stream = session.stream({"prompt": "hello", "count": 10000})
    try:
        assert stream.next()["text_delta"] == "中"
        assert stream.events.maxsize == 8 and native._response.maxsize == 8
        stream.cancel()
        assert stream.future.done() and not native.ready
    finally:
        session.close()
        native.close()


@pytest.mark.parametrize("prompt, error", [("wrong-id", WorkerProtocolError), ("hang", WorkerTimeoutError),
                                          ("crash", WorkerCrashedError)])
def test_stream_protocol_crash_and_timeout_retire_lane(tmp_path, prompt, error):
    native = worker(tmp_path, timeout=.2)
    session = native.acquire_session()
    stream = session.stream({"prompt": prompt})
    try:
        with pytest.raises(error):
            stream.next()
        assert not native.ready
    finally:
        session.close()
        native.close()


def test_protocol_versions_and_false_capability_are_rejected():
    for version in (0, 3, None, "2"):
        with pytest.raises(WorkerProtocolError):
            ModelRegistry._validate_ready({"event": "ready", "protocol_version": version,
                                          "capabilities": ["text_generation"]})


def wait_for_replacement(group, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if group.status()["idle_replicas"] == group.replicas:
            return
        time.sleep(.01)
    raise AssertionError("cancelled replica was not replaced")


def test_replacement_failure_does_not_loop_or_leak_processes(tmp_path):
    native = worker(tmp_path)
    group = WorkerGroup("test", [native])
    try:
        session = group.acquire_session()
        stream = session.stream({"prompt": "hello", "release": str(tmp_path / "never")})
        assert stream.next()["text_delta"] == "中"
        native.trtmc_binary.write_text("#!/bin/sh\nexit 1\n")
        stream.cancel()
        session.close()
        group._recovery.submit(lambda: None).result(timeout=5)
        assert not group.ready
        with pytest.raises(WorkerCrashedError):
            group.acquire_session()
        assert native._process.poll() is not None
        assert group._workers[0]._process.poll() is not None
    finally:
        group.close()


def test_shutdown_joins_a_replacement_without_leaking_workers(tmp_path, monkeypatch):
    native = worker(tmp_path)
    group = WorkerGroup("test", [native])
    started, release = threading.Event(), threading.Event()
    replacing = []
    original = native.replacement

    def delayed_replacement():
        replacement = original()
        replacing.append(replacement)
        started.set()
        assert release.wait(5)
        return replacement

    monkeypatch.setattr(native, "replacement", delayed_replacement)
    closer = None
    try:
        session = group.acquire_session()
        stream = session.stream({"prompt": "hello", "release": str(tmp_path / "never")})
        assert stream.next()["text_delta"] == "中"
        stream.cancel()
        session.close()
        assert started.wait(5)
        closer = threading.Thread(target=group.close)
        closer.start()
        release.set()
        closer.join(5)
        assert not closer.is_alive() and not group.ready
        assert native._process.poll() is not None
        assert replacing[0]._process.poll() is not None
    finally:
        release.set()
        if closer is not None:
            closer.join(5)
        group.close()
