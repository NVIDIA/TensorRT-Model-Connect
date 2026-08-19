# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from tensorrt_model_connect.models.sana_wm import native_plugin_builder


def test_native_plugin_build_is_serialized_per_cache_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    digest_barrier = threading.Barrier(2)
    calls: list[str] = []
    calls_lock = threading.Lock()

    def source_digest(_source_dir: Path) -> str:
        digest_barrier.wait(timeout=5)
        return "shared-cache-key"

    def run(command: list[str], **_kwargs) -> subprocess.CompletedProcess:
        action = command[1]
        with calls_lock:
            calls.append(action)
        if action == "-S":
            time.sleep(0.1)
        elif action == "--build":
            output = Path(command[2]) / "libtrtmc_sana_wm_native_plugin.so"
            output.write_bytes(b"plugin")
        return subprocess.CompletedProcess(command, 0)

    fake_torch = SimpleNamespace(utils=SimpleNamespace(cmake_prefix_path="/tmp/fake-torch-cmake"))
    monkeypatch.setenv(native_plugin_builder._BUILD_DIR_ENV, str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(native_plugin_builder, "_source_digest", source_digest)
    monkeypatch.setattr(native_plugin_builder.subprocess, "run", run)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(lambda _index: native_plugin_builder.ensure_native_plugin(), range(2))
        )

    assert results == [results[0], results[0]]
    assert calls == ["-S", "--build"]
