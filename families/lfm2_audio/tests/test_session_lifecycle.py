# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the production dialogue lifecycle on CPU with deterministic model callbacks.

The anonymous Session lifecycle statements are compiled unchanged from pipeline.cpp.
Fake Engines supplies deterministic model callbacks. A forwarding condition variable
can pause the reader between its predicate and wait; test-only private visibility
also permits cancellation barriers. No CUDA headers or GPU are required.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]


def extract(source, begin, end):
    assert source.count(begin) == 1, f"production boundary is not unique: {begin}"
    start = source.index(begin)
    stop = source.index(end, start)
    return source[start:stop]


def build_probe(directory, source_path=FAMILY / "runtime/pipeline.cpp"):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("CPU dialogue lifecycle tests require a C++17 compiler")
    source = source_path.read_text()
    # Extract whole definitions, never replace lifecycle statements or duplicate
    # their implementation in the fixture. A moved production boundary fails
    # explicitly so this test cannot silently execute stale copied code.
    (directory / "session_options.inc").write_text(
        extract(source, "const ConfigField kFields[] =", "std::string voice_prompt(")
        + extract(source, "struct AudioSegment {", "} // namespace")
    )
    session = extract(source, "class Session final :", "} // namespace")
    assert session.count("std::condition_variable cv_;") == 1
    # Change only the synchronization object's type. Its forwarding wrapper
    # preserves condition-variable semantics and exposes the predicate/wait gap.
    (directory / "session_under_test.inc").write_text(
        session.replace("std::condition_variable cv_;", "ScheduledConditionVariable cv_;")
    )
    output = directory / "session_lifecycle"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pthread",
            "-I",
            str(ROOT),
            "-I",
            str(ROOT / "core/runtime/include"),
            "-I",
            str(directory),
            str(FAMILY / "tests/cpp/session_lifecycle.cpp"),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return output


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    return build_probe(tmp_path_factory.mktemp("lfm2-audio-session"))


@pytest.mark.parametrize(
    "case",
    [
        "finish-completed",
        "finish-pending",
        "finish-deferred",
        "finish-empty",
        "cancel-startup",
        "cancel-delivery",
        "stop-reader-handoff",
        "fail-assistant",
        "fail-generation",
        "fail-decode",
        "fail-after-media",
        "utf8-complete",
        "utf8-truncated",
        "utf8-invalid",
    ],
)
def test_production_session_lifecycle(probe, case):
    result = subprocess.run([str(probe), case], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
