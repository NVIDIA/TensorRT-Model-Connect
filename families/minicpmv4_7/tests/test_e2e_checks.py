# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the E2E helpers: case selection, the reference interpreter, failure capture, and the
stop and logit-gap rules that the assertions and the recorded evidence share."""

from __future__ import annotations

import subprocess
import sys

import pytest

from families.minicpmv4_7.tests import test_e2e as e2e

CASE = next(iter(e2e.CASES))


class _Options:
    """pytest options as the E2E selection reads them."""

    def __init__(self, **options):
        self._values = {"--" + name.replace("_", "-"): value for name, value in options.items()}

    def getoption(self, name, default=None):
        return self._values.get(name, default)


def test_testcase_option_selects_exact_cases_only(monkeypatch):
    monkeypatch.delenv("TRTMC_E2E", raising=False)
    assert not e2e._selected(_Options(e2e_testcase=[e2e._FAMILY]), CASE)
    assert e2e._selected(_Options(e2e_testcase=[CASE]), CASE)
    assert e2e._selected(_Options(e2e_model=[e2e._FAMILY]), CASE)
    assert e2e._selected(_Options(e2e_model=[e2e.MANIFEST["name"]]), CASE)
    assert not e2e._selected(_Options(e2e_model=["another_family"]), CASE)
    assert not e2e._selected(_Options(e2e_model=[e2e._FAMILY], e2e_testcase=["another_case"]), CASE)
    assert not e2e._selected(_Options(), CASE)
    monkeypatch.setenv("TRTMC_E2E", "1")
    assert e2e._selected(_Options(), CASE)


def test_reference_interpreter_is_absolute_and_keeps_its_symlink(tmp_path, monkeypatch):
    binary = tmp_path / "trtmc"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "libtrtmc_model_minicpmv4_7.so").touch()
    (runtime / "minicpmv4_7_preprocess_dump").touch()
    interpreter = tmp_path / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    monkeypatch.setenv("TRTMC_BINARY", str(binary))
    monkeypatch.setenv("TRTMC_RUNTIME_ROOT", str(runtime))
    monkeypatch.delenv("TRTMC_NATIVE_BUILD_DIR", raising=False)
    monkeypatch.setenv("TRTMC_MINICPMV47_REFERENCE_PYTHON", "venv/bin/python")
    monkeypatch.chdir(tmp_path)
    reference = e2e._environment()[3]
    assert reference.is_absolute() and reference.is_symlink() and reference.samefile(interpreter)


def test_failed_command_raises_with_both_streams(tmp_path):
    log = tmp_path / "child.log"
    script = "import sys; print('partial output'); sys.stderr.write('child failure'); sys.exit(3)"
    with pytest.raises(subprocess.CalledProcessError) as failure:
        e2e._run([sys.executable, "-c", script], log)
    assert failure.value.returncode == 3
    assert failure.value.stdout.strip() == "partial output" and failure.value.stderr == "child failure"
    assert "child failure" in log.read_text()


def test_stop_rule_rejects_tokens_after_the_first_end_of_sequence():
    eos = {9}
    assert e2e._stops_correctly([1, 2, 9], eos, 8)
    assert e2e._stops_correctly([1, 2, 3, 4], eos, 4)
    assert not e2e._stops_correctly([9, 1, 2, 3], eos, 4)
    assert not e2e._stops_correctly([1, 2, 3], eos, 4)
    assert not e2e._stops_correctly([], eos, 4)


def test_gap_evidence_follows_the_per_token_rule():
    nan, inf = float("nan"), float("inf")
    assert e2e._gap_failures([0.1, nan, 0.9, inf, 0.85], 0.85) == [1, 2, 3]
    native = {"input_ids": [1, 2], "crops": [{"grid": [4, 4], "position_ids": [0]}],
              "positions": [[0, 0, 0], [1, 1, 1]], "decode_position": 2}
    reference = {"input_ids": [1, 2], "crop_grids": [[4, 4]], "crop_position_ids": [[0]],
                 "positions": [[0, 0, 0], [1, 1, 1]], "decode_position": 2}
    passing = e2e._comparison_checks(native, {**reference, "teacher_forced": {"candidate_gap": [0.1, 0.0]}},
                                     0, 0.85)
    failing = e2e._comparison_checks(native, {**reference, "teacher_forced": {"candidate_gap": [0.1, nan]}},
                                     0, 0.85)
    assert all(check["passed"] for check in passing)
    assert not all(check["passed"] for check in failing)
