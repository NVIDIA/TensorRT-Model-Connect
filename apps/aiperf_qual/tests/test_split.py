# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess

import pytest

from trtmc_aiperf_qual import bundles, services, split
from trtmc_aiperf_qual.config import ConfigError, Environment


def model(name, checkpoint=None):
    return {"model": name, "candidate": {"checkpoint": checkpoint or name}}


MODELS = [model("a"), model("b", "shared"), model("c", "shared"), model("d"), model("e")]
LEDGER = {"a": 100, "b": 30, "c": 40, "d": 60, "e": 60}


def test_the_assignment_is_longest_first_keeps_checkpoints_together_and_is_deterministic():
    assignment = split.assign(MODELS, LEDGER, ["h1", "h2"])
    # groups: a 100, shared (b, c) 70, d 60, e 60 -> h1 a; h2 b, c; h2 (70) vs h1 (100): d to h2; e to h1 (100 < 130)
    assert assignment["hosts"] == {"h1": ["a", "e"], "h2": ["b", "c", "d"]}
    assert assignment["predicted_s"] == {"h1": 160, "h2": 130}
    assert split.assign(list(reversed(MODELS)), LEDGER, ["h1", "h2"]) == assignment
    assert split.digest(assignment) == split.digest(json.loads(json.dumps(assignment)))
    with pytest.raises(ConfigError, match="no ledger time for e"):
        split.assign(MODELS, {key: value for key, value in LEDGER.items() if key != "e"}, ["h1", "h2"])


def test_a_host_runs_exactly_its_share_in_the_assigned_order():
    assignment = split.assign(MODELS, LEDGER, ["h1", "h2"])
    assert [m["model"] for m in split.host_models(assignment, "h2", MODELS)] == ["b", "c", "d"]
    with pytest.raises(ConfigError, match="unassigned"):
        split.host_models(assignment, "h1", [*MODELS, model("f")])
    with pytest.raises(ConfigError, match="not in the assignment"):
        split.host_models(assignment, "h3", MODELS)
    broken = {**assignment, "hosts": {"h1": ["a", "e", "b"], "h2": ["b", "c", "d"]}}
    with pytest.raises(ConfigError, match="duplicated \\['b'\\]"):
        split.host_models(broken, "h1", MODELS)


def root(tmp_path, name, assignment, host, results, inputs=None):
    path = tmp_path / name
    path.mkdir()
    (path / "plan.json").write_text(json.dumps({"host": host, "assignment": split.digest(assignment),
                                                "inputs": inputs or {"harness": "x"}}))
    for profile, kind in results.items():
        (path / profile).mkdir()
        if kind == "build-failed":
            (path / profile / "build.json").write_text(json.dumps({"status": "failed"}))
        else:
            (path / profile / "report.json").write_text(json.dumps({"mode": kind}))
    return path


def test_the_roots_merge_only_when_disjoint_complete_formal_and_alike(tmp_path):
    assignment = split.assign(MODELS, LEDGER, ["h1", "h2"])
    one = root(tmp_path, "one", assignment, "h1", {"a": "formal", "e": "formal", "e.1791000000": "smoke"})
    two = root(tmp_path, "two", assignment, "h2", {"b": "formal", "c": "build-failed", "d": "formal"})
    assert split.merge_check(assignment, [one, two]) == []

    other = root(tmp_path, "other", assignment, "h2", {"b": "formal", "c": "formal", "d": "smoke", "a": "formal"},
                 inputs={"harness": "y"})
    problems = split.merge_check(assignment, [one, other])
    assert "the roots ran with different campaign inputs (harness, code, runtime, or dependencies)" in problems
    assert any("a is assigned to another host" in problem for problem in problems)
    assert any("d is a smoke result" in problem for problem in problems)
    assert "d: no formal result" in problems

    twin = root(tmp_path, "twin", assignment, "h1", {"a": "formal"})
    problems = split.merge_check(assignment, [one, twin, two])
    assert "host h1: 2 result roots" in problems and any("a: results in" in problem for problem in problems)
    stale = root(tmp_path, "stale", {**assignment, "rule": "older"}, "h2", {})
    assert any("not run under this assignment" in problem for problem in split.merge_check(assignment, [one, stale]))


def test_the_bundle_that_was_qualified_is_identified_by_its_bytes(tmp_path):
    bundle = tmp_path / "m.bundle"
    bundle.write_bytes(b"engine")
    (tmp_path / "m.bundle.benchmark.json").write_text("{}")
    identity = bundles.identity(bundle)
    assert identity == {"bundle_bytes": 6, "bundle_sha256": hashlib.sha256(b"engine").hexdigest(),
                        "receipt_sha256": hashlib.sha256(b"{}").hexdigest()}


def test_the_report_records_the_gpu_and_runtime_it_ran_on(tmp_path, monkeypatch):
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "libnvinfer.so.10.14.1").write_bytes(b"")
    line = "GPU-aaaa, NVIDIA GB300, 595.58.03, 2032, 3996, 2032, 1400.00, Enabled, Default\n"
    monkeypatch.setattr(services.subprocess, "run",
                        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout=line))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    identity = services.gpu_identity(Environment({"runtime_root": str(tmp_path)}))
    assert identity["gpu"]["uuid"] == "GPU-aaaa" and identity["gpu"]["power.limit"] == "1400.00"
    assert identity["tensorrt"] == ["libnvinfer.so.10.14.1"] and identity["hostname"]
