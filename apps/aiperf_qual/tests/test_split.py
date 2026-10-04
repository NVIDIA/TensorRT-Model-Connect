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


def test_a_root_resumes_only_under_its_own_assignment_host_and_inputs(tmp_path):
    assignment = split.assign(MODELS, LEDGER, ["h1", "h2"])
    plan = {"host": "h1", "assignment": split.digest(assignment), "inputs": {"harness": "x"}}
    split.check_resume(tmp_path / "fresh", plan)  # nothing there yet
    old = root(tmp_path, "old", assignment, "h1", {"a": "formal"})
    split.check_resume(old, plan)  # the same plan continues
    with pytest.raises(ConfigError, match="use a fresh --out-root"):
        split.check_resume(old, {**plan, "assignment": "another"})
    with pytest.raises(ConfigError):
        split.check_resume(old, {**plan, "inputs": {"harness": "y"}})
    empty = root(tmp_path, "empty", assignment, "h2", {})
    split.check_resume(empty, plan)  # no results to relabel


def test_a_rerun_sets_every_previous_result_aside_before_its_plan(tmp_path):
    assignment = split.assign(MODELS, LEDGER, ["h1", "h2"])
    old = root(tmp_path, "old", assignment, "h1", {"a": "formal", "e": "error", "b": "formal"})
    (old / "partial").mkdir()  # no result: left as it is
    assert split.set_aside_results(old) == ["a", "b", "e"]
    remaining = sorted(path.name for path in old.iterdir() if path.is_dir())
    assert "partial" in remaining and not {"a", "b", "e"} & set(remaining)
    problems = split.merge_check(assignment, [old])
    assert all(f"{name}: no formal result" in problems for name in "abcde")  # the old ones count no more


def test_both_interpreters_count_among_the_campaign_inputs(monkeypatch):
    monkeypatch.setattr(split, "harness_digest", lambda: "h")
    monkeypatch.setattr(split, "code_digests", lambda environment, model: {"serving": "s", "family": ""})
    monkeypatch.setattr(split, "dependencies_digest", lambda python: f"deps of {python}")
    inputs = split.campaign_inputs(Environment({"serve_python": __file__}))
    assert inputs["dependencies"] == {"harness": f"deps of {split.sys.executable}", "serving": f"deps of {__file__}"}
    assert inputs["code"] == {"serving": "s"}


def root(tmp_path, name, assignment, host, results, inputs=None):
    path = tmp_path / name
    path.mkdir()
    (path / "plan.json").write_text(json.dumps({"host": host, "assignment": split.digest(assignment),
                                                "inputs": inputs or {"harness": "x"}}))
    for profile, kind in results.items():
        (path / profile).mkdir()
        if kind == "build-failed":
            (path / profile / "build.json").write_text(json.dumps({"status": "failed"}))
        elif kind == "error":
            (path / profile / "error.json").write_text(json.dumps({"category": "error"}))
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
    wrong = root(tmp_path, "wrong", assignment, "h2", {"b": "formal", "c": "error", "d": "formal", "a": "error"})
    problems = split.merge_check(assignment, [one, wrong])
    assert problems == [f"{wrong}: a is assigned to another host"]  # an error result is a result too
    smoke_one = root(tmp_path, "s1", assignment, "h1", {"a": "smoke", "e": "smoke"})
    smoke_two = root(tmp_path, "s2", assignment, "h2", {"b": "smoke", "c": "build-failed", "d": "smoke"})
    assert split.merge_check(assignment, [smoke_one, smoke_two], "smoke") == []  # smoke roots, checked as such
    assert "a: no formal result" in split.merge_check(assignment, [smoke_one, smoke_two])
    stale = root(tmp_path, "stale", {**assignment, "rule": "older"}, "h2", {})
    assert any("not run under this assignment" in problem for problem in split.merge_check(assignment, [one, stale]))


def test_the_bundle_that_was_qualified_is_identified_by_its_bytes(tmp_path):
    bundle = tmp_path / "m.bundle"
    bundle.write_bytes(b"engine")
    (tmp_path / "m.bundle.benchmark.json").write_text("{}")
    identity = bundles.identity(bundle)
    assert identity == {"bundle_bytes": 6, "bundle_sha256": hashlib.sha256(b"engine").hexdigest(),
                        "receipt_sha256": hashlib.sha256(b"{}").hexdigest()}


def test_the_libraries_a_server_group_mapped_are_recorded():
    import sys

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        for _ in range(50):
            libraries = services.loaded_libraries(child.pid, names=("libc", "libpython"))
            if libraries:
                break
            __import__("time").sleep(0.1)
        assert libraries and all("/" in path for path in libraries)
        assert services.loaded_libraries(child.pid + 10**7) == []  # no such group
    finally:
        child.kill()
        child.wait()


def test_the_report_records_the_gpu_it_ran_on(tmp_path, monkeypatch):
    line = "GPU-aaaa, NVIDIA GB300, 595.58.03, 2070, 3996, 1400.00, Enabled, Default\n"
    monkeypatch.setattr(services.subprocess, "run",
                        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout=line))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    identity = services.gpu_identity(Environment({"runtime_root": str(tmp_path)}))
    assert identity["gpu"]["uuid"] == "GPU-aaaa" and identity["gpu"]["power.limit"] == "1400.00"
    assert identity["hostname"]


def test_two_hosts_that_built_the_same_runtime_agree_on_its_digest(tmp_path):
    """The runtime digest covers what a run loads (the worker and shared libraries, by content), not build logs or
    timestamps, so separately built hosts compare equal; a changed library changes it."""
    import os

    from trtmc_aiperf_qual import campaign

    def tree(name, log, library=b"engine code"):
        root = tmp_path / name
        (root / "lib").mkdir(parents=True)
        (root / "trtmc_benchmark_worker").write_bytes(b"worker")
        (root / "lib" / "libtrtmc.so.1").write_bytes(library)
        (root / ".ninja_log").write_text(log)
        os.utime(root / "lib" / "libtrtmc.so.1", (1, len(name)))
        return root

    one, two = tree("h1", "built in 108619 ms"), tree("h22", "built in 108529 ms")
    digest = lambda root: campaign.runtime_digest.__wrapped__(str(root / "trtmc_benchmark_worker"), str(root))  # noqa: E731
    assert digest(one) == digest(two)
    assert digest(tree("h333", "x", library=b"other engine code")) != digest(one)


def test_a_local_install_path_does_not_split_the_dependency_digest(monkeypatch):
    import subprocess

    from trtmc_aiperf_qual import campaign

    freezes = iter(["aiperf==0.13.0\ntrtmc-aiperf-plugins @ file:///tmp/trtmc-plugins.jJgIu4\n",
                    "aiperf==0.13.0\ntrtmc-aiperf-plugins @ file:///tmp/trtmc-plugins.wEcs4l\n",
                    "aiperf==0.14.0\ntrtmc-aiperf-plugins @ file:///tmp/trtmc-plugins.wEcs4l\n"])
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=next(freezes)))
    first, second, third = (campaign.dependencies_digest.__wrapped__("python") for _ in range(3))
    assert first == second != third
