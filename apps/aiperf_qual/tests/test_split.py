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
    monkeypatch.setattr(split, "code_digests",
                        lambda environment, model: {"serving": "s", "family": "", "runtime": "built"})
    monkeypatch.setattr(split, "dependencies_digest", lambda python: f"deps of {python}")
    monkeypatch.setattr(split, "pinned_versions", lambda python: {"tensorrt": "11.4.0.106", "python": python})
    inputs = split.campaign_inputs(Environment({"serve_python": __file__}))
    assert inputs["versions"] == {"harness": {"tensorrt": "11.4.0.106", "python": split.sys.executable},
                                  "serving": {"tensorrt": "11.4.0.106", "python": __file__}}
    assert inputs["code"] == {"serving": "s"}
    assert inputs["informational"] == {"runtime": "built", "dependencies": {
        "harness": f"deps of {split.sys.executable}", "serving": f"deps of {__file__}"}}


def test_pinned_versions_reads_tensorrt_torch_and_transformers(monkeypatch):
    frozen = "numpy==2.5.4\ntensorrt==11.4.0.106\ntensorrt_cu13==11.4.0.106\ntorch==2.12.0+cu130\n" \
             "Transformers==5.2.0\naiperf @ file:///tmp/x\n"
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: type("Done", (), {"stdout": frozen})())
    assert split.pinned_versions("python3") == {"tensorrt": "11.4.0.106", "torch": "2.12.0+cu130",
                                                 "transformers": "5.2.0"}


def test_hosts_whose_only_differences_are_informational_still_merge(tmp_path):
    assignment = split.assign(MODELS, LEDGER, ["h1", "h2"])
    required = {"harness": "x", "code": {"core": "c"}, "versions": {"harness": {"tensorrt": "11.4.0.106"}}}
    one = root(tmp_path, "one", assignment, "h1", {"a": "formal", "e": "formal"},
               inputs={**required, "informational": {"runtime": "r1", "dependencies": {"harness": "d1"}}})
    two = root(tmp_path, "two", assignment, "h2", {"b": "formal", "c": "formal", "d": "formal"},
               inputs={**required, "informational": {"runtime": "r2", "dependencies": {"harness": "d1"}}})
    assert split.merge_check(assignment, [one, two]) == []
    assert split.informational_differences([one, two]) == [
        "the hosts' runtime differ (informational: TensorRT, torch, transformers and the sources match)"]

    newer = {**required, "versions": {"harness": {"tensorrt": "11.5.0.1"}}, "informational": {}}
    three = root(tmp_path, "three", assignment, "h2", {"b": "formal", "c": "formal", "d": "formal"}, inputs=newer)
    assert any("different campaign inputs" in problem for problem in split.merge_check(assignment, [one, three]))


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
    assert ("the roots ran with different campaign inputs (harness, code, or the TensorRT, torch or transformers "
            "version)") in problems
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


def test_scheduling_estimates_do_not_shorten_benchmark_deadlines(tmp_path):
    """Historical scheduling estimates may omit inference; only explicit deadlines override the default."""
    from trtmc_aiperf_qual import absolute

    (tmp_path / "ledger.json").write_text(json.dumps({"big": 1000, "small": 60, "wan22-ti2v-5b": 289.6501877307892}))
    environment = Environment({"run_ledger": str(tmp_path / "ledger.json"), "deadlines": {"pinned": 42}})
    timeout = lambda name: absolute.run_timeout(environment, {"catalog_profile": name})  # noqa: E731
    assert timeout("big") == timeout("small") == absolute.RUN_TIMEOUT_S
    assert timeout("wan22-ti2v-5b") > 20 * 66
    assert timeout("pinned") == 42
    assert timeout("unlisted") == absolute.RUN_TIMEOUT_S
    assert absolute.run_timeout(Environment({}), {"catalog_profile": "big"}) == absolute.RUN_TIMEOUT_S


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


def test_a_servers_memory_is_recorded_per_numa_node(tmp_path):
    """Anonymous and file mappings apart: a private file mapping with a written (copy-on-write) page stays a file
    mapping, and a buffer allocated afterwards adds to the anonymous mappings only."""
    import sys
    import time

    data = tmp_path / "data"
    data.write_bytes(b"\0" * (64 << 20))
    step = tmp_path / "step"
    script = ("import mmap, os, time\n"
              f"f = open({str(data)!r}, 'r+b'); m = mmap.mmap(f.fileno(), 0, flags=mmap.MAP_PRIVATE)\n"
              "sum(m[i] for i in range(0, len(m), 4096)); m[0] = 1\n"
              f"open({str(step)!r} + '1', 'w').close()\n"
              f"while not os.path.exists({str(step)!r} + 'go'): time.sleep(0.05)\n"
              "x = bytearray(64 << 20)\n"
              f"open({str(step)!r} + '2', 'w').close(); time.sleep(30)\n")
    child = subprocess.Popen([sys.executable, "-c", script], start_new_session=True)

    def reached(name):
        for _ in range(200):
            if (tmp_path / f"step{name}").exists():
                return services.memory_nodes(child.pid)
            time.sleep(0.05)
        raise AssertionError(f"the child did not reach step {name}")

    total = lambda held, kind: sum(held[kind].values())  # noqa: E731
    try:
        mapped = reached("1")
        (tmp_path / "stepgo").touch()
        allocated = reached("2")
        assert total(mapped, "file_mappings_gib") >= 0.06 and mapped["node_free_gib"]  # the 64 MiB file mapping
        grown = total(allocated, "anonymous_mappings_gib") - total(mapped, "anonymous_mappings_gib")
        assert 0.05 <= grown <= 0.09  # the 64 MiB buffer
        assert abs(total(allocated, "file_mappings_gib") - total(mapped, "file_mappings_gib")) < 0.01
        assert services.memory_nodes(child.pid + 10**7)["anonymous_mappings_gib"] == {}  # no such group
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


def test_a_machines_environment_names_no_run_ledger_and_run_all_takes_one(tmp_path, monkeypatch, capsys):
    from trtmc_aiperf_qual import cli

    (tmp_path / "env.yaml").write_text(f"repo: {tmp_path}\nledger: ledger.json\n")
    for command in (["plan", "--profile", "x"], ["rejudge", str(tmp_path)], ["recheck", str(tmp_path)]):
        assert cli.main([*command, "--environment", str(tmp_path / "env.yaml")]) == 2, command
        assert "pass a run's ledger with run-all --ledger" in capsys.readouterr().err
    seen = {}

    def run_all(environment, models, out_root, **kwargs):
        seen["ledger"] = environment.values.get("run_ledger")
        return []

    (tmp_path / "env.yaml").write_text(f"repo: {tmp_path}\n")
    monkeypatch.setattr("trtmc_aiperf_qual.models.resolve_model", lambda name, environment: {"model": name})
    monkeypatch.setattr("trtmc_aiperf_qual.campaign.run_all", run_all)
    monkeypatch.setattr("trtmc_aiperf_qual.campaign.write_plan", lambda *args, **kwargs: None)
    cli.main(["run-all", "--environment", str(tmp_path / "env.yaml"), "--out-root", str(tmp_path / "out"),
              "--profile", "x", "--ledger", str(tmp_path / "ledger.json")])
    assert seen["ledger"] == str((tmp_path / "ledger.json").resolve())


def fragment_root(tmp_path, name, fragment, given, results, *, campaign="c1", host="h1", inputs=None,
                  config_errors=()):
    """One fragment's root: its plan (the profiles it was given) and results of each kind."""
    path = tmp_path / name
    path.mkdir()
    (path / "plan.json").write_text(json.dumps({
        "selected": list(given), "config_errors": [{"profile": item, "reason": "broken"} for item in config_errors],
        "campaign": campaign, "fragment": fragment, "host": host, "inputs": inputs or {"harness": "x"}}))
    for profile, kind in results.items():
        (path / profile).mkdir()
        if kind == "build-failed":
            (path / profile / "build.json").write_text(json.dumps({"status": "failed"}))
        elif kind == "error":
            (path / profile / "error.json").write_text(json.dumps({"category": "error"}))
        else:
            (path / profile / "report.json").write_text(json.dumps({"mode": kind}))
    return path


def test_a_fragment_resumes_only_as_the_same_campaign_fragment_host_and_inputs(tmp_path, monkeypatch):
    plan = {"campaign": "c1", "fragment": "g-a-1", "host": "h1", "inputs": {"harness": "x"}}
    old = fragment_root(tmp_path, "old", "g-a-1", ["a"], {"a": "formal"})
    split.check_resume(old, plan)
    for change in ({"fragment": "g-a-2"}, {"campaign": "c2"}, {"host": "h2"}, {"inputs": {"harness": "y"}}):
        with pytest.raises(ConfigError, match="use a fresh --out-root"):
            split.check_resume(old, {**plan, **change})
    held = root(tmp_path, "held", split.assign(MODELS, LEDGER, ["h1", "h2"]), "h1", {"a": "formal"})
    with pytest.raises(ConfigError):  # an assignment's root is no fragment's
        split.check_resume(held, plan)

    monkeypatch.setattr(split, "campaign_inputs", lambda environment: {"harness": "x"})
    assert split.fragment_plan("c1", "g-a-1", "h1", Environment({})) == plan
    for campaign, fragment, host in (("c1", "../g", "h1"), ("", "g", "h1"), ("c1", "g", "h 1")):
        with pytest.raises(ConfigError, match="letters, digits"):
            split.fragment_plan(campaign, fragment, host, Environment({}))


def test_fragments_merge_when_every_profile_has_one_accepted_result(tmp_path):
    manifest = {"campaign": "c1", "expected": ["a", "b", "c", "d"], "accepted": ["g-a-1", "g-bc-2", "g-d-1"]}
    first = fragment_root(tmp_path, "first", "g-a-1", ["a"], {"a": "formal", "a.1791000000": "smoke"})
    lost = fragment_root(tmp_path, "lost", "g-bc-1", ["b", "c"], {"b": "formal"}, host="h2")  # superseded
    retry = fragment_root(tmp_path, "retry", "g-bc-2", ["c", "b"], {"b": "formal", "c": "build-failed"})
    broken = fragment_root(tmp_path, "broken", "g-d-1", [], {}, host="h2", config_errors=["d"])
    roots = [first, lost, retry, broken]

    assert split.fragment_merge_check(manifest, roots) == []
    assert split.accepted_roots(manifest, roots) == [first, retry, broken]


def test_fragments_do_not_merge_when_a_result_is_missing_doubled_foreign_or_unlike(tmp_path):
    manifest = {"campaign": "c1", "expected": ["a", "b", "c"], "accepted": ["g-a-1", "g-b-1", "g-c-1"]}
    one = fragment_root(tmp_path, "one", "g-a-1", ["a"], {"a": "formal", "b": "formal"})
    two = fragment_root(tmp_path, "two", "g-b-1", ["b", "z"], {"b": "formal", "z": "formal"}, inputs={"harness": "y"})
    smoke = fragment_root(tmp_path, "smoke", "g-c-1", ["c"], {"c": "smoke"})
    elsewhere = fragment_root(tmp_path, "elsewhere", "g-c-1", ["c"], {"c": "formal"}, campaign="c0")

    problems = split.fragment_merge_check(manifest, [one, two, smoke, elsewhere])

    assert f"{one}: b is not one of fragment g-a-1's profiles" in problems
    assert f"{two}: z is not in the campaign" in problems
    assert f"{smoke}: c is a smoke result" in problems and "c: no formal result" in problems
    assert any(problem.startswith(f"{elsewhere}: not run in campaign c1") for problem in problems)
    assert any("different campaign inputs" in problem for problem in problems)
    assert split.fragment_merge_check(manifest, [smoke], "smoke")[:2] == ["fragment g-a-1: 0 result roots",
                                                                          "fragment g-b-1: 0 result roots"]
    twice = {**manifest, "accepted": ["g-a-1", "g-a-2", "g-c-1"]}
    again = fragment_root(tmp_path, "again", "g-a-2", ["a"], {"a": "formal"})
    copy = fragment_root(tmp_path, "copy", "g-c-1", ["c"], {"c": "formal"})
    problems = split.fragment_merge_check(twice, [one, again, smoke, copy])
    assert any(problem.startswith("a: results in") for problem in problems)
    assert "fragment g-c-1: 2 result roots" in problems
    assert split.fragment_merge_check({"campaign": "../c", "expected": ["a", "a"], "accepted": "g"}, []) == [
        "the manifest's campaign is not an identifier: '../c'", "the manifest's expected lists a name twice",
        "the manifest's accepted is not a list of names"]


def test_run_all_runs_a_fragment_in_its_given_order_and_records_it(tmp_path, monkeypatch, capsys):
    from trtmc_aiperf_qual import cli

    (tmp_path / "env.yaml").write_text(f"repo: {tmp_path}\n")
    seen = {}

    def run_all(environment, models, out_root, **kwargs):
        seen.update(models=[item["model"] for item in models], keep_order=kwargs["keep_order"])
        return []

    monkeypatch.setattr("trtmc_aiperf_qual.models.resolve_model", lambda name, environment: {"model": name})
    monkeypatch.setattr("trtmc_aiperf_qual.campaign.run_all", run_all)
    monkeypatch.setattr(split, "campaign_inputs", lambda environment: {"harness": "x"})
    base = ["run-all", "--environment", str(tmp_path / "env.yaml"), "--out-root", str(tmp_path / "out")]
    fragment = ["--campaign", "c1", "--fragment", "g-bc-2", "--host", "h1", "--profile", "c", "--profile", "b"]

    assert cli.main([*base, *fragment]) == 0
    assert seen == {"models": ["c", "b"], "keep_order": True}
    plan = json.loads((tmp_path / "out" / "plan.json").read_text())
    assert {key: plan[key] for key in ("campaign", "fragment", "host", "selected", "inputs")} == {
        "campaign": "c1", "fragment": "g-bc-2", "host": "h1", "selected": ["c", "b"], "inputs": {"harness": "x"}}
    (tmp_path / "out" / "c").mkdir()
    (tmp_path / "out" / "c" / "report.json").write_text(json.dumps({"mode": "formal"}))
    for wrong in (fragment[:4] + fragment[6:], [*fragment, "--shard", "0/2"],
                  ["--campaign", "c1", "--fragment", "g-bc-3", "--host", "h1", "--profile", "b"],
                  ["--campaign", "c1", "--fragment", "../x", "--host", "h1", "--profile", "b"]):
        seen.clear()
        assert cli.main([*base, *wrong]) == 2, wrong
        assert seen == {}
        capsys.readouterr()


def test_summary_merges_only_the_manifests_accepted_fragments(tmp_path, monkeypatch, capsys):
    from trtmc_aiperf_qual import cli

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"campaign": "c1", "expected": ["a", "b"], "accepted": ["g-a-1", "g-b-2"]}))
    first = fragment_root(tmp_path, "a1", "g-a-1", ["a"], {"a": "formal"})
    late = fragment_root(tmp_path, "b1", "g-b-1", ["b"], {"b": "formal"}, host="h2")
    retry = fragment_root(tmp_path, "b2", "g-b-2", ["b"], {"b": "formal"})
    merged = {}

    def summary(roots, baseline):
        merged["roots"] = [path.name for path in roots]
        return "# merged\n", {}

    monkeypatch.setattr("trtmc_aiperf_qual.campaign.summary", summary)
    assert cli.main(["merge-check", "--manifest", str(manifest), str(first), str(late), str(retry)]) == 0
    assert cli.main(["summary", "--manifest", str(manifest), str(first), str(late), str(retry)]) == 0
    assert merged["roots"] == ["a1", "b2"]
    capsys.readouterr()
    assert cli.main(["summary", "--manifest", str(manifest), str(first), str(late)]) == 1
    assert "fragment g-b-2: 0 result roots" in capsys.readouterr().err
