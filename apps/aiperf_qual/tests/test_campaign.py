# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

import pytest

from trtmc_aiperf_qual import bundles, campaign, retention, selection
from trtmc_aiperf_qual.config import ConfigError, Environment

REPOSITORY = Path(__file__).resolve().parents[3]


def _model(profile, checkpoint, task="text_generation"):
    return {"model": profile, "catalog_profile": profile, "task": task, "reference": {},
            "candidate": {"bundle": f"{profile}/{profile}.bundle", "checkpoint": checkpoint}}


def test_build_uses_the_catalog_manifest_or_a_descriptor_with_build_overrides(tmp_path):
    from trtmc_aiperf_qual.models import resolve_model

    environment = Environment({"repo": str(REPOSITORY), "bundle_root": str(tmp_path / "engines"),
                               "runtime_root": "/rt", "worker": "/rt/worker", "serve_python": "/py"})
    tiny = resolve_model("tinyllama-1.1b", environment)  # no build override: the catalog manifest as is
    command = bundles.build_command(environment, tiny, tmp_path / "tiny")
    assert command[:4] == ["/py", "-m", "trtmc_benchmark", "run"] and "--prepare-only" in command
    assert command[command.index("--model") + 1] == "tinyllama-1.1b" and "--manifest-root" in command
    assert command[command.index("--bundle-cache") + 1] == str(tmp_path / "engines")

    detr = resolve_model("detr-resnet-50", environment)  # a config/models build exception: its own bundle name
    assert detr["candidate"]["bundle"] == "detr-resnet-50-coco/detr-resnet-50.bundle"
    command = bundles.build_command(environment, detr, tmp_path / "detr")
    assert "--manifest-root" not in command
    descriptor = json.loads(Path(command[command.index("--model") + 1]).read_text())
    assert (descriptor["name"], descriptor["bundle"], descriptor["image_height"], descriptor["image_width"]) == (
        "detr-resnet-50-coco", "detr-resnet-50.bundle", 1333, 1333)
    image = Path(descriptor["testcases"][0]["test_image"])
    assert image.is_absolute() and image.is_file()  # catalog assets stay reachable from the descriptor


def test_ensure_bundle_lets_trtmc_bench_decide_reuse_by_its_build_receipt(tmp_path, monkeypatch):
    environment = Environment({"repo": str(tmp_path), "bundle_root": str(tmp_path / "engines"),
                               "gpu_lock": str(tmp_path / "gpu.lock")})
    bundle = tmp_path / "engines/m/m.bundle"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("x")
    receipt = bundle.with_suffix(".bundle.benchmark.json")
    receipt.write_text('{"identity": "a"}')
    keep = [sys.executable, "-c", "pass"]  # trtmc-bench found a matching receipt
    monkeypatch.setattr(bundles, "build_command", lambda *args: keep)
    assert bundles.ensure_bundle(environment, _model("m", "org/m"), tmp_path / "out")["status"] == "reused"
    rebuild = [sys.executable, "-c", f"import pathlib; pathlib.Path({str(receipt)!r}).write_text('{{\"identity\": \"b\"}}')"]
    monkeypatch.setattr(bundles, "build_command", lambda *args: rebuild)  # the identity changed: rebuilt
    assert bundles.ensure_bundle(environment, _model("m", "org/m"), tmp_path / "out")["status"] == "built"


def test_ensure_bundle_records_a_failed_build_under_the_gpu_lock(tmp_path, monkeypatch):
    environment = Environment({"repo": str(tmp_path), "bundle_root": str(tmp_path / "engines"),
                               "gpu_lock": str(tmp_path / "gpu.lock")})
    failing = [sys.executable, "-c", "import sys; print('step'); print('ValueError: boom'); sys.exit(3)"]
    monkeypatch.setattr(bundles, "build_command", lambda *args: failing)
    result = bundles.ensure_bundle(environment, _model("m", "org/m"), tmp_path / "out")
    assert (result["status"], result["exit"]) == ("failed", 3) and "boom" in result["reason"]
    assert Path(result["log"]).is_file() and (tmp_path / "gpu.lock").is_file()


@pytest.mark.parametrize("policy, category, deleted", [
    ("retain", "pass", False),
    ("delete_on_pass", "pass", True),
    ("delete_on_pass", "acc-issue", False),
    ("delete_unless_error", "acc-issue", True),
    ("delete_unless_error", "error", False),  # a harness failure is rerun: keep its bundle
    ("delete_unless_error", "build-failed", False),
])
def test_bundle_retention_policy(policy, category, deleted):
    assert retention.should_delete_bundle(policy, category) is deleted


def test_only_bundles_the_run_built_are_deleted_by_delete_built_unless_error():
    assert retention.should_delete_bundle("delete_built_unless_error", "acc-issue", built=True)
    assert not retention.should_delete_bundle("delete_built_unless_error", "pass", built=False)  # it existed before
    assert not retention.should_delete_bundle("delete_built_unless_error", "error", built=True)


def test_retention_policies_are_validated():
    assert retention.policies(Environment({})) == ("retain", "retain")
    with pytest.raises(ConfigError):
        retention.policies(Environment({"retention": {"bundle": "delete_sometimes"}}))
    with pytest.raises(ConfigError, match="hf_hub_cache"):
        retention.policies(Environment({"retention": {"hf_cache": "delete_unused"}}))


def test_delete_bundle_removes_only_the_model_directory_under_the_bundle_root(tmp_path):
    root = tmp_path / "engines"
    (root / "m").mkdir(parents=True)
    (root / "m/m.bundle").write_bytes(b"12345")
    (root / "other").mkdir()
    (tmp_path / "outside").mkdir()
    environment = Environment({"bundle_root": str(root)})
    assert retention.delete_bundle(environment, _model("m", "org/m")) == {
        "status": "deleted", "path": str(root / "m"), "bytes": 5}
    assert not (root / "m").exists() and (root / "other").is_dir()
    assert retention.delete_bundle(environment, _model("m", "org/m"))["status"] == "absent"
    for bundle in ("../outside/x.bundle", "x.bundle"):  # escapes the root / would remove the root itself
        with pytest.raises(ConfigError):
            retention.delete_bundle(environment, {"candidate": {"bundle": bundle}})
    assert (tmp_path / "outside").is_dir() and root.is_dir()


def test_delete_checkpoint_removes_one_repository_from_the_hub_cache(tmp_path):
    hub = tmp_path / "hub"
    repo = hub / "models--org--a"
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs/x").write_bytes(b"12345")
    (repo / "snapshots/rev").mkdir(parents=True)
    (repo / "snapshots/rev/x").symlink_to(repo / "blobs/x")
    (hub / "models--org--b").mkdir()
    (hub / ".locks/models--org--a").mkdir(parents=True)
    assert retention.delete_checkpoint(hub, "org/a") == {"status": "deleted", "repo": "org/a", "bytes": 5}
    assert not repo.exists() and not (hub / ".locks/models--org--a").exists() and (hub / "models--org--b").is_dir()
    assert retention.delete_checkpoint(hub, "org/a")["status"] == "absent"
    for bad in ("../org/a", "org/..", "/abs/path", "a/b/c", ""):
        with pytest.raises(ConfigError):
            retention.delete_checkpoint(hub, bad)


def test_profiles_sharing_a_checkpoint_run_together_and_shards_keep_them_together():
    models = [_model("c", "org/c"), _model("a-fp8", "org/a"), _model("b", "org/b"), _model("a-fp16", "org/a")]
    assert [m["model"] for m in campaign.order(models)] == ["a-fp16", "a-fp8", "b", "c"]
    assert [m["model"] for m in campaign.shard(models, 0, 2)] == ["a-fp16", "a-fp8", "c"]
    assert [m["model"] for m in campaign.shard(models, 1, 2)] == ["b"]


def _record_runs(monkeypatch, events, category="pass"):
    def run_one(environment, model, out):
        events.append(("run", model["model"]))
        return {"profile": model["model"], "category": category}

    monkeypatch.setattr(campaign, "run_one", run_one)
    monkeypatch.setattr(retention, "delete_checkpoint",
                        lambda hub, repo: events.append(("delete", repo)) or {"status": "deleted"})


def test_run_all_deletes_a_checkpoint_after_the_last_profile_using_it(tmp_path, monkeypatch):
    events = []
    _record_runs(monkeypatch, events, category="error")  # checkpoints are reference-counted whatever the verdict
    environment = Environment({"hf_hub_cache": str(tmp_path / "hub"), "retention": {"hf_cache": "delete_unused"}})
    models = [_model("a-fp16", "org/a"), _model("b", "org/b"), _model("a-fp8", "org/a")]
    records = campaign.run_all(environment, models, tmp_path / "out", prefetch_next=False)
    assert events == [("run", "a-fp16"), ("run", "a-fp8"), ("delete", "org/a"), ("run", "b"), ("delete", "org/b")]
    assert [record["profile"] for record in records] == ["a-fp16", "a-fp8", "b"]
    logged = [json.loads(line) for line in (tmp_path / "out/campaign.jsonl").read_text().splitlines()]
    assert logged[1]["checkpoints_deleted"] == [{"status": "deleted"}]


def test_run_all_keeps_checkpoints_by_default(tmp_path, monkeypatch):
    events = []
    _record_runs(monkeypatch, events)
    campaign.run_all(Environment({}), [_model("a", "org/a")], tmp_path / "out", prefetch_next=False)
    assert events == [("run", "a")]


def test_run_all_resumes_only_results_of_the_same_run_key(tmp_path, monkeypatch):
    events = []

    def run_one(environment, model, out):  # what campaign.run_one leaves behind: the run key and a report
        events.append(("run", model["model"]))
        out.mkdir(parents=True, exist_ok=True)
        (out / campaign.RUN_KEY).write_text(campaign.run_key(environment, model) + "\n")
        (out / "report.json").write_text(json.dumps({"verdict": {"category": "pass"}}))
        return {"profile": model["model"], "category": "pass"}

    monkeypatch.setattr(campaign, "run_one", run_one)
    done = tmp_path / "out/a"
    done.mkdir(parents=True)
    (done / "report.json").write_text(json.dumps({"verdict": {"category": "pass"}}))
    models = [_model("a", "org/a"), _model("b", "org/b")]
    campaign.run_all(Environment({}), models, tmp_path / "out", prefetch_next=False)
    assert events == [("run", "a"), ("run", "b")]  # no run key: an older harness's result is rerun
    events.clear()
    campaign.run_all(Environment({}), models, tmp_path / "out", prefetch_next=False)
    assert events == []  # same configuration, harness, and mode: resumed
    campaign.run_all(Environment({"smoke": True}), models[:1], tmp_path / "out", prefetch_next=False)
    assert events == [("run", "a")]  # a smoke result never stands in for a formal one, nor the reverse
    campaign.run_all(Environment({"smoke": True}), models[:1], tmp_path / "out", rerun=True, prefetch_next=False)
    assert events[-1] == ("run", "a") and len(events) == 2


def _stub_phases(monkeypatch, *, build="built", qualify=None):
    monkeypatch.setattr(campaign, "reference_python", lambda environment, model: "/py")
    monkeypatch.setattr(campaign, "prefetch", lambda environment, model: None)
    monkeypatch.setattr(campaign.bundles, "ensure_bundle",
                        lambda environment, model, out, python=None: {"status": build, "reason": "ValueError: boom"})
    monkeypatch.setattr(campaign, "qualify", qualify or (lambda model, environment, out: {
        "verdict": {"category": "acc-issue", "acc": "fail", "perf": "green"}}))
    deleted = []
    monkeypatch.setattr(retention, "delete_bundle",
                        lambda environment, model: deleted.append(model["model"]) or {"status": "deleted"})
    return deleted


def test_run_one_builds_qualifies_and_applies_the_bundle_policy(tmp_path, monkeypatch):
    deleted = _stub_phases(monkeypatch)
    environment = Environment({"retention": {"bundle": "delete_unless_error"}})
    record = campaign.run_one(environment, _model("m", "org/m"), tmp_path / "m")
    assert (record["category"], record["bundle_deleted"]) == ("acc-issue", {"status": "deleted"}) and deleted == ["m"]
    assert json.loads((tmp_path / "m/build.json").read_text())["status"] == "built"


@pytest.mark.parametrize("acc, complete, timed, delete", [
    ("pass", True, True, True),
    ("pass", False, True, False),
    ("pass", True, False, False),
    ("inconclusive", True, True, False),
    ("fail", True, True, False),
])
def test_delete_on_pass_uses_the_benchmark_verdict(tmp_path, monkeypatch, acc, complete, timed, delete):
    from trtmc_aiperf_qual import judge

    result = {"performance_source": "quality", "accuracy": [{"suite": "mmlu-0shot", "status": acc}],
              "performance": [{"kind": "natural_dataset", "request": "mmlu-0shot", "gate": False,
                               "complete": complete, "candidate": {"p50_ms": 10 if timed else None},
                               "reference": {"p50_ms": 12}}]}
    verdict = judge.verdict(result, expected_suites=["mmlu-0shot"], expected_modes=0)
    deleted = _stub_phases(monkeypatch, qualify=lambda *args: {"verdict": verdict})
    environment = Environment({"retention": {"bundle": "delete_on_pass"}})
    record = campaign.run_one(environment, _model("m", "org/m"), tmp_path / "m")
    assert deleted == (["m"] if delete else [])
    assert record["category"] == verdict["category"]  # retention never rewrites the reported verdict
    assert ("bundle_deleted" in record) is delete


def test_run_one_records_build_failures_and_keeps_errored_bundles(tmp_path, monkeypatch):
    deleted = _stub_phases(monkeypatch, build="failed", qualify=lambda *a: pytest.fail("must not qualify"))
    environment = Environment({"retention": {"bundle": "delete_unless_error"}})
    record = campaign.run_one(environment, _model("m", "org/m"), tmp_path / "m")
    assert record["category"] == "build-failed" and "boom" in record["reason"] and deleted == []

    def crash(model, environment, out):
        raise RuntimeError("reference environment failed")

    deleted = _stub_phases(monkeypatch, qualify=crash)
    record = campaign.run_one(environment, _model("m", "org/m"), tmp_path / "m2")
    assert record["category"] == "error" and "reference environment failed" in record["reason"] and deleted == []


def test_summary_merges_result_roots(tmp_path):
    first = tmp_path / "gb300-1/a"
    first.mkdir(parents=True)
    (first / "report.json").write_text(json.dumps({
        "task": "text_generation", "verdict": {"category": "pass"},
        "accuracy": [{"suite": "mmlu", "passed": 10, "samples": 10, "required_passes": 9}],
        "performance": [{"reference_mode": "eager", "light": "green", "speedup": 2.5,
                            "candidate": {"p50_ms": 4.0}, "reference": {"p50_ms": 10.0}}]}))
    failed = tmp_path / "gb300-2/b"
    failed.mkdir(parents=True)
    (failed / "build.json").write_text(json.dumps({"task": "classification", "status": "failed",
                                                   "reason": "error: checkpoint is gated"}))
    text, counts = campaign.summary([tmp_path / "gb300-1", tmp_path / "gb300-2"])
    assert counts == {"pass": 1, "build-failed": 1}
    assert "| Green | a | text_generation | gb300-1 | mmlu: 10/10 within tolerance | eager: TRTMC 4.0 ms · native 10.0 ms |" in text
    assert "| White | b | classification | gb300-2 |" in text and "checkpoint is gated" in text
    assert "1 pass (Green + conclusive Yellow)" in text and "| White | 1 |" in text


CATALOG = [selection.Profile("qwen3-0.6b-fp16", "text_generation", "Qwen/Qwen3-0.6B", None),
           selection.Profile("qwen36-27b", "text_generation", "Qwen/Qwen3.6-27B", None),
           selection.Profile("flux-2-dev", "image_generation", "black-forest-labs/FLUX.2-dev", None),
           selection.Profile("flux-2-dev-fp8", "image_generation", "black-forest-labs/FLUX.2-dev", None)]


def _names(profiles):
    return [profile.name for profile in profiles]


def test_machine_model_list_includes_all_by_default_and_excludes_with_reasons():
    selected, excluded = selection.select({}, CATALOG)
    assert _names(selected) == _names(CATALOG) and excluded == []
    config = {"exclude": [{"profile": "flux-2-dev*", "reason": "does not fit in 80 GB"}]}
    selected, excluded = selection.select(config, CATALOG)
    assert _names(selected) == ["qwen3-0.6b-fp16", "qwen36-27b"]
    assert excluded == [{"profile": "flux-2-dev", "task": "image_generation", "reason": "does not fit in 80 GB"},
                        {"profile": "flux-2-dev-fp8", "task": "image_generation", "reason": "does not fit in 80 GB"}]


def test_machine_model_list_include_accepts_names_and_patterns():
    selected, excluded = selection.select({"include": ["qwen*", "flux-2-dev"]}, CATALOG)
    assert _names(selected) == ["qwen3-0.6b-fp16", "qwen36-27b", "flux-2-dev"]
    assert excluded == [{"profile": "flux-2-dev-fp8", "task": "image_generation", "reason": "not in models.include"}]


def test_machine_model_list_excludes_checkpoints_above_the_size_limit():
    sizes = {"Qwen/Qwen3-0.6B": 1.2, "Qwen/Qwen3.6-27B": 54.0, "black-forest-labs/FLUX.2-dev": None}
    selected, excluded = selection.select({"max_checkpoint_gib": 30}, CATALOG,
                                          size_gib=lambda profile: sizes[profile.checkpoint])
    assert _names(selected) == ["qwen3-0.6b-fp16", "flux-2-dev", "flux-2-dev-fp8"]  # unknown sizes are kept
    assert excluded == [{"profile": "qwen36-27b", "task": "text_generation",
                         "reason": "checkpoint 54 GiB > models.max_checkpoint_gib 30"}]


@pytest.mark.parametrize("config, message", [
    ({"exclude": [{"profile": "flux-2-dev"}]}, "reason"),
    ({"include": ["qwen3-0.6b-fp61"]}, "matches no ready catalog profile"),
    ({"include": "some"}, "include"),
    ({"exclude": "flux-2-dev"}, "exclude"),
    ({"max_checkpoint_gib": "big"}, "max_checkpoint_gib"),
    ({"exlude": []}, "unknown"),
])
def test_machine_model_list_is_validated(config, message):
    with pytest.raises(ConfigError, match=message):
        selection.select(config, CATALOG)


def test_summary_lists_excluded_models_unless_a_result_exists(tmp_path):
    root = tmp_path / "gb300-1"
    campaign.write_exclusions(root, [{"profile": "flux-2-dev", "task": "image_generation", "reason": "80 GB GPU"},
                                     {"profile": "qwen36-27b", "task": "text_generation", "reason": "80 GB GPU"}])
    other = tmp_path / "gb300-2/qwen36-27b"
    other.mkdir(parents=True)
    (other / "report.json").write_text(json.dumps({"task": "text_generation", "verdict": {"category": "pass"}}))
    text, counts = campaign.summary([tmp_path / "gb300-2", root])
    assert counts == {"pass": 1, "excluded": 1}
    assert "| White | flux-2-dev | image_generation | gb300-1 |" in text and "80 GB GPU" in text


def test_preflight_reports_missing_paths_and_interpreters(tmp_path):
    from trtmc_aiperf_qual import preflight

    (tmp_path / "repo/families").mkdir(parents=True)
    (tmp_path / "data").mkdir()
    worker = tmp_path / "rt/trtmc_benchmark_worker"
    worker.parent.mkdir()
    worker.write_text("#!/bin/sh\n")
    worker.chmod(0o755)
    environment = Environment({
        "repo": str(tmp_path / "repo"), "data_root": str(tmp_path / "data"), "bundle_root": str(tmp_path / "engines"),
        "runtime_root": str(tmp_path / "rt"), "worker": str(worker), "serve_python": str(tmp_path / "missing/python"),
        "aiperf": sys.executable,
        "hf_datasets_cache": str(tmp_path / "datasets"), "reference_env_root": str(tmp_path / "envs"),
        "ports": {"reference": 8900, "candidate": 8901}})
    checks = preflight.check_paths(environment)
    assert checks["repo"] == "ok" and checks["worker"] == "ok" and checks["bundle_root"] == "ok (created on first use)"
    assert checks["serve_python"].startswith("missing")
    assert preflight.problems(checks) == ["serve_python"]
    del environment.values["worker"]
    assert preflight.check_paths(environment)["worker"] == "missing key"


def test_summary_fetches_remote_result_roots_over_ssh(tmp_path):
    fake_ssh = tmp_path / "fake-ssh"
    fake_ssh.write_text('#!/usr/bin/env bash\n# fake-ssh HOST COMMAND: run COMMAND locally\nexec bash -c "$2"\n')
    fake_ssh.chmod(0o755)
    remote = tmp_path / "remote/results"
    (remote / "a").mkdir(parents=True)
    (remote / "a/report.json").write_text(json.dumps({"task": "classification", "verdict": {"category": "pass"}}))
    (remote / "a/model.json").write_text(json.dumps({"candidate": {"precision": "fp16"}}))
    (remote / "a/big.bin").write_bytes(b"x" * 1000)  # only result files are fetched
    roots = campaign.fetch_roots([f"gb300-1=nvidia@host:{remote}"], str(fake_ssh), tmp_path / "fetched")
    assert roots == [tmp_path / "fetched/gb300-1"] and not (roots[0] / "a/big.bin").exists()
    assert campaign.collect(roots)[0]["a"]["precision"]["trtmc"] == "fp16"  # the model's precision came along
    text, counts = campaign.summary(roots)
    assert counts == {"pass": 1} and "| Green | a | classification | gb300-1 |" in text
    local = tmp_path / "local-root"
    local.mkdir()
    assert campaign.fetch_roots([str(local)], str(fake_ssh), tmp_path / "fetched") == [local]


def test_summary_reports_an_unreachable_remote_root(tmp_path):
    fake_ssh = tmp_path / "fake-ssh"
    fake_ssh.write_text('#!/usr/bin/env bash\nexec bash -c "$2"\n')
    fake_ssh.chmod(0o755)
    with pytest.raises(ConfigError, match="cannot fetch"):
        campaign.fetch_roots([f"gb=host:{tmp_path / 'missing'}"], str(fake_ssh), tmp_path / "fetched")


def test_rejudge_can_take_todays_judging_settings(monkeypatch):
    from trtmc_aiperf_qual import cli, models

    recorded = {"catalog_profile": "bark-small", "absolute": [{"suite": "s", "gate": {"margin": 1.0}}],
                "supplementary": [{"check": "replay_parity"}],
                "performance": {"output_grader": "parity_audio", "reference_modes": ["eager"]}}
    today = {"absolute": [{"suite": "s", "gate": {"margin": 2.0}}], "accuracy_source": "absolute",
             "supplementary": [{"check": "replay_parity", "informational": True}],
             "performance": {"output_grader": "parity_audio", "output_grader_params": {"max_rms_ratio": 9.0},
                                    "reference_modes": ["eager", "compile"]}}
    monkeypatch.setattr(models, "resolve_model", lambda profile, environment: today)
    current = cli.current_settings(recorded, Environment({}))
    assert current["absolute"] == [{"suite": "s", "gate": {"margin": 2.0}}]
    assert current["supplementary"][0]["informational"]
    assert current["performance"]["output_grader_params"] == {"max_rms_ratio": 9.0}
    assert current["performance"]["reference_modes"] == ["eager"]  # what was measured stays


def test_summary_keeps_the_latest_result_of_a_profile_run_on_several_roots(tmp_path):
    for root, started, category in (("gb300-1", 200.0, "acc-issue"), ("gb300-2", 100.0, "pass")):
        (tmp_path / root / "m").mkdir(parents=True)
        (tmp_path / root / "m/report.json").write_text(json.dumps(
            {"task": "image_generation", "started": started, "verdict": {"category": category}}))
    for order in ([tmp_path / "gb300-1", tmp_path / "gb300-2"], [tmp_path / "gb300-2", tmp_path / "gb300-1"]):
        text, counts = campaign.summary(order)
        assert counts == {"acc-issue": 1} and "| Red | m | image_generation | gb300-1 |" in text


def test_summary_counts_every_planned_profile(tmp_path):
    root = tmp_path / "gb300-1"
    campaign.write_plan(root, ["a", "b"], [{"profile": "c", "reason": "no Task defaults"}])
    (root / "a").mkdir()
    (root / "a/report.json").write_text(json.dumps({"task": "t", "started": 1.0, "verdict": {"category": "pass"}}))
    text, counts = campaign.summary([root])
    assert counts == {"pass": 1, "not-run": 1, "config-error": 1}
    assert "| White | b | - | gb300-1 |" in text and "no Task defaults" in text


def test_run_all_exit_code_reports_harness_failures():
    assert campaign.exit_code([{"category": "pass"}, {"category": "acc-issue"}], []) == 0
    assert campaign.exit_code([{"category": "error"}], []) == 1
    assert campaign.exit_code([{"category": "build-failed"}], []) == 1
    assert campaign.exit_code([], [{"profile": "c", "reason": "x"}]) == 2


def test_summary_notes_trtmc_regressions_against_a_baseline(tmp_path):
    for root, p50 in (("now", 11.0), ("before", 10.0)):
        (tmp_path / root / "m").mkdir(parents=True)
        (tmp_path / root / "m/report.json").write_text(json.dumps({
            "task": "t", "started": 1.0, "verdict": {"category": "pass"},
            "performance": [{"reference_mode": "eager", "light": "green", "candidate": {"p50_ms": p50}}]}))
    text, _ = campaign.summary([tmp_path / "now"], [tmp_path / "before"])
    assert "regression: TRTMC eager p50 +10.0% vs baseline" in text


def test_html_report_lists_failures_first_with_evidence(tmp_path):
    from trtmc_aiperf_qual.report_html import render

    root = tmp_path / "gb300-1"
    for name, category in (("good", "pass"), ("bad", "acc-issue")):
        (root / name).mkdir(parents=True)
        (root / name / "report.md").write_text("x")
        (root / name / "report.json").write_text(json.dumps({
            "task": "t", "started": 1.0, "verdict": {"category": category}, "repro": f"trtmc-aiperf-qual run --profile {name}",
            "accuracy": [{"suite": "s", "status": "fail" if name == "bad" else "pass", "passed": 1, "samples": 2,
                          "required_passes": 2, "failures": [{"conversation_id": "session_000001",
                                                              "explanation": "answer differs", "actual": "B",
                                                              "expected": "C"}]}]}))
    rows, counts, rank = campaign.collect([root])
    page = render(rows, counts, rank, tmp_path / "report.html", context=campaign.run_context([root]),
                  links=[("Reruns", "reruns/report.html")], reruns={"bad": "green"}).read_text()
    assert '<a href="reruns/report.html">Reruns</a>' in page and "<th>Rerun</th>" in page
    assert page.index(">bad<") < page.index(">good<") and "answer differs" in page
    assert 'href="gb300-1/bad/report.md"' in page and "trtmc-aiperf-qual run --profile bad" in page
    bad = page[page.index(">bad<"):page.index(">good<")]
    assert "<span class='signal signal-green'><span class='light'></span>Green</span>" in bad  # the rerun's result
    assert "data-result='red' data-k='bad" in page and "<span>s</span><strong>1/2</strong>" in bad  # values only
    assert "<div class='detail'>Acc outside tolerance</div>" in page  # a short label; the reason is in the evidence
    assert "gb300-1: host -" in page and "Models <strong>2</strong>" in page
    assert "<span class='signal signal-yellow' title='Yellow'><span class='light'></span></span><strong>1</strong>" in page
    assert "Pass rate <strong>50.0%</strong>" in page  # Green + Yellow of every model
    legend = page[page.index("<dl class='legend'>"):page.index("</dl>")]
    assert legend.count("<div><dt>") == 4  # one line per result


def test_html_pass_rate_excludes_inconclusive_accuracy(tmp_path):
    from trtmc_aiperf_qual.report_html import render

    categories = {"g": "pass", "y": "acc-inconclusive", "r": "acc-issue", "w": "error"}
    rows = {name: {"category": category, "task": "t", "accuracy": [], "perf": [], "root": "h"}
            for name, category in categories.items()}
    page = render(rows, {}, {}, tmp_path / "report.html").read_text()
    green, yellow = (f"<span class='signal signal-{result}' title='{result.title()}'><span class='light'></span></span>"
                     for result in ("green", "yellow"))
    assert f"Pass {green}<span class='none'>+</span>{yellow}<strong>1</strong>" in page  # lights, not words
    assert "Pass rate <strong>25.0%</strong>" in page and "Valid comparisons <strong>3 / 4</strong>" in page
    assert "Pass rate <strong>—</strong>" in render({}, {}, {}, tmp_path / "empty.html").read_text()


def test_html_labels_say_in_a_few_words_why_a_result_is_not_green():
    from trtmc_aiperf_qual.report_html import _issue

    def timed(reason):
        return {"category": "perf-inconclusive", "accuracy": [], "perf": [{"light": "white", "reasons": [reason]}]}

    assert _issue("m", timed("TRTMC CI ±5.54% > 5.0%"), "white") == "Perf TRTMC CI ±5.54%"
    assert _issue("m", timed("reference timed at bf16, TRTMC runs fp16"), "white") == "Perf precision differs"
    assert _issue("m", timed("work differs: TRTMC [...] vs native [...]"), "white") == "Perf outputs differ"
    assert _issue("m", timed("TRTMC: 1 responses report no work"), "white") == "Perf work not reported"
    assert _issue("m", timed("native: no work evidence"), "white") == "Perf work not reported"
    scored = {"category": "acc-issue", "perf": [], "accuracy": [{"status": "fail", "metrics": {"trtmc_score": 1.0}}]}
    assert _issue("m", scored, "red") == "Acc below native"
    parity = {"category": "acc-issue", "perf": [], "accuracy": [{"status": "fail", "passed": 398, "samples": 400}]}
    assert _issue("m", parity, "red") == "Acc outside tolerance"
    unfit = {"category": "error", "perf": [], "accuracy": [
        {"status": "error", "error": "no lambada problem fits the bundle's 32-token sequence length"}]}
    assert _issue("m", unfit, "white") == "no problem fits the bundle"
    gated = {"category": "build-failed", "accuracy": [], "perf": [], "notes": "checkpoint: 403 Client Error"}
    assert _issue("m", gated, "white") == "HTTP 403"
    assert _issue("m", {"category": "not-covered", "accuracy": [], "perf": []}, "white") == "not covered"


def test_aggregate_results_report_metrics_not_a_zero_pass_count():
    from trtmc_aiperf_qual.report import counted

    assert counted({"passed": None, "samples": 0, "metrics": {"min_psnr": 10.1}}) == "min_psnr 10.1"
    assert counted({"passed": 3, "samples": 4}) == "3/4"


def test_serving_sweep_fits_the_bundle_and_compares_throughput():
    from trtmc_aiperf_qual import sweep

    assert sweep.lengths({"isl": 96, "osl": 32}, 129) == (81, 32) and sweep.lengths({}, None) == (96, 32)
    fast = [{"concurrency": 4, "request_throughput_avg": 20.0}]
    slow = [{"concurrency": 4, "request_throughput_avg": 10.0}]
    compared = sweep.compare(fast, slow, 5)
    assert compared["light"] == "green" and compared["throughput_ratio"] == 2.0
    assert sweep.compare(slow, fast, 5)["light"] == "red"
    assert sweep.compare([{**fast[0], "request_error_rate_avg": 3.0}], slow, 5)["light"] == "white"


def test_a_timed_request_must_state_its_generation_controls():
    from trtmc_aiperf_qual.suites import unstated_defaults

    assert unstated_defaults({"prompt": "cat", "num_steps": -1, "guidance_scale": -1.0, "num_frames": 17}) == \
        ["num_steps", "guidance_scale"]
    assert unstated_defaults({"prompt": "cat", "num_steps": 4}) == []


def test_latent_seeds_give_each_sample_its_own_replayed_noise():
    from trtmc_aiperf_qual.suites import Suite, request_sha, with_latent_seeds

    samples = [{"sample_id": str(i), "request": {"prompt": f"p{i}"}, "request_sha": "old"} for i in range(3)]
    seeded = with_latent_seeds(Suite("s", "key", samples, {"key": "key"}))
    assert [sample["request"]["latent_seed"] for sample in seeded.samples] == [1000, 1001, 1002]
    assert seeded.samples[1]["request_sha"] == request_sha({"prompt": "p1", "latent_seed": 1001})
    assert seeded.key != "key" and seeded.manifest["latent_seed_base"] == 1000 and "latent_seed" not in samples[0]["request"]


def test_replay_parity_compares_both_sides_with_the_full_precision_render(tmp_path):
    from trtmc_aiperf_qual import replay_parity

    def row(candidate_ref, native_ref, candidate=(20.0, 0.8)):
        return {"candidate": {"psnr": candidate[0], "ssim": candidate[1]},
                "candidate_ref": candidate_ref and {"psnr": candidate_ref[0], "ssim": candidate_ref[1]},
                "native_ref": native_ref and {"psnr": native_ref[0], "ssim": native_ref[1]}}

    samples = [{"sample_id": str(i), "request": {}} for i in range(5)]
    check = {"max_psnr_gap_db": 3.0, "max_ssim_gap": 0.05}
    same = [row((25.7, 0.91), (25.8, 0.91)), row((25.5, 0.90), (25.0, 0.90)), row((26.0, 0.92), (26.5, 0.92)),
            row(None, None), row(None, None)]
    verdict = replay_parity.judge(same, samples, check, planned=3)  # beyond the budget: informational only
    assert verdict["status"] == "pass" and verdict["samples"] == 3 and verdict["metrics"]["yardstick"] == "full precision"
    assert abs(verdict["metrics"]["psnr_gap_db"] - 0.1 / 3) < 1e-9  # gaps 0.1, -0.5, 0.5
    partial = replay_parity.judge(same, samples, check, planned=5)  # two planned renders missing: no silent removal
    assert partial["metrics"]["yardstick"] == "fallback" and partial["samples"] == 5
    assert partial["metrics"]["fallback_reason"] == "3 of 5 full-precision renders usable"
    lost = [dict(r, frames=[1, 0]) if i == 1 else r for i, r in enumerate(same[:3])]
    assert replay_parity.judge(lost, samples[:3], check)["status"] == "error"  # a native output missing
    gone = [dict(same[0], candidate=None, frames=[0, 1])] + same[1:3]
    failed_candidate = replay_parity.judge(gone, samples[:3], check)
    assert failed_candidate["status"] == "fail" and failed_candidate["failures"][0]["explanation"] == "no TRTMC output"
    worse = [row((13.9, 0.55), (24.7, 0.89)), row((14.5, 0.58), (25.0, 0.90)), row((13.0, 0.50), (24.0, 0.88))]
    failed = replay_parity.judge(worse, samples[:3], check)
    assert failed["status"] == "fail" and failed["passed"] == 0 and "vs full precision" in failed["failures"][0]["explanation"]
    noisy = [row((20.0, 0.9), (30.0, 0.9)), row((30.0, 0.9), (22.0, 0.9)), row((24.0, 0.9), (25.0, 0.9))]
    assert replay_parity.judge(noisy, samples[:3], check)["status"] == "pass"  # gap 1 dB on average
    overflow = [row((20.0, 0.8), (5.4, 0.0)), row((20.0, 0.8), (12.6, 0.04)), row((20.0, 0.8), (24.0, 0.9))]
    fallback = replay_parity.judge(overflow, samples[:3], check)  # most full-precision renders broke
    assert fallback["metrics"]["yardstick"] == "fallback" and fallback["status"] == "pass"
    assert fallback["gate"] == {"min_psnr_db": 19.0, "min_ssim": 0.8, "min_pass_rate": 0.9}
    record = {"observation": {"latent_replay": True}}
    assert replay_parity.not_replayed({"candidate": [(tmp_path, record)], "native": [(tmp_path, {"observation": {}})]}) == ["native"]


def test_sampled_frames_and_broken_floors(tmp_path):
    from trtmc_aiperf_qual import replay_parity
    from trtmc_aiperf_qual.generation import is_video, media_source

    record = {"observation": {"frame_artifacts": ["/x/000000.png", "/x/000062.png"], "artifact_indices": [0, 62]}}
    assert media_source(tmp_path, record) == {"dir": str(tmp_path), "files": ["/x/000000.png", "/x/000062.png"],
                                              "indices": [0, 62]}
    assert media_source(tmp_path, {"observation": {}}) == {"dir": str(tmp_path)}
    assert is_video({"media_type": "video"}) and is_video({"num_frames": 17}) and not is_video({"num_frames": 1})
    assert replay_parity.drop([2.0], 1.0) == (2.0, None, True)  # a single pair is judged by the limit alone
    samples = [{"sample_id": "0", "request": {"prompt": "p"}}]
    rows = [{"candidate": {"psnr": 18.0, "ssim": 0.75}, "candidate_ref": {"psnr": 9.0, "ssim": 0.1},
             "native_ref": {"psnr": 8.0, "ssim": 0.1}}]
    broken = replay_parity.judge(rows, samples, {})  # an 8 dB full-precision render broke: 19 dB / 0.8 instead
    assert broken["metrics"]["yardstick"] == "fallback" and broken["status"] == "fail"


def test_recheck_replaces_only_the_rechecked_entries(tmp_path, monkeypatch):
    import json

    from trtmc_aiperf_qual import cli, models, runner, services

    out = tmp_path / "m"
    out.mkdir()
    (out / "model.json").write_text(json.dumps({"catalog_profile": "m", "supplementary": []}))
    (out / "report.json").write_text(json.dumps({"accuracy": [
        {"suite": "family-case", "status": "pass"}, {"suite": "geneval", "status": "fail"},
        {"suite": "replay-parity", "status": "fail"}]}))
    check = {"check": "geneval", "suite": "geneval-200"}
    monkeypatch.setattr(models, "resolve_model", lambda profile, environment: {"supplementary": [check],
                                                                                "reference": {"backend": "reference"}})
    monkeypatch.setattr(services, "reference_python", lambda environment, model: "/ref/python")
    monkeypatch.setattr(runner, "supplementary", lambda environment, model, check, python, out: [
        {"suite": "geneval", "status": "pass"}])
    judged = []
    monkeypatch.setattr(cli, "rejudge_reports", lambda outs, environment: judged.append(outs) or 0)
    assert cli.recheck_reports([out], environment=object()) == 0 and judged == [[out]]
    suites = {item["suite"]: item["status"] for item in json.loads((out / "report.json").read_text())["accuracy"]}
    assert suites == {"family-case": "pass", "geneval": "pass", "replay-parity": "fail"}
    assert json.loads((out / "model.json").read_text())["supplementary"] == [check]


def test_recheck_reuses_a_generation_only_for_the_same_requests(tmp_path):
    import json

    from trtmc_aiperf_qual import generation
    from trtmc_aiperf_qual.suites import Suite

    suite = Suite("s", "k", [{"sample_id": "0", "request": {"prompt": "a", "latent_seed": 1000}}], {})
    out = tmp_path / "replay-candidate"
    (out / "scratch" / "r1").mkdir(parents=True)
    (out / "aiperf.inputs.jsonl").write_text(json.dumps({"text": json.dumps({"request": suite.samples[0]["request"]})}) + "\n")
    (out / "records.jsonl").write_text(json.dumps({"route": "/v1/tasks/generate_image", "request_id": "r1"}) + "\n")
    (out / "aiperf").mkdir()
    (out / "aiperf/inputs.json").write_text(json.dumps({"data": [{"session_id": "session_000000"}]}))
    (out / "aiperf/profile_export_raw.jsonl").write_text(json.dumps({
        "metadata": {"benchmark_phase": "profiling", "x_request_id": "r1",
                     "conversation_id": "session_000000"}, "status": 200,
        "payload": {"request": suite.samples[0]["request"]}}) + "\n")
    assert generation._earlier(out, suite) == [(out / "scratch" / "r1", {"route": "/v1/tasks/generate_image",
                                                                            "request_id": "r1"})]
    other = Suite("s", "k", [{"sample_id": "0", "request": {"prompt": "b"}}], {})
    assert generation._earlier(out, other) is None


def test_replay_parity_runs_only_for_families_that_take_caller_latents(tmp_path):
    from trtmc_aiperf_qual import replay_parity, runner

    assert replay_parity.run(None, {"family": "sana_wm"}, {"latent_replay_families": ["qwen_image"]}, "py",
                             tmp_path) == []
    assert runner.SUPPLEMENTARY_SUITES["replay_parity"] == ("replay-parity",)
    assert runner.SUPPLEMENTARY_CHECKS["replay_parity"] is replay_parity.run


def test_edit_inputs_are_center_cropped_to_cached_squares(tmp_path):
    import pytest

    Image = pytest.importorskip("PIL.Image")
    from trtmc_aiperf_qual.suites import _square_crop

    source = tmp_path / "wide.png"
    pixels = Image.new("RGB", (426, 320), (0, 0, 255))
    pixels.paste((255, 0, 0), (53, 0, 373, 320))  # the centered 320x320 square is red
    pixels.save(source)
    square = _square_crop(source, tmp_path / "cache")
    with Image.open(square) as image:
        assert image.size == (320, 320) and image.getpixel((0, 0)) == (255, 0, 0)
    assert _square_crop(source, tmp_path / "cache") == square and square.parent == tmp_path / "cache" / "trtmc-derived"


def test_gpu_busy_ignores_the_tail_of_our_own_request(monkeypatch):
    from types import SimpleNamespace

    from trtmc_aiperf_qual import runner

    def readings(values):
        sequence = iter(values)
        return lambda *args, **kwargs: SimpleNamespace(stdout=f"{next(sequence)}\n")

    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(subprocess, "run", readings([71, 40, 0, 0, 0]))
    assert runner.gpu_busy_percent(samples=5) == 0  # our own request decayed
    monkeypatch.setattr(subprocess, "run", readings([62, 66, 60, 64, 61]))
    assert runner.gpu_busy_percent(samples=5) == 60  # another process keeps the GPU busy


def test_gpu_probe_selects_the_benchmark_device(monkeypatch):
    from types import SimpleNamespace

    from trtmc_aiperf_qual import runner

    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-selected")
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: calls.append(args) or SimpleNamespace(stdout="0\n"))
    assert runner.gpu_busy_percent() == 0
    assert calls[0][1:3] == ["-i", "GPU-selected"]


def test_media_sweep_steps_decomposition_and_light(tmp_path):
    from trtmc_aiperf_qual import sweep
    from trtmc_aiperf_qual.report import write_report

    assert sweep.media_variants({"num_steps": 30}) == [15, 30] and sweep.media_variants({"num_steps": 1}) == [None]
    assert sweep.media_variants({"num_steps": 4}) == [2, 4] and sweep.media_variants({}) == [None]
    stats = sweep.media_stats([{"timing": {"model_call_ms": 300.0, "peak_memory_mb": 900.0}},
                               {"timing": {"model_call_ms": 100.0, "peak_memory_mb": 1000.0}},
                               {"timing": {"model_call_ms": 200.0}}])
    assert stats == {"measured": 3, "model_call_p50_ms": 200.0, "peak_memory_mb": 1000.0}
    candidate = [{"steps": 2, "model_call_p50_ms": 300.0}, {"steps": 4, "model_call_p50_ms": 500.0, "peak_memory_mb": 800.0}]
    reference = [{"steps": 2, "model_call_p50_ms": 900.0}, {"steps": 4, "model_call_p50_ms": 1500.0, "peak_memory_mb": 1600.0}]
    assert sweep.decompose(candidate) == {"per_step_ms": 100.0, "fixed_ms": 100.0}
    assert sweep.decompose(candidate[:1]) == {}
    compared = sweep.compare_media(candidate, reference, 5)
    assert compared["light"] == "green" and compared["speedup"] == 3.0 and compared["memory_ratio"] == 0.5
    assert sweep.compare_media(candidate, [{"steps": 4}], 5)["light"] == "white"
    rejected = [{"steps": 2, "model_call_p50_ms": None, "request_error_rate_avg": 100.0}, candidate[1]]
    partial = sweep.compare_media(rejected, reference, 5)
    assert partial["light"] == "green" and partial["notes"] == ["candidate failed at 2 steps"]
    assert sweep.decompose(rejected) == {}
    service_metrics = {"kind": "media", "endpoint": "image_generation", "prompts": 3, "requests": 3, "candidate": candidate,
          "reference": reference, "decomposition": {"candidate": sweep.decompose(candidate), "reference": {}},
          **compared}
    write_report(tmp_path, {"model": "m", "provenance": {}, "service_metrics": service_metrics})
    text = (tmp_path / "report.md").read_text()
    assert "AIPerf image_generation" in text and "100.0 ms per denoising step + 100.0 ms fixed" in text
    assert "| TRTMC | 4 | 500.000 |" in text


def test_perf_is_white_on_a_busy_gpu_or_a_fallback_reference_precision():
    from trtmc_aiperf_qual import judge

    fast = {"p50_ms": 1.0, "ci_percent": 0.1, "aggregation": "mean", "per_run_p50_ms": [1.0, 1.01, 0.99], "work": [[]]}
    slow = {"p50_ms": 10.0, "ci_percent": 0.1, "aggregation": "mean", "per_run_p50_ms": [10.0, 10.1, 9.9], "work": [[]]}
    ok = dict(margin_percent=5, max_ci_percent=5, outputs_match=True, output_reason="")
    assert judge.judge_performance(fast, slow, **ok)["light"] == "green"
    busy = judge.judge_performance({**fast, "gpu_busy_percent": 45}, slow, **ok)
    assert busy["light"] == "white" and "busy" in busy["reasons"][0]
    unmeasured = judge.judge_performance(fast, {**slow, "gpu_unmeasured_runs": 1}, **ok)  # nvidia-smi failed
    assert unmeasured["light"] == "white" and "not measured" in unmeasured["reasons"][0]
    fallback = judge.judge_performance(fast, {**slow, "precision": "fp32", "precision_fallback": "fp16: error"}, **ok)
    assert fallback["light"] == "white" and "fp32" in fallback["reasons"][0] and fallback["speedup"] == 10.0


def test_perf_lights_come_from_the_speedup_interval():
    from trtmc_aiperf_qual import judge

    ok = dict(margin_percent=5, max_ci_percent=5, outputs_match=True, output_reason="")
    side = lambda runs: {"p50_ms": sum(runs) / len(runs), "aggregation": "mean", "per_run_p50_ms": runs, "work": [[]],  # noqa: E731
                         "ci_percent": judge.across_runs(runs)["ci_percent"]}
    # 94 ms +-4% against 100 ms +-4%: the point estimate is 6% faster, but the interval reaches 1.0: not green
    noisy = judge.judge_performance(side([90.2, 94.0, 97.8]), side([96.0, 100.0, 104.0]), **ok)
    assert noisy["light"] in ("white", "yellow") and noisy["speedup_interval90"][0] < 1.05
    steady = judge.judge_performance(side([94.0, 94.1, 93.9]), side([100.0, 100.1, 99.9]), **ok)
    assert steady["light"] == "green"
    slower = judge.judge_performance(side([110.0, 110.1, 109.9]), side([100.0, 100.1, 99.9]), **ok)
    assert slower["light"] == "red"
    level = judge.judge_performance(side([100.0, 100.1, 99.9]), side([100.0, 100.1, 99.9]), **ok)
    assert level["light"] == "yellow"
    single = judge.judge_performance(side([94.0]), side([100.0]), **ok)
    assert single["light"] == "white" and "two runs" in single["reasons"][0]


def test_perf_needs_the_same_work_on_every_timed_response():
    from trtmc_aiperf_qual import judge

    ok = dict(margin_percent=5, max_ci_percent=5, outputs_match=True, output_reason="")
    runs = {"aggregation": "mean", "ci_percent": 0.1}
    candidate = {**runs, "p50_ms": 1.0, "per_run_p50_ms": [1.0, 1.0, 1.0], "work": [[["output_tokens", 20]]]}
    native = {**runs, "p50_ms": 10.0, "per_run_p50_ms": [10.0, 10.0, 10.0], "work": [[["output_tokens", 20]]]}
    assert judge.judge_performance(candidate, native, **ok)["light"] == "green"
    shorter = {**candidate, "work": [[["output_tokens", 5]], [["output_tokens", 20]]]}
    result = judge.judge_performance(shorter, native, **ok)
    assert result["light"] == "white" and "work differs" in result["reasons"][0]
    # [1, 2, 1] vs [1, 2, 2] tokens: the same sets, but responses did different work
    varied = judge.judge_performance({**candidate, "work": [[["output_tokens", 1]], [["output_tokens", 2]]]},
                                     {**native, "work": [[["output_tokens", 1]], [["output_tokens", 2]]]}, **ok)
    assert varied["light"] == "white"
    unproven = judge.judge_performance({**candidate, "work_missing": 1}, native, **ok)
    assert unproven["light"] == "white" and "report no work" in unproven["reasons"][0]
    # Text: equal counts or equal texts (one backend counts the end-of-sequence token, the other does not)
    marian = {**candidate, "work": [judge.work_signature("generate", {"output_tokens": 7, "text": "Дом."})]}
    stripped = {**native, "work": [judge.work_signature("generate", {"output_tokens": 6, "text": "Дом."})]}
    assert judge.judge_performance(marian, stripped, **ok)["light"] == "green"
    diverged = {**native, "work": [judge.work_signature("generate", {"output_tokens": 6, "text": "Дом!"})]}
    assert judge.judge_performance(marian, diverged, **ok)["light"] == "white"
    # Each operation's evidence, from fields both backends report
    assert judge.work_signature("generate", {"output_tokens": 3, "text": "a  b"}) == (("output_tokens", 3), ("text", "a  b"))
    padded = {**native, "work": [judge.work_signature("generate", {"output_tokens": 20, "text": "a" + "\n" * 18 + "b"})]}
    short = {**candidate, "work": [judge.work_signature("generate", {"output_tokens": 2, "text": "a b"})]}
    assert judge.judge_performance(short, padded, **ok)["light"] == "white"  # whitespace is generated work
    assert judge.work_signature("generate", {"scores": [1]}) is None
    assert judge.work_signature("transcribe", {"text": " a  b "}) == (("output_tokens", None), ("text", " a  b "))
    media = {"media_digest": {"frames": 17, "height": 480, "width": 832}}
    assert judge.work_signature("generate_image", media) == (("frames_height_width", (17, 480, 832)),)
    assert judge.work_signature("generate_image", {"height": 480, "width": 832}) is None
    assert judge.work_signature("generate_audio", {"audio_digest": {"seconds": 1.234}}) == (("audio_10ms", 123),)
    assert judge.work_signature("classify", {"scores": [0.1]}) == ()


def test_perf_is_white_when_the_native_model_ran_at_another_precision():
    from trtmc_aiperf_qual import judge

    ok = dict(margin_percent=5, max_ci_percent=5, outputs_match=True, output_reason="")
    fast = {"p50_ms": 1.0, "ci_percent": 0.1, "aggregation": "mean", "per_run_p50_ms": [1.0, 1.01, 0.99], "work": [[]]}
    slow = {"p50_ms": 10.0, "ci_percent": 0.1, "aggregation": "mean", "per_run_p50_ms": [10.0, 10.1, 9.9], "work": [[]]}
    declared = judge.judge_performance(fast, {**slow, "precision": "bf16"}, candidate_precision="fp16", **ok)
    assert declared["light"] == "white" and "bf16" in declared["reasons"][0]
    assert judge.judge_performance(fast, {**slow, "precision": "fp16"}, candidate_precision="fp16", **ok)["light"] == "green"


def test_long_prompts_are_not_taken_for_asset_paths(tmp_path):
    long_prompt = "Context filler. " * 100
    assert bundles._absolute_assets([{"prompt": long_prompt}], tmp_path) == [{"prompt": long_prompt}]


def test_results_follow_the_owners_four_colours_on_the_catalog_request():
    def row(category, lights, acc=()):
        return {"category": category, "accuracy": list(acc),
                "perf": [{"request": f"m-{name}", "light": light} for name, light in lights.items()]}

    green = {"catalog": "green"}
    assert campaign.signal(row("pass", green)) == "green"
    assert campaign.signal(row("perf-issue", {"catalog": "green", "catalog-near-capacity": "yellow"})) == "green"
    assert campaign.signal(row("perf-issue", {"catalog": "yellow"})) == "yellow"  # about equal: a pass
    assert campaign.signal(row("perf-issue", {"catalog": "red"})) == "red"
    assert campaign.signal(row("perf-inconclusive", {"catalog": "white"})) == "white"
    assert campaign.signal(row("acc-inconclusive", green)) == "yellow"
    assert campaign.signal(row("acc-issue", green)) == "red"
    assert campaign.signal(row("not-comparable", green)) == "white"
    assert campaign.signal(row("error", green)) == "white" and campaign.signal(row("build-failed", {})) == "white"
    assert campaign.signal(row("smoke-fail", green)) == "white"
    wer = {"suite": "librispeech", "status": "fail", "metrics": {"trtmc_score": 50.0, "native_score": 10.0}}
    assert campaign.signal_reason("m", row("acc-issue", green, [wer])) == "librispeech: TRTMC worse than native beyond the margin"
    informational = {"suite": "replay", "status": "fail", "informational": True}
    assert campaign.signal(row("pass", green, [informational])) == "green"
    compiled = row("pass", green)
    compiled["perf"].append({"request": "m-catalog", "light": "red", "reference_mode": "compile"})
    assert campaign.signal(compiled) == "green"  # torch.compile timings are informational
    shown = campaign.reported_perf(row("pass", {"catalog": "green", "catalog-near-capacity": "red"}))
    assert [campaign.request_label("m", item) for item in shown] == ["catalog"]
    assert campaign.signal_reason("m", row("perf-issue", {"catalog": "yellow"})) == "catalog: TRTMC about equal to native"


def test_image_qualification_defaults_to_the_pinned_full_geneval_selection():
    from trtmc_aiperf_qual.models import model_suite, resolve_model
    from trtmc_aiperf_qual.config import load_suite

    environment = Environment({"repo": str(REPOSITORY), "bundle_root": "/bundles",
                               "runtime_root": "/rt", "worker": "/rt/worker", "serve_python": "/py"})
    model = resolve_model("qwen-image", environment)
    check = next(item for item in model["supplementary"] if item["check"] == "geneval")
    suite = model_suite(check["suite"], model)
    assert suite["suite"] == "geneval-full" and suite["selection"]["count"] >= 553
    assert suite["source"] == load_suite("geneval-200")["source"]  # same pinned corpus and labels
    assert check["gate"] == {"margin": 5.0}


@pytest.mark.parametrize("complete, matched, warning", [
    (True, 10, ""),
    (True, 8, "work differs or is unknown on 2/10 pairs"),
    (False, 8, "Native 10/10 timed; TRTMC 8/10 timed"),
])
def test_summary_surfaces_work_and_coverage_without_changing_the_verdict(tmp_path, complete, matched, warning):
    from trtmc_aiperf_qual import report, report_html

    out = tmp_path / "model"
    out.mkdir()
    item = {"request": "evaluation", "reference_mode": "eager", "kind": "natural_dataset", "gate": False,
            "complete": complete, "comparable": matched == 10, "pairs": 10, "matched_pairs": matched,
            "measurement_status": "measured" if complete else "partial", "light": "informational",
            "reference": {"p50_ms": 123, "requests": 10, "valid_requests": 10},
            "candidate": {"p50_ms": 45, "requests": 10, "valid_requests": 10 if complete else 8}}
    result = {"model": "model", "provenance": {}, "performance_source": "quality", "performance": [item],
              "accuracy": [{"suite": "evaluation", "status": "pass"}],
              "verdict": {"acc": "pass", "perf": item["measurement_status"], "category": "measured"}}
    report.write_report(out, result)
    text, _ = campaign.summary([tmp_path])
    rows, counts, rank = campaign.collect([tmp_path])
    page = report_html.render(rows, counts, rank, tmp_path / "summary.html").read_text()
    assert warning in text and warning in page
    assert "123 ms" in page and "45.0 ms" in page and "Speedup" not in page
    assert campaign.signal(rows["model"]) == "green"  # observed work never adds a performance gate


def test_historical_preset_labels_show_the_actual_request_count():
    item = {"request": "geneval-200", "kind": "natural_dataset",
            "candidate": {"requests": 553}, "reference": {"requests": 553}}
    assert campaign.request_label("model", item) == "geneval (553 requests)"
    assert campaign.request_label("model", {**item, "request": "mmlu-0shot"}) == "mmlu-0shot"
    assert campaign.request_label("model", {**item, "kind": "fixed"}) == "geneval-200"


def test_quality_dataset_timings_are_the_main_report_and_keep_their_benchmark_label(tmp_path):
    from trtmc_aiperf_qual import cli, report, report_html

    out = tmp_path / "qwen"
    out.mkdir()
    item = {"request": "mmlu-0shot", "reference_mode": "eager", "kind": "natural_dataset", "gate": False,
            "complete": True, "comparable": True, "light": "informational", "pairs": 2, "matched_pairs": 2,
            "reference": {"p50_ms": 123.0, "precision": "fp16"},
            "candidate": {"p50_ms": 45.0, "precision": "fp16"}}
    result = {"model": "qwen", "task": "text_generation", "performance_source": "quality", "provenance": {},
              "accuracy_source": "absolute", "accuracy": [{"suite": "mmlu-0shot", "source": "absolute", "status": "pass"}],
              "performance": [item], "verdict": {"acc": "pass", "perf": "measured", "category": "measured"}}
    model = {"absolute": [{"suite": "mmlu-0shot"}], "accuracy_source": "absolute", "performance": {}}
    (out / "model.json").write_text(json.dumps(model))
    report.write_report(out, result)
    rows, counts, rank = campaign.collect([tmp_path])
    assert campaign.reported_perf(rows["qwen"]) == [item]
    page = report_html.render(rows, counts, rank, tmp_path / "summary.html").read_text()
    assert "mmlu-0shot" in page and "123 ms" in page and "45.0 ms" in page
    assert "p50 of the catalog" not in page and "Speedup" not in page
    cli.rejudge_reports([out])
    saved = json.loads((out / "report.json").read_text())
    assert saved["verdict"]["perf"] == "measured" and saved["performance"] == [item]
