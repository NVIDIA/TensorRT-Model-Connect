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
                               "runtime_root": "/rt", "worker": "/rt/worker"})
    qwen = resolve_model("qwen3-0.6b-fp16", environment)
    command = bundles.build_command(environment, qwen, "/py", tmp_path / "qwen")
    assert command[:4] == ["/py", "-m", "trtmc_benchmark", "run"] and "--prepare-only" in command
    assert command[command.index("--model") + 1] == "qwen3-0.6b-fp16" and "--manifest-root" in command
    assert command[command.index("--bundle-cache") + 1] == str(tmp_path / "engines")

    detr = resolve_model("detr-resnet-50", environment)
    assert detr["candidate"]["bundle"] == "detr-resnet-50-q1333/detr-resnet-50-q1333.bundle"
    command = bundles.build_command(environment, detr, "/py", tmp_path / "detr")
    assert "--manifest-root" not in command
    descriptor = json.loads(Path(command[command.index("--model") + 1]).read_text())
    assert (descriptor["name"], descriptor["bundle"], descriptor["image_height"]) == (
        "detr-resnet-50-q1333", "detr-resnet-50-q1333.bundle", 1333)
    image = Path(descriptor["testcases"][0]["test_image"])
    assert image.is_absolute() and image.is_file()  # catalog assets stay reachable from the descriptor


def test_build_from_a_family_prepared_model_directory(tmp_path):
    from trtmc_aiperf_qual.models import resolve_model

    environment = Environment({"repo": str(REPOSITORY), "bundle_root": str(tmp_path / "engines"),
                               "runtime_root": "/rt", "worker": "/rt/worker"})
    stereo = resolve_model("fast-foundation-stereo", environment)
    python = tmp_path / "env/bin/python"
    with pytest.raises(bundles.BuildError, match="model directory"):
        bundles.build_command(environment, stereo, str(python), tmp_path / "out")
    prepared = tmp_path / "env/trtmc-reference/Fast-FoundationStereo"
    prepared.mkdir(parents=True)
    command = bundles.build_command(environment, stereo, str(python), tmp_path / "out")
    descriptor = json.loads(Path(command[command.index("--model") + 1]).read_text())
    assert (descriptor["hf_id"], descriptor["hf_revision"]) == (str(prepared.resolve()), "")


def test_ensure_bundle_reuses_an_existing_bundle(tmp_path, monkeypatch):
    environment = Environment({"bundle_root": str(tmp_path / "engines")})
    bundle = tmp_path / "engines/m/m.bundle"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("x")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("must not build"))
    assert bundles.ensure_bundle(environment, _model("m", "org/m"), "/py", tmp_path / "out")["status"] == "reused"


def test_ensure_bundle_records_a_failed_build_under_the_gpu_lock(tmp_path, monkeypatch):
    environment = Environment({"repo": str(tmp_path), "bundle_root": str(tmp_path / "engines"),
                               "gpu_lock": str(tmp_path / "gpu.lock")})
    failing = [sys.executable, "-c", "import sys; print('step'); print('ValueError: boom'); sys.exit(3)"]
    monkeypatch.setattr(bundles, "build_command", lambda *args: failing)
    result = bundles.ensure_bundle(environment, _model("m", "org/m"), "/py", tmp_path / "out")
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


def test_run_all_resumes_and_reruns_on_request(tmp_path, monkeypatch):
    events = []
    _record_runs(monkeypatch, events)
    done = tmp_path / "out/a"
    done.mkdir(parents=True)
    (done / "report.json").write_text(json.dumps({"verdict": {"category": "pass"}}))
    models = [_model("a", "org/a"), _model("b", "org/b")]
    campaign.run_all(Environment({}), models, tmp_path / "out", prefetch_next=False)
    assert events == [("run", "b")]
    campaign.run_all(Environment({}), models[:1], tmp_path / "out", rerun=True, prefetch_next=False)
    assert events[-1] == ("run", "a")
    assert [path.name.startswith("a.") for path in (tmp_path / "out").iterdir()].count(True) == 1  # kept aside


def _stub_phases(monkeypatch, *, build="built", qualify=None):
    monkeypatch.setattr(campaign, "reference_python", lambda environment, model: "/py")
    monkeypatch.setattr(campaign, "prefetch", lambda environment, model: None)
    monkeypatch.setattr(campaign.bundles, "ensure_bundle",
                        lambda environment, model, python, out: {"status": build, "reason": "ValueError: boom"})
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
        "performance_l1": [{"reference_mode": "eager", "light": "green", "speedup": 2.5}]}))
    failed = tmp_path / "gb300-2/b"
    failed.mkdir(parents=True)
    (failed / "build.json").write_text(json.dumps({"task": "classification", "status": "failed",
                                                   "reason": "error: checkpoint is gated"}))
    text, counts = campaign.summary([tmp_path / "gb300-1", tmp_path / "gb300-2"])
    assert counts == {"pass": 1, "build-failed": 1}
    assert "| a | text_generation | gb300-1 | pass | mmlu 10/10 (need 9) | eager green 2.50x |" in text
    assert "| b | classification | gb300-2 | build-failed |" in text and "checkpoint is gated" in text


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
    assert "| flux-2-dev | image_generation | gb300-1 | excluded |" in text and "80 GB GPU" in text


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
        "aiperf": sys.executable, "golden_store": {"root": str(tmp_path / "goldens")},
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
    (remote / "a/big.bin").write_bytes(b"x" * 1000)  # only result files are fetched
    roots = campaign.fetch_roots([f"gb300-1=nvidia@host:{remote}"], str(fake_ssh), tmp_path / "fetched")
    assert roots == [tmp_path / "fetched/gb300-1"] and not (roots[0] / "a/big.bin").exists()
    text, counts = campaign.summary(roots)
    assert counts == {"pass": 1} and "| a | classification | gb300-1 | pass |" in text
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

    recorded = {"catalog_profile": "bark-small", "accuracy": [{"suite": {"suite": "s"}, "gate": {"min_pass_rate": 1.0}}],
                "performance": {"l1": {"output_grader": "parity_audio", "reference_modes": ["eager"]}}}
    today = {"accuracy": [{"suite": {"suite": "s"}, "gate": {"min_pass_rate": 0.8}, "sampled": True}],
             "performance": {"l1": {"output_grader": "parity_audio", "output_grader_params": {"max_rms_ratio": 9.0},
                                    "reference_modes": ["eager", "compile"]}}}
    monkeypatch.setattr(models, "resolve_model", lambda profile, environment: today)
    current = cli.current_settings(recorded, Environment({}))
    assert current["accuracy"][0] == {"suite": {"suite": "s"}, "gate": {"min_pass_rate": 0.8}, "sampled": True}
    assert current["performance"]["l1"]["output_grader_params"] == {"max_rms_ratio": 9.0}
    assert current["performance"]["l1"]["reference_modes"] == ["eager"]  # what was measured stays


def test_summary_keeps_the_latest_result_of_a_profile_run_on_several_roots(tmp_path):
    for root, started, category in (("gb300-1", 200.0, "acc-issue"), ("gb300-2", 100.0, "pass")):
        (tmp_path / root / "m").mkdir(parents=True)
        (tmp_path / root / "m/report.json").write_text(json.dumps(
            {"task": "image_generation", "started": started, "verdict": {"category": category}}))
    for order in ([tmp_path / "gb300-1", tmp_path / "gb300-2"], [tmp_path / "gb300-2", tmp_path / "gb300-1"]):
        text, counts = campaign.summary(order)
        assert counts == {"acc-issue": 1} and "| m | image_generation | gb300-1 | acc-issue |" in text
