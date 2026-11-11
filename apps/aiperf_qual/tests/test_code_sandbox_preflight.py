# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess

import pytest

from trtmc_aiperf_qual import campaign, gold_metrics, report_html
from trtmc_aiperf_qual.config import Environment


@pytest.mark.parametrize("failure", ["identity", "writable", "readable", "missing"])
def test_unavailable_sandbox_is_reported_before_any_model_work(tmp_path, monkeypatch, failure):
    protected = tmp_path / "private"
    protected.mkdir()
    monkeypatch.setattr(gold_metrics, "PROTECTED_READ", (str(protected),))
    calls = []

    def check(command, **kwargs):
        calls.append(command)
        if failure == "missing":
            raise FileNotFoundError("setpriv")
        code = (1 if failure == "identity" else 0) if command[-1] == "true" else (
            0 if (failure == "writable" and "test" in command)
            or (failure == "readable" and "ls" in command) else 1)
        return subprocess.CompletedProcess(command, code, stderr="Operation not permitted" if code else "")

    monkeypatch.setattr(subprocess, "run", check)
    monkeypatch.setattr(campaign, "run_key", lambda *args: "test-key")

    def forbidden(*args, **kwargs):
        pytest.fail("model preparation or generation ran without a sandbox")

    for name in ("reference_python", "prefetch", "qualify", "warm_selections"):
        monkeypatch.setattr(campaign, name, forbidden)
    monkeypatch.setattr(campaign.bundles, "ensure_bundle", forbidden)
    model = {"model": "codegen", "task": "text_generation", "candidate": {},
             "absolute": [{"suite": "code", "metric": "code_pass"}]}
    directory = tmp_path / "results/codegen"
    record = campaign.run_one(Environment({}), model, directory)
    assert record["category"] == "environment-blocked"
    assert record["failure_type"] == "environment" and record["blocked_by"] == "code_sandbox"
    assert "code sandbox" in record["reason"] and calls
    assert not (directory / "build.json").exists()
    assert json.loads((directory / "error.json").read_text())["category"] == "environment-blocked"
    assert campaign._finished(directory, (directory / campaign.RUN_KEY).read_text().strip()) == "environment-blocked"
    rows, counts, rank = campaign.collect([tmp_path / "results"])
    assert rows["codegen"]["category"] == "environment-blocked"
    assert campaign.signal(rows["codegen"]) == "white" and not campaign.is_pass(rows["codegen"])
    assert campaign.exit_code([record], []) == 1
    page = report_html.render(rows, counts, rank, tmp_path / "summary.html").read_text()
    assert "environment blocked" in page and "code sandbox" in page


def test_non_code_profile_does_not_require_a_code_sandbox(tmp_path, monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("a non-code profile invoked setpriv")

    monkeypatch.setattr(gold_metrics, "_sandbox_verified", forbidden)
    monkeypatch.setattr(campaign, "run_key", lambda *args: "test-key")
    monkeypatch.setattr(campaign, "warm_selections", lambda *args: None)
    monkeypatch.setattr(campaign, "reference_python", lambda *args: "python")
    monkeypatch.setattr(campaign, "prefetch", lambda *args: calls.append("prefetch"))
    monkeypatch.setattr(campaign.bundles, "ensure_bundle", lambda *args: {"status": "failed", "reason": "test"})
    model = {"model": "text", "task": "text_generation", "candidate": {},
             "absolute": [{"suite": "qa", "metric": "choice"}]}
    assert campaign.run_one(Environment({}), model, tmp_path / "text")["category"] == "build-failed"
    assert calls == ["prefetch"]


def test_batch_does_not_prefetch_or_delete_checkpoints_for_blocked_code_profiles(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 1))
    monkeypatch.setattr(campaign, "run_key", lambda *args: "test-key")
    monkeypatch.setattr(campaign, "prefetch", lambda *args: calls.append("prefetch"))
    monkeypatch.setattr(campaign.retention, "delete_checkpoint", lambda *args: calls.append("delete"))
    model = {"model": "codegen", "task": "text_generation", "candidate": {"checkpoint": "org/code"},
             "reference": {}, "absolute": [{"suite": "code", "metric": "code_pass"}]}
    env = Environment({"retention": {"hf_cache": "delete_unused"}, "hf_hub_cache": str(tmp_path / "cache")})
    records = campaign.run_all(env, [model, {**model, "model": "starcoder"}], tmp_path / "results")
    assert [r["category"] for r in records] == ["environment-blocked", "environment-blocked"]
    assert calls == []
    resumed = campaign.run_all(env, [model], tmp_path / "results")
    assert resumed[0]["status"] == "skipped" and resumed[0]["category"] == "environment-blocked"
    assert campaign.exit_code(resumed, []) == 1


def test_program_still_rechecks_the_boundary_after_a_successful_preflight(tmp_path, monkeypatch):
    calls = []

    def check(paths):
        calls.append(paths)
        if len(calls) > 1:
            raise gold_metrics.SandboxUnavailable("code sandbox: permissions changed")

    monkeypatch.setattr(gold_metrics, "_sandbox_verified", check)
    model = {"absolute": [{"metric": "code_pass"}]}
    gold_metrics.preflight_code_sandbox(model, {}, tmp_path)
    with pytest.raises(gold_metrics.SandboxUnavailable, match="permissions changed"):
        gold_metrics.code_pass({}, {"text": "raise SystemExit(0)"})
    assert len(calls) == 2


def test_direct_qualification_rejects_the_environment_before_starting_a_workload(tmp_path, monkeypatch):
    from trtmc_aiperf_qual import runner

    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 1))

    def forbidden(*args, **kwargs):
        pytest.fail("qualification workload started without a code sandbox")

    monkeypatch.setattr(runner, "_qualify", forbidden)
    with pytest.raises(gold_metrics.SandboxUnavailable, match="code sandbox"):
        runner.qualify({"absolute": [{"metric": "code_pass"}]}, Environment({}), tmp_path)
    assert not (tmp_path / "execution.jsonl").exists()


@pytest.mark.parametrize("batch", [False, True])
def test_sandbox_probe_precedes_reference_environment_setup_in_the_real_run_key(tmp_path, monkeypatch, batch):
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 1))

    def forbidden(*args, **kwargs):
        pytest.fail("run-key computation prepared the reference environment before sandbox preflight")

    monkeypatch.setattr(campaign, "reference_python", forbidden)
    model = {"model": "codegen", "task": "text_generation", "candidate": {"checkpoint": "org/code"},
             "reference": {"requirements": ["package"]}, "absolute": [{"suite": "code", "metric": "code_pass"}]}
    env = Environment({"reference_env_root": str(tmp_path / "envs")})
    record = (campaign.run_all(env, [model], tmp_path / "results")[0] if batch else
              campaign.run_one(env, model, tmp_path / "results/codegen"))
    assert record["category"] == "environment-blocked"


def test_resume_replaces_a_stale_environment_diagnostic_when_the_boundary_failure_changes(tmp_path, monkeypatch):
    reason = ["code sandbox: cannot switch to nobody"]

    def unavailable(paths):
        raise gold_metrics.SandboxUnavailable(reason[0])

    monkeypatch.setattr(gold_metrics, "_sandbox_verified", unavailable)
    model = {"model": "codegen", "reference": {}, "candidate": {}, "absolute": [{"metric": "code_pass"}]}
    root = tmp_path / "results"
    campaign.run_all(Environment({}), [model], root)
    reason[0] = "code sandbox: nobody can write the result root"
    record, = campaign.run_all(Environment({}), [model], root)
    assert record.get("status") != "skipped" and "can write" in record["reason"]
    assert "can write" in json.loads((root / "codegen/error.json").read_text())["reason"]
