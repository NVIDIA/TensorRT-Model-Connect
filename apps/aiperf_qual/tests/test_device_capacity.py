# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from trtmc_aiperf_qual import campaign, report_html
from trtmc_aiperf_qual.config import ConfigError, Environment

REPOSITORY = Path(__file__).resolve().parents[3]
LIMITED = {"model": "big-27b", "task": "text_generation", "candidate": {"checkpoint": "org/big"}, "reference": {},
           "capacity_limits": {"sm110": "needs more than the shared memory"}}


def _platform(monkeypatch, arch):
    monkeypatch.setattr(campaign, "_GPU_ARCH", {})
    calls = []

    def fingerprint(environment, python=None):
        calls.append(arch)
        return {"fingerprint": {"gpu_arch": arch}}

    monkeypatch.setattr(campaign, "platform_fingerprint", fingerprint)
    return calls


def _forbid_model_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a capacity-blocked profile downloaded, built or ran")

    for name in ("reference_python", "prefetch", "qualify", "warm_selections", "run_key"):
        monkeypatch.setattr(campaign, name, forbidden)
    monkeypatch.setattr(campaign.bundles, "ensure_bundle", forbidden)


def test_profile_beyond_the_device_capacity_is_reported_before_any_model_work(tmp_path, monkeypatch):
    _platform(monkeypatch, "sm110")
    _forbid_model_work(monkeypatch)
    directory = tmp_path / "results/big-27b"
    environment = Environment({"serve_python": "python"})
    record = campaign.run_one(environment, LIMITED, directory)
    assert record["category"] == "environment-blocked" and record["blocked_by"] == "device_capacity"
    assert record["reason"] == "device capacity (sm110): needs more than the shared memory"
    assert not (directory / "build.json").exists()
    assert json.loads((directory / "error.json").read_text())["blocked_by"] == "device_capacity"
    key = (directory / campaign.RUN_KEY).read_text().strip()
    assert key == campaign._sandbox_run_key(environment, LIMITED, directory, record["reason"], "device_capacity")
    assert campaign._finished(directory, key) == "environment-blocked"
    rows, counts, rank = campaign.collect([tmp_path / "results"])
    assert campaign.signal(rows["big-27b"]) == "white" and not campaign.is_pass(rows["big-27b"])
    page = report_html.render(rows, counts, rank, tmp_path / "summary.html").read_text()
    assert "device capacity (sm110)" in page


def test_batch_resumes_a_capacity_blocked_profile_without_downloading_it(tmp_path, monkeypatch):
    calls = _platform(monkeypatch, "sm110")
    _forbid_model_work(monkeypatch)
    environment = Environment({"serve_python": "python"})
    records = campaign.run_all(environment, [LIMITED, {**LIMITED, "model": "big-27b-2"}], tmp_path / "results")
    assert [r["category"] for r in records] == ["environment-blocked", "environment-blocked"]
    resumed, = campaign.run_all(environment, [LIMITED], tmp_path / "results")
    assert resumed["status"] == "skipped" and resumed["category"] == "environment-blocked"
    assert calls == ["sm110"]  # the platform is read once per serving interpreter
    assert campaign.exit_code([resumed], []) == 1


def test_other_architectures_and_profiles_without_limits_are_not_blocked(monkeypatch):
    calls = _platform(monkeypatch, "sm121")
    environment = Environment({"serve_python": "python"})
    assert campaign.device_capacity_limit(environment, LIMITED) is None
    assert campaign.device_capacity_limit(environment, {"model": "small"}) is None
    assert calls == ["sm121"]  # a profile without limits never reads the platform


def test_capacity_limits_name_gpu_architectures():
    from trtmc_aiperf_qual.models import capacity_limits

    assert capacity_limits("p", {"sm110": " too big "}) == {"sm110": "too big"}
    for bad in ({"110": "too big"}, {"sm110": ""}, ["sm110"]):
        with pytest.raises(ConfigError, match="capacity_limits"):
            capacity_limits("p", bad)


def test_27b_profiles_declare_the_thor_capacity_limit_and_quantized_variants_do_not_inherit_it():
    from trtmc_aiperf_qual.models import resolve_model

    environment = Environment({"repo": str(REPOSITORY)})
    for profile in ("gemma-3-27b", "qwen36-27b", "qwen38-27b"):
        limits = resolve_model(profile, environment)["capacity_limits"]
        assert set(limits) == {"sm110"} and "#2189" in limits["sm110"]
    for profile in ("qwen38-27b-fp8", "qwen38-27b-nvfp4"):
        assert "capacity_limits" not in resolve_model(profile, environment)


def test_a_failed_platform_probe_is_the_profile_error_and_never_builds(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "_GPU_ARCH", {})

    def broken(environment, python=None):
        raise RuntimeError("serving interpreter crashed")

    monkeypatch.setattr(campaign, "platform_fingerprint", broken)
    _forbid_model_work(monkeypatch)
    environment = Environment({"serve_python": "python"})
    records = campaign.run_all(environment, [LIMITED, {**LIMITED, "model": "big-27b-2"}], tmp_path / "results")
    assert [r["category"] for r in records] == ["error", "error"]  # the batch goes on to the next profile
    assert "platform probe for capacity_limits failed" in records[0]["reason"]
    assert not (tmp_path / "results/big-27b/build.json").exists()
    resumed = campaign.run_all(environment, [LIMITED], tmp_path / "results")
    assert resumed[0].get("status") != "skipped" and resumed[0]["category"] == "error"  # retried, not resumed
