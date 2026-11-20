# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from trtmc_aiperf_qual import absolute, gold_metrics
from trtmc_aiperf_qual.config import Environment
from trtmc_aiperf_qual.services import _serve_env, build_env

REPOSITORY = Path(__file__).resolve().parents[3]


def test_remote_code_cache_is_writable_beside_the_owned_hub_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_MODULES_CACHE", raising=False)
    env = Environment({"repo": str(REPOSITORY), "hf_hub_cache": str(tmp_path / "hf-hub")})
    assert _serve_env(env)["HF_MODULES_CACHE"] == str(tmp_path / "hf-modules")
    monkeypatch.setenv("HF_MODULES_CACHE", str(tmp_path / "explicit"))
    assert _serve_env(env)["HF_MODULES_CACHE"] == str(tmp_path / "explicit")


def test_serve_env_extends_the_environment_of_servers_and_builds():
    env = Environment({"repo": str(REPOSITORY), "serve_env": {"LD_PRELOAD": "/opt/cudnn/libcudnn.so.9", "RETRIES": 2},
                       "build_env": {"BUILD_ONLY": "1"}})
    assert {key: _serve_env(env)[key] for key in ("LD_PRELOAD", "RETRIES")} == {
        "LD_PRELOAD": "/opt/cudnn/libcudnn.so.9", "RETRIES": "2"}
    assert {key: build_env(env)[key] for key in ("LD_PRELOAD", "BUILD_ONLY")} == {
        "LD_PRELOAD": "/opt/cudnn/libcudnn.so.9", "BUILD_ONLY": "1"}
    assert "BUILD_ONLY" not in _serve_env(env)


def mask(values, height=2, width=2):
    return {"height": height, "width": width, "mask": values}


def test_binary_mask_parity_requires_valid_dimensions_labels_and_foreground_overlap():
    compare = gold_metrics.PARITY["mask_parity"]
    gate = {"min_mask_iou": 0.9}
    native = mask([0, 1, 1, 0])
    assert compare(native, native, gate)[0] is True
    assert compare(mask([0, 1, 0, 0]), native, gate)[0] is False
    assert compare(mask([0, 1, 1, 0], 1, 4), native, gate)[0] is False
    assert compare(mask([0, 2, 1, 0]), native, gate)[0] is False
    assert compare(mask([0, float("nan"), 1, 0]), native, gate)[0] is False
    assert compare(mask([0, 1]), native, gate)[0] is None
    assert compare(None, native, gate)[0] is None
    empty = mask([0, 0, 0, 0])
    assert compare(empty, empty, gate)[0] is True


def test_mask_conversion_judges_every_pair_and_keeps_native_failures_as_errors():
    item = {"suite": "binary-masks", "metric": "mask_parity", "gate": {"min_mask_iou": 0.9}}
    problems = [{"sample_id": "first"}, {"sample_id": "second"}]
    native = {"observations": {"greedy": {0: mask([0, 1, 1, 0]), 1: mask([1, 0, 0, 1])}}, "exit": {}}
    assert absolute.judge(item, problems, native, native)["status"] == "pass"
    failed = {"observations": {"greedy": {0: mask([0, 1, 1, 0]), 1: mask([0, 0, 0, 0])}}, "exit": {}}
    result = absolute.judge(item, problems, failed, native)
    assert result["status"] == "fail" and result["passed"] == 1
    assert result["failures"][0]["sample_id"] == "second"
    missing = {"observations": {"greedy": {0: mask([0, 1, 1, 0])}}, "exit": {}}
    assert absolute.judge(item, problems, native, missing)["status"] == "error"
