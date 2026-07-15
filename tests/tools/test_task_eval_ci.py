# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest

from tools import task_eval_ci


def _suite(source: str = "https://example.com/ETTh1.csv", digest: str = "a" * 64) -> dict:
    return {
        "id": "etth1_time_series_parity",
        "default_model_names": ["chronos", "timesfm"],
        "dataset": {
            "default_path": "/missing/ETTh1.csv",
            "source": source,
            "sha256": digest,
        },
        "ci": {
            "eligible": True,
            "lane": "nightly",
            "limit": 10,
            "sample_seed": 20260715,
        },
    }


def test_real_etth1_suite_is_nightly_ci_eligible() -> None:
    suite = task_eval_ci.load_ci_suite(
        task_eval_ci.DEFAULT_SUITES, "etth1_time_series_parity", "nightly"
    )

    assert suite["ci"] == {
        "eligible": True,
        "lane": "nightly",
        "limit": 10,
        "sample_seed": 20260715,
        "notes": suite["ci"]["notes"],
    }
    assert len(suite["default_model_names"]) == 5


def test_load_ci_suite_rejects_wrong_lane(tmp_path: Path) -> None:
    suites = tmp_path / "suites.yaml"
    suites.write_text(
        "suites:\n"
        "  - id: test\n"
        "    default_model_names: [model]\n"
        "    ci: {eligible: true, lane: nightly, limit: 1, sample_seed: 0}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="belongs to lane"):
        task_eval_ci.load_ci_suite(suites, "test", "premerge")


def test_ensure_dataset_downloads_and_verifies_pinned_source(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_bytes(b"date,OT\n2026-01-01,1.0\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    dataset = task_eval_ci.ensure_dataset(
        _suite(source.as_uri(), digest), explicit_path=None, cache_root=tmp_path / "cache"
    )

    assert dataset.read_bytes() == source.read_bytes()
    assert task_eval_ci.sha256_file(dataset) == digest


def test_ensure_dataset_rejects_explicit_checksum_mismatch(tmp_path: Path) -> None:
    dataset = tmp_path / "ETTh1.csv"
    dataset.write_text("wrong", encoding="utf-8")

    with pytest.raises(ValueError, match="wrong checksum"):
        task_eval_ci.ensure_dataset(
            _suite(), explicit_path=dataset, cache_root=tmp_path / "cache"
        )


def test_build_eval_command_uses_fixed_ci_models_limit_and_seed(tmp_path: Path) -> None:
    args = argparse.Namespace(
        python=sys.executable,
        suites=tmp_path / "suites.yaml",
        work_root=tmp_path / "work",
        engine_dir=tmp_path / "engines",
        waive_platform="GB300",
        trtmc_binary="trtmc",
        hf_python=sys.executable,
        model_plugin_dir=tmp_path / "plugins",
        cuda_visible_devices="1",
    )

    command = task_eval_ci.build_eval_command(args, _suite(), tmp_path / "ETTh1.csv")

    assert command.count("--model") == 2
    assert command[command.index("--limit") + 1] == "10"
    assert command[command.index("--sample-seed") + 1] == "20260715"
    assert command[command.index("--cuda-visible-devices") + 1] == "1"
    assert "--local-files-only" in command


def test_validate_eval_summary_fails_closed_on_timesfm() -> None:
    passed, results = task_eval_ci.validate_eval_summary(
        {
            "results": [
                {"model": "chronos", "status": "passed"},
                {"model": "timesfm", "status": "failed"},
            ]
        },
        ["chronos", "timesfm"],
    )

    assert passed is False
    assert len(results) == 2


def test_public_artifacts_omit_runner_paths_and_copy_numeric_summary(tmp_path: Path) -> None:
    work_root = tmp_path / "private-work"
    numeric = work_root / "etth1_time_series_parity" / "chronos" / "summary.json"
    numeric.parent.mkdir(parents=True)
    numeric.write_text(
        '{"status": "passed", "cases": [], "private_path": "/private/numeric"}\n',
        encoding="utf-8",
    )
    results = [
        {
            "suite": "etth1_time_series_parity",
            "model": "chronos",
            "status": "passed",
            "sample_agreement_rate": 1.0,
            "max_relative_l2": 1e-7,
            "max_absolute_error": 1e-6,
            "work_dir": "/private/runner/work",
            "bundle": "/private/runner/engine.trtfb",
        },
        {"model": "timesfm", "status": "failed", "error": "/private/error"},
    ]
    artifact_dir = tmp_path / "public"

    task_eval_ci.write_public_artifacts(
        suite=_suite(),
        results=results,
        work_root=work_root,
        artifact_dir=artifact_dir,
    )

    public = (artifact_dir / "eval_summary.json").read_text(encoding="utf-8")
    assert "/private" not in public
    assert "work_dir" not in public
    assert "bundle" not in public
    numeric_public = (
        artifact_dir / "models" / "chronos" / "summary.json"
    ).read_text(encoding="utf-8")
    assert "/private" not in numeric_public
    assert "private_path" not in numeric_public
    assert json.loads(numeric_public)["status"] == "passed"
