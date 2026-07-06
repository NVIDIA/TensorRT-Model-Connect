# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/generate_ci_summary.py."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


def _import_summary():
    scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return importlib.import_module("generate_ci_summary")


def _result(
    name: str,
    status: str = "pass",
    model_name: str | None = None,
) -> dict:
    return {
        "case_name": name,
        "status": status,
        "failure_type": "compare_fail" if status == "fail" else None,
        "case_config": {
            "family": "example_decoder",
            "task_strategy": "text_generation_causal",
            "metadata": {"model_name": model_name or name},
        },
        "stages": {
            "generate": {
                "status": "failed" if status == "fail" else "passed",
                "metrics": {
                    "token_agreement_rate": {
                        "value": 0.5 if status == "fail" else 1.0,
                        "threshold": 0.8,
                        "operator": ">=",
                    }
                },
            }
        },
        "timing": {"build_s": 1.0, "trt_generate_s": 2.0},
    }


def _write_result(
    artifacts_dir: Path,
    name: str,
    status: str = "pass",
    model_name: str | None = None,
) -> None:
    case_dir = artifacts_dir / name
    case_dir.mkdir(parents=True)
    (case_dir / "result.json").write_text(
        json.dumps(_result(name, status, model_name)), encoding="utf-8"
    )


def _write_junit(e2e_root: Path, body: str) -> None:
    e2e_root.mkdir(parents=True, exist_ok=True)
    (e2e_root / "junit-gpu0-shared-w0.xml").write_text(
        f'<?xml version="1.0" encoding="utf-8"?><testsuite>{body}</testsuite>',
        encoding="utf-8",
    )


def test_summary_lists_every_e2e_case_status_even_when_failure_rows_are_limited(
    tmp_path: Path,
) -> None:
    mod = _import_summary()
    artifacts_dir = tmp_path / "artifacts"
    for idx in range(6):
        _write_result(artifacts_dir, f"model-{idx}", "fail" if idx < 4 else "pass")
    report_path = tmp_path / "e2e_report.html"
    report_path.write_text("<html></html>", encoding="utf-8")

    summary = mod.render_summary(
        results=mod._load_results(artifacts_dir),
        mode="nightly",
        report_path=report_path,
        html_artifact_name="trtmc-nightly-html-report-123",
        full_artifact_name="trtmc-nightly-123",
        run_url="https://example.test/run",
        max_rows=2,
    )

    assert "HTML report artifact: `trtmc-nightly-html-report-123`" in summary
    assert "Showing 2 of 4 non-passing cases." in summary
    assert "### All E2E Model Status" in summary
    for idx in range(6):
        assert f"| model-{idx} |" in summary


def test_summary_handles_runs_without_e2e_results(tmp_path: Path) -> None:
    mod = _import_summary()

    summary = mod.render_summary(
        results=mod._load_results(tmp_path / "missing"),
        mode="premerge",
        report_path=tmp_path / "missing.html",
        html_artifact_name="trtmc-ci-html-report-123",
        full_artifact_name="trtmc-ci-123",
        run_url="",
        max_rows=40,
    )

    assert "No E2E `result.json` files were found" in summary
    assert "HTML report contents: not generated for this run" in summary


def test_summary_adds_junit_only_skips(tmp_path: Path) -> None:
    mod = _import_summary()
    e2e_root = tmp_path / "e2e_artifacts"
    artifacts_dir = e2e_root / "artifacts"
    _write_result(artifacts_dir, "example-decoder", "pass")
    _write_junit(
        e2e_root,
        """
        <testcase classname="tests.test_e2e" name="test_e2e[example-decoder]" />
        <testcase classname="tests.test_e2e" name="test_e2e[decoder-gated]">
          <skipped type="pytest.skip" message="(gated model, needs HF_TOKEN)" />
        </testcase>
        """,
    )
    report_path = e2e_root / "e2e_report.html"
    report_path.write_text("<html></html>", encoding="utf-8")

    results = mod._merge_pytest_outcomes(
        mod._load_results(artifacts_dir),
        mod._load_pytest_outcomes(e2e_root),
    )
    summary = mod.render_summary(
        results=results,
        mode="nightly",
        report_path=report_path,
        html_artifact_name="html",
        full_artifact_name="full",
        run_url="",
        max_rows=40,
    )

    assert "| skip | 1 |" in summary
    assert "| pass | 1 |" in summary
    assert "| decoder-gated |  |  | skip |  |  |" in summary
    assert "SKIPPED: gated model, needs HF_TOKEN" in summary


def test_summary_surfaces_xpass_from_console_logs(tmp_path: Path) -> None:
    mod = _import_summary()
    e2e_root = tmp_path / "e2e_artifacts"
    artifacts_dir = e2e_root / "artifacts"
    _write_result(artifacts_dir, "image-diffusion-xpass", "pass")
    e2e_root.mkdir(parents=True, exist_ok=True)
    (e2e_root / "console-gpu0-shared-w0.log").write_text(
        "tests/test_e2e.py::test_e2e[image-diffusion-xpass] "
        "XPASS ((HF diffusion reference quality should be gated)) [100%]\n",
        encoding="utf-8",
    )

    results = mod._merge_pytest_outcomes(
        mod._load_results(artifacts_dir),
        mod._load_pytest_outcomes(e2e_root),
    )
    summary = mod.render_summary(
        results=results,
        mode="nightly",
        report_path=e2e_root / "missing.html",
        html_artifact_name="html",
        full_artifact_name="full",
        run_url="",
        max_rows=40,
    )

    assert "### Pytest Waive Outcomes" in summary
    assert (
        "| image-diffusion-xpass | XPASS | pass | HF diffusion reference quality should be gated |"
        in summary
    )


def test_summary_treats_xfail_result_as_waived_not_active_failure(
    tmp_path: Path,
) -> None:
    mod = _import_summary()
    e2e_root = tmp_path / "e2e_artifacts"
    artifacts_dir = e2e_root / "artifacts"
    _write_result(artifacts_dir, "encoder-base", "fail")
    _write_junit(
        e2e_root,
        """
        <testcase classname="tests.test_e2e" name="test_e2e[encoder-base]">
          <skipped type="pytest.xfail" message="(known representation parity gap)" />
        </testcase>
        """,
    )

    results = mod._merge_pytest_outcomes(
        mod._load_results(artifacts_dir),
        mod._load_pytest_outcomes(e2e_root),
    )
    summary = mod.render_summary(
        results=results,
        mode="nightly",
        report_path=e2e_root / "missing.html",
        html_artifact_name="html",
        full_artifact_name="full",
        run_url="",
        max_rows=40,
    )

    assert "| skip | 1 |" in summary
    assert "### Failures" not in summary
    assert "| encoder-base | XFAIL | skip | known representation parity gap |" in summary


def test_summary_treats_model_owned_xfail_as_waived_not_active_failure(
    tmp_path: Path,
) -> None:
    mod = _import_summary()
    e2e_root = tmp_path / "e2e_artifacts"
    artifacts_dir = e2e_root / "artifacts"
    _write_result(artifacts_dir, "fnet-base", "fail")
    _write_junit(
        e2e_root,
        """
        <testcase classname="tests.e2e.models.fnet.test_fnet_e2e"
                  name="test_model_e2e[fnet-base]">
          <skipped type="pytest.xfail"
                   message="(encoder representation parity below minimum contract floor)" />
        </testcase>
        """,
    )

    results = mod._merge_pytest_outcomes(
        mod._load_results(artifacts_dir),
        mod._load_pytest_outcomes(e2e_root),
    )
    summary = mod.render_summary(
        results=results,
        mode="premerge",
        report_path=e2e_root / "missing.html",
        html_artifact_name="html",
        full_artifact_name="full",
        run_url="",
        max_rows=40,
    )

    assert "| skip | 1 |" in summary
    assert "### Failures" not in summary
    assert (
        "| fnet-base | XFAIL | skip | encoder representation parity below minimum contract floor |"
    ) in summary


def test_summary_groups_structured_model_testcases(tmp_path: Path) -> None:
    mod = _import_summary()
    e2e_root = tmp_path / "e2e_artifacts"
    artifacts_dir = e2e_root / "artifacts"
    _write_result(artifacts_dir, "canary-1b-v2", "pass")
    _write_result(
        artifacts_dir,
        "canary-1b-v2-asr-probe01",
        "pass",
        "canary-1b-v2",
    )
    _write_result(
        artifacts_dir,
        "canary-1b-v2-asr-probe02",
        "pass",
        "canary-1b-v2",
    )
    _write_junit(
        e2e_root,
        """
        <testcase classname="tests.e2e.models.canary.test_canary_e2e"
                  name="test_model_e2e[canary-1b-v2]" />
        """,
    )

    results = mod._merge_pytest_outcomes(
        mod._load_results(artifacts_dir),
        mod._load_pytest_outcomes(e2e_root),
    )
    summary = mod.render_summary(
        results=results,
        mode="premerge",
        report_path=e2e_root / "missing.html",
        html_artifact_name="html",
        full_artifact_name="full",
        run_url="",
        max_rows=40,
    )

    assert "| pass | 3 |" in summary
    assert "### Multi-Testcase Models" in summary
    assert "canary-1b-v2-asr-probe01" in summary
    assert "canary-1b-v2-asr-probe02" in summary
    assert (
        "| canary-1b-v2-asr-probe01 | example_decoder | "
        "text_generation_causal | pass | token_agreement_rate=1 | 3.0s |"
    ) in summary


def test_summary_surfaces_model_owned_xpass_from_console_logs(tmp_path: Path) -> None:
    mod = _import_summary()
    e2e_root = tmp_path / "e2e_artifacts"
    artifacts_dir = e2e_root / "artifacts"
    _write_result(artifacts_dir, "fnet-base", "pass")
    e2e_root.mkdir(parents=True, exist_ok=True)
    (e2e_root / "console-gpu2-shared-w3.log").write_text(
        "tests/e2e/models/fnet/test_fnet_e2e.py::test_model_e2e[fnet-base] "
        "XPASS ((FNet parity gap unexpectedly closed)) [100%]\n",
        encoding="utf-8",
    )

    results = mod._merge_pytest_outcomes(
        mod._load_results(artifacts_dir),
        mod._load_pytest_outcomes(e2e_root),
    )
    summary = mod.render_summary(
        results=results,
        mode="premerge",
        report_path=e2e_root / "missing.html",
        html_artifact_name="html",
        full_artifact_name="full",
        run_url="",
        max_rows=40,
    )

    assert "### Pytest Waive Outcomes" in summary
    assert ("| fnet-base | XPASS | pass | FNet parity gap unexpectedly closed |") in summary
