# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real family pytest/evidence/JUnit path without model inference."""

import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from tools.ci.e2e import _require_e2e_process_result
from tools.ci.process import CiError

from families.lfm2_audio.tests.test_e2e import MANIFEST

ROOT = Path(__file__).resolve().parents[3]
NAMES = tuple(case["name"] for case in MANIFEST["testcases"])


def run_routing(tmp_path, selectors, *, failed_case="", failure_stage=""):
    report = tmp_path / "junit.xml"
    calls = tmp_path / "calls.txt"
    artifacts = tmp_path / "artifacts"
    environment = os.environ | {
        "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "TRTMC_E2E_ARTIFACT_DIR": str(artifacts),
        "LFM_AUDIO_ROUTING_CALLS": str(calls),
        "LFM_AUDIO_ROUTING_FAILED_CASE": failed_case,
        "LFM_AUDIO_ROUTING_EXCEPTION": failure_stage,
    }
    environment.pop("TRTMC_E2E", None)
    environment.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "families/lfm2_audio/tests/test_e2e.py",
            "-p",
            "families.lfm2_audio.tests.fixtures.e2e_routing_plugin",
            "-p",
            "no:cacheprovider",
            "-q",
            "-o",
            "tmp_path_retention_policy=failed",
            "--junitxml",
            str(report),
            *selectors,
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result, report, calls, artifacts / "evidence/lfm2_audio"


def records(directory):
    return {
        path.parent.name: json.loads(path.read_text()) for path in directory.glob("*/evidence.json")
    }


@pytest.mark.parametrize("selected", [(NAMES[0],), (NAMES[2], NAMES[5]), NAMES])
def test_actual_pytest_routes_selected_results_and_evidence_once(tmp_path, selected):
    selectors = (
        ["--e2e-model", "lfm2_audio"]
        if selected == NAMES
        else ["--e2e-testcase", ",".join(selected)]
    )
    result, report, calls, directory = run_routing(tmp_path, selectors)
    assert result.returncode == 0, result.stdout + result.stderr
    _require_e2e_process_result(report, "lfm2_audio", selected, result.returncode)
    assert calls.read_text().splitlines() == ["executed"]
    evidence = records(directory)
    assert set(evidence) == set(selected)
    origin = selected[0]
    for name, record in evidence.items():
        assert record["status"] == "passed"
        assert record["evidence_status"] == "recorded", record["issues"]
        assert record["nodeid"].endswith(f"test_e2e[{name}]")
        assert record["native"]["comparison"]["name"] == name
        matrix = record["model_card_matrix"]
        assert matrix["executed_by_case"] == origin
        assert matrix["reused_execution"] == (name != origin)
        assert set(matrix["all_case_results"]) == set(NAMES)
        assert matrix["full_suite_passed"] and all(matrix["suite_coverage"].values())
        for role in ("native", "reference"):
            payload = json.loads(
                (directory / name / record[role]["records"]["artifact"]).read_text()
            )
            assert payload["name"] == name and payload["producer"] == "synthetic-" + role
        media = [
            item
            for item in record["artifacts"]
            if item["role"] == "native" and item["label"].endswith(".f32")
        ]
        assert all(item["label"].startswith(name + ".") for item in media)
        assert len(media) == (
            0 if MANIFEST["testcases"][NAMES.index(name)]["mode"] in {"asr", "mixed", "text"} else 4
        )
        stages = {item["stage"] for item in record["timing"]}
        assert ("build" in stages) == (name == origin)
        assert "semantic_validation" in stages
        assert ("matrix_execution" in record) == (name == origin)
        if name == origin:
            archive = record["matrix_execution"]
            assert archive["executed_by_case"] == origin
            case_file = directory / name / archive["cases"]["artifact"]
            assert {case["name"] for case in json.loads(case_file.read_text())} == set(NAMES)
            assert archive["waveforms"] and archive["asr_inputs"]
            assert all("artifact" in item for item in archive["waveforms"] + archive["asr_inputs"])


def test_unselected_case_failure_still_fails_the_full_suite(tmp_path):
    selected = (NAMES[0],)
    result, report, calls, directory = run_routing(
        tmp_path, ["--e2e-testcase", selected[0]], failed_case=NAMES[4]
    )
    assert result.returncode == 1
    with pytest.raises(CiError, match="requested E2E testcase failed"):
        _require_e2e_process_result(report, "lfm2_audio", selected, result.returncode)
    assert calls.read_text().splitlines() == ["executed"]
    record = records(directory)[selected[0]]
    assert record["native"]["comparison"]["passed"]
    assert not record["model_card_matrix"]["all_case_results"][NAMES[4]]
    assert not record["model_card_matrix"]["full_suite_passed"]
    assert record["status"] == "failed" and record["failure_stage"] == "semantic_validation"


@pytest.mark.parametrize("stage", ["build", "reference", "native", "public_api", "speech_content"])
def test_execution_failure_keeps_stage_and_is_not_reexecuted(tmp_path, stage):
    selected = (NAMES[1], NAMES[3])
    result, report, calls, directory = run_routing(
        tmp_path, ["--e2e-testcase", ",".join(selected)], failure_stage=stage
    )
    assert result.returncode == 1
    with pytest.raises(CiError, match="requested E2E testcase failed"):
        _require_e2e_process_result(report, "lfm2_audio", selected, result.returncode)
    assert calls.read_text().splitlines() == ["executed"]
    evidence = records(directory)
    assert set(evidence) == set(selected)
    assert evidence[selected[0]]["failure_stage"] == stage
    assert evidence[selected[1]]["failure_stage"] == "shared_matrix_failure"
    assert all(record["status"] == "failed" for record in evidence.values())


@pytest.mark.parametrize("selectors", [[], ["--e2e-model", "other"], ["--e2e-testcase", "unknown"]])
def test_unselected_nodes_do_not_execute_or_emit_success(tmp_path, selectors):
    result, report, calls, directory = run_routing(tmp_path, selectors)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not calls.exists()
    assert records(directory) == {}
    cases = ET.parse(report).findall(".//testcase")
    assert len(cases) == len(NAMES) and all(case.find("skipped") is not None for case in cases)
    with pytest.raises(CiError, match="requested E2E testcase skipped"):
        _require_e2e_process_result(report, "lfm2_audio", (NAMES[0],), result.returncode)
