from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import e2e_origin_main_parity
from tools.e2e_origin_main_parity import compare_results, result_signature


def _result(
    *,
    status: str = "pass",
    metric_passed: bool = True,
    artifact_path: str = "/tmp/current/qwen/output.txt",
) -> dict:
    return {
        "case_name": "qwen3-0.6b-fp16",
        "status": status,
        "failure_type": None,
        "oracle_level": "L1_external_reference",
        "stages": {
            "generate": {
                "status": "passed" if metric_passed else "failed",
                "metrics": {
                    "token_match": {
                        "value": 1.0 if metric_passed else 0.0,
                        "threshold": 1.0,
                        "operator": ">=",
                        "passed": metric_passed,
                    }
                },
            }
        },
        "artifacts": {"generate": {"text": artifact_path}},
    }


def test_result_signature_ignores_artifact_directory_prefix() -> None:
    current = _result(artifact_path="/tmp/current/qwen/output.txt")
    baseline = _result(artifact_path="/tmp/origin-main/qwen/output.txt")

    assert result_signature(current) == result_signature(baseline)


def test_compare_results_accepts_same_contract_result() -> None:
    assert compare_results(_result(), _result(artifact_path="/baseline/output.txt")) == []


def test_compare_results_reports_status_mismatch() -> None:
    errors = compare_results(_result(status="pass"), _result(status="fail"))

    assert any("status differs" in error for error in errors)


def test_compare_results_reports_metric_gate_mismatch() -> None:
    errors = compare_results(_result(metric_passed=True), _result(metric_passed=False))

    assert any("stage comparator results differ" in error for error in errors)


def test_compare_cli_returns_success_for_matching_results(tmp_path: Path) -> None:
    from tools.e2e_origin_main_parity import main

    current = tmp_path / "current.json"
    baseline = tmp_path / "baseline.json"
    current.write_text(json.dumps(_result()), encoding="utf-8")
    baseline.write_text(json.dumps(_result(artifact_path="/baseline/output.txt")), encoding="utf-8")

    assert main([
        "compare",
        "--current-result",
        str(current),
        "--baseline-result",
        str(baseline),
    ]) == 0


def test_run_cli_requires_model_plugin_dir() -> None:
    with pytest.raises(SystemExit):
        e2e_origin_main_parity.main([
            "run",
            "--model",
            "qwen3-0.6b-fp16",
            "--origin-main-dir",
            "/tmp/origin-main",
            "--current-trtmc-binary",
            "/tmp/current/trtmc",
            "--baseline-trtmc-binary",
            "/tmp/baseline/trtmc",
            "--engine-dir",
            "/tmp/current-engines",
            "--baseline-engine-dir",
            "/tmp/baseline-engines",
            "--current-artifacts-dir",
            "/tmp/current-artifacts",
            "--baseline-artifacts-dir",
            "/tmp/baseline-artifacts",
        ])


def test_run_cli_uses_isolated_current_and_origin_main_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], Path]] = []

    def fake_run_pytest(cmd: list[str], cwd: Path) -> int:
        calls.append((cmd, cwd))
        artifacts_dir = Path(cmd[cmd.index("--e2e-artifacts-dir") + 1])
        node_id = next(arg for arg in cmd if "::test_" in arg)
        model = node_id.rsplit("[", 1)[1].rstrip("]")
        result_dir = artifacts_dir / model
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "result.json").write_text(
            json.dumps(_result(artifact_path=str(result_dir / "output.txt"))),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(e2e_origin_main_parity, "_run_pytest", fake_run_pytest)

    repo_root = Path(__file__).resolve().parents[2]
    origin_main = tmp_path / "origin-main"
    current_artifacts = tmp_path / "current-artifacts"
    baseline_artifacts = tmp_path / "baseline-artifacts"
    plugin_dir = tmp_path / "only-text-generation"

    assert e2e_origin_main_parity.main([
        "run",
        "--model",
        "qwen3-0.6b-fp16",
        "--repo-root",
        str(repo_root),
        "--origin-main-dir",
        str(origin_main),
        "--current-trtmc-binary",
        str(tmp_path / "current" / "trtmc"),
        "--baseline-trtmc-binary",
        str(tmp_path / "baseline" / "trtmc"),
        "--model-plugin-dir",
        str(plugin_dir),
        "--engine-dir",
        str(tmp_path / "current-engines"),
        "--baseline-engine-dir",
        str(tmp_path / "baseline-engines"),
        "--current-artifacts-dir",
        str(current_artifacts),
        "--baseline-artifacts-dir",
        str(baseline_artifacts),
        "--hf-python",
        "/opt/venv/bin/python",
    ]) == 0

    assert len(calls) == 2
    current_cmd, current_cwd = calls[0]
    baseline_cmd, baseline_cwd = calls[1]

    assert current_cwd == repo_root
    assert baseline_cwd == origin_main
    assert any(
        "tests/e2e/models/qwen/test_qwen_e2e.py::test_model_e2e[qwen3-0.6b-fp16]"
        in arg
        for arg in current_cmd
    )
    assert "--model-plugin-dir" in current_cmd
    assert current_cmd[current_cmd.index("--model-plugin-dir") + 1] == str(plugin_dir)
    assert any(
        "tests/test_e2e.py::test_e2e[qwen3-0.6b-fp16]" in arg
        for arg in baseline_cmd
    )
    assert "--model-plugin-dir" not in baseline_cmd


def test_run_cli_passes_multi_device_selection_for_tp_cases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], Path]] = []

    def fake_run_pytest(cmd: list[str], cwd: Path) -> int:
        calls.append((cmd, cwd))
        artifacts_dir = Path(cmd[cmd.index("--e2e-artifacts-dir") + 1])
        node_id = next(arg for arg in cmd if "::test_" in arg)
        model = node_id.rsplit("[", 1)[1].rstrip("]")
        result_dir = artifacts_dir / model
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "result.json").write_text(
            json.dumps(_result(artifact_path=str(result_dir / "output.txt"))),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(e2e_origin_main_parity, "_run_pytest", fake_run_pytest)

    repo_root = Path(__file__).resolve().parents[2]
    origin_main = tmp_path / "origin-main"

    assert e2e_origin_main_parity.main([
        "run",
        "--model",
        "convbert-base-tp2",
        "--repo-root",
        str(repo_root),
        "--origin-main-dir",
        str(origin_main),
        "--current-trtmc-binary",
        str(tmp_path / "current" / "trtmc"),
        "--baseline-trtmc-binary",
        str(tmp_path / "baseline" / "trtmc"),
        "--model-plugin-dir",
        str(tmp_path / "only-encoder"),
        "--engine-dir",
        str(tmp_path / "current-engines"),
        "--baseline-engine-dir",
        str(tmp_path / "baseline-engines"),
        "--current-artifacts-dir",
        str(tmp_path / "current-artifacts"),
        "--baseline-artifacts-dir",
        str(tmp_path / "baseline-artifacts"),
    ]) == 0

    assert len(calls) == 2
    assert "--multi-device-only" in calls[0][0]
    assert "--multi-device-only" in calls[1][0]


def test_run_cli_can_set_baseline_pythonpath_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], Path, dict[str, str] | None]] = []

    def fake_run_pytest(
        cmd: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> int:
        calls.append((cmd, cwd, env))
        artifacts_dir = Path(cmd[cmd.index("--e2e-artifacts-dir") + 1])
        node_id = next(arg for arg in cmd if "::test_" in arg)
        model = node_id.rsplit("[", 1)[1].rstrip("]")
        result_dir = artifacts_dir / model
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "result.json").write_text(
            json.dumps(_result(artifact_path=str(result_dir / "output.txt"))),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(e2e_origin_main_parity, "_run_pytest", fake_run_pytest)

    repo_root = Path(__file__).resolve().parents[2]
    origin_main = tmp_path / "origin-main"
    baseline_pythonpath = origin_main / "python"

    assert e2e_origin_main_parity.main([
        "run",
        "--model",
        "qwen3-0.6b-fp16",
        "--repo-root",
        str(repo_root),
        "--origin-main-dir",
        str(origin_main),
        "--current-trtmc-binary",
        str(tmp_path / "current" / "trtmc"),
        "--baseline-trtmc-binary",
        str(tmp_path / "baseline" / "trtmc"),
        "--model-plugin-dir",
        str(tmp_path / "only-text-generation"),
        "--engine-dir",
        str(tmp_path / "current-engines"),
        "--baseline-engine-dir",
        str(tmp_path / "baseline-engines"),
        "--current-artifacts-dir",
        str(tmp_path / "current-artifacts"),
        "--baseline-artifacts-dir",
        str(tmp_path / "baseline-artifacts"),
        "--baseline-pythonpath",
        str(baseline_pythonpath),
    ]) == 0

    assert calls[0][2] is None
    assert calls[1][2] is not None
    assert calls[1][2]["PYTHONPATH"].split(":", 1)[0] == str(
        baseline_pythonpath.resolve()
    )


def test_run_cli_accepts_matching_pytest_level_skips_without_result_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], Path]] = []

    def fake_run_pytest(cmd: list[str], cwd: Path) -> int:
        calls.append((cmd, cwd))
        return 0

    monkeypatch.setattr(e2e_origin_main_parity, "_run_pytest", fake_run_pytest)

    repo_root = Path(__file__).resolve().parents[2]
    origin_main = tmp_path / "origin-main"
    current_artifacts = tmp_path / "current-artifacts"
    baseline_artifacts = tmp_path / "baseline-artifacts"

    assert e2e_origin_main_parity.main([
        "run",
        "--model",
        "qwen3-0.6b-fp16",
        "--repo-root",
        str(repo_root),
        "--origin-main-dir",
        str(origin_main),
        "--current-trtmc-binary",
        str(tmp_path / "current" / "trtmc"),
        "--baseline-trtmc-binary",
        str(tmp_path / "baseline" / "trtmc"),
        "--model-plugin-dir",
        str(tmp_path / "only-text-generation"),
        "--engine-dir",
        str(tmp_path / "current-engines"),
        "--baseline-engine-dir",
        str(tmp_path / "baseline-engines"),
        "--current-artifacts-dir",
        str(current_artifacts),
        "--baseline-artifacts-dir",
        str(baseline_artifacts),
    ]) == 0

    assert len(calls) == 2
    current_result = json.loads(
        (current_artifacts / "qwen3-0.6b-fp16" / "result.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_result = json.loads(
        (baseline_artifacts / "qwen3-0.6b-fp16" / "result.json").read_text(
            encoding="utf-8"
        )
    )
    assert current_result == baseline_result
    assert current_result["status"] == "skip"
    assert current_result["failure_type"] == "pytest_skip"


def test_batch_cli_prepares_isolated_plugin_dirs_and_writes_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], Path]] = []

    def fake_run_pytest(cmd: list[str], cwd: Path) -> int:
        calls.append((cmd, cwd))
        artifacts_dir = Path(cmd[cmd.index("--e2e-artifacts-dir") + 1])
        node_id = next(arg for arg in cmd if "::test_" in arg)
        model = node_id.rsplit("[", 1)[1].rstrip("]")
        result_dir = artifacts_dir / model
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "result.json").write_text(
            json.dumps(_result(artifact_path=str(result_dir / "output.txt"))),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(e2e_origin_main_parity, "_run_pytest", fake_run_pytest)

    repo_root = Path(__file__).resolve().parents[2]
    build_dir = tmp_path / "build"
    plugin_lib = (
        build_dir
        / "models"
        / "text_generation"
        / "libtrtmc_model_text_generation.so"
    )
    plugin_lib.parent.mkdir(parents=True)
    plugin_lib.write_bytes(b"fake-so")

    plugin_work_dir = tmp_path / "plugins"
    summary_json = tmp_path / "summary.json"
    origin_main = tmp_path / "origin-main"

    assert e2e_origin_main_parity.main([
        "batch",
        "--model",
        "qwen3-0.6b-fp16",
        "--repo-root",
        str(repo_root),
        "--origin-main-dir",
        str(origin_main),
        "--current-trtmc-binary",
        str(tmp_path / "current" / "trtmc"),
        "--baseline-trtmc-binary",
        str(tmp_path / "baseline" / "trtmc"),
        "--current-build-dir",
        str(build_dir),
        "--model-plugin-work-dir",
        str(plugin_work_dir),
        "--engine-dir",
        str(tmp_path / "current-engines"),
        "--baseline-engine-dir",
        str(tmp_path / "baseline-engines"),
        "--current-artifacts-dir",
        str(tmp_path / "current-artifacts"),
        "--baseline-artifacts-dir",
        str(tmp_path / "baseline-artifacts"),
        "--summary-json",
        str(summary_json),
        "--hf-python",
        "/opt/venv/bin/python",
    ]) == 0

    isolated_lib = (
        plugin_work_dir
        / "qwen3-0.6b-fp16"
        / "text_generation"
        / "libtrtmc_model_text_generation.so"
    )
    assert isolated_lib.read_bytes() == b"fake-so"

    assert len(calls) == 2
    current_cmd, _current_cwd = calls[0]
    baseline_cmd, baseline_cwd = calls[1]
    assert "--model-plugin-dir" in current_cmd
    assert current_cmd[current_cmd.index("--model-plugin-dir") + 1] == str(
        plugin_work_dir / "qwen3-0.6b-fp16"
    )
    assert baseline_cwd == origin_main
    assert "--model-plugin-dir" not in baseline_cmd

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["total"] == 1
    assert summary["passed"] == 1
    assert summary["failed"] == 0
    assert summary["models"][0]["model"] == "qwen3-0.6b-fp16"
    assert summary["models"][0]["status"] == "pass"

    stale_file = plugin_work_dir / "qwen3-0.6b-fp16" / "stale.txt"
    stale_file.write_text("stale", encoding="utf-8")

    assert e2e_origin_main_parity.main([
        "batch",
        "--model",
        "qwen3-0.6b-fp16",
        "--repo-root",
        str(repo_root),
        "--origin-main-dir",
        str(origin_main),
        "--current-trtmc-binary",
        str(tmp_path / "current" / "trtmc"),
        "--baseline-trtmc-binary",
        str(tmp_path / "baseline" / "trtmc"),
        "--current-build-dir",
        str(build_dir),
        "--model-plugin-work-dir",
        str(plugin_work_dir),
        "--engine-dir",
        str(tmp_path / "current-engines"),
        "--baseline-engine-dir",
        str(tmp_path / "baseline-engines"),
        "--current-artifacts-dir",
        str(tmp_path / "current-artifacts-rerun"),
        "--baseline-artifacts-dir",
        str(tmp_path / "baseline-artifacts-rerun"),
        "--clean-model-plugin-dir",
    ]) == 0

    assert not stale_file.exists()
    assert isolated_lib.read_bytes() == b"fake-so"


def test_plan_cli_reports_ready_models_and_writes_ready_file(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    build_dir = tmp_path / "build"
    current_engines = tmp_path / "current-engines"
    baseline_engines = tmp_path / "baseline-engines"
    output_json = tmp_path / "plan.json"
    ready_models = tmp_path / "ready-models.txt"

    plugin_lib = (
        build_dir
        / "models"
        / "text_generation"
        / "libtrtmc_model_text_generation.so"
    )
    plugin_lib.parent.mkdir(parents=True)
    plugin_lib.write_bytes(b"fake-so")
    current_engines.mkdir()
    baseline_engines.mkdir()
    (current_engines / "qwen3-0.6b-fp16.trtfb").write_bytes(b"current")
    (baseline_engines / "qwen3-0.6b-fp16.trtfb").write_bytes(b"baseline")

    assert e2e_origin_main_parity.main([
        "plan",
        "--model",
        "qwen3-0.6b-fp16",
        "--repo-root",
        str(repo_root),
        "--current-build-dir",
        str(build_dir),
        "--engine-dir",
        str(current_engines),
        "--baseline-engine-dir",
        str(baseline_engines),
        "--output-json",
        str(output_json),
        "--ready-models-file",
        str(ready_models),
    ]) == 0

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["total"] == 1
    assert report["ready"] == 1
    assert report["not_ready"] == 0
    assert report["ready_models"] == ["qwen3-0.6b-fp16"]
    assert ready_models.read_text(encoding="utf-8").splitlines() == [
        "qwen3-0.6b-fp16"
    ]

    entry = report["models"][0]
    assert entry["model"] == "qwen3-0.6b-fp16"
    assert entry["bundle"] == "qwen3-0.6b-fp16.trtfb"
    assert entry["current_bundle_exists"] is True
    assert entry["baseline_bundle_exists"] is True
    assert entry["ready"] is True
    assert entry["plugins"] == [{
        "model_id": "text_generation",
        "target": "trtmc_model_text_generation",
        "library": "libtrtmc_model_text_generation.so",
        "library_path": str(plugin_lib),
        "library_exists": True,
    }]


def test_plan_cli_fails_when_requested_and_baseline_bundle_is_missing(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    build_dir = tmp_path / "build"
    current_engines = tmp_path / "current-engines"
    baseline_engines = tmp_path / "baseline-engines"
    output_json = tmp_path / "plan.json"

    plugin_lib = (
        build_dir
        / "models"
        / "text_generation"
        / "libtrtmc_model_text_generation.so"
    )
    plugin_lib.parent.mkdir(parents=True)
    plugin_lib.write_bytes(b"fake-so")
    current_engines.mkdir()
    baseline_engines.mkdir()
    (current_engines / "qwen3-0.6b-fp16.trtfb").write_bytes(b"current")

    assert e2e_origin_main_parity.main([
        "plan",
        "--model",
        "qwen3-0.6b-fp16",
        "--repo-root",
        str(repo_root),
        "--current-build-dir",
        str(build_dir),
        "--engine-dir",
        str(current_engines),
        "--baseline-engine-dir",
        str(baseline_engines),
        "--output-json",
        str(output_json),
        "--fail-if-not-ready",
    ]) == 1

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["ready"] == 0
    assert report["not_ready"] == 1
    assert report["ready_models"] == []
    assert report["models"][0]["baseline_bundle_exists"] is False
    assert any(
        "origin/main model bundle not found" in error
        for error in report["models"][0]["errors"]
    )
