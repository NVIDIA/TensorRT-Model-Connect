"""Focused tests for E2EOrchestrator lifecycle phase outcomes."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.e2e_harness import orchestrator
from tests.e2e_harness.contracts import (
    CompareResult,
    E2ECase,
    E2EStatus,
    FailureType,
    MetricResult,
    PreflightRequirement,
    RunContext,
    StageOutput,
    StageSpec,
    StageStatus,
)
from tests.e2e_harness.orchestrator import E2EOrchestrator


class _FakeRunner:
    strategy_name = "unit_task"

    def __init__(
        self,
        outputs: list[StageOutput] | None = None,
        error: str | None = None,
    ) -> None:
        self._outputs = outputs or []
        self._error = error
        self.calls = 0

    def run_stage(
        self,
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
    ) -> StageOutput:
        self.calls += 1
        if self._error is not None:
            raise RuntimeError(self._error)
        if self._outputs:
            index = min(self.calls - 1, len(self._outputs) - 1)
            return self._outputs[index]
        return StageOutput(stage_name=stage.name, text="trt", data={"value": 1})


class _FakeReference:
    backend_name = "unit_ref"

    def __init__(self, error: str | None = None) -> None:
        self._error = error
        self.calls = 0

    def run_stage(
        self,
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
    ) -> StageOutput:
        self.calls += 1
        if self._error is not None:
            raise RuntimeError(self._error)
        return StageOutput(stage_name=stage.name, text="ref", data={"value": 1})


class _FakeComparator:
    task_strategy = "unit_task"

    def __init__(
        self,
        result: CompareResult | None = None,
        error: str | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls = 0

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: Any,
        stage: StageSpec,
    ) -> CompareResult:
        self.calls += 1
        if self._error is not None:
            raise RuntimeError(self._error)
        if self._result is not None:
            return self._result
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value,
            metrics={
                "value_match": MetricResult(
                    value=1.0,
                    threshold=1.0,
                    operator="==",
                    passed=True,
                ),
            },
            message="matched",
        )


@pytest.fixture(autouse=True)
def _stable_env_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tests.e2e_harness.artifact_sink._collect_env_fingerprint",
        lambda ctx=None: {"unit": "env"},
    )


def _make_case(
    name: str,
    *,
    preflight: list[PreflightRequirement] | None = None,
    determinism: dict[str, Any] | None = None,
) -> E2ECase:
    return E2ECase(
        name=name,
        hf_id="hf/unit-model",
        family="unit",
        runtime_strategy="unit_runtime",
        task_strategy="unit_task",
        reference_backend="unit_ref",
        bundle=f"{name}.trtfb",
        preflight=preflight or [],
        stages=[StageSpec(name="generate")],
        determinism=determinism or {},
    )


def _make_ctx(tmp_path: Path, case: E2ECase) -> RunContext:
    engine_dir = tmp_path / "engines"
    engine_dir.mkdir()
    return RunContext(
        case=case,
        artifacts_dir=str(tmp_path / "artifacts"),
        binary_path="/tmp/trtmc",
        hf_python=sys.executable,
        engine_dir=str(engine_dir),
    )


def _patch_bundle_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / "engines" / "unit.trtfb"
    bundle_path.parent.mkdir(exist_ok=True)
    bundle_path.write_text("bundle", encoding="utf-8")
    monkeypatch.setattr(
        orchestrator,
        "_resolve_bundle",
        lambda case, ctx: (str(bundle_path), None, "", {}),
    )


def _patch_plugins(
    monkeypatch: pytest.MonkeyPatch,
    *,
    runner: _FakeRunner | None = None,
    reference: _FakeReference | None = None,
    comparator: _FakeComparator | None = None,
) -> None:
    monkeypatch.setattr(orchestrator, "get_runner", lambda name: runner)
    monkeypatch.setattr(orchestrator, "get_reference", lambda name: reference)
    monkeypatch.setattr(orchestrator, "get_comparator", lambda name: comparator)
    monkeypatch.setattr(orchestrator, "get_contract_plugin", lambda name: None)


def _read_result_json(ctx: RunContext, case: E2ECase) -> dict[str, Any]:
    path = Path(ctx.artifacts_dir) / case.name / "result.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_run_returns_preflight_skip_without_resolving_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case(
        "preflight-skip",
        preflight=[PreflightRequirement(kind="unknown_preflight")],
    )
    ctx = _make_ctx(tmp_path, case)
    monkeypatch.setattr(
        orchestrator,
        "_resolve_bundle",
        lambda case, ctx: pytest.fail("bundle resolution should not run"),
    )

    result = E2EOrchestrator().run(case, ctx)

    assert result.status == E2EStatus.SKIP.value
    assert result.failure_type == FailureType.PRECHECK_FAIL.value
    assert result.stages == {}
    assert result.determinism["preflight"][0]["kind"] == "unknown_preflight"
    data = _read_result_json(ctx, case)
    assert data["status"] == E2EStatus.SKIP.value
    assert data["failure_type"] == FailureType.PRECHECK_FAIL.value
    assert data["stages"] == {}
    assert data["artifacts"]["preflight_details"] == "preflight_details.json"
    preflight_details = json.loads(
        (Path(ctx.artifacts_dir) / case.name / "preflight_details.json").read_text(
            encoding="utf-8"
        )
    )
    assert preflight_details[0]["kind"] == "unknown_preflight"


def test_sana_wm_preflight_finds_official_script_and_assets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    binary = tmp_path / "trtmc"
    binary.write_text("", encoding="utf-8")
    image = tmp_path / "demo_0.png"
    image.write_text("image", encoding="utf-8")
    prompt = tmp_path / "demo_0.txt"
    prompt.write_text("prompt", encoding="utf-8")
    script = tmp_path / "inference_sana_wm.py"
    script.write_text("", encoding="utf-8")
    monkeypatch.setenv("SANA_WM_SCRIPT", str(script))

    case = _make_case(
        "sana-wm-preflight",
        preflight=[
            PreflightRequirement(kind="binary_exists"),
            PreflightRequirement(kind="asset_exists", args={"path": str(image)}),
            PreflightRequirement(kind="asset_exists", args={"path": str(prompt)}),
            PreflightRequirement(kind="sana_wm_script_available"),
            PreflightRequirement(kind="sana_wm_runtime_entrypoint_available"),
        ],
    )
    ctx = _make_ctx(tmp_path, case)
    ctx.binary_path = str(binary)

    ok, details = orchestrator.run_preflight(case, ctx)

    assert ok is True
    assert all(item["passed"] for item in details)
    assert details[-2]["kind"] == "sana_wm_script_available"
    assert details[-1]["kind"] == "sana_wm_runtime_entrypoint_available"


def test_sana_wm_runtime_entrypoint_rejects_local_shim_without_model_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("SANA_WM_SCRIPT", raising=False)
    monkeypatch.delenv("SANA_REPO", raising=False)

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        assert "model_index.json" in command[-1]
        return orchestrator.subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="missing model_index.json",
        )

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)
    case = _make_case(
        "sana-wm-no-entrypoint",
        preflight=[
            PreflightRequirement(
                kind="sana_wm_runtime_entrypoint_available",
                args={
                    "hf_id": "Efficient-Large-Model/SANA-WM_bidirectional",
                    "path": "inference_video_scripts/inference_sana_wm.py",
                },
            ),
        ],
    )
    ctx = _make_ctx(tmp_path, case)

    ok, details = orchestrator.run_preflight(case, ctx)

    assert ok is False
    assert details[0]["kind"] == "sana_wm_runtime_entrypoint_available"
    assert details[0]["passed"] is False
    assert "runtime entrypoint unavailable" in details[0]["message"]
    assert "model_index.json" in details[0]["message"]
    assert "SANA_WM_SCRIPT/SANA_REPO" in details[0]["message"]


def test_sana_wm_runtime_entrypoint_rejects_unloadable_official_script(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sana_repo = tmp_path / "Sana"
    scripts_dir = sana_repo / "inference_video_scripts"
    scripts_dir.mkdir(parents=True)
    script = scripts_dir / "inference_sana_wm.py"
    script.write_text("import definitely_missing_sana_wm_dependency\n", encoding="utf-8")
    monkeypatch.setenv("SANA_REPO", str(sana_repo))
    monkeypatch.delenv("SANA_WM_SCRIPT", raising=False)

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        if str(script) in command:
            return orchestrator.subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="ModuleNotFoundError: No module named 'definitely_missing_sana_wm_dependency'",
            )
        assert "model_index.json" in command[-1]
        return orchestrator.subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="missing model_index.json",
        )

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)
    case = _make_case(
        "sana-wm-unloadable-entrypoint",
        preflight=[PreflightRequirement(kind="sana_wm_runtime_entrypoint_available")],
    )
    ctx = _make_ctx(tmp_path, case)

    ok, details = orchestrator.run_preflight(case, ctx)

    assert ok is False
    assert details[0]["passed"] is False
    assert "script load failures" in details[0]["message"]
    assert "definitely_missing_sana_wm_dependency" in details[0]["message"]


def test_sana_wm_runtime_entrypoint_allows_action_capable_diffusers_model_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("SANA_WM_SCRIPT", raising=False)
    monkeypatch.delenv("SANA_REPO", raising=False)

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        assert "model_index.json" in command[-1]
        return orchestrator.subprocess.CompletedProcess(
            command,
            0,
            stdout="pipeline_class=ActionCapableSanaWmPipeline\n",
            stderr="",
        )

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)
    case = _make_case(
        "sana-wm-diffusers-entrypoint",
        preflight=[
            PreflightRequirement(
                kind="sana_wm_script_available",
                args={"path": "inference_video_scripts/inference_sana_wm.py"},
                gating=False,
            ),
            PreflightRequirement(
                kind="sana_wm_runtime_entrypoint_available",
                args={"hf_id": "Efficient-Large-Model/SANA-WM_bidirectional"},
            ),
        ],
    )
    ctx = _make_ctx(tmp_path, case)

    ok, details = orchestrator.run_preflight(case, ctx)

    assert ok is True
    assert details[0]["kind"] == "sana_wm_script_available"
    assert details[0]["passed"] is False
    assert details[0]["gating"] is False
    assert details[1]["kind"] == "sana_wm_runtime_entrypoint_available"
    assert details[1]["passed"] is True
    assert "action-capable Diffusers entrypoint" in details[1]["message"]


def test_sana_wm_script_available_rejects_repo_local_shim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("SANA_WM_SCRIPT", raising=False)
    monkeypatch.delenv("SANA_REPO", raising=False)
    case = _make_case(
        "sana-wm-local-shim",
        preflight=[
            PreflightRequirement(
                kind="sana_wm_script_available",
                args={"path": "inference_video_scripts/inference_sana_wm.py"},
                gating=False,
            ),
        ],
    )
    ctx = _make_ctx(tmp_path, case)

    ok, details = orchestrator.run_preflight(case, ctx)

    assert ok is True
    assert details[0]["kind"] == "sana_wm_script_available"
    assert details[0]["passed"] is False
    assert details[0]["gating"] is False
    assert "compatibility shim is not sufficient" in details[0]["message"]


def test_sana_wm_preflight_skip_keeps_model_card_reference_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = E2ECase(
        name="sana-wm-bidirectional",
        hf_id="Efficient-Large-Model/SANA-WM_bidirectional",
        family="sana_wm",
        runtime_strategy="diffusion_sana_wm",
        task_strategy="diffusion_media_generation",
        reference_backend="hf_diffusers",
        bundle="sana-wm-bidirectional.trtfb",
        preflight=[PreflightRequirement(kind="unknown_preflight")],
        inputs={
            "prompt_file": "asset/sana_wm/demo_0.txt",
            "image": "asset/sana_wm/demo_0.png",
            "action": "w-80,jw-40,w-40,lw-60,w-100",
            "translation_speed": 0.055,
            "rotation_speed_deg": 1.2,
            "video_num_frames": 321,
        },
        stages=[StageSpec(name="end_to_end")],
    )
    ctx = _make_ctx(tmp_path, case)
    monkeypatch.setattr(
        orchestrator,
        "_resolve_bundle",
        lambda case, ctx: pytest.fail("bundle resolution should not run"),
    )

    result = E2EOrchestrator().run(case, ctx)

    expected = (
        f"{sys.executable} inference_video_scripts/inference_sana_wm.py "
        "--image asset/sana_wm/demo_0.png "
        "--prompt asset/sana_wm/demo_0.txt "
        '--action "w-80,jw-40,w-40,lw-60,w-100" '
        "--translation_speed 0.055 "
        "--rotation_speed_deg 1.2 "
        "--num_frames 321 "
        "--output_dir results/demo"
    )
    assert result.status == E2EStatus.SKIP.value
    assert result.repro_commands["sana_wm_python_reference"] == expected
    data = _read_result_json(ctx, case)
    assert data["repro_commands"]["sana_wm_python_reference"] == expected
    assert data["artifacts"]["preflight_details"] == "preflight_details.json"
    preflight_details = json.loads(
        (Path(ctx.artifacts_dir) / case.name / "preflight_details.json").read_text(
            encoding="utf-8"
        )
    )
    assert preflight_details[0]["kind"] == "unknown_preflight"


def test_run_returns_build_failure_and_logs_build_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case("build-failure")
    ctx = _make_ctx(tmp_path, case)
    build_info = {
        "command": [sys.executable, "-m", "builder"],
        "returncode": 2,
        "stdout": "",
        "stderr": "build stderr",
    }
    monkeypatch.setattr(
        orchestrator,
        "_resolve_bundle",
        lambda case, ctx: (None, 0.25, "build exploded", build_info),
    )

    result = E2EOrchestrator().run(case, ctx)

    assert result.status == E2EStatus.FAIL.value
    assert result.failure_type == FailureType.BUILD_FAIL.value
    assert result.determinism == {"build_error": "build exploded"}
    data = _read_result_json(ctx, case)
    assert data["failure_type"] == FailureType.BUILD_FAIL.value
    assert data["commands"][0]["command"] == build_info["command"]
    assert data["timing"]["bundle_build_s"] == 0.25
    assert "repro_commands" in data


def test_run_classifies_trt_stage_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case("trt-error")
    ctx = _make_ctx(tmp_path, case)
    _patch_bundle_success(monkeypatch, tmp_path)
    _patch_plugins(
        monkeypatch,
        runner=_FakeRunner(error="trt boom"),
        reference=_FakeReference(),
        comparator=_FakeComparator(),
    )

    result = E2EOrchestrator().run(case, ctx)

    assert result.status == E2EStatus.FAIL.value
    assert result.failure_type == FailureType.TRT_RUN_FAIL.value
    assert result.stages["generate"].status == StageStatus.ERROR.value
    assert "TRT run failed: trt boom" in result.stages["generate"].message
    data = _read_result_json(ctx, case)
    assert data["stages"]["generate"]["status"] == StageStatus.ERROR.value


def test_run_classifies_reference_stage_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case("reference-error")
    ctx = _make_ctx(tmp_path, case)
    _patch_bundle_success(monkeypatch, tmp_path)
    _patch_plugins(
        monkeypatch,
        runner=_FakeRunner(),
        reference=_FakeReference(error="reference boom"),
        comparator=_FakeComparator(),
    )

    result = E2EOrchestrator().run(case, ctx)

    assert result.status == E2EStatus.FAIL.value
    assert result.failure_type == FailureType.REFERENCE_RUN_FAIL.value
    assert result.stages["generate"].status == StageStatus.ERROR.value
    assert "Reference run failed: reference boom" in result.stages["generate"].message
    data = _read_result_json(ctx, case)
    assert data["failure_type"] == FailureType.REFERENCE_RUN_FAIL.value


def test_run_classifies_comparison_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case("comparison-failure")
    ctx = _make_ctx(tmp_path, case)
    _patch_bundle_success(monkeypatch, tmp_path)
    failed_compare = CompareResult(
        stage_name="generate",
        status=StageStatus.FAILED.value,
        metrics={"value_match": MetricResult(value=0.0, threshold=1.0, passed=False)},
        message="mismatch",
    )
    _patch_plugins(
        monkeypatch,
        runner=_FakeRunner(),
        reference=_FakeReference(),
        comparator=_FakeComparator(result=failed_compare),
    )

    result = E2EOrchestrator().run(case, ctx)

    assert result.status == E2EStatus.FAIL.value
    assert result.failure_type == FailureType.COMPARE_FAIL.value
    assert result.stages["generate"].status == StageStatus.FAILED.value
    data = _read_result_json(ctx, case)
    assert data["stages"]["generate"]["metrics"]["value_match"]["passed"] is False


def test_run_classifies_determinism_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case("determinism-failure", determinism={"reruns": 1})
    ctx = _make_ctx(tmp_path, case)
    _patch_bundle_success(monkeypatch, tmp_path)
    runner = _FakeRunner(
        outputs=[
            StageOutput(stage_name="generate", text="baseline", data={"value": 1}),
            StageOutput(stage_name="generate", text="rerun drift", data={"value": 1}),
        ]
    )
    _patch_plugins(
        monkeypatch,
        runner=runner,
        reference=_FakeReference(),
        comparator=_FakeComparator(),
    )

    result = E2EOrchestrator().run(case, ctx)

    assert result.status == E2EStatus.FAIL.value
    assert result.failure_type == FailureType.DETERMINISM_FAIL.value
    assert result.stages["generate"].status == StageStatus.PASSED.value
    assert result.determinism["status"] == "non_deterministic"
    assert result.determinism["per_stage"]["generate"][0]["text_match"] is False
    data = _read_result_json(ctx, case)
    assert data["failure_type"] == FailureType.DETERMINISM_FAIL.value
    assert data["determinism"]["status"] == "non_deterministic"
