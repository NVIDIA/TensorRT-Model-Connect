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
    OracleLevel,
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
    oracle_level: str = OracleLevel.L1_EXTERNAL_REFERENCE.value,
    metadata: dict[str, Any] | None = None,
) -> E2ECase:
    return E2ECase(
        name=name,
        hf_id="hf/unit-model",
        family="unit",
        runtime_strategy="unit_runtime",
        task_strategy="unit_task",
        reference_backend="unit_ref",
        oracle_level=oracle_level,
        bundle=f"{name}.trtfb",
        preflight=preflight or [],
        stages=[StageSpec(name="generate")],
        determinism=determinism or {},
        metadata=metadata or {},
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


def test_python_module_preflight_rejects_module_that_fails_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_name = "broken_preflight_module"
    (tmp_path / f"{module_name}.py").write_text(
        'raise OSError("native extension failed to load")\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    case = _make_case("broken-module")
    ctx = _make_ctx(tmp_path, case)

    passed, message = orchestrator._check_python_module(
        ctx,
        PreflightRequirement(
            kind="python_module_available",
            args={"module": module_name, "phase": "build"},
        ),
    )

    assert not passed
    assert "not importable in build profile" in message
    assert "native extension failed to load" in message


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


def test_pass_records_weak_validation_reason_for_invariant_oracle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case(
        "invariant-pass",
        oracle_level=OracleLevel.L4_INVARIANTS.value,
    )
    ctx = _make_ctx(tmp_path, case)
    _patch_bundle_success(monkeypatch, tmp_path)
    _patch_plugins(
        monkeypatch,
        runner=_FakeRunner(),
        reference=_FakeReference(),
        comparator=_FakeComparator(),
    )

    result = E2EOrchestrator().run(case, ctx)

    assert result.status == E2EStatus.PASS.value
    assert result.weak_validation_reason == "oracle_level is L4_invariants"
    data = _read_result_json(ctx, case)
    assert data["status"] == E2EStatus.PASS.value
    assert data["weak_validation_reason"] == "oracle_level is L4_invariants"
