# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from tests.e2e_harness import artifact_sink
from tests.e2e_harness import orchestrator as orchestrator_module
from tests.e2e_harness.contracts import (
    CompareResult,
    E2ECase,
    PluginRuntimeContext,
    RunContext,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)


class _Runner:
    @property
    def strategy_name(self) -> str:
        return "fake_task"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        return StageOutput(stage_name=stage.name, text="trt")


class _Reference:
    @property
    def backend_name(self) -> str:
        return "fake_reference"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        return StageOutput(stage_name=stage.name, text="ref")


class _RuntimeContextPlugin:
    reference_families = ["fake_family"]
    user_contract = "fake_contract"

    def __init__(self) -> None:
        self.runtime_context: PluginRuntimeContext | None = None

    def configure_reference(self, case: E2ECase) -> dict[str, str]:
        return {"configured": "yes"}

    def verify(
        self,
        trt_output: StageOutput,
        ref_output: StageOutput,
        case: E2ECase,
        threshold: ThresholdProfile,
        *,
        runtime_context: PluginRuntimeContext | None = None,
    ) -> CompareResult:
        self.runtime_context = runtime_context
        return CompareResult(
            stage_name=trt_output.stage_name,
            status=StageStatus.PASSED.value,
            message="ok",
        )


def _case() -> E2ECase:
    return E2ECase(
        name="plugin-runtime-context",
        hf_id="fake/model",
        family="fake",
        runtime_strategy="speech_to_speech",
        task_strategy="fake_task",
        reference_backend="fake_reference",
        reference_family="fake_family",
        bundle="plugin-runtime-context.bundle",
        stages=[StageSpec(name="generate")],
    )


def test_orchestrator_passes_typed_runtime_context_to_contract_plugin(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case = _case()
    plugin = _RuntimeContextPlugin()
    artifacts_dir = tmp_path / "artifacts"
    engine_dir = tmp_path / "engines"
    engine_dir.mkdir()
    ctx = RunContext(
        case=case,
        artifacts_dir=str(artifacts_dir),
        binary_path="/opt/trtmc/bin/trtmc",
        hf_python="/venvs/base/bin/python",
        runtime_python="/venvs/runtime/bin/python",
        reference_python="/venvs/reference/bin/python",
        runtime_profile="runtime-profile",
        reference_profile="reference-profile",
        engine_dir=str(engine_dir),
    )

    monkeypatch.setattr(
        artifact_sink, "_collect_env_fingerprint", lambda ctx=None: {})
    monkeypatch.setattr(
        orchestrator_module,
        "_resolve_bundle",
        lambda case, ctx: (str(engine_dir / case.bundle), None, "", {}),
    )
    monkeypatch.setattr(orchestrator_module, "get_runner", lambda name: _Runner())
    monkeypatch.setattr(orchestrator_module, "get_reference", lambda name: _Reference())
    monkeypatch.setattr(orchestrator_module, "get_comparator", lambda name: None)
    monkeypatch.setattr(
        orchestrator_module, "get_contract_plugin", lambda family: plugin)

    result = orchestrator_module.E2EOrchestrator().run(case, ctx)

    assert result.status == "pass"
    assert case.metadata["contract_config"] == {"configured": "yes"}
    assert "_ctx" not in case.metadata
    assert plugin.runtime_context == PluginRuntimeContext(
        engine_dir=str(engine_dir),
        binary_path="/opt/trtmc/bin/trtmc",
        runtime_python="/venvs/runtime/bin/python",
        reference_python="/venvs/reference/bin/python",
        artifacts_dir=str(artifacts_dir),
    )


def test_legacy_contract_plugin_verify_signature_still_supported() -> None:
    class LegacyPlugin:
        def verify(self, trt_output, ref_output, case, threshold):
            return CompareResult(
                stage_name=trt_output.stage_name,
                status=StageStatus.PASSED.value,
                message="legacy ok",
            )

    result = orchestrator_module._verify_contract_plugin(
        LegacyPlugin(),
        StageOutput(stage_name="generate"),
        StageOutput(stage_name="generate"),
        _case(),
        ThresholdProfile(task_strategy="fake_task"),
        PluginRuntimeContext(reference_python="/venvs/reference/bin/python"),
    )

    assert result.status == StageStatus.PASSED.value
