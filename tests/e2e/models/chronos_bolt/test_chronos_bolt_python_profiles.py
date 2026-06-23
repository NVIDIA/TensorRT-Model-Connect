"""Chronos-Bolt-owned Python profile tests."""

from __future__ import annotations

import subprocess

from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec
from tests.e2e_harness.python_profiles import resolve_case_profile_names
from tests.e2e.models.chronos_bolt.e2e_plugins.references.torch_reference import (
    TorchReference,
)


def _make_case(**kwargs) -> E2ECase:
    defaults = dict(
        name="chronos-bolt-case",
        hf_id="dummy/model",
        family="chronos_bolt",
        runtime_strategy="chronos_bolt_trt",
        task_strategy="neural_operator",
        bundle="chronos-bolt-case.trtfb",
        inputs={"branch_input": [1.0, 2.0, 3.0]},
        stages=[],
        reference_backend="torch_reference",
    )
    defaults.update(kwargs)
    return E2ECase(**defaults)


def test_chronos_bolt_family_metadata_declares_default_execution_profiles():
    case = _make_case()

    assert resolve_case_profile_names(case) == {
        "build": "chronos",
        "runtime": "base",
        "reference": "chronos",
    }


def test_torch_reference_time_series_uses_reference_python_subprocess(monkeypatch, tmp_path):
    case = _make_case()
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        hf_python="/usr/bin/python3",
        reference_python="/tmp/chronos-python",
    )
    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=(
                '{"output_field":[1.0,2.0,3.0],"output_shape":[1,3],'
                '"reference_output_name":"quantile_preds","model_path":"dummy/model"}'
            ),
            stderr="",
        )

    monkeypatch.setattr(
        "tests.e2e.models.chronos_bolt.e2e_plugins.references.torch_reference.subprocess.run",
        _fake_run,
    )

    out = TorchReference().run_stage(case, StageSpec(name="full_inference"), ctx)

    assert captured["cmd"][0] == "/tmp/chronos-python"
    assert out.data["reference_output_name"] == "quantile_preds"
    assert out.data["output_field"] == [1.0, 2.0, 3.0]
