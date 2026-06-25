"""Chronos Bolt-owned tests for neural-operator runner CLI behavior."""

from __future__ import annotations

import subprocess

from tests.e2e.models.chronos_bolt.e2e_plugins.runners import neural_operator
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec


def _make_case(inputs: dict | None = None, **overrides) -> E2ECase:
    defaults = dict(
        name="chronos-bolt-case",
        hf_id="dummy/model",
        family="chronos_bolt",
        runtime_strategy="chronos_bolt_trt",
        task_strategy="neural_operator",
        bundle="chronos-bolt-case.trtfb",
        inputs=inputs or {},
    )
    defaults.update(overrides)
    return E2ECase(**defaults)


def _make_ctx(case: E2ECase, tmp_path) -> RunContext:
    binary_path = tmp_path / "trtmc"
    binary_path.write_text("", encoding="utf-8")
    return RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        binary_path=str(binary_path),
        engine_dir=str(tmp_path),
    )


def test_runner_uses_solve_entrypoint_and_supported_flags(monkeypatch, tmp_path) -> None:
    case = _make_case(inputs={"field_input": [0.1, 0.2, 0.3]})
    ctx = _make_ctx(case, tmp_path)
    ctx.hf_python = str(tmp_path / "python")

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Output [3]: 1 2 3\n", stderr="")

    monkeypatch.setattr(neural_operator.subprocess, "run", _fake_run)

    out = neural_operator.NeuralOperatorRunner().run_stage(
        case, StageSpec(name="full_inference"), ctx)

    cmd = captured["cmd"]
    assert cmd[1] == "solve"
    assert "--field-input" in cmd
    assert "--input-field" not in cmd
    assert "--output-field" not in cmd
    assert "--hf-python" not in cmd
    assert out.metadata["input_mode"] == "field"
    assert out.data["output_field"] == [1.0, 2.0, 3.0]


def test_runner_accepts_branch_only_inputs(monkeypatch, tmp_path) -> None:
    case = _make_case(inputs={"branch_input": [0.1, 0.2, 0.3]})
    ctx = _make_ctx(case, tmp_path)

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Output [3]: 1 2 3\n", stderr="")

    monkeypatch.setattr(neural_operator.subprocess, "run", _fake_run)

    out = neural_operator.NeuralOperatorRunner().run_stage(
        case, StageSpec(name="full_inference"), ctx)

    cmd = captured["cmd"]
    assert "--branch-input" in cmd
    assert "--trunk-input" not in cmd
    assert out.metadata["input_mode"] == "branch"


def test_runner_wraps_distributed_command(monkeypatch, tmp_path) -> None:
    case = _make_case(
        inputs={"field_input": [0.1, 0.2, 0.3]},
        metadata={
            "distributed_runtime": {
                "enabled": True,
                "launcher": "mpirun",
                "world_size": 4,
                "export_env": ["LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES"],
            }
        },
    )
    ctx = _make_ctx(case, tmp_path)
    ctx.ld_library_path = "/tmp/lib"

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        stdout = (
            "[1,0]<stdout>:Output [3]: 1 2 3\n"
            "[1,1]<stdout>:Output [3]: 1 2 3\n"
        )
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(neural_operator.subprocess, "run", _fake_run)

    out = neural_operator.NeuralOperatorRunner().run_stage(
        case, StageSpec(name="full_inference"), ctx)

    cmd = captured["cmd"]
    assert cmd[:4] == ["mpirun", "--tag-output", "-np", "4"]
    assert "-x" in cmd
    assert "TRTMC_NCCL_RENDEZVOUS" in captured["env"]
    assert out.data["output_field"] == [1.0, 2.0, 3.0]


def test_runner_accepts_fragmented_mpi_stdout(monkeypatch, tmp_path) -> None:
    case = _make_case(
        inputs={"field_input": [0.1, 0.2, 0.3]},
        metadata={
            "distributed_runtime": {
                "enabled": True,
                "launcher": "mpirun",
                "world_size": 4,
            }
        },
    )
    ctx = _make_ctx(case, tmp_path)

    def _fake_run(cmd, **kwargs):
        stdout = (
            "[1,0]<stdout>:Output [3]: 1 2 0.084"
            "[1,1]<stdout>:Output [3]: 9 9 9\n"
            "[1,0]<stdout>:77899432182312\n"
        )
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(neural_operator.subprocess, "run", _fake_run)

    out = neural_operator.NeuralOperatorRunner().run_stage(
        case, StageSpec(name="full_inference"), ctx)

    assert out.data["output_field"] == [1.0, 2.0, 0.08477899432182312]
