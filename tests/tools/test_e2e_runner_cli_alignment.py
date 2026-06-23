"""Unit tests for E2E runner to CLI contract alignment.

Trace: ARCH-E2E-001, UD-E2E-CLI
Intent: Validate that E2E runners construct CLI commands matching the C++ binary's expected argument contract
Preconditions: Fake binary path and E2ECase with strategy-specific inputs are provided
Postconditions: Runner-generated commands contain correct subcommand aliases and required flags
"""

from __future__ import annotations

import subprocess

from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec
from tests.e2e_harness.runners import (
    embedding,
    neural_operator,
    object_detection,
    omni,
)


def _make_case(task_strategy: str, inputs: dict | None = None, **overrides) -> E2ECase:
    defaults = dict(
        name="case-a",
        hf_id="dummy/model",
        family="dummy",
        runtime_strategy=task_strategy,
        task_strategy=task_strategy,
        bundle="case-a.trtfb",
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


def test_object_detection_runner_uses_detection_alias_flags(tmp_path, monkeypatch):
    case = _make_case(
        "object_detection",
        inputs={"image": str(tmp_path / "img.jpg"), "score_threshold": 0.42},
    )
    ctx = _make_ctx(case, tmp_path)
    image_path = tmp_path / "img.jpg"
    image_path.write_text("img", encoding="utf-8")

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(object_detection.subprocess, "run", _fake_run)

    out = object_detection.ObjectDetectionRunner().run_stage(
        case, StageSpec(name="full_inference"), ctx)

    cmd = captured["cmd"]
    assert "--output-json" in cmd
    assert "--score-threshold" in cmd
    assert out.metadata["command"] == cmd


def test_neural_operator_runner_uses_solve_entrypoint_and_supported_flags(monkeypatch, tmp_path):
    case = _make_case(
        "neural_operator",
        inputs={"field_input": [0.1, 0.2, 0.3]},
    )
    ctx = _make_ctx(case, tmp_path)
    ctx.hf_python = str(tmp_path / "python")

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="Output [3]: 1 2 3\n", stderr="")

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


def test_neural_operator_runner_accepts_branch_only_inputs(monkeypatch, tmp_path):
    case = _make_case(
        "neural_operator",
        inputs={"branch_input": [0.1, 0.2, 0.3]},
    )
    ctx = _make_ctx(case, tmp_path)

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="Output [3]: 1 2 3\n", stderr="")

    monkeypatch.setattr(neural_operator.subprocess, "run", _fake_run)

    out = neural_operator.NeuralOperatorRunner().run_stage(
        case, StageSpec(name="full_inference"), ctx)

    cmd = captured["cmd"]
    assert "--branch-input" in cmd
    assert "--trunk-input" not in cmd
    assert out.metadata["input_mode"] == "branch"


def test_neural_operator_runner_wraps_distributed_command(monkeypatch, tmp_path):
    case = _make_case(
        "neural_operator",
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


def test_neural_operator_runner_accepts_fragmented_mpi_stdout(monkeypatch, tmp_path):
    case = _make_case(
        "neural_operator",
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


def test_omni_runner_thinker_stage_drops_unsupported_stage_flag(monkeypatch, tmp_path):
    case = _make_case(
        "omni_multimodal",
        inputs={"prompt": "hello", "max_new_tokens": 7},
    )
    ctx = _make_ctx(case, tmp_path)

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="hello back", stderr="")

    monkeypatch.setattr(omni.subprocess, "run", _fake_run)

    out = omni.OmniMultimodalRunner().run_stage(
        case, StageSpec(name="thinker_decode"), ctx)

    cmd = captured["cmd"]
    assert cmd[1] == "run"
    assert "--stage" not in cmd
    assert out.metadata["cli_stage_supported"] is False
    assert out.metadata["entrypoint"] == "run"


def test_omni_runner_vision_stage_maps_to_embed_without_stage_flag(monkeypatch, tmp_path):
    image_path = tmp_path / "img.jpg"
    image_path.write_text("img", encoding="utf-8")
    case = _make_case(
        "omni_multimodal",
        inputs={"image": str(image_path), "prompt": "caption me"},
    )
    ctx = _make_ctx(case, tmp_path)

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(
            cmd, 0, stdout='{"embedding": [0.1, 0.2], "dim": 2}\n', stderr="")

    monkeypatch.setattr(omni.subprocess, "run", _fake_run)

    out = omni.OmniMultimodalRunner().run_stage(
        case, StageSpec(name="vision_encode"), ctx)

    cmd = captured["cmd"]
    assert cmd[1] == "embed"
    assert "--stage" not in cmd
    assert out.metadata["entrypoint"] == "embed"
    assert out.data["embedding"] == [0.1, 0.2]


def test_embedding_parser_accepts_fragmented_json_from_mpirun() -> None:
    stdout = '{"embedding": [0.1, 0.2,\n 0.3], "dim": 3}\n'

    assert embedding._parse_embedding(stdout) == [0.1, 0.2, 0.3]


def test_composite_runner_uses_run_without_stage_flag(monkeypatch, tmp_path):
    case = _make_case(
        "composite_pipeline",
        inputs={"prompt": "hello", "max_new_tokens": 4},
    )
    ctx = _make_ctx(case, tmp_path)

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(omni.subprocess, "run", _fake_run)

    out = omni.CompositePipelineRunner().run_stage(
        case, StageSpec(name="end_to_end"), ctx)

    cmd = captured["cmd"]
    assert cmd[1] == "run"
    assert "--stage" not in cmd
    assert out.metadata["entrypoint"] == "run"
