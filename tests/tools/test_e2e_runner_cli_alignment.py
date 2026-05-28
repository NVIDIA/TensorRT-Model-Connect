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
    audio_speech,
    neural_operator,
    object_detection,
    omni,
    segmentation,
    text_generation,
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


def test_audio_runner_maps_runtime_config_to_set_flags(monkeypatch, tmp_path):
    case = _make_case(
        "text_to_audio",
        inputs={"prompt": "hello", "max_new_tokens": 12},
        family="magpie_tts",
        runtime_strategy="text_to_audio_magpie",
        metadata={
            "runtime_config": {
                "audio_magpie": {
                    "cfg_scale": 2.5,
                    "temperature": 0.6,
                    "seed": 42,
                }
            }
        },
    )
    ctx = _make_ctx(case, tmp_path)

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(audio_speech.subprocess, "run", _fake_run)

    out = audio_speech.TextToAudioRunner().run_stage(
        case, StageSpec(name="generate"), ctx)

    cmd = captured["cmd"]
    assert "--set" in cmd
    assert "audio_magpie.cfg_scale=2.5" in cmd
    assert "audio_magpie.temperature=0.6" in cmd
    assert "audio_magpie.seed=42" in cmd
    assert "TRTMC_MAGPIE_SEED" not in captured["env"]
    assert out.metadata["command"] == cmd


def test_bark_distributed_audio_runner_wraps_mpirun_once(monkeypatch, tmp_path):
    case = _make_case(
        "text_to_audio",
        inputs={"prompt": "hello"},
        family="bark",
        runtime_strategy="text_to_audio_bark",
        metadata={
            "distributed_runtime": {
                "enabled": True,
                "launcher": "mpirun",
                "world_size": 4,
            },
        },
        determinism={"seed": 42},
    )
    ctx = _make_ctx(case, tmp_path)

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(audio_speech.subprocess, "run", _fake_run)

    out = audio_speech.TextToAudioRunner().run_stage(
        case, StageSpec(name="generate"), ctx)

    cmd = captured["cmd"]
    assert cmd[:4] == ["mpirun", "--tag-output", "-np", "4"]
    assert cmd.count("mpirun") == 1
    assert "trtmc_rank_audio" in cmd
    assert "audio_bark.seed=42" in cmd
    assert out.metadata["command"] == cmd


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


def test_text_generation_runner_maps_diffusion_mode_flags(monkeypatch, tmp_path):
    case = _make_case(
        "text_generation_causal",
        inputs={
            "prompt": "hello",
            "max_new_tokens": 32,
            "generation_mode": "diffusion",
            "block_length": 32,
            "threshold": 0.9,
        },
        family="nemotron_labs_diffusion",
        runtime_strategy="nemotron_labs_diffusion",
        ci_lane="acceptance",
        reference_family="nemotron_labs_diffusion_model_card",
        user_contract="model_card_generation_parity",
        metadata={"contract_config": {}},
    )
    ctx = _make_ctx(case, tmp_path)

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        output_path = cmd[cmd.index("-o") + 1]
        tmp_path.joinpath(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write('{"id":0,"generated":"ok","token_ids":[1,2,3]}\n')
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(text_generation.subprocess, "run", _fake_run)

    out = text_generation.TextGenerationCausalRunner().run_stage(
        case, StageSpec(name="full_generation"), ctx)

    cmd = captured["cmd"]
    assert "--generation-mode" in cmd
    assert "diffusion" in cmd
    assert "--block-length" in cmd
    assert "--threshold" in cmd
    assert "-o" in cmd
    assert out.data["token_ids"] == [1, 2, 3]
    assert out.metadata["cpp"]["command"] == cmd


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


def test_prompted_segmentation_runner_uses_segment_sam_cli(monkeypatch, tmp_path):
    image_path = tmp_path / "img.jpg"
    image_path.write_text("img", encoding="utf-8")
    case = _make_case(
        "prompted_segmentation",
        inputs={"image": str(image_path), "point_x": 0.25, "point_y": 0.75},
    )
    ctx = _make_ctx(case, tmp_path)

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(segmentation.subprocess, "run", _fake_run)

    out = segmentation.PromptedSegmentationRunner().run_stage(
        case, StageSpec(name="full_inference"), ctx)

    cmd = captured["cmd"]
    assert cmd[1] == "segment-sam"
    assert "--output" in cmd
    assert "--point-x" in cmd
    assert "--point-y" in cmd
    assert "--output-dir" not in cmd
    assert "--point" not in cmd
    assert out.metadata["command"] == cmd
