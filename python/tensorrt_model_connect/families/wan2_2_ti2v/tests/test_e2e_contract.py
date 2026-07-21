# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit qualification of the Wan2.2 fixed-profile E2E contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.e2e.models.wan2_2_ti2v.e2e_plugins.comparators.diffusion import (
    DiffusionComparator,
)
from tests.e2e.models.wan2_2_ti2v.e2e_plugins.runners.diffusion import (
    DiffusionMediaRunner,
    build_generate_video_command,
    validate_official_profile,
)
from tests.e2e_harness.contracts import (
    RunContext,
    StageOutput,
    StageSpec,
    ThresholdProfile,
)
from tests.e2e_harness.manifest_loader import load_manifest
from tests.e2e_harness.orchestrator import (
    _append_declared_build_cli_args,
    _auto_register_artifacts,
    _validate_trt_runtime_path,
)


_REPO_ROOT = Path(__file__).resolve().parents[5]
_MANIFEST = _REPO_ROOT / "tests/e2e/models/wan2_2_ti2v/manifests/wan22-ti2v-5b.json"
_L0_MANIFEST = _REPO_ROOT / "tests/e2e/models/wan2_2_ti2v/manifests/wan22-ti2v-5b-l0.json"
_BUILD_CLI_ARGS = [
    {"flag": "--video-height", "input": "video_height"},
    {"flag": "--video-width", "input": "video_width"},
    {"flag": "--video-num-frames", "input": "video_num_frames"},
    {"flag": "--num-inference-steps", "input": "num_inference_steps"},
]


def _case():
    return load_manifest(_MANIFEST)


def _l0_case():
    return load_manifest(_L0_MANIFEST)


def test_manifest_is_the_official_max_profile() -> None:
    case = _case()

    assert case.hf_id == "Wan-AI/Wan2.2-TI2V-5B"
    assert case.family == "wan2_2_ti2v"
    assert case.runtime_strategy == "diffusion_wan2_2_ti2v"
    assert case.reference_backend == "hf_diffusers"
    assert case.oracle_level == "L1_external_reference"
    assert case.metadata["ci_tier"] == "nightly_only"
    assert case.metadata["runtime_timeout_s"] == 14400
    assert case.metadata["l0_replacement"] == "wan22-ti2v-5b-l0"
    assert case.metadata["build_cli_args"] == _BUILD_CLI_ARGS
    assert case.threshold_overrides == {
        "exact_num_frames": 121,
        "exact_video_height": 704,
        "exact_video_width": 1280,
        "max_pixel_mean": 0.98,
        "min_pixel_mean": 0.02,
        "min_pixel_std": 0.01,
    }
    assert case.inputs == {
        "prompt": (
            "Two anthropomorphic cats in comfy boxing gear and bright gloves "
            "fight intensely on a spotlighted stage"
        ),
        "video_num_frames": 121,
        "video_height": 704,
        "video_width": 1280,
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "flow_shift": 5.0,
        "fps": 24,
        "text_max_length": 512,
        "max_new_tokens": 30,
        "max_cache_length": 512,
        "seed": 42,
    }
    validate_official_profile(case)


def test_public_runtime_command_is_python_free_and_fully_pinned() -> None:
    case = _case()
    context = RunContext(
        case=case,
        binary_path="/opt/trtmc/bin/trtmc",
        engine_dir="/engines",
        hf_python="/venv/bin/python",
        model_plugin_dir="/opt/trtmc/models/wan2_2_ti2v",
    )

    command = build_generate_video_command(case, context, Path("/artifacts/frames"))

    assert command == [
        "/opt/trtmc/bin/trtmc",
        "generate-video",
        "/engines/wan22-ti2v-5b.trtfb",
        "--prompt",
        (
            "Two anthropomorphic cats in comfy boxing gear and bright gloves "
            "fight intensely on a spotlighted stage"
        ),
        "--output",
        "/artifacts/frames",
        "--num-steps",
        "50",
        "--cfg-scale",
        "5.0",
        "--seed",
        "42",
        "--height",
        "704",
        "--width",
        "1280",
        "--backend-dir",
        "/opt/trtmc/bin",
        "--model-plugin-dir",
        "/opt/trtmc/models/wan2_2_ti2v",
    ]
    assert "--hf-python" not in command


def test_l0_manifest_is_a_reduced_native_generation_profile() -> None:
    case = _l0_case()

    assert case.hf_id == "Wan-AI/Wan2.2-TI2V-5B"
    assert case.family == "wan2_2_ti2v"
    assert case.runtime_strategy == "diffusion_wan2_2_ti2v"
    assert case.reference_backend == "invariant_only"
    assert case.metadata["ci_tier"] == "l0_only"
    assert case.metadata["runtime_timeout_s"] == 14400
    assert case.metadata["build_cli_args"] == _BUILD_CLI_ARGS
    assert case.oracle_level == "L4_invariants"
    assert [stage.name for stage in case.stages] == ["end_to_end"]
    assert case.threshold_overrides == {
        "exact_num_frames": 5,
        "exact_video_height": 384,
        "exact_video_width": 672,
        "max_pixel_mean": 0.98,
        "min_pixel_mean": 0.02,
        "min_pixel_std": 0.01,
    }
    assert case.inputs == {
        "prompt": (
            "Two anthropomorphic cats in comfy boxing gear and bright gloves "
            "fight intensely on a spotlighted stage"
        ),
        "video_num_frames": 5,
        "video_height": 384,
        "video_width": 672,
        "num_inference_steps": 15,
        "guidance_scale": 5.0,
        "flow_shift": 5.0,
        "fps": 24,
        "text_max_length": 512,
        "max_new_tokens": 30,
        "max_cache_length": 512,
        "seed": 42,
    }
    validate_official_profile(case)


def test_l0_public_runtime_command_is_python_free_and_fully_pinned() -> None:
    case = _l0_case()
    context = RunContext(
        case=case,
        binary_path="/opt/trtmc/bin/trtmc",
        engine_dir="/engines",
        hf_python="/venv/bin/python",
        model_plugin_dir="/opt/trtmc/models/wan2_2_ti2v",
    )

    command = build_generate_video_command(
        case,
        context,
        Path("/artifacts/wan22-ti2v-5b-l0/frames"),
    )

    assert command == [
        "/opt/trtmc/bin/trtmc",
        "generate-video",
        "/engines/wan22-ti2v-5b-l0.trtfb",
        "--prompt",
        (
            "Two anthropomorphic cats in comfy boxing gear and bright gloves "
            "fight intensely on a spotlighted stage"
        ),
        "--output",
        "/artifacts/wan22-ti2v-5b-l0/frames",
        "--num-steps",
        "15",
        "--cfg-scale",
        "5.0",
        "--seed",
        "42",
        "--height",
        "384",
        "--width",
        "672",
        "--backend-dir",
        "/opt/trtmc/bin",
        "--model-plugin-dir",
        "/opt/trtmc/models/wan2_2_ti2v",
    ]
    assert "--hf-python" not in command


@pytest.mark.parametrize(
    ("case_factory", "expected_values"),
    [
        (_case, ["704", "1280", "121", "50"]),
        (_l0_case, ["384", "672", "5", "15"]),
    ],
)
def test_manifest_profile_is_forwarded_to_bundle_build(
    case_factory,
    expected_values: list[str],
) -> None:
    command: list[str] = []

    _append_declared_build_cli_args(command, case_factory())

    assert command == [
        "--video-height",
        expected_values[0],
        "--video-width",
        expected_values[1],
        "--video-num-frames",
        expected_values[2],
        "--num-inference-steps",
        expected_values[3],
    ]


def test_runner_rejects_mutated_fixed_profiles() -> None:
    case = _case()
    case.inputs["video_num_frames"] = 120

    with pytest.raises(ValueError, match="video_num_frames=120"):
        validate_official_profile(case)


def test_comparator_requires_exact_frames_dimensions_and_non_degenerate_pixels() -> None:
    comparator = DiffusionComparator()
    stage = StageSpec(name="end_to_end")
    profile = ThresholdProfile(
        task_strategy="diffusion_media_generation",
        metrics={
            "exact_num_frames": 121,
            "exact_video_height": 704,
            "exact_video_width": 1280,
            "max_pixel_mean": 0.98,
            "min_pixel_mean": 0.02,
            "min_pixel_std": 0.01,
        },
    )
    reference = StageOutput(stage_name="end_to_end", data={"_invariant_only": True})
    output = StageOutput(
        stage_name="end_to_end",
        data={
            "returncode": 0,
            "num_frames": 121,
            "frame_stats": {
                "width": 1280,
                "height": 704,
                "dimensions_consistent": True,
                "mean": 0.5,
                "std": 0.2,
            },
        },
    )

    assert comparator.compare(output, reference, profile, stage).status == "passed"

    output.data["num_frames"] = 120
    result = comparator.compare(output, reference, profile, stage)
    assert result.status == "failed"
    assert not result.metrics["exact_num_frames"].passed


def test_l0_comparator_requires_exact_reduced_profile_output() -> None:
    case = _l0_case()
    comparator = DiffusionComparator()
    stage = StageSpec(name="end_to_end")
    output = StageOutput(
        stage_name="end_to_end",
        data={
            "returncode": 0,
            "num_frames": 5,
            "frame_stats": {
                "width": 672,
                "height": 384,
                "dimensions_consistent": True,
                "mean": 0.5,
                "std": 0.2,
            },
        },
    )
    result = comparator.compare(
        output,
        StageOutput(stage_name="end_to_end", data={"_invariant_only": True}),
        ThresholdProfile(
            task_strategy="diffusion_media_generation",
            metrics=case.threshold_overrides,
        ),
        stage,
    )

    assert result.status == "passed"
    assert "exactly 5 672x384 frames" in result.composite_rule

    output.data["frame_stats"]["width"] = 671
    result = comparator.compare(
        output,
        StageOutput(stage_name="end_to_end", data={"_invariant_only": True}),
        ThresholdProfile(
            task_strategy="diffusion_media_generation",
            metrics=case.threshold_overrides,
        ),
        stage,
    )
    assert result.status == "failed"
    assert not result.metrics["exact_video_width"].passed


def test_comparator_fails_closed_without_model_sidecar_thresholds() -> None:
    comparator = DiffusionComparator()
    output = StageOutput(stage_name="end_to_end", data={"returncode": 0})
    result = comparator.compare(
        output,
        StageOutput(stage_name="end_to_end"),
        ThresholdProfile(task_strategy="diffusion_media_generation", metrics={}),
        StageSpec(name="end_to_end"),
    )

    assert result.status == "failed"
    assert "threshold sidecar is incomplete" in result.message


def test_l0_comparator_requires_runtime_strategy_and_all_bundle_sections() -> None:
    stdout = "\n".join(
        [
            "Runtime strategy:   diffusion_wan2_2_ti2v",
            "Sections:",
            "text_encoder_0_plan",
            "denoiser_plan",
            "vae_decoder_plan",
            "vae_decoder_first_frame_plan",
            "tokenizer.json",
            "config.json",
        ]
    )
    output = StageOutput(
        stage_name="bundle_contract",
        data={
            "returncode": 0,
            "stdout": stdout,
            "strict_model_plugin_probe": True,
        },
    )
    comparator = DiffusionComparator()
    result = comparator.compare(
        output,
        StageOutput(stage_name="bundle_contract"),
        ThresholdProfile(task_strategy="diffusion_media_generation", metrics={}),
        StageSpec(name="bundle_contract"),
    )

    assert result.status == "passed"

    output.data["stdout"] = stdout.replace("denoiser_plan", "")
    result = comparator.compare(
        output,
        StageOutput(stage_name="bundle_contract"),
        ThresholdProfile(task_strategy="diffusion_media_generation", metrics={}),
        StageSpec(name="bundle_contract"),
    )
    assert result.status == "failed"


def test_generation_stage_exposes_one_guard_payload_with_new_runtime_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _l0_case()
    context = RunContext(
        case=case,
        binary_path="/opt/trtmc/bin/trtmc",
        engine_dir="/engines",
        model_plugin_dir="/opt/trtmc/models",
        artifacts_dir=str(tmp_path),
    )
    marker = (
        "[trtmc] Runtime ready (backend=trt_new_runtime_default, strategy=diffusion_wan2_2_ti2v)"
    )
    expected_frames_dir = tmp_path / case.name / "frames"
    expected_frames_dir.mkdir(parents=True)
    stale_path = expected_frames_dir / "frame_9999.png"
    stale_path.touch()

    def fake_run(command, **kwargs):
        assert kwargs["env"]["TRTMC_MODEL_PLUGIN_STRICT"] == "1"
        assert kwargs["env"]["TRTMC_MODEL_PLUGIN_DIR"] == "/opt/trtmc/models"
        output_dir = Path(command[command.index("--output") + 1])
        assert output_dir == expected_frames_dir
        assert not stale_path.exists()
        for index in range(5):
            (output_dir / f"frame_{index:04d}.png").touch()
        return SimpleNamespace(returncode=0, stdout="generated", stderr=marker)

    def fake_frame_stats(frame_paths: list[Path]):
        assert frame_paths == [expected_frames_dir / f"frame_{index:04d}.png" for index in range(5)]
        return {
            "count": 5,
            "mean": 0.5,
            "std": 0.2,
            "min": 0.1,
            "max": 0.9,
            "width": 672,
            "height": 384,
            "dimensions_consistent": True,
        }

    monkeypatch.setattr(
        "tests.e2e.models.wan2_2_ti2v.e2e_plugins.runners.diffusion.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        "tests.e2e.models.wan2_2_ti2v.e2e_plugins.runners.diffusion._frame_stats",
        fake_frame_stats,
    )
    output = DiffusionMediaRunner().run_stage(
        case,
        StageSpec(name="end_to_end"),
        context,
    )

    assert output.metadata["command"][1] == "generate-video"
    assert output.metadata["returncode"] == 0
    assert output.metadata["stdout"] == "generated"
    assert output.metadata["stderr"] == marker
    assert output.metadata["strict_model_plugin_loading"] is True
    assert output.data["frames_dir"] == str(expected_frames_dir)
    assert output.data["frame_paths"] == [
        str(expected_frames_dir / f"frame_{index:04d}.png") for index in range(5)
    ]
    assert output.data["num_frames"] == 5
    assert output.data["frame_stats"]["count"] == 5
    assert output.data["frame_stats"]["width"] == 672
    assert output.data["frame_stats"]["height"] == 384
    assert output.data["frame_stats"]["dimensions_consistent"] is True
    assert _validate_trt_runtime_path(case, context, output) is None

    registered_artifacts: dict[str, str] = {}
    sink = SimpleNamespace(
        base_dir=tmp_path,
        register_artifact=registered_artifacts.__setitem__,
    )
    _auto_register_artifacts(sink, output, "trt")
    assert registered_artifacts == {
        "trt_frames": f"{case.name}/frames",
    }
