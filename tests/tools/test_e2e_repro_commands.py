"""Tests for E2E orchestrator repro command generation.

Trace: ARCH-E2E-001, UD-E2E-REPRO
Intent: Validate that E2E orchestrator generates correct reproduction commands for each task strategy
Preconditions: E2ECase and RunContext are constructed with known strategy and input parameters
Postconditions: Generated repro commands contain correct binary subcommand, flags, and input paths
"""

from __future__ import annotations

from tests.e2e_harness.contracts import E2ECase, RunContext
from tests.e2e_harness.orchestrator import _build_repro_commands
from tests.e2e_harness.registry import register_repro_command_provider, reset


def _make_ctx(tmp_path) -> RunContext:
    return RunContext(
        case=E2ECase(
            name="case-a",
            hf_id="dummy/model",
            family="dummy",
            runtime_strategy="decoder_kv_cache",
            bundle="case-a.trtfb",
            stages=[],
        ),
        artifacts_dir=str(tmp_path),
        binary_path="./build/trtmc",
        hf_python="/usr/bin/python3",
        engine_dir="/tmp/engines",
    )


class _PromptedSegmentationReproProvider:
    @property
    def family_name(self) -> str:
        return "prompted_text_segmentation_family"

    def build_trt_inference_command(
        self,
        case: E2ECase,
        ctx: RunContext,
        bundle_path: str,
    ) -> list[str]:
        image = case.inputs.get("image") or case.inputs.get("test_image") or ""
        parts = [
            ctx.binary_path,
            "segment-prompted",
            bundle_path,
            "--image",
            str(image),
            "--output",
            "/tmp/trtmc_masks",
        ]
        prompt = case.inputs.get("prompt")
        if prompt:
            parts.extend(["--prompt", str(prompt)])
        else:
            parts.extend([
                "--point-x",
                str(case.inputs.get("point_x")),
                "--point-y",
                str(case.inputs.get("point_y")),
            ])
        return parts


def _register_prompted_segmentation_provider() -> None:
    reset()
    register_repro_command_provider(_PromptedSegmentationReproProvider())


def test_repro_commands_use_segment_prompted_for_prompted_segmentation(tmp_path) -> None:
    _register_prompted_segmentation_provider()
    case = E2ECase(
        name="prompted-segmentation-point-case",
        hf_id="example-org/prompted-segmentation-point",
        family="prompted_text_segmentation_family",
        runtime_strategy="prompted_segmentation",
        task_strategy="prompted_segmentation",
        bundle="prompted-segmentation-point.trtfb",
        inputs={
            "test_image": "data/test_img.jpeg",
            "point_x": 0.5,
            "point_y": 0.25,
        },
        stages=[],
    )
    repro = _build_repro_commands(
        case,
        _make_ctx(tmp_path),
        "/tmp/engines/prompted-segmentation-point.trtfb",
        {},
    )

    cmd = repro["trt_inference"]
    assert " segment-prompted " in f" {cmd} "
    assert "--output /tmp/trtmc_masks" in cmd
    assert "--point-x 0.5" in cmd
    assert "--point-y 0.25" in cmd


def test_repro_commands_use_text_prompt_for_prompted_segmentation(tmp_path) -> None:
    _register_prompted_segmentation_provider()
    case = E2ECase(
        name="prompted-segmentation-text-case",
        hf_id="example-org/prompted-segmentation-text",
        family="prompted_text_segmentation_family",
        runtime_strategy="prompted_segmentation",
        task_strategy="prompted_segmentation",
        reference_family="prompted_text_segmentation",
        bundle="prompted-segmentation-text.trtfb",
        inputs={
            "image": "data/test_img.jpeg",
            "prompt": "car",
        },
        stages=[],
    )
    repro = _build_repro_commands(
        case,
        _make_ctx(tmp_path),
        "/tmp/engines/prompted-segmentation-text.trtfb",
        {},
    )

    cmd = repro["trt_inference"]
    assert " segment-prompted " in f" {cmd} "
    assert "--image data/test_img.jpeg" in cmd
    assert "--prompt car" in cmd
    assert "--point-x" not in cmd
    assert "--point-y" not in cmd


def test_repro_commands_use_generate_video_for_diffusion(tmp_path) -> None:
    case = E2ECase(
        name="diffusion-media-case",
        hf_id="example-org/diffusion-media",
        family="diffusion_media_family",
        runtime_strategy="diffusion",
        task_strategy="diffusion_media_generation",
        bundle="diffusion-media.trtfb",
        inputs={
            "test_prompt": "A photo of a cat sitting on a windowsill at sunset",
            "num_inference_steps": 28,
            "guidance_scale": 3.0,
            "seed": 42,
        },
        stages=[],
    )
    repro = _build_repro_commands(
        case,
        _make_ctx(tmp_path),
        "/tmp/engines/diffusion-media.trtfb",
        {},
    )

    cmd = repro["trt_inference"]
    assert " generate-video " in f" {cmd} "
    assert "--output /tmp/trtmc_frames" in cmd
    assert "--num-steps 28" in cmd
    assert "--guidance-scale 3.0" in cmd
    assert "--seed 42" in cmd
