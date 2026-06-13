"""Tests for E2E orchestrator repro command generation.

Trace: ARCH-E2E-001, UD-E2E-REPRO
Intent: Validate that E2E orchestrator generates correct reproduction commands for each task strategy
Preconditions: E2ECase and RunContext are constructed with known strategy and input parameters
Postconditions: Generated repro commands contain correct binary subcommand, flags, and input paths
"""

from __future__ import annotations

from tests.e2e_harness.contracts import E2ECase, RunContext
from tests.e2e_harness.orchestrator import _build_repro_commands


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


def test_repro_commands_use_segment_sam_for_prompted_segmentation(tmp_path) -> None:
    case = E2ECase(
        name="sam-case",
        hf_id="facebook/sam-vit-base",
        family="sam",
        runtime_strategy="prompted_segmentation",
        task_strategy="prompted_segmentation",
        bundle="sam-vit-base.trtfb",
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
        "/tmp/engines/sam-vit-base.trtfb",
        {},
    )

    cmd = repro["trt_inference"]
    assert " segment-sam " in f" {cmd} "
    assert "--output /tmp/trtmc_masks" in cmd
    assert "--point-x 0.5" in cmd
    assert "--point-y 0.25" in cmd


def test_repro_commands_use_text_prompt_for_sam3(tmp_path) -> None:
    case = E2ECase(
        name="sam3",
        hf_id="facebook/sam3",
        family="sam3",
        runtime_strategy="prompted_segmentation",
        task_strategy="prompted_segmentation",
        reference_family="prompted_segmentation_sam3",
        bundle="sam3.trtfb",
        inputs={
            "image": "data/test_img.jpeg",
            "prompt": "car",
        },
        stages=[],
    )
    repro = _build_repro_commands(
        case,
        _make_ctx(tmp_path),
        "/tmp/engines/sam3.trtfb",
        {},
    )

    cmd = repro["trt_inference"]
    assert " segment-sam " in f" {cmd} "
    assert "--image data/test_img.jpeg" in cmd
    assert "--prompt car" in cmd
    assert "--point-x" not in cmd
    assert "--point-y" not in cmd


def test_repro_commands_use_generate_video_for_diffusion(tmp_path) -> None:
    case = E2ECase(
        name="flux-case",
        hf_id="black-forest-labs/FLUX.2-dev",
        family="flux",
        runtime_strategy="diffusion",
        task_strategy="diffusion_media_generation",
        bundle="flux-2-dev.trtfb",
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
        "/tmp/engines/flux-2-dev.trtfb",
        {},
    )

    cmd = repro["trt_inference"]
    assert " generate-video " in f" {cmd} "
    assert "--output /tmp/trtmc_frames" in cmd
    assert "--num-steps 28" in cmd
    assert "--guidance-scale 3.0" in cmd
    assert "--seed 42" in cmd
