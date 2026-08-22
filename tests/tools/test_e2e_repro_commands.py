# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for E2E orchestrator repro command generation.

Trace: ARCH-E2E-001, UD-E2E-REPRO
Intent: Validate that E2E orchestrator generates correct reproduction commands for each task strategy
Preconditions: E2ECase and RunContext are constructed with known strategy and input parameters
Postconditions: Generated repro commands contain correct binary subcommand, flags, and input paths
"""

from __future__ import annotations

from pathlib import Path

from tests.e2e_harness.contracts import E2ECase, RunContext
from tests.e2e_harness.manifest_loader import load_manifest
from tests.e2e_harness.orchestrator import _build_repro_commands
from tests.e2e_harness.registry import (
    register_repro_command_provider,
    register_runner,
    reset,
)


def _make_ctx(tmp_path) -> RunContext:
    return RunContext(
        case=E2ECase(
            name="case-a",
            hf_id="dummy/model",
            family="dummy",
            runtime_strategy="dummy_decoder_kv_cache",
            bundle="case-a.bundle",
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


class _DiffusionReproProvider:
    @property
    def family_name(self) -> str:
        return "diffusion_media_family"

    def build_trt_inference_command(
        self,
        case: E2ECase,
        ctx: RunContext,
        bundle_path: str,
    ) -> list[str]:
        parts = [
            ctx.binary_path,
            "generate-video",
            bundle_path,
            "--prompt",
            str(case.inputs.get("prompt", case.inputs.get("test_prompt", ""))),
            "--output",
            "/tmp/trtmc_frames",
            "--num-steps",
            str(case.inputs.get("num_inference_steps", 30)),
        ]
        guidance_scale = case.inputs.get("guidance_scale")
        if guidance_scale is not None:
            parts.extend(["--guidance-scale", str(guidance_scale)])
        if "seed" in case.inputs:
            parts.extend(["--seed", str(case.inputs["seed"])])
        return parts


class _RunnerOwnedReproHook:
    @property
    def strategy_name(self) -> str:
        return "runner_owned_repro_strategy"

    def build_trt_inference_command(
        self,
        case: E2ECase,
        ctx: RunContext,
        bundle_path: str,
    ) -> list[str]:
        return [
            ctx.binary_path,
            "runner-owned-command",
            bundle_path,
            "--case",
            case.name,
        ]


def _register_prompted_segmentation_provider() -> None:
    reset()
    register_repro_command_provider(_PromptedSegmentationReproProvider())


def test_repro_commands_use_segment_prompted_for_prompted_segmentation(tmp_path) -> None:
    _register_prompted_segmentation_provider()
    case = E2ECase(
        name="prompted-segmentation-point-case",
        hf_id="example-org/prompted-segmentation-point",
        family="prompted_text_segmentation_family",
        runtime_strategy="prompted_text_segmentation_family_prompted_segmentation",
        task_strategy="prompted_segmentation",
        bundle="prompted-segmentation-point.bundle",
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
        "/tmp/engines/prompted-segmentation-point.bundle",
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
        runtime_strategy="prompted_text_segmentation_family_prompted_segmentation",
        task_strategy="prompted_segmentation",
        reference_family="prompted_text_segmentation",
        bundle="prompted-segmentation-text.bundle",
        inputs={
            "image": "data/test_img.jpeg",
            "prompt": "car",
        },
        stages=[],
    )
    repro = _build_repro_commands(
        case,
        _make_ctx(tmp_path),
        "/tmp/engines/prompted-segmentation-text.bundle",
        {},
    )

    cmd = repro["trt_inference"]
    assert " segment-prompted " in f" {cmd} "
    assert "--image data/test_img.jpeg" in cmd
    assert "--prompt car" in cmd
    assert "--point-x" not in cmd
    assert "--point-y" not in cmd


def test_repro_commands_use_model_owned_provider_for_diffusion(tmp_path) -> None:
    reset()
    register_repro_command_provider(_DiffusionReproProvider())
    case = E2ECase(
        name="diffusion-media-case",
        hf_id="example-org/diffusion-media",
        family="diffusion_media_family",
        runtime_strategy="diffusion",
        task_strategy="diffusion_media_generation",
        bundle="diffusion-media.bundle",
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
        "/tmp/engines/diffusion-media.bundle",
        {},
    )

    cmd = repro["trt_inference"]
    assert " generate-video " in f" {cmd} "
    assert "--output /tmp/trtmc_frames" in cmd
    assert "--num-steps 28" in cmd
    assert "--guidance-scale 3.0" in cmd
    assert "--seed 42" in cmd


def test_repro_commands_can_use_runner_owned_hook(tmp_path) -> None:
    reset()
    register_runner(_RunnerOwnedReproHook())
    case = E2ECase(
        name="runner-owned-case",
        hf_id="example-org/runner-owned",
        family="runner_owned_family",
        runtime_strategy="runner_owned_repro_strategy",
        task_strategy="runner_owned_repro_strategy",
        bundle="runner-owned.bundle",
        inputs={},
        stages=[],
    )
    repro = _build_repro_commands(
        case,
        _make_ctx(tmp_path),
        "/tmp/engines/runner-owned.bundle",
        {},
    )

    cmd = repro["trt_inference"]
    assert " runner-owned-command " in f" {cmd} "
    assert "--case runner-owned-case" in cmd


def test_qwen_native_kv_repro_preserves_model_only_build(tmp_path) -> None:
    reset()
    manifest = (
        Path(__file__).resolve().parents[1]
        / "e2e"
        / "models"
        / "qwen"
        / "manifests"
        / "qwen3-0.6b-regression-native-kv-chunked-prefill.json"
    )
    case = load_manifest(manifest)
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path / "artifacts"),
        binary_path="./build/trtmc",
        hf_python="/usr/bin/python3",
        engine_dir=str(tmp_path),
    )
    bundle = str(tmp_path / case.bundle)

    repro = _build_repro_commands(case, ctx, bundle, {})

    assert "--max-cache-length" not in repro["build_bundle"]
    assert "--precision" not in repro["build_bundle"]
    assert f"--model-revision {case.hf_revision}" in repro["build_bundle"]
    resolved_prompt = tmp_path / "artifacts" / case.name / "resolved_prompt.txt"
    assert f"--prompts-file {resolved_prompt}" in repro["trt_inference"]
    assert "--max-new-tokens 2" in repro["trt_inference"]
    assert "--temperature 0.0" in repro["trt_inference"]
    assert "--e2e-category regression" in repro["rerun_test_rebuild"]


def test_llama_chunked_prefill_repro_preserves_model_only_build(tmp_path) -> None:
    reset()
    manifest = (
        Path(__file__).resolve().parents[1]
        / "e2e"
        / "models"
        / "llama"
        / "manifests"
        / "minitron-4b-width-regression-native-kv-chunked-prefill.json"
    )
    case = load_manifest(manifest)
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path / "artifacts"),
        binary_path="./build/trtmc",
        hf_python="/usr/bin/python3",
        engine_dir=str(tmp_path),
    )
    bundle = str(tmp_path / case.bundle)

    repro = _build_repro_commands(case, ctx, bundle, {})

    assert "--max-cache-length" not in repro["build_bundle"]
    assert "--precision" not in repro["build_bundle"]
    assert f"--model-revision {case.hf_revision}" in repro["build_bundle"]
    resolved_prompt = (
        tmp_path
        / "artifacts"
        / case.name
        / "resolved_prompt.txt"
    )
    assert f"--prompts-file {resolved_prompt}" in repro["trt_inference"]
    assert "--max-new-tokens 2" in repro["trt_inference"]
    assert "--temperature 0.0" in repro["trt_inference"]
    assert "--e2e-category regression" in repro["rerun_test_rebuild"]


class _WeirdTokensProvider:
    @property
    def family_name(self) -> str:
        return "weird_family"

    def build_trt_inference_command(
        self,
        case: E2ECase,
        ctx: RunContext,
        bundle_path: str,
    ) -> list[str]:
        return [
            ctx.binary_path,
            "run",
            bundle_path,
            "--prompt", "A 'single' quote",
            "--newline", "line1\nline2",
            "--empty", "",
            "--dollar", "$HOME",
            "--backslash", "\\path\\to\\dir",
            "--spaces", "many spaces here",
        ]


def test_repro_commands_shlex_round_trip(tmp_path) -> None:
    reset()
    register_repro_command_provider(_WeirdTokensProvider())
    case = E2ECase(
        name="weird-tokens-case",
        hf_id="test",
        family="weird_family",
        runtime_strategy="weird",
        task_strategy="weird",
        bundle="test.bundle",
        inputs={},
        stages=[],
    )
    repro = _build_repro_commands(
        case,
        _make_ctx(tmp_path),
        "/tmp/engines/test.bundle",
        {},
    )

    cmd = repro["trt_inference"]
    import shlex
    parts = shlex.split(cmd)
    
    assert parts == [
        "./build/trtmc",
        "run",
        "/tmp/engines/test.bundle",
        "--prompt", "A 'single' quote",
        "--newline", "line1\nline2",
        "--empty", "",
        "--dollar", "$HOME",
        "--backslash", "\\path\\to\\dir",
        "--spaces", "many spaces here",
    ]
