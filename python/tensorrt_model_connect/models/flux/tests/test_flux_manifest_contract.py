# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flux-owned manifest contract tests."""

from __future__ import annotations

from pathlib import Path

from tests.e2e_harness.manifest_loader import load_manifest


def test_flux2_fp8_manifest_uses_end_to_end_image_contract() -> None:
    """FLUX.2 FP8 should not inherit unrelated optional debug substages."""
    manifests_dir = Path(__file__).with_name("manifests")

    for manifest_name in (
        "flux-2-dev.json",
        "flux-2-dev-fp8.json",
        "flux-2-dev-l0.json",
    ):
        case = load_manifest(manifests_dir / manifest_name)

        assert case.metadata["task_eval"]["reference_precision"] == "bf16"
        assert case.reference_family == "diffusers_image_gen"
        assert case.user_contract == "diffusion_image"
        assert [stage.name for stage in case.stages] == ["end_to_end"]
        assert all(stage.required for stage in case.stages)

    fp8_case = load_manifest(manifests_dir / "flux-2-dev-fp8.json")
    flux2_case = load_manifest(manifests_dir / "flux-2-dev.json")
    flux2_l0_case = load_manifest(manifests_dir / "flux-2-dev-l0.json")
    assert flux2_case.metadata["precision"] == "bf16"
    assert flux2_l0_case.metadata["precision"] == "bf16"
    assert "Wan-specific" in fp8_case.metadata["notes"]


def test_flux_schnell_reference_precision_matches_candidate() -> None:
    """The non-quantized FLUX.1 reference must use the bundle precision."""
    manifest_path = Path(__file__).with_name("manifests") / "flux-schnell.json"
    case = load_manifest(manifest_path)

    assert case.metadata["precision"] == "fp16"
    assert case.metadata["task_eval"]["reference_precision"] == "fp16"


def test_flux_production_defaults_run_end_to_end_contract_once() -> None:
    """Production Flux cases must not repeat the full TRT and HF pipelines."""
    manifests_dir = Path(__file__).with_name("manifests")

    for manifest_name in ("flux-2-dev.json", "flux-schnell.json"):
        case = load_manifest(manifests_dir / manifest_name)

        assert case.reference_family == "diffusers_image_gen"
        assert case.user_contract == "diffusion_image"
        assert [stage.name for stage in case.stages] == ["end_to_end"]
        assert all(stage.required for stage in case.stages)


def test_flux_batch2_manifest_declares_real_batch_contract() -> None:
    manifest_path = Path(__file__).with_name("manifests") / "flux-schnell-l0-batch2.json"
    case = load_manifest(manifest_path)

    assert case.inputs["max_batch_size"] == 2
    assert case.inputs["expected_batch_size"] == 2
    assert case.inputs["batch_prompts"] == [
        "A red cube on a white table",
        "A blue sphere on a white table",
    ]
    assert case.inputs["batch_seeds"] == [42, 42]
    assert case.threshold_overrides["batch_min_pairwise_pixel_mae"] == 0.01
    assert case.metadata["contract_config"]["use_diffusers"] is True
    assert {spec["flag"] for spec in case.metadata["build_cli_args"]} >= {
        "--max-batch-size",
    }


def test_flux_cp4_manifest_declares_ulysses_world_size() -> None:
    manifest_path = Path(__file__).with_name("manifests") / "flux-schnell-l0-cp4.json"
    case = load_manifest(manifest_path)

    assert case.name == "flux-schnell-l0-cp4"
    assert case.metadata["build_args"]["parallel"] == {
        "mode": "context_parallel",
        "cp_size": 4,
    }
    assert case.metadata["distributed_runtime"]["world_size"] == 4
    assert case.reference_family == "diffusers_image_gen"
