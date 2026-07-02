# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron Labs Diffusion-owned manifest contract tests."""

from __future__ import annotations

from pathlib import Path

from tests.e2e_harness.manifest_loader import load_manifest


def test_nemotron_labs_diffusion_manifests_cover_model_card_modes() -> None:
    """The 8B model-card generation surfaces should all have nightly cases."""
    manifest_dir = Path(__file__).with_name("manifests")
    manifest_paths = [
        manifest_dir / "nemotron-labs-diffusion-8b-ar.json",
        manifest_dir / "nemotron-labs-diffusion-8b-diffusion.json",
        manifest_dir / "nemotron-labs-diffusion-8b-linear-spec.json",
        manifest_dir / "nemotron-labs-diffusion-8b.json",
    ]
    cases = [load_manifest(path) for path in manifest_paths]

    modes = {case.inputs["generation_mode"] for case in cases}
    assert modes == {"ar", "diffusion", "linear_spec", "linear_spec_lora"}
    assert {case.bundle for case in cases} == {"nemotron-labs-diffusion-8b.trtfb"}
    assert all(case.runtime_strategy == "nemotron_labs_diffusion" for case in cases)
    assert all(
        case.reference_family == "nemotron_labs_diffusion_model_card"
        for case in cases
    )
    assert all(case.user_contract == "model_card_generation_parity" for case in cases)
    assert all(case.metadata["ci_tier"] == "nightly_only" for case in cases)
    assert all(
        case.metadata["contract_config"]["enable_thinking"] is False
        for case in cases
    )
    assert all(
        case.threshold_overrides["canonical_token_agreement_rate"] == 1.0
        for case in cases
    )
