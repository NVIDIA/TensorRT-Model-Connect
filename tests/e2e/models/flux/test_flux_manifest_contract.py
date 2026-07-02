# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flux-owned manifest contract tests."""

from __future__ import annotations

from pathlib import Path

from tests.e2e_harness.manifest_loader import load_manifest


def test_flux2_fp8_manifest_uses_end_to_end_image_contract() -> None:
    """FLUX.2 FP8 should not inherit unrelated optional debug substages."""
    manifest_path = Path(__file__).with_name("manifests") / "flux-2-dev-fp8.json"
    case = load_manifest(manifest_path)

    assert case.reference_family == "diffusers_image_gen"
    assert case.user_contract == "diffusion_image"
    assert [stage.name for stage in case.stages] == ["end_to_end"]
    assert all(stage.required for stage in case.stages)
    assert "Wan-specific" in case.metadata["notes"]
