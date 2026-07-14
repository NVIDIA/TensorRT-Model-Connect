# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image-owned manifest contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e_harness.manifest_loader import load_manifest


@pytest.mark.parametrize(
    ("manifest_name", "hf_id", "user_contract", "task_mode"),
    [
        ("qwen-image-l0.json", "Qwen/Qwen-Image", "diffusion_image", None),
        ("qwen-image.json", "Qwen/Qwen-Image", "text-to-image", None),
        ("qwen-image-2512.json", "Qwen/Qwen-Image-2512", "text-to-image", None),
        (
            "qwen-image-edit-2511.json",
            "Qwen/Qwen-Image-Edit-2511",
            "image-to-image",
            "edit",
        ),
    ],
)
def test_qwen_image_manifest_declares_hf_image_contract(
    manifest_name: str,
    hf_id: str,
    user_contract: str,
    task_mode: str | None,
) -> None:
    manifest_path = Path(__file__).with_name("manifests") / manifest_name
    case = load_manifest(manifest_path)

    assert case.hf_id == hf_id
    assert case.task_strategy == "diffusion_media_generation"
    assert case.user_contract == user_contract
    assert case.metadata.get("task_mode") == task_mode


def test_qwen_image_nightly_defaults_run_end_to_end_contract_once() -> None:
    """Nightly Qwen-Image cases must not repeat the full TRT and HF pipelines."""
    manifests = Path(__file__).with_name("manifests")

    for manifest_name in (
        "qwen-image.json",
        "qwen-image-2512.json",
        "qwen-image-edit-2511.json",
    ):
        case = load_manifest(manifests / manifest_name)

        assert case.task_strategy == "diffusion_media_generation"
        assert [stage.name for stage in case.stages] == ["end_to_end"]
        assert all(stage.required for stage in case.stages)


def test_qwen_image_l0_keeps_nightly_quality_gate_at_reduced_scale() -> None:
    manifests = Path(__file__).with_name("manifests")
    l0_case = load_manifest(manifests / "qwen-image-l0.json")
    nightly_case = load_manifest(manifests / "qwen-image.json")

    assert l0_case.metadata["ci_tier"] == "l0_only"
    assert l0_case.inputs["image_height"] == 512
    assert l0_case.inputs["image_width"] == 512
    assert l0_case.inputs["num_inference_steps"] == 20
    assert l0_case.threshold_overrides == nightly_case.threshold_overrides
