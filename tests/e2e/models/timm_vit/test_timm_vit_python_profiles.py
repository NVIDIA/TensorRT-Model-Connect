# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm ViT-owned Python profile contracts."""

from pathlib import Path

from tests.e2e_harness.contracts import E2ECase
from tests.e2e_harness.python_profiles import resolve_case_profile_names


def _make_case() -> E2ECase:
    return E2ECase(
        name="timm-vit-case",
        hf_id="timm/example",
        family="timm_vit",
        runtime_strategy="timm_vit_image_classification",
        task_strategy="image_classification",
        bundle="timm-vit-case.bundle",
        inputs={"image_path": "data/test_img.jpeg"},
        stages=[],
        reference_backend="hf_transformers",
    )


def test_timm_vit_family_owns_a_pinned_reference_profile() -> None:
    repository = Path(__file__).resolve().parents[4]
    requirements = (
        repository
        / "python/tensorrt_model_connect/families/timm_vit/python_profile_requirements"
        / "timm_reference.lock.txt"
    ).read_text(encoding="utf-8")

    assert "timm==1.0.28" in requirements
    assert resolve_case_profile_names(_make_case())["reference"] == "timm_reference"
