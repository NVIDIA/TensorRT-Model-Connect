# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SANA-WM-owned HF cache warm dependency metadata tests."""

from __future__ import annotations

from tensorrt_model_connect.models import (
    family_hf_warm_dependencies,
    family_hf_warm_files,
)


def test_sana_wm_stage1_text_encoder_dependency_is_family_owned() -> None:
    dependencies = dict(family_hf_warm_dependencies("sana_wm"))

    assert (
        dependencies["stage-1-text-encoder"]
        == "Efficient-Large-Model/gemma-2-2b-it"
    )


def test_sana_wm_official_reference_license_is_warmed() -> None:
    files = family_hf_warm_files("sana_wm")

    assert (
        "official-reference-license",
        "Efficient-Large-Model/SANA-WM_bidirectional",
        "LICENSE",
    ) in files
