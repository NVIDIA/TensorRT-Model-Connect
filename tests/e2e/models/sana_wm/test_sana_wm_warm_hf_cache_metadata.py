# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SANA-WM-owned HF cache warm dependency metadata tests."""

from __future__ import annotations

from tensorrt_model_connect.families import family_hf_warm_dependencies


def test_sana_wm_stage1_text_encoder_dependency_is_family_owned() -> None:
    dependencies = dict(family_hf_warm_dependencies("sana_wm"))

    assert (
        dependencies["stage-1-text-encoder"]
        == "Efficient-Large-Model/gemma-2-2b-it"
    )
