# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron Labs Diffusion-owned HF cache warm metadata tests."""

from __future__ import annotations

from tensorrt_model_connect.families import family_hf_required_files_by_id


def test_nemotron_labs_diffusion_required_files_are_family_owned() -> None:
    required = family_hf_required_files_by_id()

    assert sorted(required["nvidia/Nemotron-Labs-Diffusion-8B"]) == [
        "linear_spec_lora/adapter_config.json",
        "linear_spec_lora/adapter_model.safetensors",
    ]
