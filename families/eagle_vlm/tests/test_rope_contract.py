# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eagle VLM checkpoint RoPE schema contracts."""

from __future__ import annotations

from families.eagle_vlm import model


class _Config:
    raw: dict = {}


def test_eagle_vlm_resolves_nested_legacy_rope_scaling() -> None:
    class Config(_Config):
        raw = {
            "llm_config": {
                "rope_scaling": {
                    "rope_type": "llama3",
                    "factor": 32.0,
                },
            },
        }

    assert model._resolve_rope_scaling(Config()) == {
        "rope_type": "llama3",
        "factor": 32.0,
    }


def test_eagle_vlm_prefers_rope_parameters_over_legacy_alias() -> None:
    class Config(_Config):
        raw = {
            "llm_config": {
                "rope_parameters": {"rope_type": "llama3", "factor": 8.0},
                "rope_scaling": {"rope_type": "llama3", "factor": 32.0},
            },
        }

    assert model._resolve_rope_scaling(Config())["factor"] == 8.0
