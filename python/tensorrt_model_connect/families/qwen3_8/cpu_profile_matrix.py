# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU profile matrix defaults owned by the Qwen3.8 family."""

from __future__ import annotations


def cpu_profile_matrix_specs() -> list[dict]:
    return [{
        "order": 51,
        "strategy": "qwen3_8_hybrid_mamba_attention",
        "label": "hybrid_mamba_attn\n(qwen38-27b)",
        "hf_id": "Qwen/Qwen3.8-27B",
        "bundle": "qwen38-27b.bundle",
        "runner": "decoder",
    }]
