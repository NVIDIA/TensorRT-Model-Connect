# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused checks for the family-owned StableLM TensorRT build policy."""

from pathlib import Path


FAMILY_ROOT = Path(__file__).resolve().parent.parent


def test_stablelm_builders_preserve_mainline_optimization_level() -> None:
    policies = (
        (
            "utils.py",
            "config = builder.create_builder_config()",
            "config.builder_optimization_level = 3",
        ),
        (
            "dual_profile_decoder_tp_builder.py",
            "trt_config = builder.create_builder_config()",
            "trt_config.builder_optimization_level = 3",
        ),
    )

    for filename, creation, policy in policies:
        source = (FAMILY_ROOT / filename).read_text(encoding="utf-8")
        assert source.count(creation) == 1
        assert source.count(policy) == 1
