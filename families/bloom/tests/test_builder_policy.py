# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BLOOM-owned TensorRT builder precision policy."""

from pathlib import Path


FAMILY = Path(__file__).resolve().parent.parent
FP32_BUILDERS = (
    FAMILY / "standard_decoder_builder.py",
    FAMILY / "dual_profile_decoder_builder.py",
    FAMILY / "dual_profile_decoder_tp_builder.py",
)


def test_bloom_fp32_builders_disable_tf32() -> None:
    """BLOOM FP32 must not silently use lower-precision TF32 matmuls."""
    missing = [
        path.name
        for path in FP32_BUILDERS
        if "trt_config.clear_flag(trt.BuilderFlag.TF32)" not in path.read_text(encoding="utf-8")
    ]

    assert missing == []
