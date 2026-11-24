# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ImageNet gate of the FP8 ResNet-50 profile."""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

AIPERF = Path(__file__).resolve().parents[2] / "config" / "models"  # the profiles' AIPerf settings


def test_the_fp8_resnet50_gets_twice_the_imagenet_margin() -> None:
    """Two FP8 builds differ by up to 0.13 top-1 points, so the quantized bundle is held to 1 point (the shared
    ImageNet margin is 0.5); the fp16 path keeps the shared gate."""
    benchmarks = yaml.safe_load((AIPERF / "resnet50-a1-in1k-fp8.yaml").read_text())["benchmarks"]
    assert benchmarks == {"imagenet": {"quantized_gate": {"margin": 1.0}}}
