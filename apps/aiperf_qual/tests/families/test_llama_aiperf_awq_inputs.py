# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPOSITORY = Path(__file__).resolve().parents[4]


def test_the_profile_declares_what_the_preparation_reads_as_its_inputs():
    profile = yaml.safe_load(
        (REPOSITORY / "apps/aiperf_qual/config/models/nemotron-nano-4b-int4-awq.yaml").read_text()
    )
    inputs = profile["reference"]["prepare_inputs"]
    # A change to the quantizer, the committed scales or a manifest must prepare the reference weights again: the
    # family's quantization module, its AWQ_SCALES_DIR and the manifests prepare_int4_awq reads.
    assert inputs == [
        "families/llama/quantization.py",
        "families/llama/awq_scales",
        "families/llama/tests/manifests",
    ]
    assert all((REPOSITORY / path).exists() for path in inputs)
