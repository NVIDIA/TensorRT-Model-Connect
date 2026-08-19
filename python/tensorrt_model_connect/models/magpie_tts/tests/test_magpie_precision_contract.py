# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precision regression coverage for the long-form Magpie TTS contract."""

from __future__ import annotations

import json
from pathlib import Path


def test_long_form_magpie_build_keeps_the_complete_pipeline_in_fp32() -> None:
    manifest = json.loads(
        (Path(__file__).parent / "manifests" / "magpie-tts-357m.json").read_text()
    )

    assert manifest["precision"] == "fp32"
    assert "fp32_layers" not in manifest

    long_form = next(
        testcase
        for testcase in manifest["testcases"]
        if testcase["name"] == "magpie-tts-357m"
    )
    assert long_form["max_new_tokens"] == 750
