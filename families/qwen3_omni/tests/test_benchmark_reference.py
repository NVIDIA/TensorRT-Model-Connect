# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent


def test_text_performance_reference_is_family_owned() -> None:
    profile = yaml.safe_load(
        (ROOT / "benchmark/qwen3-omni-30b-a3b-instruct.yaml").read_text(encoding="utf-8")
    )
    case = profile["performance"][0]
    assert case["operation"] == "generate"
    assert case["request"]["max_new_tokens"] == 16
    assert case["reference"]["script"] == "tests/benchmark/reference.py"
    assert "adapter" not in case["reference"]

    source = (ROOT / "benchmark/reference.py").read_text(encoding="utf-8")
    assert "enable_audio_output=False" in source
    assert "return_audio=False" in source
