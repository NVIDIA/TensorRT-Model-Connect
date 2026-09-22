# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent


def test_performance_reference_is_family_owned() -> None:
    profile = yaml.safe_load(
        (ROOT / "benchmark/locateanything-3b.yaml").read_text(encoding="utf-8")
    )
    reference = profile["performance"][0]["reference"]
    assert reference["script"] == "tests/benchmark/reference.py"
    assert reference["output_contract"] == "localization"
    assert "adapter" not in reference
