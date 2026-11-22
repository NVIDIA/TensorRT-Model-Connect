# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from trtmc_benchmark.task_adapters import resolve_task_case


@pytest.mark.parametrize("profile", ["edge", "lite", "nano", "mini", "flash"])
def test_d3_text_image_manifest_passes_image_to_worker(profile):
    tests = Path(__file__).resolve().parent
    manifest = json.loads((tests / f"manifests/d3-{profile}.json").read_text())
    case = manifest["testcases"][0]
    request = resolve_task_case("structured_decision", case, tests).request["requests"][0]
    assert request["image_paths"] == [str(tests / "fixtures/red.png")]
