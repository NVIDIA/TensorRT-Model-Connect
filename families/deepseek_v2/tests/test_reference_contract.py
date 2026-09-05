# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path


MANIFESTS = Path(__file__).with_name("manifests")


def test_tiny_reference_uses_batched_expert_matmuls() -> None:
    for name in ("deepseek-v2-tiny.json", "deepseek-v2-tiny-tp2.json"):
        manifest = json.loads((MANIFESTS / name).read_text(encoding="utf-8"))
        case = manifest["testcases"][0]
        assert case["reference_experts_implementation"] == "batched_mm"
