# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualification configuration contracts for Mistral generation."""

import json
from pathlib import Path

MANIFESTS = Path(__file__).with_name("manifests")


def test_mistral_7b_manifests_use_one_dual_profile_engine() -> None:
    for name in ("mistral-7b.json", "mistral-7b-l0.json"):
        manifest = json.loads((MANIFESTS / name).read_text(encoding="utf-8"))
        assert manifest["build_args"]["decoder_engine_layout"] == "dual_profile"
