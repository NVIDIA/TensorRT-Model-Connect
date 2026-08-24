# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualification configuration contracts for Mistral generation."""

import json
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
MANIFESTS = Path(__file__).with_name("manifests")


def test_mistral_7b_manifests_use_one_dual_profile_engine() -> None:
    for name in ("mistral-7b.json", "mistral-7b-l0.json"):
        manifest = json.loads((MANIFESTS / name).read_text(encoding="utf-8"))
        assert manifest["build_args"]["decoder_engine_layout"] == "dual_profile"


def test_release_case_enables_gpu_greedy_with_aligned_fp16_reference() -> None:
    release = yaml.safe_load(
        (REPOSITORY_ROOT / "benchmarks" / "performance" / "release.yaml").read_text(
            encoding="utf-8"
        )
    )
    entry = next(item for item in release["entries"] if item["id"] == "mistral.generate")

    assert entry["workload"]["runtime"]["config"] == {"runtime.prefer_gpu_greedy": True}
    assert entry["baseline"]["precision"] == "fp16"
