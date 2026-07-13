# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned CI build contract tests for Bark."""

from __future__ import annotations

import json
from pathlib import Path


def test_acceptance_build_reserves_gpu_for_stable_tactic_selection() -> None:
    manifest_path = Path(__file__).parent / "manifests" / "bark-small.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["e2e_parallel_resource"] == "exclusive_gpu"


def test_acceptance_build_records_gb300_timing_cache_artifact() -> None:
    manifest_path = Path(__file__).parent / "manifests" / "bark-small.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    build_env = manifest["build_env"]

    assert manifest["precision"] == "fp16"
    assert build_env["TRTMC_BARK_TIMING_CACHE_MODE"] == "record"
    assert (
        build_env["TRTMC_BARK_TIMING_CACHE_PATH"]
        == "/artifacts/bark-small-gb300-trt11.2-fp16.cache"
    )
    assert "TRTMC_BARK_TIMING_CACHE_SHA256" not in build_env
    assert "TRTMC_BUILDER_OPTIMIZATION_LEVEL" not in build_env
