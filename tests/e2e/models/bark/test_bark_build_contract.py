# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned CI build contract tests for Bark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_acceptance_build_reserves_gpu_for_stable_tactic_selection() -> None:
    manifest_path = Path(__file__).parent / "manifests" / "bark-small.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["e2e_parallel_resource"] == "exclusive_gpu"


def test_acceptance_build_uses_verified_gb300_timing_cache() -> None:
    model_dir = Path(__file__).parent
    manifest_path = model_dir / "manifests" / "bark-small.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    build_env = manifest["build_env"]
    cache_spec = build_env["TRTMC_BARK_TIMING_CACHE_PATH"]
    cache_path = model_dir / cache_spec["path"]
    cache_digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()

    assert manifest["precision"] == "fp16"
    assert build_env["TRTMC_BARK_TIMING_CACHE_MODE"] == "verified"
    assert cache_path.name == "bark-small-gb300-trt11.2-cuda13.3-fp16.cache"
    assert cache_spec["relative_to"] == "model"
    assert cache_digest == build_env["TRTMC_BARK_TIMING_CACHE_SHA256"]
    assert "TRTMC_BUILDER_OPTIMIZATION_LEVEL" not in build_env
