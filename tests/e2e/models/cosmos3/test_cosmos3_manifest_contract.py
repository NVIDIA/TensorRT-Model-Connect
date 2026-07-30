# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static qualification contract for the Cosmos3-Nano CP4 model proof."""

from __future__ import annotations

import json
from pathlib import Path

_MANIFEST = Path(__file__).parent / "manifests" / "cosmos3-nano-cp4.json"


def _load_manifest() -> dict:
    return json.loads(_MANIFEST.read_text(encoding="utf-8"))


def test_cp4_build_and_runtime_world_sizes_match() -> None:
    manifest = _load_manifest()
    parallel = manifest["build_args"]["parallel"]
    distributed = manifest["distributed_runtime"]

    assert parallel == {"mode": "context_parallel", "cp_size": 4}
    assert distributed["enabled"] is True
    assert distributed["launcher"] == "mpirun"
    assert distributed["world_size"] == parallel["cp_size"]


def test_cp4_model_proof_uses_the_qualified_full_profile() -> None:
    testcase = _load_manifest()["testcases"][0]

    assert testcase["video_width"] == 1280
    assert testcase["video_height"] == 720
    assert testcase["video_num_frames"] == 189
    assert testcase["num_inference_steps"] == 35
    assert testcase["guidance_scale"] == 6.0
    assert testcase["seed"] == 42
