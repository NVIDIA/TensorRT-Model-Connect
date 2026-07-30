# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static qualification contracts for Cosmos3-Nano SD and CP4 proofs."""

from __future__ import annotations

import json
from pathlib import Path

_MANIFEST_ROOT = Path(__file__).parent / "manifests"
_SD_MANIFEST = _MANIFEST_ROOT / "cosmos3-nano.json"
_CP4_MANIFEST = _MANIFEST_ROOT / "cosmos3-nano-cp4.json"


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _qualified_profile(path: Path) -> dict:
    testcase = _load_manifest(path)["testcases"][0]
    return {
        "video_width": testcase["video_width"],
        "video_height": testcase["video_height"],
        "video_num_frames": testcase["video_num_frames"],
        "num_inference_steps": testcase["num_inference_steps"],
        "guidance_scale": testcase["guidance_scale"],
        "seed": testcase["seed"],
    }


def test_single_device_model_proof_has_no_distributed_configuration() -> None:
    manifest = _load_manifest(_SD_MANIFEST)

    assert "distributed_runtime" not in manifest
    assert "parallel" not in manifest.get("build_args", {})


def test_cp4_build_and_runtime_world_sizes_match() -> None:
    manifest = _load_manifest(_CP4_MANIFEST)
    parallel = manifest["build_args"]["parallel"]
    distributed = manifest["distributed_runtime"]

    assert parallel == {"mode": "context_parallel", "cp_size": 4}
    assert distributed["enabled"] is True
    assert distributed["launcher"] == "mpirun"
    assert distributed["world_size"] == parallel["cp_size"]


def test_sd_and_cp4_model_proofs_use_the_same_qualified_full_profile() -> None:
    expected = {
        "video_width": 1280,
        "video_height": 720,
        "video_num_frames": 189,
        "num_inference_steps": 35,
        "guidance_scale": 6.0,
        "seed": 42,
    }

    assert _qualified_profile(_SD_MANIFEST) == expected
    assert _qualified_profile(_CP4_MANIFEST) == expected
