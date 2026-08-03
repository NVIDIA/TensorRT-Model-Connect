# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static qualification contract for the Cosmos3-Nano CP4 proof."""

from __future__ import annotations

from pathlib import Path

from tests.e2e_harness.manifest_loader import load_manifest


def test_cosmos3_cp4_manifest_declares_context_parallel_runtime() -> None:
    manifest_path = Path(__file__).with_name("manifests") / "cosmos3-nano-l0-cp4.json"
    case = load_manifest(manifest_path)
    parallel = case.metadata["build_args"]["parallel"]
    distributed = case.metadata["distributed_runtime"]

    assert parallel == {"mode": "context_parallel", "cp_size": 4}
    assert distributed["enabled"] is True
    assert distributed["launcher"] == "mpirun"
    assert distributed["world_size"] == parallel["cp_size"]
