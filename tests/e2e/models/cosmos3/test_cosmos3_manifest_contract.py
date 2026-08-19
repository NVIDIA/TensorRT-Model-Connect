# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static qualification contract for the Cosmos3-Nano CP4 proof."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.e2e_harness.manifest_loader import load_manifest


MODEL_ID = "nvidia/Cosmos3-Nano"
MODEL_REVISION = "411f42a8fdfb8c5b2583cb8786e0938f49796eaa"


@pytest.mark.parametrize(
    "filename",
    ("cosmos3-nano-l0.json", "cosmos3-nano-l0-cp4.json"),
)
def test_cosmos3_manifests_declare_public_pinned_checkpoint(filename: str) -> None:
    manifest_path = Path(__file__).with_name("manifests") / filename
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = load_manifest(manifest_path)

    assert raw["hf_id"] == MODEL_ID
    assert raw["hf_revision"] == MODEL_REVISION
    assert "gated" not in raw
    assert case.hf_id == MODEL_ID
    assert case.hf_revision == MODEL_REVISION
    assert all(item.kind != "hf_auth_token_present" for item in case.preflight)


def test_cosmos3_cp4_manifest_declares_context_parallel_runtime() -> None:
    manifest_path = Path(__file__).with_name("manifests") / "cosmos3-nano-l0-cp4.json"
    case = load_manifest(manifest_path)
    parallel = case.metadata["build_args"]["parallel"]
    distributed = case.metadata["distributed_runtime"]

    assert parallel == {"mode": "context_parallel", "cp_size": 4}
    assert distributed["enabled"] is True
    assert distributed["launcher"] == "mpirun"
    assert distributed["world_size"] == parallel["cp_size"]
