# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned-checkpoint provenance shared by this family's validation paths."""

import json
from pathlib import Path

from tensorrt_model_connect.provenance import checkpoint_identity


def manifest():
    return json.loads((Path(__file__).parent / "manifests/qwen3-next-bf16.json").read_text())


def assert_pinned_checkpoint(directory, selected=None):
    selected = manifest() if selected is None else selected
    root = Path(directory).resolve()
    if root.parent.name == "snapshots":
        assert checkpoint_identity(root) == {
            "hf_id": selected["hf_id"], "revision": selected["hf_revision"],
        }, "Qwen3-Next validation requires the pinned checkpoint repository and revision"
        return
    for name in ("config.json", "model.safetensors.index.json",
                 *sorted(path.name for path in root.glob("*.safetensors"))):
        receipt = root / ".cache/huggingface/download" / (name + ".metadata")
        assert receipt.is_file(), f"Qwen3-Next checkpoint lacks a pinned receipt: {name}"
        lines = receipt.read_text().splitlines()
        assert lines and lines[0] == selected["hf_revision"], (
            f"Qwen3-Next checkpoint receipt has the wrong revision: {name}"
        )


def assert_bundle_checkpoint(info, selected):
    provenance = info.get("provenance")
    assert isinstance(provenance, dict), "prebuilt Qwen3-Next bundle lacks provenance"
    assert provenance.get("version") == 1, "unsupported Qwen3-Next bundle provenance version"
    assert provenance.get("checkpoint") == {
        "hf_id": selected["hf_id"], "revision": selected["hf_revision"],
    }, "prebuilt Qwen3-Next bundle checkpoint differs from the pinned reference"
