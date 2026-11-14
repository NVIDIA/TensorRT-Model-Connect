# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Both reference setup entry points reuse the pinned checkpoint in the HF cache."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import shutil

import huggingface_hub
import pytest


@pytest.mark.parametrize("module", [
    "families.fast_foundation_stereo.reference.prepare",
    "families.fast_foundation_stereo.tests.benchmark.prepare_environment",
])
def test_preparation_reuses_checkpoint_and_preserves_it_on_environment_removal(
    module, monkeypatch, tmp_path,
):
    prepare = import_module(module)
    cache = tmp_path / "cache"
    blob = cache / "blobs/weights"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"checkpoint")
    checkpoint = cache / "snapshots/revision/model.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.symlink_to("../../blobs/weights")
    environment = tmp_path / "environment"
    downloads, checkouts = [], []

    def download(**kwargs):
        downloads.append(kwargs)
        return str(checkpoint)

    def run(command, **_kwargs):
        if command[0] == "git" and "checkout" in command:
            root = Path(command[2])
            (root / "core").mkdir()
            for name in ("foundation_stereo.py", "submodule.py"):
                (root / "core" / name).write_text("# source\n")
            checkouts.append(root)

    monkeypatch.setattr(prepare.sys, "prefix", str(environment))
    monkeypatch.setattr(prepare.subprocess, "run", run)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    if hasattr(prepare, "hf_hub_download"):
        monkeypatch.setattr(prepare, "hf_hub_download", download)

    assert prepare.main() == 0
    target = environment / "trtmc-reference/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth"
    assert target.is_symlink()
    assert target.samefile(blob)
    assert target.read_bytes() == b"checkpoint"
    assert downloads == [{
        "repo_id": "nvidia/c-fast-foundationstereo",
        "revision": "9b446878c81ddb27593036767b29b2859d46103e",
        "filename": "model_best_bp2_serialize.pth",
    }]
    assert prepare.main() == 0
    assert len(downloads) == len(checkouts) == 1
    shutil.rmtree(environment)
    assert blob.read_bytes() == b"checkpoint"
