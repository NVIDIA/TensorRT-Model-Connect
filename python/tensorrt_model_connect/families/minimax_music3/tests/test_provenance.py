# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MiniMax-Music3 pinned sources."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

provenance = importlib.import_module(
    "tensorrt_model_connect.families.minimax_music3.provenance"
)

_FAMILY_DIR = Path(provenance.__file__).parent
_LOCK = (
    _FAMILY_DIR
    / "python_profile_requirements"
    / "minimax_music3_reference.lock.txt"
)


def test_checkpoint_revision_is_a_full_git_sha() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", provenance.CHECKPOINT_REVISION)
    assert provenance.CHECKPOINT_REPOSITORY == "MiniMaxAI/MiniMax-Music3"


def test_upstream_commit_is_recorded_but_not_installed() -> None:
    """The model card pins a pull-request commit; the merged release is used."""

    assert re.fullmatch(r"[0-9a-f]{40}", provenance.DIFFUSERS_UPSTREAM_COMMIT)
    assert provenance.DIFFUSERS_UPSTREAM_PULL_REQUEST == 14456
    assert provenance.DIFFUSERS_UPSTREAM_COMMIT not in provenance.reference_requirement()


def test_reference_requirement_pins_an_exact_release() -> None:
    assert provenance.reference_requirement() == "diffusers==0.40.0"


def test_reference_lock_file_matches_the_pin() -> None:
    lines = [
        line.strip()
        for line in _LOCK.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert lines == [provenance.reference_requirement()]


def test_reference_modules_cover_the_three_pipeline_pieces() -> None:
    modules = provenance.DIFFUSERS_REFERENCE_MODULES

    assert any(m.endswith("transformer_minimax_music3") for m in modules)
    assert any(m.endswith("minimax_music3_rvq_depth_decoder") for m in modules)
    assert any("modular_pipelines.minimax_music3" in m for m in modules)


def test_required_snapshot_covers_every_declared_component() -> None:
    plugin_mod = importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.plugin"
    )

    for component in plugin_mod.REQUIRED_COMPONENTS:
        assert any(
            name.startswith(f"{component}/")
            for name in provenance.REQUIRED_SNAPSHOT_FILES
        ), component


def test_required_snapshot_excludes_the_unreferenced_weights() -> None:
    joined = " ".join(provenance.REQUIRED_SNAPSHOT_FILES)

    assert "qwen_7B" not in joined
    assert "flowmatching_vae.pth" not in joined
    assert "dav.pth" not in joined


def test_missing_snapshot_files_reports_gaps() -> None:
    complete = set(provenance.REQUIRED_SNAPSHOT_FILES)

    assert provenance.missing_snapshot_files(complete) == ()

    partial = complete - {"vocoder/config.json"}
    assert provenance.missing_snapshot_files(partial) == ("vocoder/config.json",)


def test_reference_call_matches_the_documented_example() -> None:
    call = provenance.REFERENCE_CALL

    assert call["audio_duration_seconds"] == 60.0
    assert call["seed"] == 7
    assert call["dtype"] == "bfloat16"
