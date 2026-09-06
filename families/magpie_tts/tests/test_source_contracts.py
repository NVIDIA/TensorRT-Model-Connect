# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Magpie-TTS pinned checkpoint and precision contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from families.magpie_tts.model import _validate_supported_checkpoint_architecture


def test_latest_upstream_architecture_fails_with_actionable_revision_error() -> None:
    latest_checkpoint = {
        **{f"audio_embeddings.{index}.weight": object() for index in range(16)},
    }

    with pytest.raises(
        ValueError,
        match=r"supports 8 codebooks.*hf_revision",
    ):
        _validate_supported_checkpoint_architecture(latest_checkpoint)


def test_pinned_checkpoint_architecture_is_supported() -> None:
    supported_checkpoint = {
        **{f"audio_embeddings.{index}.weight": object() for index in range(8)},
    }

    _validate_supported_checkpoint_architecture(supported_checkpoint)


def test_long_form_build_keeps_the_complete_pipeline_in_fp32() -> None:
    manifest = json.loads(
        (Path(__file__).parent / "manifests" / "magpie-tts-357m.json").read_text(encoding="utf-8")
    )

    assert manifest["precision"] == "fp32"
    assert "fp32_layers" not in manifest
    long_form = next(
        testcase for testcase in manifest["testcases"] if testcase["name"] == "magpie-tts-357m"
    )
    assert long_form["max_new_tokens"] == 750
