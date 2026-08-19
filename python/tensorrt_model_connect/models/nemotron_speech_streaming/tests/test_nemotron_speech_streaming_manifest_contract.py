# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron Speech Streaming-owned manifest contract tests."""

from __future__ import annotations

from pathlib import Path

from tests.e2e_harness.manifest_loader import load_manifest


def test_nemotron_35_asr_manifest_declares_hf_asr_contract() -> None:
    manifest_path = (
        Path(__file__).with_name("manifests")
        / "nemotron-3.5-asr-streaming-0.6b.json"
    )
    case = load_manifest(manifest_path)

    assert case.hf_id == "nvidia/nemotron-3.5-asr-streaming-0.6b"
    assert case.task_strategy == "speech_to_text"
    assert case.user_contract == "automatic-speech-recognition"
