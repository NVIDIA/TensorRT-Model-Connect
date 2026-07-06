# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-owned manifest contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e_harness.manifest_loader import load_manifest


@pytest.mark.parametrize(
    "manifest_name",
    ["qwen3-0.6b-fp8.json", "qwen3-0.6b-fp8-tp4.json"],
)
def test_qwen3_fp8_manifest_declares_hf_text_generation_contract(
    manifest_name: str,
) -> None:
    manifest_path = Path(__file__).with_name("manifests") / manifest_name
    case = load_manifest(manifest_path)

    assert case.hf_id == "Qwen/Qwen3-0.6B"
    assert case.task_strategy == "text_generation_causal"
    assert case.user_contract == "text-generation"
    assert not case.metadata.get("skip_reason")
