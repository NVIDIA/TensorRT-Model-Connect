# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-MoE-owned manifest contract tests."""

from __future__ import annotations

from pathlib import Path

from tests.e2e_harness.manifest_loader import load_manifest


def test_tiny_random_qwen3_moe_does_not_infer_an_hf_contract() -> None:
    """Keep the missing HF contract explicit while allowing HF parity."""
    manifest_path = Path(__file__).with_name("manifests") / "qwen3-moe-tiny-random.json"
    case = load_manifest(manifest_path)

    assert case.hf_id == "amd-quark/tiny-random-qwen3_moe"
    assert case.task_strategy == "text_generation_causal"
    assert case.user_contract == ""
    assert case.metadata["single_process_debug_generation"] is True
    assert "skip_comparison" not in case.metadata
    assert "skip_comparison_reason" not in case.metadata
