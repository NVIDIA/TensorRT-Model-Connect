"""Qwen3-MoE-owned manifest contract tests."""

from __future__ import annotations

from pathlib import Path

from tests.e2e_harness.manifest_loader import load_manifest


def test_tiny_random_qwen3_moe_is_an_explicit_runtime_smoke_case() -> None:
    """Keep the missing HF contract intentional until HF declares one."""
    manifest_path = Path(__file__).with_name("manifests") / "qwen3-moe-tiny-random.json"
    case = load_manifest(manifest_path)

    assert case.hf_id == "amd-quark/tiny-random-qwen3_moe"
    assert case.task_strategy == "text_generation_causal"
    assert case.user_contract == ""
    assert case.metadata["single_process_debug_generation"] is True
    skip_reason = case.metadata["skip_comparison_reason"]
    assert "runtime smoke test" in skip_reason
    assert "HF text parity is not meaningful" in skip_reason
