# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned CI build contracts for Llama models."""

from __future__ import annotations

import json
from pathlib import Path

from tests.e2e_harness.manifest_loader import load_manifest


def test_falcon3_split_decoder_build_reserves_an_exclusive_gpu() -> None:
    manifest_path = Path(__file__).parent / "manifests" / "falcon3-1b.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["e2e_parallel_resource"] == "exclusive_gpu"


def test_native_minitron_manifests_use_family_build_defaults() -> None:
    for manifest_name in (
        "minitron-4b-width.json",
        "minitron-4b-width-l0.json",
        "minitron-4b-width-regression-native-kv-chunked-prefill.json",
    ):
        manifest_path = Path(__file__).parent / "manifests" / manifest_name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        case = load_manifest(manifest_path)

        assert "precision" not in manifest, manifest_name
        assert "max_cache_length" not in manifest, manifest_name
        assert "precision" not in case.metadata, manifest_name
        assert "max_cache_length" not in case.inputs, manifest_name


def test_native_minitron_regression_exceeds_one_prefill_profile() -> None:
    manifest_path = (
        Path(__file__).parent
        / "manifests"
        / "minitron-4b-width-regression-native-kv-chunked-prefill.json"
    )
    case = load_manifest(manifest_path)

    assert case.hf_revision == "5205ef7d36204947e3b973cb8b147a816ccd7e6a"
    assert case.metadata["test_category"] == "regression"
    assert case.metadata["ci_tier"] == "default"
    assert case.reference_backend == "invariant_only"
    assert case.oracle_level == "L4_invariants"
    assert case.reference_family == "llama_native_kv_chunked_prefill_regression"
    assert case.user_contract == "runtime_invariants"
    assert case.inputs["prompt_repeat"] == {
        "text": "a",
        "separator": " ",
        "count": 32768,
        "suffix": "\n",
    }
    assert case.inputs["expected_prompt_tokens"] == 32769
    assert case.metadata["expected_kv_cache_rows"] == 131072
    assert case.metadata["expected_prefill_chunks"] == 2
    assert case.metadata["expected_prefill_chunk_limit"] == 32768
    assert case.inputs["max_new_tokens"] == 2


def test_tinyllama_keeps_legacy_build_contract() -> None:
    manifest_path = (
        Path(__file__).parent / "manifests" / "tinyllama-1.1b.json"
    )
    case = load_manifest(manifest_path)

    assert case.metadata["precision"] == "fp16"
    assert case.inputs["max_cache_length"] == 256
