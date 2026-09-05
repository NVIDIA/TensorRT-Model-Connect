# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned CI build contracts for SmolLM3 models."""

from __future__ import annotations

import json
from pathlib import Path

from tests.e2e_harness.manifest_loader import load_manifest

MANIFEST = Path(__file__).parent / "manifests" / "smollm3-3b.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_pins_an_immutable_revision() -> None:
    assert load_manifest(MANIFEST).hf_revision == (
        "a07cc9a04f16550a088caea529712d1d335b0ac1"
    )


def test_manifest_declares_the_family_runtime_contract() -> None:
    manifest = _manifest()
    assert manifest["family"] == "smollm3"
    assert manifest["runtime_strategy"] == "smollm3_decoder_kv_cache"
    assert manifest["task_strategy"] == "text_generation_causal"


def test_manifest_builds_bf16_for_the_native_kv_path() -> None:
    # build_routing accepts only BF16 for the native KV decoder, and the
    # plugin's default_build_precision returns bf16 once the architecture
    # qualifies, so the declared precision has to agree with both.
    manifest = _manifest()
    assert manifest["precision"] == "bf16"
    assert load_manifest(MANIFEST).metadata["precision"] == "bf16"


def test_manifest_reserves_an_exclusive_gpu() -> None:
    assert _manifest()["e2e_parallel_resource"] == "exclusive_gpu"


def test_manifest_uses_the_hf_transformers_oracle() -> None:
    case = load_manifest(MANIFEST)
    assert case.reference_backend == "hf_transformers"
    assert case.oracle_level == "L1_external_reference"
    assert case.reference_family == "causal_base_continuation"
    assert case.user_contract == "continuation_parity"


def test_manifest_needs_no_remote_code() -> None:
    # SmolLM3 is native in transformers; the bundle must not depend on
    # trust_remote_code.
    assert _manifest()["trust_remote_code"] is False
