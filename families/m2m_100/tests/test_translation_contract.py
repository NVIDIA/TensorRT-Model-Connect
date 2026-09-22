# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import yaml


TEST_ROOT = Path(__file__).resolve().parent


def test_qualification_profile_owns_source_language_placement() -> None:
    profile = yaml.safe_load(
        (TEST_ROOT / "benchmark/nllb-200-distilled-600m.yaml").read_text(encoding="utf-8")
    )

    cases = [*profile["accuracy"], *profile["performance"]]
    assert cases
    assert all(
        case["reference"]["source_language_placement"] == "replace-final-unk" for case in cases
    )


def test_nllb_case_keeps_the_explicit_english_to_french_contract() -> None:
    manifest = json.loads((TEST_ROOT / "manifests/nllb-200.json").read_text(encoding="utf-8"))
    case = manifest["testcases"][0]
    assert case["source_language"] == "eng_Latn"
    assert case["source_language_token_id"] == 256047
    assert case["target_language"] == "fra_Latn"
    assert case["forced_bos_token_id"] == 256057

    runner = (TEST_ROOT / "test_e2e.py").read_text(encoding="utf-8")
    assert '"--source-language-token-id"' in runner
    assert '"--forced-bos-token-id"' in runner
    assert "forced_bos_token_id=forced_bos_token_id" in runner
