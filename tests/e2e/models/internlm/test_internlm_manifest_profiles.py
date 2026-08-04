# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InternLM-owned E2E manifest profile tests."""

from __future__ import annotations

import json
from pathlib import Path

from tests.e2e_harness.manifest_loader import load_manifest


_MANIFEST_DIR = Path(__file__).with_name("manifests")
_FAMILY_DIR = (
    Path(__file__).resolve().parents[4]
    / "python/tensorrt_model_connect/families/internlm"
)


def _write_manifest(tmp_path, data: dict) -> str:
    path = tmp_path / "manifest.json"
    model_fields = {
        key: data[key] for key in ("name", "hf_id", "family", "runtime_strategy") if key in data
    }
    testcase = {
        key: value for key, value in data.items() if key not in model_fields or key == "name"
    }
    model_fields["testcases"] = [testcase]
    path.write_text(json.dumps(model_fields), encoding="utf-8")
    return str(path)


def test_load_manifest_applies_internlm_default_execution_profiles(tmp_path) -> None:
    path = _write_manifest(
        tmp_path,
        {
            "name": "internlm-case",
            "hf_id": "internlm/internlm-test",
            "family": "internlm",
            "runtime_strategy": "internlm_decoder_kv_cache",
            "reference_backend": "torch_reference",
        },
    )
    case = load_manifest(path)

    assert case.execution_profiles["build"] == "internlm"
    assert case.execution_profiles["runtime"] == "internlm"
    assert case.execution_profiles["reference"] == "internlm"


def test_load_manifest_preserves_internlm_execution_profile_overrides(tmp_path) -> None:
    path = _write_manifest(
        tmp_path,
        {
            "name": "internlm-case",
            "hf_id": "internlm/internlm-test",
            "family": "internlm",
            "runtime_strategy": "internlm_decoder_kv_cache",
            "reference_backend": "torch_reference",
            "execution_profiles": {"runtime": "custom-runtime"},
        },
    )
    case = load_manifest(path)

    assert case.execution_profiles["build"] == "internlm"
    assert case.execution_profiles["runtime"] == "custom-runtime"
    assert case.execution_profiles["reference"] == "internlm"


def test_internlm_builds_reserve_an_exclusive_gpu() -> None:
    for manifest_name in ("internlm2-1.8b.json", "internlm2-1.8b-tp4.json"):
        manifest = json.loads(
            (_MANIFEST_DIR / manifest_name).read_text(encoding="utf-8"))

        assert manifest["e2e_parallel_resource"] == "exclusive_gpu"


def test_reference_profile_supports_current_internlm_dynamic_cache_api() -> None:
    lock = (
        _FAMILY_DIR / "python_profile_requirements/internlm.lock.txt"
    ).read_text(encoding="utf-8")
    verify = (_FAMILY_DIR / "python_profile_verify.py").read_text(encoding="utf-8")

    assert "huggingface-hub==0.26.5" in lock
    assert "tokenizers==0.20.3" in lock
    assert "transformers==4.46.3" in lock
    assert "protobuf==" not in lock
    assert "sentencepiece==" not in lock
    assert "import google.protobuf" not in verify
    assert "import sentencepiece" not in verify
    assert 'hasattr(DynamicCache, "get_max_cache_shape")' in verify
