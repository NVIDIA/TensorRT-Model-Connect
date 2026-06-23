"""InternLM-owned E2E manifest profile tests."""

from __future__ import annotations

import json

from tests.e2e_harness.manifest_loader import load_manifest


def _write_manifest(tmp_path, data: dict) -> str:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_load_manifest_applies_internlm_default_execution_profiles(tmp_path) -> None:
    path = _write_manifest(
        tmp_path,
        {
            "name": "internlm-case",
            "hf_id": "internlm/internlm-test",
            "family": "internlm",
            "runtime_strategy": "decoder_kv_cache",
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
            "runtime_strategy": "decoder_kv_cache",
            "reference_backend": "torch_reference",
            "execution_profiles": {"runtime": "custom-runtime"},
        },
    )
    case = load_manifest(path)

    assert case.execution_profiles["build"] == "internlm"
    assert case.execution_profiles["runtime"] == "custom-runtime"
    assert case.execution_profiles["reference"] == "internlm"
