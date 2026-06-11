"""Tests for E2E Python profile resolution and phase-specific interpreter use."""

from __future__ import annotations

import sys
from pathlib import Path

import tensorrt_model_connect.python_profiles as shared_profiles
from tests.e2e_harness.contracts import E2ECase, RunContext
from tests.e2e_harness.orchestrator import _build_repro_commands
from tests.e2e_harness.python_profiles import (
    profile_env_var,
    resolve_case_profile_names,
    resolve_case_python_profiles,
)


def _make_case(runtime_strategy: str = "decoder_kv_cache", **kwargs) -> E2ECase:
    defaults = dict(
        name="case-a",
        hf_id="dummy/model",
        family="dummy",
        runtime_strategy=runtime_strategy,
        task_strategy=kwargs.pop("task_strategy", runtime_strategy),
        bundle="case-a.trtfb",
        inputs=kwargs.pop("inputs", {}),
        stages=[],
    )
    defaults.update(kwargs)
    return E2ECase(**defaults)


def test_resolve_case_python_profiles_defaults_to_base():
    case = _make_case()
    profiles = resolve_case_python_profiles(case, "/usr/bin/python3")
    assert profiles == {
        "build": "/usr/bin/python3",
        "runtime": "/usr/bin/python3",
        "reference": "/usr/bin/python3",
    }


def test_resolve_case_profile_names_apply_family_defaults():
    case = _make_case(
        family="internlm",
        runtime_strategy="decoder_kv_cache",
        reference_backend="torch_reference",
    )
    assert resolve_case_profile_names(case) == {
        "build": "internlm",
        "runtime": "internlm",
        "reference": "internlm",
    }


def test_resolve_case_python_profiles_uses_family_default_named_env(monkeypatch, tmp_path):
    wrapper = tmp_path / "internlm-python"
    wrapper.write_text("", encoding="utf-8")
    monkeypatch.setenv(profile_env_var("internlm"), str(wrapper))
    case = _make_case(
        family="internlm",
        runtime_strategy="decoder_kv_cache",
        reference_backend="torch_reference",
    )
    profiles = resolve_case_python_profiles(case, "/usr/bin/python3")
    assert profiles["build"] == str(wrapper)
    assert profiles["runtime"] == str(wrapper)
    assert profiles["reference"] == str(wrapper)


def test_resolve_profile_python_materializes_declared_venv(monkeypatch, tmp_path):
    requirements = tmp_path / "empty.lock.txt"
    requirements.write_text("", encoding="utf-8")
    monkeypatch.setenv("TRTMC_PYTHON_PROFILE_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setattr(
        shared_profiles,
        "load_python_profile_registry",
        lambda: {
            "profiles": {
                "custom": {
                    "kind": "venv",
                    "requirements": str(requirements),
                    "system_site_packages": False,
                    "verification_script": "import sys; print(sys.executable)",
                }
            }
        },
    )

    python = shared_profiles.resolve_profile_python("custom", sys.executable)
    ready = Path(python).parent.parent / ".ready"

    assert Path(python).is_file()
    assert ready.is_file()
    assert shared_profiles.resolve_profile_python("custom", sys.executable) == python


def test_runtime_cli_hf_python_only_applies_to_speech_to_speech(tmp_path):
    base_ctx = RunContext(
        case=_make_case(runtime_strategy="decoder_kv_cache"),
        hf_python="/usr/bin/python3",
        runtime_python="/tmp/runtime-python",
    )
    assert base_ctx.runtime_cli_hf_python() == ""

    speech_ctx = RunContext(
        case=_make_case(
            runtime_strategy="speech_to_speech",
            task_strategy="speech_to_speech",
        ),
        hf_python="/usr/bin/python3",
        runtime_python="/tmp/runtime-python",
    )
    assert speech_ctx.runtime_cli_hf_python() == "/tmp/runtime-python"


def test_repro_commands_record_profile_exports(tmp_path):
    case = _make_case(
        runtime_strategy="decoder_kv_cache",
        task_strategy="text_generation_causal",
    )
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        binary_path="./build/trtmc",
        hf_python="/usr/bin/python3",
        build_python="/tmp/internlm-python",
        runtime_python="/tmp/internlm-python",
        reference_python="/tmp/internlm-python",
        build_profile="internlm",
        runtime_profile="internlm",
        reference_profile="internlm",
        engine_dir="/tmp/engines",
    )
    repro = _build_repro_commands(case, ctx, "/tmp/engines/case-a.trtfb", {})
    assert repro["build_bundle"].startswith("/tmp/internlm-python -m tensorrt_model_connect.__main__ build")
    assert "TRTMC_PYTHON_PROFILE_INTERNLM_PYTHON=/tmp/internlm-python" in repro["profile_env"]
