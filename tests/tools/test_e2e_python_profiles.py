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


def test_resolve_case_profile_names_apply_manifest_profiles():
    case = _make_case(
        runtime_strategy="decoder_kv_cache",
        reference_backend="torch_reference",
        execution_profiles={
            "build": "specialized",
            "runtime": "specialized",
            "reference": "specialized",
        },
    )
    assert resolve_case_profile_names(case) == {
        "build": "specialized",
        "runtime": "specialized",
        "reference": "specialized",
    }


def test_resolve_case_python_profiles_uses_manifest_profile_named_env(monkeypatch, tmp_path):
    wrapper = tmp_path / "specialized-python"
    wrapper.write_text("", encoding="utf-8")
    monkeypatch.setenv(profile_env_var("specialized"), str(wrapper))
    case = _make_case(
        runtime_strategy="decoder_kv_cache",
        reference_backend="torch_reference",
        execution_profiles={
            "build": "specialized",
            "runtime": "specialized",
            "reference": "specialized",
        },
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


def test_runtime_cli_hf_python_is_manifest_metadata_controlled(tmp_path):
    base_ctx = RunContext(
        case=_make_case(runtime_strategy="speech_to_speech", task_strategy="speech_to_speech"),
        hf_python="/usr/bin/python3",
        runtime_python="/tmp/runtime-python",
    )
    assert base_ctx.runtime_cli_hf_python() == ""

    opted_in_ctx = RunContext(
        case=_make_case(
            runtime_strategy="speech_to_speech",
            task_strategy="speech_to_speech",
            metadata={"runtime_cli_requires_hf_python": True},
        ),
        hf_python="/usr/bin/python3",
        runtime_python="/tmp/runtime-python",
    )
    assert opted_in_ctx.runtime_cli_hf_python() == "/tmp/runtime-python"


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
        build_python="/tmp/specialized-python",
        runtime_python="/tmp/specialized-python",
        reference_python="/tmp/specialized-python",
        build_profile="specialized",
        runtime_profile="specialized",
        reference_profile="specialized",
        engine_dir="/tmp/engines",
    )
    repro = _build_repro_commands(case, ctx, "/tmp/engines/case-a.trtfb", {})
    assert repro["build_bundle"].startswith(
        "/tmp/specialized-python -m tensorrt_model_connect.__main__ build"
    )
    assert (
        "TRTMC_PYTHON_PROFILE_SPECIALIZED_PYTHON=/tmp/specialized-python"
        in repro["profile_env"]
    )
