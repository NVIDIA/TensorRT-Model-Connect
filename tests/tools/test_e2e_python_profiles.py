# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for E2E Python profile resolution and phase-specific interpreter use."""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import pytest
import tensorrt_model_connect.python_profiles as shared_profiles
from tests.e2e_harness.contracts import E2ECase, RunContext
from tests.e2e_harness.orchestrator import _build_repro_commands
from tests.e2e_harness.python_profiles import (
    profile_env_var,
    resolve_case_profile_names,
    resolve_case_python_profiles,
)


def _make_case(runtime_strategy: str = "example_decoder_decoder_kv_cache", **kwargs) -> E2ECase:
    defaults = dict(
        name="case-a",
        hf_id="dummy/model",
        family="dummy",
        runtime_strategy=runtime_strategy,
        task_strategy=kwargs.pop("task_strategy", runtime_strategy),
        bundle="case-a.bundle",
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


def test_python_profile_key_normalizes_virtualenv_interpreter_aliases(tmp_path):
    environment = tmp_path / "venv"
    bin_dir = environment / "bin"
    bin_dir.mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    canonical = bin_dir / "python"
    canonical.write_text("", encoding="utf-8")

    assert shared_profiles._absolute_python(str(canonical)) == str(canonical)
    assert shared_profiles._absolute_python(str(bin_dir / "python3")) == str(canonical)
    assert shared_profiles._absolute_python(str(bin_dir / "python3.12")) == str(canonical)


def test_python_profile_key_preserves_non_virtualenv_interpreter_path(tmp_path):
    interpreter = tmp_path / "python3"

    assert shared_profiles._absolute_python(str(interpreter)) == str(interpreter)


def test_resolve_case_profile_names_apply_manifest_profiles():
    case = _make_case(
        runtime_strategy="example_decoder_decoder_kv_cache",
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


def test_golden_snapshot_overrides_family_reference_profile():
    case = _make_case(
        family="lance",
        runtime_strategy="lance_vision_language",
        reference_backend="golden_snapshot",
    )

    assert resolve_case_profile_names(case)["reference"] == "base"


def test_resolve_case_python_profiles_uses_manifest_profile_named_env(monkeypatch, tmp_path):
    wrapper = tmp_path / "specialized-python"
    wrapper.write_text("", encoding="utf-8")
    monkeypatch.setenv(profile_env_var("specialized"), str(wrapper))
    case = _make_case(
        runtime_strategy="example_decoder_decoder_kv_cache",
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


@pytest.mark.skipif(
    shared_profiles.fcntl is None,
    reason="runtime profile materialization requires POSIX advisory locks",
)
def test_resolve_profile_python_materializes_declared_venv(monkeypatch, tmp_path):
    requirements = tmp_path / "empty.lock.txt"
    requirements.write_text("", encoding="utf-8")
    monkeypatch.delenv(shared_profiles.PREBUILT_ONLY_ENV, raising=False)
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

    created = []
    python = shared_profiles.resolve_profile_python(
        "custom",
        sys.executable,
        on_create=created.append,
    )
    ready = Path(python).parent.parent / ".ready"

    assert Path(python).is_file()
    assert ready.is_file()
    assert created == ["custom"]
    assert (
        shared_profiles.resolve_profile_python(
            "custom",
            sys.executable,
            on_create=created.append,
        )
        == python
    )
    assert created == ["custom"]


def test_materialize_venv_profile_rejects_platform_without_posix_locks(
    monkeypatch, tmp_path
):
    requirements = tmp_path / "empty.lock.txt"
    requirements.write_text("", encoding="utf-8")
    monkeypatch.setattr(shared_profiles, "fcntl", None)
    monkeypatch.setenv(shared_profiles.PROFILE_ROOT_ENV, str(tmp_path / "profiles"))

    with pytest.raises(
        RuntimeError,
        match="Materializing non-base Python profiles is not supported on Windows",
    ):
        shared_profiles._materialize_venv_profile(
            "custom",
            {
                "kind": "venv",
                "requirements": str(requirements),
                "system_site_packages": False,
            },
            sys.executable,
        )


def test_profile_source_builds_use_a_safe_default_job_limit(monkeypatch, tmp_path):
    requirements = tmp_path / "requirements.lock.txt"
    requirements.write_text("demo-package==1.0\n", encoding="utf-8")
    monkeypatch.delenv("MAX_JOBS", raising=False)
    monkeypatch.setenv("PYTHONPATH", "/untrusted/profile/source")
    monkeypatch.delenv(shared_profiles.PREBUILT_ONLY_ENV, raising=False)
    monkeypatch.setenv(shared_profiles.PROFILE_ROOT_ENV, str(tmp_path / "profiles"))
    monkeypatch.setattr(
        shared_profiles,
        "_verify_exact_requirements",
        lambda *_args, **_kwargs: None,
    )

    commands = []

    def run_command(cmd, *, description, timeout=1800, **kwargs):
        commands.append((cmd, description, timeout, kwargs))
        if description.startswith("create Python profile"):
            python = Path(cmd[-1]) / "bin" / "python"
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("", encoding="utf-8")

    monkeypatch.setattr(shared_profiles, "_run_profile_command", run_command)

    shared_profiles._materialize_venv_profile(
        "custom",
        {
            "requirements": str(requirements),
            "system_site_packages": False,
            "verification_script": "print('verified')",
        },
        sys.executable,
    )

    install = next(call for call in commands if call[1].startswith("install "))
    assert install[3]["env"]["MAX_JOBS"] == "4"
    assert "PYTHONPATH" not in install[3]["env"]
    assert install[2] == 7200
    verify = next(call for call in commands if call[1].startswith("verify "))
    assert "PYTHONPATH" not in verify[3]["env"]


def test_profile_source_builds_respect_an_explicit_job_limit(monkeypatch):
    monkeypatch.setenv("MAX_JOBS", "2")

    assert shared_profiles._profile_install_environment()["MAX_JOBS"] == "2"


def test_profile_command_timeout_terminates_descendants(tmp_path):
    sentinel = tmp_path / "orphan-finished"
    child = (
        "import pathlib, time; "
        "time.sleep(0.3); "
        f"pathlib.Path({str(sentinel)!r}).write_text('orphaned')"
    )
    parent = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
        "time.sleep(60)"
    )

    error = None
    try:
        shared_profiles._run_profile_command(
            [sys.executable, "-c", parent],
            description="run descendant timeout regression",
            timeout=0.05,
        )
    except Exception as caught:  # noqa: BLE001 - assert the public failure below.
        error = caught

    time.sleep(0.5)
    assert isinstance(error, RuntimeError)
    assert "timed out" in str(error)
    assert not sentinel.exists()


def test_process_session_members_ignores_vanished_proc_entry(monkeypatch):
    class VanishedStat:
        def read_text(self, *, encoding):
            raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr(
        shared_profiles.Path,
        "glob",
        lambda _path, _pattern: [VanishedStat()],
    )

    assert shared_profiles._process_session_members(123) == []


def test_family_profile_registry_is_fully_exact_pinned():
    expected = {
        "chronos",
        "deepseek_ocr",
        "elf_flow",
        "elf_flow_reference",
        "internlm",
        "lance_reference",
        "magpie_tts_reference",
        "nemotron_h_reference",
        "personaplex_full_duplex_evaluator",
        "phi4_multimodal",
        "sana_wm_reference",
        "reference_common",
    }
    profiles = shared_profiles.load_python_profile_registry()["profiles"]

    assert set(profiles) - {shared_profiles.DEFAULT_PROFILE} == expected
    for name in expected:
        requirements = shared_profiles._read_requirements_text(
            profiles[name]["requirements"]
        )
        pins = shared_profiles._exact_pinned_requirements(requirements)
        assert pins, name


def test_profile_contract_has_one_family_owned_source_of_truth() -> None:
    package_root = Path(shared_profiles.__file__).resolve().parent
    with (package_root / "python_profiles.toml").open("rb") as stream:
        shared_registry = shared_profiles.tomllib.load(stream)
    from tensorrt_model_connect.families import family_python_profile_specs

    shared_names = set(shared_registry["profiles"])
    family_names = set(family_python_profile_specs())
    merged_names = set(shared_profiles.load_python_profile_registry()["profiles"])

    assert shared_names == {"base", "reference_common"}
    assert family_names
    assert shared_names.isdisjoint(family_names)
    assert merged_names == shared_names | family_names


def test_lazy_profiles_are_excluded_from_the_shared_ci_image() -> None:
    registry = shared_profiles.load_python_profile_registry()
    prebuilt = shared_profiles.prebuilt_python_profile_names(registry)

    assert "personaplex_full_duplex_evaluator" not in prebuilt
    assert "reference_common" in prebuilt

def test_profile_lock_rejects_non_exact_or_duplicate_requirements():
    with pytest.raises(ValueError, match="exact name==version pins"):
        shared_profiles._exact_pinned_requirements("transformers>=4.48\n")

    with pytest.raises(ValueError, match="more than once"):
        shared_profiles._exact_pinned_requirements(
            "huggingface-hub==0.28.1\nhuggingface_hub==0.28.1\n"
        )


@pytest.mark.parametrize(
    "requirement",
    (
        "demo==1.*",
        "demo===latest",
        "demo==https://example.invalid/demo.whl",
    ),
)
def test_profile_lock_rejects_non_deterministic_exact_pin_lookalikes(requirement):
    with pytest.raises(ValueError, match="exact name==version pins"):
        shared_profiles._exact_pinned_requirements(requirement + "\n")


def test_profile_registry_validates_supported_global_defaults():
    registry = copy.deepcopy(shared_profiles.load_python_profile_registry())
    registry["runtime_strategy_defaults"] = {
        "demo": {"runtime": "reference_common"}
    }

    shared_profiles._validate_python_profile_registry(registry)

    registry["runtime_strategy_defaults"]["demo"]["runtime"] = "missing"
    with pytest.raises(ValueError, match="undeclared profile"):
        shared_profiles._validate_python_profile_registry(registry)


def test_family_default_profile_must_be_declared(monkeypatch):
    import tensorrt_model_connect.families as family_profiles

    monkeypatch.setattr(
        family_profiles,
        "family_default_execution_profiles",
        lambda _family: {"runtime": "missing"},
    )

    with pytest.raises(ValueError, match="selects undeclared profile 'missing'"):
        shared_profiles.default_execution_profiles(family="demo")


@pytest.mark.parametrize(
    ("path_spec", "message"),
    (
        ("/tmp/absolute.lock.txt", "unsafe requirements path"),
        ("../outside.lock.txt", "unsafe requirements path"),
        ("missing.lock.txt", "missing requirements asset"),
    ),
)
def test_profile_registry_rejects_unsafe_or_missing_assets(
    monkeypatch, tmp_path, path_spec, message
):
    package_root = tmp_path / "package"
    package_root.mkdir()
    monkeypatch.setattr(shared_profiles, "_PACKAGE_DIR", package_root)

    with pytest.raises(ValueError, match=message):
        shared_profiles._profile_asset_path(
            path_spec,
            field="requirements",
            profile_name="demo",
        )


def test_profile_registry_rejects_symlink_escape(monkeypatch, tmp_path):
    package_root = tmp_path / "package"
    package_root.mkdir()
    outside = tmp_path / "outside.lock.txt"
    outside.write_text("demo==1.0\n", encoding="utf-8")
    (package_root / "escaped.lock.txt").symlink_to(outside)
    monkeypatch.setattr(shared_profiles, "_PACKAGE_DIR", package_root)

    with pytest.raises(ValueError, match="unsafe requirements path"):
        shared_profiles._profile_asset_path(
            "escaped.lock.txt",
            field="requirements",
            profile_name="demo",
        )


def test_profile_registry_rejects_ambiguous_schema_fields():
    def add_unknown_top_level(registry):
        registry["unknown"] = True

    def add_unknown_profile_field(registry):
        registry["profiles"]["base"]["unknown"] = True

    def use_unknown_kind(registry):
        registry["profiles"]["base"]["kind"] = "dynamic"

    def use_non_boolean_flag(registry):
        registry["profiles"]["reference_common"]["prebuild"] = 1

    def declare_two_verification_sources(registry):
        registry["profiles"]["reference_common"]["verification_script_file"] = (
            registry["profiles"]["reference_common"]["requirements"]
        )

    cases = (
        (add_unknown_top_level, "unknown top-level keys"),
        (add_unknown_profile_field, "unknown keys"),
        (use_unknown_kind, "unsupported kind"),
        (use_non_boolean_flag, "must be a bool"),
        (declare_two_verification_sources, "exactly one"),
    )
    baseline = shared_profiles.load_python_profile_registry()
    for mutate, message in cases:
        registry = copy.deepcopy(baseline)
        mutate(registry)
        with pytest.raises(ValueError, match=message):
            shared_profiles._validate_python_profile_registry(registry)


def test_exact_profile_pin_accepts_only_local_builds_of_same_public_version():
    assert shared_profiles._pinned_version_matches(
        "3.1.0",
        "3.1.0+c9040511b",
    )
    assert shared_profiles._pinned_version_matches("3.1.0", "3.1.0")
    assert not shared_profiles._pinned_version_matches("3.1.0", "3.1.1")
    assert not shared_profiles._pinned_version_matches(
        "3.1.0+expected",
        "3.1.0+different",
    )


def test_prebuilt_only_profile_fails_before_creating_a_runtime_cache(
    monkeypatch, tmp_path
):
    requirements = tmp_path / "empty.lock.txt"
    requirements.write_text("", encoding="utf-8")
    profile_root = tmp_path / "profiles"
    monkeypatch.setenv(shared_profiles.PROFILE_ROOT_ENV, str(profile_root))
    monkeypatch.setenv(shared_profiles.PREBUILT_ONLY_ENV, "1")
    monkeypatch.setattr(
        shared_profiles,
        "load_python_profile_registry",
        lambda: {
            "profiles": {
                "custom": {
                    "kind": "venv",
                    "requirements": str(requirements),
                    "system_site_packages": False,
                }
            }
        },
    )

    with pytest.raises(RuntimeError, match="CI image is stale or incomplete"):
        shared_profiles.resolve_profile_python("custom", sys.executable)

    assert not profile_root.exists()


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
        runtime_strategy="example_decoder_decoder_kv_cache",
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
    repro = _build_repro_commands(case, ctx, "/tmp/engines/case-a.bundle", {})
    assert repro["build_bundle"].startswith(
        "/tmp/specialized-python -m tensorrt_model_connect.__main__ build"
    )
    assert (
        "TRTMC_PYTHON_PROFILE_SPECIALIZED_PYTHON=/tmp/specialized-python"
        in repro["profile_env"]
    )
