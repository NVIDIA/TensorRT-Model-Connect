# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for uniform model-manifest E2E execution."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pytest

from tests.e2e_harness.contracts import E2ECase, E2EResult
from tests.e2e_harness.manifest_loader import get_model_by_name
from tests.e2e_harness import model_runner
from tests.e2e_harness.orchestrator import BundleResolution


MODELS_DIR = Path(__file__).resolve().parents[1] / "e2e" / "models"


class _Config:
    def __init__(self, **options):
        self._options = options

    def getoption(self, name: str, default=None):
        return self._options.get(name, default)


class _Request:
    def __init__(self, config: _Config):
        self.config = config


def _case_matches(_case, _filters) -> bool:
    return True


def _is_multi_device(case) -> bool:
    return case.metadata.get("ci_tier") == "multi_device"


def test_canary_collects_as_one_model_manifest() -> None:
    names = model_runner.model_names_for_dir(
        config=_Config(**{"--e2e-exclude-ci-tier": []}),
        model_dir=MODELS_DIR / "canary",
        case_matches_model=_case_matches,
        is_multi_device_case=_is_multi_device,
    )

    assert "canary-1b-v2" in names
    assert "canary-1b-v2-asr-probe01" not in names
    assert "canary-1b-v2-tp4" not in names


def test_model_collection_supports_direct_call_without_pytest_config() -> None:
    names = model_runner.model_names_for_dir(
        config=None,
        model_dir=MODELS_DIR / "canary",
        case_matches_model=_case_matches,
        is_multi_device_case=_is_multi_device,
    )

    assert names == ["canary-1b-v2", "canary-1b-v2-tp4"]


def test_ci_tier_filters_children_without_changing_model_identity() -> None:
    model = get_model_by_name("canary-1b-v2", MODELS_DIR / "canary")
    assert model is not None

    selected = model_runner.selected_testcases(
        model,
        config=_Config(**{"--e2e-exclude-ci-tier": ["nightly_only"]}),
        case_matches_model=_case_matches,
        is_multi_device_case=_is_multi_device,
    )

    assert [case.name for case in selected] == ["canary-1b-v2"]


def test_category_filter_separates_e2e_from_historical_regressions() -> None:
    qwen_dir = MODELS_DIR / "qwen"
    regression = get_model_by_name(
        "qwen3-0.6b-regression-native-kv-chunked-prefill", qwen_dir
    )
    ordinary = get_model_by_name("qwen3-0.6b-fp16", qwen_dir)
    assert regression is not None
    assert ordinary is not None

    config = _Config(
        **{
            "--e2e-category": "regression",
            "--e2e-exclude-ci-tier": [],
        }
    )
    selected_regressions = model_runner.selected_testcases(
        regression,
        config=config,
        case_matches_model=_case_matches,
        is_multi_device_case=_is_multi_device,
    )
    selected_ordinary = model_runner.selected_testcases(
        ordinary,
        config=config,
        case_matches_model=_case_matches,
        is_multi_device_case=_is_multi_device,
    )

    assert [case.name for case in selected_regressions] == [
        "qwen3-0.6b-regression-native-kv-chunked-prefill"
    ]
    assert selected_ordinary == []


def test_platform_threshold_overrides_are_scoped_to_matching_platform() -> None:
    case = E2ECase(
        name="elf-b-owt-l0",
        hf_id="embedded-language-flows/ELF-B-owt",
        family="elf_flow",
        runtime_strategy="elf_flow",
        task_strategy="diffusion_text_generation",
        threshold_overrides={
            "contract_max_upstream_text_ned": 0.01,
            "contract_min_upstream_token_agreement_rate": 0.99,
        },
        metadata={
            "platform_threshold_overrides": {
                "THOR": {"contract_max_upstream_text_ned": 0.011}
            }
        },
    )

    thor_case = model_runner._case_with_platform_thresholds(case, "THOR")
    default_case = model_runner._case_with_platform_thresholds(case, "")

    assert thor_case.threshold_overrides == {
        "contract_max_upstream_text_ned": 0.011,
        "contract_min_upstream_token_agreement_rate": 0.99,
    }
    assert default_case is case
    assert case.threshold_overrides["contract_max_upstream_text_ned"] == 0.01


def test_elf_flow_thor_threshold_override_keeps_default_contract() -> None:
    model = get_model_by_name("elf-b-owt-l0", MODELS_DIR / "elf_flow")
    assert model is not None
    case = model.testcases[0]

    assert case.threshold_overrides["contract_max_upstream_text_ned"] == 0.01
    assert case.metadata["platform_threshold_overrides"]["THOR"] == {
        "contract_max_upstream_text_ned": 0.011
    }


def test_model_collection_applies_worker_partition() -> None:
    options = {
        "--e2e-exclude-ci-tier": [],
        "--e2e-partition-id": 1,
        "--e2e-partition-size": 2,
    }
    names = model_runner.model_names_for_dir(
        config=_Config(**options),
        model_dir=MODELS_DIR,
        case_matches_model=_case_matches,
        is_multi_device_case=_is_multi_device,
    )
    models = model_runner.load_all_model_manifests(MODELS_DIR)
    selected_models = [
        model
        for model in models
        if model_runner.selected_testcases(
            model,
            config=_Config(**options),
            case_matches_model=_case_matches,
            is_multi_device_case=_is_multi_device,
        )
    ]

    assert (
        names == [model.name for model in sorted(selected_models, key=lambda item: item.name)][1::2]
    )


def test_model_runner_builds_once_then_runs_all_children(monkeypatch, tmp_path) -> None:
    model = get_model_by_name("canary-1b-v2", MODELS_DIR / "canary")
    assert model is not None
    build_calls = []
    run_calls = []

    monkeypatch.setattr(model_runner, "get_model_by_name", lambda *_args: model)

    def resolve_bundle(self, case, ctx):
        del self
        build_calls.append((case.name, ctx.rebuild))
        return BundleResolution(str(tmp_path / model.bundle), 1.0)

    def run_case(self, case, ctx, prepared=None):
        del self, ctx
        run_calls.append((case.name, prepared is not None))
        return E2EResult(case_name=case.name, status="pass")

    monkeypatch.setattr(
        model_runner.E2EOrchestrator,
        "resolve_model_bundle",
        resolve_bundle,
    )
    monkeypatch.setattr(model_runner.E2EOrchestrator, "run", run_case)
    monkeypatch.setattr(model_runner, "run_preflight", lambda *_args: (True, []))

    config = _Config(
        **{
            "--rebuild-engines": True,
            "--e2e-exclude-ci-tier": [],
            "--e2e-platform": "",
        }
    )
    model_runner.run_model_e2e(
        model_name=model.name,
        request=_Request(config),
        model_dir=MODELS_DIR / "canary",
        load_waives=lambda _platform: {},
        case_matches_model=_case_matches,
        is_multi_device_case=_is_multi_device,
        resolve_hf_python=lambda _config: "python",
        resolve_artifacts_dir=lambda _config: str(tmp_path / "artifacts"),
        resolve_binary=lambda _config: "trtmc",
        resolve_ld_library_path=lambda: "",
        resolve_engine_dir=lambda _config: str(tmp_path),
        resolve_model_plugin_dir=lambda _config: "",
        model_plugin_dir_env=lambda _path: nullcontext(),
    )

    assert build_calls == [("canary-1b-v2", True)]
    assert [name for name, _prepared in run_calls] == [case.name for case in model.testcases]
    assert [prepared for _name, prepared in run_calls] == [True] * 8


def test_model_runner_does_not_build_when_all_preflights_skip(
    monkeypatch, tmp_path
) -> None:
    model = get_model_by_name("canary-1b-v2", MODELS_DIR / "canary")
    assert model is not None
    build_calls = []
    run_calls = []

    monkeypatch.setattr(model_runner, "get_model_by_name", lambda *_args: model)
    monkeypatch.setattr(model_runner, "run_preflight", lambda *_args: (False, []))

    def resolve_bundle(self, case, ctx):
        del self, case, ctx
        build_calls.append(True)
        return BundleResolution(str(tmp_path / model.bundle), 1.0)

    def run_case(self, case, ctx, prepared=None):
        del self, ctx
        run_calls.append((case.name, prepared))
        return E2EResult(case_name=case.name, status="skip")

    monkeypatch.setattr(
        model_runner.E2EOrchestrator,
        "resolve_model_bundle",
        resolve_bundle,
    )
    monkeypatch.setattr(model_runner.E2EOrchestrator, "run", run_case)

    config = _Config(
        **{
            "--rebuild-engines": True,
            "--e2e-exclude-ci-tier": [],
            "--e2e-platform": "",
        }
    )
    with pytest.raises(pytest.skip.Exception):
        model_runner.run_model_e2e(
            model_name=model.name,
            request=_Request(config),
            model_dir=MODELS_DIR / "canary",
            load_waives=lambda _platform: {},
            case_matches_model=_case_matches,
            is_multi_device_case=_is_multi_device,
            resolve_hf_python=lambda _config: "python",
            resolve_artifacts_dir=lambda _config: str(tmp_path / "artifacts"),
            resolve_binary=lambda _config: "trtmc",
            resolve_ld_library_path=lambda: "",
            resolve_engine_dir=lambda _config: str(tmp_path),
            resolve_model_plugin_dir=lambda _config: "",
            model_plugin_dir_env=lambda _path: nullcontext(),
        )

    assert build_calls == []
    assert [name for name, _prepared in run_calls] == [
        case.name for case in model.testcases
    ]
    assert all(prepared is None for _name, prepared in run_calls)
