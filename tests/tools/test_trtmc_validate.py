# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest

from tools.validation import catalog as validation_catalog
from tools import trtmc_compare
from tools import trtmc_disagreements
from tools import trtmc_reference
from tools import trtmc_validate


def test_validation_entrypoints_use_narrow_engine_boundaries():
    for entrypoint in ("trtmc_validate.py", "trtmc_reference.py"):
        source = (trtmc_validate.REPO_ROOT / "tools" / entrypoint).read_text(encoding="utf-8")
        assert "validation import engine" not in source


def test_model_workload_catalog_covers_every_ready_model():
    catalog = trtmc_validate.load_catalog()
    suites = validation_catalog.load_suites()
    task_models = validation_catalog.load_manifest_records_by_name(trtmc_validate.DEFAULT_MODELS)
    ready_models = trtmc_validate.ready_model_names()

    trtmc_validate.audit_catalog(
        catalog,
        ready_models=ready_models,
        suite_names=(suite["id"] for suite in suites),
    )
    trtmc_validate.audit_workload_compatibility(
        catalog,
        suites={suite["id"]: suite for suite in suites},
        task_models=task_models,
    )

    assert len(catalog["models"]) == len(ready_models) == 119
    assert sum("not_compared_reason" in spec for spec in catalog["models"].values()) == 0
    assert all("e2e" not in spec.get("workloads", []) for spec in catalog["models"].values())
    assert "reference_cache_identity" not in catalog["models"]["personaplex-7b"]
    assert (
        catalog["models"]["flux-2-dev"]["reference_cache_identity"]
        == catalog["models"]["flux-2-dev-fp8"]["reference_cache_identity"]
    )
    assert catalog["models"]["flux-2-dev"]["reference_cache_identity"] == "flux-2-dev-dpg-v2"
    qwen_identities = {
        catalog["models"][name]["reference_cache_identity"]
        for name in (
            "qwen3-0.6b-fp16",
            "qwen3-0.6b-fp8",
            "qwen3-0.6b-topp",
        )
    }
    assert len(qwen_identities) == 1
    bindings = trtmc_validate.resolve_bindings(catalog, catalog["models"])
    assert len(bindings) == 120
    assert {
        binding.model for binding in bindings if binding.workload == "mmlu_continuation_parity"
    } >= {
        "lfm2-1.2b",
        "lfm2-2.6b",
        "lfm2-350m-fp16",
        "lfm2-700m",
    }
    assert trtmc_validate.resolve_binding(
        catalog,
        "lfm2-350m-bf16-model-card",
    ) == trtmc_validate.Binding(
        "lfm2-350m-bf16-model-card",
        "lfm2_model_card_sampling_parity",
    )
    assert [binding.workload for binding in bindings if binding.model == "personaplex-7b"] == [
        "full_duplex_bench_behavior_parity"
    ]
    assert trtmc_validate.resolve_binding(
        catalog,
        "personaplex-7b",
        "full_duplex_bench_speech_parity",
    ) == trtmc_validate.Binding(
        "personaplex-7b",
        "full_duplex_bench_speech_parity",
    )
    assert [binding.workload for binding in bindings if binding.model == "qwen25vl-3b"] == [
        "vlm_mmmu_pro_vision_fixed_mcq"
    ]


def test_fast_foundation_stereo_catalog_binds_middlebury_task_accuracy() -> None:
    catalog = trtmc_validate.load_catalog()

    assert catalog["sample_limits"]["fast_foundation_stereo_middlebury_q_task_accuracy"] == 15
    assert catalog["models"]["fast-foundation-stereo"]["workloads"] == [
        "fast_foundation_stereo_synthetic_parity",
        "fast_foundation_stereo_middlebury_q_task_accuracy",
    ]


def test_minimax_h3_catalog_uses_model_owned_official_profile() -> None:
    catalog = trtmc_validate.load_catalog()
    suites = validation_catalog.load_suites()
    suite = next(value for value in suites if value["id"] == "minimax_h3_official_profile_parity")
    model = next(
        value
        for value in validation_catalog.load_manifest_records(trtmc_validate.DEFAULT_MODELS)
        if value["name"] == "minimax-h3-768p"
    )

    assert catalog["models"]["minimax-h3-768p"] == {
        "workloads": [
            "minimax_h3_official_profile_parity",
            "minimax_h3_vbench_siglip_task_quality",
        ],
    }
    assert validation_catalog.suite_match_reason(suite, model) == (
        True,
        "selected",
    )
    assert suite["dataset"] == {
        "kind": "model_plugin_json",
        "default_path": ("tests/e2e/models/minimax_h3/validation/minimax-h3-768p.json"),
    }
    assert suite["scoring"] == {"scorer": "model_plugin_parity"}
    assert suite["gates"] == {"min_sample_pass_rate": 1.0}

    dataset_path = trtmc_validate.REPO_ROOT / suite["dataset"]["default_path"]
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert dataset["requests"] == [
        {
            "sample_id": "minimax-h3-768p-official-profile",
            "testcase": "minimax-h3-768p",
            "stage": "end_to_end",
            "category": "official-profile",
            "inputs": {},
        }
    ]
    resolved = validation_catalog.resolve_suite_for_model(suite, model)
    assert resolved["generation"] == {
        "video_num_frames": 124,
        "video_height": 768,
        "video_width": 1344,
        "num_inference_steps": 50,
    }

    task_quality = next(
        value for value in suites if value["id"] == "minimax_h3_vbench_siglip_task_quality"
    )
    assert catalog["sample_limits"][task_quality["id"]] == 100
    assert task_quality["reference"] == {"mode": "metric_only"}
    assert task_quality["dataset"] == {
        "kind": "model_plugin_json",
        "default_path": ("/mnt/data/vbench-fd18b3d-minimax-h3-siglip-v1/dataset.json"),
        "input_asset_fields": ["prompt_file"],
    }
    assert task_quality["scoring"] == {
        "scorer": "vbench_siglip",
        "python_profile": "reference_common",
        "device": "cuda:0",
    }
    assert task_quality["gates"] == {
        "required_sample_count": 100,
        "min_structural_pass_rate": 1.0,
    }
    assert validation_catalog.suite_match_reason(task_quality, model) == (
        True,
        "selected",
    )


def test_dataset_path_keeps_repository_owned_default_with_dataset_root(
    tmp_path: Path,
) -> None:
    suite = {
        "id": "repo-owned",
        "dataset": {"default_path": "tests/e2e/models/minimax_h3/validation/minimax-h3-768p.json"},
    }

    assert trtmc_validate._dataset_path(suite, tmp_path / "datasets") == (
        trtmc_validate.REPO_ROOT / "tests/e2e/models/minimax_h3/validation/minimax-h3-768p.json"
    )


def test_dataset_path_rebases_mounted_defaults_under_dataset_root(
    tmp_path: Path,
) -> None:
    suite = {
        "id": "mounted",
        "dataset": {"default_path": "/mnt/data/example/dataset.json"},
    }

    assert trtmc_validate._dataset_path(suite, tmp_path / "datasets") == (
        tmp_path / "datasets/example/dataset.json"
    )


def test_validation_ready_models_exclude_l0_and_regression_profiles():
    records = validation_catalog.load_manifest_records(trtmc_validate.DEFAULT_MODELS)
    eligible = {
        str(record["name"])
        for record in records
        if not record["requires_multi_device"] and not record.get("skip")
    }
    l0_only = {str(record["name"]) for record in records if record.get("ci_tier") == "l0_only"}
    regressions = {
        str(record["name"]) for record in records if record.get("test_category") == "regression"
    }
    selected = set(trtmc_validate.ready_model_names())

    assert l0_only
    assert regressions == {
        "minitron-4b-width-regression-native-kv-chunked-prefill",
        "qwen3-0.6b-regression-native-kv-chunked-prefill",
        "sam2-l4-local",
    }
    assert selected == eligible - l0_only - regressions


def test_catalog_defines_sample_limit_for_every_dataset_workload():
    catalog = trtmc_validate.load_catalog()
    configured = set(catalog["sample_limits"])
    declared = {
        workload
        for spec in catalog["models"].values()
        for workload in trtmc_validate.declared_workloads(spec)
    }

    assert declared <= configured
    assert configured - declared == {
        "full_duplex_bench_speech_parity",
        "refcoco_grounding",
        "vlm_mmmu_pro_vision_mcq",
    }
    assert min(catalog["sample_limits"].values()) >= 1
    assert {workload for workload, limit in catalog["sample_limits"].items() if limit == 1} == {
        "dinov3_image_feature_extraction_parity",
        "fast_foundation_stereo_synthetic_parity",
        "lfm2_model_card_sampling_parity",
        "minimax_h3_official_profile_parity",
        "moge_monocular_geometry_fp32_parity",
        "nemotron_voicechat_model_card_general_conversation",
        "seedtts_en_omni_audio_parity",
        "vbench_ti2v_official_profile_parity",
    }
    assert max(catalog["sample_limits"].values()) == 150
    assert catalog["sample_limits"]["full_duplex_bench_behavior_parity"] == 150
    assert catalog["sample_limits"]["mmlu_five_shot_mcq"] == 20
    assert catalog["sample_limits"]["dpg_bench_diffusion_image"] == 5
    assert catalog["sample_limits"]["gedit_bench_image_edit"] == 5


def test_standard_validation_suites_have_report_task_types():
    missing = [
        suite["id"]
        for suite in validation_catalog.load_suites()
        if not trtmc_validate._suite_task_metadata(suite)[0]
    ]

    assert not missing


def test_every_dataset_backed_validation_binding_has_native_reference_runner():
    catalog = trtmc_validate.load_catalog()
    suites = {suite["id"]: suite for suite in validation_catalog.load_suites()}
    bindings = [
        (model_name, workload)
        for model_name, spec in catalog["models"].items()
        for workload in spec.get("workloads", [])
    ]
    missing = []
    for model_name, workload in bindings:
        dataset_kind = str(suites[workload]["dataset"]["kind"])
        if trtmc_reference.native_reference_runner_for_dataset_kind(dataset_kind) is None:
            missing.append((model_name, workload, dataset_kind))

    assert not missing
    assert len({model for model, _workload in bindings}) == 119


def test_shadow_gate_metrics_include_plugin_task_accuracy() -> None:
    metrics = trtmc_validate._shadow_gate_metrics(
        {"metrics": {"sample_pass_rate": 1.0}},
        {
            "task_accuracy": {
                "candidate_nonocc_epe_px": 0.44,
                "reference_nonocc_epe_px": 0.43,
            }
        },
    )

    assert metrics == {
        "sample_pass_rate": 1.0,
        "candidate_nonocc_epe_px": 0.44,
        "reference_nonocc_epe_px": 0.43,
    }


def test_resolve_binding_requires_an_explicit_choice_for_multi_workload_model():
    catalog = {
        "sample_limits": {
            "workload-a": 5,
            "workload-b": 5,
            "workload-c": 5,
        },
        "models": {
            "model-a": {
                "workloads": ["workload-a", "workload-b"],
                "reference_cache_identity": "org/model/reference-contract-v1",
            }
        },
    }

    with pytest.raises(trtmc_validate.ValidationError, match="selects 2 workloads"):
        trtmc_validate.resolve_binding(catalog, "model-a")
    assert trtmc_validate.resolve_binding(catalog, "model-a", "workload-b") == (
        trtmc_validate.Binding(
            "model-a",
            "workload-b",
            reference_cache_identity="org/model/reference-contract-v1",
        )
    )
    assert trtmc_validate.resolve_binding(catalog, "model-a", "workload-c") == (
        trtmc_validate.Binding(
            "model-a",
            "workload-c",
            reference_cache_identity="org/model/reference-contract-v1",
        )
    )
    with pytest.raises(trtmc_validate.ValidationError, match="unknown workload"):
        trtmc_validate.resolve_binding(catalog, "model-a", "missing-workload")


def test_resolve_bindings_expands_every_model_workload():
    catalog = {
        "sample_limits": {
            "workload-a": 5,
            "workload-b": 5,
            "workload-c": 5,
        },
        "models": {
            "model-a": {
                "workloads": ["workload-a", "workload-b"],
            },
            "model-b": {
                "workloads": ["workload-c"],
            },
        },
    }

    assert trtmc_validate.resolve_bindings(
        catalog,
        ["model-a", "model-b"],
    ) == [
        trtmc_validate.Binding("model-a", "workload-a"),
        trtmc_validate.Binding("model-a", "workload-b"),
        trtmc_validate.Binding("model-b", "workload-c"),
    ]


def test_resolve_binding_allows_globally_configured_unmapped_workload():
    catalog = {
        "sample_limits": {"workload-a": 5, "workload-b": 9},
        "models": {
            "model-a": {
                "workloads": ["workload-a"],
            }
        },
    }

    assert trtmc_validate.resolve_binding(catalog, "model-a", "workload-b") == (
        trtmc_validate.Binding("model-a", "workload-b")
    )


def test_explicit_binding_still_has_to_match_suite_selectors():
    binding = trtmc_validate.Binding("model-a", "workload-b")
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="model-a/workload-b: model=model-a not selected",
    ):
        trtmc_validate.audit_binding_compatibility(
            [binding],
            suites={"workload-b": {"selectors": {"model_names": ["model-b"]}}},
            task_models={"model-a": {"name": "model-a"}},
        )


def test_list_shows_mapped_workloads_and_sample_limits(capsys, monkeypatch):
    catalog = {
        "sample_limits": {"workload-a": 5, "workload-b": 9},
        "models": {
            "model-a": {
                "workloads": ["workload-a"],
            }
        },
    }
    arguments = trtmc_validate.build_parser().parse_args(["--list"])
    monkeypatch.setattr(
        trtmc_validate,
        "_load_validation_inputs",
        lambda _arguments: (catalog, {}, (), {}),
    )

    assert trtmc_validate._main(arguments) == 0
    assert capsys.readouterr().out.strip() == "model-a: workload-a (5 samples)"


def test_gate_census_groups_resolved_variants_and_exposes_review_gaps() -> None:
    catalog = {
        "models": {
            "default-model": {"workloads": ["quality"]},
            "strict-model": {"workloads": ["quality"]},
            "observer": {"workloads": ["diagnostic"]},
        },
        "sample_limits": {"quality": 20, "diagnostic": 5, "explicit": 5},
    }
    suites = {
        "quality": {
            "id": "quality",
            "description": "Sampled quality parity.",
            "gates": {"min_prediction_agreement": 0.98},
            "model_profiles": {"strict-model": {"gates": {"min_prediction_agreement": 1.0}}},
        },
        "diagnostic": {
            "id": "diagnostic",
            "description": "Diagnostic only.",
            "gates": {},
            "gate_policy": "observation_only",
        },
        "unbound": {
            "id": "unbound",
            "description": "Not selected by the current model inventory.",
            "gates": {"min_sample_pass_rate": 1.0},
        },
        "explicit": {
            "id": "explicit",
            "description": "Selected only by an explicit command.",
            "selection": "explicit_only",
            "gates": {"min_sample_pass_rate": 1.0},
        },
    }
    task_models = {
        name: {
            "name": name,
            "family": "fixture",
            "task_strategy": "fixture",
            "runtime_strategy": "fixture",
            "user_contract": "fixture",
            "skip": "",
        }
        for name in catalog["models"]
    }

    census = trtmc_validate.build_gate_census(
        catalog=catalog,
        suites=suites,
        task_models=task_models,
    )

    assert census["schema_version"] == "trtmc.validation-gate-census/v1"
    assert census["summary"] == {
        "suites": 4,
        "bindings": 3,
        "variants": 5,
        "blocking_variants": 4,
        "observation_only_variants": 1,
        "invalid_variants": 1,
        "review_required_suites": 1,
    }
    quality = next(row for row in census["suites"] if row["id"] == "quality")
    assert quality["owner"] == {"kind": "workload", "id": "quality"}
    assert quality["rationale"] == "Sampled quality parity."
    assert quality["configured_sample_count"] == 20
    assert [variant["models"] for variant in quality["variants"]] == [
        ["default-model"],
        ["strict-model"],
    ]
    assert quality["review"] == []
    diagnostic = next(row for row in census["suites"] if row["id"] == "diagnostic")
    assert diagnostic["review"] == []
    unbound = next(row for row in census["suites"] if row["id"] == "unbound")
    assert unbound["review"] == [
        {"code": "no_selected_models"},
        {"code": "sample_limit_unconfigured"},
    ]
    explicit = next(row for row in census["suites"] if row["id"] == "explicit")
    assert explicit["selection"] == "explicit_only"
    assert explicit["review"] == []


def test_gate_census_expands_sample_acceptance_at_configured_count() -> None:
    catalog = {
        "models": {"model-a": {"workloads": ["quality"]}},
        "sample_limits": {"quality": 20},
    }
    suites = {
        "quality": {
            "id": "quality",
            "description": "Sampled quality parity.",
            "gates": {},
            "sample_acceptance": {
                "min_pass_rate": 0.98,
                "min_allowed_failures": 1,
            },
        }
    }
    task_models = {
        "model-a": {
            "name": "model-a",
            "family": "fixture",
            "task_strategy": "fixture",
            "runtime_strategy": "fixture",
            "user_contract": "fixture",
            "skip": "",
        }
    }

    census = trtmc_validate.build_gate_census(
        catalog=catalog,
        suites=suites,
        task_models=task_models,
    )

    assert census["summary"]["review_required_suites"] == 0
    quality = census["suites"][0]
    assert quality["review"] == []
    assert quality["variants"][0]["policy"]["policy_mode"] == "blocking"
    assert quality["variants"][0]["sample_acceptance"] == {
        "sample_count": 20,
        "min_pass_rate": 0.98,
        "min_allowed_failures": 1,
        "allowed_failures": 1,
        "issues": [],
    }


def test_gate_census_cli_prints_machine_readable_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        trtmc_validate,
        "_load_validation_inputs",
        lambda arguments: (
            {"models": {}, "sample_limits": {}},
            {},
            (),
            {},
        ),
    )
    monkeypatch.setattr(
        trtmc_validate,
        "build_gate_census",
        lambda **kwargs: {
            "schema_version": "trtmc.validation-gate-census/v1",
            "summary": {"suites": 0},
            "suites": [],
        },
    )

    arguments = trtmc_validate.build_parser().parse_args(["--gate-census"])

    assert trtmc_validate._main(arguments) == 0
    assert json.loads(capsys.readouterr().out) == {
        "schema_version": "trtmc.validation-gate-census/v1",
        "summary": {"suites": 0},
        "suites": [],
    }


def test_gate_census_cli_rejects_model_selection(monkeypatch) -> None:
    monkeypatch.setattr(
        trtmc_validate,
        "_load_validation_inputs",
        lambda arguments: (
            {"models": {}, "sample_limits": {}},
            {},
            (),
            {},
        ),
    )
    arguments = trtmc_validate.build_parser().parse_args(["gpt2-125m", "--gate-census"])

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="global inventory",
    ):
        trtmc_validate._main(arguments)


def test_default_gate_census_covers_every_suite_and_binding() -> None:
    catalog = trtmc_validate.load_catalog()
    suites = {suite["id"]: suite for suite in validation_catalog.load_suites()}
    task_models = validation_catalog.load_manifest_records_by_name(trtmc_validate.DEFAULT_MODELS)

    census = trtmc_validate.build_gate_census(
        catalog=catalog,
        suites=suites,
        task_models=task_models,
    )

    assert {row["id"] for row in census["suites"]} == set(suites)
    assert census["summary"]["bindings"] == sum(
        len(spec.get("workloads", [])) for spec in catalog["models"].values()
    )
    invalid = [
        row["id"]
        for row in census["suites"]
        if any(
            variant["policy"]["issues"] or variant.get("sample_acceptance", {}).get("issues")
            for variant in row["variants"]
        )
    ]
    assert invalid == []
    assert census["summary"]["review_required_suites"] == 0
    explicit_only = {row["id"] for row in census["suites"] if row["selection"] == "explicit_only"}
    assert explicit_only == {
        "full_duplex_bench_speech_parity",
        "refcoco_grounding",
        "vlm_mmmu_pro_vision_mcq",
    }
    mmlu = next(row for row in census["suites"] if row["id"] == "mmlu_five_shot_mcq")
    assert len(mmlu["variants"]) == 2
    assert [variant["sample_acceptance"]["min_pass_rate"] for variant in mmlu["variants"]] == [
        0.98,
        0.95,
    ]
    assert [variant["sample_acceptance"]["allowed_failures"] for variant in mmlu["variants"]] == [
        1,
        1,
    ]
    asr = next(row for row in census["suites"] if row["id"] == "librispeech_clean_asr")
    assert asr["variants"][0]["policy"]["gates"][1]["effective"]["kind"] == ("continuous")
    marian = next(
        row
        for row in census["suites"]
        if row["id"] == "newstest2019_en_ru_marian_translation_parity"
    )
    assert marian["review"] == []


def test_resolve_bindings_selects_multiple_explicit_workloads():
    catalog = {
        "sample_limits": {"workload-a": 5, "workload-b": 5},
        "models": {
            "model-a": {
                "workloads": ["workload-a", "workload-b"],
            }
        },
    }

    assert trtmc_validate.resolve_bindings(
        catalog,
        ["model-a"],
        workloads=["workload-b", "workload-a", "workload-b"],
    ) == [
        trtmc_validate.Binding("model-a", "workload-b"),
        trtmc_validate.Binding("model-a", "workload-a"),
    ]


def test_select_bindings_reads_model_ci_selection_and_expands_workloads(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps({"matrix": {"include": [{"model": "model-a"}]}}),
        encoding="utf-8",
    )
    arguments = trtmc_validate.build_parser().parse_args(
        ["--model-selection", str(selection), "--dry-run"]
    )
    catalog = {
        "sample_limits": {"workload-a": 5, "workload-b": 5},
        "models": {
            "model-a": {
                "workloads": ["workload-a", "workload-b"],
            }
        },
    }

    assert trtmc_validate._select_bindings(
        arguments,
        catalog,
        ("model-a",),
        {"model-a": {"family": "model-a"}},
    ) == [
        trtmc_validate.Binding("model-a", "workload-a"),
        trtmc_validate.Binding("model-a", "workload-b"),
    ]


def test_model_ci_family_selection_expands_ready_accuracy_profiles():
    assert trtmc_validate.model_profiles_for_families(
        {
            "model-a-small": {"family": "family-a"},
            "model-a-large": {"family": "family-a"},
            "model-b": {"family": "family-b"},
        },
        ("model-a-small", "model-a-large", "model-b"),
        ("family-a",),
    ) == ("model-a-large", "model-a-small")


def test_select_bindings_rejects_ambiguous_selection_modes():
    arguments = trtmc_validate.build_parser().parse_args(
        ["model-a", "--model", "model-b", "--dry-run"]
    )

    with pytest.raises(trtmc_validate.ValidationError, match="choose exactly one"):
        trtmc_validate._select_bindings(
            arguments,
            {"models": {}},
            (),
        )


def test_select_bindings_requires_one_binding_for_explicit_dataset(tmp_path):
    dataset = tmp_path / "dataset.json"
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--model",
            "model-a",
            "--dataset",
            str(dataset),
            "--dry-run",
        ]
    )
    catalog = {
        "sample_limits": {"workload-a": 5, "workload-b": 5},
        "models": {
            "model-a": {
                "workloads": ["workload-a", "workload-b"],
            }
        },
    }

    with pytest.raises(trtmc_validate.ValidationError, match="exactly one"):
        trtmc_validate._select_bindings(arguments, catalog, ("model-a",))


def test_select_bindings_accepts_multiple_exact_model_workload_pairs():
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--binding",
            "model-a=workload-b",
            "--binding",
            "model-b=workload-c",
            "--binding",
            "model-a=workload-b",
            "--dry-run",
        ]
    )
    catalog = {
        "sample_limits": {
            "workload-a": 5,
            "workload-b": 5,
            "workload-c": 5,
        },
        "models": {
            "model-a": {
                "workloads": ["workload-a", "workload-b"],
            },
            "model-b": {
                "workloads": ["workload-c"],
            },
        },
    }

    assert trtmc_validate._select_bindings(
        arguments,
        catalog,
        ("model-a", "model-b"),
    ) == [
        trtmc_validate.Binding("model-a", "workload-b"),
        trtmc_validate.Binding("model-b", "workload-c"),
    ]


def test_select_bindings_rejects_malformed_exact_binding():
    arguments = trtmc_validate.build_parser().parse_args(["--binding", "model-a", "--dry-run"])

    with pytest.raises(trtmc_validate.ValidationError, match="MODEL=WORKLOAD"):
        trtmc_validate._select_bindings(
            arguments,
            {"models": {}},
            (),
        )


def test_binding_scoped_engines_are_isolated_across_suites_and_deleted_on_pass(
    tmp_path,
    monkeypatch,
):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--output",
            str(tmp_path / "results"),
            "--model-work-dir",
            str(tmp_path / "work"),
            "--engine-retention",
            "delete_on_pass",
            "--hf-cache-mode",
            "per_model",
            "--hf-cache-retention",
            "delete_on_pass",
            "--reference-cache-dir",
            str(tmp_path / "references"),
        ]
    )
    bindings = [
        trtmc_validate.Binding("model-a", "suite-a"),
        trtmc_validate.Binding("model-a", "suite-b"),
    ]
    engine_directories = []

    def run_worker(binding, *, arguments, catalog, on_retry=None):
        del on_retry
        engine_directories.append(arguments.engine_dir)
        artifact = arguments.engine_dir / "model-a.bundle"
        assert not artifact.exists()
        artifact.write_text("engine", encoding="utf-8")
        cache_blob = arguments.hf_cache_dir / "blob"
        if binding.workload == "suite-b":
            assert cache_blob.is_file()
        cache_blob.write_text("cache", encoding="utf-8")
        case_dir = trtmc_validate._case_directory(arguments.output, binding)
        case_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "model": binding.model,
            "workload": binding.workload,
            "execution": {"status": "completed"},
            "validation": {"status": "passed"},
        }
        (case_dir / "comparison.json").write_text(
            json.dumps(result),
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(
        trtmc_validate,
        "_run_supervised_binding_with_retries",
        run_worker,
    )
    monkeypatch.setattr(trtmc_validate, "write_run_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(trtmc_validate, "finalize_run_metadata", lambda *args: None)
    monkeypatch.setattr(
        trtmc_validate,
        "write_report",
        lambda output: (
            output / "report.json",
            output / "report.html",
            {"model_source_identity": {"consistent": True}},
        ),
    )
    monkeypatch.setattr(trtmc_validate, "_print_result", lambda *args: None)

    returncode = trtmc_validate._run_all_bindings(
        bindings,
        arguments=arguments,
        catalog={"sample_limits": {"suite-a": 1, "suite-b": 1}},
    )

    assert returncode == 0
    assert engine_directories == [
        tmp_path / "work" / "model-a" / "bindings" / "suite-a" / "engines",
        tmp_path / "work" / "model-a" / "bindings" / "suite-b" / "engines",
    ]
    for suite in ("suite-a", "suite-b"):
        assert not (tmp_path / "work" / "model-a" / "bindings" / suite / "engines").exists()
        result = json.loads(
            (tmp_path / f"results/model-a/{suite}/comparison.json").read_text(encoding="utf-8")
        )
        assert result["resource_cleanup"]["engine"]["status"] == "deleted"
    first_result = json.loads(
        (tmp_path / "results/model-a/suite-a/comparison.json").read_text(encoding="utf-8")
    )
    assert first_result["resource_cleanup"]["hf_cache"]["status"] == "retained_until_model_complete"
    final_result = json.loads(
        (tmp_path / "results/model-a/suite-b/comparison.json").read_text(encoding="utf-8")
    )
    assert final_result["resource_cleanup"]["hf_cache"]["status"] == "deleted"
    assert not (tmp_path / "work" / "model-a").exists()


def test_binding_failure_retains_engine_and_per_model_hf_cache(tmp_path):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--model-work-dir",
            str(tmp_path / "work"),
            "--engine-retention",
            "delete_on_pass",
            "--hf-cache-mode",
            "per_model",
            "--hf-cache-retention",
            "delete_on_pass",
        ]
    )
    binding = trtmc_validate.Binding("model-a", "suite-a")
    selected, binding_work, model_work = trtmc_validate._binding_resource_arguments(
        arguments,
        binding,
    )
    assert binding_work is not None
    assert model_work is not None
    (selected.engine_dir / "model.bundle").write_text("engine", encoding="utf-8")
    (selected.hf_cache_dir / "blob").write_text("cache", encoding="utf-8")

    engine_cleanup = trtmc_validate._cleanup_binding_engine(
        arguments,
        binding_work,
        passed=False,
    )
    hf_cleanup = trtmc_validate._cleanup_model_hf_cache(
        arguments,
        model_work,
        passed=False,
        model_complete=True,
    )

    assert engine_cleanup["status"] == "retained"
    assert hf_cleanup["status"] == "retained"
    assert (binding_work / "engines/model.bundle").is_file()
    assert (model_work / "hf-cache/blob").is_file()


def test_per_model_hf_cache_hardlinks_seed_and_deletes_only_working_copy(tmp_path):
    seed = tmp_path / "seed"
    blob = seed / "hub/models--org--model/blobs/content"
    blob.parent.mkdir(parents=True)
    blob.write_text("weights", encoding="utf-8")
    snapshot = seed / "hub/models--org--model/snapshots/revision"
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").symlink_to("../../blobs/content")
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--storage-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "results"),
            "--model-work-dir",
            str(tmp_path / "work"),
            "--engine-dir",
            str(tmp_path / "engines"),
            "--engine-retention",
            "delete_always",
            "--hf-cache-mode",
            "per_model",
            "--hf-cache-retention",
            "delete_always",
            "--hf-cache-seed-dir",
            str(seed),
            "--reference-cache-dir",
            str(tmp_path / "references"),
        ]
    )
    trtmc_validate._prepare_run_directories(arguments)

    selected, _, model_work = trtmc_validate._binding_resource_arguments(
        arguments,
        trtmc_validate.Binding("model-a", "suite-a"),
    )

    linked_blob = selected.hf_cache_dir / "hub/models--org--model/blobs/content"
    linked_snapshot = selected.hf_cache_dir / "hub/models--org--model/snapshots/revision/model.bin"
    assert linked_blob.stat().st_ino == blob.stat().st_ino
    assert linked_snapshot.is_symlink()
    assert linked_snapshot.resolve() == linked_blob
    cleanup = trtmc_validate._cleanup_model_hf_cache(
        arguments,
        model_work,
        passed=False,
        model_complete=True,
    )
    assert cleanup["status"] == "deleted"
    assert blob.read_text(encoding="utf-8") == "weights"


def test_per_model_hf_cache_accepts_hub_cache_as_seed(tmp_path):
    seed = tmp_path / "seed-hub"
    blob = seed / "models--org--model/blobs/content"
    blob.parent.mkdir(parents=True)
    blob.write_text("weights", encoding="utf-8")
    snapshot = seed / "models--org--model/snapshots/revision"
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").symlink_to("../../blobs/content")
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--storage-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "results"),
            "--model-work-dir",
            str(tmp_path / "work"),
            "--engine-dir",
            str(tmp_path / "engines"),
            "--engine-retention",
            "delete_always",
            "--hf-cache-mode",
            "per_model",
            "--hf-cache-retention",
            "delete_always",
            "--hf-cache-seed-dir",
            str(seed),
            "--reference-cache-dir",
            str(tmp_path / "references"),
        ]
    )
    trtmc_validate._prepare_run_directories(arguments)

    selected, _, _ = trtmc_validate._binding_resource_arguments(
        arguments,
        trtmc_validate.Binding("model-a", "suite-a"),
    )

    linked_blob = selected.hf_cache_dir / "hub/models--org--model/blobs/content"
    linked_snapshot = selected.hf_cache_dir / "hub/models--org--model/snapshots/revision/model.bin"
    assert linked_blob.stat().st_ino == blob.stat().st_ino
    assert linked_snapshot.resolve() == linked_blob


def test_prepare_hf_on_demand_uses_binding_cache_and_selected_model(
    tmp_path,
    monkeypatch,
):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--prepare-hf-on-demand",
            "--hf-python",
            sys.executable,
            "--output",
            str(tmp_path / "results"),
        ]
    )
    arguments.hf_cache_dir = tmp_path / "hf-home"
    arguments.hf_cache_dir.mkdir()
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(trtmc_validate.subprocess, "run", run)

    trtmc_validate._prepare_hf_on_demand(
        trtmc_validate.Binding("model-a", "suite-a"),
        arguments,
    )

    command, kwargs = calls[0]
    assert command[:2] == [sys.executable, str(trtmc_validate.HF_WARM_SCRIPT)]
    assert command[command.index("--models-file") + 1].endswith(
        "model-a/suite-a/hf_prepare.models.txt"
    )
    assert "--strict" in command
    assert "--fail-fast" in command
    assert kwargs["env"]["HF_HOME"] == str(arguments.hf_cache_dir)
    assert (tmp_path / "results/model-a/suite-a/hf_prepare.models.txt").read_text(
        encoding="utf-8"
    ) == "model-a\n"


def test_worker_command_propagates_on_demand_hf_preparation(tmp_path):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--prepare-hf-on-demand",
            "--output",
            str(tmp_path / "results"),
        ]
    )

    command = trtmc_validate._worker_command(
        trtmc_validate.Binding("model-a", "suite-a"),
        arguments,
    )

    assert "--prepare-hf-on-demand" in command


def test_worker_command_propagates_hf_reference_device(tmp_path):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--hf-device",
            "cpu",
            "--hf-device-map",
            "balanced",
            "--output",
            str(tmp_path / "results"),
        ]
    )

    command = trtmc_validate._worker_command(
        trtmc_validate.Binding("model-a", "suite-a"),
        arguments,
    )

    assert command[command.index("--hf-device") + 1] == "cpu"
    assert command[command.index("--hf-device-map") + 1] == "balanced"


def test_hf_cache_seed_requires_per_model_mode(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    arguments = trtmc_validate.build_parser().parse_args(
        ["--all", "--hf-cache-seed-dir", str(seed)]
    )

    with pytest.raises(trtmc_validate.ValidationError, match="requires.*per_model"):
        trtmc_validate._prepare_run_directories(arguments)


def test_hf_cache_seed_must_be_disjoint_from_model_work(tmp_path):
    work = tmp_path / "work"
    seed = work / "seed"
    seed.mkdir(parents=True)
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--model-work-dir",
            str(work),
            "--hf-cache-mode",
            "per_model",
            "--hf-cache-seed-dir",
            str(seed),
        ]
    )

    with pytest.raises(trtmc_validate.ValidationError, match="must be disjoint"):
        trtmc_validate._prepare_run_directories(arguments)


def test_resume_existing_keeps_terminal_binding_without_rerunning_worker(
    tmp_path,
    monkeypatch,
):
    revision = "a" * 40
    output = tmp_path / "results"
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--binding",
            "model-a=suite-a",
            "--output",
            str(output),
            "--model-work-dir",
            str(tmp_path / "work"),
            "--engine-retention",
            "delete_on_pass",
            "--reference-cache-dir",
            str(tmp_path / "references"),
            "--resume-existing",
        ]
    )
    binding = trtmc_validate.Binding("model-a", "suite-a")
    case_dir = trtmc_validate._case_directory(output, binding)
    case_dir.mkdir(parents=True)
    (output / "run.json").write_text(
        json.dumps(
            {
                    "source_revision": revision,
                "command": "tools/trtmc_validate.py --binding model-a=suite-a "
                f"--output {output} --model-work-dir {tmp_path / 'work'} "
                "--engine-retention delete_on_pass "
                f"--reference-cache-dir {tmp_path / 'references'}",
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                    "model": "model-a",
                    "workload": "suite-a",
                    "source_revision": revision,
                "execution": {"status": "completed"},
                "validation": {"status": "passed"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(trtmc_validate, "_source_revision", lambda: revision)
    monkeypatch.setattr(
        trtmc_validate.sys,
        "argv",
        [
            "tools/trtmc_validate.py",
            "--binding",
            "model-a=suite-a",
            "--output",
            str(output),
            "--model-work-dir",
            str(tmp_path / "work"),
            "--engine-retention",
            "delete_on_pass",
            "--reference-cache-dir",
            str(tmp_path / "references"),
            "--resume-existing",
        ],
    )
    monkeypatch.setattr(
        trtmc_validate,
        "_run_supervised_binding_with_retries",
        lambda *args, **kwargs: pytest.fail("terminal binding was rerun"),
    )
    monkeypatch.setattr(trtmc_validate, "write_run_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(trtmc_validate, "finalize_run_metadata", lambda *args: None)
    monkeypatch.setattr(
        trtmc_validate,
        "write_report",
        lambda output: (
            output / "report.json",
            output / "report.html",
            {"model_source_identity": {"consistent": True}},
        ),
    )
    monkeypatch.setattr(trtmc_validate, "_print_result", lambda *args: None)

    returncode = trtmc_validate._run_all_bindings(
        [binding],
        arguments=arguments,
        catalog={"sample_limits": {"suite-a": 1}},
    )

    assert returncode == 0
    result = json.loads((case_dir / "comparison.json").read_text(encoding="utf-8"))
    assert result["resource_cleanup"]["engine"]["status"] == "deleted"
    receipt = trtmc_validate.ExecutionLedger.load(output, task_kind="accuracy").receipt(
        "model-a::suite-a"
    )
    assert receipt["state"] == "terminal"
    comparison_mtime = (case_dir / "comparison.json").stat().st_mtime_ns

    assert (
        trtmc_validate._run_all_bindings(
            [binding],
            arguments=arguments,
            catalog={"sample_limits": {"suite-a": 1}},
        )
        == 0
    )
    assert (case_dir / "comparison.json").stat().st_mtime_ns == comparison_mtime


def test_accuracy_report_is_rebuilt_from_ordered_live_receipts(tmp_path):
    cases = [
        {
            "id": "model-a::suite-a",
            "result_path": "model-a/suite-a/comparison.json",
            "report": {"model": "model-a", "workload": "suite-a", "sample_limit": 20},
        },
        {
            "id": "model-b::suite-b",
            "result_path": "model-b/suite-b/comparison.json",
            "report": {"model": "model-b", "workload": "suite-b", "sample_limit": 10},
        },
    ]
    ledger = trtmc_validate.ExecutionLedger.open(
        tmp_path,
        campaign_id="run-1",
        task_kind="accuracy",
        fingerprint="revision-1",
        cases=cases,
    )
    result = trtmc_validate._normalize_result(
        {
            "model": "model-a",
            "workload": "suite-a",
            "execution": {"status": "completed", "exit_code": 0},
            "comparison": {
                "status": "agreement",
                "mode": "exact",
                "primary_metric": None,
                "metrics": {},
                "failures": [],
            },
            "validation": {"status": "passed"},
            "precision_contract": {
                "reference_precision": "fp16",
                "trtmc_base_precision": "fp16",
            },
        }
    )
    ledger.begin("model-a::suite-a", stage="candidate")
    ledger.finish("model-a::suite-a", result="green", payload=result)

    _, _, report = trtmc_validate.write_report(tmp_path)

    assert [(row["id"], row["state"], row["result"]) for row in report["results"]] == [
        ("model-a::suite-a", "terminal", "green"),
        ("model-b::suite-b", "pending", None),
    ]
    assert [row["progress"] for row in report["results"]] == [
        {"stage": "candidate", "attempt": 1},
        {"stage": None, "attempt": 0},
    ]
    assert (
        json.loads((tmp_path / "model-a/suite-a/comparison.json").read_text(encoding="utf-8"))
        == result
    )
    assert report["accounting"]["progress"] == {
        "pending": 1,
        "running": 0,
        "terminal": 1,
    }


def test_accuracy_report_exposes_backend_sample_acceptance(tmp_path):
    cases = [
        {
            "id": "model-a::suite-a",
            "result_path": "model-a/suite-a/comparison.json",
            "report": {"model": "model-a", "workload": "suite-a", "sample_limit": 20},
        }
    ]
    ledger = trtmc_validate.ExecutionLedger.open(
        tmp_path,
        campaign_id="run-1",
        task_kind="accuracy",
        fingerprint="revision-1",
        cases=cases,
    )
    result = trtmc_validate._normalize_result(
        {
            "model": "model-a",
            "workload": "suite-a",
            "execution": {"status": "completed", "exit_code": 0},
            "comparison": {
                "status": "agreement",
                "mode": "mcq",
                "primary_metric": None,
                "metrics": {
                    "prediction_agreement_rate": 0.95,
                    "accuracy_drop_from_hf": 0.0,
                },
                "failures": [],
            },
            "validation": {"status": "passed"},
            "precision_contract": {
                "reference_precision": "fp16",
                "trtmc_base_precision": "fp16",
            },
            "raw_result": {
                "configured_gates": {"max_accuracy_drop_from_hf": 0.01},
                "gate_policy": "blocking",
                "sample_acceptance": {
                    "sample_count": 20,
                    "passed_count": 19,
                    "failed_count": 1,
                    "min_pass_rate": 0.98,
                    "min_allowed_failures": 1,
                    "allowed_failures": 1,
                    "verdict": "pass",
                    "issues": [],
                },
            },
            "reproduce": {
                "dataset": {
                    "sample_limit": 20,
                    "prepared_input_count": 20,
                }
            },
        }
    )
    ledger.begin("model-a::suite-a", stage="compare")
    ledger.finish("model-a::suite-a", result="green", payload=result)

    _, _, report = trtmc_validate.write_report(tmp_path)

    row = report["results"][0]
    assert row["result"] == "green"
    assert row["comparison"]["sample_acceptance"] == {
        "sample_count": 20,
        "passed_count": 19,
        "failed_count": 1,
        "min_pass_rate": 0.98,
        "min_allowed_failures": 1,
        "allowed_failures": 1,
        "verdict": "pass",
        "issues": [],
    }
    assert row["comparison"]["gate_evaluation"]["status"] == "pass"


def test_accuracy_shadow_gate_uses_valid_pairs_not_prepared_inputs(tmp_path):
    case_dir = tmp_path / "model-a" / "suite-a"
    case_dir.mkdir(parents=True)
    result_path = case_dir / "comparison.json"
    result_path.write_text("{}", encoding="utf-8")
    result = trtmc_validate._normalize_result(
        {
            "model": "model-a",
            "workload": "suite-a",
            "execution": {"status": "completed", "exit_code": 0},
            "comparison": {
                "status": "disagreement",
                "mode": "mcq",
                "primary_metric": None,
                "metrics": {
                    "prediction_agreement_rate": 18 / 19,
                    "valid_count": 19,
                },
                "failures": [],
            },
            "validation": {"status": "failed"},
            "raw_result": {
                "configured_gates": {"min_prediction_agreement": 0.95},
                "gate_policy": "blocking",
            },
            "reproduce": {
                "dataset": {
                    "sample_limit": 20,
                    "prepared_input_count": 20,
                }
            },
        }
    )

    public = trtmc_validate._public_accuracy_result(tmp_path, result_path, result)

    assert public["samples"] == {"planned": 20, "evaluated": 20}
    evaluation = public["comparison"]["gate_evaluation"]
    assert evaluation["sample_count"] == 19
    assert evaluation["checks"][0]["effective"]["observed_failures"] == 1


def test_accuracy_shadow_gate_preserves_worst_nested_metric(tmp_path):
    case_dir = tmp_path / "model-a" / "suite-a"
    case_dir.mkdir(parents=True)
    result_path = case_dir / "comparison.json"
    result_path.write_text("{}", encoding="utf-8")
    result = trtmc_validate._normalize_result(
        {
            "model": "model-a",
            "workload": "suite-a",
            "execution": {"status": "completed", "exit_code": 0},
            "raw_result": {
                "status": "failed",
                "valid_count": 5,
                "metrics": {"pixel_mean": {"mean": 0.5, "min": 0.1, "max": 0.9}},
                "configured_gates": {"max_pixel_mean": 0.85},
                "gate_policy": "blocking",
            },
            "reproduce": {"dataset": {"sample_limit": 5, "prepared_input_count": 5}},
        }
    )

    public = trtmc_validate._public_accuracy_result(tmp_path, result_path, result)

    evaluation = public["comparison"]["gate_evaluation"]
    assert evaluation["status"] == "fail"
    assert evaluation["checks"][0]["metric"] == "max_pixel_mean"
    assert evaluation["checks"][0]["actual"] == 0.9


def test_accuracy_adapter_resumes_an_interrupted_case_as_a_new_attempt(tmp_path, monkeypatch):
    output = tmp_path / "results"
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--binding",
            "model-a=suite-a",
            "--output",
            str(output),
            "--model-work-dir",
            str(tmp_path / "work"),
            "--reference-cache-dir",
            str(tmp_path / "references"),
            "--engine-retention",
            "retain",
            "--hf-cache-retention",
            "retain",
        ]
    )
    binding = trtmc_validate.Binding("model-a", "suite-a")
    catalog = {"sample_limits": {"suite-a": 1}}
    monkeypatch.setattr(trtmc_validate, "write_run_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(trtmc_validate, "finalize_run_metadata", lambda *args: None)
    monkeypatch.setattr(trtmc_validate, "_print_result", lambda *args: None)
    monkeypatch.setattr(trtmc_validate, "_check_free_space", lambda *args: 100.0)
    monkeypatch.setattr(
        trtmc_validate,
        "_run_supervised_binding_with_retries",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        trtmc_validate._run_all_bindings([binding], arguments=arguments, catalog=catalog)
    ledger = trtmc_validate.ExecutionLedger.load(output, task_kind="accuracy")
    running = ledger.receipt("model-a::suite-a")
    assert running["state"] == "running"
    evidence = running["attempts"][0]["evidence"]
    assert evidence["commands"]["worker"]["rendered"]
    assert (output / evidence["logs"][0]["href"]).is_file()
    live = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert live["results"][0]["progress"] == {"stage": "compare", "attempt": 1}
    assert live["results"][0]["commands"]["worker"]["rendered"]

    arguments.resume_existing = True
    monkeypatch.setattr(trtmc_validate, "_validate_resume_request", lambda *args: None)
    monkeypatch.setattr(
        trtmc_validate,
        "_run_supervised_binding_with_retries",
        lambda *args, **kwargs: trtmc_validate._normalize_result(
            {
                "model": "model-a",
                "workload": "suite-a",
                "execution": {"status": "completed", "exit_code": 0},
                "comparison": {
                    "status": "agreement",
                    "mode": "exact",
                    "primary_metric": None,
                    "metrics": {},
                    "failures": [],
                },
                "validation": {"status": "passed"},
                "precision_contract": {
                    "reference_precision": "fp16",
                    "trtmc_base_precision": "fp16",
                },
            }
        ),
    )

    assert trtmc_validate._run_all_bindings([binding], arguments=arguments, catalog=catalog) == 0
    receipt = ledger.receipt("model-a::suite-a")
    assert receipt["result"] == "green"
    assert [(attempt["attempt"], attempt["state"]) for attempt in receipt["attempts"]] == [
        (1, "interrupted"),
        (2, "completed"),
    ]


def test_accuracy_adapter_records_each_worker_retry_and_reference_stage(
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "results"
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--binding",
            "model-a=suite-a",
            "--output",
            str(output),
            "--model-work-dir",
            str(tmp_path / "work"),
            "--reference-cache-dir",
            str(tmp_path / "references"),
            "--engine-retention",
            "retain",
            "--hf-cache-retention",
            "retain",
            "--model-attempts",
            "2",
            "--model-retry-delay-seconds",
            "0",
        ]
    )
    binding = trtmc_validate.Binding("model-a", "suite-a")
    monkeypatch.setattr(trtmc_validate, "write_run_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(trtmc_validate, "finalize_run_metadata", lambda *args: None)
    monkeypatch.setattr(trtmc_validate, "_print_result", lambda *args: None)
    monkeypatch.setattr(trtmc_validate, "_check_free_space", lambda *args: 100.0)

    def fail_reference(binding, *, arguments, catalog, attempt):
        case_dir = trtmc_validate._case_directory(arguments.output, binding)
        case_dir.mkdir(parents=True, exist_ok=True)
        worker_log = trtmc_validate._worker_log_path(
            case_dir,
            case_attempt=arguments.case_attempt,
            worker_attempt=attempt,
        )
        worker_log.write_text("ReferenceExecutionError: reference crashed\n", encoding="utf-8")
        result = trtmc_validate._normalize_result(
            {
                "model": binding.model,
                "workload": binding.workload,
                "executor": "trtmc_compare",
                "execution": {"status": "error", "exit_code": 1, "retryable": True},
                "comparison": {
                    "status": "not_run",
                    "mode": "",
                    "primary_metric": None,
                    "metrics": {},
                    "failures": [],
                },
                "validation": {"status": "failed"},
                "raw_result": {
                    "status": "failed",
                    "error_type": "ReferenceExecutionError",
                    "error": "HF reference subprocess failed rc=-11",
                },
                "worker_log": str(worker_log),
            }
        )
        (case_dir / "comparison.json").write_text(
            json.dumps(result),
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(trtmc_validate, "_run_supervised_binding", fail_reference)

    assert (
        trtmc_validate._run_all_bindings(
            [binding],
            arguments=arguments,
            catalog={"sample_limits": {"suite-a": 1}},
        )
        == 1
    )

    receipt = trtmc_validate.ExecutionLedger.load(
        output,
        task_kind="accuracy",
    ).receipt("model-a::suite-a")
    assert receipt["result"] == "white"
    assert receipt["stage"] == "reference"
    assert [
        (attempt["attempt"], attempt["state"], attempt["stage"]) for attempt in receipt["attempts"]
    ] == [
        (1, "failed", "reference"),
        (2, "failed", "reference"),
    ]
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["results"][0]["progress"] == {
        "stage": "reference",
        "attempt": 2,
    }
    assert report["results"][0]["issue"]["stage"] == "reference"


def test_resume_existing_allows_a_new_execution_revision(tmp_path, monkeypatch):
    output = tmp_path / "results"
    output.mkdir()
    (output / "run.json").write_text(
        json.dumps(
            {
                "source_revision": "a" * 40,
                "command": "tools/trtmc_validate.py --model model-a",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(trtmc_validate, "_source_revision", lambda: "b" * 40)
    monkeypatch.setattr(
        trtmc_validate.sys,
        "argv",
        ["tools/trtmc_validate.py", "--model", "model-a", "--resume-existing"],
    )

    trtmc_validate._validate_resume_request(output)


def test_resume_command_ignores_model_invalidation_control() -> None:
    assert trtmc_validate._resume_command(
        "tools/trtmc_validate.py --model model-a --resume-existing "
        "--invalidate-model model-a --verbose"
    ) == ["tools/trtmc_validate.py", "--model", "model-a"]


def test_resume_existing_rejects_different_command(tmp_path, monkeypatch):
    output = tmp_path / "results"
    output.mkdir()
    (output / "run.json").write_text(
        json.dumps(
            {
                "source_revision": "same-revision",
                "command": "tools/trtmc_validate.py --model model-a --limit 5",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(trtmc_validate, "_source_revision", lambda: "same-revision")
    monkeypatch.setattr(
        trtmc_validate.sys,
        "argv",
        [
            "tools/trtmc_validate.py",
            "--model",
            "model-a",
            "--limit",
            "10",
            "--resume-existing",
        ],
    )

    with pytest.raises(trtmc_validate.ValidationError, match="different resolved command"):
        trtmc_validate._validate_resume_request(output)


def test_shared_hf_cache_cannot_be_deleted_by_accuracy_runner(tmp_path):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--output",
            str(tmp_path / "results"),
            "--engine-dir",
            str(tmp_path / "engines"),
            "--reference-cache-dir",
            str(tmp_path / "references"),
            "--hf-cache-retention",
            "delete_always",
        ]
    )

    with pytest.raises(trtmc_validate.ValidationError, match="shared Hugging Face"):
        trtmc_validate._prepare_run_directories(arguments)


def test_storage_root_rejects_reference_source_cache_outside_root(tmp_path):
    storage = tmp_path / "nvme"
    storage.mkdir()
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--storage-root",
            str(storage),
            "--output",
            str(storage / "results"),
            "--engine-dir",
            str(storage / "engines"),
            "--reference-cache-dir",
            str(storage / "references"),
            "--reference-source-cache-dir",
            str(tmp_path / "outside-references"),
        ]
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="reference source cache directory must stay below storage root",
    ):
        trtmc_validate._prepare_run_directories(arguments)


def test_resolve_binding_keeps_unimplemented_model_visible_but_not_runnable():
    catalog = {
        "models": {
            "model-a": {
                "not_compared_reason": "Reference comparator is missing.",
            }
        }
    }

    binding = trtmc_validate.resolve_binding(catalog, "model-a")

    assert binding == trtmc_validate.Binding(
        "model-a",
        None,
        "Reference comparator is missing.",
    )
    assert not binding.runnable
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="has no reference-consistency workloads",
    ):
        trtmc_validate.resolve_binding(catalog, "model-a", "workload-a")


def test_catalog_rejects_e2e_as_reference_consistency_workload(tmp_path):
    catalog_path = tmp_path / "model_workloads.yaml"
    catalog_path.write_text(
        """
version: 1
sample_limits:
  workload-a: 1
models:
  model-a:
    workloads: [e2e]
""",
        encoding="utf-8",
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="cannot use e2e",
    ):
        trtmc_validate.load_catalog(catalog_path)


@pytest.mark.parametrize("invalid_limit", [0, -2])
def test_catalog_sample_limit_is_full_or_positive(tmp_path, invalid_limit):
    catalog_path = tmp_path / "model_workloads.yaml"
    catalog_path.write_text(
        f"""
version: 1
sample_limits:
  workload-a: {invalid_limit}
models:
  model-a:
    workloads: [workload-a]
""",
        encoding="utf-8",
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="must be -1 or a positive integer",
    ):
        trtmc_validate.load_catalog(catalog_path)


@pytest.mark.parametrize(
    "obsolete_field",
    ["default", "additional_workloads", "diagnostic_workloads"],
)
def test_catalog_rejects_obsolete_workload_categories(tmp_path, obsolete_field):
    catalog_path = tmp_path / "model_workloads.yaml"
    catalog_path.write_text(
        f"""
version: 1
sample_limits:
  workload-a: 5
models:
  model-a:
    workloads: [workload-a]
    {obsolete_field}: workload-a
""",
        encoding="utf-8",
    )

    with pytest.raises(trtmc_validate.ValidationError, match="uses obsolete fields"):
        trtmc_validate.load_catalog(catalog_path)


def test_catalog_rejects_cache_identity_across_different_reference_contracts(
    monkeypatch,
) -> None:
    catalog = {
        "models": {
            "model-a": {
                "workloads": ["workload-a"],
                "reference_cache_identity": "shared-reference",
            },
            "model-b": {
                "workloads": ["workload-a"],
                "reference_cache_identity": "shared-reference",
            },
        }
    }
    task_models = {
        "model-a": {
            "hf_id": "org/model-a",
            "family": "family",
            "reference_backend": "hf_transformers",
            "reference_family": "causal",
        },
        "model-b": {
            "hf_id": "org/model-b",
            "family": "family",
            "reference_backend": "hf_transformers",
            "reference_family": "causal",
        },
    }
    monkeypatch.setattr(
        trtmc_validate.validation_catalog,
        "suite_match_reason",
        lambda _suite, _model: (True, ""),
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="spans different reference contracts",
    ):
        trtmc_validate.audit_workload_compatibility(
            catalog,
            suites={"workload-a": {}},
            task_models=task_models,
        )


def test_resolve_sample_limit_uses_workload_policy_and_cli_override():
    catalog = {
        "sample_limits": {"workload-a": 50, "workload-all": -1},
        "models": {
            "model-a": {
                "workloads": ["workload-a"],
            },
            "model-not-compared": {
                "not_compared_reason": "Reference comparator is missing.",
            },
        },
    }

    assert (
        trtmc_validate.resolve_sample_limit(
            catalog,
            trtmc_validate.Binding("model-a", "workload-a"),
            None,
        )
        == 50
    )
    assert (
        trtmc_validate.resolve_sample_limit(
            catalog,
            trtmc_validate.Binding("model-a", "workload-a"),
            7,
        )
        == 7
    )
    assert (
        trtmc_validate.resolve_sample_limit(
            catalog,
            trtmc_validate.Binding("model-a", "workload-a"),
            0,
        )
        == 0
    )
    assert (
        trtmc_validate.resolve_sample_limit(
            catalog,
            trtmc_validate.Binding("model-a", "workload-all"),
            None,
        )
        == 0
    )
    assert (
        trtmc_validate.resolve_sample_limit(
            catalog,
            trtmc_validate.Binding("model-a", "workload-a"),
            -1,
        )
        == 0
    )
    with pytest.raises(trtmc_validate.ValidationError, match="-1 or greater"):
        trtmc_validate.resolve_sample_limit(
            catalog,
            trtmc_validate.Binding("model-a", "workload-a"),
            -2,
        )
    assert (
        trtmc_validate.resolve_sample_limit(
            catalog,
            trtmc_validate.Binding(
                "model-not-compared",
                None,
                "Reference comparator is missing.",
            ),
            7,
        )
        == 0
    )
    assert (
        trtmc_validate.resolve_sample_limit(
            catalog,
            trtmc_validate.Binding(
                "model-not-compared",
                None,
                "Reference comparator is missing.",
            ),
            None,
        )
        == 0
    )


def test_all_defaults_to_continue_and_accepts_stop_policy():
    parser = trtmc_validate.build_parser()

    default = parser.parse_args(["--all"])
    stop = parser.parse_args(["--all", "--on-model-failure", "stop"])

    assert default.on_model_failure == "continue"
    assert stop.on_model_failure == "stop"
    assert default.model_attempts == 2
    assert default.model_retry_delay_seconds == 5.0
    assert default.model_timeout_seconds == 0.0
    assert default.reference_source_cache_dir is None
    assert default.reused_bundle_revalidation_limit == 1
    assert default.reused_bundle_revalidation_attempts_used == 0


def test_reused_bundle_revalidation_limit_rejects_negative_values():
    parser = trtmc_validate.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--all", "--reused-bundle-revalidation-limit", "-1"])


@pytest.mark.parametrize(
    ("policy", "expected_models"),
    [
        ("continue", ["model-a", "model-b"]),
        ("stop", ["model-a"]),
    ],
)
def test_all_supervisor_applies_model_failure_policy(
    tmp_path,
    monkeypatch,
    policy,
    expected_models,
):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--on-model-failure",
            policy,
            "--output",
            str(tmp_path / "results"),
            "--engine-dir",
            str(tmp_path / "engines"),
            "--reference-cache-dir",
            str(tmp_path / "references"),
        ]
    )
    bindings = [
        trtmc_validate.Binding("model-a", "workload-a"),
        trtmc_validate.Binding("model-b", "workload-b"),
    ]
    catalog = {
        "sample_limits": {
            "workload-a": 5,
            "workload-b": 10,
        }
    }
    attempted = []

    def run_worker(binding, *, arguments, catalog, on_retry=None):
        del on_retry
        attempted.append(binding.model)
        status = "failed" if binding.model == "model-a" else "passed"
        return {
            "model": binding.model,
            "workload": binding.workload,
            "execution": {"status": "completed"},
            "validation": {"status": status},
        }

    monkeypatch.setattr(
        trtmc_validate,
        "_run_supervised_binding_with_retries",
        run_worker,
    )
    monkeypatch.setattr(
        trtmc_validate,
        "write_run_metadata",
        lambda output, **_kwargs: output,
    )
    monkeypatch.setattr(
        trtmc_validate,
        "finalize_run_metadata",
        lambda output: output,
    )
    monkeypatch.setattr(
        trtmc_validate,
        "write_report",
        lambda output: (
            output / "report.json",
            output / "report.html",
            {"model_source_identity": {"consistent": True}},
        ),
    )
    monkeypatch.setattr(trtmc_validate, "_print_result", lambda *args: None)

    returncode = trtmc_validate._run_all_bindings(
        bindings,
        arguments=arguments,
        catalog=catalog,
    )

    assert returncode == 1
    assert attempted == expected_models


def test_supervisor_retries_execution_error_but_not_disagreement(
    tmp_path,
    monkeypatch,
    capsys,
):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--model-retry-delay-seconds",
            "0",
            "--output",
            str(tmp_path / "results"),
            "--engine-dir",
            str(tmp_path / "engines"),
            "--reference-cache-dir",
            str(tmp_path / "references"),
        ]
    )
    binding = trtmc_validate.Binding("model-a", "workload-a")
    attempts = []

    def run_worker(binding, *, arguments, catalog, attempt):
        attempts.append(attempt)
        execution_status = "error" if attempt == 1 else "completed"
        validation_status = "failed" if attempt == 1 else "passed"
        result = {
            "model": binding.model,
            "workload": binding.workload,
            "execution": {
                "status": execution_status,
                "exit_code": 1 if attempt == 1 else 0,
            },
            "validation": {"status": validation_status},
            "raw_result": {
                "status": validation_status,
                "error_type": "WorkerProcessError" if attempt == 1 else "",
                "error": "RuntimeError: stale Python profile" if attempt == 1 else "",
            },
            "worker_log": str(tmp_path / f"worker-{attempt}.log"),
        }
        case_dir = trtmc_validate._case_directory(arguments.output, binding)
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "comparison.json").write_text(
            json.dumps(result),
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(trtmc_validate, "_run_supervised_binding", run_worker)

    result = trtmc_validate._run_supervised_binding_with_retries(
        binding,
        arguments=arguments,
        catalog={"sample_limits": {"workload-a": 5}},
    )

    assert attempts == [1, 2]
    assert result["execution"]["status"] == "completed"
    assert result["execution"]["attempt_count"] == 2
    assert result["execution"]["retry_count"] == 1
    output = capsys.readouterr().out
    assert "Attempt 1/2: FAILED" in output
    assert "Error: RuntimeError: stale Python profile" in output
    assert f"Worker log: {tmp_path / 'worker-1.log'}" in output

    attempts.clear()

    def disagree(binding, *, arguments, catalog, attempt):
        attempts.append(attempt)
        return {
            "model": binding.model,
            "workload": binding.workload,
            "execution": {"status": "completed", "exit_code": 1},
            "validation": {"status": "failed"},
            "raw_result": {"status": "failed"},
            "worker_log": str(tmp_path / "worker-disagreement.log"),
        }

    monkeypatch.setattr(trtmc_validate, "_run_supervised_binding", disagree)

    disagreement = trtmc_validate._run_supervised_binding_with_retries(
        binding,
        arguments=arguments,
        catalog={"sample_limits": {"workload-a": 5}},
    )

    assert attempts == [1]
    assert disagreement["execution"]["status"] == "completed"
    assert disagreement["execution"]["attempt_count"] == 1


def test_supervisor_propagates_revalidation_budget_to_next_worker(
    tmp_path,
    monkeypatch,
):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--model-attempts",
            "1",
            "--output",
            str(tmp_path / "results"),
            "--engine-dir",
            str(tmp_path / "engines"),
            "--reference-cache-dir",
            str(tmp_path / "references"),
        ]
    )
    seen = []

    def run_worker(binding, *, arguments, catalog, attempt):
        command = trtmc_validate._worker_command(binding, arguments)
        option = "--reused-bundle-revalidation-attempts-used"
        seen.append((binding.model, int(command[command.index(option) + 1])))
        case_dir = trtmc_validate._case_directory(
            arguments.output,
            binding,
        )
        case_dir.mkdir(parents=True, exist_ok=True)
        attempted = binding.model == "model-a"
        return {
            "model": binding.model,
            "workload": binding.workload,
            "execution": {"status": "completed", "exit_code": 0},
            "validation": {
                "status": "passed" if attempted else "failed",
            },
            "raw_result": {"status": "passed" if attempted else "failed"},
            "bundle_revalidation": {
                "attempted": attempted,
                "attempt_count": 1 if attempted else 0,
            },
        }

    monkeypatch.setattr(trtmc_validate, "_run_supervised_binding", run_worker)
    catalog = {"sample_limits": {"workload-a": 1}}

    trtmc_validate._run_supervised_binding_with_retries(
        trtmc_validate.Binding("model-a", "workload-a"),
        arguments=arguments,
        catalog=catalog,
    )
    trtmc_validate._run_supervised_binding_with_retries(
        trtmc_validate.Binding("model-b", "workload-a"),
        arguments=arguments,
        catalog=catalog,
    )

    assert seen == [("model-a", 0), ("model-b", 1)]


def test_supervisor_retry_cannot_reset_revalidation_budget(
    tmp_path,
    monkeypatch,
):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--model-attempts",
            "2",
            "--model-retry-delay-seconds",
            "0",
            "--output",
            str(tmp_path / "results"),
            "--engine-dir",
            str(tmp_path / "engines"),
            "--reference-cache-dir",
            str(tmp_path / "references"),
        ]
    )
    binding = trtmc_validate.Binding("model-a", "workload-a")
    seen = []

    def run_worker(binding, *, arguments, catalog, attempt):
        command = trtmc_validate._worker_command(binding, arguments)
        option = "--reused-bundle-revalidation-attempts-used"
        seen.append((attempt, int(command[command.index(option) + 1])))
        case_dir = trtmc_validate._case_directory(
            arguments.output,
            binding,
        )
        case_dir.mkdir(parents=True, exist_ok=True)
        rebuild_failed = attempt == 1
        return {
            "model": binding.model,
            "workload": binding.workload,
            "execution": {
                "status": "error" if rebuild_failed else "completed",
                "exit_code": 1,
            },
            "validation": {"status": "failed"},
            "raw_result": {
                "status": "failed",
                "error_type": ("RebuildExecutionError" if rebuild_failed else ""),
            },
            "bundle_revalidation": {
                "attempted": rebuild_failed,
                "attempt_count": 1 if rebuild_failed else 0,
            },
        }

    monkeypatch.setattr(trtmc_validate, "_run_supervised_binding", run_worker)

    result = trtmc_validate._run_supervised_binding_with_retries(
        binding,
        arguments=arguments,
        catalog={"sample_limits": {"workload-a": 1}},
    )

    assert seen == [(1, 0), (2, 1)]
    assert result["execution"]["attempt_count"] == 2
    assert result["execution"]["retry_count"] == 1


def test_all_supervisor_records_not_compared_without_launching_worker(
    tmp_path,
    monkeypatch,
):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--output",
            str(tmp_path / "results"),
            "--engine-dir",
            str(tmp_path / "engines"),
            "--reference-cache-dir",
            str(tmp_path / "references"),
        ]
    )
    binding = trtmc_validate.Binding(
        "model-a",
        None,
        "Reference comparator is missing.",
    )

    def unexpected_worker(*_args, **_kwargs):
        raise AssertionError("not-compared models must not launch a worker")

    monkeypatch.setattr(
        trtmc_validate,
        "_run_supervised_binding_with_retries",
        unexpected_worker,
    )
    monkeypatch.setattr(
        trtmc_validate,
        "write_run_metadata",
        lambda output, **_kwargs: output,
    )
    monkeypatch.setattr(
        trtmc_validate,
        "finalize_run_metadata",
        lambda output: output,
    )
    monkeypatch.setattr(
        trtmc_validate,
        "write_report",
        lambda output: (
            output / "report.json",
            output / "report.html",
            {"model_source_identity": {"consistent": True}},
        ),
    )
    monkeypatch.setattr(trtmc_validate, "_print_result", lambda *args: None)

    returncode = trtmc_validate._run_all_bindings(
        [binding],
        arguments=arguments,
        catalog={"sample_limits": {}},
    )

    comparison = (
        arguments.output / "model-a" / trtmc_validate.NOT_COMPARED_DIRECTORY / "comparison.json"
    )
    result = json.loads(comparison.read_text(encoding="utf-8"))
    assert returncode == 0
    assert result["execution"]["status"] == "not_run"
    assert result["comparison"]["status"] == "not_run"
    assert result["validation"]["status"] == "not_compared"
    assert result["not_compared_reason"] == "Reference comparator is missing."


def test_supervised_binding_replaces_stale_result_with_worker_crash(tmp_path, monkeypatch):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--output",
            str(tmp_path / "results"),
            "--engine-dir",
            str(tmp_path / "engines"),
            "--reference-cache-dir",
            str(tmp_path / "references"),
            "--local-files-only",
        ]
    )
    binding = trtmc_validate.Binding("model-a", "workload-a")
    catalog = {"sample_limits": {"workload-a": 5}}
    case_dir = arguments.output / binding.model / binding.workload
    case_dir.mkdir(parents=True)
    comparison = case_dir / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "model": binding.model,
                "workload": binding.workload,
                "status": "passed",
            }
        ),
        encoding="utf-8",
    )

    def crash(command, log_path, env):
        log_path.write_text(
            "Traceback (most recent call last):\n"
            "  worker setup failed\n"
            "RuntimeError: required Python profile is not prebuilt\n",
            encoding="utf-8",
        )
        return 2

    monkeypatch.setattr(trtmc_validate, "_run_subprocess", crash)

    result = trtmc_validate._run_supervised_binding(
        binding,
        arguments=arguments,
        catalog=catalog,
    )

    assert result["execution"] == {"status": "error", "exit_code": 2}
    assert result["comparison"]["status"] == "not_run"
    assert result["validation"]["status"] == "failed"
    assert result["raw_result"]["error_type"] == "WorkerProcessError"
    assert result["raw_result"]["error"] == (
        "RuntimeError: required Python profile is not prebuilt"
    )
    assert result["reproduce"]["dataset"]["sample_limit"] == 5
    assert "--model-worker" in result["reproduce"]["dataset"]["command"]
    assert "--local-files-only" in result["reproduce"]["dataset"]["command"]
    assert json.loads(comparison.read_text(encoding="utf-8")) == result


def test_supervised_binding_records_worker_timeout(tmp_path, monkeypatch):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--model-timeout-seconds",
            "42",
            "--output",
            str(tmp_path / "results"),
            "--engine-dir",
            str(tmp_path / "engines"),
            "--reference-cache-dir",
            str(tmp_path / "references"),
        ]
    )
    binding = trtmc_validate.Binding("model-a", "workload-a")

    def timeout(command, log_path, env, timeout_seconds):
        assert timeout_seconds == 42
        log_path.write_text("worker timed out\n", encoding="utf-8")
        raise trtmc_validate.WorkerTimeoutError("model worker exceeded 42 seconds")

    monkeypatch.setattr(trtmc_validate, "_run_supervised_subprocess", timeout)

    result = trtmc_validate._run_supervised_binding(
        binding,
        arguments=arguments,
        catalog={"sample_limits": {"workload-a": 5}},
    )

    assert result["execution"] == {"status": "error", "exit_code": 124}
    assert result["raw_result"]["error_type"] == "WorkerTimeoutError"
    assert result["raw_result"]["error"] == "model worker exceeded 42 seconds"
    assert result["validation"]["status"] == "failed"


def test_supervised_subprocess_terminates_on_timeout(tmp_path):
    log_path = tmp_path / "worker.log"

    with pytest.raises(
        trtmc_validate.WorkerTimeoutError,
        match="exceeded 0.05 seconds",
    ):
        trtmc_validate._run_supervised_subprocess(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            log_path,
            os.environ,
            0.05,
        )

    assert "terminating process group" in log_path.read_text(encoding="utf-8")


def test_supervised_binding_accepts_fresh_worker_result(tmp_path, monkeypatch):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--output",
            str(tmp_path / "results"),
            "--engine-dir",
            str(tmp_path / "engines"),
            "--reference-cache-dir",
            str(tmp_path / "references"),
        ]
    )
    binding = trtmc_validate.Binding("model-a", "workload-a")
    catalog = {"sample_limits": {"workload-a": 5}}
    comparison = arguments.output / binding.model / binding.workload / "comparison.json"

    def pass_worker(command, log_path, env):
        comparison.write_text(
            json.dumps(
                {
                    "model": binding.model,
                    "workload": binding.workload,
                    "status": "passed",
                    "returncode": 0,
                    "raw_result": {"status": "passed"},
                    "reproduce": {
                        "dataset": {
                            "command": "internal worker command",
                            "sample_limit": 5,
                            "prepared_input_count": 5,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(trtmc_validate, "_run_subprocess", pass_worker)

    result = trtmc_validate._run_supervised_binding(
        binding,
        arguments=arguments,
        catalog=catalog,
    )

    assert result["execution"]["status"] == "completed"
    assert result["comparison"]["status"] == "agreement"
    assert result["validation"]["status"] == "passed"
    assert result["worker_log"].endswith("/model-a/workload-a/worker.log")
    assert "--model-worker" not in result["reproduce"]["dataset"]["command"]


def test_all_dry_run_emits_machine_readable_ci_cases(monkeypatch, capsys):
    catalog = {
        "sample_limits": {"workload-a": 5},
        "models": {
            "model-a": {
                "default": "workload-a",
                "workloads": ["workload-a"],
            },
            "model-not-compared": {
                "not_compared_reason": "Reference comparator is missing.",
            },
        },
    }
    monkeypatch.setattr(
        trtmc_validate,
        "_load_validation_inputs",
        lambda arguments: (
            catalog,
            {"workload-a": {}},
            ("model-a", "model-not-compared"),
            {"model-a": {"name": "model-a"}},
        ),
    )

    returncode = trtmc_validate.main(["--all", "--dry-run"])

    assert returncode == 0
    assert json.loads(capsys.readouterr().out) == [
        {
            "model": "model-a",
            "workload": "workload-a",
            "sample_limit": 5,
        },
        {
            "model": "model-not-compared",
            "workload": None,
            "sample_limit": 0,
            "status": "not_compared",
            "reason": "Reference comparator is missing.",
        },
    ]


@pytest.mark.parametrize(
    ("validation_status", "comparison_status", "expected_returncode"),
    [
        ("passed", "agreement", 0),
        ("failed", "disagreement", 1),
    ],
)
def test_single_ci_case_writes_stable_result_and_exit_code(
    tmp_path,
    monkeypatch,
    validation_status,
    comparison_status,
    expected_returncode,
):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "model-a",
            "workload-a",
            "--output",
            str(tmp_path / "results"),
            "--engine-dir",
            str(tmp_path / "engines"),
            "--reference-cache-dir",
            str(tmp_path / "references"),
        ]
    )
    binding = trtmc_validate.Binding("model-a", "workload-a")
    catalog = {"sample_limits": {"workload-a": 5}}
    monkeypatch.setattr(
        trtmc_validate,
        "_runtime_gpu_devices",
        lambda _visible: [
            {
                "cuda_logical_index": 0,
                "nvidia_smi_index": 0,
                "uuid": "GPU-test",
                "name": "NVIDIA test GPU",
                "pci_bus_id": "00000000:01:00.0",
            }
        ],
    )

    def run_binding(binding, *, arguments, task_models, suites):
        result = {
            "schema_version": "trtmc.validation-result/v2",
            "model": binding.model,
            "workload": binding.workload,
            "execution": {"status": "completed", "exit_code": 0},
            "comparison": {
                "status": comparison_status,
                "mode": "test",
                "primary_metric": None,
                "metrics": {},
                "failures": [],
            },
            "validation": {"status": validation_status},
            "reproduce": {
                "dataset": {
                    "command": "python tools/trtmc_validate.py model-a workload-a",
                    "sample_limit": arguments.limit,
                    "prepared_input_count": arguments.limit,
                },
                "hf": [],
                "trtmc": [],
            },
        }
        case_dir = arguments.output / binding.model / binding.workload
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "comparison.json").write_text(
            json.dumps(result),
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(trtmc_validate, "run_binding", run_binding)

    returncode = trtmc_validate._run_bindings(
        [binding],
        arguments=arguments,
        catalog=catalog,
        task_models={},
        suites={"workload-a": {}},
    )

    case_dir = arguments.output / "model-a" / "workload-a"
    assert returncode == expected_returncode
    assert (case_dir / "comparison.json").is_file()
    assert (arguments.output / "report.json").is_file()
    assert (arguments.output / "report.html").is_file()


def test_single_ci_case_returns_exit_two_for_setup_error(monkeypatch):
    def fail(arguments):
        raise trtmc_validate.ValidationError("missing CI dataset")

    monkeypatch.setattr(trtmc_validate, "_load_validation_inputs", fail)

    with pytest.raises(SystemExit) as error:
        trtmc_validate.main(["model-a", "workload-a"])

    assert error.value.code == 2


@pytest.mark.parametrize(
    "reference_backend",
    ["hf_transformers", "torch_reference", "diffusers_reference"],
)
def test_default_reference_backends_share_common_environment(reference_backend):
    assert (
        trtmc_validate._declared_profile(
            family="",
            runtime_strategy="",
            reference_backend=reference_backend,
            execution_profiles=None,
        )
        == trtmc_validate.COMMON_REFERENCE_PROFILE
    )


def test_model_specific_reference_environment_keeps_common_validation_base() -> None:
    profiles = trtmc_validate.binding_profiles(
        trtmc_validate.Binding("elf", "dataset"),
        task_models={
            "elf": {
                "family": "elf_flow",
                "runtime_strategy": "elf_flow",
                "reference_backend": "hf_transformers",
            }
        },
    )

    assert profiles == (
        trtmc_validate.COMMON_REFERENCE_PROFILE,
        "elf_flow_reference",
    )


def test_suite_specific_scorer_environment_is_materialized_on_demand() -> None:
    profiles = trtmc_validate.binding_profiles(
        trtmc_validate.Binding("personaplex-7b", "full-duplex"),
        task_models={
            "personaplex-7b": {
                "family": "personaplex",
                "runtime_strategy": "personaplex_speech_to_speech",
                "reference_backend": "torch_reference",
            }
        },
        suites={
            "full-duplex": {"scoring": {"python_profile": "personaplex_full_duplex_evaluator"}}
        },
    )

    assert profiles == (
        trtmc_validate.COMMON_REFERENCE_PROFILE,
        "personaplex_full_duplex_evaluator",
    )


def test_minimax_h3_vbench_scorer_reuses_common_environment() -> None:
    profiles = trtmc_validate.binding_profiles(
        trtmc_validate.Binding("minimax-h3-768p", "minimax_h3_vbench_siglip_task_quality"),
        task_models={
            "minimax-h3-768p": {
                "family": "minimax_h3",
                "runtime_strategy": "diffusion_minimax_h3",
                "reference_backend": "hf_diffusers",
            }
        },
        suites={
            "minimax_h3_vbench_siglip_task_quality": {
                "scoring": {"python_profile": "reference_common"}
            }
        },
    )

    assert profiles == (
        trtmc_validate.COMMON_REFERENCE_PROFILE,
        "minimax_h3_reference",
    )


def test_ensure_environments_reports_create_only_when_resolver_creates(monkeypatch, capsys):
    calls = 0

    def resolve(name, base_python, *, on_create):
        nonlocal calls
        calls += 1
        if calls == 1:
            on_create(name)
        return f"/profiles/{name}/bin/python"

    monkeypatch.setattr(trtmc_validate, "resolve_profile_python", resolve)

    cold = trtmc_validate.ensure_environments(
        [trtmc_validate.COMMON_REFERENCE_PROFILE],
        "/base/python",
    )
    cold_output = capsys.readouterr().out
    warm = trtmc_validate.ensure_environments(
        [trtmc_validate.COMMON_REFERENCE_PROFILE],
        "/base/python",
    )
    warm_output = capsys.readouterr().out

    assert "Creating reference environment: reference_common" in cold_output
    assert "Using reference environment: /profiles/reference_common/bin/python" in (cold_output)
    assert "Creating reference environment" not in warm_output
    assert "Using reference environment: /profiles/reference_common/bin/python" in (warm_output)
    assert cold.base_python == "/profiles/reference_common/bin/python"
    assert warm.base_python == cold.base_python


def test_reference_sources_create_once_then_reuse(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    source = trtmc_validate.ReferenceSource(
        name="ELF",
        repository="https://example.invalid/ELF.git",
        revision="0123456789abcdef",
        relative_checkout=Path("elf/reference/ELF-0123456789ab"),
        entrypoint=Path("src/entrypoint.py"),
    )
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        if command[1] == "-C":
            checkout = Path(command[2])
            entrypoint = checkout / source.entrypoint
            entrypoint.parent.mkdir(parents=True)
            entrypoint.write_text("# reference\n", encoding="utf-8")
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(trtmc_validate.subprocess, "run", fake_run)

    cold = trtmc_validate._ensure_reference_source(source, tmp_path)
    cold_output = capsys.readouterr().out
    command_count = len(commands)
    warm = trtmc_validate._ensure_reference_source(source, tmp_path)
    warm_output = capsys.readouterr().out

    assert cold == warm == tmp_path / source.relative_checkout
    assert command_count == 2
    assert len(commands) == command_count
    assert "Creating reference source: ELF" in cold_output
    assert f"Using reference source: {cold}" in cold_output
    assert "Creating reference source" not in warm_output
    assert f"Using reference source: {warm}" in warm_output


def test_reference_source_prebuilt_only_rejects_a_missing_checkout(tmp_path: Path) -> None:
    source = trtmc_validate.ReferenceSource(
        name="ELF",
        repository="https://example.invalid/ELF.git",
        revision="0123456789abcdef",
        relative_checkout=Path("elf/reference/ELF-0123456789ab"),
        entrypoint=Path("src/entrypoint.py"),
    )

    with pytest.raises(trtmc_validate.ValidationError, match="prepare"):
        trtmc_validate._ensure_reference_source(source, tmp_path, prebuilt_only=True)

    assert not (tmp_path / source.relative_checkout).exists()


def test_elf_reference_source_is_pinned_to_upstream_pytorch_implementation() -> None:
    assert trtmc_validate.ELF_SOURCE.revision == ("b29d8833609e9ab7f67cd9da39435ac5cea04837")
    assert trtmc_validate.ELF_SOURCE.relative_checkout == Path("elf/reference/ELF-b29d8833609e")


def test_reference_sources_select_model_specific_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared: list[str] = []

    def prepare(source, cache_root):
        prepared.append(source.name)
        checkout = cache_root / source.relative_checkout
        entrypoint = checkout / source.entrypoint
        entrypoint.parent.mkdir(parents=True)
        entrypoint.write_text("# reference\n", encoding="utf-8")
        return checkout

    monkeypatch.setattr(trtmc_validate, "_ensure_reference_source", prepare)

    elf = trtmc_validate.ensure_reference_sources("elf_flow", tmp_path)
    sana = trtmc_validate.ensure_reference_sources(
        "sana_wm",
        tmp_path,
        {
            "repository": trtmc_validate.SANA_WM_SOURCE.repository,
            "revision": trtmc_validate.SANA_WM_SOURCE.revision,
            "relative_path": str(trtmc_validate.SANA_WM_SOURCE.relative_checkout),
            "entrypoint": str(trtmc_validate.SANA_WM_SOURCE.entrypoint),
        },
    )
    wan22 = trtmc_validate.ensure_reference_sources(
        "wan2_2_ti2v",
        tmp_path,
        {
            "repository": "https://example.invalid/Wan2.2.git",
            "revision": "42bf4cfaa384bc21833865abc2f9e6c0e67233dc",
            "relative_path": "wan2_2_ti2v/reference/Wan2.2-42bf4cfaa384",
            "entrypoint": "wan/textimage2video.py",
            "environment_variable": "TRTMC_WAN_REFERENCE_REPO",
        },
    )
    lance = trtmc_validate.ensure_reference_sources(
        "lance",
        tmp_path,
        {
            "repository": "https://example.invalid/Lance.git",
            "revision": "4baeee086648996f6ab12e673cbe461b0b149997",
            "relative_path": "lance/reference/Lance-4baeee086648",
            "entrypoint": "inference_lance.py",
            "environment_variable": "TRTMC_LANCE_REFERENCE_REPO",
        },
    )
    common = trtmc_validate.ensure_reference_sources("bert", tmp_path)

    assert prepared == ["ELF", "sana_wm", "wan2_2_ti2v", "lance"]
    assert elf.elf_reference_repo == tmp_path / trtmc_validate.ELF_SOURCE.relative_checkout
    assert elf.environment["TRTMC_STORAGE_ROOT"] == str(tmp_path)
    assert sana.environment["SANA_WM_SCRIPT"] == str(
        tmp_path
        / trtmc_validate.SANA_WM_SOURCE.relative_checkout
        / trtmc_validate.SANA_WM_SOURCE.entrypoint
    )
    assert common.elf_reference_repo is None
    assert common.environment == {"TRTMC_STORAGE_ROOT": str(tmp_path)}
    assert wan22.environment == {
        "TRTMC_STORAGE_ROOT": str(tmp_path),
        "TRTMC_WAN_REFERENCE_REPO": str(tmp_path / "wan2_2_ti2v/reference/Wan2.2-42bf4cfaa384"),
    }
    assert lance.environment == {
        "TRTMC_STORAGE_ROOT": str(tmp_path),
        "TRTMC_LANCE_REFERENCE_REPO": str(tmp_path / "lance/reference/Lance-4baeee086648"),
    }


def test_reference_sources_keep_outputs_separate_from_pinned_checkouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_cache = tmp_path / "model-work" / "references"
    source_cache = tmp_path / "reference-sources"

    def prepare(source, cache_root):
        checkout = cache_root / source.relative_checkout
        entrypoint = checkout / source.entrypoint
        entrypoint.parent.mkdir(parents=True)
        entrypoint.write_text("# reference\n", encoding="utf-8")
        return checkout

    monkeypatch.setattr(trtmc_validate, "_ensure_reference_source", prepare)

    selection = trtmc_validate.ensure_reference_sources(
        "elf_flow",
        output_cache,
        source_cache_root=source_cache,
    )

    assert selection.environment["TRTMC_STORAGE_ROOT"] == str(output_cache)
    assert selection.elf_reference_repo == (
        source_cache / trtmc_validate.ELF_SOURCE.relative_checkout
    )


def test_reference_sources_reject_incomplete_model_contract(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        trtmc_validate.ValidationError,
        match="wan2_2_ti2v model reference source is missing: entrypoint",
    ):
        trtmc_validate.ensure_reference_sources(
            "wan2_2_ti2v",
            tmp_path,
            {
                "repository": "https://example.invalid/Wan2.2.git",
                "revision": "42bf4cfaa384bc21833865abc2f9e6c0e67233dc",
                "relative_path": "wan2_2_ti2v/reference/Wan2.2-42bf4cfaa384",
            },
        )


def test_print_result_verbose_exposes_raw_commands_and_result_locations(tmp_path, capsys):
    comparison = tmp_path / "comparison.json"
    report = tmp_path / "report.html"
    trtmc_validate._print_result(
        {
            "validation": {"status": "passed"},
            "reproduce": {
                "dataset": {
                    "command": "python tools/trtmc_validate.py model-a --limit 1000",
                    "sample_limit": 1000,
                    "prepared_input_count": 1000,
                },
                "hf": ["python hf_reference.py --model model-a"],
                "trtmc": ["trtmc run --model model-a"],
            },
        },
        comparison,
        report,
        verbose=True,
    )

    output = capsys.readouterr().out
    assert output == (
        "\n"
        "Status: PASSED\n"
        "\n"
        "Reproduce dataset run:\n"
        "  python tools/trtmc_validate.py model-a --limit 1000\n"
        "\n"
        "Reproduce representative HF:\n"
        "  python hf_reference.py --model model-a\n"
        "\n"
        "Reproduce representative TRTMC:\n"
        "  trtmc run --model model-a\n"
        "\n"
        f"Compare result: {comparison}\n"
        f"Report data:   {report.with_name('report.json')}\n"
        f"Report:         {report}\n"
    )
    assert "package" not in output.lower()
    assert "token-agreement" not in output
    assert "env action" not in output.lower()


def test_print_result_does_not_mislabel_validation_wrapper_as_raw_command(tmp_path, capsys):
    comparison = tmp_path / "comparison.json"
    report = tmp_path / "report.html"
    trtmc_validate._print_result(
        {
            "reproduce": {
                "hf": [],
                "trtmc": [],
                "validation": "python tools/trtmc_validate.py model-a",
            }
        },
        comparison,
        report,
        verbose=True,
    )

    output = capsys.readouterr().out
    assert output.count("unavailable; see comparison result") == 3
    assert "python tools/trtmc_validate.py model-a" not in output


def test_print_result_default_is_concise_and_shows_execution_error(tmp_path, capsys):
    comparison = tmp_path / "comparison.json"
    report = tmp_path / "report.html"
    worker_log = tmp_path / "worker.log"

    trtmc_validate._print_result(
        {
            "execution": {"status": "error", "exit_code": 1},
            "validation": {"status": "failed"},
            "raw_result": {
                "error_type": "WorkerProcessError",
                "error": "RuntimeError: required Python profile is not prebuilt",
            },
            "worker_log": str(worker_log),
            "reproduce": {
                "dataset": {"command": "python very-long-worker-command"},
                "hf": [],
                "trtmc": [],
            },
        },
        comparison,
        report,
    )

    output = capsys.readouterr().out
    assert "Status: FAILED" in output
    assert "Error: RuntimeError: required Python profile is not prebuilt" in output
    assert f"Worker log: {worker_log}" in output
    assert f"Compare result: {comparison}" in output
    assert f"Report: {report}" in output
    assert "Reproduce" not in output
    assert "very-long-worker-command" not in output


def test_write_report_links_each_comparison(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    comparison = case_dir / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "family": "example",
                "operation": "generate_audio",
                "task_strategy": "text_to_audio",
                "task_type": "Text → Audio",
                "user_contract": "tts_audio",
                "status": "passed",
                "precision_contract": {
                    "trtmc_base_precision": "fp16",
                    "reference_precision": "fp16",
                },
                "reference_environment": [
                    {"name": "reference_common", "python": "/profiles/python"}
                ],
                "reproduce": {
                    "hf": ["python hf.py"],
                    "trtmc": ["trtmc run"],
                    "dataset": {
                        "command": "python tools/trtmc_validate.py model-a",
                        "sample_limit": 500,
                        "prepared_input_count": 100,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    json_path, html_path, report = trtmc_validate.write_report(tmp_path)

    assert report["summary"] == {
        "cases": 1,
        "execution_completed": 1,
        "execution_errors": 0,
        "agreements": 1,
        "disagreements": 0,
        "not_compared": 0,
        "validation_passed": 1,
        "validation_failed": 0,
        "validation_skipped": 0,
        "selected_samples": 100,
    }
    assert report["validation_status"] == "passed"
    assert report["results"][0]["execution"]["status"] == "completed"
    assert report["results"][0]["comparison"]["status"] == "agreement"
    assert report["results"][0]["validation"]["status"] == "passed"
    assert "status" not in report["results"][0]
    assert report["schema_version"] == "trtmc.qualification-report/v1"
    assert report["accounting"]["selected"] == 1
    assert report["accounting"]["comparable"] == 1
    assert report["accounting"]["outcomes"] == {
        "green": 1,
        "red": 0,
        "white": 0,
        "yellow": 0,
    }
    assert report["results"][0]["result"] == "green"
    assert json_path == tmp_path / "report.json"
    assert html_path == tmp_path / "report.html"
    document = html_path.read_text(encoding="utf-8")
    assert 'data-report="report.json"' in document
    assert "model-a" not in document
    assert (tmp_path / "assets/qualification-report.js").is_file()
    assert (tmp_path / "assets/qualification-report.css").is_file()
    assert report["summary"]["selected_samples"] == 100
    frontend = (tmp_path / "assets/qualification-report.js").read_text(encoding="utf-8")
    assert "Complete qualification register" in frontend
    assert "Vanilla reproduction" in frontend


@pytest.mark.parametrize(
    ("limit", "prepared", "expected"),
    [
        (500, 100, 100),
        (0, 83, 83),
        (5, 0, 0),
    ],
)
def test_selected_sample_count_uses_actual_prepared_count(limit, prepared, expected):
    result = {
        "reproduce": {
            "dataset": {
                "sample_limit": limit,
                "prepared_input_count": prepared,
            }
        }
    }

    assert trtmc_validate._selected_sample_count(result) == expected


def test_write_report_surfaces_quantized_reference_precision_contract(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "quantized-model" / "mmlu_five_shot_mcq"
    case_dir.mkdir(parents=True)
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "quantized-model",
                "workload": "mmlu_five_shot_mcq",
                "status": "passed",
                "raw_result": {
                    "status": "passed",
                    "mode": "mcq",
                    "precision_contract": {
                        "trtmc_base_precision": "bf16",
                        "trtmc_quantization": "fp8",
                        "reference_precision": "bf16",
                        "reference_dtype": "bfloat16",
                        "comparison": "quantized_vs_unquantized_reference",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    _, html_path, report = trtmc_validate.write_report(tmp_path)

    assert report["results"][0]["precision_contract"] == {
        "trtmc_base_precision": "bf16",
        "trtmc_quantization": "fp8",
        "reference_precision": "bf16",
        "reference_dtype": "bfloat16",
        "comparison": "quantized_vs_unquantized_reference",
    }
    assert report["results"][0]["precision"] == {
        "reference": "bf16",
        "candidate": "fp8 (bf16 base)",
    }
    assert "quantized-model" not in html_path.read_text(encoding="utf-8")


def test_report_infers_task_type_for_legacy_standard_result(tmp_path):
    case_dir = tmp_path / "bark-large" / "seedtts_en_tts_intelligibility"
    case_dir.mkdir(parents=True)
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "bark-large",
                "workload": "seedtts_en_tts_intelligibility",
                "status": "passed",
                "reproduce": {
                    "dataset": {
                        "command": "python tools/trtmc_validate.py bark-large",
                        "sample_limit": 3,
                        "prepared_input_count": 3,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    _, html_path, report = trtmc_validate.write_report(tmp_path)

    result = report["results"][0]
    assert result["family"] == "bark"
    assert result["operation"] == "generate_audio"
    assert result["task_strategy"] == "text_to_audio"
    assert result["task_type"] == "Text → Audio"
    assert result["user_contract"] == "tts_audio"
    assert result["samples"] == {"planned": 3, "evaluated": 3}
    assert "bark-large" not in html_path.read_text(encoding="utf-8")
    assert "Samples" in (tmp_path / "assets" / "qualification-report.js").read_text(
        encoding="utf-8"
    )


def test_accuracy_traffic_light_statuses_are_mutually_exclusive():
    def result(validation, comparison):
        return {
            "execution": {"status": "completed"},
            "validation": {"status": validation},
            "comparison": {"status": comparison},
            "precision_contract": {
                "trtmc_base_precision": "fp16",
                "reference_precision": "fp16",
            },
        }

    statuses = [
        trtmc_validate._traffic_light_status(value)
        for value in [
            result("passed", "agreement"),
            result("skipped", "not_run"),
            result("failed", "disagreement"),
            result("not_compared", "not_run"),
        ]
    ]

    assert Counter(statuses) == {
        "green": 1,
        "red": 1,
        "white": 2,
    }


def test_accuracy_result_with_unknown_precision_has_no_result_light() -> None:
    result = {
        "execution": {"status": "completed"},
        "validation": {"status": "passed"},
        "comparison": {"status": "agreement"},
    }

    assert trtmc_validate._traffic_light_status(result) == "white"
    assert trtmc_validate._accuracy_issue(result) == {
        "priority": "P1",
        "stage": "preflight",
        "domain": "policy-config",
        "code": "comparison_contract",
        "message": "Reference and TRTMC compute precision were not both recorded",
    }


def test_accuracy_result_with_incomplete_samples_is_white() -> None:
    result = trtmc_validate._normalize_result(
        {
            "raw_result": {
                "status": "invalid",
                "error_type": "SampleEvidenceError",
                "error": "incomplete_samples",
                "sample_acceptance": {
                    "sample_count": 19,
                    "passed_count": 19,
                    "failed_count": 0,
                    "min_pass_rate": 0.98,
                    "min_allowed_failures": 1,
                    "allowed_failures": 1,
                    "verdict": "invalid",
                    "issues": [{"code": "incomplete_samples", "expected": 20, "actual": 19}],
                },
                "precision_contract": {
                    "reference_precision": "fp16",
                    "trtmc_base_precision": "fp16",
                },
            }
        }
    )

    assert trtmc_validate._traffic_light_status(result) == "white"
    assert trtmc_validate._accuracy_issue(result) == {
        "priority": "P1",
        "stage": "compare",
        "domain": "data-artifact",
        "code": "incomplete_samples",
        "message": "incomplete_samples",
    }


def test_write_report_removes_legacy_platform_exclusion_rows(tmp_path: Path) -> None:
    selected = tmp_path / "model-a" / "suite-a"
    selected.mkdir(parents=True)
    (selected / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "suite-a",
                "execution": {"status": "completed", "exit_code": 0},
                "comparison": {
                    "status": "agreement",
                    "mode": "test",
                    "primary_metric": None,
                    "metrics": {},
                    "failures": [],
                },
                "validation": {"status": "passed"},
            }
        ),
        encoding="utf-8",
    )
    excluded = tmp_path / "excluded-model" / "suite-b"
    excluded.mkdir(parents=True)
    (excluded / "comparison.json").write_text(
        json.dumps(
            {
                "model": "excluded-model",
                "workload": "suite-b",
                "execution": {"status": "not_run", "exit_code": None},
                "comparison": {
                    "status": "not_run",
                    "mode": "platform_exclusion",
                    "primary_metric": None,
                    "metrics": {},
                    "failures": [],
                },
                "validation": {"status": "not_compared"},
                "platform_exclusion": {"reason": "not supported"},
            }
        ),
        encoding="utf-8",
    )

    json_path, _, report = trtmc_validate.write_report(tmp_path)

    assert report["accounting"]["selected"] == 1
    assert [row["model"] for row in report["results"]] == ["model-a"]
    assert "excluded-model" not in json_path.read_text(encoding="utf-8")


def test_accuracy_report_publishes_direct_relative_log_links(tmp_path: Path) -> None:
    case_dir = tmp_path / "model-a" / "suite-a"
    case_dir.mkdir(parents=True)
    (case_dir / "worker.log").write_text("raw worker output\n", encoding="utf-8")
    (case_dir / "execution.log").write_text("raw execution output\n", encoding="utf-8")
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "suite-a",
                "execution": {"status": "error", "exit_code": 1},
                "comparison": {
                    "status": "not_run",
                    "mode": "",
                    "primary_metric": None,
                    "metrics": {},
                    "failures": [],
                },
                "validation": {"status": "failed"},
            }
        ),
        encoding="utf-8",
    )

    _, _, report = trtmc_validate.write_report(tmp_path)

    row = report["results"][0]
    assert row["result"] == "white"
    assert row["issue"]["stage"] == "candidate"
    assert row["debug"]["logs"] == [
        {
            "label": "execution.log",
            "href": "artifacts/cases/model-a/suite-a/logs/execution.log",
        },
        {
            "label": "worker.log",
            "href": "artifacts/cases/model-a/suite-a/logs/worker.log",
        },
    ]
    assert all((tmp_path / item["href"]).is_file() for item in row["debug"]["logs"])


def test_accuracy_report_materializes_symlinked_logs_inside_report(
    tmp_path: Path,
) -> None:
    output = tmp_path / "accuracy"
    case_dir = output / "model-a" / "suite-a"
    case_dir.mkdir(parents=True)
    reference_log = tmp_path / "reference-cache" / "hf_native_run.log"
    reference_log.parent.mkdir()
    reference_log.write_text("reference output\n", encoding="utf-8")
    (case_dir / "hf_native_run.log").symlink_to(reference_log)
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "suite-a",
                "execution": {"status": "completed", "exit_code": 0},
                "comparison": {
                    "status": "agreement",
                    "mode": "test",
                    "primary_metric": None,
                    "metrics": {},
                    "failures": [],
                },
                "validation": {"status": "passed"},
                "precision_contract": {
                    "trtmc_base_precision": "fp16",
                    "reference_precision": "fp16",
                },
            }
        ),
        encoding="utf-8",
    )

    _, _, report = trtmc_validate.write_report(output)

    log = next(
        item
        for item in report["results"][0]["debug"]["logs"]
        if item["label"] == "hf_native_run.log"
    )
    published = output / log["href"]
    assert published.read_text(encoding="utf-8") == "reference output\n"
    assert not published.is_symlink()


def test_run_binding_records_missing_default_dataset_as_preflight_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "model-a",
            "suite-a",
            "--output",
            str(tmp_path / "results"),
            "--dataset-root",
            str(tmp_path / "datasets"),
            "--reference-cache-dir",
            str(tmp_path / "references"),
            "--limit",
            "50",
        ]
    )
    monkeypatch.setattr(
        trtmc_validate,
        "ensure_environments",
        lambda *_args, **_kwargs: pytest.fail(
            "missing datasets must fail before reference environment preparation"
        ),
    )

    result = trtmc_validate.run_binding(
        trtmc_validate.Binding("model-a", "suite-a"),
        arguments=arguments,
        task_models={
            "model-a": {
                "family": "albert",
                "task_strategy": "encoder_only_nlp",
                "execution_profiles": {},
            }
        },
        suites={
            "suite-a": {
                "id": "suite-a",
                "dataset": {"default_path": "/mnt/data/missing/data.jsonl"},
            }
        },
    )

    assert result["execution"] == {
        "status": "error",
        "exit_code": 1,
        "retryable": False,
    }
    assert result["failure_stage"] == "preflight"
    assert result["failure_domain"] == "data-artifact"
    assert result["failure_code"] == "dataset_missing"
    assert result["reproduce"]["dataset"]["sample_limit"] == 50
    assert result["reproduce"]["dataset"]["prepared_input_count"] == 0
    assert "missing/data.jsonl" in result["raw_result"]["error"]


def test_diffusion_report_flattens_nested_reference_metrics():
    comparison = trtmc_validate._comparison_details(
        {
            "status": "passed",
            "mode": "diffusion_image_clip_parity",
            "overall_pass_rate": 1.0,
            "passed_count": 5,
            "valid_count": 5,
            "skipped_count": 0,
            "metrics": {
                "trt_hf_image_clip_cosine": {
                    "mean": 0.91,
                    "min": 0.87,
                    "max": 0.95,
                    "count": 5,
                },
                "psnr": {
                    "mean": 12.5,
                    "min": 11.0,
                    "max": 14.0,
                    "count": 5,
                },
            },
        },
        {"status": "completed"},
    )

    assert comparison["primary_metric"] == {
        "name": "overall_pass_rate",
        "value": 1.0,
    }
    assert comparison["metrics"]["trt_hf_image_clip_cosine"] == 0.91
    assert comparison["metrics"]["psnr"] == 12.5
    assert "No metrics" not in trtmc_validate._render_metrics({"comparison": comparison})


def test_model_plugin_report_uses_sample_pass_rate_and_nested_metrics():
    comparison = trtmc_validate._comparison_details(
        {
            "status": "passed",
            "mode": "model_plugin_parity",
            "sample_pass_rate": 1.0,
            "passed_count": 4,
            "valid_count": 4,
            "metrics": {
                "token_agreement_rate": {
                    "mean": 0.99,
                    "min": 0.97,
                    "max": 1.0,
                    "count": 4,
                }
            },
        },
        {"status": "completed"},
    )

    assert comparison["primary_metric"] == {
        "name": "sample_pass_rate",
        "value": 1.0,
    }
    assert comparison["metrics"]["token_agreement_rate"] == 0.99


def test_mcq_report_exposes_reference_tie_equivalence_metrics():
    comparison = trtmc_validate._comparison_details(
        {
            "status": "passed",
            "mode": "mcq",
            "prediction_agreement_rate": 1.0,
            "accuracy_delta_bundle_minus_hf": -0.05,
            "tie_adjusted_accuracy_delta_bundle_minus_hf": 0.0,
            "raw_accuracy_drop_from_hf": 0.05,
            "accuracy_drop_from_hf": 0.0,
            "reference_tie_equivalent_count": 1,
        },
        {"status": "completed"},
    )

    assert comparison["status"] == "agreement"
    assert comparison["metrics"]["accuracy_delta_bundle_minus_hf"] == -0.05
    assert comparison["metrics"]["tie_adjusted_accuracy_delta_bundle_minus_hf"] == 0.0
    assert comparison["metrics"]["raw_accuracy_drop_from_hf"] == 0.05
    assert comparison["metrics"]["accuracy_drop_from_hf"] == 0.0
    assert comparison["metrics"]["reference_tie_equivalent_count"] == 1


def test_full_duplex_report_uses_metric_gate_pass_rate() -> None:
    comparison = trtmc_validate._comparison_details(
        {
            "status": "passed",
            "mode": "full_duplex_bench_behavior_parity",
            "metric_gate_pass_rate": 1.0,
            "metrics": {
                "icc_backchannel.jsd.abs_delta": {"mean": 0.006},
            },
        },
        {"status": "completed"},
    )

    assert comparison["status"] == "agreement"
    assert comparison["primary_metric"] == {
        "name": "metric_gate_pass_rate",
        "value": 1.0,
    }
    assert comparison["metrics"]["icc_backchannel.jsd.abs_delta"] == 0.006


def test_legacy_e2e_result_is_not_reported_as_reference_agreement(tmp_path):
    case_dir = tmp_path / "model-a" / "e2e"
    case_dir.mkdir(parents=True)
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "e2e",
                "executor": "e2e",
                "status": "passed",
                "returncode": 0,
                "raw_results": [{"status": "pass"}],
            }
        ),
        encoding="utf-8",
    )

    _, html_path, report = trtmc_validate.write_report(tmp_path)

    result = report["results"][0]
    assert report["validation_status"] == "incomplete"
    assert report["summary"]["agreements"] == 0
    assert report["summary"]["not_compared"] == 1
    assert result["execution"]["status"] == "not_run"
    assert result["comparison"]["status"] == "not_run"
    assert result["validation"]["status"] == "not_compared"
    assert result["not_compared_reason"] == trtmc_validate.LEGACY_E2E_REASON
    assert report["accounting"]["outcomes"] == {
        "green": 0,
        "yellow": 0,
        "red": 0,
        "white": 1,
    }
    assert "model-a" not in html_path.read_text(encoding="utf-8")


def test_not_compared_result_replaces_legacy_e2e_row_without_deleting_evidence(
    tmp_path,
):
    legacy_dir = tmp_path / "model-a" / "e2e"
    legacy_dir.mkdir(parents=True)
    legacy_comparison = legacy_dir / "comparison.json"
    legacy_comparison.write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "e2e",
                "executor": "e2e",
                "status": "passed",
                "raw_results": [{"status": "pass"}],
            }
        ),
        encoding="utf-8",
    )
    trtmc_validate._write_not_compared_case(
        trtmc_validate.Binding(
            "model-a",
            None,
            "Reference comparator is missing.",
        ),
        tmp_path,
    )

    _, _, report = trtmc_validate.write_report(tmp_path)

    assert legacy_comparison.is_file()
    assert report["summary"]["cases"] == 1
    assert report["summary"]["not_compared"] == 1
    assert report["results"][0]["not_compared_reason"] == "Reference comparator is missing."


def test_write_report_records_total_duration(tmp_path, monkeypatch):
    started_at = "2026-07-25T01:02:03+00:00"
    finished_at = datetime(2026, 7, 25, 4, 4, 6, 500000, tzinfo=timezone.utc)
    (tmp_path / "run.json").write_text(
        json.dumps({"started_at": started_at}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        trtmc_validate,
        "_utc_now",
        lambda: finished_at,
    )

    _, html_path, report = trtmc_validate.write_report(tmp_path)

    assert report["summary"]["duration_seconds"] == 10_923.5
    assert "10923.5" not in html_path.read_text(encoding="utf-8")


def test_write_report_preserves_finalized_duration(tmp_path, monkeypatch):
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "started_at": "2026-07-25T01:02:03+00:00",
                "finished_at": "2026-07-25T01:02:13+00:00",
                "duration_seconds": 10.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        trtmc_validate,
        "_utc_now",
        lambda: datetime(2026, 7, 25, 4, 4, 6, tzinfo=timezone.utc),
    )

    _, html_path, report = trtmc_validate.write_report(tmp_path)

    assert report["summary"]["duration_seconds"] == 10.0
    assert "10.0" not in html_path.read_text(encoding="utf-8")


def test_write_report_does_not_render_validation_wrapper(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "failed",
                "reproduce": {
                    "hf": ["python hf.py --prompt '<hello>'"],
                    "trtmc": [],
                    "validation": "python tools/validation/engine.py eval --model model-a",
                },
            }
        ),
        encoding="utf-8",
    )

    _, html_path, _ = trtmc_validate.write_report(tmp_path)

    document = html_path.read_text(encoding="utf-8")
    assert "python hf.py" not in document
    migrated = json.loads((case_dir / "comparison.json").read_text(encoding="utf-8"))
    assert "validation" not in migrated["reproduce"]
    assert set(migrated["reproduce"]) == {
        "command_count",
        "command_logs",
        "commands_shown",
        "dataset",
        "hf",
        "representative",
        "trtmc",
    }


def test_write_report_recovers_json_logged_runner_command(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    work_dir = case_dir / "validation"
    work_dir.mkdir(parents=True)
    (work_dir / "bundle_run.log").write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "command": ["trtmc", "solve", "model.bundle", "--field-input", "1,2"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "passed",
                "raw_result": {"work_dir": str(work_dir)},
                "reproduce": {"hf": ["python hf.py"], "trtmc": ["trtmc build"]},
            }
        ),
        encoding="utf-8",
    )

    _, html_path, report = trtmc_validate.write_report(tmp_path)

    assert report["results"][0]["reproduce"]["trtmc"] == [
        "trtmc build",
        "trtmc solve model.bundle --field-input 1,2",
    ]
    assert "trtmc solve model.bundle --field-input 1,2" in json.dumps(report)


def test_report_bounds_large_sample_commands_and_selects_disagreement(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    work_dir = case_dir / "validation" / "workload-a" / "model-a"
    work_dir.mkdir(parents=True)
    sample_count = 10_000
    (work_dir / "prompts.jsonl").write_text(
        "".join(
            json.dumps({"sample_id": f"sample-{index}", "prompt": f"prompt-{index}"}) + "\n"
            for index in range(sample_count)
        ),
        encoding="utf-8",
    )
    (work_dir / "bundle_run.log").write_text(
        "".join(
            f"$ trtmc run model.bundle --prompt prompt-{index}\n" for index in range(sample_count)
        ),
        encoding="utf-8",
    )
    (work_dir / "summary.json").write_text(
        json.dumps({"disagreements": [{"sample_id": "sample-9999"}]}),
        encoding="utf-8",
    )
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "failed",
                "raw_result": {
                    "status": "failed",
                    "work_dir": str(work_dir),
                },
                "reproduce": {
                    "dataset": {
                        "command": ("python tools/trtmc_validate.py model-a --limit 10000"),
                        "prepared_input_count": sample_count,
                    },
                    "hf": [],
                    "trtmc": [],
                },
            }
        ),
        encoding="utf-8",
    )

    _, html_path, report = trtmc_validate.write_report(tmp_path)

    reproduction = report["results"][0]["reproduce"]
    assert reproduction["command_count"]["trtmc"] == sample_count
    assert reproduction["commands_shown"]["trtmc"] == 1
    assert reproduction["trtmc"] == ["trtmc run model.bundle --prompt prompt-9999"]
    assert reproduction["representative"] == {
        "sample_id": "sample-9999",
        "reason": "first_disagreement",
    }
    assert reproduction["command_logs"]["trtmc"] == ["bundle_run.log"]
    assert "prompt-5000" not in json.dumps(report)
    document = html_path.read_text(encoding="utf-8")
    assert "prompt-9999" not in document
    assert "prompt-5000" not in document
    assert (case_dir / "comparison.json").stat().st_size < 20_000
    assert (tmp_path / "report.json").stat().st_size < 20_000


def test_report_adds_failed_sample_results_and_native_commands(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    work_dir = case_dir / "validation" / "workload-a" / "model-a"
    work_dir.mkdir(parents=True)
    prompt = {
        "sample_id": "sample-7",
        "eval_index": 7,
        "prompt": "Complete this sentence",
    }
    (work_dir / "prompts.jsonl").write_text(
        json.dumps(prompt) + "\n",
        encoding="utf-8",
    )
    (work_dir / "summary.json").write_text(
        json.dumps(
            {
                "disagreements": [
                    {
                        "sample_id": "sample-7",
                        "hf_prediction": "reference answer",
                        "bundle_prediction": "TRTMC answer",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (work_dir / "hf_predictions.json").write_text(
        json.dumps(
            {
                "responses": [
                    {
                        "sample_id": "sample-7",
                        "output_text": "reference answer",
                        "generated_token_ids": [1, 2],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (work_dir / "bundle_predictions.json").write_text(
        json.dumps(
            {
                "responses": [
                    {
                        "sample_id": "sample-7",
                        "output_text": "TRTMC answer",
                        "generated_token_ids": [1, 3],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (work_dir / "hf_native_repro.json").write_text(
        json.dumps(
            {
                "command": [
                    "/profiles/reference/bin/python",
                    "/workspace/trtmc/tools/reference/transformers_text.py",
                    "--prompts",
                    "{work_dir}/prompts.jsonl",
                    "--sample-id",
                    "{sample_id}",
                    "--predictions",
                    "{reference_predictions_json}",
                ]
            }
        ),
        encoding="utf-8",
    )
    (work_dir / "bundle_repro.json").write_text(
        json.dumps(
            {
                "command": [
                    "/workspace/build/trtmc_dataset_benchmark",
                    "model.bundle",
                    "{input_jsonl}",
                    "{trtmc_raw_jsonl}",
                    "--max-new-tokens",
                    "8",
                    "--seed",
                    "{sample_seed}",
                ],
                "base_seed": 42,
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "failed",
                "raw_result": {
                    "status": "failed",
                    "work_dir": str(work_dir),
                    "precision_contract": {
                        "trtmc_base_precision": "fp16",
                        "reference_precision": "fp16",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    _, html_path, report = trtmc_validate.write_report(tmp_path)

    metadata = report["results"][0]["disagreements"]
    assert metadata["count"] == 1
    artifact = case_dir / metadata["path"]
    records = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines()]
    assert records[0]["input"] == prompt
    assert records[0]["reference_result"]["output_text"] == "reference answer"
    assert records[0]["trtmc_result"]["output_text"] == "TRTMC answer"
    assert records[0]["reproduce"]["reference"].startswith(
        "/profiles/reference/bin/python /workspace/trtmc/tools/reference/transformers_text.py"
    )
    assert records[0]["reproduce"]["trtmc"].startswith(
        "/workspace/build/trtmc_dataset_benchmark model.bundle"
    )
    assert records[0]["reproduce"]["trtmc"].endswith("--seed 49")
    assert (case_dir / records[0]["artifacts"]["trtmc_input"]).read_text(
        encoding="utf-8"
    ) == json.dumps(prompt, ensure_ascii=False) + "\n"
    public_differences = report["results"][0]["sample_differences"]
    assert public_differences["count"] == 1
    assert public_differences["classification"] == "failed_samples"
    assert public_differences["preview"][0]["sample_id"] == "sample-7"
    assert public_differences["preview"][0]["reason"] == "comparison_threshold"
    assert public_differences["preview"][0]["reference_result"]["output_text"] == (
        "reference answer"
    )
    assert public_differences["preview"][0]["trtmc_result"]["output_text"] == ("TRTMC answer")
    assert public_differences["preview"][0]["reproduce"]["reference"].startswith(
        "/profiles/reference/bin/python"
    )
    assert (tmp_path / public_differences["href"]).is_file()
    rendered = html_path.read_text(encoding="utf-8")
    assert "reference answer" not in rendered
    frontend = (tmp_path / "assets" / "qualification-report.js").read_text(encoding="utf-8")
    assert "sampleDifferences" in frontend
    assert "results and vanilla commands" in frontend
    assert "sameMetricValue" in frontend
    assert report["results"][0]["result"] == "red"
    for wrapper in (
        "validation/engine.py",
        "trtmc_compare.py",
        "trtmc_reference.py",
        "trtmc_validate.py",
    ):
        assert wrapper not in json.dumps(records)


def test_cached_reference_command_is_relocated_to_current_work_dir(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "current run"
    work_dir.mkdir()
    (work_dir / "prompts.jsonl").write_text(
        json.dumps({"sample_id": "sample-1"}) + "\n",
        encoding="utf-8",
    )
    old_work_dir = Path("/runs/results/old-run/model/workload")
    (work_dir / "hf_native_run.log").write_text(
        "$ python reference.py "
        f"--prompts {shlex.quote(str(old_work_dir / 'prompts.jsonl'))} "
        f"--answers {shlex.quote(str(old_work_dir / 'answers.json'))} "
        f"--manifest {shlex.quote(str(old_work_dir / 'manifest.json'))} "
        f"--output {shlex.quote(str(old_work_dir / 'hf_predictions.json'))}\n",
        encoding="utf-8",
    )

    command = trtmc_validate._commands_from_logs(work_dir)["hf"][0]

    assert str(old_work_dir) not in command
    assert shlex.quote(str(work_dir / "prompts.jsonl")) in command
    assert shlex.quote(str(work_dir / "hf_predictions.json")) in command


def test_commands_from_logs_use_native_trtmc_jsonl(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "prompts.jsonl").write_text(
        "\n".join(json.dumps({"sample_id": sample_id}) for sample_id in ("sample-1", "sample-2"))
        + "\n",
        encoding="utf-8",
    )
    (work_dir / "bundle_run.log").write_text(
        "$ python validation/engine.py run-bundle\n",
        encoding="utf-8",
    )
    (work_dir / "full_duplex_bench_score.log").write_text(
        "$ python tools/full_duplex_bench_score.py --trtmc-predictions out.json\n",
        encoding="utf-8",
    )
    commands = (
        {
            "sample_id": "sample-1",
            "command": ["trtmc", "segment-prompted", "model.bundle", "--prompt", "cat"],
        },
        {
            "sample_id": "sample-2",
            "command": ["trtmc", "segment-prompted", "model.bundle", "--prompt", "dog"],
        },
    )
    (work_dir / "bundle_native_commands.jsonl").write_text(
        "".join(json.dumps(command) + "\n" for command in commands),
        encoding="utf-8",
    )

    reproduction = trtmc_validate._commands_from_logs(work_dir)

    assert reproduction["trtmc"] == ["trtmc segment-prompted model.bundle --prompt cat"]
    assert reproduction["command_count"]["trtmc"] == 2
    assert reproduction["command_logs"]["trtmc"] == ["bundle_native_commands.jsonl"]
    assert "full_duplex_bench_score.py" not in json.dumps(reproduction)


def test_commands_from_logs_prefer_native_reference_jsonl(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "prompts.jsonl").write_text(
        json.dumps({"sample_id": "sample-1"}) + "\n",
        encoding="utf-8",
    )
    (work_dir / "hf_run.log").write_text(
        "$ python trtmc_reference.py run\n",
        encoding="utf-8",
    )
    (work_dir / "hf_native_run.log").write_text(
        "$ python plugin_reference.py\n",
        encoding="utf-8",
    )
    (work_dir / "hf_native_commands.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "command": ["python", "model_reference.py", "--prompt", "cat"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    reproduction = trtmc_validate._commands_from_logs(work_dir)

    assert reproduction["hf"] == ["python model_reference.py --prompt cat"]
    assert reproduction["command_count"]["hf"] == 1
    assert reproduction["command_logs"]["hf"] == ["hf_native_commands.jsonl"]


def test_failed_sample_uses_recorded_trtmc_command_and_copies_media(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    work_dir = case_dir / "validation" / "workload-a" / "model-a"
    frames = work_dir / "reference_frames"
    frames.mkdir(parents=True)
    input_image = work_dir / "input.png"
    reference_image = frames / "000.png"
    reference_visualization = work_dir / "reference_visualization.png"
    trtmc_visualization = work_dir / "trtmc_visualization.png"
    trtmc_audio = work_dir / "output.wav"
    input_image.write_bytes(b"input-image")
    reference_image.write_bytes(b"reference-image")
    reference_visualization.write_bytes(b"reference-visualization")
    trtmc_visualization.write_bytes(b"trtmc-visualization")
    trtmc_audio.write_bytes(b"RIFFfake-wave")
    prompt = {
        "sample_id": "sample-9",
        "prompt": "Describe",
        "images": [str(input_image)],
    }
    (work_dir / "prompts.jsonl").write_text(
        json.dumps(prompt) + "\n",
        encoding="utf-8",
    )
    (work_dir / "answers.json").write_text(
        json.dumps({"requests": [{"sample_id": "sample-9", "answer": "A"}]}),
        encoding="utf-8",
    )
    (work_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "backend_mean_iou": 0.90,
                "gates": {"min_backend_mean_iou": 0.95},
                "cases": [
                    {
                        "sample_id": "sample-9",
                        "backend_mean_iou": 0.90,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (work_dir / "hf_predictions.json").write_text(
        json.dumps(
            {
                "responses": [
                    {
                        "sample_id": "sample-9",
                        "frames_dir": str(frames),
                        "visualization_path": str(reference_visualization),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (work_dir / "bundle_predictions.json").write_text(
        json.dumps(
            {
                "responses": [
                    {
                        "sample_id": "sample-9",
                        "wav_path": str(trtmc_audio),
                        "visualization_path": str(trtmc_visualization),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    native_command = [
        "/workspace/build/trtmc",
        "run",
        "/runs/engines/model.bundle",
        "--prompt",
        "Describe",
    ]
    (work_dir / "bundle_native_commands.jsonl").write_text(
        json.dumps({"sample_id": "sample-9", "command": native_command}) + "\n",
        encoding="utf-8",
    )
    reference_command = [
        "/profiles/reference/bin/python",
        "/workspace/model/reference.py",
        "--input",
        str(input_image),
    ]
    (work_dir / "hf_native_commands.jsonl").write_text(
        json.dumps({"sample_id": "sample-9", "command": reference_command}) + "\n",
        encoding="utf-8",
    )

    metadata = trtmc_disagreements.build_disagreement_artifact(
        work_dir=work_dir,
        case_dir=case_dir,
    )

    record = json.loads((case_dir / metadata["path"]).read_text(encoding="utf-8"))
    assert record["reproduce"]["trtmc"] == (
        "/workspace/build/trtmc run /runs/engines/model.bundle --prompt Describe"
    )
    assert record["reproduce"]["reference"].startswith(
        "/profiles/reference/bin/python /workspace/model/reference.py"
    )
    media = record["artifacts"]["media"]
    assert {item["kind"] for item in media} == {"image", "audio"}
    assert len(media) == 5
    assert {item["label"] for item in media} >= {
        "Reference visualization_path",
        "TRTMC visualization_path",
    }
    assert all((case_dir / item["path"]).is_file() for item in media)
    rendered = trtmc_validate._render_disagreement_record(
        record,
        asset_base=Path("model-a/workload-a"),
    )
    assert "<img " in rendered
    assert "<audio " in rendered
    assert "validation/engine.py" not in rendered


def test_failed_encoder_pair_expands_to_both_reproducible_samples():
    rows = [
        {
            "pair_id": "sts-4",
            "passed": False,
            "cosine_abs_delta": 0.2,
        }
    ]
    prompts = {
        "sts-4-a": {
            "sample_id": "sts-4-a",
            "pair_id": "sts-4",
            "pair_side": "sentence1",
        },
        "sts-4-b": {
            "sample_id": "sts-4-b",
            "pair_id": "sts-4",
            "pair_side": "sentence2",
        },
    }

    expanded = trtmc_disagreements._expand_pair_disagreements(rows, prompts)

    assert [row["sample_id"] for row in expanded] == [
        "sts-4-a",
        "sts-4-b",
    ]


def test_summary_gate_failure_selects_worst_sample_for_reproduction():
    rows = trtmc_disagreements._summary_disagreements(
        {
            "status": "failed",
            "backend_mean_iou": 0.92,
            "gates": {"min_backend_mean_iou": 0.95},
            "cases": [
                {"sample_id": "sample-good", "backend_mean_iou": 0.97},
                {"sample_id": "sample-worst", "backend_mean_iou": 0.90},
            ],
        }
    )

    assert rows == [
        {
            "sample_id": "sample-worst",
            "backend_mean_iou": 0.90,
            "status": "failed",
            "reason": "summary_gate_failure",
            "failed_gates": [
                {
                    "gate": "min_backend_mean_iou",
                    "metric": "backend_mean_iou",
                    "actual": 0.92,
                    "threshold": 0.95,
                }
            ],
        }
    ]


def test_report_bounds_inline_failed_samples_but_keeps_full_artifact(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    work_dir = case_dir / "validation" / "workload-a" / "model-a"
    work_dir.mkdir(parents=True)
    sample_count = 25
    prompts = [
        {"sample_id": f"sample-{index}", "prompt": f"prompt-{index}"}
        for index in range(sample_count)
    ]
    (work_dir / "prompts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in prompts),
        encoding="utf-8",
    )
    (work_dir / "summary.json").write_text(
        json.dumps(
            {
                "disagreements": [
                    {"sample_id": row["sample_id"], "reason": "token_mismatch"} for row in prompts
                ]
            }
        ),
        encoding="utf-8",
    )
    for name, prefix in (
        ("hf_predictions.json", "reference"),
        ("bundle_predictions.json", "trtmc"),
    ):
        (work_dir / name).write_text(
            json.dumps(
                {
                    "responses": [
                        {
                            "sample_id": row["sample_id"],
                            "output_text": f"{prefix}-{index}",
                        }
                        for index, row in enumerate(prompts)
                    ]
                }
            ),
            encoding="utf-8",
        )
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "failed",
                "raw_result": {"status": "failed", "work_dir": str(work_dir)},
            }
        ),
        encoding="utf-8",
    )

    _, html_path, report = trtmc_validate.write_report(tmp_path)

    metadata = report["results"][0]["disagreements"]
    artifact = case_dir / metadata["path"]
    assert metadata["count"] == sample_count
    assert len(artifact.read_text(encoding="utf-8").splitlines()) == sample_count
    rendered = html_path.read_text(encoding="utf-8")
    assert "sample-19" not in rendered
    assert "sample-20" not in rendered
    assert (case_dir / "comparison.json").stat().st_size < 20_000
    assert (tmp_path / "report.json").stat().st_size < 20_000


def test_report_does_not_treat_shared_task_failure_as_disagreement(tmp_path):
    case_dir = tmp_path / "model-a" / "workload-a"
    work_dir = case_dir / "validation" / "workload-a" / "model-a"
    work_dir.mkdir(parents=True)
    (work_dir / "prompts.jsonl").write_text(
        json.dumps({"sample_id": "sample-0", "prompt": "hello"}) + "\n",
        encoding="utf-8",
    )
    (work_dir / "bundle_run.log").write_text(
        "$ trtmc run model.bundle --prompt hello\n",
        encoding="utf-8",
    )
    (work_dir / "summary.json").write_text(
        json.dumps(
            {
                "disagreements": [],
                "hf": {"samples": [{"sample_id": "sample-0", "passed": False}]},
                "bundle": {"samples": [{"sample_id": "sample-0", "passed": False}]},
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "comparison.json").write_text(
        json.dumps(
            {
                "model": "model-a",
                "workload": "workload-a",
                "status": "passed",
                "raw_result": {"work_dir": str(work_dir)},
            }
        ),
        encoding="utf-8",
    )

    _, _, report = trtmc_validate.write_report(tmp_path)

    assert report["results"][0]["reproduce"]["representative"] == {
        "sample_id": "sample-0",
        "reason": "first_input",
    }


def test_run_metadata_records_source_and_exact_command(monkeypatch, tmp_path):
    monkeypatch.setenv("TRTMC_VALIDATION_SOURCE_REVISION", "abc123")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        trtmc_validate.sys,
        "argv",
        ["tools/trtmc_validate.py", "model-a"],
    )
    monkeypatch.setattr(
        trtmc_validate,
        "_runtime_gpu_devices",
        lambda visible: [
            {
                "cuda_logical_index": 0,
                "nvidia_smi_index": 0,
                "uuid": "GPU-7980c63d",
                "name": "NVIDIA GB300",
                "pci_bus_id": "00000009:06:00.0",
            }
        ],
    )

    path = trtmc_validate.write_run_metadata(tmp_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))

    assert metadata["source_revision"] == "abc123"
    assert metadata["cuda_visible_devices"] == "0"
    assert metadata["gpu_devices"] == [
        {
            "cuda_logical_index": 0,
            "nvidia_smi_index": 0,
            "uuid": "GPU-7980c63d",
            "name": "NVIDIA GB300",
            "pci_bus_id": "00000009:06:00.0",
        }
    ]
    assert metadata["command"] == "tools/trtmc_validate.py model-a"
    assert metadata["finished_at"] is None
    assert metadata["duration_seconds"] is None


def test_run_metadata_prefers_cli_gpu_selector_over_parent_environment(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    received = []

    def resolve(visible):
        received.append(visible)
        return [
            {
                "cuda_logical_index": 0,
                "nvidia_smi_index": 1,
                "uuid": "GPU-gb300",
                "name": "NVIDIA GB300",
                "pci_bus_id": "00000009:06:00.0",
            }
        ]

    monkeypatch.setattr(trtmc_validate, "_runtime_gpu_devices", resolve)

    path = trtmc_validate.write_run_metadata(
        tmp_path,
        cuda_visible_devices="1",
    )
    metadata = json.loads(path.read_text(encoding="utf-8"))

    assert received == ["1"]
    assert metadata["cuda_visible_devices"] == "1"
    assert metadata["gpu_devices"][0]["uuid"] == "GPU-gb300"


def test_cuda_visible_device_ordinal_resolves_to_stable_gpu_identity():
    inventory = [
        {
            "nvidia_smi_index": 0,
            "uuid": "GPU-rtx",
            "name": "NVIDIA RTX PRO 6000",
            "pci_bus_id": "00000004:01:00.0",
        },
        {
            "nvidia_smi_index": 1,
            "uuid": "GPU-gb300",
            "name": "NVIDIA GB300",
            "pci_bus_id": "00000009:06:00.0",
        },
    ]

    assert trtmc_validate._resolve_cuda_devices("1", inventory) == [
        {
            "cuda_logical_index": 0,
            "nvidia_smi_index": 1,
            "uuid": "GPU-gb300",
            "name": "NVIDIA GB300",
            "pci_bus_id": "00000009:06:00.0",
        }
    ]


def test_nvidia_smi_inventory_records_stable_gpu_identity(monkeypatch):
    command = []

    def fake_run(args, **kwargs):
        command.extend(args)
        assert kwargs == {
            "check": False,
            "capture_output": True,
            "text": True,
            "timeout": 10,
        }
        return trtmc_validate.subprocess.CompletedProcess(
            args,
            0,
            stdout=(
                "0, GPU-ae3e92b4, NVIDIA RTX PRO 6000, 00000004:01:00.0\n"
                "1, GPU-7980c63d, NVIDIA GB300, 00000009:06:00.0\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(trtmc_validate.subprocess, "run", fake_run)

    assert trtmc_validate._query_nvidia_smi_gpus() == [
        {
            "nvidia_smi_index": 0,
            "uuid": "GPU-ae3e92b4",
            "name": "NVIDIA RTX PRO 6000",
            "pci_bus_id": "00000004:01:00.0",
        },
        {
            "nvidia_smi_index": 1,
            "uuid": "GPU-7980c63d",
            "name": "NVIDIA GB300",
            "pci_bus_id": "00000009:06:00.0",
        },
    ]
    assert command == [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,pci.bus_id",
        "--format=csv,noheader,nounits",
    ]


def test_runtime_gpu_identity_falls_back_to_cuda_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        trtmc_validate,
        "_query_nvidia_smi_gpus",
        lambda: (_ for _ in ()).throw(trtmc_validate.ValidationError("nvidia-smi was not found")),
    )
    monkeypatch.setattr(
        trtmc_validate,
        "_query_cuda_runtime_gpus",
        lambda: [
            {
                "cuda_runtime_index": 0,
                "uuid": "GPU-7d4db97a-e132-503a-816d-a252f371ec4c",
                "name": "Thor",
                "pci_bus_id": "0000:00:00.0",
            }
        ],
    )

    path = trtmc_validate.write_run_metadata(tmp_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))

    assert metadata["gpu_identity_source"] == "cuda-runtime"
    assert metadata["gpu_devices"] == [
        {
            "cuda_logical_index": 0,
            "cuda_runtime_index": 0,
            "uuid": "GPU-7d4db97a-e132-503a-816d-a252f371ec4c",
            "name": "Thor",
            "pci_bus_id": "0000:00:00.0",
        }
    ]


def test_run_metadata_rejects_missing_gpu_identity(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        trtmc_validate,
        "_runtime_gpu_devices",
        lambda _visible: [],
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="GPU identity",
    ):
        trtmc_validate.write_run_metadata(tmp_path)

    assert not (tmp_path / "run.json").exists()


def test_report_provenance_disambiguates_process_local_cuda_ordinal():
    provenance = trtmc_validate._report_provenance(
        {
            "source_revision": "abc123",
            "hostname": "container-id",
            "cuda_visible_devices": "0",
            "gpu_devices": [
                {
                    "cuda_logical_index": 0,
                    "nvidia_smi_index": 0,
                    "uuid": "GPU-7980c63d",
                    "name": "NVIDIA GB300",
                    "pci_bus_id": "00000009:06:00.0",
                }
            ],
        }
    )

    assert "GPU logical 0=NVIDIA GB300" in provenance
    assert "uuid=GPU-7980c63d" in provenance
    assert "pci=00000009:06:00.0" in provenance
    assert "runtime-nvidia-smi-index=0" in provenance
    assert "CUDA_VISIBLE_DEVICES(process-local)=0" in provenance


def test_report_provenance_marks_legacy_gpu_identity_as_missing():
    provenance = trtmc_validate._report_provenance(
        {
            "hostname": "legacy-container-id",
            "cuda_visible_devices": "0",
        }
    )

    assert "GPU identity=not recorded" in provenance
    assert "CUDA_VISIBLE_DEVICES(process-local)=0" in provenance


def test_run_metadata_preserves_campaign_start_when_results_exist(
    monkeypatch,
    tmp_path,
):
    original_start = "2026-07-25T01:02:03+00:00"
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "started_at": original_start,
                "finished_at": "2026-07-25T01:12:03+00:00",
                "duration_seconds": 600.0,
            }
        ),
        encoding="utf-8",
    )
    case_dir = tmp_path / "model-a" / "workload-a"
    case_dir.mkdir(parents=True)
    (case_dir / "comparison.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        trtmc_validate,
        "_utc_now",
        lambda: datetime(2026, 7, 25, 2, 0, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        trtmc_validate,
        "_runtime_gpu_devices",
        lambda _visible: [
            {
                "cuda_logical_index": 0,
                "nvidia_smi_index": 0,
                "uuid": "GPU-test",
                "name": "NVIDIA test GPU",
                "pci_bus_id": "00000000:01:00.0",
            }
        ],
    )

    path = trtmc_validate.write_run_metadata(tmp_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))

    assert metadata["started_at"] == original_start
    assert metadata["finished_at"] is None
    assert metadata["duration_seconds"] is None


def _build_identity_arguments(
    tmp_path: Path,
    *,
    include_worker: bool = True,
) -> argparse.Namespace:
    build = tmp_path / "build"
    models = build / "models"
    models.mkdir(parents=True)
    for name in (
        "trtmc",
        "trtmc_dataset_benchmark",
        "libtrtmc_backend_trt.so",
    ):
        path = build / name
        path.write_bytes(name.encode())
        path.chmod(0o755)
    if include_worker:
        worker = build / "trtmc_benchmark_worker"
        worker.write_bytes(b"worker")
        worker.chmod(0o755)
    return trtmc_validate.build_parser().parse_args(
        [
            "model-a",
            "--trtmc-binary",
            str(build / "trtmc"),
            "--benchmark-binary",
            str(build / "trtmc_dataset_benchmark"),
            "--backend-dir",
            str(build),
            "--model-plugin-dir",
            str(models),
        ]
    )


def test_build_identity_preflight_accepts_exact_native_build(
    monkeypatch,
    tmp_path,
):
    arguments = _build_identity_arguments(tmp_path)
    monkeypatch.setattr(trtmc_validate, "_source_revision", lambda: "tested-revision")
    monkeypatch.setattr(
        trtmc_validate,
        "worker_metadata",
        lambda _worker: {
            "schema_version": "trtmc.benchmark-worker-metadata/v1",
            "build": {
                "configuration": "Release",
                "source_revision": "tested-revision",
            },
        },
    )

    identity = trtmc_validate._validate_build_identity(arguments)

    assert identity["source_revision"] == "tested-revision"
    assert identity["embedded_source_revision"] == "tested-revision"
    assert identity["build_configuration"] == "Release"
    assert set(identity["artifacts"]) == {
        "trtmc binary",
        "dataset benchmark",
        "benchmark worker",
        "TensorRT backend",
    }
    assert all(len(artifact["sha256"]) == 64 for artifact in identity["artifacts"].values())


def test_build_identity_preflight_rejects_stale_native_build(
    monkeypatch,
    tmp_path,
):
    arguments = _build_identity_arguments(tmp_path)
    monkeypatch.setattr(trtmc_validate, "_source_revision", lambda: "current-revision")
    monkeypatch.setattr(
        trtmc_validate,
        "worker_metadata",
        lambda _worker: {
            "schema_version": "trtmc.benchmark-worker-metadata/v1",
            "build": {
                "configuration": "Release",
                "source_revision": "stale-revision",
            },
        },
    )

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="embedded stale-revision, expected current-revision",
    ):
        trtmc_validate._validate_build_identity(arguments)


def test_build_identity_preflight_rejects_missing_worker(monkeypatch, tmp_path):
    arguments = _build_identity_arguments(tmp_path, include_worker=False)
    monkeypatch.setattr(trtmc_validate, "_source_revision", lambda: "tested-revision")

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="benchmark worker is missing",
    ):
        trtmc_validate._validate_build_identity(arguments)


def test_build_identity_preflight_rejects_non_executable_worker(monkeypatch, tmp_path):
    arguments = _build_identity_arguments(tmp_path)
    worker = arguments.benchmark_binary.parent / "trtmc_benchmark_worker"
    worker.chmod(0o644)
    monkeypatch.setattr(trtmc_validate, "_source_revision", lambda: "tested-revision")

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="benchmark worker is not executable",
    ):
        trtmc_validate._validate_build_identity(arguments)


def test_build_identity_preflight_rejects_mixed_plugin_directory(
    monkeypatch,
    tmp_path,
):
    arguments = _build_identity_arguments(tmp_path)
    mixed_plugins = tmp_path / "stale-build" / "models"
    mixed_plugins.mkdir(parents=True)
    arguments.model_plugin_dir = mixed_plugins
    monkeypatch.setattr(trtmc_validate, "_source_revision", lambda: "tested-revision")

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="model plugin directory resolves .* outside",
    ):
        trtmc_validate._validate_build_identity(arguments)


def test_build_identity_preflight_rejects_missing_source_revision(
    monkeypatch,
    tmp_path,
):
    arguments = _build_identity_arguments(tmp_path)
    monkeypatch.setattr(trtmc_validate, "_source_revision", lambda: "")

    with pytest.raises(
        trtmc_validate.ValidationError,
        match="cannot determine the validation source revision",
    ):
        trtmc_validate._validate_build_identity(arguments)


def test_finalize_run_metadata_records_completion(monkeypatch, tmp_path):
    started_at = datetime(2026, 7, 25, 1, 2, 3, tzinfo=timezone.utc)
    finished_at = datetime(2026, 7, 25, 4, 4, 6, 500000, tzinfo=timezone.utc)
    (tmp_path / "run.json").write_text(
        json.dumps({"started_at": started_at.isoformat()}),
        encoding="utf-8",
    )
    monkeypatch.setattr(trtmc_validate, "_utc_now", lambda: finished_at)

    path = trtmc_validate.finalize_run_metadata(tmp_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))

    assert metadata["finished_at"] == finished_at.isoformat()
    assert metadata["duration_seconds"] == 10_923.5


def test_comparison_command_uses_validation_entrypoint(tmp_path):
    arguments = argparse.Namespace(
        models_dir=tmp_path / "models",
        engine_dir=tmp_path / "engines",
        reference_cache_dir=tmp_path / "references",
        trtmc_binary=tmp_path / "trtmc",
        benchmark_binary=tmp_path / "trtmc_dataset_benchmark",
        limit=2,
        force_hf=True,
        force_build=False,
        no_build=True,
        local_files_only=True,
        backend_dir=None,
        model_plugin_dir=None,
        cuda_visible_devices="1",
        hf_device="cpu",
        hf_device_map="balanced",
    )

    command = trtmc_validate._comparison_command(
        trtmc_validate.Binding(
            "model-a",
            "workload-a",
            reference_cache_identity="org/model/reference-contract-v1",
        ),
        case_dir=tmp_path / "case",
        dataset=tmp_path / "dataset.json",
        arguments=arguments,
        reference_python="/profiles/python",
    )

    assert command[:2] == [
        "/profiles/python",
        str(trtmc_validate.REPO_ROOT / "tools" / "trtmc_compare.py"),
    ]
    assert "validation/engine.py" not in " ".join(command)
    assert command[command.index("--work-root") + 1] == str(tmp_path / "case" / "validation")
    assert command[command.index("--model") + 1] == "model-a"
    assert command[command.index("--suite") + 1] == "workload-a"
    assert command[command.index("--models-dir") + 1] == str(tmp_path / "models")
    assert command[command.index("--hf-python") + 1] == "/profiles/python"
    assert command[command.index("--reference-cache-dir") + 1] == str(tmp_path / "references")
    assert (
        command[command.index("--reference-cache-identity") + 1]
        == "org/model/reference-contract-v1"
    )
    assert "--replace-bundle-on-build" in command
    assert "--force-hf" in command
    assert "--require-prebuilt-bundles" in command
    assert "--local-files-only" in command
    assert command[command.index("--hf-device") + 1] == "cpu"
    assert command[command.index("--hf-device-map") + 1] == "balanced"


def test_comparison_command_passes_elf_reference_checkout(tmp_path):
    arguments = trtmc_validate.build_parser().parse_args([])
    arguments.engine_dir = tmp_path / "engines"
    arguments.reference_cache_dir = tmp_path / "references"
    arguments.trtmc_binary = tmp_path / "trtmc"
    arguments.benchmark_binary = tmp_path / "trtmc_dataset_benchmark"
    arguments.limit = 1
    reference_sources = trtmc_validate.ReferenceSourceSelection(
        environment={},
        elf_reference_repo=tmp_path / "sources" / "elf",
    )

    command = trtmc_validate._comparison_command(
        trtmc_validate.Binding("elf-b", "elf-workload"),
        case_dir=tmp_path / "case",
        dataset=tmp_path / "dataset.json",
        arguments=arguments,
        reference_python="/profiles/python",
        reference_sources=reference_sources,
    )

    assert command[command.index("--elf-reference-repo") + 1] == str(
        reference_sources.elf_reference_repo
    )


def test_run_binding_wires_reference_source_command_and_environment(
    tmp_path,
    monkeypatch,
):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "elf-b",
            "elf-workload",
            "--output",
            str(tmp_path / "results"),
            "--dataset",
            str(tmp_path / "dataset.json"),
            "--reference-cache-dir",
            str(tmp_path / "references"),
        ]
    )
    selection = trtmc_validate.ReferenceSourceSelection(
        environment={
            "TRTMC_STORAGE_ROOT": str(tmp_path / "references"),
            "EXTERNAL_REFERENCE_SENTINEL": "present",
        },
        elf_reference_repo=tmp_path / "references" / "elf",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        trtmc_validate,
        "ensure_environments",
        lambda _profiles, _base: trtmc_validate.EnvironmentSelection(
            base_python="/profiles/python",
            names_and_paths=(),
            overrides={},
        ),
    )
    monkeypatch.setattr(
        trtmc_validate,
        "ensure_reference_sources",
        lambda _family, _cache, _contract=None, **_kwargs: selection,
    )

    def run(command, _log_path, environment):
        captured["command"] = command
        captured["environment"] = environment
        return 0

    monkeypatch.setattr(trtmc_validate, "_run_subprocess", run)
    monkeypatch.setattr(
        trtmc_validate,
        "_comparison_result",
        lambda binding, **_kwargs: {
            "model": binding.model,
            "workload": binding.workload,
        },
    )

    trtmc_validate.run_binding(
        trtmc_validate.Binding("elf-b", "elf-workload"),
        arguments=arguments,
        task_models={
            "elf-b": {
                "family": "elf_flow",
                "runtime_strategy": "elf_flow",
                "reference_backend": "torch_reference",
                "execution_profiles": {},
            }
        },
        suites={"elf-workload": {}},
    )

    command = captured["command"]
    assert command[command.index("--elf-reference-repo") + 1] == str(selection.elf_reference_repo)
    assert captured["environment"]["EXTERNAL_REFERENCE_SENTINEL"] == "present"


def _run_binding_with_comparison_results(
    *,
    tmp_path,
    monkeypatch,
    raw_results,
    extra_args=(),
):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "model-a",
            "workload-a",
            "--output",
            str(tmp_path / "results"),
            "--dataset",
            str(tmp_path / "dataset.json"),
            "--engine-dir",
            str(tmp_path / "engines"),
            "--reference-cache-dir",
            str(tmp_path / "references"),
            *extra_args,
        ]
    )
    commands = []
    monkeypatch.setattr(
        trtmc_validate,
        "ensure_environments",
        lambda _profiles, _base: trtmc_validate.EnvironmentSelection(
            base_python="/profiles/python",
            names_and_paths=(),
            overrides={},
        ),
    )
    monkeypatch.setattr(
        trtmc_validate,
        "ensure_reference_sources",
        lambda _family, _cache, _contract=None, **_kwargs: trtmc_validate.ReferenceSourceSelection(
            environment={},
        ),
    )

    def run(command, log_path, _environment):
        commands.append(command)
        log_path.write_text(f"run {len(commands)}\n", encoding="utf-8")
        summary = log_path.parent / "validation" / "workload-a" / "eval_summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(
            json.dumps({"run": len(commands)}),
            encoding="utf-8",
        )
        return 0

    comparisons = iter(raw_results)

    def comparison_result(binding, **_kwargs):
        raw_result = next(comparisons)
        return trtmc_validate._normalize_result(
            {
                "model": binding.model,
                "workload": binding.workload,
                "status": raw_result["status"],
                "returncode": 0,
                "raw_result": raw_result,
            }
        )

    monkeypatch.setattr(trtmc_validate, "_run_subprocess", run)
    monkeypatch.setattr(trtmc_validate, "_comparison_result", comparison_result)
    result = trtmc_validate.run_binding(
        trtmc_validate.Binding("model-a", "workload-a"),
        arguments=arguments,
        task_models={
            "model-a": {
                "family": "family-a",
                "runtime_strategy": "text_generation",
                "reference_backend": "hf_transformers",
                "execution_profiles": {},
            }
        },
        suites={"workload-a": {}},
    )
    return result, commands, arguments


def _run_multiple_bindings_with_comparison_results(
    *,
    tmp_path,
    monkeypatch,
    models,
    raw_results,
    extra_args=(),
):
    arguments = trtmc_validate.build_parser().parse_args(
        [
            "--all",
            "--model-worker",
            "--output",
            str(tmp_path / "results"),
            "--dataset",
            str(tmp_path / "dataset.json"),
            "--engine-dir",
            str(tmp_path / "engines"),
            "--reference-cache-dir",
            str(tmp_path / "references"),
            *extra_args,
        ]
    )
    commands = []
    monkeypatch.setattr(
        trtmc_validate,
        "ensure_environments",
        lambda _profiles, _base: trtmc_validate.EnvironmentSelection(
            base_python="/profiles/python",
            names_and_paths=(),
            overrides={},
        ),
    )
    monkeypatch.setattr(
        trtmc_validate,
        "ensure_reference_sources",
        lambda _family, _cache, _contract=None, **_kwargs: trtmc_validate.ReferenceSourceSelection(
            environment={},
        ),
    )

    def run(command, log_path, _environment):
        commands.append(command)
        log_path.write_text(f"run {len(commands)}\n", encoding="utf-8")
        workload = log_path.parent.name
        summary = log_path.parent / "validation" / workload / "eval_summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(
            json.dumps({"run": len(commands)}),
            encoding="utf-8",
        )
        return 0

    comparisons = iter(raw_results)

    def comparison_result(binding, **_kwargs):
        raw_result = next(comparisons)
        return trtmc_validate._normalize_result(
            {
                "model": binding.model,
                "workload": binding.workload,
                "status": raw_result["status"],
                "returncode": 0,
                "raw_result": raw_result,
            }
        )

    monkeypatch.setattr(trtmc_validate, "_run_subprocess", run)
    monkeypatch.setattr(trtmc_validate, "_comparison_result", comparison_result)
    bindings = [trtmc_validate.Binding(model, "workload-a") for model in models]
    task_models = {
        model: {
            "family": "family-a",
            "runtime_strategy": "text_generation",
            "reference_backend": "hf_transformers",
            "execution_profiles": {},
        }
        for model in models
    }
    returncode = trtmc_validate._run_bindings(
        bindings,
        arguments=arguments,
        catalog={
            "sample_limits": {"workload-a": 1},
            "models": {
                model: {
                    "default": "workload-a",
                    "workloads": ["workload-a"],
                }
                for model in models
            },
        },
        task_models=task_models,
        suites={"workload-a": {}},
    )
    results = {
        model: json.loads(
            (arguments.output / model / "workload-a" / "comparison.json").read_text(
                encoding="utf-8"
            )
        )
        for model in models
    }
    return results, commands, returncode, arguments


def test_reused_bundle_accuracy_failure_rebuilds_once_and_recovers(
    tmp_path,
    monkeypatch,
):
    result, commands, arguments = _run_binding_with_comparison_results(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        raw_results=[
            {
                "status": "failed",
                "bundle_built": False,
                "prediction_agreement_rate": 0.25,
                "gate_failures": [
                    {
                        "gate": "min_prediction_agreement_rate",
                        "actual": 0.25,
                        "required": 1.0,
                    }
                ],
                "error_type": "BenchmarkGateError",
                "error": ("min_prediction_agreement_rate actual=0.25 required=1.0"),
            },
            {
                "status": "passed",
                "bundle_built": True,
                "prediction_agreement_rate": 1.0,
            },
        ],
    )

    assert len(commands) == 2
    assert "--force-build" not in commands[0]
    assert commands[1][-1] == "--force-build"
    assert result["validation"]["status"] == "passed"
    assert result["bundle_revalidation"]["outcome"] == "recovered_after_rebuild"
    initial_receipt = result["bundle_revalidation"]["initial"]
    assert initial_receipt["validation_status"] == "failed"
    assert initial_receipt["bundle_built"] is False
    assert initial_receipt["error_type"] == "BenchmarkGateError"
    assert "actual=0.25" in initial_receipt["error"]
    assert initial_receipt["metrics"]["prediction_agreement_rate"] == 0.25
    initial = Path(initial_receipt["artifacts"]["comparison_result"])
    assert initial.is_file()
    initial_summary = Path(initial_receipt["artifacts"]["eval_summary.json"])
    assert json.loads(initial_summary.read_text(encoding="utf-8")) == {"run": 1}
    assert (arguments.output / "model-a" / "workload-a" / "execution.reused-bundle.log").is_file()


def test_reused_bundle_accuracy_failure_rebuilds_once_and_confirms_failure(
    tmp_path,
    monkeypatch,
):
    result, commands, _arguments = _run_binding_with_comparison_results(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        raw_results=[
            {
                "status": "failed",
                "bundle_built": False,
                "prediction_agreement_rate": 0.25,
            },
            {
                "status": "failed",
                "bundle_built": True,
                "prediction_agreement_rate": 0.50,
            },
        ],
    )

    assert len(commands) == 2
    assert commands[1].count("--force-build") == 1
    assert result["validation"]["status"] == "failed"
    assert result["bundle_revalidation"]["outcome"] == "confirmed_after_rebuild"


def test_reused_bundle_revalidation_limit_caps_multi_binding_run(
    tmp_path,
    monkeypatch,
):
    results, commands, returncode, arguments = _run_multiple_bindings_with_comparison_results(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        models=("model-a", "model-b"),
        raw_results=[
            {
                "status": "failed",
                "bundle_built": False,
                "prediction_agreement_rate": 0.25,
            },
            {
                "status": "passed",
                "bundle_built": True,
                "prediction_agreement_rate": 1.0,
            },
            {
                "status": "failed",
                "bundle_built": False,
                "prediction_agreement_rate": 0.50,
            },
        ],
    )

    assert arguments.reused_bundle_revalidation_limit == 1
    assert len(commands) == 3
    assert sum("--force-build" in command for command in commands) == 1
    assert results["model-a"]["validation"]["status"] == "passed"
    assert results["model-a"]["bundle_revalidation"]["attempted"] is True
    assert results["model-a"]["bundle_revalidation"]["outcome"] == "recovered_after_rebuild"
    capped = results["model-b"]
    assert capped["validation"]["status"] == "failed"
    assert capped["comparison"]["status"] == "disagreement"
    assert capped["raw_result"]["bundle_built"] is False
    assert capped["bundle_revalidation"]["attempted"] is False
    assert capped["bundle_revalidation"]["attempt_count"] == 0
    assert capped["bundle_revalidation"]["run_attempt_limit"] == 1
    assert capped["bundle_revalidation"]["run_attempts_used"] == 1
    assert capped["bundle_revalidation"]["outcome"] == "not_attempted_run_limit_reached"
    assert capped["bundle_revalidation"]["initial"]["validation_status"] == "failed"
    assert returncode == 1


def test_zero_revalidation_limit_preserves_original_disagreement(
    tmp_path,
    monkeypatch,
):
    result, commands, _arguments = _run_binding_with_comparison_results(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        raw_results=[
            {
                "status": "failed",
                "bundle_built": False,
                "prediction_agreement_rate": 0.25,
            }
        ],
        extra_args=("--reused-bundle-revalidation-limit", "0"),
    )

    assert len(commands) == 1
    assert "--force-build" not in commands[0]
    assert result["comparison"]["status"] == "disagreement"
    assert result["validation"]["status"] == "failed"
    assert result["raw_result"]["bundle_built"] is False
    assert result["bundle_revalidation"]["attempted"] is False
    assert result["bundle_revalidation"]["run_attempt_limit"] == 0
    assert result["bundle_revalidation"]["run_attempts_used"] == 0
    assert result["bundle_revalidation"]["outcome"] == "not_attempted_run_limit_reached"


def test_nonaccuracy_failure_does_not_consume_revalidation_budget(
    tmp_path,
    monkeypatch,
):
    results, commands, returncode, _arguments = _run_multiple_bindings_with_comparison_results(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        models=("model-error", "model-accuracy"),
        raw_results=[
            {
                "status": "failed",
                "bundle_built": False,
                "error_type": "ReferenceSetupError",
                "error": "reference environment is unavailable",
            },
            {
                "status": "failed",
                "bundle_built": False,
                "prediction_agreement_rate": 0.25,
            },
            {
                "status": "passed",
                "bundle_built": True,
                "prediction_agreement_rate": 1.0,
            },
        ],
    )

    assert len(commands) == 3
    assert sum("--force-build" in command for command in commands) == 1
    assert results["model-error"]["execution"]["status"] == "error"
    assert "bundle_revalidation" not in results["model-error"]
    assert results["model-accuracy"]["validation"]["status"] == "passed"
    assert results["model-accuracy"]["bundle_revalidation"]["outcome"] == "recovered_after_rebuild"
    assert returncode == 1


def test_passing_reused_bundle_does_not_trigger_rebuild(
    tmp_path,
    monkeypatch,
):
    result, commands, _arguments = _run_binding_with_comparison_results(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        raw_results=[
            {
                "status": "passed",
                "bundle_built": False,
                "prediction_agreement_rate": 1.0,
            }
        ],
    )

    assert len(commands) == 1
    assert "--force-build" not in commands[0]
    assert result["validation"]["status"] == "passed"
    assert "bundle_revalidation" not in result


@pytest.mark.parametrize(
    ("flag", "forwarded"),
    [
        ("--force-build", "--force-build"),
        ("--no-build", "--require-prebuilt-bundles"),
    ],
)
def test_explicit_bundle_build_policy_does_not_trigger_revalidation(
    tmp_path,
    monkeypatch,
    flag,
    forwarded,
):
    result, commands, _arguments = _run_binding_with_comparison_results(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        raw_results=[
            {
                "status": "failed",
                "bundle_built": False,
                "prediction_agreement_rate": 0.25,
            }
        ],
        extra_args=(flag,),
    )

    assert len(commands) == 1
    assert forwarded in commands[0]
    assert result["validation"]["status"] == "failed"
    assert "bundle_revalidation" not in result


def test_compare_entrypoint_forwards_to_validation_backend(monkeypatch):
    captured = []

    def run(arguments):
        captured.extend(arguments)
        return 7

    monkeypatch.setattr(trtmc_compare.engine, "main", run)

    assert trtmc_compare.main(["--suite", "suite-a"]) == 7
    assert captured == ["eval", "--suite", "suite-a"]


@pytest.mark.parametrize(
    ("raw_result", "execution", "comparison", "validation"),
    [
        (
            {"status": "passed", "prediction_agreement_rate": 1.0},
            "completed",
            "agreement",
            "passed",
        ),
        (
            {"status": "failed", "prediction_agreement_rate": 0.5},
            "completed",
            "disagreement",
            "failed",
        ),
        (
            {
                "status": "failed",
                "prediction_agreement_rate": 0.5,
                "gate_failures": [
                    {
                        "gate": "min_prediction_agreement_rate",
                        "actual": 0.5,
                        "required": 0.98,
                    }
                ],
                "error_type": "BenchmarkGateError",
                "error": ("min_prediction_agreement_rate actual=0.5 required=0.98"),
            },
            "completed",
            "disagreement",
            "failed",
        ),
        (
            {"status": "failed", "error": "runner crashed"},
            "error",
            "not_run",
            "failed",
        ),
    ],
)
def test_result_statuses_separate_execution_from_agreement(
    raw_result,
    execution,
    comparison,
    validation,
):
    result = trtmc_validate._normalize_result(
        {
            "model": "model-a",
            "workload": "workload-a",
            "status": raw_result["status"],
            "raw_result": raw_result,
        }
    )

    assert result["execution"]["status"] == execution
    assert result["comparison"]["status"] == comparison
    assert result["validation"]["status"] == validation


@pytest.mark.parametrize("returncode", [0, 1])
def test_comparison_result_marks_missing_summary_as_execution_error(
    tmp_path,
    returncode,
):
    result = trtmc_validate._comparison_result(
        trtmc_validate.Binding("model-a", "workload-a"),
        case_dir=tmp_path,
        returncode=returncode,
        reference_environment=trtmc_validate.EnvironmentSelection(
            base_python="/profiles/python",
            names_and_paths=(("reference_common", "/profiles/python"),),
            overrides={},
        ),
        dataset_command="python tools/trtmc_validate.py model-a",
    )

    assert result["execution"] == {
        "status": "error",
        "exit_code": returncode,
    }
    assert result["comparison"]["status"] == "not_run"
    assert result["validation"]["status"] == "failed"
    assert result["raw_result"]["error_type"] == "ComparisonProcessError"
    assert "without writing" in result["raw_result"]["error"]
