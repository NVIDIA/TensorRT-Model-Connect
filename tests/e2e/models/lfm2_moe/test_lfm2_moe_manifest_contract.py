# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static acceptance contracts for the pinned LFM2-8B-A1B sparse checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib

from tests.e2e_harness.manifest_loader import load_model_manifest
from tools.ci.model_proof_selection import ModelProofSelector
from tools.model_plugin_isolation import discover_e2e_manifests

_ROOT = Path(__file__).resolve().parent
_PINNED_8B_A1B_REVISION = "c1c44ff9fc00db3ebf4516970563f5f383d23670"
_STRATEGY = "lfm2_moe_hybrid_conv_attention"


def _read_manifest(name: str) -> dict:
    return json.loads((_ROOT / "manifests" / name).read_text(encoding="utf-8"))


def test_model_root_registers_the_single_moe_manifest() -> None:
    model = tomllib.loads((_ROOT / "MODEL.toml").read_text(encoding="utf-8"))
    assert model["id"] == "lfm2_moe"
    assert model["plugin"] == "lfm2_moe"
    assert model["test_manifests"] == ["manifests/lfm2-8b-a1b.json", "manifests/lfm2-8b-a1b-l0.json"]
    defaults = model["e2e_defaults"]["text_generation_causal"]
    assert defaults["reference_backend"] == "hf_transformers"
    assert defaults["oracle_level"] == "L1_external_reference"


def test_core_bf16_manifest_is_revision_pinned() -> None:
    manifest = _read_manifest("lfm2-8b-a1b.json")
    assert manifest["hf_id"] == "LiquidAI/LFM2-8B-A1B"
    assert manifest["hf_revision"] == _PINNED_8B_A1B_REVISION
    assert manifest["family"] == "lfm2_moe"
    assert manifest["runtime_strategy"] == _STRATEGY
    assert manifest["task_strategy"] == "text_generation_causal"
    assert manifest["precision"] == "bf16"
    assert manifest["max_cache_length"] == 512
    assert manifest["trust_remote_code"] is False

    loaded = load_model_manifest(_ROOT / "manifests" / "lfm2-8b-a1b.json")
    assert loaded.hf_revision == _PINNED_8B_A1B_REVISION
    assert loaded.build_case.runtime_strategy == _STRATEGY


def test_cases_cover_nightly_continuation_and_l0_replacement() -> None:
    manifest = _read_manifest("lfm2-8b-a1b.json")
    cases = {case["name"]: case for case in manifest["testcases"]}
    assert set(cases) == {"lfm2-8b-a1b"}
    l0_manifest = _read_manifest("lfm2-8b-a1b-l0.json")
    l0_cases = {case["name"]: case for case in l0_manifest["testcases"]}
    assert set(l0_cases) == {"lfm2-8b-a1b-l0"}

    nightly = cases["lfm2-8b-a1b"]
    assert nightly["ci_tier"] == "nightly_only"
    assert nightly["l0_replacement"] == "lfm2-8b-a1b-l0"
    assert nightly["reference_family"] == "lfm2_moe_greedy_continuation"
    assert nightly["user_contract"] == "continuation_parity"
    assert nightly["inputs"]["do_sample"] is False
    assert nightly["contract_config"]["use_chat_template"] is False

    smoke = l0_cases["lfm2-8b-a1b-l0"]
    assert smoke["ci_tier"] == "l0_only"
    assert smoke["reference_family"] == "lfm2_moe_greedy_continuation"
    assert smoke["inputs"]["do_sample"] is False
    assert smoke["max_new_tokens"] < nightly["max_new_tokens"]


def test_architecture_contract_matches_the_pinned_8b_a1b_schema() -> None:
    manifest = _read_manifest("lfm2-8b-a1b.json")
    nightly = next(case for case in manifest["testcases"] if case["name"] == "lfm2-8b-a1b")
    assert nightly["architecture_contract"] == {
        "layers": 24,
        "hidden_size": 2048,
        "intermediate_size": 7168,
        "attention_layers": [2, 6, 10, 14, 18, 21],
        "num_experts": 32,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 1792,
        "num_dense_layers": 2,
    }


def test_ci_selection_keeps_the_8b_run_nightly_with_an_l0_premerge_smoke() -> None:
    repo_root = _ROOT.parents[3]
    manifest = discover_e2e_manifests(repo_root)["lfm2-8b-a1b"]
    loaded = load_model_manifest(manifest.path)
    premerge_selector = ModelProofSelector("lfm2_moe", "premerge", "HEAD", repo_root)
    premerge = premerge_selector._select_cases(premerge_selector._cases(_ROOT), "lfm2_moe")
    nightly_selector = ModelProofSelector("lfm2_moe", "nightly", "HEAD", repo_root)
    nightly = nightly_selector._select_cases(nightly_selector._cases(_ROOT), "lfm2_moe")

    assert manifest.result_case == manifest.name == "lfm2-8b-a1b"
    assert loaded.build_case.name == manifest.result_case
    assert [case["name"] for case in premerge] == ["lfm2-8b-a1b-l0"]
    assert all(case["ci_tier"] != "nightly_only" for case in premerge)
    assert {case["name"] for case in nightly} == {"lfm2-8b-a1b"}


def test_manifest_excludes_dense_vl_and_external_runtime_dependencies() -> None:
    manifests = [_read_manifest(path.name) for path in sorted((_ROOT / "manifests").glob("*.json"))]
    assert len(manifests) == 2
    model_ids = {manifest["hf_id"] for manifest in manifests}
    assert all("-VL" not in model_id for model_id in model_ids)
    assert all("8B-A1B" in model_id for model_id in model_ids)

    local_runtime_sources = "\n".join(
        path.read_text(encoding="utf-8") for path in (_ROOT / "e2e_plugins").rglob("*.py")
    ).lower()
    for external_runtime in ("import vllm", "import sglang", "llama.cpp", "trtexec", "onnxruntime"):
        assert external_runtime not in local_runtime_sources


def test_reference_and_runner_stay_family_owned() -> None:
    reference = (_ROOT / "e2e_plugins" / "references" / "hf_transformers.py").read_text(
        encoding="utf-8"
    )
    runner = (_ROOT / "e2e_plugins" / "runners" / "text_generation.py").read_text(encoding="utf-8")

    assert "return_dict=True" in reference
    assert 'inputs["input_ids"]' in reference
    assert '"torch_dtype":' in reference
    assert 'model_kwargs["revision"] = revision' in reference
    for parameter in ("do_sample", "max_new_tokens"):
        assert parameter in reference

    assert "from tensorrt_model_connect.families.lfm2_moe.debug_runner import" in runner
    assert "family_runner_from_bundle(" in runner
    assert "from tensorrt_model_connect.families.lfm2." not in runner


def test_report_hook_degrades_without_raising() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "lfm2_moe_e2e_report",
        _ROOT / "e2e_plugins" / "report.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fragment = module.render({}, project_dir=str(_ROOT.parents[3]))
    assert isinstance(fragment, str) and fragment
    assert "render" in dir(module)
    with_counts = module.render(
        {
            "case": "lfm2-8b-a1b",
            "stage_outputs": {
                "trt_full_generation": {
                    "data": {"text": "x", "moe_expert_counts": [3, 1, 0, 5]},
                },
                "ref_full_generation": {"data": {"text": "x"}},
            },
        },
        project_dir=str(_ROOT.parents[3]),
    )
    assert "Routing summary" in with_counts
    without_counts = module.render(
        {"stage_outputs": {"trt_full_generation": {"data": {"text": "x"}}}},
        project_dir=str(_ROOT.parents[3]),
    )
    assert "debug runner" in without_counts


def test_every_case_has_a_family_local_threshold_sidecar() -> None:
    for manifest_path in sorted((_ROOT / "manifests").glob("*.json")):
        loaded = load_model_manifest(manifest_path)
        manifest = _read_manifest(manifest_path.name)
        assert {case.name for case in loaded.testcases} == {
            case["name"] for case in manifest["testcases"]
        }
        for case in manifest["testcases"]:
            threshold = _ROOT / "thresholds" / f"{case['name']}.json"
            assert threshold.is_file(), threshold
