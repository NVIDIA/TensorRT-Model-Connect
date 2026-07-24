# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from tools import model_plugin_isolation


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "tools" / "model_plugin_isolation.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def _make_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    manifests_dir = repo_root / "tests" / "e2e" / "models" / "decoder_family" / "manifests"
    manifests_dir.mkdir(parents=True)
    (manifests_dir / "decoder-small.json").write_text(
        json.dumps({
            "name": "decoder-small",
            "family": "decoder_family",
            "runtime_strategy": "llama_decoder_kv_cache",
        }),
        encoding="utf-8",
    )
    runtime_dir = repo_root / "src" / "runtime" / "models" / "llama"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "MODEL.toml").write_text(
        'id = "llama"\n'
        'runtime_library = "libtrtmc_model_llama.so"\n'
        'runtime_strategies = ["llama_decoder_kv_cache"]\n',
        encoding="utf-8",
    )
    return repo_root


def _add_case(
    repo_root: Path,
    *,
    name: str,
    family: str,
    runtime_id: str,
    runtime_strategy: str,
) -> None:
    manifests_dir = repo_root / "tests" / "e2e" / "models" / family / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / f"{name}.json").write_text(
        json.dumps(
            {
                "name": name,
                "family": family,
                "runtime_strategy": runtime_strategy,
            }
        ),
        encoding="utf-8",
    )
    runtime_dir = repo_root / "src" / "runtime" / "models" / runtime_id
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "MODEL.toml").write_text(
        f'id = "{runtime_id}"\n'
        f'runtime_library = "libtrtmc_model_{runtime_id}.so"\n'
        f'runtime_strategies = ["{runtime_strategy}"]\n',
        encoding="utf-8",
    )


def test_targets_resolve_e2e_model_to_runtime_plugin_owner(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path)
    result = _run(
        "targets",
        "--repo-root",
        str(repo_root),
        "--model",
        "decoder-small",
    )

    assert result.stdout.splitlines() == ["trtmc_model_llama"]


@pytest.mark.parametrize(
    ("family", "strategy", "target"),
    [
        ("qwen", "qwen_decoder_kv_cache", "trtmc_model_qwen"),
        ("deepseek_v2", "deepseek_v2_decoder_kv_cache", "trtmc_model_deepseek_v2"),
        ("olmo2", "olmo2_decoder_kv_cache", "trtmc_model_olmo2"),
        ("mixtral", "mixtral_decoder_moe", "trtmc_model_mixtral"),
        ("gpt_oss", "gpt_oss_decoder_moe", "trtmc_model_gpt_oss"),
        ("qwen_moe", "qwen_moe_decoder_moe", "trtmc_model_qwen_moe"),
        (
            "nemotron_labs_diffusion",
            "nemotron_labs_diffusion",
            "trtmc_model_nemotron_labs_diffusion",
        ),
        ("bloom", "bloom_decoder_kv_cache", "trtmc_model_bloom"),
        ("codegen", "codegen_decoder_kv_cache", "trtmc_model_codegen"),
        ("falcon", "falcon_decoder_kv_cache", "trtmc_model_falcon"),
        ("gemma", "gemma_decoder_kv_cache", "trtmc_model_gemma"),
        ("glm", "glm_decoder_kv_cache", "trtmc_model_glm"),
        ("gpt2", "gpt2_decoder_kv_cache", "trtmc_model_gpt2"),
        ("gpt_neo", "gpt_neo_decoder_kv_cache", "trtmc_model_gpt_neo"),
        ("gpt_neox", "gpt_neox_decoder_kv_cache", "trtmc_model_gpt_neox"),
        ("granite", "granite_decoder_kv_cache", "trtmc_model_granite"),
        ("internlm", "internlm_decoder_kv_cache", "trtmc_model_internlm"),
        ("llama", "llama_decoder_kv_cache", "trtmc_model_llama"),
        ("mistral", "mistral_decoder_kv_cache", "trtmc_model_mistral"),
        ("nemotron", "nemotron_decoder_kv_cache", "trtmc_model_nemotron"),
        ("olmo", "olmo_decoder_kv_cache", "trtmc_model_olmo"),
        ("opt", "opt_decoder_kv_cache", "trtmc_model_opt"),
        ("phi", "phi_decoder_kv_cache", "trtmc_model_phi"),
        ("phi_moe", "phi_moe_decoder_kv_cache", "trtmc_model_phi_moe"),
        ("stablelm", "stablelm_decoder_kv_cache", "trtmc_model_stablelm"),
        ("starcoder2", "starcoder2_decoder_kv_cache", "trtmc_model_starcoder2"),
        ("xglm", "xglm_decoder_kv_cache", "trtmc_model_xglm"),
    ],
)
def test_targets_resolve_model_owned_runtime_plugin(
    tmp_path: Path,
    family: str,
    strategy: str,
    target: str,
) -> None:
    repo_root = _make_repo(tmp_path)
    model_name = f"{family}-case"
    manifests_dir = repo_root / "tests" / "e2e" / "models" / family / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / f"{model_name}.json").write_text(
        json.dumps({
            "name": model_name,
            "family": family,
            "runtime_strategy": strategy,
        }),
        encoding="utf-8",
    )
    runtime_dir = repo_root / "src" / "runtime" / "models" / family
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "MODEL.toml").write_text(
        f'id = "{family}"\n'
        f'runtime_library = "libtrtmc_model_{family}.so"\n'
        f'runtime_strategies = ["{strategy}"]\n',
        encoding="utf-8",
    )

    result = _run(
        "targets",
        "--repo-root",
        str(repo_root),
        "--model",
        model_name,
    )

    assert result.stdout.splitlines() == [target]


def test_targets_resolve_model_owned_node_id_from_tests_file(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path)
    tests_file = tmp_path / "tests.txt"
    tests_file.write_text(
        "tests/e2e/models/decoder_family/test_decoder_family_e2e.py::test_model_e2e[decoder-small]\n",
        encoding="utf-8",
    )

    result = _run(
        "targets",
        "--repo-root",
        str(repo_root),
        "--tests-file",
        str(tests_file),
    )

    assert result.stdout.splitlines() == ["trtmc_model_llama"]


def test_prepare_copies_only_selected_runtime_plugin(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path)
    build_dir = tmp_path / "build"
    source_dir = build_dir / "models" / "llama"
    source_dir.mkdir(parents=True)
    source = source_dir / "libtrtmc_model_llama.so"
    source.write_bytes(b"fake-so")

    output_dir = tmp_path / "only-selected"
    result = _run(
        "prepare",
        "--repo-root",
        str(repo_root),
        "--model",
        "decoder-small",
        "--build-dir",
        str(build_dir),
        "--output-dir",
        str(output_dir),
    )

    copied = output_dir / "llama" / "libtrtmc_model_llama.so"
    assert copied.read_bytes() == b"fake-so"
    assert result.stdout.splitlines() == [f"trtmc_model_llama {copied}"]


def _add_projection_fixture_files(repo_root: Path) -> None:
    files = {
        "README.md": "generic root\n",
        "python/tensorrt_model_connect/families/__init__.py": "# registry\n",
        "python/tensorrt_model_connect/families/base.py": "# protocol\n",
        "python/tensorrt_model_connect/families/decoder_family/MODEL.toml": (
            'id = "decoder_family"\n'
        ),
        "python/tensorrt_model_connect/families/decoder_family/plugin.py": (
            "# selected builder\n"
        ),
        "python/tensorrt_model_connect/families/sibling/MODEL.toml": 'id = "sibling"\n',
        "python/tensorrt_model_connect/families/sibling/plugin.py": "# sibling builder\n",
        "src/runtime/core/core.cpp": "// shared runtime\n",
        "src/runtime/models/llama/plugin.cpp": "// selected runtime\n",
        "src/runtime/models/sibling/MODEL.toml": (
            'id = "sibling"\n'
            'runtime_library = "libtrtmc_model_sibling.so"\n'
            'runtime_plugins = ["plugin.cpp|register_sibling"]\n'
            'runtime_strategies = ["sibling_runtime"]\n'
        ),
        "src/runtime/models/sibling/plugin.cpp": "// sibling runtime\n",
        "tests/e2e_harness/contracts.py": "# shared harness\n",
        "tests/e2e/models/decoder_family/MODEL.toml": (
            'id = "decoder_family"\n'
        ),
        "tests/e2e/models/decoder_family/runner.py": "# selected E2E\n",
        "tests/e2e/models/sibling/MODEL.toml": 'id = "sibling"\n',
        "tests/e2e/models/sibling/runner.py": "# sibling E2E\n",
        "tests/cpp/models/llama/test_runtime.cpp": "// selected runtime test\n",
        "tests/cpp/models/sibling/test_runtime.cpp": "// sibling runtime test\n",
    }
    for relative, content in files.items():
        path = repo_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    subprocess.run(["git", "init", "-q", str(repo_root)], check=True)
    subprocess.run(["git", "-C", str(repo_root), "add", "."], check=True)


def test_stage_source_masks_sibling_model_roots(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path)
    _add_projection_fixture_files(repo_root)
    output_dir = tmp_path / "isolated"

    result = _run(
        "stage-source",
        "--repo-root",
        str(repo_root),
        "--model",
        "decoder-small",
        "--output-dir",
        str(output_dir),
    )

    assert "families=decoder_family" in result.stdout
    assert "runtime_plugins=llama" in result.stdout
    assert (output_dir / "README.md").read_text() == "generic root\n"
    assert (
        output_dir / "python/tensorrt_model_connect/families/__init__.py"
    ).is_file()
    assert (
        output_dir / "python/tensorrt_model_connect/families/base.py"
    ).is_file()
    assert (
        output_dir
        / "python/tensorrt_model_connect/families/decoder_family/plugin.py"
    ).is_file()
    assert not (
        output_dir / "python/tensorrt_model_connect/families/sibling"
    ).exists()
    assert (output_dir / "src/runtime/models/llama/plugin.cpp").is_file()
    assert not (output_dir / "src/runtime/models/sibling").exists()
    assert (
        output_dir / "tests/e2e/models/decoder_family/runner.py"
    ).is_file()
    assert not (output_dir / "tests/e2e/models/sibling").exists()
    assert (
        output_dir / "tests/cpp/models/llama/test_runtime.cpp"
    ).is_file()
    assert not (output_dir / "tests/cpp/models/sibling").exists()

    manifest = json.loads(
        (output_dir / ".trtmc-isolation.json").read_text(encoding="utf-8")
    )
    assert manifest["selected_models"] == ["decoder-small"]
    assert manifest["builder_families"] == ["decoder_family"]
    assert manifest["e2e_families"] == ["decoder_family"]
    assert manifest["runtime_plugins"] == [
        {
            "model_id": "llama",
            "library": "libtrtmc_model_llama.so",
            "strategies": ["llama_decoder_kv_cache"],
            "target": "trtmc_model_llama",
        }
    ]
    assert all(value > 0 for value in manifest["excluded_model_files"].values())


def test_stage_source_requires_clean_to_replace_output(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path)
    _add_projection_fixture_files(repo_root)
    output_dir = tmp_path / "isolated"
    output_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "stage-source",
            "--repo-root",
            str(repo_root),
            "--model",
            "decoder-small",
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "pass --clean to replace it" in result.stderr


@pytest.mark.parametrize("output", ["repo", "."])
def test_stage_source_rejects_output_that_contains_repo(
    tmp_path: Path,
    output: str,
) -> None:
    repo_root = _make_repo(tmp_path)
    _add_projection_fixture_files(repo_root)
    output_dir = repo_root if output == "repo" else tmp_path

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "stage-source",
            "--repo-root",
            str(repo_root),
            "--model",
            "decoder-small",
            "--output-dir",
            str(output_dir),
            "--clean",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "must not be the repository root or one of its parents" in result.stderr


def test_plan_groups_shared_family_runtime_and_splits_siblings(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path)
    _add_case(
        repo_root,
        name="decoder-medium",
        family="decoder_family",
        runtime_id="llama",
        runtime_strategy="llama_decoder_kv_cache",
    )
    _add_case(
        repo_root,
        name="encoder-small",
        family="encoder_family",
        runtime_id="bert",
        runtime_strategy="bert_encoder",
    )
    output_dir = tmp_path / "plan"

    result = _run(
        "plan",
        "--repo-root",
        str(repo_root),
        "--model",
        "decoder-small",
        "--model",
        "decoder-medium",
        "--model",
        "encoder-small",
        "--output-dir",
        str(output_dir),
    )

    assert "3 model(s) in 2 single-family isolation group(s)" in result.stdout
    plan = json.loads((output_dir / "plan.json").read_text(encoding="utf-8"))
    assert [group["id"] for group in plan["groups"]] == [
        "decoder_family--llama",
        "encoder_family--bert",
    ]
    assert plan["groups"][0]["models"] == ["decoder-medium", "decoder-small"]
    assert (output_dir / plan["groups"][0]["models_file"]).read_text(
        encoding="utf-8"
    ).splitlines() == [
        "decoder-medium",
        "decoder-small",
    ]
    assert plan["groups"][1]["runtime_plugin"]["target"] == "trtmc_model_bert"


def test_schedule_balances_groups_across_gpu_queues(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path)
    _add_case(
        repo_root,
        name="encoder-small",
        family="encoder_family",
        runtime_id="bert",
        runtime_strategy="bert_encoder",
    )
    plan_dir = tmp_path / "plan"
    _run(
        "plan",
        "--repo-root",
        str(repo_root),
        "--model",
        "decoder-small",
        "--model",
        "encoder-small",
        "--output-dir",
        str(plan_dir),
    )
    timing_path = tmp_path / "timing.json"
    timing_path.write_text(
        json.dumps(
            {
                "estimates_s": {
                    "decoder-small": 100,
                    "encoder-small": 20,
                }
            }
        ),
        encoding="utf-8",
    )
    schedule_dir = tmp_path / "schedule"

    result = _run(
        "schedule",
        "--plan",
        str(plan_dir / "plan.json"),
        "--output-dir",
        str(schedule_dir),
        "--timing-estimates",
        str(timing_path),
        "--gpu-id",
        "0",
        "--gpu-id",
        "2",
        "--build-overhead-seconds",
        "10",
    )

    assert "2 isolation group(s) across 2 GPU queue(s)" in result.stdout
    schedule = json.loads(
        (schedule_dir / "schedule.json").read_text(encoding="utf-8")
    )
    assert [
        item["group_id"] for item in schedule["assignments"]["0"]
    ] == ["decoder_family--llama"]
    assert [
        item["group_id"] for item in schedule["assignments"]["2"]
    ] == ["encoder_family--bert"]
    assert schedule["queue_estimated_seconds"] == {"0": 110.0, "2": 30.0}
    assert (schedule_dir / "gpu-0.txt").read_text(encoding="utf-8").strip().endswith(
        "groups/decoder_family--llama/group.json"
    )


def test_impact_models_selects_owned_rules_and_l0_replacements(tmp_path: Path) -> None:
    impact_path = tmp_path / "impact.json"
    impact_path.write_text(
        json.dumps(
            {
                "e2e_models": ["decoder-small", "decoder-large-l0", "shared-case"],
                "e2e_test_ids": [
                    "tests/e2e/models/decoder/test_decoder_e2e.py"
                    "::test_model_e2e[decoder-small]"
                ],
                "matched_rules": [
                    {
                        "file": "python/tensorrt_model_connect/families/decoder/plugin.py",
                        "rule": "family_package",
                        "models": ["decoder-small", "decoder-large"],
                    },
                    {
                        "file": "python/tensorrt_model_connect/checkpoint_mapper.py",
                        "rule": "shared_builder_module",
                        "models": ["shared-case"],
                    },
                ],
                "l0_replacements": [
                    {
                        "model": "decoder-large",
                        "replacement": "decoder-large-l0",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = _run("impact-models", "--impact-json", str(impact_path))

    assert result.stdout.splitlines() == ["decoder-large-l0", "decoder-small"]


def test_impact_models_excludes_non_runnable_ci_tiers(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path)
    manifest_path = (
        repo_root
        / "tests"
        / "e2e"
        / "models"
        / "decoder_family"
        / "manifests"
        / "decoder-small.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["testcases"] = [
        {"name": "decoder-small", "ci_tier": "multi_device"}
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    impact_path = tmp_path / "impact.json"
    impact_path.write_text(
        json.dumps(
            {
                "e2e_models": ["decoder-small"],
                "matched_rules": [
                    {
                        "file": "tests/e2e/models/decoder_family/runner.py",
                        "rule": "e2e_model_owned_test",
                        "models": ["decoder-small"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = _run(
        "impact-models",
        "--repo-root",
        str(repo_root),
        "--impact-json",
        str(impact_path),
        "--exclude-ci-tier",
        "multi_device",
    )

    assert result.stdout == ""


def _passing_result(model_name: str) -> dict[str, object]:
    return {
        "case_name": model_name,
        "status": "pass",
        "failure_type": None,
        "oracle_level": "L1_external_reference",
        "stages": {
            "full_inference": {
                "status": "passed",
                "metrics": {
                    "cosine": {
                        "value": 1.0,
                        "threshold": 0.99,
                        "passed": True,
                    }
                },
            }
        },
        "stage_outputs": {
            "trt_full_inference": {"metadata": {"returncode": 0}},
            "ref_full_inference": {"metadata": {"returncode": 0}},
        },
        "commands": [{"command": ["trtmc", "run"], "returncode": 0}],
    }


def _write_result(artifacts_dir: Path, model_name: str, result: object) -> None:
    model_dir = artifacts_dir / model_name
    model_dir.mkdir(parents=True)
    (model_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")


def test_verify_results_accepts_complete_passing_result(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    _write_result(artifacts_dir, "decoder-small", _passing_result("decoder-small"))
    report_path = tmp_path / "verification.json"

    result = _run(
        "verify-results",
        "--repo-root",
        str(repo_root),
        "--model",
        "decoder-small",
        "--artifacts-dir",
        str(artifacts_dir),
        "--report",
        str(report_path),
    )

    assert "PASS decoder-small" in result.stdout
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["results"][0]["proof_kind"] == "reference"


@pytest.mark.parametrize(
    ("oracle_level", "proof_kind"),
    [
        ("L1_external_reference", "reference"),
        ("L2_internal_reference", "reference"),
        ("L3_snapshot_regression", "snapshot_regression"),
        ("L4_invariants", "functional_invariant"),
        (None, "invalid"),
        ("", "invalid"),
        ("L5_unknown", "invalid"),
    ],
)
def test_verify_result_maps_oracle_without_overclaiming_reference_parity(
    tmp_path: Path,
    oracle_level: str | None,
    proof_kind: str,
) -> None:
    artifacts_dir = tmp_path / "artifacts"
    result_data = _passing_result("decoder-small")
    if oracle_level is None:
        result_data.pop("oracle_level")
    else:
        result_data["oracle_level"] = oracle_level
    _write_result(artifacts_dir, "decoder-small", result_data)
    verified = model_plugin_isolation._verify_model_result(
        "decoder-small", "decoder-small", artifacts_dir
    )

    assert verified["proof_kind"] == proof_kind
    assert verified["passed"] is (proof_kind != "invalid")
    if proof_kind == "invalid":
        assert any("oracle_level is" in error for error in verified["errors"])


def test_verify_results_accepts_advisory_metric_failure_in_passing_stage(
    tmp_path: Path,
) -> None:
    repo_root = _make_repo(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    result_data = _passing_result("decoder-small")
    metric = result_data["stages"]["full_inference"]["metrics"]["cosine"]
    metric["passed"] = False
    _write_result(artifacts_dir, "decoder-small", result_data)

    result = _run(
        "verify-results",
        "--repo-root",
        str(repo_root),
        "--model",
        "decoder-small",
        "--artifacts-dir",
        str(artifacts_dir),
    )

    assert "PASS decoder-small" in result.stdout


def test_verify_results_accepts_skipped_optional_stage(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    result_data = _passing_result("decoder-small")
    result_data["case_config"] = {
        "stages": [
            {"name": "advisory_probe", "required": False},
            {"name": "full_inference", "required": True},
        ]
    }
    result_data["stages"]["advisory_probe"] = {
        "status": "skipped",
        "metrics": {},
        "message": "No comparison logic for advisory probe",
    }
    _write_result(artifacts_dir, "decoder-small", result_data)

    result = _run(
        "verify-results",
        "--repo-root",
        str(repo_root),
        "--model",
        "decoder-small",
        "--artifacts-dir",
        str(artifacts_dir),
    )

    assert "PASS decoder-small" in result.stdout


@pytest.mark.parametrize(
    ("required", "stage_status", "returncode", "expected_error"),
    [
        (True, "skipped", 0, "status is 'skipped'"),
        (False, "failed", 0, "status is 'failed'"),
        (False, "error", 0, "status is 'error'"),
        (False, "skipped", 1, "returncode is 1"),
    ],
)
def test_verify_results_rejects_invalid_optional_stage_execution(
    tmp_path: Path,
    required: bool,
    stage_status: str,
    returncode: int,
    expected_error: str,
) -> None:
    repo_root = _make_repo(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    result_data = _passing_result("decoder-small")
    result_data["case_config"] = {
        "stages": [{"name": "advisory_probe", "required": required}]
    }
    result_data["stages"]["advisory_probe"] = {
        "status": stage_status,
        "metrics": {},
    }
    result_data["stage_outputs"]["advisory_probe"] = {
        "metadata": {"returncode": returncode}
    }
    _write_result(artifacts_dir, "decoder-small", result_data)

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "verify-results",
            "--repo-root",
            str(repo_root),
            "--model",
            "decoder-small",
            "--artifacts-dir",
            str(artifacts_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "FAIL decoder-small" in result.stdout
    assert expected_error in result.stderr


def test_verify_results_uses_first_testcase_when_model_has_no_same_named_case(
    tmp_path: Path,
) -> None:
    repo_root = _make_repo(tmp_path)
    manifest_path = (
        repo_root
        / "tests"
        / "e2e"
        / "models"
        / "decoder_family"
        / "manifests"
        / "decoder-small.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["name"] = "decoder-bundle"
    manifest["testcases"] = [
        {"name": "decoder-bundle-probe01"},
        {"name": "decoder-bundle-probe02"},
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    artifacts_dir = tmp_path / "artifacts"
    _write_result(
        artifacts_dir,
        "decoder-bundle-probe01",
        _passing_result("decoder-bundle-probe01"),
    )

    result = _run(
        "verify-results",
        "--repo-root",
        str(repo_root),
        "--model",
        "decoder-bundle",
        "--artifacts-dir",
        str(artifacts_dir),
    )

    assert "PASS decoder-bundle" in result.stdout


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (lambda result: result.update(status="skip"), "status is 'skip'"),
        (
            lambda result: result["stages"]["full_inference"].update(status="failed"),
            "status is 'failed'",
        ),
        (
            lambda result: result["stage_outputs"]["ref_full_inference"]["metadata"].update(
                returncode=1
            ),
            "returncode is 1",
        ),
    ],
)
def test_verify_results_rejects_incomplete_execution(
    tmp_path: Path,
    mutation,
    expected_error: str,
) -> None:
    repo_root = _make_repo(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    result_data = _passing_result("decoder-small")
    mutation(result_data)
    _write_result(artifacts_dir, "decoder-small", result_data)

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "verify-results",
            "--repo-root",
            str(repo_root),
            "--model",
            "decoder-small",
            "--artifacts-dir",
            str(artifacts_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "FAIL decoder-small" in result.stdout
    assert expected_error in result.stderr


def test_verify_results_rejects_missing_result(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "verify-results",
            "--repo-root",
            str(repo_root),
            "--model",
            "decoder-small",
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "result.json is missing" in result.stderr


def _write_build_ledger(
    ledger_dir: Path,
    model_name: str,
    bundle_path: Path,
    timing_path: Path,
    **overrides,
) -> None:
    ledger_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "identity": model_name,
        "status": "passed",
        "invocation_count": 1,
        "attempt_count": 1,
        "builder_pid": 222,
        "started_at": "2026-01-01T00:00:02+00:00",
        "recovery_attempts": [],
        "returncode": 0,
        "source_revision": "abc123",
        "bundle_path": str(bundle_path),
        "build_timing_path": str(timing_path),
    }
    payload.update(overrides)
    (ledger_dir / f"{model_name}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_verify_builds_accepts_exactly_one_completed_build_per_model(
    tmp_path: Path,
) -> None:
    models_file = tmp_path / "models.txt"
    models_file.write_text("decoder-small\nencoder-small\n", encoding="utf-8")
    ledger_dir = tmp_path / "engine-builds"
    for model_name in ("decoder-small", "encoder-small"):
        bundle_path = tmp_path / f"{model_name}.trtfb"
        bundle_path.write_bytes(b"bundle")
        timing_path = tmp_path / f"{model_name}-timing.json"
        timing_path.write_text("{}\n", encoding="utf-8")
        _write_build_ledger(
            ledger_dir,
            model_name,
            bundle_path,
            timing_path,
        )
    report_path = tmp_path / "build-verification.json"

    result = _run(
        "verify-builds",
        "--models-file",
        str(models_file),
        "--ledger-dir",
        str(ledger_dir),
        "--source-revision",
        "abc123",
        "--report",
        str(report_path),
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result.stdout.count("PASS") == 2
    assert report["passed"] is True
    assert report["builds_per_model"] == 1
    assert report["expected_models"] == ["decoder-small", "encoder-small"]


def test_verify_builds_accepts_one_recorded_sigsegv_recovery(
    tmp_path: Path,
) -> None:
    models_file = tmp_path / "models.txt"
    models_file.write_text("decoder-small\n", encoding="utf-8")
    ledger_dir = tmp_path / "engine-builds"
    bundle_path = tmp_path / "decoder-small.trtfb"
    bundle_path.write_bytes(b"bundle")
    timing_path = tmp_path / "decoder-small-timing.json"
    timing_path.write_text("{}\n", encoding="utf-8")
    _write_build_ledger(
        ledger_dir,
        "decoder-small",
        bundle_path,
        timing_path,
        attempt_count=2,
        recovery_attempts=[
            {
                "attempt": 1,
                "returncode": -signal.SIGSEGV,
                "signal": signal.SIGSEGV,
                "builder_pid": 111,
                "started_at": "2026-01-01T00:00:01+00:00",
                "recovered_at": "2026-01-01T00:00:02+00:00",
            }
        ],
    )

    result = _run(
        "verify-builds",
        "--models-file",
        str(models_file),
        "--ledger-dir",
        str(ledger_dir),
        "--source-revision",
        "abc123",
    )

    assert result.returncode == 0
    assert "PASS decoder-small" in result.stdout


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_recovery_builder_pid",
        "boolean_recovery_builder_pid",
        "same_builder_pid",
        "invalid_recovery_started_at",
        "invalid_recovered_at",
        "recovered_at_mismatch",
        "reversed_recovery_time",
        "extra_recovery_field",
    ],
)
def test_verify_builds_rejects_incomplete_fresh_process_recovery_evidence(
    tmp_path: Path,
    mutation: str,
) -> None:
    models_file = tmp_path / "models.txt"
    models_file.write_text("decoder-small\n", encoding="utf-8")
    ledger_dir = tmp_path / "engine-builds"
    bundle_path = tmp_path / "decoder-small.trtfb"
    bundle_path.write_bytes(b"bundle")
    timing_path = tmp_path / "decoder-small-timing.json"
    timing_path.write_text("{}\n", encoding="utf-8")
    recovery = {
        "attempt": 1,
        "returncode": -signal.SIGSEGV,
        "signal": signal.SIGSEGV,
        "builder_pid": 111,
        "started_at": "2026-01-01T00:00:01+00:00",
        "recovered_at": "2026-01-01T00:00:02+00:00",
    }
    if mutation == "missing_recovery_builder_pid":
        recovery.pop("builder_pid")
    elif mutation == "boolean_recovery_builder_pid":
        recovery["builder_pid"] = True
    elif mutation == "same_builder_pid":
        recovery["builder_pid"] = 222
    elif mutation == "invalid_recovery_started_at":
        recovery["started_at"] = "not-a-timestamp"
    elif mutation == "invalid_recovered_at":
        recovery["recovered_at"] = "2026-01-01T00:00:02"
    elif mutation == "recovered_at_mismatch":
        recovery["recovered_at"] = "2026-01-01T00:00:03+00:00"
    elif mutation == "reversed_recovery_time":
        recovery["started_at"] = "2026-01-01T00:00:03+00:00"
    else:
        recovery["unexpected"] = "field"
    _write_build_ledger(
        ledger_dir,
        "decoder-small",
        bundle_path,
        timing_path,
        attempt_count=2,
        recovery_attempts=[recovery],
    )

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        _run(
            "verify-builds",
            "--models-file",
            str(models_file),
            "--ledger-dir",
            str(ledger_dir),
            "--source-revision",
            "abc123",
        )

    assert exc_info.value.returncode == 1
    assert "complete ordered fresh-process SIGSEGV evidence" in exc_info.value.stderr


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("schema_version", True, "schema_version is True, expected 1"),
        ("schema_version", 2, "schema_version is 2, expected 1"),
        ("builder_pid", True, "builder_pid is True, expected a positive integer"),
        ("builder_pid", 0, "builder_pid is 0, expected a positive integer"),
        ("started_at", "not-a-timestamp", "expected a UTC timestamp"),
        ("started_at", "2026-01-01T00:00:02", "expected a UTC timestamp"),
        ("returncode", False, "returncode is False, expected 0"),
        ("returncode", 0.0, "returncode is 0.0, expected 0"),
    ],
)
def test_verify_builds_rejects_malformed_process_evidence(
    tmp_path: Path,
    field: str,
    value: object,
    expected_error: str,
) -> None:
    models_file = tmp_path / "models.txt"
    models_file.write_text("decoder-small\n", encoding="utf-8")
    ledger_dir = tmp_path / "engine-builds"
    bundle_path = tmp_path / "decoder-small.trtfb"
    bundle_path.write_bytes(b"bundle")
    timing_path = tmp_path / "decoder-small-timing.json"
    timing_path.write_text("{}\n", encoding="utf-8")
    _write_build_ledger(
        ledger_dir,
        "decoder-small",
        bundle_path,
        timing_path,
        **{field: value},
    )

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        _run(
            "verify-builds",
            "--models-file",
            str(models_file),
            "--ledger-dir",
            str(ledger_dir),
            "--source-revision",
            "abc123",
        )

    assert exc_info.value.returncode == 1
    assert expected_error in exc_info.value.stderr


@pytest.mark.parametrize(
    ("attempt_count", "recovery_attempts", "expected_error"),
    [
        (True, [], "attempt_count is True, expected 1 or 2"),
        (3, [], "attempt_count is 3, expected 1 or 2"),
        (2, [], "recovery_attempts does not match attempt_count"),
        (
            2,
            [{"attempt": 1, "returncode": -signal.SIGABRT, "signal": signal.SIGABRT}],
            "recovery_attempts must contain only ordered SIGSEGV recoveries",
        ),
    ],
)
def test_verify_builds_rejects_invalid_recovery_records(
    tmp_path: Path,
    attempt_count: int,
    recovery_attempts: list[dict[str, int]],
    expected_error: str,
) -> None:
    models_file = tmp_path / "models.txt"
    models_file.write_text("decoder-small\n", encoding="utf-8")
    ledger_dir = tmp_path / "engine-builds"
    bundle_path = tmp_path / "decoder-small.trtfb"
    bundle_path.write_bytes(b"bundle")
    timing_path = tmp_path / "decoder-small-timing.json"
    timing_path.write_text("{}\n", encoding="utf-8")
    _write_build_ledger(
        ledger_dir,
        "decoder-small",
        bundle_path,
        timing_path,
        attempt_count=attempt_count,
        recovery_attempts=recovery_attempts,
    )

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        _run(
            "verify-builds",
            "--models-file",
            str(models_file),
            "--ledger-dir",
            str(ledger_dir),
            "--source-revision",
            "abc123",
        )
    result = exc_info.value

    assert result.returncode == 1
    assert "FAIL decoder-small" in result.stdout
    assert expected_error in result.stderr


def test_verify_builds_rejects_boolean_invocation_count(tmp_path: Path) -> None:
    models_file = tmp_path / "models.txt"
    models_file.write_text("decoder-small\n", encoding="utf-8")
    ledger_dir = tmp_path / "engine-builds"
    bundle_path = tmp_path / "decoder-small.trtfb"
    bundle_path.write_bytes(b"bundle")
    timing_path = tmp_path / "decoder-small-timing.json"
    timing_path.write_text("{}\n", encoding="utf-8")
    _write_build_ledger(
        ledger_dir,
        "decoder-small",
        bundle_path,
        timing_path,
        invocation_count=True,
    )

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        _run(
            "verify-builds",
            "--models-file",
            str(models_file),
            "--ledger-dir",
            str(ledger_dir),
            "--source-revision",
            "abc123",
        )

    assert "invocation_count is True, expected 1" in exc_info.value.stderr


def test_verify_builds_rejects_a_missing_or_failed_build(tmp_path: Path) -> None:
    models_file = tmp_path / "models.txt"
    models_file.write_text("decoder-small\nencoder-small\n", encoding="utf-8")
    ledger_dir = tmp_path / "engine-builds"
    bundle_path = tmp_path / "decoder-small.trtfb"
    bundle_path.write_bytes(b"bundle")
    timing_path = tmp_path / "decoder-small-timing.json"
    timing_path.write_text("{}\n", encoding="utf-8")
    _write_build_ledger(
        ledger_dir,
        "decoder-small",
        bundle_path,
        timing_path,
        status="failed",
        returncode=1,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "verify-builds",
            "--models-file",
            str(models_file),
            "--ledger-dir",
            str(ledger_dir),
            "--source-revision",
            "abc123",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "missing build ledgers: encoder-small" in result.stderr
    assert "decoder-small: status is 'failed'" in result.stderr
