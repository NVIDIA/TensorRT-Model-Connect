#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import model_ci


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "tools" / "model_ci.py"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write(repo: Path, relative: str, content: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _add_model(
    repo: Path,
    logical_id: str,
    *,
    runtime_id: str | None = None,
    strategy: str | None = None,
) -> None:
    runtime_id = runtime_id or logical_id
    strategy = strategy or f"{runtime_id}_runtime"
    _write(
        repo,
        f"python/tensorrt_model_connect/families/{logical_id}/MODEL.toml",
        f'id = "{logical_id}"\n',
    )
    _write(
        repo,
        f"python/tensorrt_model_connect/families/{logical_id}/plugin.py",
        f'MODEL = "{logical_id}"\n',
    )
    _write(
        repo,
        f"src/runtime/models/{runtime_id}/MODEL.toml",
        f'id = "{runtime_id}"\n'
        f'runtime_library = "libtrtmc_model_{runtime_id}.so"\n'
        f'runtime_strategies = ["{strategy}"]\n',
    )
    _write(
        repo,
        f"src/runtime/models/{runtime_id}/plugin.cpp",
        f"// {runtime_id}\n",
    )
    _write(
        repo,
        f"tests/e2e/models/{logical_id}/MODEL.toml",
        f'id = "{logical_id}"\ntest_manifests = ["manifests/{logical_id}.json"]\n',
    )
    _write(
        repo,
        f"tests/e2e/models/{logical_id}/manifests/{logical_id}.json",
        json.dumps(
            {
                "name": logical_id,
                "family": logical_id,
                "runtime_strategy": strategy,
                "testcases": [{"name": logical_id}],
            }
        )
        + "\n",
    )
    _write(
        repo,
        f"tests/cpp/models/{runtime_id}/test_{runtime_id}.cpp",
        f"// {runtime_id} test\n",
    )


def _make_repo(
    tmp_path: Path,
    *,
    model_ids: tuple[str, ...] = ("model_a", "model_b"),
) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Model CI Test")
    _git(repo, "config", "user.email", "model-ci@example.com")
    for model_id in model_ids:
        _add_model(repo, model_id)
    _write(repo, "python/tensorrt_model_connect/families/__init__.py", "# registry\n")
    _write(
        repo,
        "pyproject.toml",
        "[project]\n"
        'dependencies = ["runtime-package>=1"]\n\n'
        "[project.optional-dependencies]\n"
        'test = ["pytest>=7"]\n',
    )
    _write(repo, "CMakeLists.txt", "# platform build\n")
    _write(repo, "examples/byok/identity_copy_kernel.cpp", "// BYOK example\n")
    _write(repo, "src/runtime/core/core.cpp", "// platform core\n")
    _write(repo, "README.md", "# Documentation\n")
    _write(repo, "tests/__init__.py", "")
    _write(repo, "tests/builder/__init__.py", "")
    for support_path in (
        "tests/builder/conftest.py",
        "tests/builder/debug_runner_test_support.py",
        "tests/builder/family_plugin_test_mixin.py",
        "tests/builder/family_plugin_test_support.py",
        "tests/builder/family_plugin_tester.py",
    ):
        _write(repo, support_path, "# shared test support\n")
    _write(repo, "tests/builder/test_checkpoint_mapper.py", "# unrelated test suite\n")
    _write(repo, "tests/builder/test_debug_runner.py", "# unrelated test suite\n")
    _write(repo, "tests/runtime_strategy_matrix.yaml", "strategies: []\n")
    for source in sorted((REPO_ROOT / "tools/ci").glob("*.py")):
        _write(repo, f"tools/ci/{source.name}", f"# projected CI module: {source.name}\n")
    _write(
        repo,
        ".github/scripts/write-model-proof-fallback-report.py",
        "#!/usr/bin/env python3\n",
    )
    _write(repo, "scripts/generate_e2e_report.py", "# report generator\n")
    _write(repo, "scripts/generate_e2e_report_assets/e2e_report.css", "/* report */\n")
    _write(repo, "scripts/generate_e2e_report_assets/e2e_report.js", "// report\n")
    _write(repo, "scripts/reporting/__init__.py", "")
    _write(repo, "scripts/reporting/vlm_assessment.py", "# report component\n")
    _write(repo, "scripts/schedule_e2e.py", "# shared E2E scheduler\n")
    _write(repo, "scripts/hf_cache_download_worker.py", "# cache download worker\n")
    _write(repo, "scripts/warm_hf_cache.py", "# cache check\n")
    _write(repo, "tools/__init__.py", "")
    _write(repo, "tools/diff_logits.py", "# shared logits diff\n")
    _write(repo, "tools/diff_vl.py", "# shared vision-language diff\n")
    _write(repo, "tools/diffusion_helpers.py", "# shared diffusion helpers\n")
    _write(repo, "tools/elf_hf_reference.py", "# ELF reference helper\n")
    _write(repo, "tools/model_plugin_isolation.py", "# proof verifier\n")
    for reference_runner in (
        "tools/reference/elf_prepared.py",
        "tools/reference/plugin_reference.py",
        "tools/reference/speech.py",
        "tools/reference/transformers_encoder.py",
        "tools/reference/transformers_text.py",
        "tools/reference/transformers_vlm.py",
    ):
        _write(repo, reference_runner, "# task-eval reference runner\n")
    _write(repo, "tools/test_impact.py", "# shared impact analyzer\n")
    _write(repo, "tools/trtmc_reference.py", "# task-eval reference runner\n")
    for validation_module in (
        "tools/validation/__init__.py",
        "tools/validation/artifacts.py",
        "tools/validation/catalog.py",
        "tools/validation/engine.py",
        "tools/validation/gate_policy.py",
        "tools/validation/model_plugin_contract.py",
    ):
        _write(repo, validation_module, "# validation module\n")
    _write(repo, "tests/validation/workloads.yaml", "suites: []\n")
    _write(repo, "tests/tools/test_validation_engine.py", "# validation unit tests\n")
    _write(repo, "tools/tool_helpers.py", "# shared tool helpers\n")
    _write(repo, "scripts/repro_trt_fp8_mha.py", "# unrelated model script\n")
    _write(repo, "tools/diff_t5.py", "# unrelated model diff\n")
    _write(repo, "tools/validate_t5.py", "# unrelated model validator\n")
    return repo, _commit(repo, "initial")


def _run(
    repo: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *args, "--repo-root", str(repo)],
        check=check,
        capture_output=True,
        text=True,
    )


def _impact(
    repo: Path,
    base: str,
    head: str,
    *,
    fallback_models: tuple[str, ...] = ("model_a",),
) -> dict[str, object]:
    args = [
        "impact",
        "--base",
        base,
        "--head",
        head,
        "--platform-change-policy",
        "fallback",
    ]
    for model in fallback_models:
        args.extend(("--fallback-model", model))
    result = _run(repo, *args)
    return json.loads(result.stdout)


def test_validate_and_all_emit_deterministic_matrix_and_github_outputs(
    tmp_path: Path,
) -> None:
    repo, revision = _make_repo(tmp_path)
    validated = json.loads(_run(repo, "validate", "--revision", revision).stdout)
    output = tmp_path / "github-output"

    result = json.loads(
        _run(
            repo,
            "all",
            "--revision",
            revision,
            "--github-output",
            str(output),
        ).stdout
    )

    assert validated["models"] == ["model_a", "model_b"]
    assert result["schema_version"] == 3
    assert result["direct_models"] == ["model_a", "model_b"]
    assert result["fallback_models"] == []
    assert result["matrix"] == {
        "include": [
            {"model": "model_a", "selection_kind": "direct"},
            {"model": "model_b", "selection_kind": "direct"},
        ]
    }
    assert result["expected_cases_by_model"] == {
        "model_a": ["model_a"],
        "model_b": ["model_b"],
    }
    assert result["expected_result_count"] == 2
    assert output.read_text(encoding="utf-8").splitlines() == [
        'matrix={"include":[{"model":"model_a","selection_kind":"direct"},'
        '{"model":"model_b","selection_kind":"direct"}]}',
        "has_models=true",
        'affected_models=["model_a","model_b"]',
        'direct_models=["model_a","model_b"]',
        "fallback_models=[]",
        "expected_count=2",
        "mode=all",
        "run_unit_tests=false",
        "unit_scope=none",
        'expected_cases_by_model={"model_a":["model_a"],"model_b":["model_b"]}',
        "expected_result_count=2",
    ]


def test_validate_rejects_multi_gpu_case_outside_multi_device_tier(tmp_path: Path) -> None:
    repo, _ = _make_repo(tmp_path)
    manifest_path = repo / "tests/e2e/models/model_a/manifests/model_a.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["build_args"] = {"parallel": {"mode": "tensor_parallel", "tp_size": 4}}
    manifest["distributed_runtime"] = {"enabled": True, "world_size": 4}
    manifest["testcases"] = [{"name": "model_a-tp4", "ci_tier": "nightly_only"}]
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    revision = _commit(repo, "add invalid tp4 tier")

    result = _run(repo, "validate", "--revision", revision, check=False)

    assert result.returncode != 0
    assert "requires 4 GPUs" in result.stderr
    assert "ci_tier='multi_device'" in result.stderr


def test_validate_uses_gpu_count_preflight_for_device_tier(tmp_path: Path) -> None:
    repo, _ = _make_repo(tmp_path)
    manifest_path = repo / "tests/e2e/models/model_a/manifests/model_a.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["testcases"] = [
        {
            "name": "model_a-multi-gpu",
            "ci_tier": "nightly_only",
            "preflight_requirements": [
                {"kind": "gpu_count_min", "args": {"count": 2}, "gating": True}
            ],
        }
    ]
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    revision = _commit(repo, "add invalid multi-gpu preflight tier")

    result = _run(repo, "validate", "--revision", revision, check=False)

    assert result.returncode != 0
    assert "requires 2 GPUs" in result.stderr
    assert "ci_tier='multi_device'" in result.stderr


def test_nightly_matrix_schedules_exclusive_then_shared_and_longest_first(
    tmp_path: Path,
) -> None:
    repo, _ = _make_repo(tmp_path)
    for model in ("model_c", "model_d", "model_e", "model_f"):
        _add_model(repo, model)
    estimates = {
        "model_a-fast": 100,
        "model_a-slow": 150,
        "model_b-case": 200,
        "model_d-fast": 150,
        "model_d-slow": 200,
        "model_e-case": 300,
    }
    for model in ("model_a", "model_b", "model_c", "model_d", "model_e", "model_f"):
        manifest_path = repo / f"tests/e2e/models/{model}/manifests/{model}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if model == "model_a":
            manifest["testcases"] = [
                {"name": "model_a-fast", "ci_tier": "contract_only"},
                {"name": "model_a-slow", "ci_tier": "contract_only"},
                {"name": "model_a-tp", "ci_tier": "multi_device"},
            ]
        elif model == "model_d":
            manifest["testcases"] = [
                {"name": "model_d-fast", "ci_tier": "l0_only"},
                {"name": "model_d-slow", "ci_tier": "l0_only"},
            ]
        elif model in {"model_c", "model_f"}:
            manifest["testcases"] = [{"name": f"{model}-nightly", "ci_tier": "nightly_only"}]
        else:
            manifest["testcases"] = [{"name": f"{model}-case", "ci_tier": "l0_only"}]
        if model in {"model_d", "model_e", "model_f"}:
            manifest["e2e_parallel_resource"] = "exclusive_gpu"
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    _write(
        repo,
        "tests/e2e/timing_estimates.json",
        json.dumps(
            {
                "schema_version": 1,
                "estimates_s": estimates,
            }
        )
        + "\n",
    )
    revision = _commit(repo, "add timing estimates")

    result = json.loads(_run(repo, "all", "--revision", revision).stdout)

    # Exclusive-GPU models are emitted first. A selected nightly-only case with
    # no timing makes the owner unknown. Known durations sum every selected
    # production case, or every L0 case when it is the owner's nightly fallback.
    assert result["affected_models"] == [
        "model_a",
        "model_b",
        "model_c",
        "model_d",
        "model_e",
        "model_f",
    ]
    assert result["matrix"] == {
        "include": [
            {"model": "model_f", "selection_kind": "direct"},
            {"model": "model_d", "selection_kind": "direct"},
            {"model": "model_e", "selection_kind": "direct"},
            {"model": "model_c", "selection_kind": "direct"},
            {"model": "model_a", "selection_kind": "direct"},
            {"model": "model_b", "selection_kind": "direct"},
        ]
    }
    assert result["expected_cases_by_model"] == {
        "model_a": ["model_a-fast", "model_a-slow"],
        "model_b": ["model_b-case"],
        "model_c": ["model_c-nightly"],
        "model_d": ["model_d-fast", "model_d-slow"],
        "model_e": ["model_e-case"],
        "model_f": ["model_f-nightly"],
    }
    assert result["expected_result_count"] == 8


def test_all_rejects_an_owner_without_a_single_gpu_nightly_case(tmp_path: Path) -> None:
    repo, _ = _make_repo(tmp_path, model_ids=("model_a",))
    manifest_path = repo / "tests/e2e/models/model_a/manifests/model_a.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["testcases"] = [{"name": "model_a-tp", "ci_tier": "multi_device"}]
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    revision = _commit(repo, "leave no single gpu case")

    result = _run(repo, "all", "--revision", revision, check=False)

    assert result.returncode == 2
    assert "model 'model_a' has no active single-GPU nightly E2E case" in result.stderr


def test_impact_selects_only_model_a(tmp_path: Path) -> None:
    repo, base = _make_repo(tmp_path)
    _write(
        repo,
        "python/tensorrt_model_connect/families/model_a/plugin.py",
        'MODEL = "model_a_changed"\n',
    )
    head = _commit(repo, "change a")

    result = _impact(repo, base, head)

    assert result["mode"] == "models"
    assert result["affected_models"] == ["model_a"]
    assert result["direct_models"] == ["model_a"]
    assert result["fallback_models"] == []
    assert result["matrix"] == {"include": [{"model": "model_a", "selection_kind": "direct"}]}
    assert result["run_unit_tests"] is True
    assert result["unit_scope"] == "all"


@pytest.mark.parametrize("move_nested_contract", (False, True))
def test_runtime_test_manifest_metadata_runs_units_without_model_proof(
    tmp_path: Path,
    move_nested_contract: bool,
) -> None:
    repo, _ = _make_repo(tmp_path)
    manifest = repo / "src/runtime/models/model_a/MODEL.toml"
    prefix = (
        'id = "model_a"\n'
        'runtime_library = "libtrtmc_model_model_a.so"\n'
        'runtime_strategies = ["model_a_runtime"]\n'
    )
    test_entry = '  "test_model_a|test_model_a.cpp|_|_|REQUIRES_TRT",\n'
    if move_nested_contract:
        manifest.write_text(
            prefix
            + '[validation_profiles]\n'
            + 'decoder_debug = ["model_a_runtime"]\n'
            + "runtime_tests = [\n"
            + test_entry
            + "]\n",
            encoding="utf-8",
        )
    else:
        manifest.write_text(
            prefix + "runtime_tests = [\n" + test_entry + "]\n",
            encoding="utf-8",
        )
    base = _commit(repo, "add runtime test metadata")

    head_entry = test_entry.replace("REQUIRES_TRT", "REQUIRES_TRT,REQUIRES_GPU")
    suffix = (
        '[validation_profiles]\n'
        'decoder_debug = ["model_a_runtime"]\n'
        if move_nested_contract
        else ""
    )
    manifest.write_text(
        prefix + "runtime_tests = [\n" + head_entry + "]\n" + suffix,
        encoding="utf-8",
    )
    head = _commit(repo, "update runtime test metadata")

    result = _impact(repo, base, head)

    assert result["mode"] == "unit"
    assert result["affected_models"] == []
    assert result["direct_models"] == []
    assert result["fallback_models"] == []
    assert result["matrix"] == {"include": []}
    assert result["run_unit_tests"] is True
    assert result["unit_scope"] == "all"
    assert {
        classification["kind"]
        for change in result["changes"]
        for classification in change["classifications"]
    } == {"unit_tests"}


def test_runtime_manifest_semantic_change_keeps_direct_model_proof(tmp_path: Path) -> None:
    repo, _ = _make_repo(tmp_path)
    manifest = repo / "src/runtime/models/model_a/MODEL.toml"
    manifest.write_text(
        'id = "model_a"\n'
        'runtime_library = "libtrtmc_model_model_a.so"\n'
        'runtime_strategies = ["model_a_runtime"]\n'
        'runtime_tests = ["test_model_a|test_model_a.cpp|_|_|REQUIRES_TRT"]\n',
        encoding="utf-8",
    )
    base = _commit(repo, "add runtime test metadata")
    manifest.write_text(
        'id = "model_a"\n'
        'runtime_library = "libtrtmc_model_model_a_v2.so"\n'
        'runtime_strategies = ["model_a_runtime"]\n'
        'runtime_tests = '
        '["test_model_a|test_model_a.cpp|_|_|REQUIRES_TRT,REQUIRES_GPU"]\n',
        encoding="utf-8",
    )
    head = _commit(repo, "change runtime library and tests")

    result = _impact(repo, base, head)

    assert result["mode"] == "models"
    assert result["affected_models"] == ["model_a"]
    assert result["direct_models"] == ["model_a"]
    assert result["matrix"] == {
        "include": [{"model": "model_a", "selection_kind": "direct"}]
    }


@pytest.mark.parametrize("module_marker", (False, True))
def test_pytest_resource_marker_only_change_runs_units_without_model_proof(
    tmp_path: Path,
    module_marker: bool,
) -> None:
    repo, _ = _make_repo(tmp_path)
    path = "tests/e2e/models/model_a/test_builder_contract.py"
    _write(repo, path, "import pytest\n\ndef test_contract():\n    assert True\n")
    base = _commit(repo, "add model test contract")
    if module_marker:
        updated = (
            "import pytest\n\n"
            "pytestmark = [pytest.mark.gpu, pytest.mark.trt]\n\n"
            "def test_contract():\n    assert True\n"
        )
    else:
        updated = (
            "import pytest\n\n"
            "@pytest.mark.gpu\n"
            "@pytest.mark.trt\n"
            "def test_contract():\n    assert True\n"
        )
    _write(repo, path, updated)
    head = _commit(repo, "classify model test resources")

    result = _impact(repo, base, head)

    assert result["mode"] == "unit"
    assert result["affected_models"] == []
    assert result["direct_models"] == []
    assert result["matrix"] == {"include": []}
    assert result["unit_scope"] == "all"


def test_pytest_marker_and_test_semantic_change_keeps_direct_model_proof(
    tmp_path: Path,
) -> None:
    repo, _ = _make_repo(tmp_path)
    path = "tests/e2e/models/model_a/test_builder_contract.py"
    _write(repo, path, "import pytest\n\ndef test_contract():\n    assert True\n")
    base = _commit(repo, "add model test contract")
    _write(
        repo,
        path,
        "import pytest\n\n"
        "@pytest.mark.gpu\n"
        "def test_contract():\n    assert 2 + 2 == 4\n",
    )
    head = _commit(repo, "change marker and test behavior")

    result = _impact(repo, base, head)

    assert result["mode"] == "models"
    assert result["direct_models"] == ["model_a"]


def test_family_python_unit_test_change_does_not_select_model_proof(tmp_path: Path) -> None:
    repo, base = _make_repo(tmp_path)
    _write(
        repo,
        "python/tensorrt_model_connect/families/model_a/tests/test_family.py",
        "def test_family():\n    assert True\n",
    )
    head = _commit(repo, "add family unit test")

    result = _impact(repo, base, head)

    assert result["mode"] == "unit"
    assert result["affected_models"] == []
    assert result["direct_models"] == []
    assert result["matrix"] == {"include": []}
    assert result["unit_scope"] == "all"


@pytest.mark.parametrize(
    "path",
    (
        "python/tensorrt_model_connect/families/model_a/graph_ops.py",
        "python/tensorrt_model_connect/families/model_a/graph_blocks.py",
        "python/tensorrt_model_connect/families/model_a/model/model.py",
        "python/tensorrt_model_connect/families/model_a/debug_runner.py",
        "tests/e2e/models/model_a/model.l0.json",
        "src/runtime/models/model_a/model_a.cpp",
    ),
)
def test_model_owned_change_runs_all_cpu_units(
    tmp_path: Path,
    path: str,
) -> None:
    repo, base = _make_repo(tmp_path)
    _write(repo, path, "# graph implementation\n")
    head = _commit(repo, "change family graph source")

    result = _impact(repo, base, head)

    assert result["mode"] == "models"
    assert result["affected_models"] == ["model_a"]
    assert result["direct_models"] == ["model_a"]
    assert result["run_unit_tests"] is True
    assert result["unit_scope"] == "all"


def test_model_owned_deletion_runs_all_cpu_units(tmp_path: Path) -> None:
    repo, _ = _make_repo(tmp_path)
    path = "python/tensorrt_model_connect/families/model_a/graph_ops.py"
    _write(repo, path, "# graph implementation\n")
    base = _commit(repo, "add family graph source")
    (repo / path).unlink()
    head = _commit(repo, "delete family graph source")

    result = _impact(repo, base, head)

    assert result["mode"] == "models"
    assert result["affected_models"] == ["model_a"]
    assert result["run_unit_tests"] is True
    assert result["unit_scope"] == "all"


def test_mixed_model_and_cli_change_runs_all_units(tmp_path: Path) -> None:
    repo, base = _make_repo(tmp_path)
    _write(
        repo,
        "python/tensorrt_model_connect/families/model_a/graph_ops.py",
        "# graph implementation\n",
    )
    _write(repo, "src/runtime/config/cli_support.cpp", "// CLI implementation\n")
    head = _commit(repo, "change graph and CLI sources")

    result = _impact(repo, base, head)

    assert result["mode"] == "models"
    assert result["affected_models"] == ["model_a"]
    assert result["run_unit_tests"] is True
    assert result["unit_scope"] == "all"


@pytest.mark.parametrize(
    ("path", "content"),
    (
        (
            "python/tensorrt_model_connect/families/model_b/optimized_adapter/adapter.py",
            'MODEL = "model_b"\n',
        ),
        (
            "python/tensorrt_model_connect/families/model_b/optimized_adapter/"
            "profiles/example.toml",
            'profile_id = "example"\n',
        ),
        (
            "src/runtime/models/model_b/optimized_adapter/adapter.cpp",
            "// model_b optimized runtime adapter\n",
        ),
        (
            "tests/e2e/models/model_b/optimized_adapter/test_contract.py",
            "def test_contract():\n    assert True\n",
        ),
    ),
)
def test_model_owned_adapter_change_selects_only_its_family(
    tmp_path: Path,
    path: str,
    content: str,
) -> None:
    repo, base = _make_repo(tmp_path)
    _write(repo, path, content)
    head = _commit(repo, "add model b adapter file")

    result = _impact(repo, base, head)

    assert result["mode"] == "models"
    assert result["affected_models"] == ["model_b"]
    assert result["direct_models"] == ["model_b"]
    assert result["fallback_models"] == []
    assert result["matrix"] == {"include": [{"model": "model_b", "selection_kind": "direct"}]}
    assert {
        classification["kind"]
        for change in result["changes"]
        for classification in change["classifications"]
    } == {"model"}


def test_impact_allows_head_to_migrate_legacy_multi_gpu_tier(tmp_path: Path) -> None:
    repo, _ = _make_repo(tmp_path)
    manifest_path = repo / "tests/e2e/models/model_a/manifests/model_a.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["build_args"] = {"parallel": {"mode": "tensor_parallel", "tp_size": 4}}
    manifest["distributed_runtime"] = {"enabled": True, "world_size": 4}
    manifest["testcases"] = [{"name": "model_a-tp4", "ci_tier": "nightly_only"}]
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    base = _commit(repo, "add legacy tp4 tier")

    base_validation = _run(repo, "validate", "--revision", base, check=False)
    assert base_validation.returncode == 2
    assert "ci_tier='multi_device'" in base_validation.stderr

    manifest["testcases"] = [{"name": "model_a-tp4", "ci_tier": "multi_device"}]
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    head = _commit(repo, "migrate tp4 tier")

    result = _impact(repo, base, head)

    assert result["mode"] == "models"
    assert result["affected_models"] == ["model_a"]
    assert result["direct_models"] == ["model_a"]
    assert json.loads(_run(repo, "validate", "--revision", head).stdout)["models"] == [
        "model_a",
        "model_b",
    ]


def test_impact_selects_each_modified_model_once(tmp_path: Path) -> None:
    repo, base = _make_repo(tmp_path)
    _write(repo, "src/runtime/models/model_a/plugin.cpp", "// changed a\n")
    _write(
        repo,
        "tests/e2e/models/model_b/manifests/model_b.json",
        json.dumps(
            {
                "name": "model_b",
                "family": "model_b",
                "runtime_strategy": "model_b_runtime",
                "changed": True,
            }
        )
        + "\n",
    )
    head = _commit(repo, "change a and b")

    result = _impact(repo, base, head)

    assert result["affected_models"] == ["model_a", "model_b"]
    assert result["expected_count"] == 2


def test_impact_treats_legal_and_docs_as_no_model_change(tmp_path: Path) -> None:
    repo, base = _make_repo(tmp_path)
    _write(repo, "README.md", "# Updated documentation\n")
    _write(repo, "NOTICE", "Legal notice\n")
    head = _commit(repo, "docs")

    result = _impact(repo, base, head)

    assert result["mode"] == "none"
    assert result["has_models"] is False
    assert result["matrix"] == {"include": []}
    assert result["run_unit_tests"] is True
    assert result["unit_scope"] == "all"


@pytest.mark.parametrize(
    ("path", "expected_scope", "expected_kind"),
    (
        (
            "website/docs/wiki/Agentic-Quantization-Core-Minimal-Plan.md",
            "all",
            "unit_builder",
        ),
        ("tools/ci/README.md", "all", "unit_tests"),
    ),
)
def test_impact_runs_units_for_test_consumed_docs(
    tmp_path: Path,
    path: str,
    expected_scope: str,
    expected_kind: str,
) -> None:
    repo, base = _make_repo(tmp_path)
    _write(repo, path, "# Updated test-consumed documentation\n")
    head = _commit(repo, "update test-consumed documentation")

    result = _impact(repo, base, head)

    assert result["mode"] == "unit"
    assert result["affected_models"] == []
    assert result["direct_models"] == []
    assert result["fallback_models"] == []
    assert result["matrix"] == {"include": []}
    assert result["run_unit_tests"] is True
    assert result["unit_scope"] == expected_scope
    assert {
        classification["kind"]
        for change in result["changes"]
        for classification in change["classifications"]
    } == {expected_kind}


def test_release_performance_config_runs_source_contracts_without_model_fallback(
    tmp_path: Path,
) -> None:
    repo, base = _make_repo(tmp_path)
    _write(
        repo,
        "benchmarks/performance/release.yaml",
        "entries:\n  - id: model-a.generate\n",
    )
    head = _commit(repo, "update release performance contract")

    result = _impact(repo, base, head)

    assert result["mode"] == "unit"
    assert result["affected_models"] == []
    assert result["direct_models"] == []
    assert result["fallback_models"] == []
    assert result["matrix"] == {"include": []}
    assert result["run_unit_tests"] is True
    assert result["unit_scope"] == "all"
    assert {
        classification["kind"]
        for change in result["changes"]
        for classification in change["classifications"]
    } == {"unit_tests"}


@pytest.mark.parametrize(
    "dockerfile", ("Dockerfile.dev.aarch64", "Dockerfile.dev.x86")
)
def test_source_container_runs_units_without_model_fallback(
    tmp_path: Path, dockerfile: str,
) -> None:
    repo, base = _make_repo(tmp_path)
    _write(repo, dockerfile, "FROM scratch\n")
    head = _commit(repo, "add source container")

    result = _impact(repo, base, head)

    assert result["mode"] == "unit"
    assert result["affected_models"] == []
    assert result["fallback_models"] == []
    assert result["matrix"] == {"include": []}
    assert result["run_unit_tests"] is True
    assert result["unit_scope"] == "all"


@pytest.mark.parametrize(
    "path",
    (
        ".pre-commit-config.yaml",
        "Dockerfile.community-cpu",
        "requirements/community-ci.txt",
        "tools/community_ci.py",
    ),
)
def test_community_cpu_contract_runs_all_units_without_model_fallback(
    tmp_path: Path, path: str,
) -> None:
    repo, base = _make_repo(tmp_path)
    _write(repo, path, "# community CPU contract\n")
    head = _commit(repo, "update community CPU contract")

    result = _impact(repo, base, head)

    assert result["mode"] == "unit"
    assert result["affected_models"] == []
    assert result["matrix"] == {"include": []}
    assert result["run_unit_tests"] is True
    assert result["unit_scope"] == "all"


@pytest.mark.parametrize(
    "path",
    (
        "benchmarks/performance/baselines/task_reference.py",
        "tools/perf_matrix.py",
    ),
)
def test_release_performance_runtime_changes_keep_model_fallback(
    tmp_path: Path,
    path: str,
) -> None:
    repo, base = _make_repo(tmp_path)
    _write(repo, path, "# changed release performance runtime\n")
    head = _commit(repo, "update release performance runtime")

    result = _impact(repo, base, head)

    assert result["mode"] == "fallback"
    assert result["affected_models"] == ["model_a"]
    assert result["direct_models"] == []
    assert result["fallback_models"] == ["model_a"]
    assert result["matrix"] == {
        "include": [{"model": "model_a", "selection_kind": "fallback"}]
    }
    assert result["run_unit_tests"] is True
    assert result["unit_scope"] == "all"


def test_impact_treats_platform_change_as_fixed_fallback(tmp_path: Path) -> None:
    repo, base = _make_repo(tmp_path)
    _write(repo, "src/runtime/core/core.cpp", "// changed platform core\n")
    head = _commit(repo, "platform")

    result = _impact(repo, base, head)

    assert result["mode"] == "fallback"
    assert result["affected_models"] == ["model_a"]
    assert result["direct_models"] == []
    assert result["fallback_models"] == ["model_a"]
    assert result["matrix"] == {"include": [{"model": "model_a", "selection_kind": "fallback"}]}
    assert result["run_unit_tests"] is True
    assert result["unit_scope"] == "all"


def test_tokenizer_platform_change_always_selects_known_consumers(tmp_path: Path) -> None:
    repo, base = _make_repo(tmp_path, model_ids=("flux", "model_a", "sam3"))
    _write(repo, "src/tokenizer/bpe_tokenizer.cpp", "// changed shared tokenizer\n")
    head = _commit(repo, "change shared tokenizer")

    result = _impact(repo, base, head, fallback_models=("model_a",))

    assert result["mode"] == "fallback"
    assert result["affected_models"] == ["flux", "model_a", "sam3"]
    assert result["direct_models"] == ["flux", "sam3"]
    assert result["fallback_models"] == ["model_a"]
    assert result["matrix"] == {
        "include": [
            {"model": "flux", "selection_kind": "direct"},
            {"model": "model_a", "selection_kind": "fallback"},
            {"model": "sam3", "selection_kind": "direct"},
        ]
    }
    tokenizer_classification = next(
        classification
        for change in result["changes"]
        for classification in change["classifications"]
        if classification["path"] == "src/tokenizer/bpe_tokenizer.cpp"
    )
    assert tokenizer_classification == {
        "path": "src/tokenizer/bpe_tokenizer.cpp",
        "kind": "platform",
        "consumer_models": ["flux", "sam3"],
    }


def test_impact_treats_shared_family_registry_as_platform(tmp_path: Path) -> None:
    repo, base = _make_repo(tmp_path)
    _write(repo, "python/tensorrt_model_connect/families/__init__.py", "# changed registry\n")
    head = _commit(repo, "shared family registry")

    result = _impact(repo, base, head)

    assert result["mode"] == "fallback"
    assert result["affected_models"] == ["model_a"]
    assert result["run_unit_tests"] is True
    assert result["unit_scope"] == "all"


@pytest.mark.parametrize(
    "path, expected_scope",
    (
        ("src/runtime/config/cli_support.cpp", "all"),
        ("include/trtmc/config/cli_support.h", "all"),
        ("python/tensorrt_model_connect/build_cli.py", "all"),
        ("python/tensorrt_model_connect/runtime_config/cli_support.py", "all"),
        ("tests/builder/test_cli.py", "all"),
        ("tests/cpp/test_cli_args.cpp", "all"),
        ("tests/tools/test_cli_contract.py", "all"),
        ("examples/models/cosmos3/dual_spark/run_dual_spark.py", "all"),
        ("examples/models/cosmos3/dual_spark/Dockerfile", "all"),
        ("examples/models/nemotron_voicechat/full_duplex/main.cpp", "all"),
        ("examples/models/nemotron_voicechat/full_duplex/Dockerfile", "all"),
        ("examples/models/nemotron_voicechat/full_duplex/CMakeLists.txt", "all"),
    ),
)
def test_cli_and_unit_test_changes_run_units_without_model_proofs(
    tmp_path: Path,
    path: str,
    expected_scope: str,
) -> None:
    repo, base = _make_repo(tmp_path)
    _write(repo, path, "// changed\n" if path.endswith((".cpp", ".h")) else "# changed\n")
    head = _commit(repo, "unit-only change")

    result = _impact(repo, base, head)

    assert result["mode"] == "unit"
    assert result["affected_models"] == []
    assert result["direct_models"] == []
    assert result["fallback_models"] == []
    assert result["matrix"] == {"include": []}
    assert result["run_unit_tests"] is True
    assert result["unit_scope"] == expected_scope


@pytest.mark.parametrize("path", ("src/cli/main.cpp", "src/cli/args.h"))
def test_runtime_cli_changes_run_model_fallback(tmp_path: Path, path: str) -> None:
    repo, base = _make_repo(tmp_path)
    _write(repo, path, "// changed\n")
    head = _commit(repo, "runtime CLI change")

    result = _impact(repo, base, head)

    assert result["mode"] == "fallback"
    assert result["affected_models"] == ["model_a"]
    assert result["run_unit_tests"] is True
    assert result["unit_scope"] == "all"


@pytest.mark.parametrize(
    "path",
    (
        "CMakeLists.txt",
        ".github/workflows/internal-ci-bridge.yml",
        "src/runtime/providers/optimized_runtime_host.cpp",
        "tools/model_ci.py",
        "new_platform/implementation.py",
    ),
)
def test_broad_or_unknown_changes_select_only_fixed_fallback(
    tmp_path: Path,
    path: str,
) -> None:
    repo, base = _make_repo(tmp_path)
    _write(repo, path, "# changed\n")
    head = _commit(repo, "broad change")

    result = _impact(repo, base, head)

    assert result["mode"] == "fallback"
    assert result["affected_models"] == ["model_a"]
    assert result["run_unit_tests"] is True
    assert result["unit_scope"] == "all"


def test_mixed_model_and_broad_change_keeps_direct_model_plus_fallback(
    tmp_path: Path,
) -> None:
    repo, base = _make_repo(tmp_path)
    _write(repo, "src/runtime/models/model_b/plugin.cpp", "// changed model b\n")
    _write(repo, "CMakeLists.txt", "# changed platform\n")
    head = _commit(repo, "mixed change")

    result = _impact(repo, base, head)

    assert result["mode"] == "fallback"
    assert result["affected_models"] == ["model_a", "model_b"]
    assert result["direct_models"] == ["model_b"]
    assert result["fallback_models"] == ["model_a"]
    assert result["matrix"] == {
        "include": [
            {"model": "model_a", "selection_kind": "fallback"},
            {"model": "model_b", "selection_kind": "direct"},
        ]
    }
    assert result["run_unit_tests"] is True
    assert result["unit_scope"] == "all"


def test_validation_only_pr_runs_units_without_model_proofs(tmp_path: Path) -> None:
    repo, base = _make_repo(tmp_path)
    _write(
        repo,
        "pyproject.toml",
        "[project]\n"
        'dependencies = ["runtime-package>=1"]\n\n'
        "[project.optional-dependencies]\n"
        'test = ["pytest>=7"]\n'
        'validation = ["rouge-score>=0.1.2", "sacrebleu>=2.4"]\n',
    )
    _write(
        repo,
        "tests/validation/workloads.yaml",
        "suites:\n  - id: elf_text_parity\n",
    )
    _write(repo, "tests/tools/test_validation_engine.py", "# expanded validation unit tests\n")
    _write(repo, "tests/tools/test_test_impact.py", "# validation impact tests\n")
    _write(repo, "tools/validation/engine.py", "# expanded validation runner\n")
    _write(repo, "tools/elf_hf_reference.py", "# isolated ELF reference\n")
    _write(
        repo,
        "tools/prepare_elf_validation_datasets.py",
        "# ELF validation dataset preparation\n",
    )
    _write(
        repo,
        "tools/prepare_media_validation_datasets.py",
        "# media validation dataset preparation\n",
    )
    _write(repo, "tools/test_impact.py", "# validation impact refinement\n")
    head = _commit(repo, "add validation coverage")

    result = _impact(repo, base, head)

    assert result["mode"] == "unit"
    assert result["affected_models"] == []
    assert result["direct_models"] == []
    assert result["fallback_models"] == []
    assert result["matrix"] == {"include": []}
    assert result["run_unit_tests"] is True
    assert result["unit_scope"] == "all"
    assert {
        classification["kind"]
        for change in result["changes"]
        for classification in change["classifications"]
    } == {"unit_tests"}


def test_validation_optional_extra_mixed_with_runtime_dependency_uses_fallback(
    tmp_path: Path,
) -> None:
    repo, base = _make_repo(tmp_path)
    _write(
        repo,
        "pyproject.toml",
        "[project]\n"
        'dependencies = ["runtime-package>=1", "new-runtime-package>=1"]\n\n'
        "[project.optional-dependencies]\n"
        'test = ["pytest>=7"]\n'
        'validation = ["rouge-score>=0.1.2"]\n',
    )
    head = _commit(repo, "change runtime and validation dependencies")

    result = _impact(repo, base, head)

    assert result["mode"] == "fallback"
    assert result["direct_models"] == []
    assert result["fallback_models"] == ["model_a"]


def test_validation_and_model_change_runs_direct_model_proof_plus_units(
    tmp_path: Path,
) -> None:
    repo, base = _make_repo(tmp_path)
    _write(repo, "tools/validation/engine.py", "# expanded validation runner\n")
    _write(repo, "src/runtime/models/model_b/plugin.cpp", "// changed model b\n")
    head = _commit(repo, "change validation and model b")

    result = _impact(repo, base, head)

    assert result["mode"] == "models"
    assert result["affected_models"] == ["model_b"]
    assert result["direct_models"] == ["model_b"]
    assert result["fallback_models"] == []
    assert result["matrix"] == {"include": [{"model": "model_b", "selection_kind": "direct"}]}
    assert result["run_unit_tests"] is True
    assert result["unit_scope"] == "all"


def test_impact_includes_deletions_and_both_sides_of_rename(tmp_path: Path) -> None:
    repo, base = _make_repo(tmp_path)
    (repo / "tests/cpp/models/model_a/test_model_a.cpp").unlink()
    _git(
        repo,
        "mv",
        "python/tensorrt_model_connect/families/model_a/plugin.py",
        "python/tensorrt_model_connect/families/model_b/from_a.py",
    )
    head = _commit(repo, "delete and rename")

    result = _impact(repo, base, head)

    assert result["affected_models"] == ["model_a", "model_b"]
    assert any(change["status"] == "D" for change in result["changes"])
    assert any(str(change["status"]).startswith("R") for change in result["changes"])


def test_whole_model_deletion_runs_units_and_fixed_fallback(tmp_path: Path) -> None:
    repo, base = _make_repo(tmp_path)
    _git(
        repo,
        "rm",
        "-r",
        "python/tensorrt_model_connect/families/model_b",
        "src/runtime/models/model_b",
        "tests/e2e/models/model_b",
        "tests/cpp/models/model_b",
    )
    head = _commit(repo, "delete model b")

    result = _impact(repo, base, head)

    assert result["mode"] == "fallback"
    assert result["affected_models"] == ["model_a"]
    assert result["run_unit_tests"] is True
    assert result["unit_scope"] == "all"


def test_deleting_configured_fallback_requires_policy_update(tmp_path: Path) -> None:
    repo, base = _make_repo(tmp_path)
    _git(
        repo,
        "rm",
        "-r",
        "python/tensorrt_model_connect/families/model_a",
        "src/runtime/models/model_a",
        "tests/e2e/models/model_a",
        "tests/cpp/models/model_a",
    )
    head = _commit(repo, "delete fallback model")

    result = _run(
        repo,
        "impact",
        "--base",
        base,
        "--head",
        head,
        "--platform-change-policy",
        "fallback",
        "--fallback-model",
        "model_a",
        check=False,
    )

    assert result.returncode == 2
    assert "fallback models are absent from the head catalog" in result.stderr


def test_impact_rejects_unowned_path_below_model_root(tmp_path: Path) -> None:
    repo, base = _make_repo(tmp_path)
    _write(repo, "tests/e2e/models/unowned/test_unowned.py", "VALUE = 1\n")
    head = _commit(repo, "unowned model source")

    result = _run(
        repo,
        "impact",
        "--base",
        base,
        "--head",
        head,
        "--platform-change-policy",
        "fallback",
        "--fallback-model",
        "model_a",
        check=False,
    )

    assert result.returncode == 2
    assert "under a model root but has no MODEL.toml owner" in result.stderr


@pytest.mark.parametrize(
    "path",
    (
        "tests/builder/test_dynamic_batch_profile.py",
        "tests/builder/test_flashinfer_benchmark.py",
        "tests/builder/test_graph_blocks.py",
        "tests/builder/test_tvm_ffi_plugin.py",
        "tests/cpp/test_cuda_buffer.cpp",
        "tests/cpp/test_cuda_graph.cpp",
        "tests/cpp/test_cuda_stream.cpp",
        "tests/cpp/test_device_tensor.cpp",
        "tests/cpp/test_model_plugin_loader.cpp",
        "tests/cpp/test_trt_module.cpp",
        "tests/cpp/test_trt_runtime_lifetime.cpp",
        "tests/cpp/test_tvm_ffi_module_loader.cpp",
        "tests/cpp/test_tvm_ffi_plugin.cpp",
        "tests/cpp/test_tvm_ffi_plugin_v2.cpp",
    ),
)
def test_model_coupled_shared_tests_fail_closed_until_owned(
    tmp_path: Path,
    path: str,
) -> None:
    repo, base = _make_repo(tmp_path)
    _write(repo, path, "// changed coupled test\n")
    head = _commit(repo, "change coupled test")

    result = _run(
        repo,
        "impact",
        "--base",
        base,
        "--head",
        head,
        "--platform-change-policy",
        "fallback",
        "--fallback-model",
        "model_a",
        check=False,
    )

    assert result.returncode == 2
    assert "model-coupled test has no isolated model owner" in result.stderr


@pytest.mark.parametrize(
    "fallback_args, message",
    (
        ((), "requires at least one --fallback-model"),
        (("model_a", "model_a"), "contains duplicates"),
        (("missing",), "absent from the head catalog"),
        (("../unsafe",), "contains unsafe ids"),
    ),
)
def test_broad_impact_rejects_invalid_fallback_configuration(
    tmp_path: Path,
    fallback_args: tuple[str, ...],
    message: str,
) -> None:
    repo, base = _make_repo(tmp_path)
    _write(repo, "CMakeLists.txt", "# changed platform\n")
    head = _commit(repo, "broad change")
    args = [
        "impact",
        "--base",
        base,
        "--head",
        head,
        "--platform-change-policy",
        "fallback",
    ]
    for model in fallback_args:
        args.extend(("--fallback-model", model))

    result = _run(repo, *args, check=False)

    assert result.returncode == 2
    assert message in result.stderr


def test_validate_rejects_overlapping_runtime_ownership(tmp_path: Path) -> None:
    repo, _ = _make_repo(tmp_path)
    manifest = repo / "tests/e2e/models/model_b/manifests/model_b.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["runtime_strategy"] = "model_a_runtime"
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    _commit(repo, "introduce overlap")

    result = _run(repo, "validate", check=False)

    assert result.returncode == 2
    assert "depends on multiple runtime models" in result.stderr


def test_projection_contains_only_selected_model_and_stable_git_blobs(
    tmp_path: Path,
) -> None:
    repo, revision = _make_repo(tmp_path)
    source = repo / "python/tensorrt_model_connect/families/model_a/plugin.py"
    expected = source.read_bytes()
    source.write_text('MODEL = "dirty_worktree_value"\n', encoding="utf-8")
    output = tmp_path / "projection"

    manifest = json.loads(
        _run(
            repo,
            "project",
            "--revision",
            revision,
            "--model",
            "model_a",
            "--output-dir",
            str(output),
        ).stdout
    )

    copied = output / "python/tensorrt_model_connect/families/model_a/plugin.py"
    assert copied.read_bytes() == expected
    assert not (output / "python/tensorrt_model_connect/families/model_b").exists()
    assert not (output / "src/runtime/models/model_b").exists()
    assert not (output / "tests/e2e/models/model_b").exists()
    assert not (output / "tests/cpp/models/model_b").exists()
    assert (output / "tests/cpp/models/model_a/test_model_a.cpp").is_file()
    assert (output / "src/runtime/core/core.cpp").is_file()
    assert (output / "examples/byok/identity_copy_kernel.cpp").is_file()
    assert (output / "python/tensorrt_model_connect/families/__init__.py").is_file()
    assert (output / "tests/__init__.py").is_file()
    assert (output / "tests/builder/__init__.py").is_file()
    for support_path in (
        "tests/builder/conftest.py",
        "tests/builder/debug_runner_test_support.py",
        "tests/builder/family_plugin_test_mixin.py",
        "tests/builder/family_plugin_test_support.py",
        "tests/builder/family_plugin_tester.py",
    ):
        assert (output / support_path).is_file()
    for unrelated_path in (
        "tests/builder/test_checkpoint_mapper.py",
        "tests/builder/test_debug_runner.py",
    ):
        assert not (output / unrelated_path).exists()
    assert (output / "tests/runtime_strategy_matrix.yaml").is_file()
    assert (output / "tools/ci/__main__.py").is_file()
    assert (output / "tools/ci/model_proof.py").is_file()
    fallback = output / ".github/scripts/write-model-proof-fallback-report.py"
    assert fallback.is_file()
    assert not os.access(fallback, os.X_OK)
    for report_path in (
        "scripts/generate_e2e_report.py",
        "scripts/generate_e2e_report_assets/e2e_report.css",
        "scripts/generate_e2e_report_assets/e2e_report.js",
        "scripts/reporting/__init__.py",
        "scripts/reporting/vlm_assessment.py",
        "scripts/schedule_e2e.py",
        "scripts/hf_cache_download_worker.py",
        "scripts/warm_hf_cache.py",
        "tools/__init__.py",
        "tools/diff_logits.py",
        "tools/diff_vl.py",
        "tools/diffusion_helpers.py",
        "tools/elf_hf_reference.py",
        "tools/model_plugin_isolation.py",
        "tools/reference/elf_prepared.py",
        "tools/reference/plugin_reference.py",
        "tools/reference/speech.py",
        "tools/reference/transformers_encoder.py",
        "tools/reference/transformers_text.py",
        "tools/reference/transformers_vlm.py",
        "tools/test_impact.py",
        "tools/tool_helpers.py",
        "tools/trtmc_reference.py",
        "tools/validation/__init__.py",
        "tools/validation/artifacts.py",
        "tools/validation/catalog.py",
        "tools/validation/engine.py",
        "tools/validation/gate_policy.py",
        "tools/validation/model_plugin_contract.py",
        "tools/ci/__init__.py",
        "tools/ci/__main__.py",
        "tools/ci/model_proof.py",
    ):
        assert (output / report_path).is_file()
    assert (output / "tests/validation/workloads.yaml").is_file()
    for unrelated_tool in (
        "scripts/repro_trt_fp8_mha.py",
        "tools/diff_t5.py",
        "tools/validate_t5.py",
    ):
        assert not (output / unrelated_tool).exists()
    assert manifest["runtime_model"] == "model_a"
    assert manifest["build_target"] == "trtmc_model_model_a"
    entry = next(
        item for item in manifest["files"] if item["path"] == copied.relative_to(output).as_posix()
    )
    assert entry["sha256"] == hashlib.sha256(expected).hexdigest()


def test_projection_closes_validation_module_imports() -> None:
    projected = set(model_ci.PLATFORM_PROJECTION_EXACT)
    dependencies = {"tools/validation/__init__.py"}

    for relative in projected:
        source = REPO_ROOT / relative
        if source.suffix != ".py" or not source.is_file():
            continue
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if node.module == "tools.validation":
                for imported in node.names:
                    candidate = f"tools/validation/{imported.name}.py"
                    if (REPO_ROOT / candidate).is_file():
                        dependencies.add(candidate)
            elif node.module.startswith("tools.validation."):
                dependencies.add(node.module.replace(".", "/") + ".py")

    assert dependencies <= projected


def test_projection_includes_only_the_selected_family_adapter_subtrees(
    tmp_path: Path,
) -> None:
    repo, _ = _make_repo(tmp_path)
    selected_paths = (
        "python/tensorrt_model_connect/families/model_b/optimized_adapter/adapter.py",
        "python/tensorrt_model_connect/families/model_b/optimized_adapter/dependency.lock",
        "python/tensorrt_model_connect/families/model_b/optimized_adapter/profiles/example.toml",
        "src/runtime/models/model_b/optimized_adapter/adapter.cpp",
        "tests/e2e/models/model_b/optimized_adapter/test_contract.py",
    )
    sibling_paths = tuple(path.replace("model_b", "model_a") for path in selected_paths)
    for path in selected_paths:
        _write(repo, path, "# selected model adapter\n")
    for path in sibling_paths:
        _write(repo, path, "# sibling model adapter\n")
    generic_host = "src/runtime/providers/optimized_runtime_host.cpp"
    _write(repo, generic_host, "// shared provider host\n")
    revision = _commit(repo, "add model-owned adapters")
    output = tmp_path / "projection-model-b"

    manifest = json.loads(
        _run(
            repo,
            "project",
            "--revision",
            revision,
            "--model",
            "model_b",
            "--output-dir",
            str(output),
        ).stdout
    )

    for path in selected_paths:
        assert (output / path).is_file()
    for path in sibling_paths:
        assert not (output / path).exists()
    assert (output / generic_host).is_file()
    assert "optimized_runtime_roots" not in manifest
    assert "optimized_runtime_files" not in manifest


def test_affected_model_projections_include_only_shared_support_and_owned_roots(
    tmp_path: Path,
) -> None:
    repo, revision = _make_repo(
        tmp_path,
        model_ids=("ltx_video", "qwen3_5", "dpr"),
    )
    cases = (
        ("ltx_video", "tools/diffusion_helpers.py"),
        ("qwen3_5", "scripts/schedule_e2e.py"),
        ("dpr", "tools/test_impact.py"),
    )

    for selected, required_shared_file in cases:
        output = tmp_path / f"projection-{selected}"
        _run(
            repo,
            "project",
            "--revision",
            revision,
            "--model",
            selected,
            "--output-dir",
            str(output),
        )

        assert (output / required_shared_file).is_file()
        for model_root in (
            "python/tensorrt_model_connect/families",
            "src/runtime/models",
            "tests/e2e/models",
            "tests/cpp/models",
        ):
            assert (output / model_root / selected).is_dir()
            for sibling in {model_id for model_id, _ in cases} - {selected}:
                assert not (output / model_root / sibling).exists()


def test_projection_and_impact_normalize_logical_runtime_owner(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Model CI Test")
    _git(repo, "config", "user.email", "model-ci@example.com")
    _add_model(repo, "logical_model", runtime_id="runtime_model", strategy="runtime_strategy")
    _write(repo, "CMakeLists.txt", "# platform\n")
    base = _commit(repo, "initial")
    _write(repo, "src/runtime/models/runtime_model/plugin.cpp", "// changed runtime\n")
    head = _commit(repo, "runtime change")

    impact = _impact(repo, base, head)
    output = tmp_path / "projection"
    projection = json.loads(
        _run(
            repo,
            "project",
            "--revision",
            head,
            "--model",
            "logical_model",
            "--output-dir",
            str(output),
        ).stdout
    )

    assert impact["affected_models"] == ["logical_model"]
    assert projection["runtime_model"] == "runtime_model"
    assert projection["build_target"] == "trtmc_model_runtime_model"
    assert (output / "src/runtime/models/runtime_model/plugin.cpp").is_file()


def test_projection_rejects_symlink_that_escapes_allowlist(tmp_path: Path) -> None:
    repo, revision = _make_repo(tmp_path)
    link = repo / "python/tensorrt_model_connect/families/model_a/escape"
    link.symlink_to("/etc/passwd")
    revision = _commit(repo, "escaping symlink")

    result = _run(
        repo,
        "project",
        "--revision",
        revision,
        "--model",
        "model_a",
        "--output-dir",
        str(tmp_path / "projection"),
        check=False,
    )

    assert result.returncode == 2
    assert "symlink escapes projection" in result.stderr
