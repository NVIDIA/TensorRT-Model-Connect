# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the hermetic single-model proof runner."""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / ".github" / "scripts" / "run-model-proof.sh"
IMAGE_ENSURE = REPO_ROOT / ".github" / "scripts" / "ensure-ci-docker-image.sh"
PROOF_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "model-proof.yml"
FALLBACK_WRITER = REPO_ROOT / ".github" / "scripts" / "write-model-proof-fallback-report.py"
PLUGIN_CMAKE = REPO_ROOT / "cmake" / "trtmc_pipeline_plugins.cmake"


def _write_successful_fake_docker(tmp_path: Path) -> tuple[Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$*\" >> \"$DOCKER_LOG\"\n"
        "case \"${1:-}\" in\n"
        "  image|rm) exit 0 ;;\n"
        "  run)\n"
        "    if [[ \" $* \" == *\" /src/scripts/warm_hf_cache.py \"* ]]; then\n"
        "      exit 0\n"
        "    fi\n"
        "    if [[ \" $* \" == *\" --inner \"* ]]; then\n"
        "      sleep \"${FAKE_PROOF_DELAY_SECONDS:-0}\"\n"
        "      mkdir -p \"$FAKE_ARTIFACTS\"\n"
        "      printf '{}\\n' > \"$FAKE_ARTIFACTS/proof.json\"\n"
        "      printf '<html></html>\\n' > \"$FAKE_ARTIFACTS/model-proof-report.html\"\n"
        "      exit 0\n"
        "    fi\n"
        "    ;;\n"
        "esac\n"
        "exit 99\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    return fake_bin, docker_log


def _fake_proof_environment(
    tmp_path: Path,
    fake_bin: Path,
    docker_log: Path,
    output: Path,
) -> dict[str, str]:
    (tmp_path / "hf-cache" / "hub").mkdir(parents=True, exist_ok=True)
    (tmp_path / "hf-cache" / "modules").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("TRTMC_GPU_ID", None)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "DOCKER_LOG": str(docker_log),
            "FAKE_ARTIFACTS": str(output / "artifacts"),
            "TRTMC_HF_CACHE": str(tmp_path / "hf-cache"),
            "TRTMC_MODEL_PROOF_GPU_LOCK_DIR": str(tmp_path / "gpu-locks"),
            "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": "5",
        }
    )
    return env


def _proof_gpu_ids(docker_log: Path) -> list[str]:
    proof_runs = [
        line
        for line in docker_log.read_text(encoding="utf-8").splitlines()
        if " --inner " in f" {line} "
    ]
    assert len(proof_runs) == 1, proof_runs
    matches = re.findall(r"--gpus device=([0-9]+)", proof_runs[0])
    assert len(matches) == 1, proof_runs[0]
    return matches


def _selection_program() -> str:
    text = RUNNER.read_text(encoding="utf-8")
    marker = '> "$config_file" <<\'PY\'\n'
    return text.split(marker, maxsplit=1)[1].split("\nPY\n", maxsplit=1)[0]


def _run_test_selection(tmp_path: Path, family: str, suite: str) -> dict:
    source = tmp_path / f"{family}-{suite}"
    e2e_source = REPO_ROOT / "tests" / "e2e" / "models" / family
    e2e_target = source / "tests" / "e2e" / "models" / family
    shutil.copytree(e2e_source, e2e_target)
    shutil.copy2(
        REPO_ROOT / "tests" / "e2e" / "timing_estimates.json",
        source / "tests" / "e2e" / "timing_estimates.json",
    )

    runtime = source / "src" / "runtime" / "models" / "fixture_runtime"
    runtime.mkdir(parents=True)
    (runtime / "MODEL.toml").write_text(
        'id = "fixture_runtime"\n'
        'runtime_library = "libtrtmc_model_fixture_runtime.so"\n',
        encoding="utf-8",
    )
    (source / "python" / "tensorrt_model_connect" / "families").mkdir(parents=True)
    revision = "a" * 40
    (source / ".trtmc-model-projection.json").write_text(
        json.dumps(
            {
                "revision": revision,
                "model": family,
                "runtime_model": "fixture_runtime",
                "e2e_family": family,
            }
        ),
        encoding="utf-8",
    )
    selection_path = tmp_path / f"{family}-{suite}-selection.json"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _selection_program(),
            family,
            suite,
            revision,
            str(source),
            str(selection_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(selection_path.read_text(encoding="utf-8"))


def _add_runtime_model(source: Path, model: str) -> None:
    model_dir = source / "src" / "runtime" / "models" / model
    model_dir.mkdir(parents=True)
    (model_dir / "plugin.cpp").write_text("// fixture\n", encoding="utf-8")
    (model_dir / "MODEL.toml").write_text(
        f'id = "{model}"\n'
        f'runtime_library = "libtrtmc_model_{model}.so"\n'
        f'runtime_plugins = ["plugin.cpp|register_{model}"]\n'
        f'runtime_strategies = ["{model}_strategy"]\n',
        encoding="utf-8",
    )


def _configure(source: Path, requested_model: str) -> subprocess.CompletedProcess[str]:
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(model_proof_contract NONE)\n"
        f'include("{PLUGIN_CMAKE}")\n',
        encoding="utf-8",
    )
    return subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(source / "build"),
            f"-DTRTMC_MODEL_PROOF_MODEL={requested_model}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cmake_model_proof_accepts_one_matching_runtime_model(tmp_path: Path) -> None:
    _add_runtime_model(tmp_path, "alpha")

    result = _configure(tmp_path, "alpha")

    assert result.returncode == 0, result.stdout + result.stderr


def test_cmake_model_proof_rejects_a_sibling_manifest(tmp_path: Path) -> None:
    _add_runtime_model(tmp_path, "alpha")
    _add_runtime_model(tmp_path, "beta")

    result = _configure(tmp_path, "alpha")

    assert result.returncode != 0
    assert "requires exactly one runtime model manifest" in result.stdout + result.stderr


def test_cmake_model_proof_rejects_the_wrong_runtime_model(tmp_path: Path) -> None:
    _add_runtime_model(tmp_path, "beta")

    result = _configure(tmp_path, "alpha")

    assert result.returncode != 0
    output = " ".join((result.stdout + result.stderr).split())
    assert "projected source contains runtime model 'beta'" in output


def test_runner_rejects_an_unknown_suite_before_starting_docker() -> None:
    result = subprocess.run(
        ["bash", str(RUNNER), "--model", "alpha", "--suite", "everything"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--suite must be premerge or nightly" in result.stderr


@pytest.mark.parametrize(
    ("family", "expected_case"),
    (
        ("flux", "flux-schnell-l0"),
        ("personaplex", "personaplex-7b-l0"),
        ("canary", "canary-1b-v2"),
        ("nemotron_labs_diffusion", "nemotron-labs-diffusion-8b-l0"),
        ("qwen_image", "qwen-image-l0"),
    ),
)
def test_premerge_selects_one_nested_l0_replacement(
    tmp_path: Path,
    family: str,
    expected_case: str,
) -> None:
    selection = _run_test_selection(tmp_path, family, "premerge")

    assert selection["suite"] == "premerge"
    assert [case["name"] for case in selection["e2e_cases"]] == [expected_case]
    assert selection["e2e_cases"][0]["ci_tier"] != "nightly_only"
    assert selection["e2e_cases"][0]["model"] == expected_case


def test_every_owned_e2e_family_has_one_premerge_case(tmp_path: Path) -> None:
    model_root = REPO_ROOT / "tests" / "e2e" / "models"
    families = sorted(
        path.parent.name for path in model_root.glob("*/MODEL.toml")
    )

    assert families
    for family in families:
        selection = _run_test_selection(tmp_path, family, "premerge")
        assert len(selection["e2e_cases"]) == 1, family
        assert selection["e2e_cases"][0]["ci_tier"] != "nightly_only", family


@pytest.mark.parametrize(
    ("family", "expected_cases"),
    (
        (
            "flux",
            {
                "flux-2-dev-fp8-l0",
                "flux-2-dev-fp8",
                "flux-2-dev-l0",
                "flux-2-dev",
                "flux-schnell-l0-batch2",
                "flux-schnell-l0",
                "flux-schnell",
            },
        ),
        ("personaplex", {"personaplex-7b-l0", "personaplex-7b"}),
        (
            "canary",
            {
                "canary-1b-v2",
                "canary-1b-v2-asr-probe01",
                "canary-1b-v2-asr-probe02",
                "canary-1b-v2-asr-probe03",
                "canary-1b-v2-asr-probe04",
                "canary-1b-v2-asr-probe05",
                "canary-1b-v2-asr-probe06",
                "canary-1b-v2-asr-probe08",
            },
        ),
    ),
)
def test_nightly_selects_the_full_nested_single_gpu_suite(
    tmp_path: Path,
    family: str,
    expected_cases: set[str],
) -> None:
    selection = _run_test_selection(tmp_path, family, "nightly")

    assert selection["suite"] == "nightly"
    assert {case["name"] for case in selection["e2e_cases"]} == expected_cases
    assert any(case["ci_tier"] == "nightly_only" for case in selection["e2e_cases"])
    assert all(case["ci_tier"] != "multi_device" for case in selection["e2e_cases"])


def test_runner_declares_the_hermetic_container_boundary() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    warm = text.split("local -a cache_check_docker_args=(", maxsplit=1)[1].split(
        "local -a docker_args=(", maxsplit=1
    )[0]
    proof = text.split("local -a docker_args=(", maxsplit=1)[1].split(
        "set +e", maxsplit=1
    )[0]

    for contract in (
        "--read-only",
        "--network none",
        "--cap-drop ALL",
        "dst=/src,readonly",
        "TORCHINDUCTOR_CACHE_DIR=/work/torch-cache",
        "TRTMC_MODEL_PLUGIN_STRICT=1",
        "scratch build produced ${#built_dsos[@]} model DSOs; expected exactly one",
    ):
        assert contract in text
    assert "--network none" in warm
    assert "dst=/hf-cache/hub,readonly" in warm
    assert "dst=/hf-cache/modules,readonly" in warm
    assert "-e HF_TOKEN" not in warm
    assert "-e HUGGING_FACE_HUB_TOKEN" not in warm
    assert "--network none" in proof
    assert "dst=/hf-cache/hub,readonly" in proof
    assert "dst=/hf-cache/modules,readonly" in proof
    assert "-e HF_TOKEN" not in proof
    assert "-e HUGGING_FACE_HUB_TOKEN" not in proof


def test_runner_warms_the_exact_shared_selection_before_the_proof() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    host = text.split("run_host() {", maxsplit=1)[1]
    warm = host.split("local -a cache_check_docker_args=(", maxsplit=1)[1].split(
        "local -a docker_args=(", maxsplit=1
    )[0]

    assert host.count("write_model_proof_selection") == 1
    assert 'sed -n \'s/^e2e_model=//p\' "$host_config_file"' in host
    assert "cache-check-models.txt" in warm
    assert "scripts/warm_hf_cache.py" in warm
    assert "--models-file /artifacts/cache-check-models.txt --local-only --strict" in warm
    assert host.index("scripts/warm_hf_cache.py") < host.index("local -a docker_args=(")
    assert 'die "offline HF cache readiness check failed' in warm


def test_runner_keeps_local_fallback_and_workflow_uses_runner_cache_paths() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    workflow = PROOF_WORKFLOW.read_text(encoding="utf-8")

    assert '${HF_HOME:-$HOME/.cache/huggingface}' in runner
    assert (
        "TRTMC_HF_HUB_CACHE: ${{ vars.TRTMC_HF_HUB_CACHE || "
        "'/workspace/users/yifeif/tensorrt-model-connect/hf-cache/hub' }}"
    ) in workflow
    assert (
        "TRTMC_HF_MODULES_CACHE: ${{ vars.TRTMC_HF_MODULES_CACHE || "
        "'/workspace/users/yifeif/tensorrt-model-connect/hf-cache/modules' }}"
    ) in workflow


def test_hf_token_is_not_exposed_to_pull_request_model_proof_code() -> None:
    workflow = PROOF_WORKFLOW.read_text(encoding="utf-8")
    job_environment = workflow.split("\n    steps:", maxsplit=1)[0]
    proof_step = workflow.split("- name: Run affected model proofs", maxsplit=1)[1].split(
        "- name: Finalize batch proof fallbacks", maxsplit=1
    )[0]

    assert "HF_TOKEN:" not in job_environment
    assert "HUGGING_FACE_HUB_TOKEN:" not in job_environment
    assert "HF_TOKEN:" not in proof_step
    assert "HUGGING_FACE_HUB_TOKEN:" not in proof_step


def test_runner_removes_only_its_container_without_masking_exit_status() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    cleanup = text.split("cleanup_proof_container() {", maxsplit=1)[1].split("\n}\n", maxsplit=1)[0]

    assert 'local rc="$1"' in cleanup
    assert 'docker rm -f "$proof_container_name"' in cleanup
    assert 'exit "$rc"' in cleanup
    assert "artifacts" not in cleanup
    assert 'proof_container_name="$container_name"' in text
    assert "trap 'cleanup_proof_container \"$?\"' EXIT" in text
    assert "trap 'cleanup_proof_container 130' INT" in text
    assert "trap 'cleanup_proof_container 143' TERM" in text


def test_model_proof_serializes_image_setup_and_uses_the_verified_image_id() -> None:
    ensure = IMAGE_ENSURE.read_text(encoding="utf-8")
    workflow = PROOF_WORKFLOW.read_text(encoding="utf-8")

    for contract in (
        "TRTMC_CI_IMAGE_LOCK_FILE",
        'flock -w "$lock_timeout" 9',
        "docker image inspect --format '{{.Id}}'",
        'echo "image_ref=$image_ref" >> "$GITHUB_OUTPUT"',
    ):
        assert contract in ensure
    assert "id: ci_image" in workflow
    assert 'timeout-minutes: 90' in workflow
    assert "TRTMC_CI_IMAGE: ${{ steps.ci_image.outputs.image_ref }}" in workflow


def test_model_proof_uses_a_dedicated_self_hosted_checkout() -> None:
    workflow = PROOF_WORKFLOW.read_text(encoding="utf-8")
    checkout = workflow.split("- name: Check out exact source revision once", maxsplit=1)[
        1
    ].split(
        "- name: Ensure CI Docker image once", maxsplit=1
    )[0]

    assert "path: model-proof-source" in checkout
    assert "clean: true" in checkout
    assert "persist-credentials: false" in checkout
    assert workflow.count("working-directory: ${{ github.workspace }}/model-proof-source") == 2


def test_model_proof_bootstraps_html_without_a_checkout_dependency() -> None:
    workflow = PROOF_WORKFLOW.read_text(encoding="utf-8")
    bootstrap = workflow.split(
        "- name: Bootstrap batch HTML before checkout", maxsplit=1
    )[1].split("- name: Check out exact source revision once", maxsplit=1)[0]
    checkout_failure = workflow.split(
        "- name: Finalize batch proof fallbacks", maxsplit=1
    )[1].split("- name: Upload batch proof evidence", maxsplit=1)[0]

    assert workflow.index("Bootstrap batch HTML before checkout") < workflow.index(
        "Check out exact source revision once"
    )
    assert "model-proof-report.html" in bootstrap
    assert "model-proof-status.json" in bootstrap
    assert "model-proof-index.html" in bootstrap
    assert "working-directory:" not in bootstrap
    assert ".github/scripts/" not in bootstrap
    assert "if: always()" in checkout_failure
    assert "CHECKOUT_OUTCOME: ${{ steps.checkout.outcome }}" in checkout_failure
    assert "model-proof-report.html" in checkout_failure
    assert "working-directory:" not in checkout_failure
    assert ".github/scripts/" not in checkout_failure


def test_model_proof_resolves_runner_temp_only_after_runner_assignment() -> None:
    workflow = PROOF_WORKFLOW.read_text(encoding="utf-8")
    job_configuration = workflow.split("\n    steps:", maxsplit=1)[0]
    bootstrap = workflow.split(
        "- name: Bootstrap batch HTML before checkout", maxsplit=1
    )[1].split("- name: Check out exact source revision once", maxsplit=1)[0]
    proof = workflow.split(
        "- name: Run affected model proofs", maxsplit=1
    )[1].split("- name: Finalize batch proof fallbacks", maxsplit=1)[0]

    assert "MODEL_PROOF_BATCH_OUTPUT_DIR:" not in job_configuration
    assert "MODEL_PROOF_BATCH_OUTPUT_DIR: ${{ runner.temp }}" in bootstrap
    assert (
        'echo "MODEL_PROOF_BATCH_OUTPUT_DIR=$MODEL_PROOF_BATCH_OUTPUT_DIR" >> "$GITHUB_ENV"'
        in bootstrap
    )
    assert "${{ env.MODEL_PROOF_BATCH_OUTPUT_DIR }}" not in proof
    assert '--output-dir "$MODEL_PROOF_BATCH_OUTPUT_DIR"' in proof


def test_model_proof_checks_disk_headroom_before_checkout() -> None:
    workflow = PROOF_WORKFLOW.read_text(encoding="utf-8")
    disk_check = workflow.split(
        "- name: Check model proof disk headroom", maxsplit=1
    )[1].split("- name: Check out exact source revision once", maxsplit=1)[0]

    assert workflow.index("Check model proof disk headroom") < workflow.index(
        "Check out exact source revision once"
    )
    assert "TRTMC_MODEL_PROOF_MIN_FREE_GIB:" in workflow
    assert "TRTMC_MODEL_PROOF_STALE_MINUTES:" in workflow
    assert "-mmin \"+$TRTMC_MODEL_PROOF_STALE_MINUTES\"" in disk_check
    assert "-name work -o -name projection" in disk_check
    assert "-exec rm -rf -- {} +" in disk_check
    assert "active_workers=" in disk_check
    assert "min(len(models), len(gpu_ids))" in disk_check
    assert "required_gib=\"$((TRTMC_MODEL_PROOF_MIN_FREE_GIB * active_workers))\"" in disk_check
    assert 'df -Pk "$RUNNER_TEMP"' in disk_check
    assert "Insufficient model-proof disk headroom" in disk_check


def test_model_proof_cleans_scratch_only_after_both_artifact_uploads() -> None:
    workflow = PROOF_WORKFLOW.read_text(encoding="utf-8")
    cleanup = workflow.split(
        "- name: Clean batch proof scratch space", maxsplit=1
    )[1]

    assert workflow.index("Upload batch proof evidence") < workflow.index(
        "Clean batch proof scratch space"
    )
    assert workflow.index("Upload batch model proof HTML reports") < workflow.index(
        "Clean batch proof scratch space"
    )
    assert "id: proof_upload" in workflow
    assert "id: html_upload" in workflow
    assert '"$RUNNER_TEMP"/model-proof-batch-*' in cleanup
    assert "-name work -o -name projection" in cleanup
    assert 'PROOF_UPLOAD_OUTCOME" = "success"' in cleanup
    assert 'HTML_UPLOAD_OUTCOME" = "success"' in cleanup
    assert 'rm -rf -- "$MODEL_PROOF_BATCH_OUTPUT_DIR"' in cleanup


def test_model_proof_always_generates_a_strict_self_contained_html_report() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    workflow = PROOF_WORKFLOW.read_text(encoding="utf-8")

    for contract in (
        "trap 'finalize_model_report \"$?\"' EXIT",
        "/src/scripts/generate_e2e_report.py",
        "--artifacts-dir /artifacts/e2e",
        "--output /artifacts/model-proof-report.html",
        "--project-dir /src",
        "--proof-status /artifacts/model-proof-status.json",
        "--proof-json /artifacts/proof.json",
        "--selection-json /artifacts/selection.json",
        "--strict-evidence",
        "--max-embed-bytes 33554432",
        "--junitxml=/artifacts/e2e/junit.xml",
        "generate_host_fallback_report",
        'proof_artifacts_dir="$artifacts_dir"',
        'die "model proof did not emit model-proof-report.html"',
    ):
        assert contract in runner

    assert 'if [ "$validation_rc" -eq 0 ] && [ "$report_rc" -ne 0 ]; then' in runner
    assert 'exit "$validation_rc"' in runner
    assert 'payload["validation_exit_code"] = rc' in runner
    assert 'payload["report_exit_code"] = report_rc' in runner
    assert "Upload batch model proof HTML reports" in workflow
    assert "Bootstrap batch HTML before checkout" in workflow
    assert "Finalize batch proof fallbacks" in workflow
    assert "ci-image.log" in workflow
    assert "/*/artifacts/model-proof-report.html" in workflow
    assert "/model-proof-index.html" in workflow
    assert "if-no-files-found: error" in workflow


def test_model_proof_report_assets_are_inside_the_positive_projection() -> None:
    model_ci = (REPO_ROOT / "tools" / "model_ci.py").read_text(encoding="utf-8")

    assert '"scripts/",' in model_ci
    for path in (
        REPO_ROOT / "scripts" / "generate_e2e_report.py",
        REPO_ROOT / "scripts" / "generate_e2e_report_assets" / "e2e_report.css",
        REPO_ROOT / "scripts" / "generate_e2e_report_assets" / "e2e_report.js",
        REPO_ROOT / "scripts" / "reporting" / "vlm_assessment.py",
    ):
        assert path.is_file(), path


def test_fallback_writer_embeds_host_diagnostics(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "host-error.log").write_text(
        "model-ci: error: unknown model <unsafe>\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(FALLBACK_WRITER),
            "--artifacts-dir", str(artifacts),
            "--model", "missing-model",
            "--revision", "a" * 40,
            "--suite", "premerge",
            "--outcome", "failed",
            "--phase", "host-setup",
            "--exit-code", "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = (artifacts / "model-proof-report.html").read_text(encoding="utf-8")
    status = json.loads(
        (artifacts / "model-proof-status.json").read_text(encoding="utf-8")
    )
    assert "host-error.log" in report
    assert "unknown model &lt;unsafe&gt;" in report
    assert status["outcome"] == "failed"
    assert status["exit_code"] == 2


def test_host_projection_failure_preserves_error_and_html(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text("#!/usr/bin/env bash\nexit 99\n", encoding="utf-8")
    docker.chmod(0o755)
    output = tmp_path / "proof"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash", str(RUNNER),
            "--model", "model-that-does-not-exist",
            "--revision", "HEAD",
            "--output-dir", str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    artifacts = output / "artifacts"
    report = (artifacts / "model-proof-report.html").read_text(encoding="utf-8")
    status = json.loads(
        (artifacts / "model-proof-status.json").read_text(encoding="utf-8")
    )
    assert "projection.stderr.log" in report
    assert "unknown model" in report
    assert status["outcome"] == "failed"
    assert status["exit_code"] == result.returncode


def test_strict_cache_warm_failure_stops_before_hermetic_proof(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$DOCKER_LOG\"\n"
        "if [ \"${1:-}\" = image ] || [ \"${1:-}\" = rm ]; then exit 0; fi\n"
        "if [ \"${1:-}\" = run ]; then exit 23; fi\n"
        "exit 99\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    output = tmp_path / "proof"
    (tmp_path / "hf-cache" / "hub").mkdir(parents=True)
    (tmp_path / "hf-cache" / "modules").mkdir(parents=True)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "DOCKER_LOG": str(docker_log),
            "TRTMC_HF_CACHE": str(tmp_path / "hf-cache"),
        }
    )

    result = subprocess.run(
        [
            "bash", str(RUNNER),
            "--model", "convbert",
            "--revision", "HEAD",
            "--output-dir", str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "offline HF cache readiness check failed for convbert" in result.stderr
    assert (output / "artifacts" / "cache-check-models.txt").read_text().splitlines() == [
        "convbert-base"
    ]
    docker_runs = [
        line for line in docker_log.read_text(encoding="utf-8").splitlines()
        if line.startswith("run ")
    ]
    assert len(docker_runs) == 1
    assert "scripts/warm_hf_cache.py" in docker_runs[0]
    assert "--local-only" in docker_runs[0]
    assert "--strict" in docker_runs[0]
    assert "--network none" in docker_runs[0]


def test_host_cache_existence_is_delegated_to_read_only_docker_mounts() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    host = text.split("run_host() {", maxsplit=1)[1]
    cache_check = host.split(
        "local -a cache_check_docker_args=(", maxsplit=1
    )[1].split("set +e", maxsplit=1)[0]
    proof = host.split("local -a docker_args=(", maxsplit=1)[1].split(
        "set +e", maxsplit=1
    )[0]

    assert '[ -d "$hf_hub_cache" ]' not in host
    assert '[ -d "$hf_modules_cache" ]' not in host
    assert "HF Hub cache directory does not exist" not in host
    assert "HF modules cache directory does not exist" not in host
    assert '[ "$hf_hub_cache" != "/" ]' in host
    assert '[ "$hf_modules_cache" != "/" ]' in host
    for docker_args in (cache_check, proof):
        assert (
            '--mount "type=bind,src=$hf_hub_cache,dst=/hf-cache/hub,readonly"'
            in docker_args
        )
        assert (
            '--mount "type=bind,src=$hf_modules_cache,dst=/hf-cache/modules,readonly"'
            in docker_args
        )


def test_distinct_explicit_hf_cache_paths_reach_both_containers(
    tmp_path: Path,
) -> None:
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    hub_cache = tmp_path / "explicit-hub-cache"
    modules_cache = tmp_path / "explicit-modules-cache"
    hub_cache.mkdir()
    modules_cache.mkdir()
    env.update(
        {
            "TRTMC_HF_HUB_CACHE": str(hub_cache),
            "TRTMC_HF_MODULES_CACHE": str(modules_cache),
        }
    )

    result = subprocess.run(
        [
            "bash", str(RUNNER),
            "--model", "convbert",
            "--revision", "HEAD",
            "--output-dir", str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    docker_runs = [
        line for line in docker_log.read_text(encoding="utf-8").splitlines()
        if line.startswith("run ")
    ]
    assert len(docker_runs) == 2
    for docker_run in docker_runs:
        assert (
            f"--mount type=bind,src={hub_cache},dst=/hf-cache/hub,readonly"
            in docker_run
        )
        assert (
            f"--mount type=bind,src={modules_cache},dst=/hf-cache/modules,readonly"
            in docker_run
        )


def test_docker_bind_mount_fails_closed_when_host_cache_source_is_absent(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\n' \"$*\" >> \"$DOCKER_LOG\"\n"
        "if [ \"${1:-}\" = image ] || [ \"${1:-}\" = rm ]; then exit 0; fi\n"
        "if [ \"${1:-}\" = run ]; then exit 23; fi\n"
        "exit 99\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    hub_cache = tmp_path / "missing-hub-cache"
    modules_cache = tmp_path / "missing-modules-cache"
    assert not hub_cache.exists()
    assert not modules_cache.exists()
    output = tmp_path / "proof"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "DOCKER_LOG": str(docker_log),
            "TRTMC_HF_HUB_CACHE": str(hub_cache),
            "TRTMC_HF_MODULES_CACHE": str(modules_cache),
        }
    )

    result = subprocess.run(
        [
            "bash", str(RUNNER),
            "--model", "convbert",
            "--revision", "HEAD",
            "--output-dir", str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "offline HF cache readiness check failed for convbert" in result.stderr
    docker_runs = [
        line for line in docker_log.read_text(encoding="utf-8").splitlines()
        if line.startswith("run ")
    ]
    assert len(docker_runs) == 1
    assert (
        f"--mount type=bind,src={hub_cache},dst=/hf-cache/hub,readonly"
        in docker_runs[0]
    )
    assert (
        f"--mount type=bind,src={modules_cache},dst=/hf-cache/modules,readonly"
        in docker_runs[0]
    )
    assert "--network none" in docker_runs[0]


def test_explicit_runner_gpu_id_bypasses_automatic_leasing(tmp_path: Path) -> None:
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    env.update(
        {
            "TRTMC_GPU_ID": "7",
            "TRTMC_MODEL_PROOF_GPU_IDS": "invalid-auto-config-is-ignored",
        }
    )

    result = subprocess.run(
        [
            "bash", str(RUNNER),
            "--model", "convbert",
            "--revision", "HEAD",
            "--output-dir", str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Using explicit model-proof GPU 7" in result.stdout
    assert _proof_gpu_ids(docker_log) == ["7"]
    assert (output / "artifacts" / "gpu-id.txt").read_text().strip() == "7"
    assert not (tmp_path / "gpu-locks").exists()


def test_automatic_gpu_leases_are_unique_across_parallel_proofs(
    tmp_path: Path,
) -> None:
    processes: list[tuple[subprocess.Popen[str], Path, Path]] = []
    for index in range(2):
        case_dir = tmp_path / f"case-{index}"
        case_dir.mkdir()
        fake_bin, docker_log = _write_successful_fake_docker(case_dir)
        output = case_dir / "proof"
        env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
        env.update(
            {
                "TRTMC_MODEL_PROOF_GPU_IDS": "2,3",
                "FAKE_PROOF_DELAY_SECONDS": "2",
            }
        )
        process = subprocess.Popen(
            [
                "bash", str(RUNNER),
                "--model", "convbert",
                "--revision", "HEAD",
                "--output-dir", str(output),
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append((process, docker_log, output))

    selected: list[str] = []
    for process, docker_log, output in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stdout + stderr
        selected.extend(_proof_gpu_ids(docker_log))
        assert (output / "artifacts" / "gpu-id.txt").is_file()

    assert sorted(selected) == ["2", "3"]


def test_automatic_gpu_lease_rejects_invalid_id_configuration(
    tmp_path: Path,
) -> None:
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    env["TRTMC_MODEL_PROOF_GPU_IDS"] = "0,,1"

    result = subprocess.run(
        [
            "bash", str(RUNNER),
            "--model", "convbert",
            "--revision", "HEAD",
            "--output-dir", str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "TRTMC_MODEL_PROOF_GPU_IDS must be a comma-separated list" in result.stderr
    assert not _proof_gpu_ids_if_present(docker_log)


def test_automatic_gpu_lease_times_out_when_every_gpu_is_busy(
    tmp_path: Path,
) -> None:
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    env.update(
        {
            "TRTMC_MODEL_PROOF_GPU_IDS": "9",
            "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": "1",
        }
    )
    lock_dir = tmp_path / "gpu-locks"
    lock_dir.mkdir()
    with (lock_dir / "gpu-9.lock").open("w", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            [
                "bash", str(RUNNER),
                "--model", "convbert",
                "--revision", "HEAD",
                "--output-dir", str(output),
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )

    assert result.returncode != 0
    assert "timed out after 1s waiting for a model-proof GPU lease from: 9" in result.stderr
    assert not _proof_gpu_ids_if_present(docker_log)


def _proof_gpu_ids_if_present(docker_log: Path) -> list[str]:
    proof_runs = [
        line
        for line in docker_log.read_text(encoding="utf-8").splitlines()
        if " --inner " in f" {line} "
    ]
    return [
        gpu_id
        for line in proof_runs
        for gpu_id in re.findall(r"--gpus device=([0-9]+)", line)
    ]


def test_gpu_mapping_exists_only_on_the_hermetic_proof_container() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    workflow = PROOF_WORKFLOW.read_text(encoding="utf-8")
    host = text.split("run_host() {", maxsplit=1)[1]
    warm = host.split("local -a cache_check_docker_args=(", maxsplit=1)[1].split(
        "local -a docker_args=(", maxsplit=1
    )[0]
    proof = host.split("local -a docker_args=(", maxsplit=1)[1].split(
        "set +e", maxsplit=1
    )[0]

    assert host.index("warm_hf_cache.py") < host.index("select_proof_gpu")
    assert "--gpus" not in warm
    assert "TRTMC_MODEL_PROOF_GPU_ID" not in warm
    assert '--gpus "device=$gpu_id"' in proof
    assert '-e "TRTMC_MODEL_PROOF_GPU_ID=$gpu_id"' in proof
    assert 'update_proof_fact gpu_id "$TRTMC_MODEL_PROOF_GPU_ID"' in text
    assert '"gpu_id": gpu_id' in text
    assert "TRTMC_MODEL_PROOF_GPU_IDS: ${{ vars.TRTMC_MODEL_PROOF_GPU_IDS || '0,1,2,3' }}" in workflow
    assert "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS:" in workflow
