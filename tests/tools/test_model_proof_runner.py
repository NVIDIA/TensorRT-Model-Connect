# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the hermetic single-model proof runner."""

from __future__ import annotations

from collections.abc import Callable
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / ".github" / "scripts" / "run-model-proof.sh"
IMAGE_ENSURE = REPO_ROOT / ".github" / "scripts" / "ensure-ci-docker-image.sh"
PROOF_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "model-proof.yml"
FALLBACK_WRITER = REPO_ROOT / ".github" / "scripts" / "write-model-proof-fallback-report.py"
PLUGIN_CMAKE = REPO_ROOT / "cmake" / "trtmc_pipeline_plugins.cmake"
SANA_REFERENCE_REVISION = "59629fdf790850797cb657bad014fce432bd713d"
SANA_REFERENCE_RELATIVE_PATH = "sana_wm/reference/Sana-59629fdf7908"
SANA_REFERENCE_ENTRYPOINT = "inference_video_scripts/wm/inference_sana_wm.py"


def _write_successful_fake_docker(tmp_path: Path) -> tuple[Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$*" >> "$DOCKER_LOG"\n'
        'case "${1:-}" in\n'
        "  image) exit 0 ;;\n"
        "  rm)\n"
        '    if [ "${3:-}" = "${FAKE_ORPHAN_CONTAINER_ID:-}" ]; then\n'
        '      exit "${FAKE_ORPHAN_RM_EXIT_CODE:-0}"\n'
        "    fi\n"
        "    exit 0\n"
        "    ;;\n"
        "  ps)\n"
        '    if [ "${FAKE_DOCKER_PS_EXIT_CODE:-0}" -ne 0 ]; then\n'
        '      exit "$FAKE_DOCKER_PS_EXIT_CODE"\n'
        "    fi\n"
        '    if [[ " $* " == *" -a "* ]]; then\n'
        '      record="${FAKE_ORPHAN_CONFIRM_RECORD:-}"\n'
        "    else\n"
        '      record="${FAKE_ORPHAN_CONTAINER_RECORD:-}"\n'
        "    fi\n"
        '    if [ -n "$record" ]; then\n'
        '      printf \'%s\\n\' "$record"\n'
        "    fi\n"
        "    exit 0\n"
        "    ;;\n"
        "  run)\n"
        '    if [[ " $* " == *" /src/scripts/warm_hf_cache.py "* ]]; then\n'
        '      mkdir -p "$FAKE_ARTIFACTS"\n'
        '      if [ "${FAKE_CACHE_EVIDENCE_MODE:-valid}" = escape ]; then\n'
        '        printf \'%s\\n\' \'{"schema_version":1,"hub_cache":"/hf-cache/hub","repositories":[{"repo_id":"fixture/model","repo_type":"model","cache_folder":"../escape","cache_path":"/hf-cache/escape"}]}\' > "$FAKE_ARTIFACTS/hf-cache-repos.json"\n'
        "      else\n"
        '        printf \'%s\\n\' \'{"schema_version":1,"hub_cache":"/hf-cache/hub","repositories":[{"repo_id":"fixture/model","repo_type":"model","cache_folder":"models--fixture--model","cache_path":"/hf-cache/hub/models--fixture--model"}]}\' > "$FAKE_ARTIFACTS/hf-cache-repos.json"\n'
        "      fi\n"
        "      exit 0\n"
        "    fi\n"
        '    if [[ " $* " == *"dst=/selected-hf-repo,readonly"* ]]; then\n'
        '      if [ "${FAKE_REFLINK_EXIT_CODE:-0}" -ne 0 ]; then\n'
        '        exit "$FAKE_REFLINK_EXIT_CODE"\n'
        "      fi\n"
        '      selected_repo=""\n'
        '      private_repo=""\n'
        '      for argument in "$@"; do\n'
        '        case "$argument" in\n'
        "          type=bind,src=*,dst=/selected-hf-repo,readonly)\n"
        '            selected_repo="${argument#type=bind,src=}"\n'
        '            selected_repo="${selected_repo%,dst=/selected-hf-repo,readonly}"\n'
        "            ;;\n"
        "          type=bind,src=*,dst=/private-hf-repo)\n"
        '            private_repo="${argument#type=bind,src=}"\n'
        '            private_repo="${private_repo%,dst=/private-hf-repo}"\n'
        "            ;;\n"
        "        esac\n"
        "      done\n"
        '      [ -n "$selected_repo" ] && [ -n "$private_repo" ] || exit 96\n'
        '      if [ "${FAKE_UNREADABLE_REFLINK:-0}" = 1 ]; then\n'
        "        printf '%s\\n' \"${FAKE_UNREADABLE_REFLINK_CONTENT:-private copy}\" > "
        '"$private_repo/unreadable-cache-marker"\n'
        "      else\n"
        '        /bin/cp -a --reflink=always -- "$selected_repo/." "$private_repo/"\n'
        "      fi\n"
        "      exit 0\n"
        "    fi\n"
        '    if [[ " $* " == *" --inner "* ]]; then\n'
        '      if [ -n "${FAKE_PROOF_RELEASE_FILE:-}" ]; then\n'
        "        deadline=$((SECONDS + 600))\n"
        '        while [ ! -e "$FAKE_PROOF_RELEASE_FILE" ]; do\n'
        '          [ "$SECONDS" -lt "$deadline" ] || exit 98\n'
        "          sleep 0.05\n"
        "        done\n"
        "      fi\n"
        '      sleep "${FAKE_PROOF_DELAY_SECONDS:-0}"\n'
        '      mkdir -p "$FAKE_ARTIFACTS"\n'
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
    (tmp_path / "hf-cache" / "hub" / "models--fixture--model").mkdir(parents=True, exist_ok=True)
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
            # Fast state transitions keep the multi-process coordination tests
            # deterministic without widening their assertion windows.
            "TRTMC_MODEL_PROOF_POLL_INTERVAL": "0.05",
            "TRTMC_MODEL_PROOF_FLOCK_WATCHDOG_SECONDS": "2",
        }
    )
    return env


def _run_fake_proof(env: dict[str, str], output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--model",
            "convbert",
            "--revision",
            "HEAD",
            "--output-dir",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_fake_pinned_model_reference(
    tmp_path: Path,
    fake_bin: Path,
) -> tuple[Path, Path]:
    cache_root = tmp_path / "model-reference-cache"
    source = cache_root / SANA_REFERENCE_RELATIVE_PATH
    entrypoint = source / SANA_REFERENCE_ENTRYPOINT
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# pinned SANA-WM reference\n", encoding="utf-8")
    (source / "reference-marker.txt").write_text("selected only\n", encoding="utf-8")

    real_git = shutil.which("git")
    assert real_git is not None
    fake_git = fake_bin / "git"
    fake_git.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import os
            from pathlib import Path
            import subprocess
            import sys

            args = sys.argv[1:]
            source = Path(os.environ["FAKE_REFERENCE_SOURCE"]).resolve()
            if len(args) >= 3 and args[0] == "-C" and Path(args[1]).resolve() == source:
                command = args[2]
                if command == "rev-parse":
                    value = args[3]
                    if value == "HEAD^{{commit}}":
                        print(os.environ["FAKE_REFERENCE_REVISION"])
                        raise SystemExit(0)
                    if value.endswith("^{{tree}}"):
                        print("1" * 40)
                        raise SystemExit(0)
                if command == "config" and args[3:] == ["--get", "remote.origin.url"]:
                    print("https://github.com/NVlabs/Sana.git")
                    raise SystemExit(0)
                if command == "cat-file":
                    revision_and_path = args[-1]
                    relative = revision_and_path.split(":", 1)[1]
                    raise SystemExit(0 if (source / relative).is_file() else 1)
                if command == "archive":
                    raise SystemExit(subprocess.run(
                        ["tar", "--exclude=.git", "-C", str(source), "-cf", "-", "."],
                        check=False,
                    ).returncode)
            os.execv({real_git!r}, [{real_git!r}, *args])
            """
        ),
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    return cache_root, source


def _write_sana_reference_config(path: Path) -> None:
    path.write_text(
        "model_reference_repository=https://github.com/NVlabs/Sana.git\n"
        f"model_reference_revision={SANA_REFERENCE_REVISION}\n"
        f"model_reference_relative_path={SANA_REFERENCE_RELATIVE_PATH}\n"
        f"model_reference_entrypoint={SANA_REFERENCE_ENTRYPOINT}\n",
        encoding="utf-8",
    )


def _prepare_reference_program() -> str:
    text = RUNNER.read_text(encoding="utf-8")
    function = text.split("prepare_private_model_reference_cache() {", maxsplit=1)[1]
    function = (
        "prepare_private_model_reference_cache() {"
        + function.split("\n\nrun_host() {", maxsplit=1)[0]
    )
    return (
        "set -euo pipefail\n"
        'model="sana_wm"\n'
        'die() { echo "ERROR: $*" >&2; exit 1; }\n'
        f"{function}\n"
        'prepare_private_model_reference_cache "$1" "$2" "$3" "$4"\n'
    )


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


def _gpu_lease(output: Path) -> dict:
    return json.loads((output / "artifacts" / "gpu-lease.json").read_text(encoding="utf-8"))


def _lock_is_busy(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
    return False


def _selection_program() -> str:
    text = RUNNER.read_text(encoding="utf-8")
    marker = "> \"$config_file\" <<'PY'\n"
    return text.split(marker, maxsplit=1)[1].split("\nPY\n", maxsplit=1)[0]


def _workflow_singleton_gate_program() -> str:
    workflow = PROOF_WORKFLOW.read_text(encoding="utf-8")
    step = workflow.split("- name: Enforce isolated model certification", maxsplit=1)[1].split(
        "- name: Clean model proof scratch space", maxsplit=1
    )[0]
    program = step.split("<<'PY'\n", maxsplit=1)[1].split("\n          PY", maxsplit=1)[0]
    return textwrap.dedent(program)


def _run_test_selection(
    tmp_path: Path,
    family: str,
    suite: str,
    *,
    lease_env: dict[str, str] | None = None,
) -> dict:
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
        'id = "fixture_runtime"\nruntime_library = "libtrtmc_model_fixture_runtime.so"\n',
        encoding="utf-8",
    )
    family_source = REPO_ROOT / "python" / "tensorrt_model_connect" / "families" / family
    family_root = source / "python" / "tensorrt_model_connect" / "families"
    if family_source.is_dir():
        shutil.copytree(family_source, family_root / family)
    else:
        family_root.mkdir(parents=True)
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
    env = os.environ.copy()
    for name in (
        "TRTMC_MODEL_PROOF_GPU_ID",
        "TRTMC_MODEL_PROOF_GPU_SLOT_IDS",
        "TRTMC_MODEL_PROOF_SLOTS_PER_GPU",
        "TRTMC_MODEL_PROOF_RESOURCE_CLASS",
    ):
        env.pop(name, None)
    env.update(lease_env or {})
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
        env=env,
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
    families = sorted(path.parent.name for path in model_root.glob("*/MODEL.toml"))

    assert families
    for family in families:
        selection = _run_test_selection(tmp_path, family, "premerge")
        assert len(selection["e2e_cases"]) == 1, family
        assert selection["e2e_cases"][0]["ci_tier"] != "nightly_only", family


@pytest.mark.parametrize(
    ("family", "expected_family_tests"),
    (
        ("flux", {"python/tensorrt_model_connect/families/flux/tests/test_family.py"}),
        (
            "sana_wm",
            {
                "python/tensorrt_model_connect/families/sana_wm/tests/test_family.py",
                "python/tensorrt_model_connect/families/sana_wm/tests/test_native_plugin_builder.py",
                "python/tensorrt_model_connect/families/sana_wm/tests/test_stage1_dit_builder.py",
            },
        ),
    ),
)
def test_selection_includes_every_owned_python_family_test(
    tmp_path: Path,
    family: str,
    expected_family_tests: set[str],
) -> None:
    selection = _run_test_selection(tmp_path, family, "premerge")

    assert selection["python_family"] == family
    assert expected_family_tests <= set(selection["python_tests"])


def test_sana_selection_declares_its_pinned_model_reference_cache(
    tmp_path: Path,
) -> None:
    selection = _run_test_selection(tmp_path, "sana_wm", "premerge")

    assert selection["model_reference_cache"] == {
        "repository": "https://github.com/NVlabs/Sana.git",
        "revision": SANA_REFERENCE_REVISION,
        "relative_path": SANA_REFERENCE_RELATIVE_PATH,
        "entrypoint": SANA_REFERENCE_ENTRYPOINT,
    }


def test_inner_proof_runs_the_exact_model_owned_python_test_selection() -> None:
    runner = RUNNER.read_text(encoding="utf-8")

    assert "for test in python_tests:" in runner
    assert 'print(f"python_test={test.relative_to(root)}")' in runner
    assert "sed -n 's/^python_test=//p'" in runner
    assert 'python_tests+=("/src/$python_test")' in runner
    assert 'find "$python_test_dir" -maxdepth 1' not in runner


@pytest.mark.parametrize(
    ("family", "expected_resource"),
    (
        ("bark", "exclusive_gpu"),
        ("convbert", "shared"),
        ("flux", "exclusive_gpu"),
        ("gpt2", "exclusive_gpu"),
        ("mixtral", "exclusive_gpu"),
        ("timesfm", "exclusive_gpu"),
    ),
)
def test_selection_derives_the_most_restrictive_gpu_resource_class(
    tmp_path: Path,
    family: str,
    expected_resource: str,
) -> None:
    selection = _run_test_selection(tmp_path, family, "premerge")

    assert selection["resource_class"] == expected_resource
    assert {case["resource_class"] for case in selection["e2e_cases"]} == {expected_resource}


def test_inner_selection_records_the_leased_gpu_evidence(tmp_path: Path) -> None:
    selection = _run_test_selection(
        tmp_path,
        "convbert",
        "premerge",
        lease_env={
            "TRTMC_MODEL_PROOF_GPU_ID": "2",
            "TRTMC_MODEL_PROOF_GPU_SLOT_IDS": "3",
            "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "4",
            "TRTMC_MODEL_PROOF_RESOURCE_CLASS": "shared",
        },
    )

    assert selection["gpu_id"] == "2"
    assert selection["gpu_slot_ids"] == [3]
    assert selection["gpu_slots_per_device"] == 4
    assert selection["gpu_resource_class"] == "shared"
    assert selection["gpu_lease_evidence"] == "gpu-lease.json"


@pytest.mark.parametrize(
    ("family", "expected_cases"),
    (
        (
            "flux",
            {
                "flux-2-dev-fp8",
                "flux-2-dev",
                "flux-schnell",
            },
        ),
        ("personaplex", {"personaplex-7b"}),
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
def test_nightly_selects_production_single_gpu_cases_without_redundant_l0(
    tmp_path: Path,
    family: str,
    expected_cases: set[str],
) -> None:
    selection = _run_test_selection(tmp_path, family, "nightly")

    assert selection["suite"] == "nightly"
    assert {case["name"] for case in selection["e2e_cases"]} == expected_cases
    assert any(case["ci_tier"] == "nightly_only" for case in selection["e2e_cases"])
    assert all(case["ci_tier"] != "l0_only" for case in selection["e2e_cases"])
    assert all(case["ci_tier"] != "multi_device" for case in selection["e2e_cases"])


def test_flux_nightly_model_proof_reserves_an_exclusive_gpu(tmp_path: Path) -> None:
    selection = _run_test_selection(
        tmp_path,
        "flux",
        "nightly",
        lease_env={
            "TRTMC_MODEL_PROOF_GPU_ID": "2",
            "TRTMC_MODEL_PROOF_GPU_SLOT_IDS": "0,1,2,3",
            "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "4",
            "TRTMC_MODEL_PROOF_RESOURCE_CLASS": "exclusive_gpu",
        },
    )

    assert selection["resource_class"] == "exclusive_gpu"
    assert selection["gpu_resource_class"] == "exclusive_gpu"
    assert selection["gpu_slot_ids"] == [0, 1, 2, 3]
    assert selection["gpu_slot"] is None
    assert {
        case["name"]: case["gpu_resource_class"]
        for case in selection["e2e_cases"]
    } == {
        "flux-2-dev": "exclusive_gpu",
        "flux-2-dev-fp8": "shared",
        "flux-schnell": "shared",
    }


@pytest.mark.parametrize("family", ("locateanything", "ltx_video"))
def test_nightly_retains_l0_as_an_owners_only_single_gpu_fallback(
    tmp_path: Path, family: str
) -> None:
    selection = _run_test_selection(tmp_path, family, "nightly")

    assert selection["e2e_cases"]
    assert all(case["ci_tier"] == "l0_only" for case in selection["e2e_cases"])


@pytest.mark.parametrize(
    ("family", "multi_gpu_case"),
    (
        ("internvl", "internvl3-8b-tp4"),
        ("qwen_moe", "qwen3-moe-30b-a3b-tp4"),
    ),
)
def test_nightly_single_gpu_selection_excludes_tp4_cases(
    tmp_path: Path,
    family: str,
    multi_gpu_case: str,
) -> None:
    selection = _run_test_selection(tmp_path, family, "nightly")

    selected = {case["name"] for case in selection["e2e_cases"]}
    assert multi_gpu_case not in selected
    assert selected
    assert all(case["ci_tier"] != "multi_device" for case in selection["e2e_cases"])


def test_runner_declares_the_hermetic_container_boundary() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    warm = text.split("local -a cache_check_docker_args=(", maxsplit=1)[1].split(
        "local -a docker_args=(", maxsplit=1
    )[0]
    proof = text.split("local -a docker_args=(", maxsplit=1)[1].split("set +e", maxsplit=1)[0]

    for contract in (
        "--read-only",
        "--network none",
        "--cap-drop ALL",
        "dst=/src,readonly",
        "TMPDIR=/work/tmp",
        "TORCHINDUCTOR_CACHE_DIR=/work/torch-cache",
        "TRTMC_MODEL_PLUGIN_STRICT=1",
        "scratch build produced ${#built_dsos[@]} model DSOs; expected exactly one",
        "staged plugin directory contains ${#staged_dsos[@]} model DSOs; expected exactly one",
        "staged plugin DSO does not byte-match the scratch-built DSO",
        "staged plugin DSO SHA-256 does not match the scratch-built DSO",
    ):
        assert contract in text
    assert "--network none" in warm
    assert "dst=/hf-cache/hub,readonly" in warm
    assert "dst=/hf-cache/modules" not in warm
    assert "-e HF_HOME=/tmp/hf-home" in warm
    assert "-e HF_MODULES_CACHE=/tmp/hf-modules" in warm
    assert 'dst=/artifacts"' in warm
    assert "dst=/artifacts,readonly" not in warm
    assert "-e HF_TOKEN" not in warm
    assert "-e HUGGING_FACE_HUB_TOKEN" not in warm
    assert "--network none" in proof
    assert "-e TMPDIR=/work/tmp" in proof
    assert "-e TMPDIR=/work/tmp" not in warm
    assert '--mount "type=bind,src=$hf_private_hub,dst=/hf-cache/hub"' in proof
    assert "src=$hf_private_hub,dst=/hf-cache/hub,readonly" not in proof
    assert "dst=/hf-cache/hub/$hf_repo_folder" not in text
    assert "src=$hf_hub_cache,dst=/hf-cache/hub" not in proof
    assert "dst=/hf-cache/modules" not in proof
    assert "-e HF_HOME=/work/hf-home" in proof
    assert "-e HF_MODULES_CACHE=/work/hf-modules" in proof
    assert "-e TRANSFORMERS_CACHE=/hf-cache/hub" in proof
    assert "cp -a --reflink=always --no-preserve=ownership --" in text
    assert "chmod -R u+rwX --" in text
    assert "-e TRTMC_STORAGE_ROOT=/work/reference-private" in proof
    assert "TRTMC_MODEL_REFERENCE_CACHE_ROOT" not in proof
    assert "src=$reference_cache_root" not in proof
    assert "src=$reference_source" not in proof
    assert "-e HF_TOKEN" not in proof
    assert "-e HUGGING_FACE_HUB_TOKEN" not in proof


def test_runner_warms_the_exact_shared_selection_before_the_proof() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    host = text.split("run_host() {", maxsplit=1)[1]
    warm = host.split("local -a cache_check_docker_args=(", maxsplit=1)[1].split(
        "local -a docker_args=(", maxsplit=1
    )[0]

    assert host.count("write_model_proof_selection") == 1
    assert "sed -n 's/^e2e_model=//p' \"$host_config_file\"" in host
    assert "cache-check-models.txt" in warm
    assert "scripts/warm_hf_cache.py" in warm
    assert "--models-file /artifacts/cache-check-models.txt --local-only --strict" in warm
    assert "--emit-cache-repos /artifacts/hf-cache-repos.json" in warm
    assert host.index("scripts/warm_hf_cache.py") < host.index("local -a docker_args=(")
    assert 'die "offline HF cache readiness check failed' in warm


def test_runner_keeps_local_fallback_and_workflow_uses_runner_cache_paths() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    workflow = PROOF_WORKFLOW.read_text(encoding="utf-8")

    assert "${HF_HOME:-$HOME/.cache/huggingface}" in runner
    assert "TRTMC_HF_CACHE: ${{ vars.TRTMC_HF_HOME || " in workflow
    assert "TRTMC_HF_HUB_CACHE: ${{ vars.TRTMC_HF_HUB_CACHE || " in workflow
    assert "format('{0}/hub', vars.TRTMC_HF_HOME || " in workflow
    assert (
        "TRTMC_MODEL_REFERENCE_CACHE_ROOT: ${{ vars.TRTMC_MODEL_REFERENCE_CACHE_ROOT || "
        in workflow
    )
    assert "TRTMC_HF_MODULES_CACHE:" not in workflow
    assert "${TRTMC_HF_HUB_CACHE:-$hf_cache_root/hub}" in runner
    assert "TRTMC_HF_MODULES_CACHE" not in runner


def test_hf_token_is_not_exposed_to_pull_request_model_proof_code() -> None:
    workflow = PROOF_WORKFLOW.read_text(encoding="utf-8")
    job_environment = workflow.split("\n    steps:", maxsplit=1)[0]
    proof_step = workflow.split("- name: Run isolated model proof", maxsplit=1)[1].split(
        "- name: Finalize model proof fallback", maxsplit=1
    )[0]

    assert "HF_TOKEN:" not in job_environment
    assert "HUGGING_FACE_HUB_TOKEN:" not in job_environment
    assert "HF_TOKEN:" not in proof_step
    assert "HUGGING_FACE_HUB_TOKEN:" not in proof_step


def test_runner_removes_only_its_container_without_masking_exit_status() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    cleanup = text.split("cleanup_proof_container() {", maxsplit=1)[1].split("\n}\n", maxsplit=1)[0]

    assert 'local rc="$1"' in cleanup
    assert "trap - EXIT" in cleanup
    assert "trap '' INT TERM" in cleanup
    assert 'docker rm -f "$proof_container_name"' in cleanup
    assert cleanup.index('docker rm -f "$proof_container_name"') < cleanup.index(
        "release_proof_gpu_lease"
    )
    assert 'exit "$rc"' in cleanup
    assert "artifacts" not in cleanup
    assert 'proof_container_name="$container_name"' in text
    assert "trap 'cleanup_proof_container \"$?\"' EXIT" in text
    assert "trap 'cleanup_proof_container 130' INT" in text
    assert "trap 'cleanup_proof_container 143' TERM" in text


def test_workflow_reconciles_exact_job_containers_after_a_cancelled_proof() -> None:
    workflow = PROOF_WORKFLOW.read_text(encoding="utf-8")
    reconciliation = workflow.split(
        "- name: Reconcile model proof containers", maxsplit=1
    )[1].split("- name: Finalize model proof fallback", maxsplit=1)[0]

    assert "if: ${{ always() && steps.checkout.outcome == 'success' }}" in reconciliation
    assert "--cleanup-containers" in reconciliation
    assert '--model "$MODEL"' in reconciliation
    assert workflow.index("Reconcile model proof containers") < workflow.index(
        "Finalize model proof fallback"
    )


def test_every_host_container_has_exact_workflow_job_identity_labels(tmp_path: Path) -> None:
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    env.update({"GITHUB_RUN_ID": "4242", "GITHUB_RUN_ATTEMPT": "3"})

    result = _run_fake_proof(env, output)

    assert result.returncode == 0, result.stdout + result.stderr
    runs = [
        line
        for line in docker_log.read_text(encoding="utf-8").splitlines()
        if line.startswith("run ")
    ]
    assert len(runs) == 3
    for run in runs:
        assert "--label com.nvidia.trtmc.model-proof.job=1" in run
        assert "--label com.nvidia.trtmc.model-proof.run-id=4242" in run
        assert "--label com.nvidia.trtmc.model-proof.run-attempt=3" in run
        assert "--label com.nvidia.trtmc.model-proof.model=convbert" in run


def test_cleanup_mode_removes_only_exact_labeled_job_containers(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    state = tmp_path / "container-present"
    state.write_text("present\n", encoding="utf-8")
    remove_count = tmp_path / "remove-count"
    container_id = "d" * 64
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$*" >> "$DOCKER_LOG"\n'
        'case "${1:-}" in\n'
        "  ps)\n"
        '    if [ -e "$FAKE_CONTAINER_STATE" ]; then\n'
        f'      printf \'%s\\n\' "{container_id} 4242 3 convbert"\n'
        "    fi\n"
        "    ;;\n"
        "  rm)\n"
        "    count=0\n"
        '    if [ -e "$FAKE_REMOVE_COUNT" ]; then read -r count < "$FAKE_REMOVE_COUNT"; fi\n'
        "    count=$((count + 1))\n"
        '    printf \'%s\\n\' "$count" > "$FAKE_REMOVE_COUNT"\n'
        '    if [ "$count" -ge 3 ]; then rm -f -- "$FAKE_CONTAINER_STATE"; fi\n'
        "    ;;\n"
        "  *) exit 99 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "DOCKER_LOG": str(docker_log),
            "FAKE_CONTAINER_STATE": str(state),
            "FAKE_REMOVE_COUNT": str(remove_count),
            "GITHUB_RUN_ID": "4242",
            "GITHUB_RUN_ATTEMPT": "3",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--cleanup-containers",
            "--model",
            "convbert",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    lines = docker_log.read_text(encoding="utf-8").splitlines()
    inventory = next(line for line in lines if line.startswith("ps "))
    assert "label=com.nvidia.trtmc.model-proof.job=1" in inventory
    assert "label=com.nvidia.trtmc.model-proof.run-id=4242" in inventory
    assert "label=com.nvidia.trtmc.model-proof.run-attempt=3" in inventory
    assert "label=com.nvidia.trtmc.model-proof.model=convbert" in inventory
    assert f"rm -f {container_id}" in lines
    assert remove_count.read_text(encoding="utf-8").strip() == "3"
    assert not state.exists()


@pytest.mark.parametrize(("orphan_slots", "removed"), [("0", True), ("1", False)])
def test_runner_reclaims_only_a_labeled_orphan_overlapping_its_lease(
    tmp_path: Path,
    orphan_slots: str,
    removed: bool,
) -> None:
    orphan_id = "d" * 64
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    env.update(
        {
            "FAKE_ORPHAN_CONTAINER_ID": orphan_id,
            "FAKE_ORPHAN_CONTAINER_RECORD": f"{orphan_id} {orphan_slots}",
            "TRTMC_MODEL_PROOF_GPU_IDS": "7",
            "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "2",
        }
    )

    result = _run_fake_proof(env, output)

    assert result.returncode == 0, result.stdout + result.stderr
    docker_lines = docker_log.read_text(encoding="utf-8").splitlines()
    orphan_remove = f"rm -f {orphan_id}"
    assert (orphan_remove in docker_lines) is removed
    proof_run = next(line for line in docker_lines if " --inner " in f" {line} ")
    assert "--label com.nvidia.trtmc.model-proof=1" in proof_run
    assert "--label com.nvidia.trtmc.model-proof.gpu=7" in proof_run
    assert "--label com.nvidia.trtmc.model-proof.slots=0" in proof_run
    namespace_match = re.search(
        r"--label com\.nvidia\.trtmc\.model-proof\.lock-namespace=([a-f0-9]{64})",
        proof_run,
    )
    assert namespace_match is not None
    inspection = next(line for line in docker_lines if line.startswith("ps "))
    assert inspection.startswith("ps --no-trunc ")
    assert (
        f"label=com.nvidia.trtmc.model-proof.lock-namespace={namespace_match.group(1)}"
        in inspection
    )
    if removed:
        inspection_index = next(
            index
            for index, line in enumerate(docker_lines)
            if line.startswith("ps ") and "model-proof.gpu=7" in line
        )
        assert inspection_index < docker_lines.index(orphan_remove) < docker_lines.index(proof_run)


@pytest.mark.parametrize(
    ("record", "ps_exit_code", "expected_error"),
    [
        (
            f"{'d' * 64} invalid",
            "0",
            f"existing model-proof container {'d' * 64} has invalid GPU slot labels",
        ),
        ("", "75", "could not inspect existing model-proof containers on GPU 7"),
    ],
)
def test_orphan_reclamation_fails_closed_on_untrusted_or_unavailable_inventory(
    tmp_path: Path,
    record: str,
    ps_exit_code: str,
    expected_error: str,
) -> None:
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    env.update(
        {
            "FAKE_ORPHAN_CONTAINER_RECORD": record,
            "FAKE_DOCKER_PS_EXIT_CODE": ps_exit_code,
            "TRTMC_MODEL_PROOF_GPU_IDS": "7",
        }
    )

    result = _run_fake_proof(env, output)

    assert result.returncode != 0
    assert expected_error in result.stderr
    docker_lines = docker_log.read_text(encoding="utf-8").splitlines()
    assert not any(" --inner " in f" {line} " for line in docker_lines)
    assert f"rm -f {'d' * 64}" not in docker_lines


def test_orphan_reclamation_accepts_only_the_auto_remove_race(tmp_path: Path) -> None:
    orphan_id = "d" * 64
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    env.update(
        {
            "FAKE_ORPHAN_CONTAINER_ID": orphan_id,
            "FAKE_ORPHAN_CONTAINER_RECORD": f"{orphan_id} 0",
            "FAKE_ORPHAN_RM_EXIT_CODE": "1",
            "TRTMC_MODEL_PROOF_GPU_IDS": "7",
        }
    )

    result = _run_fake_proof(env, output)

    assert result.returncode == 0, result.stdout + result.stderr
    docker_lines = docker_log.read_text(encoding="utf-8").splitlines()
    assert f"rm -f {orphan_id}" in docker_lines
    assert any(
        line.startswith("ps -a --no-trunc ") and f"--filter id={orphan_id}" in line
        for line in docker_lines
    )


def test_orphan_reclamation_rejects_a_failed_remove_when_full_id_remains(
    tmp_path: Path,
) -> None:
    orphan_id = "d" * 64
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    env.update(
        {
            "FAKE_ORPHAN_CONTAINER_ID": orphan_id,
            "FAKE_ORPHAN_CONTAINER_RECORD": f"{orphan_id} 0",
            "FAKE_ORPHAN_CONFIRM_RECORD": orphan_id,
            "FAKE_ORPHAN_RM_EXIT_CODE": "1",
            "TRTMC_MODEL_PROOF_GPU_IDS": "7",
        }
    )

    result = _run_fake_proof(env, output)

    assert result.returncode != 0
    assert f"could not remove orphaned model-proof container {orphan_id}" in result.stderr
    docker_lines = docker_log.read_text(encoding="utf-8").splitlines()
    assert not any(" --inner " in f" {line} " for line in docker_lines)


def test_model_proof_serializes_image_setup_and_uses_the_verified_image_id() -> None:
    ensure = IMAGE_ENSURE.read_text(encoding="utf-8")
    workflow = PROOF_WORKFLOW.read_text(encoding="utf-8")

    for contract in (
        "TRTMC_CI_IMAGE_LOCK_FILE",
        'flock -w "$lock_timeout" 9',
        "docker image inspect --format '{{.Id}}'",
        'printf \'image_ref=%s\\n\' "$resolved_ref" >> "$GITHUB_OUTPUT"',
        'mkdir -p "$(dirname "$GITHUB_OUTPUT")"',
        "verification_stamp",
    ):
        assert contract in ensure
    assert "id: ci_image" in workflow
    assert "timeout-minutes: 90" in workflow
    assert "TRTMC_CI_IMAGE: ${{ steps.ci_image.outputs.image_ref }}" in workflow


def test_model_proof_job_budget_reserves_singleton_finalization() -> None:
    workflow = PROOF_WORKFLOW.read_text(encoding="utf-8")
    job_configuration = workflow.split("jobs:\n  prove:", maxsplit=1)[1].split(
        "\n    steps:", maxsplit=1
    )[0]
    proof = workflow.split("- name: Run isolated model proof", maxsplit=1)[1].split(
        "- name: Finalize model proof fallback", maxsplit=1
    )[0]

    assert (
        "timeout-minutes: ${{ inputs.suite == 'nightly' && 480 || 300 }}"
        in job_configuration
    )
    assert "timeout-minutes: ${{ inputs.suite == 'nightly' && 360 || 150 }}" in proof
    assert "inputs.suite == 'nightly' && '5400' || '3600'" in job_configuration
    assert "inputs.suite == 'nightly' && '600' || '360'" in job_configuration
    assert "inputs.expected_count" not in workflow

    nightly_job_minutes = 480
    image_minutes = 90
    nightly_proof_minutes = 360
    finalization_margin_minutes = 30
    lease_minutes = 5400 // 60
    sana_build_minutes = json.loads(
        (
            REPO_ROOT
            / "tests/e2e/models/sana_wm/manifests/sana-wm-bidirectional.json"
        ).read_text(encoding="utf-8")
    )["build_timeout_s"] // 60
    e2e_and_report_margin_minutes = 150
    assert nightly_proof_minutes >= (
        lease_minutes + sana_build_minutes + e2e_and_report_margin_minutes
    )
    assert nightly_job_minutes >= (
        image_minutes + nightly_proof_minutes + finalization_margin_minutes
    )
    assert nightly_proof_minutes <= 360


def test_model_proof_uses_a_dedicated_self_hosted_checkout() -> None:
    workflow = PROOF_WORKFLOW.read_text(encoding="utf-8")
    checkout = workflow.split("- name: Check out exact source revision", maxsplit=1)[1].split(
        "- name: Ensure CI Docker image", maxsplit=1
    )[0]

    assert "path: model-proof-source" in checkout
    assert "clean: true" in checkout
    assert "persist-credentials: false" in checkout
    assert workflow.count("working-directory: ${{ github.workspace }}/model-proof-source") == 3


def test_model_proof_bootstraps_html_without_a_checkout_dependency() -> None:
    workflow = PROOF_WORKFLOW.read_text(encoding="utf-8")
    bootstrap = workflow.split("- name: Bootstrap model HTML before checkout", maxsplit=1)[1].split(
        "- name: Check model proof disk headroom", maxsplit=1
    )[0]
    checkout_failure = workflow.split("- name: Finalize model proof fallback", maxsplit=1)[1].split(
        "- name: Upload isolated model proof artifact", maxsplit=1
    )[0]

    assert workflow.index("Bootstrap model HTML before checkout") < workflow.index(
        "Check out exact source revision"
    )
    assert "model-proof-report.html" in bootstrap
    assert "model-proof-status.json" in bootstrap
    assert "working-directory:" not in bootstrap
    assert ".github/scripts/" not in bootstrap
    assert "if: ${{ always() }}" in checkout_failure
    assert "CHECKOUT_OUTCOME: ${{ steps.checkout.outcome }}" in checkout_failure
    assert "model-proof-report.html" in checkout_failure
    assert "working-directory:" not in checkout_failure
    assert "write-model-proof-fallback-report.py" in checkout_failure


def _write_certified_singleton_artifacts(
    root: Path,
    *,
    model: str = "alpha",
    revision: str = "a" * 40,
) -> None:
    root.mkdir(parents=True)
    (root / "model-proof-status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": model,
                "source_revision": revision,
                "suite": "premerge",
                "outcome": "passed",
            }
        ),
        encoding="utf-8",
    )
    (root / "proof.json").write_text(
        json.dumps(
            {
                "model": model,
                "source_revision": revision,
                "passed": True,
            }
        ),
        encoding="utf-8",
    )
    (root / "selection.json").write_text(json.dumps({"requested_model": model}), encoding="utf-8")
    (root / "model-proof-report.html").write_text(
        "<!doctype html><title>complete proof</title>", encoding="utf-8"
    )


def _run_workflow_singleton_gate(
    root: Path,
    *,
    model: str = "alpha",
    revision: str = "a" * 40,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _workflow_singleton_gate_program(),
            str(root),
            model,
            revision,
            "premerge",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_workflow_singleton_gate_accepts_complete_certification(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_certified_singleton_artifacts(artifacts)

    result = _run_workflow_singleton_gate(artifacts)

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("filename", "field", "value", "message"),
    [
        ("model-proof-status.json", "outcome", "failed", "status is not"),
        ("proof.json", "passed", False, "passed=true"),
        ("selection.json", "requested_model", "beta", "requested model"),
    ],
)
def test_workflow_singleton_gate_rejects_invalid_certification(
    tmp_path: Path,
    filename: str,
    field: str,
    value: object,
    message: str,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_certified_singleton_artifacts(artifacts)
    path = artifacts / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_workflow_singleton_gate(artifacts)

    assert result.returncode == 1
    assert message in result.stderr


def test_model_proof_resolves_runner_temp_only_after_runner_assignment() -> None:
    workflow = PROOF_WORKFLOW.read_text(encoding="utf-8")
    job_configuration = workflow.split("\n    steps:", maxsplit=1)[0]
    bootstrap = workflow.split("- name: Bootstrap model HTML before checkout", maxsplit=1)[1].split(
        "- name: Check model proof disk headroom", maxsplit=1
    )[0]
    proof = workflow.split("- name: Run isolated model proof", maxsplit=1)[1].split(
        "- name: Finalize model proof fallback", maxsplit=1
    )[0]

    assert "MODEL_PROOF_OUTPUT_DIR:" not in job_configuration
    assert "MODEL_PROOF_OUTPUT_DIR: ${{ runner.temp }}" in bootstrap
    assert 'echo "MODEL_PROOF_OUTPUT_DIR=$MODEL_PROOF_OUTPUT_DIR" >> "$GITHUB_ENV"' in bootstrap
    assert "${{ env.MODEL_PROOF_OUTPUT_DIR }}" not in proof
    assert '--output-dir "$MODEL_PROOF_OUTPUT_DIR"' in proof


def test_model_proof_checks_disk_headroom_before_checkout() -> None:
    workflow = PROOF_WORKFLOW.read_text(encoding="utf-8")
    disk_check = workflow.split("- name: Check model proof disk headroom", maxsplit=1)[1].split(
        "- name: Check out exact source revision", maxsplit=1
    )[0]

    assert workflow.index("Check model proof disk headroom") < workflow.index(
        "Check out exact source revision"
    )
    assert "TRTMC_MODEL_PROOF_MIN_FREE_GIB:" in workflow
    assert "TRTMC_MODEL_PROOF_STALE_MINUTES:" in workflow
    assert '-mmin "+$TRTMC_MODEL_PROOF_STALE_MINUTES"' in disk_check
    assert "-name work -o -name projection" in disk_check
    assert "-exec rm -rf -- {} +" in disk_check
    assert 'proof_capacity="$((${#gpu_ids[@]} * TRTMC_MODEL_PROOF_SLOTS_PER_GPU))"' in disk_check
    assert 'required_gib="$((TRTMC_MODEL_PROOF_MIN_FREE_GIB * proof_capacity))"' in disk_check
    assert 'required_kib="$((required_gib * 1024 * 1024))"' in disk_check
    assert 'df -Pk "$RUNNER_TEMP"' in disk_check
    assert "Insufficient model-proof disk headroom" in disk_check


def test_model_proof_uploads_before_singleton_gate_and_cleanup() -> None:
    workflow = PROOF_WORKFLOW.read_text(encoding="utf-8")
    gate = workflow.split("- name: Enforce isolated model certification", maxsplit=1)[1].split(
        "- name: Clean model proof scratch space", maxsplit=1
    )[0]
    cleanup = workflow.split("- name: Clean model proof scratch space", maxsplit=1)[1]

    assert workflow.index("Upload isolated model proof artifact") < workflow.index(
        "Enforce isolated model certification"
    )
    assert workflow.index("Enforce isolated model certification") < workflow.index(
        "Clean model proof scratch space"
    )
    assert "if: ${{ always() }}" in gate
    assert "id: artifact_upload" in workflow
    assert '"$RUNNER_TEMP"/model-proof-*' in cleanup
    assert "-name work -o -name projection" in cleanup
    assert 'ARTIFACT_UPLOAD_OUTCOME" = "success"' in cleanup
    assert 'rm -rf -- "$MODEL_PROOF_OUTPUT_DIR"' in cleanup


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
    assert "Upload isolated model proof artifact" in workflow
    assert "Bootstrap model HTML before checkout" in workflow
    assert "Finalize model proof fallback" in workflow
    assert "ci-image.log" in workflow
    assert "model-proof-report.html" in workflow
    assert "model-proof-index.html" not in workflow
    assert "if-no-files-found: error" in workflow


def test_model_proof_fallback_step_has_valid_shell_heredocs() -> None:
    workflow = yaml.safe_load(PROOF_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["prove"]["steps"]
    script = next(
        step["run"] for step in steps if step.get("name") == "Finalize model proof fallback"
    )

    result = subprocess.run(
        ["bash", "-n"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_model_proof_enforces_one_full_bundle_build_per_selected_model() -> None:
    runner = RUNNER.read_text(encoding="utf-8")

    for contract in (
        "TRTMC_ENGINE_BUILD_GUARD_DIR=/artifacts/engine-builds",
        'TRTMC_ENGINE_BUILD_REVISION="$revision"',
        "model_plugin_isolation.py verify-builds",
        "--ledger-dir /artifacts/engine-builds",
        '--source-revision "$revision"',
        "--report /artifacts/engine-build-verification.json",
        "update_proof_step engine_build_budget passed",
        '"engine_builds_per_model": build_verification["builds_per_model"]',
        '"engine_build_count": len(build_verification["records"])',
    ):
        assert contract in runner

    assert runner.index("model_plugin_isolation.py verify-results") < runner.index(
        "model_plugin_isolation.py verify-builds"
    )
    assert runner.index("model_plugin_isolation.py verify-results") < runner.index(
        'update_proof_step e2e_reference passed'
    )
    assert '"$py" -m pytest "$e2e_test" -v -rs' in runner


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
            "--artifacts-dir",
            str(artifacts),
            "--model",
            "missing-model",
            "--revision",
            "a" * 40,
            "--suite",
            "premerge",
            "--outcome",
            "failed",
            "--phase",
            "host-setup",
            "--exit-code",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = (artifacts / "model-proof-report.html").read_text(encoding="utf-8")
    status = json.loads((artifacts / "model-proof-status.json").read_text(encoding="utf-8"))
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
            "bash",
            str(RUNNER),
            "--model",
            "model-that-does-not-exist",
            "--revision",
            "HEAD",
            "--output-dir",
            str(output),
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
    status = json.loads((artifacts / "model-proof-status.json").read_text(encoding="utf-8"))
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
        'printf \'%s\\n\' "$*" >> "$DOCKER_LOG"\n'
        'if [ "${1:-}" = image ] || [ "${1:-}" = rm ]; then exit 0; fi\n'
        'if [ "${1:-}" = run ]; then exit 23; fi\n'
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
            "bash",
            str(RUNNER),
            "--model",
            "convbert",
            "--revision",
            "HEAD",
            "--output-dir",
            str(output),
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
        line
        for line in docker_log.read_text(encoding="utf-8").splitlines()
        if line.startswith("run ")
    ]
    assert len(docker_runs) == 1
    assert "scripts/warm_hf_cache.py" in docker_runs[0]
    assert "--local-only" in docker_runs[0]
    assert "--strict" in docker_runs[0]
    assert "--network none" in docker_runs[0]


def test_host_cache_uses_full_hub_only_for_check_and_positive_view_for_proof() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    host = text.split("run_host() {", maxsplit=1)[1]
    cache_check = host.split("local -a cache_check_docker_args=(", maxsplit=1)[1].split(
        "set +e", maxsplit=1
    )[0]
    proof = host.split("local -a docker_args=(", maxsplit=1)[1].split("set +e", maxsplit=1)[0]

    assert '[ -d "$hf_hub_cache" ]' not in host
    assert "HF Hub cache directory does not exist" not in host
    assert '[ "$hf_hub_cache" != "/" ]' in host
    assert "hf_modules_cache" not in host
    assert '--mount "type=bind,src=$hf_hub_cache,dst=/hf-cache/hub,readonly"' in cache_check
    assert "dst=/hf-cache/modules" not in cache_check
    assert '--mount "type=bind,src=$hf_private_hub,dst=/hf-cache/hub"' in proof
    assert "src=$hf_private_hub,dst=/hf-cache/hub,readonly" not in proof
    assert "hf_repo_mount_args" not in host
    assert "src=$hf_hub_cache,dst=/hf-cache/hub" not in proof
    assert "dst=/hf-cache/modules" not in proof
    assert "cp -a --reflink=always --no-preserve=ownership --" in host


def test_selected_hf_cache_reflink_helper_has_a_minimal_mount_and_capability_boundary() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    helper = text.split("local -a cache_copy_docker_args=(", maxsplit=1)[1].split(
        "select_proof_gpu", maxsplit=1
    )[0]

    for contract in (
        "--read-only",
        "--network none",
        "--cap-drop ALL",
        "--cap-add DAC_OVERRIDE",
        "--cap-add CHOWN",
        "--security-opt no-new-privileges",
        "--pids-limit 32",
        "--user 0:0",
        "type=bind,src=$hf_repo_source,dst=/selected-hf-repo,readonly",
        "type=bind,src=$hf_repo_destination,dst=/private-hf-repo",
        "--entrypoint /bin/bash",
        "cp -a --reflink=always --no-preserve=ownership --",
        'chown -hR -- "$runner_owner" /private-hf-repo',
    ):
        assert contract in helper
    assert helper.count("--cap-add") == 2
    assert "dst=/selected-hf-repo,readonly" in helper
    assert "dst=/private-hf-repo,readonly" not in helper
    assert "src=$hf_hub_cache" not in helper
    assert "src=$hf_private_hub" not in helper
    assert "src=$projection_dir" not in helper
    assert "src=$artifacts_dir" not in helper
    assert "--gpus" not in helper
    assert "HF_TOKEN" not in helper
    assert "/var/run/docker.sock" not in helper
    assert "if ! cp -a" not in helper


def test_sana_reference_cache_is_copied_to_selected_private_view(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    cache_root, source = _write_fake_pinned_model_reference(tmp_path, fake_bin)
    config = tmp_path / "model-proof-config.txt"
    _write_sana_reference_config(config)
    work_dir = tmp_path / "work"
    artifacts = tmp_path / "artifacts"
    work_dir.mkdir()
    artifacts.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "TRTMC_MODEL_REFERENCE_CACHE_ROOT": str(cache_root),
            "FAKE_REFERENCE_SOURCE": str(source),
            "FAKE_REFERENCE_REVISION": SANA_REFERENCE_REVISION,
        }
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            _prepare_reference_program(),
            "--",
            str(config),
            str(work_dir),
            str(artifacts),
            str(REPO_ROOT),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    private_root = work_dir / "reference-private"
    private_reference = private_root / SANA_REFERENCE_RELATIVE_PATH
    assert (private_reference / SANA_REFERENCE_ENTRYPOINT).is_file()
    assert (private_reference / "reference-marker.txt").read_text(encoding="utf-8") == (
        "selected only\n"
    )
    assert not any(path.name == ".git" for path in private_root.rglob(".git"))
    assert {path.name for path in private_root.iterdir()} == {"sana_wm"}

    evidence = json.loads((artifacts / "model-reference-cache.json").read_text(encoding="utf-8"))
    assert evidence == {
        "schema_version": 1,
        "model": "sana_wm",
        "isolation": "selected-pinned-private",
        "repository": "https://github.com/NVlabs/Sana.git",
        "reference_revision": SANA_REFERENCE_REVISION,
        "reference_tree": "1" * 40,
        "relative_path": SANA_REFERENCE_RELATIVE_PATH,
        "entrypoint": SANA_REFERENCE_ENTRYPOINT,
        "container_storage_root": "/work/reference-private",
        "copy_method": "git-archive",
    }


def test_sana_reference_cache_missing_checkout_fails_before_docker(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "model-reference-cache"
    cache_root.mkdir()
    config = tmp_path / "model-proof-config.txt"
    _write_sana_reference_config(config)
    work_dir = tmp_path / "work"
    artifacts = tmp_path / "artifacts"
    work_dir.mkdir()
    artifacts.mkdir()
    env = os.environ.copy()
    env["TRTMC_MODEL_REFERENCE_CACHE_ROOT"] = str(cache_root)

    result = subprocess.run(
        [
            "bash",
            "-c",
            _prepare_reference_program(),
            "--",
            str(config),
            str(work_dir),
            str(artifacts),
            str(REPO_ROOT),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "selected model reference cache is unavailable" in result.stderr


def test_sana_reference_cache_wrong_revision_fails_before_docker(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    cache_root, source = _write_fake_pinned_model_reference(tmp_path, fake_bin)
    config = tmp_path / "model-proof-config.txt"
    _write_sana_reference_config(config)
    work_dir = tmp_path / "work"
    artifacts = tmp_path / "artifacts"
    work_dir.mkdir()
    artifacts.mkdir()
    env = os.environ.copy()
    wrong_revision = "0" * 40
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "TRTMC_MODEL_REFERENCE_CACHE_ROOT": str(cache_root),
            "FAKE_REFERENCE_SOURCE": str(source),
            "FAKE_REFERENCE_REVISION": wrong_revision,
        }
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            _prepare_reference_program(),
            "--",
            str(config),
            str(work_dir),
            str(artifacts),
            str(REPO_ROOT),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert (
        "selected model reference cache revision mismatch: "
        f"expected {SANA_REFERENCE_REVISION}, found {wrong_revision}"
    ) in result.stderr


def test_distinct_explicit_hf_cache_paths_reach_both_containers(
    tmp_path: Path,
) -> None:
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    hub_cache = tmp_path / "explicit-hub-cache"
    modules_cache = tmp_path / "explicit-modules-cache"
    selected_repo = hub_cache / "models--fixture--model"
    selected_repo.mkdir(parents=True)
    (selected_repo / "selected-cache-marker").write_text("selected\n", encoding="utf-8")
    modules_cache.mkdir()
    env.update(
        {
            "TRTMC_HF_HUB_CACHE": str(hub_cache),
            "TRTMC_HF_MODULES_CACHE": str(modules_cache),
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--model",
            "convbert",
            "--revision",
            "HEAD",
            "--output-dir",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    docker_runs = [
        line
        for line in docker_log.read_text(encoding="utf-8").splitlines()
        if line.startswith("run ")
    ]
    assert len(docker_runs) == 3
    warm, cache_copy, proof = docker_runs
    assert f"--mount type=bind,src={hub_cache},dst=/hf-cache/hub,readonly" in warm
    assert f"src={modules_cache}" not in warm
    assert f"--mount type=bind,src={selected_repo},dst=/selected-hf-repo,readonly" in cache_copy
    private_repo = output / "work" / "hf-private" / "hub" / "models--fixture--model"
    assert f"--mount type=bind,src={private_repo},dst=/private-hf-repo" in cache_copy
    assert "--user 0:0" in cache_copy
    assert "--cap-drop ALL --cap-add DAC_OVERRIDE --cap-add CHOWN" in cache_copy
    assert "--network none" in cache_copy
    assert f"src={hub_cache},dst=/hf-cache/hub" not in cache_copy
    assert f"src={modules_cache}" not in cache_copy
    assert f"src={hub_cache},dst=/hf-cache/hub" not in proof
    assert f"src={selected_repo}" not in proof
    assert f"src={modules_cache}" not in proof
    private_hub = output / "work" / "hf-private" / "hub"
    assert f"--mount type=bind,src={private_hub},dst=/hf-cache/hub" in proof
    assert f"src={private_hub},dst=/hf-cache/hub,readonly" not in proof
    assert (private_repo / "selected-cache-marker").read_text(encoding="utf-8") == "selected\n"
    write_probe = private_repo / "test-write-probe"
    write_probe.write_text("writable\n", encoding="utf-8")
    assert write_probe.read_text(encoding="utf-8") == "writable\n"


def test_selected_hf_cache_with_unreadable_file_is_delegated_to_root_helper(
    tmp_path: Path,
) -> None:
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    selected_repo = tmp_path / "hf-cache" / "hub" / "models--fixture--model"
    unreadable = selected_repo / "unreadable-cache-marker"
    unreadable.write_text("persistent cache\n", encoding="utf-8")
    unreadable.chmod(0)
    # Root can read mode-000 files through DAC_OVERRIDE, so assert the fixture's
    # permissions directly. The proof below still verifies that the dedicated
    # root helper receives and reflinks this exact repository.
    assert unreadable.stat().st_mode & 0o777 == 0
    env.update(
        {
            "FAKE_UNREADABLE_REFLINK": "1",
            "FAKE_UNREADABLE_REFLINK_CONTENT": "private reflink",
        }
    )

    try:
        result = subprocess.run(
            [
                "bash",
                str(RUNNER),
                "--model",
                "convbert",
                "--revision",
                "HEAD",
                "--output-dir",
                str(output),
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        source_mode = unreadable.stat().st_mode & 0o777
    finally:
        unreadable.chmod(0o600)

    assert result.returncode == 0, result.stdout + result.stderr
    assert source_mode == 0
    assert unreadable.read_text(encoding="utf-8") == "persistent cache\n"
    private_repo = output / "work" / "hf-private" / "hub" / "models--fixture--model"
    assert (private_repo / "unreadable-cache-marker").read_text(encoding="utf-8") == (
        "private reflink\n"
    )
    docker_runs = [
        line
        for line in docker_log.read_text(encoding="utf-8").splitlines()
        if line.startswith("run ")
    ]
    assert len(docker_runs) == 3
    _warm, cache_copy, proof = docker_runs
    assert f"--mount type=bind,src={selected_repo},dst=/selected-hf-repo,readonly" in cache_copy
    assert "--user 0:0" in cache_copy
    assert "--cap-add DAC_OVERRIDE" in cache_copy
    assert f"src={selected_repo}" not in proof


def test_selected_hf_cache_fails_closed_when_reflink_is_unavailable(
    tmp_path: Path,
) -> None:
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    env["FAKE_REFLINK_EXIT_CODE"] = "73"

    result = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--model",
            "convbert",
            "--revision",
            "HEAD",
            "--output-dir",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "selected Hugging Face cache repository could not be reflinked" in result.stderr
    docker_text = docker_log.read_text(encoding="utf-8")
    assert "cp -a --reflink=always --no-preserve=ownership --" in docker_text
    docker_runs = [line for line in docker_text.splitlines() if line.startswith("run ")]
    assert len(docker_runs) == 2
    assert "warm_hf_cache.py" in docker_runs[0]
    assert "dst=/selected-hf-repo,readonly" in docker_runs[1]
    assert "dst=/private-hf-repo" in docker_runs[1]
    assert "--cap-drop ALL --cap-add DAC_OVERRIDE --cap-add CHOWN" in docker_runs[1]
    assert " --inner " not in f" {docker_text} "


def test_selected_hf_cache_evidence_rejects_path_escape_before_proof(
    tmp_path: Path,
) -> None:
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    env["FAKE_CACHE_EVIDENCE_MODE"] = "escape"

    result = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--model",
            "convbert",
            "--revision",
            "HEAD",
            "--output-dir",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "cache evidence failed closed validation" in result.stderr
    docker_runs = [
        line
        for line in docker_log.read_text(encoding="utf-8").splitlines()
        if line.startswith("run ")
    ]
    assert len(docker_runs) == 1
    assert "warm_hf_cache.py" in docker_runs[0]


def test_docker_bind_mount_fails_closed_when_host_cache_source_is_absent(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\n\' "$*" >> "$DOCKER_LOG"\n'
        'if [ "${1:-}" = image ] || [ "${1:-}" = rm ]; then exit 0; fi\n'
        'if [ "${1:-}" = run ]; then exit 23; fi\n'
        "exit 99\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    hub_cache = tmp_path / "missing-hub-cache"
    assert not hub_cache.exists()
    output = tmp_path / "proof"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "DOCKER_LOG": str(docker_log),
            "TRTMC_HF_HUB_CACHE": str(hub_cache),
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--model",
            "convbert",
            "--revision",
            "HEAD",
            "--output-dir",
            str(output),
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
        line
        for line in docker_log.read_text(encoding="utf-8").splitlines()
        if line.startswith("run ")
    ]
    assert len(docker_runs) == 1
    assert f"--mount type=bind,src={hub_cache},dst=/hf-cache/hub,readonly" in docker_runs[0]
    assert "dst=/hf-cache/modules" not in docker_runs[0]
    assert "--network none" in docker_runs[0]


def test_explicit_runner_gpu_id_still_acquires_a_slot_lease(tmp_path: Path) -> None:
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    env.update(
        {
            "TRTMC_GPU_ID": "7",
            "TRTMC_MODEL_PROOF_GPU_IDS": "7",
            "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "4",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--model",
            "convbert",
            "--revision",
            "HEAD",
            "--output-dir",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Leased shared model-proof GPU 7 slot 0" in result.stdout
    assert _proof_gpu_ids(docker_log) == ["7"]
    assert (output / "artifacts" / "gpu-id.txt").read_text().strip() == "7"
    assert _gpu_lease(output) == {
        "schema_version": 1,
        "model": "convbert",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "gpu_id": "7",
        "gpu_slot": 0,
        "gpu_slots": [0],
        "gpu_slot_ids": [0],
        "slots_per_gpu": 4,
        "gpu_slots_per_device": 4,
        "resource_class": "shared",
        "gpu_resource_class": "shared",
    }
    assert (tmp_path / "gpu-locks" / "gpu-7-slot-0.lock").is_file()


def test_explicit_runner_gpu_id_cannot_bypass_a_busy_slot(tmp_path: Path) -> None:
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    env.update(
        {
            "TRTMC_GPU_ID": "7",
            "TRTMC_MODEL_PROOF_GPU_IDS": "7",
            "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "1",
            "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": "1",
        }
    )
    lock_dir = tmp_path / "gpu-locks"
    lock_dir.mkdir()
    with (lock_dir / "gpu-7-slot-0.lock").open("w", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            [
                "bash",
                str(RUNNER),
                "--model",
                "convbert",
                "--revision",
                "HEAD",
                "--output-dir",
                str(output),
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            # Source projection and cache preparation happen before GPU
            # admission so queued jobs do not hold scarce slots during CPU
            # setup. Allow that setup to finish on a loaded CI host; the
            # in-runner lease timeout remains one second and is asserted below.
            timeout=60,
        )

    assert result.returncode != 0
    assert "waiting for a shared model-proof GPU lease from: 7" in result.stderr
    assert not _proof_gpu_ids_if_present(docker_log)


def test_model_proof_cannot_bypass_an_exclusive_whole_machine_lock(
    tmp_path: Path,
) -> None:
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    env.update(
        {
            "TRTMC_GPU_ID": "0",
            "TRTMC_MODEL_PROOF_GPU_IDS": "0",
            "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": "1",
        }
    )
    lock_dir = tmp_path / "gpu-locks"
    lock_dir.mkdir()
    with (lock_dir / "whole-machine.lock").open("w", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            [
                "bash",
                str(RUNNER),
                "--model",
                "convbert",
                "--revision",
                "HEAD",
                "--output-dir",
                str(output),
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

    assert result.returncode != 0
    assert "waiting for the whole-machine GPU lock" in result.stderr
    assert not _proof_gpu_ids_if_present(docker_log)


def test_explicit_runner_gpu_must_be_in_the_configured_allowlist(tmp_path: Path) -> None:
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    env.update({"TRTMC_GPU_ID": "7", "TRTMC_MODEL_PROOF_GPU_IDS": "0,1"})

    result = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--model",
            "convbert",
            "--revision",
            "HEAD",
            "--output-dir",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "TRTMC_GPU_ID must be present in TRTMC_MODEL_PROOF_GPU_IDS" in result.stderr
    assert not _proof_gpu_ids_if_present(docker_log)


@pytest.mark.model_proof_allocator
def test_four_shared_proofs_use_unique_slots_on_one_gpu(
    tmp_path: Path,
) -> None:
    processes: list[tuple[subprocess.Popen[str], Path, Path]] = []
    release_file = tmp_path / "release-four-shared"
    for index in range(4):
        case_dir = tmp_path / f"case-{index}"
        case_dir.mkdir()
        fake_bin, docker_log = _write_successful_fake_docker(case_dir)
        output = case_dir / "proof"
        env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
        env.update(
            {
                "TRTMC_MODEL_PROOF_GPU_IDS": "2",
                "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "4",
                "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": "180",
                "FAKE_PROOF_RELEASE_FILE": str(release_file),
            }
        )
        process = subprocess.Popen(
            [
                "bash",
                str(RUNNER),
                "--model",
                "convbert",
                "--revision",
                "HEAD",
                "--output-dir",
                str(output),
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append((process, docker_log, output))

    deadline = time.monotonic() + 90
    while time.monotonic() < deadline and not all(
        (output / "artifacts" / "gpu-lease.json").is_file() for _, _, output in processes
    ):
        time.sleep(0.05)
    all_leased_together = all(
        (output / "artifacts" / "gpu-lease.json").is_file() for _, _, output in processes
    )
    release_file.touch()

    selected: list[str] = []
    selected_slots: list[int] = []
    for process, docker_log, output in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stdout + stderr
        selected.extend(_proof_gpu_ids(docker_log))
        assert (output / "artifacts" / "gpu-id.txt").is_file()
        lease = _gpu_lease(output)
        selected_slots.extend(lease["gpu_slots"])
        assert lease["resource_class"] == "shared"
        assert lease["slots_per_gpu"] == 4

    assert all_leased_together
    assert selected == ["2", "2", "2", "2"]
    assert sorted(selected_slots) == [0, 1, 2, 3]


@pytest.mark.model_proof_allocator
def test_shared_slot_allocator_spreads_across_gpus_before_using_second_slots(
    tmp_path: Path,
) -> None:
    processes: list[tuple[subprocess.Popen[str], Path]] = []
    release_file = tmp_path / "release-spread"
    for index in range(4):
        case_dir = tmp_path / f"spread-{index}"
        case_dir.mkdir()
        fake_bin, docker_log = _write_successful_fake_docker(case_dir)
        output = case_dir / "proof"
        env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
        env.update(
            {
                "TRTMC_MODEL_PROOF_GPU_IDS": "2,3",
                "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "2",
                "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": "180",
                "FAKE_PROOF_RELEASE_FILE": str(release_file),
            }
        )
        process = subprocess.Popen(
            [
                "bash",
                str(RUNNER),
                "--model",
                "convbert",
                "--revision",
                "HEAD",
                "--output-dir",
                str(output),
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append((process, output))

    deadline = time.monotonic() + 90
    while time.monotonic() < deadline and not all(
        (output / "artifacts" / "gpu-lease.json").is_file() for _, output in processes
    ):
        time.sleep(0.05)
    all_leased_together = all(
        (output / "artifacts" / "gpu-lease.json").is_file() for _, output in processes
    )
    release_file.touch()

    assignments: set[tuple[str, int]] = set()
    for process, output in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stdout + stderr
        lease = _gpu_lease(output)
        assignments.add((lease["gpu_id"], lease["gpu_slot_ids"][0]))

    assert all_leased_together
    assert assignments == {("2", 0), ("3", 0), ("2", 1), ("3", 1)}


def test_automatic_gpu_lease_rejects_invalid_id_configuration(
    tmp_path: Path,
) -> None:
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    env["TRTMC_MODEL_PROOF_GPU_IDS"] = "0,,1"

    result = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--model",
            "convbert",
            "--revision",
            "HEAD",
            "--output-dir",
            str(output),
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


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("TRTMC_MODEL_PROOF_SLOTS_PER_GPU", "0", "must be an integer from 1 to 16"),
        ("TRTMC_MODEL_PROOF_SLOTS_PER_GPU", "17", "must be an integer from 1 to 16"),
        (
            "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS",
            "21601",
            "must be an integer from 1 to 21600",
        ),
        (
            "TRTMC_MODEL_PROOF_POLL_INTERVAL",
            "9223372036854775808",
            "must be a positive number no greater than 21600 seconds",
        ),
        (
            "TRTMC_MODEL_PROOF_POLL_INTERVAL",
            "21600.1",
            "must be a positive number no greater than 21600 seconds",
        ),
        (
            "TRTMC_MODEL_PROOF_FLOCK_WATCHDOG_SECONDS",
            "9223372036854775808",
            "must be an integer from 1 to 21600",
        ),
        (
            "TRTMC_MODEL_PROOF_FLOCK_WATCHDOG_SECONDS",
            "21601",
            "must be an integer from 1 to 21600",
        ),
    ],
)
def test_gpu_lease_rejects_invalid_numeric_configuration(
    tmp_path: Path,
    name: str,
    value: str,
    message: str,
) -> None:
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    env[name] = value

    result = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--model",
            "convbert",
            "--revision",
            "HEAD",
            "--output-dir",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert message in result.stderr
    assert not _proof_gpu_ids_if_present(docker_log)


def test_expected_resource_class_must_match_projected_e2e_manifest(
    tmp_path: Path,
) -> None:
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    env["TRTMC_MODEL_PROOF_EXPECTED_RESOURCE_CLASS"] = "exclusive_gpu"

    result = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--model",
            "convbert",
            "--revision",
            "HEAD",
            "--output-dir",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "does not match selected E2E resource class shared" in result.stderr
    assert not _proof_gpu_ids_if_present(docker_log)


@pytest.mark.model_proof_allocator
def test_fifth_shared_proof_times_out_when_all_four_slots_are_busy(
    tmp_path: Path,
) -> None:
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    env.update(
        {
            "TRTMC_MODEL_PROOF_GPU_IDS": "9",
            "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "4",
            "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": "1",
        }
    )
    lock_dir = tmp_path / "gpu-locks"
    lock_dir.mkdir()
    lock_streams = [
        (lock_dir / f"gpu-9-slot-{slot}.lock").open("w", encoding="utf-8") for slot in range(4)
    ]
    try:
        for lock_stream in lock_streams:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            [
                "bash",
                str(RUNNER),
                "--model",
                "convbert",
                "--revision",
                "HEAD",
                "--output-dir",
                str(output),
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=90,
        )
    finally:
        for lock_stream in lock_streams:
            lock_stream.close()

    assert result.returncode != 0
    assert "timed out after 1s waiting for a shared model-proof GPU lease from: 9" in result.stderr
    assert not _proof_gpu_ids_if_present(docker_log)
    assert not list(lock_dir.glob("admission-global-*.lock"))


@pytest.mark.model_proof_allocator
def test_poll_interval_cannot_sleep_past_the_lease_timeout(tmp_path: Path) -> None:
    """A poll interval larger than the lease budget must not delay the timeout.

    The capacity-poll sleep is clamped to the remaining deadline; without the
    clamp a 300s interval would sleep far past a 1s lease timeout while still
    reporting "timed out after 1s". Measure from the queue-joined marker to the
    timeout error so loaded-runner host setup cannot weaken the assertion.
    """
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    env.update(
        {
            "TRTMC_MODEL_PROOF_GPU_IDS": "9",
            "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "1",
            "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": "1",
            "TRTMC_MODEL_PROOF_POLL_INTERVAL": "300",
        }
    )
    lock_dir = tmp_path / "gpu-locks"
    lock_dir.mkdir()
    with (lock_dir / "gpu-9-slot-0.lock").open("w", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process = subprocess.Popen(
            [
                "bash",
                str(RUNNER),
                "--model",
                "convbert",
                "--revision",
                "HEAD",
                "--output-dir",
                str(output),
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            joined_file = output / "artifacts" / "gpu-queue-joined.txt"
            join_deadline = time.monotonic() + 90
            while time.monotonic() < join_deadline and not joined_file.is_file():
                assert process.poll() is None
                time.sleep(0.05)
            assert joined_file.is_file()

            timeout_message = (
                "timed out after 1s waiting for a shared model-proof GPU lease from: 9"
            )
            error_file = output / "artifacts" / "host-error.log"
            started = time.monotonic()
            error_deadline = started + 10
            while time.monotonic() < error_deadline:
                if error_file.is_file() and timeout_message in error_file.read_text(
                    encoding="utf-8"
                ):
                    break
                time.sleep(0.05)
            lease_elapsed = time.monotonic() - started
            stdout, stderr = process.communicate(timeout=90)
        finally:
            _finish_proof_cases([process])

    assert process.returncode != 0, stdout + stderr
    assert timeout_message in stderr
    assert lease_elapsed < 10, (
        f"lease timeout was pierced by the poll interval: {lease_elapsed:.3f}s"
    )
    assert not list(lock_dir.glob("admission-global-*.lock"))


@pytest.mark.model_proof_allocator
def test_gpu_admission_queue_prunes_a_stale_ticket(tmp_path: Path) -> None:
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    env.update(
        {
            "TRTMC_GPU_ID": "9",
            "TRTMC_MODEL_PROOF_GPU_IDS": "9",
            "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "1",
        }
    )
    lock_dir = tmp_path / "gpu-locks"
    lock_dir.mkdir()
    stale_ticket = lock_dir / "admission-global-00000000000000000001.lock"
    stale_ticket.write_text("pid=999999 model=stale\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--model",
            "convbert",
            "--revision",
            "HEAD",
            "--output-dir",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not stale_ticket.exists()
    assert not list(lock_dir.glob("admission-global-*.lock"))
    assert (lock_dir / "admission-global.next").read_text(encoding="utf-8") == "2\n"
    assert _proof_gpu_ids(docker_log) == ["9"]


@pytest.mark.model_proof_allocator
def test_gpu_allocator_mutex_contention_obeys_lease_timeout(
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
    with (lock_dir / "allocator.lock").open("w", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process = subprocess.Popen(
            [
                "bash",
                str(RUNNER),
                "--model",
                "convbert",
                "--revision",
                "HEAD",
                "--output-dir",
                str(output),
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            requested = output / "artifacts" / "gpu-lease-requested.txt"
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline and not requested.is_file():
                if process.poll() is not None:
                    break
                time.sleep(0.05)
            assert requested.is_file(), "proof never reached GPU lease acquisition"
            started = time.monotonic()
            stdout, stderr = process.communicate(timeout=15)
            elapsed = time.monotonic() - started
        finally:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=10)

    assert process.returncode != 0, stdout + stderr
    assert elapsed < 5
    assert "timed out after 1s waiting for a shared model-proof GPU lease" in stderr
    assert not _proof_gpu_ids_if_present(docker_log)


@pytest.mark.model_proof_allocator
def test_exclusive_gpu_ticket_drains_existing_shared_and_blocks_new_shared(
    tmp_path: Path,
) -> None:
    lock_dir = tmp_path / "gpu-locks"
    first_release_file = tmp_path / "release-first-shared"

    first_dir = tmp_path / "first-shared"
    first_dir.mkdir()
    first_bin, first_log = _write_successful_fake_docker(first_dir)
    first_output = first_dir / "proof"
    first_env = _fake_proof_environment(tmp_path, first_bin, first_log, first_output)
    first_env.update(
        {
            "TRTMC_GPU_ID": "6",
            "TRTMC_GPU_SLOT_ID": "0",
            "TRTMC_MODEL_PROOF_GPU_IDS": "6",
            "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "2",
            "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": "10",
            "FAKE_PROOF_RELEASE_FILE": str(first_release_file),
        }
    )
    first = subprocess.Popen(
        [
            "bash",
            str(RUNNER),
            "--model",
            "convbert",
            "--revision",
            "HEAD",
            "--output-dir",
            str(first_output),
        ],
        cwd=REPO_ROOT,
        env=first_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    exclusive_dir = tmp_path / "exclusive"
    exclusive_dir.mkdir()
    exclusive_bin, exclusive_log = _write_successful_fake_docker(exclusive_dir)
    exclusive_output = exclusive_dir / "proof"
    exclusive_env = _fake_proof_environment(
        tmp_path, exclusive_bin, exclusive_log, exclusive_output
    )
    exclusive_env.update(
        {
            "TRTMC_GPU_ID": "6",
            "TRTMC_MODEL_PROOF_GPU_IDS": "6",
            "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "2",
            "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": "600",
        }
    )
    exclusive: subprocess.Popen[str] | None = None
    try:
        deadline = time.monotonic() + 90
        while (
            time.monotonic() < deadline
            and not (first_output / "artifacts" / "gpu-lease.json").is_file()
        ):
            time.sleep(0.05)
        assert (first_output / "artifacts" / "gpu-lease.json").is_file()

        exclusive = subprocess.Popen(
            [
                "bash",
                str(RUNNER),
                "--model",
                "flux",
                "--revision",
                "HEAD",
                "--output-dir",
                str(exclusive_output),
            ],
            cwd=REPO_ROOT,
            env=exclusive_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        reservation = lock_dir / "gpu-6-reservation.lock"
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and not list(
            lock_dir.glob("admission-global-*.lock")
        ):
            time.sleep(0.05)
        tickets = list(lock_dir.glob("admission-global-*.lock"))
        assert len(tickets) == 1
        assert "model=flux" in tickets[0].read_text(encoding="utf-8")
        assert _lock_is_busy(tickets[0])
        assert not _lock_is_busy(reservation)

        blocked_dir = tmp_path / "blocked-shared"
        blocked_dir.mkdir()
        blocked_bin, blocked_log = _write_successful_fake_docker(blocked_dir)
        blocked_output = blocked_dir / "proof"
        blocked_env = _fake_proof_environment(tmp_path, blocked_bin, blocked_log, blocked_output)
        blocked_env.update(
            {
                "TRTMC_GPU_ID": "6",
                "TRTMC_GPU_SLOT_ID": "1",
                "TRTMC_MODEL_PROOF_GPU_IDS": "6",
                "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "2",
                "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": "1",
            }
        )
        blocked = subprocess.run(
            [
                "bash",
                str(RUNNER),
                "--model",
                "convbert",
                "--revision",
                "HEAD",
                "--output-dir",
                str(blocked_output),
            ],
            cwd=REPO_ROOT,
            env=blocked_env,
            capture_output=True,
            text=True,
            check=False,
            timeout=90,
        )
        assert blocked.returncode != 0
        assert "waiting for a shared model-proof GPU lease" in blocked.stderr
        assert not _proof_gpu_ids_if_present(blocked_log)

        first_release_file.touch()
        first_stdout, first_stderr = first.communicate(timeout=30)
        assert first.returncode == 0, first_stdout + first_stderr
        exclusive_stdout, exclusive_stderr = exclusive.communicate(timeout=30)
        assert exclusive.returncode == 0, exclusive_stdout + exclusive_stderr
        assert _proof_gpu_ids(exclusive_log) == ["6"]
        assert _gpu_lease(exclusive_output)["resource_class"] == "exclusive_gpu"
        assert _gpu_lease(exclusive_output)["gpu_slots"] == [0, 1]
        assert not _lock_is_busy(reservation)
        assert not _lock_is_busy(lock_dir / "gpu-6-slot-0.lock")
        assert not _lock_is_busy(lock_dir / "gpu-6-slot-1.lock")
    finally:
        first_release_file.touch(exist_ok=True)
        if first.poll() is None:
            first.terminate()
            first.communicate(timeout=10)
        if exclusive is not None and exclusive.poll() is None:
            exclusive.terminate()
            exclusive.communicate(timeout=10)


@pytest.mark.model_proof_allocator
def test_oldest_exclusive_waiter_takes_the_first_idle_gpu(
    tmp_path: Path,
) -> None:
    lock_dir = tmp_path / "gpu-locks"
    coordination_timeout_s = 90
    processes: list[subprocess.Popen[str]] = []
    release_files: list[Path] = []

    def start_case(
        name: str,
        model: str,
        release_file: Path,
        *,
        explicit_gpu: str | None = None,
    ) -> tuple[subprocess.Popen[str], Path, Path]:
        case_dir = tmp_path / name
        case_dir.mkdir()
        fake_bin, docker_log = _write_successful_fake_docker(case_dir)
        output = case_dir / "proof"
        env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
        env.update(
            {
                "TRTMC_MODEL_PROOF_GPU_IDS": "6,7",
                "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "4",
                "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": "600",
                "FAKE_PROOF_RELEASE_FILE": str(release_file),
            }
        )
        if explicit_gpu is not None:
            env["TRTMC_GPU_ID"] = explicit_gpu
        process = subprocess.Popen(
            [
                "bash",
                str(RUNNER),
                "--model",
                model,
                "--revision",
                "HEAD",
                "--output-dir",
                str(output),
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(process)
        release_files.append(release_file)
        return process, docker_log, output

    def wait_for(predicate: Callable[[], bool], message: str) -> None:
        deadline = time.monotonic() + coordination_timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        raise AssertionError(message)

    gpu6_release = tmp_path / "release-gpu6-holder"
    gpu7_release = tmp_path / "release-gpu7-holder"
    exclusive_release = tmp_path / "release-exclusive"
    younger_release = tmp_path / "release-younger-shared"
    try:
        gpu6_holder, _, gpu6_output = start_case(
            "gpu6-holder", "convbert", gpu6_release, explicit_gpu="6"
        )
        wait_for(
            lambda: (gpu6_output / "artifacts" / "gpu-lease.json").is_file(),
            "GPU 6 holder never acquired its lease",
        )
        gpu7_holder, _, gpu7_output = start_case(
            "gpu7-holder", "m2m_100", gpu7_release, explicit_gpu="7"
        )
        wait_for(
            lambda: (gpu7_output / "artifacts" / "gpu-lease.json").is_file(),
            "GPU 7 holder never acquired its lease",
        )

        exclusive, exclusive_log, exclusive_output = start_case(
            "oldest-exclusive", "flux", exclusive_release
        )
        gpu6_reservation = lock_dir / "gpu-6-reservation.lock"
        wait_for(
            lambda: len(list(lock_dir.glob("admission-global-*.lock"))) == 1,
            "exclusive proof did not retain its admission ticket while draining",
        )
        exclusive_ticket = list(lock_dir.glob("admission-global-*.lock"))[0]
        assert "model=flux" in exclusive_ticket.read_text(encoding="utf-8")
        assert not (exclusive_output / "artifacts" / "gpu-lease.json").exists()
        assert not _lock_is_busy(gpu6_reservation)

        younger, younger_log, younger_output = start_case(
            "younger-shared", "convbert", younger_release, explicit_gpu="7"
        )
        wait_for(
            lambda: len(list(lock_dir.glob("admission-global-*.lock"))) == 2,
            "younger proof never entered the admission queue",
        )
        assert not (younger_output / "artifacts" / "gpu-lease.json").exists()

        # GPU 6 remains occupied, but GPU 7 becomes idle. The oldest exclusive
        # request must claim all four slots on GPU 7 before the younger shared
        # request can steal any of that newly available capacity.
        gpu7_release.touch()
        gpu7_stdout, gpu7_stderr = gpu7_holder.communicate(timeout=30)
        assert gpu7_holder.returncode == 0, gpu7_stdout + gpu7_stderr
        wait_for(
            lambda: (exclusive_output / "artifacts" / "gpu-lease.json").is_file(),
            "exclusive proof did not claim GPU 7 after it became idle",
        )
        exclusive_lease = _gpu_lease(exclusive_output)
        assert exclusive_lease["gpu_id"] == "7"
        assert exclusive_lease["gpu_slots"] == [0, 1, 2, 3]
        assert exclusive_lease["slots_per_gpu"] == 4
        assert not (younger_output / "artifacts" / "gpu-lease.json").exists()
        assert not _lock_is_busy(gpu6_reservation)
        assert _lock_is_busy(lock_dir / "gpu-7-reservation.lock")

        exclusive_release.touch()
        exclusive_stdout, exclusive_stderr = exclusive.communicate(timeout=30)
        assert exclusive.returncode == 0, exclusive_stdout + exclusive_stderr
        assert _proof_gpu_ids(exclusive_log) == ["7"]
        wait_for(
            lambda: (younger_output / "artifacts" / "gpu-lease.json").is_file(),
            "younger shared proof did not run after the exclusive proof",
        )
        assert _gpu_lease(younger_output)["gpu_id"] == "7"
        younger_release.touch()
        younger_stdout, younger_stderr = younger.communicate(timeout=30)
        assert younger.returncode == 0, younger_stdout + younger_stderr
        assert _proof_gpu_ids(younger_log) == ["7"]

        gpu6_release.touch()
        gpu6_stdout, gpu6_stderr = gpu6_holder.communicate(timeout=30)
        assert gpu6_holder.returncode == 0, gpu6_stdout + gpu6_stderr
        assert not list(lock_dir.glob("admission-global-*.lock"))
    finally:
        for release_file in release_files:
            release_file.touch(exist_ok=True)
        for process in processes:
            if process.poll() is None:
                try:
                    process.communicate(timeout=30)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.communicate(timeout=10)


@pytest.mark.model_proof_allocator
def test_gpu_admission_ticket_queue_prevents_younger_requests_overtaking_shared_waiter(
    tmp_path: Path,
) -> None:
    lock_dir = tmp_path / "gpu-locks"
    coordination_timeout_s = 90
    common_env = {
        "TRTMC_GPU_ID": "6",
        "TRTMC_MODEL_PROOF_GPU_IDS": "6",
        "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "1",
        # This test deliberately holds several waiters while it observes their
        # ordering.  The lease timeout must outlive the coordinated setup on
        # loaded CI hosts; otherwise an older ticket can expire before the last
        # waiter starts.
        "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": "600",
    }

    def start_case(
        name: str, model: str, release_file: Path
    ) -> tuple[subprocess.Popen[str], Path, Path]:
        case_dir = tmp_path / name
        case_dir.mkdir()
        fake_bin, docker_log = _write_successful_fake_docker(case_dir)
        output = case_dir / "proof"
        env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
        env.update(common_env)
        env["FAKE_PROOF_RELEASE_FILE"] = str(release_file)
        process = subprocess.Popen(
            [
                "bash",
                str(RUNNER),
                "--model",
                model,
                "--revision",
                "HEAD",
                "--output-dir",
                str(output),
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return process, docker_log, output

    first_release = tmp_path / "release-first-exclusive"
    oldest_release = tmp_path / "release-oldest-shared"
    younger_shared_release = tmp_path / "release-younger-shared"
    younger_exclusive_release = tmp_path / "release-younger-exclusive"
    first, _, first_output = start_case("first-exclusive", "flux", first_release)
    oldest: subprocess.Popen[str] | None = None
    younger_shared: subprocess.Popen[str] | None = None
    younger_exclusive: subprocess.Popen[str] | None = None
    try:
        deadline = time.monotonic() + coordination_timeout_s
        while (
            time.monotonic() < deadline
            and not (first_output / "artifacts" / "gpu-lease.json").is_file()
        ):
            time.sleep(0.05)
        assert (first_output / "artifacts" / "gpu-lease.json").is_file()

        oldest, _, oldest_output = start_case("oldest-shared", "m2m_100", oldest_release)
        deadline = time.monotonic() + coordination_timeout_s
        while time.monotonic() < deadline and not list(lock_dir.glob("admission-global-*.lock")):
            time.sleep(0.05)
        admission_tickets = sorted(lock_dir.glob("admission-global-*.lock"))
        assert len(admission_tickets) == 1
        assert "model=m2m_100" in admission_tickets[0].read_text(encoding="utf-8")
        assert _lock_is_busy(admission_tickets[0])

        younger_exclusive, _, younger_exclusive_output = start_case(
            "younger-exclusive", "bark", younger_exclusive_release
        )
        deadline = time.monotonic() + coordination_timeout_s
        while time.monotonic() < deadline:
            admission_tickets = sorted(lock_dir.glob("admission-global-*.lock"))
            if len(admission_tickets) == 2 and all(
                ticket.stat().st_size > 0 for ticket in admission_tickets
            ):
                break
            time.sleep(0.05)
        admission_tickets = sorted(lock_dir.glob("admission-global-*.lock"))
        assert len(admission_tickets) == 2
        assert [
            ticket.read_text(encoding="utf-8").split("model=", maxsplit=1)[1].split()[0]
            for ticket in admission_tickets
        ] == ["m2m_100", "bark"]
        younger_shared, _, younger_shared_output = start_case(
            "younger-shared", "convbert", younger_shared_release
        )
        deadline = time.monotonic() + coordination_timeout_s
        while time.monotonic() < deadline:
            admission_tickets = sorted(lock_dir.glob("admission-global-*.lock"))
            if len(admission_tickets) == 3 and all(
                ticket.stat().st_size > 0 for ticket in admission_tickets
            ):
                break
            time.sleep(0.05)
        admission_tickets = sorted(lock_dir.glob("admission-global-*.lock"))
        assert len(admission_tickets) == 3
        assert [
            ticket.read_text(encoding="utf-8").split("model=", maxsplit=1)[1].split()[0]
            for ticket in admission_tickets
        ] == ["m2m_100", "bark", "convbert"]

        first_release.touch()
        deadline = time.monotonic() + coordination_timeout_s
        while (
            time.monotonic() < deadline
            and not (oldest_output / "artifacts" / "gpu-lease.json").is_file()
            and not (younger_exclusive_output / "artifacts" / "gpu-lease.json").is_file()
            and not (younger_shared_output / "artifacts" / "gpu-lease.json").is_file()
        ):
            time.sleep(0.05)

        assert (oldest_output / "artifacts" / "gpu-lease.json").is_file()
        assert not (younger_exclusive_output / "artifacts" / "gpu-lease.json").exists()
        assert not (younger_shared_output / "artifacts" / "gpu-lease.json").exists()
        assert _gpu_lease(oldest_output)["resource_class"] == "shared"

        oldest_release.touch()
        deadline = time.monotonic() + coordination_timeout_s
        while (
            time.monotonic() < deadline
            and not (younger_exclusive_output / "artifacts" / "gpu-lease.json").is_file()
        ):
            time.sleep(0.05)
        assert (younger_exclusive_output / "artifacts" / "gpu-lease.json").is_file()
        assert not (younger_shared_output / "artifacts" / "gpu-lease.json").exists()
        assert _gpu_lease(younger_exclusive_output)["resource_class"] == "exclusive_gpu"

        younger_exclusive_release.touch()
        deadline = time.monotonic() + coordination_timeout_s
        while (
            time.monotonic() < deadline
            and not (younger_shared_output / "artifacts" / "gpu-lease.json").is_file()
        ):
            time.sleep(0.05)
        assert (younger_shared_output / "artifacts" / "gpu-lease.json").is_file()
        assert _gpu_lease(younger_shared_output)["resource_class"] == "shared"
        younger_shared_release.touch()

        first_stdout, first_stderr = first.communicate(timeout=30)
        assert first.returncode == 0, first_stdout + first_stderr
        oldest_stdout, oldest_stderr = oldest.communicate(timeout=30)
        assert oldest.returncode == 0, oldest_stdout + oldest_stderr
        younger_shared_stdout, younger_shared_stderr = younger_shared.communicate(timeout=30)
        assert younger_shared.returncode == 0, younger_shared_stdout + younger_shared_stderr
        younger_exclusive_stdout, younger_exclusive_stderr = younger_exclusive.communicate(
            timeout=30
        )
        assert younger_exclusive.returncode == 0, (
            younger_exclusive_stdout + younger_exclusive_stderr
        )
        assert not list(lock_dir.glob("admission-global-*.lock"))
    finally:
        for release_file in (
            first_release,
            oldest_release,
            younger_shared_release,
            younger_exclusive_release,
        ):
            release_file.touch()
        for process in (first, oldest, younger_shared, younger_exclusive):
            if process is not None and process.poll() is None:
                try:
                    process.communicate(timeout=30)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.communicate(timeout=10)


def _start_proof_case(
    tmp_path: Path,
    name: str,
    model: str,
    release_file: Path,
    common_env: dict[str, str],
) -> tuple[subprocess.Popen[str], Path]:
    case_dir = tmp_path / name
    case_dir.mkdir()
    fake_bin, docker_log = _write_successful_fake_docker(case_dir)
    output = case_dir / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    env.update(common_env)
    env["FAKE_PROOF_RELEASE_FILE"] = str(release_file)
    process = subprocess.Popen(
        [
            "bash",
            str(RUNNER),
            "--model",
            model,
            "--revision",
            "HEAD",
            "--output-dir",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return process, output


def _finish_proof_cases(processes: list[subprocess.Popen[str] | None]) -> None:
    for process in processes:
        if process is not None and process.poll() is None:
            try:
                process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.communicate(timeout=10)


@pytest.mark.model_proof_allocator
def test_gpu_lock_directory_file_protocol_is_frozen(tmp_path: Path) -> None:
    """The lock-directory layout and its live semantics are a contract.

    Old and new script revisions share one lock directory on a runner while
    premerge branches overlap, so the file names, the ticket flock semantics
    (a live ticket is exclusively flocked by its owner; the slot lease stays
    flocked while the proof runs), and the single-hard-link publication must
    never change in place; a protocol change requires a new lock-directory
    generation.
    """
    lock_dir = tmp_path / "gpu-locks"
    coordination_timeout_s = 90
    common_env = {
        "TRTMC_GPU_ID": "6",
        "TRTMC_MODEL_PROOF_GPU_IDS": "6",
        "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "1",
        "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": "600",
    }
    holder_release = tmp_path / "release-holder"
    waiter_release = tmp_path / "release-waiter"
    waiter_release.touch()
    holder, holder_output = _start_proof_case(
        tmp_path, "holder", "convbert", holder_release, common_env
    )
    waiter: subprocess.Popen[str] | None = None
    try:
        deadline = time.monotonic() + coordination_timeout_s
        while (
            time.monotonic() < deadline
            and not (holder_output / "artifacts" / "gpu-lease.json").is_file()
        ):
            time.sleep(0.05)
        assert (holder_output / "artifacts" / "gpu-lease.json").is_file()
        # A held lease is an exclusively flocked slot file.
        assert _lock_is_busy(lock_dir / "gpu-6-slot-0.lock")
        # Every model proof shares the machine-wide fence while it owns GPU work.
        assert _lock_is_busy(lock_dir / "whole-machine.lock")

        waiter, waiter_output = _start_proof_case(
            tmp_path, "waiter", "convbert", waiter_release, common_env
        )
        deadline = time.monotonic() + coordination_timeout_s
        while time.monotonic() < deadline and not list(lock_dir.glob("admission-global-*.lock")):
            time.sleep(0.05)
        tickets = sorted(lock_dir.glob("admission-global-*.lock"))
        assert len(tickets) == 1
        ticket = tickets[0]
        # Live-ticket semantics: exclusively flocked by its owner from the
        # moment it becomes visible.
        assert re.fullmatch(r"admission-global-[0-9]{20}\.lock", ticket.name)
        assert _lock_is_busy(ticket)
        # Publication settles to a single hard link with no temporary alias:
        # the ticket becomes visible via ln just before its .tmp source and
        # the counter .tmp are removed, so wait out that window first.
        joined_file = waiter_output / "artifacts" / "gpu-queue-joined.txt"
        deadline = time.monotonic() + coordination_timeout_s
        while time.monotonic() < deadline and (
            list(lock_dir.glob("*.tmp.*")) or not joined_file.is_file()
        ):
            time.sleep(0.05)
        assert not list(lock_dir.glob("*.tmp.*"))
        assert ticket.stat().st_nlink == 1
        assert joined_file.read_text(encoding="utf-8") == ticket.name + "\n"

        holder_release.touch()
        waiter_stdout, waiter_stderr = waiter.communicate(timeout=coordination_timeout_s)
        assert waiter.returncode == 0, waiter_stdout + waiter_stderr
        holder_stdout, holder_stderr = holder.communicate(timeout=30)
        assert holder.returncode == 0, holder_stdout + holder_stderr

        # End-state layout: exactly the frozen protocol files, nothing else.
        assert sorted(path.name for path in lock_dir.iterdir()) == [
            "admission-global.enqueue.lock",
            "admission-global.next",
            "allocator.lock",
            "gpu-6-reservation.lock",
            "gpu-6-slot-0.lock",
            "whole-machine.lock",
        ]
        # Both GPU fencing layers are released once no proof runs.
        assert not _lock_is_busy(lock_dir / "gpu-6-slot-0.lock")
        assert not _lock_is_busy(lock_dir / "whole-machine.lock")
    finally:
        holder_release.touch()
        _finish_proof_cases([holder, waiter])


@pytest.mark.model_proof_allocator
def test_killed_queue_predecessor_wakes_waiter_and_is_pruned(tmp_path: Path) -> None:
    lock_dir = tmp_path / "gpu-locks"
    coordination_timeout_s = 90
    common_env = {
        "TRTMC_GPU_ID": "6",
        "TRTMC_MODEL_PROOF_GPU_IDS": "6",
        "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "1",
        "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": "600",
    }
    holder_release = tmp_path / "release-holder"
    waiter_release = tmp_path / "release-waiter"
    waiter_release.touch()
    holder, holder_output = _start_proof_case(
        tmp_path, "holder", "m2m_100", holder_release, common_env
    )
    doomed: subprocess.Popen[str] | None = None
    waiter: subprocess.Popen[str] | None = None
    try:
        deadline = time.monotonic() + coordination_timeout_s
        while (
            time.monotonic() < deadline
            and not (holder_output / "artifacts" / "gpu-lease.json").is_file()
        ):
            time.sleep(0.05)
        assert (holder_output / "artifacts" / "gpu-lease.json").is_file()

        doomed, _ = _start_proof_case(tmp_path, "doomed", "convbert", waiter_release, common_env)
        deadline = time.monotonic() + coordination_timeout_s
        while (
            time.monotonic() < deadline and len(list(lock_dir.glob("admission-global-*.lock"))) < 1
        ):
            time.sleep(0.05)
        assert len(list(lock_dir.glob("admission-global-*.lock"))) == 1

        waiter, waiter_output = _start_proof_case(
            tmp_path, "waiter", "convbert", waiter_release, common_env
        )
        deadline = time.monotonic() + coordination_timeout_s
        while (
            time.monotonic() < deadline and len(list(lock_dir.glob("admission-global-*.lock"))) < 2
        ):
            time.sleep(0.05)
        assert len(list(lock_dir.glob("admission-global-*.lock"))) == 2

        # SIGKILL the middle of the chain: the kernel drops its ticket flock,
        # which must wake the waiter behind it; the waiter then prunes the
        # stale ticket and inherits the queue head.
        doomed.kill()
        doomed.communicate(timeout=30)
        holder_release.touch()

        waiter_stdout, waiter_stderr = waiter.communicate(timeout=coordination_timeout_s)
        assert waiter.returncode == 0, waiter_stdout + waiter_stderr
        assert (waiter_output / "artifacts" / "gpu-lease.json").is_file()
        holder_stdout, holder_stderr = holder.communicate(timeout=30)
        assert holder.returncode == 0, holder_stdout + holder_stderr
        assert not list(lock_dir.glob("admission-global-*.lock"))
    finally:
        holder_release.touch()
        _finish_proof_cases([holder, doomed, waiter])


@pytest.mark.model_proof_allocator
def test_exclusive_ticket_waits_for_all_holders_in_any_order(
    tmp_path: Path,
) -> None:
    lock_dir = tmp_path / "gpu-locks"
    coordination_timeout_s = 90
    common_env = {
        "TRTMC_GPU_ID": "6",
        "TRTMC_MODEL_PROOF_GPU_IDS": "6",
        "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "3",
        "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": "600",
    }
    releases = [tmp_path / f"release-shared-{index}" for index in range(3)]
    exclusive_release = tmp_path / "release-exclusive"
    exclusive_release.touch()
    holders: list[subprocess.Popen[str] | None] = [None, None, None]
    holder_outputs: list[Path] = []
    exclusive: subprocess.Popen[str] | None = None
    try:
        for index in range(3):
            process, output = _start_proof_case(
                tmp_path, f"shared-{index}", "m2m_100", releases[index], common_env
            )
            holders[index] = process
            holder_outputs.append(output)
            deadline = time.monotonic() + coordination_timeout_s
            while (
                time.monotonic() < deadline
                and not (output / "artifacts" / "gpu-lease.json").is_file()
            ):
                time.sleep(0.05)
            assert (output / "artifacts" / "gpu-lease.json").is_file()

        exclusive, exclusive_output = _start_proof_case(
            tmp_path, "exclusive", "bark", exclusive_release, common_env
        )
        # The exclusive proof keeps the oldest queue position without pinning
        # itself to a partially occupied GPU.
        deadline = time.monotonic() + coordination_timeout_s
        while time.monotonic() < deadline and not list(
            lock_dir.glob("admission-global-*.lock")
        ):
            time.sleep(0.05)
        tickets = list(lock_dir.glob("admission-global-*.lock"))
        assert len(tickets) == 1
        assert "model=bark" in tickets[0].read_text(encoding="utf-8")
        assert _lock_is_busy(tickets[0])
        assert not _lock_is_busy(lock_dir / "gpu-6-reservation.lock")
        assert not (exclusive_output / "artifacts" / "gpu-lease.json").exists()

        # Release holders out of order. The exclusive proof must not publish a
        # partial lease; it can proceed only after every slot is simultaneously
        # available for one atomic acquisition.
        for index in (1, 0, 2):
            releases[index].touch()
            holder = holders[index]
            assert holder is not None
            holder_stdout, holder_stderr = holder.communicate(timeout=coordination_timeout_s)
            assert holder.returncode == 0, f"shared-{index}: " + holder_stdout + holder_stderr
            if index != 2:
                assert not (exclusive_output / "artifacts" / "gpu-lease.json").exists()
                assert _lock_is_busy(tickets[0])
                assert not _lock_is_busy(lock_dir / "gpu-6-reservation.lock")

        exclusive_stdout, exclusive_stderr = exclusive.communicate(timeout=coordination_timeout_s)
        assert exclusive.returncode == 0, exclusive_stdout + exclusive_stderr
        lease = _gpu_lease(exclusive_output)
        assert lease["resource_class"] == "exclusive_gpu"
        assert sorted(lease["gpu_slots"]) == [0, 1, 2]
    finally:
        for release_file in releases:
            release_file.touch()
        _finish_proof_cases([*holders, exclusive])


@pytest.mark.model_proof_allocator
def test_long_queue_is_served_in_strict_fifo_order(tmp_path: Path) -> None:
    lock_dir = tmp_path / "gpu-locks"
    coordination_timeout_s = 90
    waiter_count = 6
    common_env = {
        "TRTMC_GPU_ID": "6",
        "TRTMC_MODEL_PROOF_GPU_IDS": "6",
        "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "1",
        "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": "600",
    }
    holder_release = tmp_path / "release-holder"
    waiter_release = tmp_path / "release-waiter"
    waiter_release.touch()
    holder, holder_output = _start_proof_case(
        tmp_path, "holder", "m2m_100", holder_release, common_env
    )
    waiters: list[subprocess.Popen[str] | None] = [None] * waiter_count
    waiter_outputs: list[Path] = []
    try:
        deadline = time.monotonic() + coordination_timeout_s
        while (
            time.monotonic() < deadline
            and not (holder_output / "artifacts" / "gpu-lease.json").is_file()
        ):
            time.sleep(0.05)
        assert (holder_output / "artifacts" / "gpu-lease.json").is_file()

        # Start all waiters at once so their host setup runs concurrently
        # (sequential starts made this test dominate the CI step budget).
        # The enqueue order is whatever the race produced; the FIFO contract
        # is asserted afterwards from the recorded ticket numbers.
        for index in range(waiter_count):
            process, output = _start_proof_case(
                tmp_path, f"waiter-{index}", "convbert", waiter_release, common_env
            )
            waiters[index] = process
            waiter_outputs.append(output)
        deadline = time.monotonic() + coordination_timeout_s
        while (
            time.monotonic() < deadline
            and len(list(lock_dir.glob("admission-global-*.lock"))) < waiter_count
        ):
            time.sleep(0.05)
        assert len(list(lock_dir.glob("admission-global-*.lock"))) == waiter_count

        holder_release.touch()
        for index, process in enumerate(waiters):
            assert process is not None
            stdout, stderr = process.communicate(timeout=coordination_timeout_s)
            assert process.returncode == 0, f"waiter-{index}: " + stdout + stderr
        holder.communicate(timeout=30)
        assert holder.returncode == 0

        # Enqueue order (ticket numbers) must equal service order (lease
        # creation times): the chain wakes exactly one successor per release
        # and nobody overtakes.
        tickets = [
            (output / "artifacts" / "gpu-queue-joined.txt").read_text(encoding="utf-8")
            for output in waiter_outputs
        ]
        assert len(set(tickets)) == waiter_count
        order_by_ticket = sorted(range(waiter_count), key=lambda i: tickets[i])
        lease_times = [
            (output / "artifacts" / "gpu-lease.json").stat().st_mtime_ns
            for output in waiter_outputs
        ]
        order_by_service = sorted(range(waiter_count), key=lambda i: lease_times[i])
        assert order_by_service == order_by_ticket
        assert not list(lock_dir.glob("admission-global-*.lock"))
    finally:
        holder_release.touch()
        _finish_proof_cases([holder, *waiters])


def _proof_gpu_ids_if_present(docker_log: Path) -> list[str]:
    if not docker_log.is_file():
        return []
    proof_runs = [
        line
        for line in docker_log.read_text(encoding="utf-8").splitlines()
        if " --inner " in f" {line} "
    ]
    return [gpu_id for line in proof_runs for gpu_id in re.findall(r"--gpus device=([0-9]+)", line)]


def test_gpu_mapping_exists_only_on_the_hermetic_proof_container() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    workflow = PROOF_WORKFLOW.read_text(encoding="utf-8")
    host = text.split("run_host() {", maxsplit=1)[1]
    warm = host.split("local -a cache_check_docker_args=(", maxsplit=1)[1].split(
        "local -a docker_args=(", maxsplit=1
    )[0]
    proof = host.split("local -a docker_args=(", maxsplit=1)[1].split("set +e", maxsplit=1)[0]

    assert host.index("warm_hf_cache.py") < host.index("select_proof_gpu")
    assert "--gpus" not in warm
    assert "TRTMC_MODEL_PROOF_GPU_ID" not in warm
    assert '--gpus "device=$gpu_id"' in proof
    assert '-e "TRTMC_MODEL_PROOF_GPU_ID=$gpu_id"' in proof
    assert '-e "TRTMC_MODEL_PROOF_GPU_SLOT_IDS=$gpu_slot_ids"' in proof
    assert '-e "TRTMC_MODEL_PROOF_SLOTS_PER_GPU=$proof_gpu_slots_per_gpu"' in proof
    assert '-e "TRTMC_MODEL_PROOF_RESOURCE_CLASS=$proof_gpu_resource_class"' in proof
    assert "TRTMC_MODEL_PROOF_SLOTS_PER_GPU:-4" in text
    assert "gpu-$gpu_id-slot-$slot.lock" in text
    assert "gpu-$gpu_id-reservation.lock" in text
    assert "TRTMC_GPU_ID must be present in TRTMC_MODEL_PROOF_GPU_IDS" in text
    assert '"gpu_id": gpu_id' in text
    assert '"gpu_slot": gpu_slots[0] if resource_class == "shared" else None' in text
    assert '"gpu_slots": gpu_slots' in text
    assert '"gpu_slot_ids": gpu_slots' in text
    assert '"slots_per_gpu": int(slots_per_gpu_text)' in text
    assert '"gpu_slots_per_device": int(slots_per_gpu_text)' in text
    assert '"resource_class": resource_class' in text
    assert '"gpu_resource_class": resource_class' in text
    assert '"gpu_lease_evidence": "gpu-lease.json"' in text
    assert (
        "TRTMC_MODEL_PROOF_GPU_IDS: ${{ vars.TRTMC_MODEL_PROOF_GPU_IDS || '0,1,2,3' }}" in workflow
    )
    assert (
        "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS: "
        "${{ vars.TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS || "
        "(inputs.suite == 'nightly' && '5400' || '3600') }}"
        in workflow
    )
