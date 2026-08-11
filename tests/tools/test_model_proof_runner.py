# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the hermetic single-model proof runner."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.ci import gpu_lease as gpu_lease_module
from tools.ci.context import CiContext
from tools.ci.gpu_lease import GpuLease
from tools.ci.model_reference_cache import ModelReferenceCacheWarmer
from tools.ci.model_proof import ModelProofRequest, ModelProofRunner, ModelReferenceCache
from tools.ci.model_proof_inner import ModelProofInnerPipeline
from tools.ci.model_proof_selection import ModelProofSelector
from tools.ci.process import CiError


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "tools" / "ci" / "model_proof.py"
RUNNER_COMMAND = [sys.executable, "-m", "tools.ci", "model-proof"]
GPU_LEASE_WORKER = "tests.tools.gpu_lease_worker"
PROJECT_CACHE = REPO_ROOT / "tests" / "tools" / "model_ci_project_cache.py"
MODEL_CI = REPO_ROOT / "tools" / "model_ci.py"
IMAGE_ENSURE = REPO_ROOT / "tools" / "ci" / "docker_image.py"
FALLBACK_WRITER = REPO_ROOT / ".github" / "scripts" / "write-model-proof-fallback-report.py"
PLUGIN_CMAKE = REPO_ROOT / "cmake" / "trtmc_pipeline_plugins.cmake"
SANA_REFERENCE_REVISION = "59629fdf790850797cb657bad014fce432bd713d"
SANA_REFERENCE_RELATIVE_PATH = "sana_wm/reference/Sana-59629fdf7908"
SANA_REFERENCE_ENTRYPOINT = "inference_video_scripts/wm/inference_sana_wm.py"


@pytest.fixture(scope="session", autouse=True)
def _cache_model_proof_source_projection(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    base = tmp_path_factory.getbasetemp()
    shared = base.parent if base.name.startswith("popen-gw") else base
    cache_root = shared / "model-proof-projection-cache"
    wrapper_bin = base / "model-proof-python"
    wrapper_bin.mkdir()
    wrapper = wrapper_bin / "python3"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'if [ "${{1:-}}" = {shlex.quote(str(MODEL_CI))} ] '
        '&& [ "${2:-}" = project ]; then\n'
        f"  exec {shlex.quote(sys.executable)} {shlex.quote(str(PROJECT_CACHE))} "
        f'--cache-root {shlex.quote(str(cache_root))} -- "$@"\n'
        "fi\n"
        f'exec {shlex.quote(sys.executable)} "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    previous = os.environ["PATH"]
    os.environ["PATH"] = f"{wrapper_bin}:{previous}"
    try:
        yield
    finally:
        os.environ["PATH"] = previous


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
        "      printf '%s\\n' \"$record\"\n"
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


def _write_fake_nvidia_smi(fake_bin: Path) -> None:
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import os
            import sys
            import time

            time.sleep(float(os.environ.get("FAKE_NVIDIA_SMI_DELAY_SECONDS", "0")))
            if "--query-compute-apps=pid" in sys.argv:
                output = os.environ.get("FAKE_NVIDIA_SMI_COMPUTE_ROWS", "")
            elif sys.argv[1:3] == ["topo", "--gpu-numa-id"]:
                output = os.environ.get("FAKE_NVIDIA_SMI_TOPOLOGY", "")
            elif any(
                argument.startswith("--query-gpu=pci.bus_id")
                for argument in sys.argv
            ):
                output = os.environ.get("FAKE_NVIDIA_SMI_IDENTITY_ROWS", "")
            else:
                output = os.environ.get("FAKE_NVIDIA_SMI_ROWS", "")
            if output:
                print(output)
            exit_code = int(os.environ.get("FAKE_NVIDIA_SMI_EXIT_CODE", "0"))
            if exit_code:
                print("fake nvidia-smi failure", file=sys.stderr)
            raise SystemExit(exit_code)
            """
        ),
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)


def _fake_gpu_lease_context(
    tmp_path: Path,
    rows: str,
    *,
    timeout_seconds: int = 3,
) -> CiContext:
    fake_bin = tmp_path / "nvidia-bin"
    fake_bin.mkdir()
    _write_fake_nvidia_smi(fake_bin)
    env = os.environ.copy()
    env.pop("TRTMC_GPU_ID", None)
    env.pop("TRTMC_GPU_SLOT_ID", None)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "FAKE_NVIDIA_SMI_ROWS": rows,
            "TRTMC_MODEL_PROOF_GPU_IDS": "2,3",
            "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "2",
            "TRTMC_MODEL_PROOF_GPU_LOCK_DIR": str(tmp_path / "gpu-locks"),
            "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": str(timeout_seconds),
            "TRTMC_MODEL_PROOF_POLL_INTERVAL": "0.01",
            "TRTMC_MODEL_PROOF_FLOCK_WATCHDOG_SECONDS": "2",
        }
    )
    return CiContext(REPO_ROOT, env)


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
    env.pop("TRTMC_GPU_SLOT_ID", None)
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
            *RUNNER_COMMAND,
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


def _sana_reference_contract() -> dict[str, str]:
    return {
        "repository": "https://github.com/NVlabs/Sana.git",
        "revision": SANA_REFERENCE_REVISION,
        "relative_path": SANA_REFERENCE_RELATIVE_PATH,
        "entrypoint": SANA_REFERENCE_ENTRYPOINT,
    }


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




def _copy_selection_inputs(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        selected = path.name == "MODEL.toml" or (
            path.parent.name == "manifests" and path.suffix == ".json"
        )
        python_test = path.name.startswith("test_") and path.suffix == ".py"
        if not path.is_file() or not (selected or python_test):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if python_test and not selected:
            target.touch()
        else:
            shutil.copy2(path, target)


def _run_test_selection(
    tmp_path: Path,
    family: str,
    suite: str,
    *,
    lease_env: dict[str, str] | None = None,
    projection_setup: Callable[[Path, dict[str, object]], None] | None = None,
) -> dict:
    source = tmp_path / f"{family}-{suite}"
    e2e_source = REPO_ROOT / "tests" / "e2e" / "models" / family
    e2e_target = source / "tests" / "e2e" / "models" / family
    _copy_selection_inputs(e2e_source, e2e_target)
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
        _copy_selection_inputs(family_source, family_root / family)
    else:
        family_root.mkdir(parents=True)
    revision = "a" * 40
    projection: dict[str, object] = {
        "revision": revision,
        "model": family,
        "runtime_model": "fixture_runtime",
        "e2e_family": family,
    }
    if projection_setup is not None:
        projection_setup(source, projection)
    (source / ".trtmc-model-projection.json").write_text(json.dumps(projection), encoding="utf-8")
    selection_path = tmp_path / f"{family}-{suite}-selection.json"
    env = os.environ.copy()
    for name in (
        "TRTMC_MODEL_PROOF_GPU_ID",
        "TRTMC_MODEL_PROOF_GPU_SLOT_IDS",
        "TRTMC_MODEL_PROOF_SLOTS_PER_GPU",
        "TRTMC_MODEL_PROOF_RESOURCE_CLASS",
        "TRTMC_MODEL_PROOF_MIN_FREE_GPU_MEMORY_MIB",
    ):
        env.pop(name, None)
    env.update(lease_env or {})
    lease = None
    if lease_env:
        slots = [int(item) for item in lease_env["TRTMC_MODEL_PROOF_GPU_SLOT_IDS"].split(",")]
        capacity = int(lease_env["TRTMC_MODEL_PROOF_SLOTS_PER_GPU"])
        resource = lease_env["TRTMC_MODEL_PROOF_RESOURCE_CLASS"]
        lease = {
            "gpu_id": lease_env["TRTMC_MODEL_PROOF_GPU_ID"],
            "gpu_slot_ids": slots,
            "slots_per_gpu": capacity,
            "resource_class": resource,
            "min_free_gpu_memory_mib": int(
                lease_env.get("TRTMC_MODEL_PROOF_MIN_FREE_GPU_MEMORY_MIB", "0")
            ),
        }
    ModelProofSelector(family, suite, revision, source).select(selection_path, lease)
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
        [*RUNNER_COMMAND, "--model", "alpha", "--suite", "everything"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--suite must be premerge or nightly" in result.stderr


@pytest.mark.parametrize(
    ("family", "expected_case"),
    (
        ("bark", "bark-small-fp32-l0"),
        ("flux", "flux-schnell-l0"),
        ("personaplex", "personaplex-7b-l0"),
        ("canary", "canary-1b-v2"),
        ("nemotron_labs_diffusion", "nemotron-labs-diffusion-8b-l0"),
        ("qwen_image", "qwen-image-l0"),
        ("elf_flow", "elf-b-xsum-l0"),
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


def test_wan22_premerge_selects_standalone_l0_manifest(tmp_path: Path) -> None:
    selection = _run_test_selection(tmp_path, "wan2_2_ti2v", "premerge")

    assert [case["name"] for case in selection["e2e_cases"]] == ["wan22-ti2v-5b-l0"]
    assert selection["e2e_cases"][0]["model"] == "wan22-ti2v-5b-l0"
    assert selection["e2e_cases"][0]["ci_tier"] == "l0_only"
    assert "model_reference_cache" not in selection


def test_qwen_premerge_selects_native_defaults_l0(tmp_path: Path) -> None:
    selection = _run_test_selection(tmp_path, "qwen", "premerge")

    assert selection["suite"] == "premerge"
    assert [
        (
            case["name"],
            case["model"],
            case["manifest"],
            case["ci_tier"],
        )
        for case in selection["e2e_cases"]
    ] == [
        (
            "qwen3-0.6b-native-l0",
            "qwen3-0.6b-native-l0",
            "qwen3-0.6b-native-l0.json",
            "l0_only",
        )
    ]


def test_qwen_nightly_keeps_production_cases(tmp_path: Path) -> None:
    selection = _run_test_selection(tmp_path, "qwen", "nightly")

    assert selection["suite"] == "nightly"
    assert {case["name"] for case in selection["e2e_cases"]} == {
        "qwen3-0.6b-fp16",
        "qwen3-0.6b-fp8",
        "qwen3-0.6b-topp",
        "qwen3-4b-instruct-2507",
    }
    assert all(case["ci_tier"] != "l0_only" for case in selection["e2e_cases"])


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


def _selection_with_nested_adapter_and_unselected_sibling(
    tmp_path: Path,
) -> tuple[dict, str, str]:
    selected_root = "tests/e2e/models/flux/optimized_adapter"
    sibling_root = "tests/e2e/models/sibling_model/optimized_adapter"

    def project_adapter_tests(source: Path, _projection: dict[str, object]) -> None:
        selected_tests = source / selected_root
        selected_tests.mkdir(parents=True)
        for name in ("test_capsule.py", "test_runtime_contract.py", "test_adapter_e2e.py"):
            (selected_tests / name).write_text("# projected fixture\n", encoding="utf-8")
        sibling_tests = source / sibling_root
        sibling_tests.mkdir(parents=True)
        (sibling_tests / "test_sibling_capsule.py").write_text(
            "def test_sibling_capsule(): pass\n", encoding="utf-8"
        )

    return (
        _run_test_selection(
            tmp_path,
            "flux",
            "premerge",
            projection_setup=project_adapter_tests,
        ),
        selected_root,
        sibling_root,
    )


def test_selection_includes_nested_model_owned_adapter_tests(
    tmp_path: Path,
) -> None:
    selection, selected_root, _ = _selection_with_nested_adapter_and_unselected_sibling(tmp_path)

    selected = set(selection["python_tests"])
    expected_adapter_tests = {
        f"{selected_root}/{name}" for name in ("test_capsule.py", "test_runtime_contract.py")
    }
    assert expected_adapter_tests == {
        path for path in selected if path.startswith(f"{selected_root}/")
    }
    assert expected_adapter_tests
    assert f"{selected_root}/test_adapter_e2e.py" not in selected


def test_selection_excludes_unselected_sibling_model_tests(
    tmp_path: Path,
) -> None:
    selection, _, sibling_root = _selection_with_nested_adapter_and_unselected_sibling(tmp_path)

    assert not any(path.startswith(f"{sibling_root}/") for path in selection["python_tests"])


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


def test_wan22_nightly_selection_emits_only_the_pinned_source_contract(
    tmp_path: Path,
) -> None:
    selection = _run_test_selection(tmp_path, "wan2_2_ti2v", "nightly")

    assert selection["model_reference_cache"] == {
        "repository": "https://github.com/Wan-Video/Wan2.2.git",
        "revision": "42bf4cfaa384bc21833865abc2f9e6c0e67233dc",
        "relative_path": "wan2_2_ti2v/reference/Wan2.2-42bf4cfaa384",
        "entrypoint": "wan/textimage2video.py",
    }


def test_lance_selection_projects_reference_environment_into_proof(
    tmp_path: Path,
) -> None:
    selection = _run_test_selection(tmp_path, "lance", "premerge")
    contract = selection["model_reference_cache"]
    assert contract["environment_variable"] == "TRTMC_LANCE_REFERENCE_REPO"

    runner = ModelProofRunner(
        CiContext(REPO_ROOT, os.environ.copy()),
        ModelProofRequest("lance"),
    )
    runner.lease = SimpleNamespace(
        gpu_id=0,
        slots_per_gpu=4,
        resource_class="exclusive_gpu",
        min_free_gpu_memory_mib=0,
    )

    environment = runner._proof_environment("0,1,2,3", contract)

    assert (
        "TRTMC_LANCE_REFERENCE_REPO="
        "/work/reference-private/lance/reference/Lance-4baeee086648"
    ) in environment


def test_inner_proof_runs_the_exact_model_owned_python_test_selection() -> None:
    selector = (REPO_ROOT / "tools/ci/model_proof_selection.py").read_text(encoding="utf-8")
    inner = (REPO_ROOT / "tools/ci/model_proof_inner.py").read_text(encoding="utf-8")

    assert '"python_tests": [' in selector
    assert "for path in sorted(set(python_tests))" in selector
    assert 'payload["python_tests"]' in inner
    assert "self.source / path" in inner
    assert 'glob("test_*.py")' in selector
    assert '"TRTMC_BINARY": self._trtmc()' in inner
    assert '"TRTMC_ELF_TIMING_CACHE_PATH": ""' in inner
    assert '"TRTMC_ELF_TIMING_CACHE_METADATA_PATH": ""' in inner
    assert '"TRTMC_ELF_TIMING_CACHE_GENERATE": "0"' in inner


@pytest.mark.parametrize(
    ("family", "expected_resource"),
    (
        ("bark", "exclusive_gpu"),
        ("convbert", "shared"),
        ("flux", "exclusive_gpu"),
        ("gpt2", "exclusive_gpu"),
        ("m2m_100", "exclusive_gpu"),
        ("mixtral", "exclusive_gpu"),
        ("timesfm", "exclusive_gpu"),
        ("whisper", "exclusive_gpu"),
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


def test_whisper_nightly_selection_leases_one_complete_gpu(
    tmp_path: Path,
) -> None:
    selection = _run_test_selection(
        tmp_path,
        "whisper",
        "nightly",
        lease_env={
            "TRTMC_MODEL_PROOF_GPU_ID": "2",
            "TRTMC_MODEL_PROOF_GPU_SLOT_IDS": "0,1,2,3",
            "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "4",
            "TRTMC_MODEL_PROOF_RESOURCE_CLASS": "exclusive_gpu",
        },
    )

    assert len(selection["e2e_cases"]) == 16
    assert {case["manifest"] for case in selection["e2e_cases"]} == {
        "whisper-large-v3-turbo.json",
        "whisper-tiny-fp16.json",
    }
    assert {case["resource_class"] for case in selection["e2e_cases"]} == {"exclusive_gpu"}
    assert selection["gpu_resource_class"] == "exclusive_gpu"
    assert selection["gpu_slot_ids"] == [0, 1, 2, 3]


def test_qwen3_omni_selection_requires_clean_gpu_capacity(tmp_path: Path) -> None:
    selection = _run_test_selection(tmp_path, "qwen3_omni", "nightly")

    assert selection["resource_class"] == "exclusive_gpu"
    assert selection["min_free_gpu_memory_mib"] == 280000
    assert {case["min_free_gpu_memory_mib"] for case in selection["e2e_cases"]} == {280000}


def test_qwen_moe_selection_requires_clean_gpu_capacity(tmp_path: Path) -> None:
    selection = _run_test_selection(tmp_path, "qwen_moe", "nightly")

    assert selection["resource_class"] == "exclusive_gpu"
    assert selection["min_free_gpu_memory_mib"] == 122880
    assert {case["min_free_gpu_memory_mib"] for case in selection["e2e_cases"]} == {122880}


def test_minimax_h3_selection_requires_native_runtime_capacity(tmp_path: Path) -> None:
    selection = _run_test_selection(tmp_path, "minimax_h3", "premerge")

    assert selection["resource_class"] == "exclusive_gpu"
    assert selection["min_free_gpu_memory_mib"] == 184320
    assert {case["min_free_gpu_memory_mib"] for case in selection["e2e_cases"]} == {184320}


def test_selection_without_a_capacity_requirement_normalizes_to_zero(
    tmp_path: Path,
) -> None:
    selection = _run_test_selection(tmp_path, "convbert", "nightly")

    assert selection["min_free_gpu_memory_mib"] == 0
    assert {case["min_free_gpu_memory_mib"] for case in selection["e2e_cases"]} == {0}


def test_nightly_capacity_requirement_uses_maximum_instead_of_sum(
    tmp_path: Path,
) -> None:
    def configure(source: Path, _projection: dict[str, object]) -> None:
        manifests = source / "tests/e2e/models/flux/manifests"
        requirements = {
            "flux-2-dev.json": 100000,
            "flux-2-dev-fp8.json": 200000,
            "flux-schnell.json": 230000,
        }
        for name, requirement in requirements.items():
            path = manifests / name
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["e2e_parallel_resource"] = "exclusive_gpu"
            payload["e2e_min_free_gpu_memory_mib"] = requirement
            path.write_text(json.dumps(payload), encoding="utf-8")

    selection = _run_test_selection(
        tmp_path,
        "flux",
        "nightly",
        projection_setup=configure,
    )

    assert selection["min_free_gpu_memory_mib"] == 230000


@pytest.mark.parametrize("value", [None, True, 0, -1, 1.5, "240000"])
def test_selector_rejects_invalid_gpu_capacity_requirements(
    tmp_path: Path,
    value: object,
) -> None:
    def configure(source: Path, _projection: dict[str, object]) -> None:
        path = next((source / "tests/e2e/models/convbert/manifests").glob("*.json"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["e2e_parallel_resource"] = "exclusive_gpu"
        payload["e2e_min_free_gpu_memory_mib"] = value
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CiError, match="e2e_min_free_gpu_memory_mib"):
        _run_test_selection(
            tmp_path,
            "convbert",
            "nightly",
            projection_setup=configure,
        )


def test_selector_rejects_gpu_capacity_on_a_shared_manifest(
    tmp_path: Path,
) -> None:
    def configure(source: Path, _projection: dict[str, object]) -> None:
        path = next((source / "tests/e2e/models/convbert/manifests").glob("*.json"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["e2e_parallel_resource"] = "shared"
        payload["e2e_min_free_gpu_memory_mib"] = 240000
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CiError, match="requires.*exclusive_gpu"):
        _run_test_selection(
            tmp_path,
            "convbert",
            "nightly",
            projection_setup=configure,
        )


def test_selector_rejects_testcase_level_gpu_capacity(
    tmp_path: Path,
) -> None:
    def configure(source: Path, _projection: dict[str, object]) -> None:
        path = next((source / "tests/e2e/models/convbert/manifests").glob("*.json"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["testcases"][0]["e2e_min_free_gpu_memory_mib"] = 240000
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CiError, match="model-only"):
        _run_test_selection(
            tmp_path,
            "convbert",
            "nightly",
            projection_setup=configure,
        )


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


def _inner_gpu_lease_fixture(
    tmp_path: Path,
) -> tuple[ModelProofInnerPipeline, Path, dict[str, object]]:
    revision = "a" * 40
    env = {
        "TRTMC_MODEL_PROOF_GPU_ID": "3",
        "TRTMC_MODEL_PROOF_RESOURCE_CLASS": "exclusive_gpu",
        "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "4",
        "TRTMC_MODEL_PROOF_GPU_SLOT_IDS": "0,1,2,3",
        "TRTMC_MODEL_PROOF_MIN_FREE_GPU_MEMORY_MIB": "240000",
    }
    pipeline = ModelProofInnerPipeline(
        CiContext(REPO_ROOT, env),
        ModelProofRequest("qwen3_omni", revision=revision),
    )
    pipeline.artifacts = tmp_path / "artifacts"
    pipeline.artifacts.mkdir()
    evidence_path = pipeline.artifacts / "gpu-lease.json"
    evidence: dict[str, object] = {
        "schema_version": 1,
        "model": "qwen3_omni",
        "source_revision": revision,
        "gpu_id": "3",
        "gpu_slot": None,
        "gpu_slots": [0, 1, 2, 3],
        "gpu_slot_ids": [0, 1, 2, 3],
        "slots_per_gpu": 4,
        "gpu_slots_per_device": 4,
        "resource_class": "exclusive_gpu",
        "gpu_resource_class": "exclusive_gpu",
        "min_free_gpu_memory_mib": 240000,
        "gpu_memory_admission": {
            "source": "nvidia-smi",
            "required_free_mib": 240000,
            "observed_total_mib": 284208,
            "observed_used_mib": 34208,
            "observed_free_mib": 250000,
        },
    }
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    return pipeline, evidence_path, evidence


def test_inner_gpu_lease_accepts_valid_memory_admission(tmp_path: Path) -> None:
    pipeline, _, _ = _inner_gpu_lease_fixture(tmp_path)

    lease = pipeline._validate_gpu_lease()

    assert lease["min_free_gpu_memory_mib"] == 240000
    assert lease["gpu_memory_admission"] == {
        "source": "nvidia-smi",
        "required_free_mib": 240000,
        "observed_total_mib": 284208,
        "observed_used_mib": 34208,
        "observed_free_mib": 250000,
    }


def test_inner_gpu_lease_accepts_linux_numa_memory_admission(tmp_path: Path) -> None:
    pipeline, evidence_path, evidence = _inner_gpu_lease_fixture(tmp_path)
    admission = evidence["gpu_memory_admission"]
    assert isinstance(admission, dict)
    admission.update(
        {
            "source": "linux-numa-meminfo",
            "observed_total_mib": 283136,
            "observed_used_mib": 2675,
            "observed_free_mib": 280461,
        }
    )
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    lease = pipeline._validate_gpu_lease()

    assert lease["gpu_memory_admission"] == admission


@pytest.mark.parametrize("schema_version", [2, True])
def test_inner_gpu_lease_rejects_admission_schema_mismatch(
    tmp_path: Path,
    schema_version: object,
) -> None:
    pipeline, evidence_path, evidence = _inner_gpu_lease_fixture(tmp_path)
    evidence["schema_version"] = schema_version
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(CiError, match="unsupported schema"):
        pipeline._validate_gpu_lease()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("gpu_slots", [0, True, 2, 3]),
        ("gpu_slot_ids", [0, True, 2, 3]),
        ("slots_per_gpu", True),
        ("gpu_slots_per_device", True),
    ),
)
def test_inner_gpu_lease_rejects_boolean_integer_fields(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    pipeline, evidence_path, evidence = _inner_gpu_lease_fixture(tmp_path)
    evidence[field] = value
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(CiError, match=f"invalid {field}"):
        pipeline._validate_gpu_lease()


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_inner_gpu_lease_rejects_admission_field_set_mismatch(
    tmp_path: Path,
    mutation: str,
) -> None:
    pipeline, evidence_path, evidence = _inner_gpu_lease_fixture(tmp_path)
    admission = evidence["gpu_memory_admission"]
    assert isinstance(admission, dict)
    if mutation == "missing":
        del admission["observed_used_mib"]
    else:
        admission["unexpected"] = 1
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(CiError, match="unexpected or missing fields"):
        pipeline._validate_gpu_lease()


@pytest.mark.parametrize("requirement", [239999, True])
def test_inner_gpu_lease_rejects_admission_requirement_mismatch(
    tmp_path: Path,
    requirement: object,
) -> None:
    pipeline, evidence_path, evidence = _inner_gpu_lease_fixture(tmp_path)
    admission = evidence["gpu_memory_admission"]
    assert isinstance(admission, dict)
    admission["required_free_mib"] = requirement
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(CiError, match="requirement does not match"):
        pipeline._validate_gpu_lease()


def test_inner_gpu_lease_rejects_free_memory_below_requirement(
    tmp_path: Path,
) -> None:
    pipeline, evidence_path, evidence = _inner_gpu_lease_fixture(tmp_path)
    admission = evidence["gpu_memory_admission"]
    assert isinstance(admission, dict)
    admission["observed_free_mib"] = 239999
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(CiError, match="did not satisfy the required free memory"):
        pipeline._validate_gpu_lease()


@pytest.mark.parametrize("minimum", [239999, True])
def test_inner_gpu_lease_rejects_top_level_minimum_mismatch(
    tmp_path: Path,
    minimum: object,
) -> None:
    pipeline, evidence_path, evidence = _inner_gpu_lease_fixture(tmp_path)
    evidence["min_free_gpu_memory_mib"] = minimum
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(CiError, match="invalid minimum free GPU memory"):
        pipeline._validate_gpu_lease()


@pytest.mark.parametrize(
    ("family", "expected_cases"),
    (
        (
            "bark",
            {
                "bark-large",
                "bark-small",
                "bark-small-tts-probe01",
                "bark-small-tts-probe02",
            },
        ),
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
        ("wan2_2_ti2v", {"wan22-ti2v-5b"}),
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
    assert {case["name"]: case["gpu_resource_class"] for case in selection["e2e_cases"]} == {
        "flux-2-dev": "exclusive_gpu",
        "flux-2-dev-fp8": "shared",
        "flux-schnell": "shared",
    }


def test_llama_nightly_model_proof_reserves_an_exclusive_gpu(tmp_path: Path) -> None:
    selection = _run_test_selection(
        tmp_path,
        "llama",
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
    assert {case["name"]: case["gpu_resource_class"] for case in selection["e2e_cases"]} == {
        "falcon3-1b": "exclusive_gpu",
        "minitron-4b-depth": "shared",
        "minitron-4b-width": "shared",
        "nemotron-nano-4b": "shared",
        "tinyllama-1.1b": "shared",
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


def test_nightly_inventory_exactly_matches_every_model_proof_selection(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [sys.executable, str(MODEL_CI), "all", "--revision", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    inventory = json.loads(result.stdout)
    selected_cases_by_model = {}
    for model in inventory["affected_models"]:
        selection = _run_test_selection(tmp_path, model, "nightly")
        selected_cases_by_model[model] = [case["name"] for case in selection["e2e_cases"]]

    assert inventory["expected_cases_by_model"] == selected_cases_by_model
    assert inventory["expected_result_count"] == sum(
        len(cases) for cases in selected_cases_by_model.values()
    )


def test_runner_declares_the_hermetic_container_boundary() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    inner = (REPO_ROOT / "tools/ci/model_proof_inner.py").read_text(encoding="utf-8")
    warm = text.split("def _prepare_hf_cache(", maxsplit=1)[1].split(
        "def _validated_cache_evidence", maxsplit=1
    )[0]
    proof = text.split("def _run_proof_container(", maxsplit=1)[1].split(
        "def _proof_environment", maxsplit=1
    )[0]

    for contract in (
        '"--read-only"',
        '"--network"',
        '"none"',
        '"--cap-drop"',
        '"ALL"',
        "dst=/src,readonly",
        '"TMPDIR": "/work/tmp"',
        '"TORCHINDUCTOR_CACHE_DIR": "/work/torch-cache"',
        '"TRTMC_MODEL_PLUGIN_STRICT": "1"',
    ):
        assert contract in text
    assert "scratch build produced" in inner
    assert "staged plugin DSO does not byte-match" in inner
    assert '"--network"' in warm and '"none"' in warm
    assert "dst=/hf-cache/hub,readonly" in warm
    assert "dst=/hf-cache/modules" not in warm
    assert '"HF_HOME=/tmp/hf-home"' in warm
    assert '"HF_MODULES_CACHE=/tmp/hf-modules"' in warm
    assert "dst=/artifacts" in warm
    assert "dst=/artifacts,readonly" not in warm
    assert "-e HF_TOKEN" not in warm
    assert "-e HUGGING_FACE_HUB_TOKEN" not in warm
    assert '"--network"' in proof and '"none"' in proof
    assert 'f"type=bind,src={private_hub},dst=/hf-cache/hub"' in proof
    assert "src={private_hub},dst=/hf-cache/hub,readonly" not in proof
    assert "dst=/hf-cache/hub/$hf_repo_folder" not in text
    assert "dst=/hf-cache/modules" not in proof
    assert '"HF_HOME": "/work/hf-home"' in text
    assert '"HF_MODULES_CACHE": "/work/hf-modules"' in text
    assert '"TRANSFORMERS_CACHE": "/hf-cache/hub"' in text
    assert '"--reflink=always"' in text
    assert '"TRTMC_STORAGE_ROOT"] = "/work/reference-private"' in text
    assert "TRTMC_MODEL_REFERENCE_CACHE_ROOT" not in proof
    assert "src=$reference_cache_root" not in proof
    assert "src=$reference_source" not in proof
    assert "-e HF_TOKEN" not in proof
    assert "-e HUGGING_FACE_HUB_TOKEN" not in proof


def test_runner_warms_the_exact_shared_selection_before_the_proof() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    host = text.split("def _run_host(self)", maxsplit=1)[1]
    warm = host.split("def _prepare_hf_cache(", maxsplit=1)[1].split(
        "def _validated_cache_evidence", maxsplit=1
    )[0]

    assert host.count("ModelProofSelector(") == 1
    assert "selection.e2e_models" in host
    assert "cache-check-models.txt" in warm
    assert "scripts/warm_hf_cache.py" in warm
    assert '"--models-file"' in warm and '"/artifacts/cache-check-models.txt"' in warm
    assert '"--local-only"' in warm and '"--strict"' in warm
    assert '"--emit-cache-repos"' in warm and '"/artifacts/hf-cache-repos.json"' in warm
    assert host.index("_prepare_hf_cache") < host.index("_run_proof_container")
    assert "offline HF cache readiness check failed" in warm




def test_runner_removes_only_its_container_without_masking_exit_status() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    cleanup = text.split("def _cleanup(self)", maxsplit=1)[1].split("def _signal", maxsplit=1)[0]

    assert '["docker", "rm", "-f", self.container_name]' in cleanup
    assert cleanup.index('["docker", "rm"') < cleanup.index("self.lease.release()")
    assert "self.lease.release()" in cleanup
    assert "for number in (signal.SIGINT, signal.SIGTERM)" in text
    assert "raise SystemExit(130 if number == signal.SIGINT else 143)" in text




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
        f"      printf '%s\\n' \"{container_id} 4242 3 convbert\"\n"
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
            *RUNNER_COMMAND,
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




def test_model_proof_enforces_one_full_bundle_build_per_selected_model() -> None:
    runner = (REPO_ROOT / "tools/ci/model_proof_inner.py").read_text(encoding="utf-8")

    for contract in (
        '"TRTMC_ENGINE_BUILD_GUARD_DIR": str(self.artifacts / "engine-builds")',
        '"TRTMC_ENGINE_BUILD_REVISION": self.request.revision',
        '"verify-builds"',
        '"--ledger-dir"',
        'self.artifacts / "engine-builds"',
        '"--source-revision"',
        "self.request.revision",
        'self.artifacts / "engine-build-verification.json"',
        '"--build-verification-report"',
        'self.status.step("engine_build_budget", "passed")',
        '"engine_builds_per_model": verification["builds_per_model"]',
        '"engine_build_count": len(verification["records"])',
    ):
        assert contract in runner

    assert runner.index('"verify-builds"') < runner.index('"verify-results"')
    assert runner.index('"verify-results"') < runner.index(
        'result.get("proof_kind") for result in e2e_verification.get("results", [])'
    )
    assert "if len(proof_kinds) != 1:" in runner
    assert 'self.status.fact("e2e_proof_kind", e2e_proof_kind)' in runner
    assert "self._python()" in runner
    assert '"pytest"' in runner
    assert 'self.source / str(payload["e2e_test"])' in runner


def test_model_proof_report_assets_are_inside_the_positive_projection() -> None:
    model_ci = (REPO_ROOT / "tools" / "model_ci.py").read_text(encoding="utf-8")

    assert '"scripts/generate_e2e_report.py"' in model_ci
    assert '"tools/ci/",' in model_ci
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
    python = fake_bin / "python3"
    python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'if [ "${{1:-}}" = {shlex.quote(str(MODEL_CI))} ] '
        '&& [ "${2:-}" = project ]; then\n'
        "  printf '%s\\n' 'unknown model: model-that-does-not-exist' >&2\n"
        "  exit 2\n"
        "fi\n"
        f'exec {shlex.quote(sys.executable)} "$@"\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    docker = fake_bin / "docker"
    docker.write_text("#!/usr/bin/env bash\nexit 99\n", encoding="utf-8")
    docker.chmod(0o755)
    output = tmp_path / "proof"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            *RUNNER_COMMAND,
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
            *RUNNER_COMMAND,
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
    host = text.split("def _prepare_hf_cache(", maxsplit=1)[1]
    cache_check = host.split("result = self._run_logged", maxsplit=1)[0]
    proof = text.split("def _run_proof_container(", maxsplit=1)[1].split(
        "def _proof_environment", maxsplit=1
    )[0]

    assert "hub.is_dir()" not in cache_check
    assert "HF Hub cache directory does not exist" not in host
    assert 'hub in {Path("/"), self.context.repository}' in host
    assert "hf_modules_cache" not in host
    assert 'f"type=bind,src={hub},dst=/hf-cache/hub,readonly"' in cache_check
    assert "dst=/hf-cache/modules" not in cache_check
    assert 'f"type=bind,src={private_hub},dst=/hf-cache/hub"' in proof
    assert "src={private_hub},dst=/hf-cache/hub,readonly" not in proof
    assert "hf_repo_mount_args" not in host
    assert "src={hub},dst=/hf-cache/hub" not in proof
    assert "dst=/hf-cache/modules" not in proof
    assert '"--reflink=always"' in text


def test_selected_hf_cache_reflink_helper_has_a_minimal_mount_and_capability_boundary() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    helper = text.split("copy = [", maxsplit=1)[1].split("if self.context.run(copy", maxsplit=1)[0]
    program = text.split('CACHE_COPY_PROGRAM = r"""', maxsplit=1)[1].split('"""', maxsplit=1)[0]

    for contract in (
        '"--read-only"',
        '"--network"',
        '"none"',
        '"--cap-drop"',
        '"ALL"',
        '"--cap-add"',
        '"DAC_OVERRIDE"',
        '"CHOWN"',
        '"--security-opt"',
        '"no-new-privileges"',
        '"--pids-limit"',
        '"32"',
        '"--user"',
        '"0:0"',
        "dst=/selected-hf-repo,readonly",
        "dst=/private-hf-repo",
        '"--entrypoint"',
        '"/usr/bin/python3"',
    ):
        assert contract in helper
    assert helper.count('"--cap-add"') == 2
    assert "dst=/selected-hf-repo,readonly" in helper
    assert "dst=/private-hf-repo,readonly" not in helper
    assert "src={hub}" not in helper
    assert "src={private_hub}" not in helper
    assert "src={projection}" not in helper
    assert "src={self.artifacts_dir}" not in helper
    assert "--gpus" not in helper
    assert "HF_TOKEN" not in helper
    assert "/var/run/docker.sock" not in helper
    assert "CACHE_COPY_PROGRAM" in helper
    assert "os.chown(destination, 0, 0)" in program
    assert program.index("os.chown(destination, 0, 0)") < program.index('"--reflink=always"')


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

    ModelReferenceCache(CiContext(REPO_ROOT, env), ModelProofRequest("sana_wm")).prepare(
        _sana_reference_contract(), work_dir, artifacts
    )
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


def test_sana_reference_cache_missing_checkout_is_warmed_before_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    cache_root = tmp_path / "model-reference-cache"
    cache_root.mkdir()
    config = tmp_path / "model-proof-config.txt"
    _write_sana_reference_config(config)
    work_dir = tmp_path / "work"
    artifacts = tmp_path / "artifacts"
    work_dir.mkdir()
    artifacts.mkdir()
    env = os.environ.copy()
    source = cache_root / SANA_REFERENCE_RELATIVE_PATH
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "TRTMC_MODEL_REFERENCE_CACHE_ROOT": str(cache_root),
            "FAKE_REFERENCE_SOURCE": str(source),
            "FAKE_REFERENCE_REVISION": SANA_REFERENCE_REVISION,
        }
    )
    warmed: list[str] = []

    def fake_warm_contract(_self, contract):
        warmed.append(contract.relative_path)
        _cache_root, prepared = _write_fake_pinned_model_reference(tmp_path, fake_bin)
        return prepared

    monkeypatch.setattr(
        ModelReferenceCacheWarmer,
        "warm_contract",
        fake_warm_contract,
    )

    ModelReferenceCache(CiContext(REPO_ROOT, env), ModelProofRequest("sana_wm")).prepare(
        _sana_reference_contract(), work_dir, artifacts
    )

    assert warmed == [SANA_REFERENCE_RELATIVE_PATH]
    assert (work_dir / "reference-private" / SANA_REFERENCE_RELATIVE_PATH).is_dir()


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

    with pytest.raises(
        CiError,
        match=(
            f"model reference cache revision mismatch for {SANA_REFERENCE_RELATIVE_PATH}: "
            f"expected {SANA_REFERENCE_REVISION}, found {wrong_revision}"
        ),
    ):
        ModelReferenceCache(CiContext(REPO_ROOT, env), ModelProofRequest("sana_wm")).prepare(
            _sana_reference_contract(), work_dir, artifacts
        )


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
            *RUNNER_COMMAND,
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
                *RUNNER_COMMAND,
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
            *RUNNER_COMMAND,
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
    assert '"cp", "-a", "--reflink=always", "--no-preserve=ownership"' in docker_text
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
            *RUNNER_COMMAND,
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
            *RUNNER_COMMAND,
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
            *RUNNER_COMMAND,
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
        "min_free_gpu_memory_mib": 0,
    }
    assert (tmp_path / "gpu-locks" / "gpu-7-slot-0.lock").is_file()


@pytest.mark.model_proof_allocator
def test_capacity_gated_exclusive_lease_skips_a_memory_busy_gpu(
    tmp_path: Path,
) -> None:
    context = _fake_gpu_lease_context(
        tmp_path,
        "2, 284208, 184208, 100000\n3, 284208, 34208, 250000",
    )
    context.env["FAKE_NVIDIA_SMI_DELAY_SECONDS"] = "0.25"
    artifacts = tmp_path / "artifacts"
    lease = GpuLease(
        context,
        "qwen3_omni",
        "exclusive_gpu",
        artifacts,
        min_free_gpu_memory_mib=240000,
    )
    prepared: list[int] = []

    def prepare_candidate() -> None:
        assert lease.gpu_id is not None
        prepared.append(lease.gpu_id)
        assert _lock_is_busy(lease.lock_dir / f"gpu-{lease.gpu_id}-reservation.lock")
        assert all(
            _lock_is_busy(lease.lock_dir / f"gpu-{lease.gpu_id}-slot-{slot}.lock")
            for slot in range(lease.slots_per_gpu)
        )
        assert not _lock_is_busy(lease.lock_dir / "allocator.lock")

    try:
        lease.acquire(prepare_candidate=prepare_candidate)

        assert prepared == [2, 3]
        assert lease.gpu_id == 3
        assert not _lock_is_busy(lease.lock_dir / "gpu-2-reservation.lock")
        assert not _lock_is_busy(lease.lock_dir / "gpu-2-slot-0.lock")
        assert not _lock_is_busy(lease.lock_dir / "gpu-2-slot-1.lock")
        assert _lock_is_busy(lease.lock_dir / "gpu-3-reservation.lock")
        assert _lock_is_busy(lease.lock_dir / "gpu-3-slot-0.lock")
        assert _lock_is_busy(lease.lock_dir / "gpu-3-slot-1.lock")
        assert lease.evidence("a" * 40)["gpu_memory_admission"] == {
            "source": "nvidia-smi",
            "required_free_mib": 240000,
            "observed_total_mib": 284208,
            "observed_used_mib": 34208,
            "observed_free_mib": 250000,
        }
    finally:
        lease.release()

    for gpu in (2, 3):
        assert not _lock_is_busy(lease.lock_dir / f"gpu-{gpu}-reservation.lock")
        assert not _lock_is_busy(lease.lock_dir / f"gpu-{gpu}-slot-0.lock")
        assert not _lock_is_busy(lease.lock_dir / f"gpu-{gpu}-slot-1.lock")


def test_capacity_gate_counts_reclaimable_clean_file_cache_on_gpu_numa_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _fake_gpu_lease_context(
        tmp_path,
        "2, 284208, 91595, 192613\n3, 284208, 124790, 159418",
    )
    lease = GpuLease(
        context,
        "qwen3_omni",
        "exclusive_gpu",
        min_free_gpu_memory_mib=280000,
    )
    lease._gpu_numa_identity = (  # type: ignore[method-assign]
        lambda gpu, *, timeout_seconds: ("00000008:06:00.0", 1072, 2)
    )
    lease._gpu_has_compute_processes = (  # type: ignore[method-assign]
        lambda gpu, *, timeout_seconds: False
    )
    lease._node_meminfo = lambda node: {  # type: ignore[method-assign]
        "MemTotal": 289931264,
        "MemFree": 197234880,
        "Active(file)": 14362240,
        "Inactive(file)": 78329984,
        "Unevictable": 0,
        "Dirty": 320,
        "Writeback": 0,
        "Mapped": 225728,
        "Shmem": 0,
    }
    lease._node_high_watermark_kib = (  # type: ignore[method-assign]
        lambda node, *, page_size_kib: 39188 * page_size_kib
    )
    monkeypatch.setattr("tools.ci.gpu_lease.os.sysconf", lambda name: 65536)

    try:
        lease.acquire()

        assert lease.gpu_id == 2
        assert lease.evidence("a" * 40)["gpu_memory_admission"] == {
            "source": "linux-numa-meminfo",
            "required_free_mib": 280000,
            "observed_total_mib": 283136,
            "observed_used_mib": 2675,
            "observed_free_mib": 280461,
        }
    finally:
        lease.release()


def test_gpu_numa_meminfo_parser_ignores_real_huge_page_counter_format() -> None:
    meminfo = GpuLease._parse_node_meminfo(
        2,
        """\
Node 2 MemTotal:       289931264 kB
Node 2 MemFree:        197234880 kB
Node 2 MemUsed:         92696384 kB
Node 2 Active:          14362240 kB
Node 2 Inactive:        78329984 kB
Node 2 Active(anon):           0 kB
Node 2 Inactive(anon):         0 kB
Node 2 Active(file):    14362240 kB
Node 2 Inactive(file):  78329984 kB
Node 2 Unevictable:            0 kB
Node 2 Mlocked:                0 kB
Node 2 Dirty:                320 kB
Node 2 Writeback:              0 kB
Node 2 FilePages:       92692352 kB
Node 2 Mapped:            225728 kB
Node 2 AnonPages:              0 kB
Node 2 Shmem:                  0 kB
Node 2 KernelStack:            0 kB
Node 2 PageTables:             0 kB
Node 2 NFS_Unstable:           0 kB
Node 2 Bounce:                 0 kB
Node 2 WritebackTmp:           0 kB
Node 2 KReclaimable:           0 kB
Node 2 Slab:                   0 kB
Node 2 SReclaimable:           0 kB
Node 2 SUnreclaim:             0 kB
Node 2 AnonHugePages:          0 kB
Node 2 ShmemHugePages:         0 kB
Node 2 ShmemPmdMapped:         0 kB
Node 2 FileHugePages:          0 kB
Node 2 FilePmdMapped:          0 kB
Node 2 HugePages_Total:        0
Node 2 HugePages_Free:         0
Node 2 HugePages_Surp:         0
""",
    )

    assert meminfo == {
        "MemTotal": 289931264,
        "MemFree": 197234880,
        "Active(file)": 14362240,
        "Inactive(file)": 78329984,
        "Unevictable": 0,
        "Dirty": 320,
        "Writeback": 0,
        "Mapped": 225728,
        "Shmem": 0,
    }


def test_capacity_gate_rejects_raw_capacity_when_a_compute_process_appears(
    tmp_path: Path,
) -> None:
    context = _fake_gpu_lease_context(
        tmp_path,
        "2, 284208, 3208, 281000",
    )
    lease = GpuLease(
        context,
        "qwen3_omni",
        "exclusive_gpu",
        min_free_gpu_memory_mib=280000,
    )
    lease.gpu_id = 2
    lease._gpu_has_compute_processes = (  # type: ignore[method-assign]
        lambda gpu, *, timeout_seconds: True
    )
    lease._gpu_memory_snapshot = (  # type: ignore[method-assign]
        lambda gpu, *, timeout_seconds: {
            "source": "nvidia-smi",
            "total_mib": 284208,
            "used_mib": 3208,
            "free_mib": 281000,
        }
    )

    admitted = lease._candidate_has_capacity(
        time.monotonic() + 0.05,
        candidates_remaining=1,
    )

    assert not admitted
    assert lease.gpu_memory_admission is None


def test_capacity_gate_reserves_numa_high_watermark_from_reclaimable_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _fake_gpu_lease_context(
        tmp_path,
        "2, 284208, 5208, 279000",
    )
    lease = GpuLease(
        context,
        "qwen3_omni",
        "exclusive_gpu",
        min_free_gpu_memory_mib=280000,
    )
    lease._gpu_numa_identity = (  # type: ignore[method-assign]
        lambda gpu, *, timeout_seconds: ("00000008:06:00.0", 1072, 2)
    )
    lease._gpu_has_compute_processes = (  # type: ignore[method-assign]
        lambda gpu, *, timeout_seconds: False
    )
    lease._node_meminfo = lambda node: {  # type: ignore[method-assign]
        "MemTotal": 289931264,
        "MemFree": 280100 * 1024,
        "Active(file)": 0,
        "Inactive(file)": 0,
        "Unevictable": 0,
        "Dirty": 0,
        "Writeback": 0,
        "Mapped": 0,
        "Shmem": 0,
    }
    lease._node_high_watermark_kib = (  # type: ignore[method-assign]
        lambda node, *, page_size_kib: 39184 * page_size_kib
    )
    monkeypatch.setattr("tools.ci.gpu_lease.os.sysconf", lambda name: 65536)

    snapshot = lease._gpu_memory_snapshot(2)

    assert snapshot == {
        "source": "linux-numa-meminfo",
        "total_mib": 283136,
        "used_mib": 5485,
        "free_mib": 277651,
    }
    assert snapshot["free_mib"] < lease.min_free_gpu_memory_mib


def test_capacity_gate_rejects_a_mismatched_gpu_numa_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _fake_gpu_lease_context(
        tmp_path,
        "2, 284208, 91595, 192613\n3, 284208, 124790, 159418",
    )
    lease = GpuLease(
        context,
        "qwen3_omni",
        "exclusive_gpu",
        min_free_gpu_memory_mib=280000,
    )
    lease._gpu_numa_identity = (  # type: ignore[method-assign]
        lambda gpu, *, timeout_seconds: ("00000008:06:00.0", 1072, 0)
    )
    lease._gpu_has_compute_processes = (  # type: ignore[method-assign]
        lambda gpu, *, timeout_seconds: False
    )
    lease._node_meminfo = lambda node: {  # type: ignore[method-assign]
        "MemTotal": 501667072,
        "MemFree": 432153600,
        "Active(file)": 3178496,
        "Inactive(file)": 10366784,
        "Unevictable": 0,
        "Dirty": 0,
        "Writeback": 0,
        "Mapped": 0,
        "Shmem": 0,
    }
    lease._node_high_watermark_kib = (  # type: ignore[method-assign]
        lambda node, *, page_size_kib: 168996 * page_size_kib
    )
    monkeypatch.setattr("tools.ci.gpu_lease.os.sysconf", lambda name: 65536)

    snapshot = lease._gpu_memory_snapshot(2)

    assert snapshot["source"] == "nvidia-smi"
    assert snapshot["free_mib"] == 192613
    assert any("total does not match" in warning for warning in lease.numa_fallback_warnings)


def test_gpu_numa_identity_uses_bounded_direct_topology_query(
    tmp_path: Path,
) -> None:
    context = _fake_gpu_lease_context(tmp_path, "2, 284208, 91595, 192613")
    context.env["FAKE_NVIDIA_SMI_IDENTITY_ROWS"] = "00000008:06:00.0, 1072"
    context.env["FAKE_NVIDIA_SMI_TOPOLOGY"] = "not-a-node"
    lease = GpuLease(
        context,
        "qwen3_omni",
        "exclusive_gpu",
        min_free_gpu_memory_mib=280000,
    )

    with pytest.raises(CiError, match="invalid NUMA ID"):
        lease._gpu_numa_identity(2, timeout_seconds=1)
    assert lease.gpu_numa_nodes == {}

    context.env["FAKE_NVIDIA_SMI_TOPOLOGY"] = "2"
    assert lease._gpu_numa_identity(2, timeout_seconds=1) == (
        "00000008:06:00.0",
        1072,
        2,
    )
    assert lease.gpu_numa_nodes == {"00000008:06:00.0": 2}


def test_gpu_numa_query_obeys_probe_deadline(tmp_path: Path) -> None:
    context = _fake_gpu_lease_context(tmp_path, "2, 284208, 91595, 192613")
    context.env["FAKE_NVIDIA_SMI_DELAY_SECONDS"] = "2"
    lease = GpuLease(
        context,
        "qwen3_omni",
        "exclusive_gpu",
        min_free_gpu_memory_mib=280000,
    )
    started = time.monotonic()

    with pytest.raises(CiError):
        lease._gpu_numa_node(2, timeout_seconds=0.05)

    assert time.monotonic() - started < 1


def test_gpu_compute_process_query_is_strict(tmp_path: Path) -> None:
    context = _fake_gpu_lease_context(tmp_path, "2, 284208, 91595, 192613")
    lease = GpuLease(
        context,
        "qwen3_omni",
        "exclusive_gpu",
        min_free_gpu_memory_mib=280000,
    )

    assert not lease._gpu_has_compute_processes(2, timeout_seconds=1)
    context.env["FAKE_NVIDIA_SMI_COMPUTE_ROWS"] = "123\n456"
    assert lease._gpu_has_compute_processes(2, timeout_seconds=1)
    context.env["FAKE_NVIDIA_SMI_COMPUTE_ROWS"] = "No running processes found"
    with pytest.raises(CiError, match="invalid compute process rows"):
        lease._gpu_has_compute_processes(2, timeout_seconds=1)


def test_gpu_numa_zoneinfo_parser_sums_high_watermarks() -> None:
    zoneinfo = """\
Node 2, zone      DMA
        high     0
Node 2, zone    DMA32
        high     0
Node 2, zone   Normal
        high     0
Node 2, zone  Movable
        high     39188
Node 2, zone   Device
        high     0
Node 3, zone  Movable
        high     123
"""

    assert (
        GpuLease._parse_node_high_watermark_kib(
            2,
            zoneinfo,
            page_size_kib=64,
        )
        == 2508032
    )


def test_capacity_gated_lease_waits_for_memory_reclaim_without_requeueing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Drive the short capacity settle window explicitly so host scheduling
    # cannot turn the second memory sample into an unrelated GPU requeue.
    clock = SimpleNamespace(now=0.0)

    def advance_clock(seconds: float) -> None:
        clock.now += seconds

    monkeypatch.setattr(
        gpu_lease_module,
        "time",
        SimpleNamespace(monotonic=lambda: clock.now, sleep=advance_clock),
    )
    context = _fake_gpu_lease_context(
        tmp_path,
        "2, 284208, 184208, 100000\n3, 284208, 174208, 110000",
        timeout_seconds=2,
    )
    lease = GpuLease(
        context,
        "qwen3_omni",
        "exclusive_gpu",
        min_free_gpu_memory_mib=240000,
    )
    prepared: list[int] = []
    snapshots = iter(
        (
            {
                "source": "nvidia-smi",
                "total_mib": 284208,
                "used_mib": 184208,
                "free_mib": 100000,
            },
            {
                "source": "nvidia-smi",
                "total_mib": 284208,
                "used_mib": 34208,
                "free_mib": 250000,
            },
        )
    )
    sampled: list[int] = []

    def memory_snapshot(
        gpu: int,
        *,
        timeout_seconds: float = 10.0,
    ) -> dict[str, object]:
        assert gpu == 2
        assert timeout_seconds <= 1
        sampled.append(gpu)
        assert _lock_is_busy(lease.lock_dir / "gpu-2-reservation.lock")
        assert all(
            _lock_is_busy(lease.lock_dir / f"gpu-2-slot-{slot}.lock")
            for slot in range(lease.slots_per_gpu)
        )
        assert not list(lease.lock_dir.glob("admission-global-*.lock"))
        return next(snapshots)

    lease._gpu_memory_snapshot = memory_snapshot  # type: ignore[method-assign]

    def prepare_candidate() -> None:
        assert lease.gpu_id is not None
        prepared.append(lease.gpu_id)
        assert _lock_is_busy(lease.lock_dir / "gpu-2-reservation.lock")
        assert all(
            _lock_is_busy(lease.lock_dir / f"gpu-2-slot-{slot}.lock")
            for slot in range(lease.slots_per_gpu)
        )
        assert not list(lease.lock_dir.glob("admission-global-*.lock"))

    try:
        lease.acquire(prepare_candidate=prepare_candidate)

        assert prepared == [2]
        assert sampled == [2, 2]
        assert lease.gpu_id == 2
        assert lease.gpu_memory_admission
        assert lease.gpu_memory_admission["observed_free_mib"] == 250000
    finally:
        lease.release()


@pytest.mark.model_proof_allocator
def test_capacity_waiter_reserves_one_gpu_without_blocking_other_gpu(
    tmp_path: Path,
) -> None:
    context = _fake_gpu_lease_context(
        tmp_path,
        "2, 284208, 34208, 250000\n3, 284208, 34208, 250000",
        timeout_seconds=10,
    )
    capacity_lease = GpuLease(
        context,
        "qwen3_omni",
        "exclusive_gpu",
        min_free_gpu_memory_mib=240000,
    )
    lock_dir = capacity_lease.lock_dir
    gpu2_holder = lock_dir / "gpu-2-slot-0.lock"
    gpu3_holder = lock_dir / "gpu-3-slot-0.lock"
    gpu2_holder.parent.mkdir(parents=True, exist_ok=True)
    acquired = threading.Event()
    failure: list[BaseException] = []
    capacity_thread: threading.Thread | None = None

    def acquire_capacity() -> None:
        try:
            capacity_lease.acquire()
            acquired.set()
        except BaseException as error:
            failure.append(error)
            acquired.set()

    try:
        with (
            gpu2_holder.open("w", encoding="utf-8") as gpu2_stream,
            gpu3_holder.open("w", encoding="utf-8") as gpu3_stream,
        ):
            fcntl.flock(gpu2_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(gpu3_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            capacity_thread = threading.Thread(target=acquire_capacity)
            capacity_thread.start()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not (
                _lock_is_busy(lock_dir / "gpu-2-reservation.lock")
                and not list(lock_dir.glob("admission-global-*.lock"))
            ):
                time.sleep(0.01)
            assert _lock_is_busy(lock_dir / "gpu-2-reservation.lock")
            assert not _lock_is_busy(lock_dir / "gpu-3-reservation.lock")
            assert not list(lock_dir.glob("admission-global-*.lock"))

            younger_env = context.env.copy()
            younger = GpuLease(CiContext(REPO_ROOT, younger_env), "convbert", "shared")
            try:
                younger.acquire()
                assert younger.gpu_id == 3
                assert younger.slot_ids == [1]
                assert not acquired.is_set()
            finally:
                younger.release()

            fcntl.flock(gpu2_stream, fcntl.LOCK_UN)
            capacity_thread.join(timeout=5)
            assert not capacity_thread.is_alive()
            assert not failure
            assert acquired.is_set()
            assert capacity_lease.gpu_id == 2
            assert capacity_lease.slot_ids == [0, 1]
    finally:
        if capacity_thread is not None:
            capacity_thread.join(timeout=11)
        capacity_lease.release()

    for gpu in (2, 3):
        assert not _lock_is_busy(lock_dir / f"gpu-{gpu}-reservation.lock")
        assert not _lock_is_busy(lock_dir / f"gpu-{gpu}-slot-0.lock")
        assert not _lock_is_busy(lock_dir / f"gpu-{gpu}-slot-1.lock")


@pytest.mark.model_proof_allocator
def test_capacity_waiter_switches_to_an_alternate_that_drains_first(
    tmp_path: Path,
) -> None:
    context = _fake_gpu_lease_context(
        tmp_path,
        "2, 284208, 34208, 250000\n3, 284208, 34208, 250000",
        timeout_seconds=3,
    )
    capacity_lease = GpuLease(
        context,
        "minimax_h3",
        "exclusive_gpu",
        min_free_gpu_memory_mib=184320,
    )
    younger_lease = GpuLease(
        CiContext(REPO_ROOT, context.env.copy()),
        "qwen_vl",
        "shared",
    )
    lock_dir = capacity_lease.lock_dir
    gpu2_holder = lock_dir / "gpu-2-slot-0.lock"
    gpu3_reservation = lock_dir / "gpu-3-reservation.lock"
    gpu3_slots = [lock_dir / f"gpu-3-slot-{slot}.lock" for slot in range(2)]
    lock_dir.mkdir(parents=True, exist_ok=True)
    acquired = threading.Event()
    candidate_reserved = threading.Event()
    younger_queued = threading.Event()
    younger_acquired = threading.Event()
    release_younger = threading.Event()
    failure: list[BaseException] = []
    younger_failure: list[BaseException] = []
    capacity_thread: threading.Thread | None = None
    younger_thread: threading.Thread | None = None

    reserve_candidate = capacity_lease._reserve_capacity_candidate
    create_younger_ticket = younger_lease._create_ticket

    def record_candidate(deadline: float, *, exclude: set[int]) -> bool:
        reserved = reserve_candidate(deadline, exclude=exclude)
        if reserved:
            candidate_reserved.set()
        return reserved

    capacity_lease._reserve_capacity_candidate = record_candidate  # type: ignore[method-assign]

    def record_younger_ticket(scope: str, deadline: float) -> gpu_lease_module.FileLock:
        ticket = create_younger_ticket(scope, deadline)
        younger_queued.set()
        return ticket

    younger_lease._create_ticket = record_younger_ticket  # type: ignore[method-assign]

    def acquire_capacity() -> None:
        try:
            capacity_lease.acquire()
        except BaseException as error:
            failure.append(error)
        finally:
            acquired.set()

    def acquire_younger() -> None:
        try:
            younger_lease.acquire()
            younger_acquired.set()
            release_younger.wait(timeout=6)
        except BaseException as error:
            younger_failure.append(error)
        finally:
            younger_lease.release()

    try:
        with (
            gpu2_holder.open("w", encoding="utf-8") as gpu2_stream,
            gpu3_reservation.open("w", encoding="utf-8") as gpu3_reservation_stream,
            gpu3_slots[0].open("w", encoding="utf-8") as gpu3_slot0_stream,
            gpu3_slots[1].open("w", encoding="utf-8") as gpu3_slot1_stream,
        ):
            fcntl.flock(gpu2_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(gpu3_reservation_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(gpu3_slot0_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(gpu3_slot1_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

            capacity_thread = threading.Thread(target=acquire_capacity)
            capacity_thread.start()
            assert candidate_reserved.wait(timeout=2)
            assert _lock_is_busy(lock_dir / "gpu-2-reservation.lock")
            assert not acquired.is_set()

            younger_thread = threading.Thread(target=acquire_younger)
            younger_thread.start()
            assert younger_queued.wait(timeout=2)
            assert not younger_acquired.is_set()

            fcntl.flock(gpu3_slot0_stream, fcntl.LOCK_UN)
            fcntl.flock(gpu3_slot1_stream, fcntl.LOCK_UN)
            fcntl.flock(gpu3_reservation_stream, fcntl.LOCK_UN)

            assert acquired.wait(timeout=4)
            capacity_thread.join(timeout=1)
            assert not capacity_thread.is_alive()
            assert not failure
            assert capacity_lease.gpu_id == 3
            assert capacity_lease.slot_ids == [0, 1]
            assert younger_acquired.wait(timeout=2)
            assert not younger_failure
            assert younger_lease.gpu_id == 2
            assert younger_lease.slot_ids == [1]
            assert not list(lock_dir.glob("admission-global-*.lock"))
            assert capacity_lease.evidence("a" * 40)["gpu_memory_admission"] == {
                "source": "nvidia-smi",
                "required_free_mib": 184320,
                "observed_total_mib": 284208,
                "observed_used_mib": 34208,
                "observed_free_mib": 250000,
            }
            assert _lock_is_busy(lock_dir / "gpu-3-reservation.lock")
            assert all(_lock_is_busy(path) for path in gpu3_slots)
            assert not _lock_is_busy(lock_dir / "gpu-2-reservation.lock")
            assert _lock_is_busy(lock_dir / "gpu-2-slot-1.lock")
            assert _lock_is_busy(gpu2_holder)
    finally:
        release_younger.set()
        if younger_thread is not None:
            younger_thread.join(timeout=7)
        if capacity_thread is not None:
            capacity_thread.join(timeout=4)
        younger_lease.release()
        capacity_lease.release()

    for gpu in (2, 3):
        assert not _lock_is_busy(lock_dir / f"gpu-{gpu}-reservation.lock")
        assert not _lock_is_busy(lock_dir / f"gpu-{gpu}-slot-0.lock")
        assert not _lock_is_busy(lock_dir / f"gpu-{gpu}-slot-1.lock")


@pytest.mark.model_proof_allocator
def test_capacity_waiter_timeout_releases_retained_ticket_and_partial_lease(
    tmp_path: Path,
) -> None:
    context = _fake_gpu_lease_context(
        tmp_path,
        "2, 284208, 34208, 250000\n3, 284208, 34208, 250000",
        timeout_seconds=1,
    )
    lease = GpuLease(
        context,
        "minimax_h3",
        "exclusive_gpu",
        min_free_gpu_memory_mib=184320,
    )
    lock_dir = lease.lock_dir
    gpu2_holder = lock_dir / "gpu-2-slot-0.lock"
    gpu3_reservation = lock_dir / "gpu-3-reservation.lock"
    gpu3_slots = [lock_dir / f"gpu-3-slot-{slot}.lock" for slot in range(2)]
    retained_ticket: list[Path] = []
    drain_reserved_gpu = lease._drain_reserved_gpu

    def observe_retained_ticket(
        deadline: float,
        *,
        exclude: set[int] | None = None,
    ) -> None:
        assert lease.hold_ticket_while_capacity_drains
        assert lease.ticket is not None
        assert _lock_is_busy(lease.ticket.path)
        retained_ticket.append(lease.ticket.path)
        drain_reserved_gpu(deadline, exclude=exclude)

    lease._drain_reserved_gpu = observe_retained_ticket  # type: ignore[method-assign]

    with (
        gpu2_holder.open("w", encoding="utf-8") as gpu2_stream,
        gpu3_reservation.open("w", encoding="utf-8") as gpu3_reservation_stream,
        gpu3_slots[0].open("w", encoding="utf-8") as gpu3_slot0_stream,
        gpu3_slots[1].open("w", encoding="utf-8") as gpu3_slot1_stream,
    ):
        fcntl.flock(gpu2_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu3_reservation_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu3_slot0_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu3_slot1_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

        with pytest.raises(
            CiError,
            match="timed out after 1s.*exclusive_gpu.*from: 2",
        ):
            lease.acquire()

        assert len(retained_ticket) == 1
        assert not retained_ticket[0].exists()
        assert lease.ticket is None
        assert lease.reservation is None
        assert lease.slots == []
        assert lease.slot_ids == []
        assert lease.gpu_id is None
        assert lease.machine is None
        assert not lease.hold_ticket_while_capacity_drains
        assert not _lock_is_busy(lock_dir / "gpu-2-reservation.lock")
        assert not _lock_is_busy(lock_dir / "gpu-2-slot-1.lock")
        assert _lock_is_busy(gpu2_holder)
        assert _lock_is_busy(gpu3_reservation)
        assert all(_lock_is_busy(path) for path in gpu3_slots)

    assert not list(lock_dir.glob("admission-global-*.lock"))
    for gpu in (2, 3):
        assert not _lock_is_busy(lock_dir / f"gpu-{gpu}-reservation.lock")
        assert not _lock_is_busy(lock_dir / f"gpu-{gpu}-slot-0.lock")
        assert not _lock_is_busy(lock_dir / f"gpu-{gpu}-slot-1.lock")


@pytest.mark.model_proof_allocator
def test_capacity_waiter_reconsiders_recovered_gpu_when_other_gpu_is_reserved(
    tmp_path: Path,
) -> None:
    context = _fake_gpu_lease_context(
        tmp_path,
        "2, 284208, 184208, 100000\n3, 284208, 174208, 110000",
    )
    lease = GpuLease(
        context,
        "qwen3_omni",
        "exclusive_gpu",
        min_free_gpu_memory_mib=240000,
    )
    gpu3_reservation = lease.lock_dir / "gpu-3-reservation.lock"
    gpu3_reservation.parent.mkdir(parents=True, exist_ok=True)
    sampled: list[int] = []

    def candidate_has_capacity(
        deadline: float,
        *,
        candidates_remaining: int,
    ) -> bool:
        assert lease.gpu_id == 2
        assert deadline > time.monotonic()
        assert candidates_remaining == 2
        sampled.append(2)
        lease.last_observed_total_mib[2] = 284208
        lease.last_observed_free_mib[2] = 100000
        if len(sampled) == 1:
            return False
        lease.gpu_memory_admission = {
            "source": "nvidia-smi",
            "required_free_mib": 240000,
            "observed_total_mib": 284208,
            "observed_used_mib": 34208,
            "observed_free_mib": 250000,
        }
        return True

    lease._candidate_has_capacity = candidate_has_capacity  # type: ignore[method-assign]

    with gpu3_reservation.open("w", encoding="utf-8") as reservation_stream:
        fcntl.flock(reservation_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            lease.acquire()
            assert sampled == [2, 2]
            assert lease.gpu_id == 2
            assert lease.gpu_memory_admission
            assert lease.gpu_memory_admission["observed_free_mib"] == 250000
        finally:
            lease.release()

    assert not list(lease.lock_dir.glob("admission-global-*.lock"))
    assert not _lock_is_busy(lease.lock_dir / "gpu-2-reservation.lock")
    assert not _lock_is_busy(lease.lock_dir / "gpu-2-slot-0.lock")
    assert not _lock_is_busy(lease.lock_dir / "gpu-2-slot-1.lock")


def test_capacity_gated_lease_times_out_with_last_observed_memory(
    tmp_path: Path,
) -> None:
    context = _fake_gpu_lease_context(
        tmp_path,
        "2, 284208, 184208, 100000\n3, 284208, 174208, 110000",
        timeout_seconds=3,
    )
    lease = GpuLease(
        context,
        "qwen3_omni",
        "exclusive_gpu",
        min_free_gpu_memory_mib=240000,
    )

    started = time.monotonic()
    with pytest.raises(CiError, match="240000 MiB.*GPU 2=100000 MiB.*GPU 3=110000 MiB"):
        lease.acquire()
    elapsed = time.monotonic() - started

    assert elapsed < 5
    assert not list(lease.lock_dir.glob("admission-global-*.lock"))
    for gpu in (2, 3):
        assert not _lock_is_busy(lease.lock_dir / f"gpu-{gpu}-reservation.lock")
        assert not _lock_is_busy(lease.lock_dir / f"gpu-{gpu}-slot-0.lock")
        assert not _lock_is_busy(lease.lock_dir / f"gpu-{gpu}-slot-1.lock")


def test_capacity_gate_fails_closed_when_nvidia_smi_fails(tmp_path: Path) -> None:
    context = _fake_gpu_lease_context(
        tmp_path,
        "2, 284208, 1000, 283208\n3, 284208, 1000, 283208",
    )
    context.env["FAKE_NVIDIA_SMI_EXIT_CODE"] = "1"
    lease = GpuLease(
        context,
        "qwen3_omni",
        "exclusive_gpu",
        min_free_gpu_memory_mib=240000,
    )

    with pytest.raises(CiError, match="Command failed"):
        lease.acquire()

    assert not list(lease.lock_dir.glob("admission-global-*.lock"))
    for gpu in (2, 3):
        assert not _lock_is_busy(lease.lock_dir / f"gpu-{gpu}-reservation.lock")
        assert not _lock_is_busy(lease.lock_dir / f"gpu-{gpu}-slot-0.lock")
        assert not _lock_is_busy(lease.lock_dir / f"gpu-{gpu}-slot-1.lock")


def test_capacity_gate_fails_fast_when_gpu_total_memory_is_too_small(
    tmp_path: Path,
) -> None:
    context = _fake_gpu_lease_context(
        tmp_path,
        "2, 200000, 1000, 199000\n3, 220000, 1000, 219000",
    )
    lease = GpuLease(
        context,
        "qwen3_omni",
        "exclusive_gpu",
        min_free_gpu_memory_mib=240000,
    )

    def unexpected_coherent_fallback(
        gpu: int,
        raw: dict[str, object],
        *,
        timeout_seconds: float,
    ) -> dict[str, object] | None:
        pytest.fail(f"GPU {gpu} cannot gain capacity beyond its {raw['total_mib']} MiB total")

    def unexpected_requeue(deadline: float) -> bool:
        pytest.fail(f"physically impossible capacity must not requeue before {deadline}")

    lease._coherent_gpu_memory_snapshot = unexpected_coherent_fallback  # type: ignore[method-assign]
    lease._requeue_after_capacity_rejection = unexpected_requeue  # type: ignore[method-assign]

    started = time.monotonic()
    with pytest.raises(
        CiError,
        match=("cannot meet.*240000 MiB.*GPU 2=200000 MiB.*GPU 3=220000 MiB"),
    ):
        lease.acquire()

    assert time.monotonic() - started < 2
    assert lease.last_observed_total_mib == {2: 200000, 3: 220000}
    assert lease.gpu_memory_admission is None
    assert not list(lease.lock_dir.glob("admission-global-*.lock"))


@pytest.mark.parametrize(
    "rows",
    (
        "malformed",
        "2, 284208, 1000, 283208\n2, 284208, 1000, 283208",
        "0, 284208, 1000, 283208\n1, 284208, 1000, 283208",
    ),
)
def test_capacity_gate_fails_closed_on_invalid_nvidia_smi_output(
    tmp_path: Path,
    rows: str,
) -> None:
    context = _fake_gpu_lease_context(tmp_path, rows)
    lease = GpuLease(
        context,
        "qwen3_omni",
        "exclusive_gpu",
        min_free_gpu_memory_mib=240000,
    )

    with pytest.raises(CiError, match="nvidia-smi"):
        lease.acquire()

    assert not list(lease.lock_dir.glob("admission-global-*.lock"))
    for gpu in (2, 3):
        assert not _lock_is_busy(lease.lock_dir / f"gpu-{gpu}-reservation.lock")
        assert not _lock_is_busy(lease.lock_dir / f"gpu-{gpu}-slot-0.lock")
        assert not _lock_is_busy(lease.lock_dir / f"gpu-{gpu}-slot-1.lock")


def test_capacity_candidate_callback_failure_releases_every_lock(
    tmp_path: Path,
) -> None:
    context = _fake_gpu_lease_context(
        tmp_path,
        "2, 284208, 1000, 283208\n3, 284208, 1000, 283208",
    )
    lease = GpuLease(
        context,
        "qwen3_omni",
        "exclusive_gpu",
        min_free_gpu_memory_mib=240000,
    )

    def reject_candidate() -> None:
        raise CiError("orphan cleanup failed")

    with pytest.raises(CiError, match="orphan cleanup failed"):
        lease.acquire(prepare_candidate=reject_candidate)

    assert not list(lease.lock_dir.glob("admission-global-*.lock"))
    for gpu in (2, 3):
        assert not _lock_is_busy(lease.lock_dir / f"gpu-{gpu}-reservation.lock")
        assert not _lock_is_busy(lease.lock_dir / f"gpu-{gpu}-slot-0.lock")
        assert not _lock_is_busy(lease.lock_dir / f"gpu-{gpu}-slot-1.lock")


@pytest.mark.model_proof_allocator
def test_capacity_drain_and_probe_share_one_deadline(
    tmp_path: Path,
) -> None:
    context = _fake_gpu_lease_context(
        tmp_path,
        "2, 284208, 1000, 283208",
        timeout_seconds=3,
    )
    context.env["TRTMC_MODEL_PROOF_GPU_IDS"] = "2"
    context.env["FAKE_NVIDIA_SMI_DELAY_SECONDS"] = "2"
    lease = GpuLease(
        context,
        "qwen3_omni",
        "exclusive_gpu",
        min_free_gpu_memory_mib=240000,
    )
    busy_slot = lease.lock_dir / "gpu-2-slot-0.lock"
    busy_slot.parent.mkdir(parents=True, exist_ok=True)

    with busy_slot.open("w", encoding="utf-8") as busy_stream:
        fcntl.flock(busy_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def finish_existing_work() -> None:
            time.sleep(2)
            fcntl.flock(busy_stream, fcntl.LOCK_UN)

        holder = threading.Thread(target=finish_existing_work)
        holder.start()
        started = time.monotonic()
        try:
            with pytest.raises(CiError, match="timed out after 3s"):
                lease.acquire()
        finally:
            holder.join(timeout=5)
            lease.release()
        elapsed = time.monotonic() - started

    assert not holder.is_alive()
    assert 2.5 <= elapsed < 3.7
    assert lease.gpu_id is None
    assert lease.gpu_memory_admission is None
    assert not lease.last_observed_total_mib
    assert not list(lease.lock_dir.glob("admission-global-*.lock"))
    assert not _lock_is_busy(lease.lock_dir / "gpu-2-reservation.lock")
    assert not _lock_is_busy(lease.lock_dir / "gpu-2-slot-0.lock")
    assert not _lock_is_busy(lease.lock_dir / "gpu-2-slot-1.lock")


@pytest.mark.model_proof_allocator
def test_explicit_gpu_id_cannot_bypass_a_busy_slot(tmp_path: Path) -> None:
    context = _fake_gpu_lease_context(tmp_path, "")
    context.env.update(
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
        lease = GpuLease(context, "convbert", "shared")
        started = time.monotonic()
        with pytest.raises(
            CiError,
            match="timed out after 1s waiting for a shared model-proof GPU lease from: 7",
        ):
            lease.acquire()
        lease_elapsed = time.monotonic() - started

    assert lease_elapsed < 5


@pytest.mark.model_proof_allocator
def test_gpu_lease_cannot_bypass_an_exclusive_whole_machine_lock(
    tmp_path: Path,
) -> None:
    context = _fake_gpu_lease_context(tmp_path, "")
    context.env.update(
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
        lease = GpuLease(context, "convbert", "shared")
        with pytest.raises(CiError, match="waiting for the whole-machine GPU lock"):
            lease.acquire()


def test_explicit_runner_gpu_must_be_in_the_configured_allowlist(tmp_path: Path) -> None:
    fake_bin, docker_log = _write_successful_fake_docker(tmp_path)
    output = tmp_path / "proof"
    env = _fake_proof_environment(tmp_path, fake_bin, docker_log, output)
    env.update({"TRTMC_GPU_ID": "7", "TRTMC_MODEL_PROOF_GPU_IDS": "0,1"})

    result = subprocess.run(
        [
            *RUNNER_COMMAND,
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
def test_four_shared_leases_use_unique_slots_and_reject_a_fifth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRTMC_GPU_ID", "7")
    monkeypatch.setenv("TRTMC_GPU_SLOT_ID", "3")
    processes: list[tuple[subprocess.Popen[str], Path]] = []
    release_file = tmp_path / "release-four-shared"
    common_env = {
        "TRTMC_MODEL_PROOF_GPU_IDS": "2",
        "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "4",
        "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": "180",
    }
    for index in range(4):
        process, output = _start_lease_case(
            tmp_path,
            f"case-{index}",
            "convbert",
            "shared",
            release_file,
            common_env,
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

    fifth_context = _fake_gpu_lease_context(tmp_path, "")
    fifth_context.env.update(common_env)
    fifth_context.env["TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS"] = "1"
    fifth = GpuLease(fifth_context, "convbert", "shared")
    with pytest.raises(
        CiError,
        match="timed out after 1s waiting for a shared model-proof GPU lease from: 2",
    ):
        fifth.acquire()

    release_file.touch()
    selected_slots: list[int] = []
    for process, output in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stdout + stderr
        assert (output / "artifacts" / "gpu-id.txt").is_file()
        lease = _gpu_lease(output)
        selected_slots.extend(lease["gpu_slots"])
        assert lease["gpu_id"] == "2"
        assert lease["resource_class"] == "shared"
        assert lease["slots_per_gpu"] == 4

    assert all_leased_together
    assert sorted(selected_slots) == [0, 1, 2, 3]


@pytest.mark.model_proof_allocator
def test_shared_slot_allocator_spreads_across_gpus_before_using_second_slots(
    tmp_path: Path,
) -> None:
    processes: list[tuple[subprocess.Popen[str], Path]] = []
    release_file = tmp_path / "release-spread"
    common_env = {
        "TRTMC_MODEL_PROOF_GPU_IDS": "2,3",
        "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "2",
        "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": "180",
    }
    for index in range(4):
        process, output = _start_lease_case(
            tmp_path,
            f"spread-{index}",
            "convbert",
            "shared",
            release_file,
            common_env,
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
            *RUNNER_COMMAND,
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
            *RUNNER_COMMAND,
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
            *RUNNER_COMMAND,
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
def test_poll_interval_cannot_sleep_past_the_lease_timeout(tmp_path: Path) -> None:
    """A poll interval larger than the lease budget must not delay the timeout.

    The capacity-poll sleep is clamped to the remaining deadline; without the
    clamp a 300s interval would sleep far past a 1s lease timeout while still
    reporting "timed out after 1s". The focused lease call excludes unrelated
    runner setup from the measured interval.
    """
    context = _fake_gpu_lease_context(tmp_path, "")
    context.env.update(
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
        lease = GpuLease(context, "convbert", "shared")
        started = time.monotonic()
        with pytest.raises(
            CiError,
            match="timed out after 1s waiting for a shared model-proof GPU lease from: 9",
        ):
            lease.acquire()
        lease_elapsed = time.monotonic() - started

    assert lease_elapsed < 10, (
        f"lease timeout was pierced by the poll interval: {lease_elapsed:.3f}s"
    )
    assert not list(lock_dir.glob("admission-global-*.lock"))


@pytest.mark.model_proof_allocator
def test_exclusive_drain_observes_release_before_timeout_with_long_poll(
    tmp_path: Path,
) -> None:
    context = _fake_gpu_lease_context(tmp_path, "", timeout_seconds=1)
    context.env.update(
        {
            "TRTMC_MODEL_PROOF_GPU_IDS": "9",
            "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "1",
            "TRTMC_MODEL_PROOF_POLL_INTERVAL": "300",
        }
    )
    lease = GpuLease(context, "minimax_h3", "exclusive_gpu")
    lock_dir = lease.lock_dir
    busy_slot = lock_dir / "gpu-9-slot-0.lock"
    reservation_acquired = threading.Event()
    holder_failure: list[BaseException] = []
    reserve_one_gpu = lease._reserve_one_gpu

    def record_reservation(deadline: float) -> bool:
        reserved = reserve_one_gpu(deadline)
        if reserved:
            reservation_acquired.set()
        return reserved

    lease._reserve_one_gpu = record_reservation  # type: ignore[method-assign]

    with busy_slot.open("w", encoding="utf-8") as busy_stream:
        fcntl.flock(busy_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def release_slot() -> None:
            try:
                assert reservation_acquired.wait(timeout=1)
                time.sleep(0.1)
                fcntl.flock(busy_stream, fcntl.LOCK_UN)
            except BaseException as error:
                holder_failure.append(error)

        holder = threading.Thread(target=release_slot)
        holder.start()
        try:
            lease.acquire()
            assert lease.gpu_id == 9
            assert lease.slot_ids == [0]
            assert _lock_is_busy(lock_dir / "gpu-9-reservation.lock")
            assert _lock_is_busy(busy_slot)
        finally:
            holder.join(timeout=2)
            lease.release()

        assert not holder.is_alive()
        assert not holder_failure

    assert not list(lock_dir.glob("admission-global-*.lock"))
    assert not _lock_is_busy(lock_dir / "gpu-9-reservation.lock")
    assert not _lock_is_busy(busy_slot)


@pytest.mark.model_proof_allocator
def test_gpu_admission_queue_prunes_a_stale_ticket(tmp_path: Path) -> None:
    context = _fake_gpu_lease_context(tmp_path, "")
    context.env.update(
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
    stale_handoff = lock_dir / "admission-global-00000000000000000000.lock.handoff.999999"
    stale_handoff.write_text("pid=999999 model=stale-handoff\n", encoding="utf-8")

    lease = GpuLease(context, "convbert", "shared")
    try:
        lease.acquire()
        assert lease.gpu_id == 9
    finally:
        lease.release()

    assert not stale_ticket.exists()
    assert not stale_handoff.exists()
    assert not list(lock_dir.glob("admission-global-*.lock"))
    assert (lock_dir / "admission-global.next").read_text(encoding="utf-8") == "2\n"


@pytest.mark.model_proof_allocator
def test_gpu_allocator_mutex_contention_obeys_lease_timeout(
    tmp_path: Path,
) -> None:
    context = _fake_gpu_lease_context(tmp_path, "")
    context.env.update(
        {
            "TRTMC_MODEL_PROOF_GPU_IDS": "9",
            "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": "1",
        }
    )
    lock_dir = tmp_path / "gpu-locks"
    lock_dir.mkdir()
    with (lock_dir / "allocator.lock").open("w", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lease = GpuLease(context, "convbert", "shared")
        started = time.monotonic()
        with pytest.raises(
            CiError,
            match="timed out after 1s waiting for a shared model-proof GPU lease",
        ):
            lease.acquire()
        elapsed = time.monotonic() - started

    assert elapsed < 5


@pytest.mark.model_proof_allocator
def test_exclusive_gpu_reservation_drains_shared_in_any_order_and_blocks_new_shared(
    tmp_path: Path,
) -> None:
    lock_dir = tmp_path / "gpu-locks"
    releases = [tmp_path / f"release-shared-{index}" for index in range(3)]
    exclusive_release = tmp_path / "release-exclusive"
    exclusive_release.touch()
    common_env = {
        "TRTMC_GPU_ID": "6",
        "TRTMC_MODEL_PROOF_GPU_IDS": "6",
        "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "3",
        "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": "600",
    }
    holders: list[subprocess.Popen[str] | None] = [None, None, None]
    exclusive: subprocess.Popen[str] | None = None
    try:
        for index in range(3):
            holder_env = dict(common_env)
            holder_env["TRTMC_GPU_SLOT_ID"] = str(index)
            holder, output = _start_lease_case(
                tmp_path,
                f"shared-{index}",
                "convbert",
                "shared",
                releases[index],
                holder_env,
            )
            holders[index] = holder
            deadline = time.monotonic() + 90
            while (
                time.monotonic() < deadline
                and not (output / "artifacts" / "gpu-lease.json").is_file()
            ):
                time.sleep(0.05)
            assert (output / "artifacts" / "gpu-lease.json").is_file()

        exclusive, exclusive_output = _start_lease_case(
            tmp_path,
            "exclusive",
            "flux",
            "exclusive_gpu",
            exclusive_release,
            common_env,
        )

        reservation = lock_dir / "gpu-6-reservation.lock"
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and not _lock_is_busy(reservation):
            time.sleep(0.05)
        assert _lock_is_busy(reservation), "exclusive proof never reserved GPU 6"
        assert not list(lock_dir.glob("admission-global-*.lock"))

        blocked_context = _fake_gpu_lease_context(tmp_path, "")
        blocked_context.env.update(common_env)
        blocked_context.env["TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS"] = "1"
        blocked = GpuLease(blocked_context, "convbert", "shared")
        with pytest.raises(CiError, match="waiting for a shared model-proof GPU lease"):
            blocked.acquire()

        for index in (1, 0, 2):
            releases[index].touch()
            holder = holders[index]
            assert holder is not None
            holder_stdout, holder_stderr = holder.communicate(timeout=30)
            assert holder.returncode == 0, holder_stdout + holder_stderr

        exclusive_stdout, exclusive_stderr = exclusive.communicate(timeout=30)
        assert exclusive.returncode == 0, exclusive_stdout + exclusive_stderr
        assert _gpu_lease(exclusive_output)["resource_class"] == "exclusive_gpu"
        assert _gpu_lease(exclusive_output)["gpu_slots"] == [0, 1, 2]
        assert not _lock_is_busy(reservation)
        for slot in range(3):
            assert not _lock_is_busy(lock_dir / f"gpu-6-slot-{slot}.lock")
    finally:
        for release_file in releases:
            release_file.touch()
        _finish_proof_cases([*holders, exclusive])


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
        resource_class: str,
        release_file: Path,
        *,
        explicit_gpu: str | None = None,
    ) -> tuple[subprocess.Popen[str], Path]:
        env = {
            "TRTMC_MODEL_PROOF_GPU_IDS": "6,7",
            "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": "4",
            "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": "600",
        }
        if explicit_gpu is not None:
            env["TRTMC_GPU_ID"] = explicit_gpu
        process, output = _start_lease_case(
            tmp_path,
            name,
            model,
            resource_class,
            release_file,
            env,
        )
        processes.append(process)
        release_files.append(release_file)
        return process, output

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
        gpu6_holder, gpu6_output = start_case(
            "gpu6-holder", "convbert", "shared", gpu6_release, explicit_gpu="6"
        )
        wait_for(
            lambda: (gpu6_output / "artifacts" / "gpu-lease.json").is_file(),
            "GPU 6 holder never acquired its lease",
        )
        gpu7_holder, gpu7_output = start_case(
            "gpu7-holder", "albert", "shared", gpu7_release, explicit_gpu="7"
        )
        wait_for(
            lambda: (gpu7_output / "artifacts" / "gpu-lease.json").is_file(),
            "GPU 7 holder never acquired its lease",
        )

        exclusive, exclusive_output = start_case(
            "oldest-exclusive", "flux", "exclusive_gpu", exclusive_release
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

        younger, younger_output = start_case(
            "younger-shared",
            "convbert",
            "shared",
            younger_release,
            explicit_gpu="7",
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
        wait_for(
            lambda: (younger_output / "artifacts" / "gpu-lease.json").is_file(),
            "younger shared proof did not run after the exclusive proof",
        )
        assert _gpu_lease(younger_output)["gpu_id"] == "7"
        younger_release.touch()
        younger_stdout, younger_stderr = younger.communicate(timeout=30)
        assert younger.returncode == 0, younger_stdout + younger_stderr

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
        name: str,
        model: str,
        resource_class: str,
        release_file: Path,
    ) -> tuple[subprocess.Popen[str], Path]:
        return _start_lease_case(
            tmp_path,
            name,
            model,
            resource_class,
            release_file,
            common_env,
        )

    first_release = tmp_path / "release-first-exclusive"
    oldest_release = tmp_path / "release-oldest-shared"
    younger_shared_release = tmp_path / "release-younger-shared"
    younger_exclusive_release = tmp_path / "release-younger-exclusive"
    first, first_output = start_case("first-exclusive", "flux", "exclusive_gpu", first_release)
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

        oldest, oldest_output = start_case("oldest-shared", "albert", "shared", oldest_release)
        deadline = time.monotonic() + coordination_timeout_s
        while time.monotonic() < deadline and not list(lock_dir.glob("admission-global-*.lock")):
            time.sleep(0.05)
        admission_tickets = sorted(lock_dir.glob("admission-global-*.lock"))
        assert len(admission_tickets) == 1
        assert "model=albert" in admission_tickets[0].read_text(encoding="utf-8")
        assert _lock_is_busy(admission_tickets[0])

        younger_exclusive, younger_exclusive_output = start_case(
            "younger-exclusive",
            "bark",
            "exclusive_gpu",
            younger_exclusive_release,
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
        ] == ["albert", "bark"]
        younger_shared, younger_shared_output = start_case(
            "younger-shared",
            "convbert",
            "shared",
            younger_shared_release,
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
        ] == ["albert", "bark", "convbert"]

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


def _start_lease_case(
    tmp_path: Path,
    name: str,
    model: str,
    resource_class: str,
    release_file: Path,
    common_env: dict[str, str],
) -> tuple[subprocess.Popen[str], Path]:
    case_dir = tmp_path / name
    case_dir.mkdir()
    output = case_dir / "lease"
    env = os.environ.copy()
    env.pop("TRTMC_GPU_ID", None)
    env.pop("TRTMC_GPU_SLOT_ID", None)
    env.update(
        {
            "TRTMC_MODEL_PROOF_GPU_LOCK_DIR": str(tmp_path / "gpu-locks"),
            "TRTMC_MODEL_PROOF_POLL_INTERVAL": "0.05",
            "TRTMC_MODEL_PROOF_FLOCK_WATCHDOG_SECONDS": "2",
        }
    )
    env.update(common_env)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            GPU_LEASE_WORKER,
            "--model",
            model,
            "--resource-class",
            resource_class,
            "--artifacts-dir",
            str(output / "artifacts"),
            "--release-file",
            str(release_file),
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
    holder, holder_output = _start_lease_case(
        tmp_path, "holder", "convbert", "shared", holder_release, common_env
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

        waiter, waiter_output = _start_lease_case(
            tmp_path, "waiter", "convbert", "shared", waiter_release, common_env
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
    holder, holder_output = _start_lease_case(
        tmp_path, "holder", "albert", "shared", holder_release, common_env
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

        doomed, _ = _start_lease_case(
            tmp_path, "doomed", "convbert", "shared", waiter_release, common_env
        )
        deadline = time.monotonic() + coordination_timeout_s
        while (
            time.monotonic() < deadline and len(list(lock_dir.glob("admission-global-*.lock"))) < 1
        ):
            time.sleep(0.05)
        assert len(list(lock_dir.glob("admission-global-*.lock"))) == 1

        waiter, waiter_output = _start_lease_case(
            tmp_path, "waiter", "convbert", "shared", waiter_release, common_env
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
    holder, holder_output = _start_lease_case(
        tmp_path, "holder", "albert", "shared", holder_release, common_env
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
            process, output = _start_lease_case(
                tmp_path,
                f"waiter-{index}",
                "convbert",
                "shared",
                waiter_release,
                common_env,
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


def test_runner_keeps_local_hugging_face_cache_fallbacks() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert (
        'self.context.env.get("HF_HOME", str(Path.home() / ".cache/huggingface"))'
        in source
    )
    assert 'self.context.env.get("TRTMC_HF_HUB_CACHE"' in source
    assert "TRTMC_HF_MODULES_CACHE" not in source


def test_model_proof_always_generates_strict_self_contained_html() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    inner = (REPO_ROOT / "tools/ci/model_proof_inner.py").read_text(
        encoding="utf-8"
    )
    for contract in (
        "report_rc = self._finalize_report(validation_rc)",
        'self.source / "scripts/generate_e2e_report.py"',
        '"--artifacts-dir"',
        'self.artifacts / "e2e"',
        'self.artifacts / "model-proof-report.html"',
        '"--project-dir"',
        "self.source",
        'self.artifacts / "model-proof-status.json"',
        'self.artifacts / "proof.json"',
        'self.artifacts / "selection.json"',
        '"--strict-evidence"',
        '"--max-embed-bytes"',
        '"33554432"',
        "f\"--junitxml={self.artifacts / 'e2e/junit.xml'}\"",
    ):
        assert contract in inner

    assert "self._fallback_report()" in runner
    assert 'raise CiError(f"model proof did not emit {name}")' in runner
    assert 'self.payload["validation_exit_code"] = returncode' in inner
    assert 'self.payload["report_exit_code"] = report_rc' in inner


def test_gpu_mapping_exists_only_on_the_hermetic_proof_container() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    allocator = (REPO_ROOT / "tools/ci/gpu_lease.py").read_text(encoding="utf-8")
    inner = (REPO_ROOT / "tools/ci/model_proof_inner.py").read_text(
        encoding="utf-8"
    )
    host = source.split("def _run_host(self)", maxsplit=1)[1]
    warm = source.split("def _prepare_hf_cache(", maxsplit=1)[1].split(
        "def _validated_cache_evidence", maxsplit=1
    )[0]
    proof = source.split("def _run_proof_container(", maxsplit=1)[1].split(
        "def _proof_environment", maxsplit=1
    )[0]

    assert host.index("_prepare_hf_cache") < host.index("GpuLease(")
    assert "--gpus" not in warm
    assert "TRTMC_MODEL_PROOF_GPU_ID" not in warm
    assert '"--gpus"' in proof
    assert 'f"device={self.lease.gpu_id}"' in proof
    assert '"TRTMC_MODEL_PROOF_GPU_ID": str(self.lease.gpu_id)' in source
    assert '"TRTMC_MODEL_PROOF_GPU_SLOT_IDS": slots' in source
    assert (
        '"TRTMC_MODEL_PROOF_RESOURCE_CLASS": self.lease.resource_class'
        in source
    )
    assert 'f"gpu-{gpu}-slot-{slot}.lock"' in allocator
    assert 'f"gpu-{gpu}-reservation.lock"' in allocator
    assert "TRTMC_GPU_ID must be present in TRTMC_MODEL_PROOF_GPU_IDS" in allocator
    assert '"gpu_lease_evidence": "gpu-lease.json"' in inner
