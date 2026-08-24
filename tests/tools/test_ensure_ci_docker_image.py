# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for CI Docker image resolution and validation."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.ci.docker_image import CiError, DockerImageManager, WorkflowImageLock


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "tools" / "ci" / "docker_image.py"
DEFAULT_PROFILES = (
    "chronos,deepseek_ocr,elf_flow,elf_flow_reference,internlm,lance_reference,magpie_tts_reference,"
    "nemotron_h_reference,phi4_multimodal,reference_common,sana_wm_reference"
)
TENSORRT_VERSION = "11.1.0.106"
TENSORRT_APT_VERSION = "11.1.0.106-1+cuda13.3"
TENSORRT_DISTRIBUTIONS = {
    name: TENSORRT_VERSION
    for name in (
        "tensorrt",
        "tensorrt_cu13",
        "tensorrt_cu13_bindings",
        "tensorrt_cu13_libs",
    )
}
TENSORRT_APT_PACKAGES = {
    name: TENSORRT_APT_VERSION
    for name in (
        "libnvinfer-dev",
        "libnvinfer-headers-dev",
        "libnvinfer-headers-plugin-dev",
        "libnvinfer-safe-headers-dev",
        "libnvinfer11",
        "libnvonnxparsers-dev",
        "libnvonnxparsers11",
    )
}


def _record_lock_open_modes(monkeypatch, lock_path: Path, *, lose_create_race=False) -> list[str]:
    modes: list[str] = []
    path_open = Path.open

    def traced_open(path: Path, mode: str = "r", *args, **kwargs):
        if path == lock_path:
            modes.append(mode)
            if mode == "x+" and lose_create_race:
                path_open(path, mode, *args, **kwargs).close()
                raise FileExistsError(lock_path)
        return path_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", traced_open)
    return modes


def test_workflow_image_lock_opens_existing_file_without_create(
    tmp_path: Path, monkeypatch
) -> None:
    lock_path = tmp_path / "ci-image.lock"
    lock_path.touch()
    modes = _record_lock_open_modes(monkeypatch, lock_path)

    with WorkflowImageLock(lock_path, timeout=1):
        pass

    assert modes == ["r+"]


def test_workflow_image_lock_creates_missing_file(tmp_path: Path, monkeypatch) -> None:
    lock_path = tmp_path / "ci-image.lock"
    modes = _record_lock_open_modes(monkeypatch, lock_path)

    with WorkflowImageLock(lock_path, timeout=1):
        assert lock_path.is_file()

    assert modes == ["r+", "x+"]


def test_workflow_image_lock_retries_when_another_runner_creates_file(
    tmp_path: Path, monkeypatch
) -> None:
    lock_path = tmp_path / "ci-image.lock"
    modes = _record_lock_open_modes(monkeypatch, lock_path, lose_create_race=True)

    with WorkflowImageLock(lock_path, timeout=1):
        pass

    assert modes == ["r+", "x+", "r+"]


def test_default_image_lock_wait_covers_one_profile_install_budget() -> None:
    manager = DockerImageManager(REPO_ROOT, {})

    assert manager.config.lock_timeout >= 7200


def _write_fake_docker(tmp_path: Path, existing_images: dict[str, str]) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    state_path = tmp_path / "images"
    state_path.write_text(
        "".join(f"{image}|{fingerprint}\n" for image, fingerprint in existing_images.items())
    )
    log_path = tmp_path / "docker.log"
    docker_path = bin_dir / "docker"
    docker_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%q ' "$@" >> "$FAKE_DOCKER_LOG"
printf '\n' >> "$FAKE_DOCKER_LOG"

if [ "${1:-}" = "image" ] && [ "${2:-}" = "inspect" ]; then
  image="${!#}"
  record="$(awk -F'|' -v image="$image" '$1 == image { print; exit }' "$FAKE_DOCKER_STATE")"
  [ -n "$record" ] || exit 1
  if [[ " $* " == *" --format "* ]]; then
    if [[ " $* " == *"{{.Id}}"* ]]; then
      printf 'sha256:%s\n' "${record#*|}"
    else
      printf '%s\n' "${record#*|}"
    fi
  fi
  exit 0
fi

if [ "${1:-}" = "run" ]; then
  capability="${FAKE_DOCKER_CAPABILITY:-available}"
  profiles="${FAKE_DOCKER_PROFILES-chronos,deepseek_ocr,elf_flow,elf_flow_reference,internlm,lance_reference,magpie_tts_reference,nemotron_h_reference,phi4_multimodal,reference_common,sana_wm_reference}"
  if [ -f "$FAKE_DOCKER_REBUILT" ]; then
    capability="available"
    profiles="$FAKE_DOCKER_REBUILT_PROFILES"
  fi
  cat <<'EOF'
TENSORRT_VERSION=11.1.0.106
TENSORRT_PYTHON_DISTRIBUTIONS={"tensorrt":"11.1.0.106","tensorrt_cu13":"11.1.0.106","tensorrt_cu13_bindings":"11.1.0.106","tensorrt_cu13_libs":"11.1.0.106"}
TENSORRT_APT_PACKAGES={"libnvinfer-dev":"11.1.0.106-1+cuda13.3","libnvinfer-headers-dev":"11.1.0.106-1+cuda13.3","libnvinfer-headers-plugin-dev":"11.1.0.106-1+cuda13.3","libnvinfer-safe-headers-dev":"11.1.0.106-1+cuda13.3","libnvinfer11":"11.1.0.106-1+cuda13.3","libnvonnxparsers-dev":"11.1.0.106-1+cuda13.3","libnvonnxparsers11":"11.1.0.106-1+cuda13.3"}
TENSORRT_HEADER_VERSION=11.1.0.106
TENSORRT_OVERLAY_FILES=present
TENSORRT_NATIVE_VERSION=11.1.0.106
MODELOPT_VERSION=0.44.0
NLOHMANN_JSON_HEADER=present
EOF
  printf 'NEMO_PROMPT_RNNT=%s\n' "$capability"
  printf 'PYTHON_PROFILES=%s\n' "$profiles"
  exit 0
fi

if [ "${1:-}" = "build" ]; then
  shift
  tag=""
  label=""
  while [ "$#" -gt 0 ]; do
    case "$1" in
      -t)
        tag="$2"
        shift 2
        ;;
      --label)
        label="$2"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done
  [ -n "$tag" ]
  fingerprint="${label#*=}"
  printf '%s|%s\n' "$tag" "$fingerprint" >> "$FAKE_DOCKER_STATE"
  touch "$FAKE_DOCKER_REBUILT"
  exit 0
fi

echo "unsupported fake docker invocation: $*" >&2
exit 2
"""
    )
    docker_path.chmod(0o755)
    return bin_dir, log_path


def _run_ensure_script(
    tmp_path: Path,
    *,
    existing_images: dict[str, str],
    capability: str = "available",
    profiles: str = DEFAULT_PROFILES,
    rebuilt_profiles: str = DEFAULT_PROFILES,
    changed_paths: tuple[str, ...] = (),
    repo_root: Path = REPO_ROOT,
    run_id: str = "",
    verification_dir: Path | None = None,
    github_output: Path | None = None,
) -> tuple[subprocess.CompletedProcess[str], str, str]:
    bin_dir, log_path = _write_fake_docker(tmp_path, existing_images)
    if changed_paths:
        git_path = bin_dir / "git"
        git_path.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
if [ "${1:-}" = "cat-file" ]; then
  exit 0
fi
if [ "${1:-}" = "diff" ]; then
  printf '%s\n' "$FAKE_GIT_CHANGED_PATHS"
  exit 0
fi
echo "unsupported fake git invocation: $*" >&2
exit 2
"""
        )
        git_path.chmod(0o755)
    github_env = tmp_path / "github.env"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "FAKE_DOCKER_LOG": str(log_path),
            "FAKE_DOCKER_STATE": str(tmp_path / "images"),
            "FAKE_DOCKER_CAPABILITY": capability,
            "FAKE_DOCKER_PROFILES": profiles,
            "FAKE_DOCKER_REBUILT_PROFILES": rebuilt_profiles,
            "FAKE_DOCKER_REBUILT": str(tmp_path / "rebuilt"),
            "GITHUB_ENV": str(github_env),
            "RUNNER_TEMP": str(tmp_path / "runner-temp"),
            "TRTMC_CI_IMAGE_LOCK_FILE": str(tmp_path / "ci-image.lock"),
            "TRTMC_CI_IMAGE": "trtmc-dev-gb300:manylinux_2_39",
            "CI_BASE_REF": "fake-base" if changed_paths else "",
            "FAKE_GIT_CHANGED_PATHS": "\n".join(changed_paths),
            "GITHUB_RUN_ID": run_id,
            "GITHUB_RUN_ATTEMPT": "1",
        }
    )
    if verification_dir is not None:
        env["TRTMC_CI_IMAGE_VERIFICATION_DIR"] = str(verification_dir)
    if github_output is not None:
        env["GITHUB_OUTPUT"] = str(github_output)
    result = subprocess.run(
        [sys.executable, "-m", "tools.ci", "image", "ensure"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    github_env_text = github_env.read_text() if github_env.exists() else ""
    docker_log = log_path.read_text() if log_path.exists() else ""
    return result, github_env_text, docker_log


def _write_profile_fingerprint_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create the smallest source tree needed to exercise image fingerprinting."""
    repo_root = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / ".github" / "scripts", repo_root / ".github" / "scripts")
    shutil.copytree(REPO_ROOT / "tools" / "ci", repo_root / "tools" / "ci")
    shutil.copy2(REPO_ROOT / "tools" / "__init__.py", repo_root / "tools" / "__init__.py")

    package_root = repo_root / "python" / "tensorrt_model_connect"
    families_root = package_root / "families"
    demo_root = families_root / "demo"
    demo_root.mkdir(parents=True)
    for source in (
        REPO_ROOT / "python" / "tensorrt_model_connect" / "__init__.py",
        REPO_ROOT / "python" / "tensorrt_model_connect" / "python_profiles.py",
    ):
        shutil.copy2(source, package_root / source.name)
    shutil.copy2(
        REPO_ROOT / "python" / "tensorrt_model_connect" / "families" / "__init__.py",
        families_root / "__init__.py",
    )

    manifest = demo_root / "MODEL.toml"
    manifest.write_text(
        """# Synthetic family used only by the image-fingerprint tests.
id = "demo"
plugin = "demo"
module = "plugin"
aliases = ["demo"]
prefixes = ["demo"]
default_execution_profiles = ["reference|demo"]
python_profile_specs = [
  "demo|families/demo/requirements.lock.txt|families/demo/verify.py|true",
  "lazy_demo|families/demo/requirements.lock.txt|families/demo/verify.py|true|false",
]
""",
        encoding="utf-8",
    )
    profile_registry = package_root / "python_profiles.toml"
    profile_registry.write_text(
        """version = 1

[profiles.base]
kind = "passthrough"

[profiles.reference_common]
kind = "venv"
requirements = "families/demo/requirements.lock.txt"
system_site_packages = true
verification_script = "import demo_package"

""",
        encoding="utf-8",
    )
    requirements = demo_root / "requirements.lock.txt"
    requirements.write_text("demo-package==1.0.0\n", encoding="utf-8")
    (demo_root / "verify.py").write_text("import demo_package\n", encoding="utf-8")

    (repo_root / "Dockerfile").write_text(
        "ARG TENSORRT_VERSION=11.1.0.106\n"
        "ARG TENSORRT_APT_VERSION=11.1.0.106-1+cuda13.3\n"
        "ARG MODELOPT_VERSION=0.44.0\n",
        encoding="utf-8",
    )
    return repo_root, manifest, requirements


def _resolved_image_for_repo(
    tmp_path: Path,
    repo_root: Path,
) -> tuple[str, str]:
    result, github_env, docker_log = _run_ensure_script(
        tmp_path,
        existing_images={},
        profiles="demo,reference_common",
        rebuilt_profiles="demo,reference_common",
        repo_root=repo_root,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    match = re.search(r"^TRTMC_CI_IMAGE=(.+)$", github_env, re.MULTILINE)
    assert match, github_env
    fingerprint_match = re.search(
        r"--label org\.nvidia\.trtmc\.ci-input-fingerprint=([0-9a-f]{64})",
        docker_log,
    )
    assert fingerprint_match, docker_log
    return match.group(1), fingerprint_match.group(1)


def test_missing_fingerprint_image_builds_and_exports_resolved_tag(tmp_path: Path) -> None:
    base_image = "trtmc-dev-gb300:manylinux_2_39"

    result, github_env, docker_log = _run_ensure_script(
        tmp_path,
        existing_images={base_image: "legacy-image-without-source-provenance"},
    )

    assert result.returncode == 0, result.stderr
    match = re.search(
        rf"^TRTMC_CI_IMAGE=({re.escape(base_image)}-[0-9a-f]{{12}})$",
        github_env,
        re.MULTILINE,
    )
    assert match, github_env
    resolved_image = match.group(1)
    assert f"-t {resolved_image}" in docker_log


def test_missing_required_capability_rebuilds_matching_image(tmp_path: Path) -> None:
    bootstrap_result, bootstrap_env, bootstrap_log = _run_ensure_script(
        tmp_path / "bootstrap",
        existing_images={},
    )
    assert bootstrap_result.returncode == 0, bootstrap_result.stderr
    resolved_image = re.search(r"^TRTMC_CI_IMAGE=(.+)$", bootstrap_env, re.MULTILINE).group(1)
    fingerprint = re.search(
        r"--label org\.nvidia\.trtmc\.ci-input-fingerprint=([0-9a-f]{64})",
        bootstrap_log,
    ).group(1)

    result, _, docker_log = _run_ensure_script(
        tmp_path / "capability-missing",
        existing_images={resolved_image: fingerprint},
        capability="missing",
    )

    assert result.returncode == 0, result.stderr
    assert f"-t {resolved_image}" in docker_log
    assert "required NeMo prompt RNN-T capability is missing" in result.stdout


def test_matching_fingerprint_image_is_reused_for_pr_rerun(tmp_path: Path) -> None:
    bootstrap_result, bootstrap_env, bootstrap_log = _run_ensure_script(
        tmp_path / "bootstrap",
        existing_images={},
    )
    assert bootstrap_result.returncode == 0, bootstrap_result.stderr
    resolved_image = re.search(r"^TRTMC_CI_IMAGE=(.+)$", bootstrap_env, re.MULTILINE).group(1)
    fingerprint = re.search(
        r"--label org\.nvidia\.trtmc\.ci-input-fingerprint=([0-9a-f]{64})",
        bootstrap_log,
    ).group(1)

    result, _, docker_log = _run_ensure_script(
        tmp_path / "matching",
        existing_images={resolved_image: fingerprint},
        changed_paths=("tools/ci/docker_image.py",),
    )

    assert result.returncode == 0, result.stderr
    assert "build " not in docker_log
    assert f"CI Docker image '{resolved_image}' already matches" in result.stdout


def test_reused_image_creates_missing_github_output_parent(tmp_path: Path) -> None:
    bootstrap_result, bootstrap_env, bootstrap_log = _run_ensure_script(
        tmp_path / "bootstrap",
        existing_images={},
    )
    assert bootstrap_result.returncode == 0, bootstrap_result.stderr
    resolved_image = re.search(r"^TRTMC_CI_IMAGE=(.+)$", bootstrap_env, re.MULTILINE).group(1)
    fingerprint = re.search(
        r"--label org\.nvidia\.trtmc\.ci-input-fingerprint=([0-9a-f]{64})",
        bootstrap_log,
    ).group(1)
    github_output = tmp_path / "missing-file-command-dir" / "set_output"

    result, _, _ = _run_ensure_script(
        tmp_path / "matching",
        existing_images={resolved_image: fingerprint},
        changed_paths=("tools/ci/docker_image.py",),
        github_output=github_output,
    )

    assert result.returncode == 0, result.stderr
    assert github_output.read_text(encoding="utf-8") == f"image_ref=sha256:{fingerprint}\n"


def test_matching_image_is_fully_validated_once_per_workflow_run(tmp_path: Path) -> None:
    verification_dir = tmp_path / "verification"
    bootstrap_result, bootstrap_env, bootstrap_log = _run_ensure_script(
        tmp_path / "bootstrap",
        existing_images={},
        run_id="12345",
        verification_dir=verification_dir,
    )
    assert bootstrap_result.returncode == 0, bootstrap_result.stderr
    resolved_image = re.search(r"^TRTMC_CI_IMAGE=(.+)$", bootstrap_env, re.MULTILINE).group(1)
    fingerprint = re.search(
        r"--label org\.nvidia\.trtmc\.ci-input-fingerprint=([0-9a-f]{64})",
        bootstrap_log,
    ).group(1)

    result, _, docker_log = _run_ensure_script(
        tmp_path / "sibling",
        existing_images={resolved_image: fingerprint},
        run_id="12345",
        verification_dir=verification_dir,
    )

    assert result.returncode == 0, result.stderr
    assert " run " not in f" {docker_log} "
    assert "reused from this workflow run's verified image" in result.stdout


def test_missing_prebuilt_profiles_rebuilds_the_image(tmp_path: Path) -> None:
    bootstrap_result, bootstrap_env, bootstrap_log = _run_ensure_script(
        tmp_path / "bootstrap",
        existing_images={},
    )
    assert bootstrap_result.returncode == 0, bootstrap_result.stderr
    resolved_image = re.search(r"^TRTMC_CI_IMAGE=(.+)$", bootstrap_env, re.MULTILINE).group(1)
    fingerprint = re.search(
        r"--label org\.nvidia\.trtmc\.ci-input-fingerprint=([0-9a-f]{64})",
        bootstrap_log,
    ).group(1)

    result, _, docker_log = _run_ensure_script(
        tmp_path / "profiles-missing",
        existing_images={resolved_image: fingerprint},
        profiles="",
    )

    assert result.returncode == 0, result.stderr
    assert f"-t {resolved_image}" in docker_log
    assert "prebuilt Python profiles differ" in result.stdout


def test_tensorrt_overlay_contract_reports_each_mismatch(tmp_path: Path) -> None:
    repo_root, _, _ = _write_profile_fingerprint_repo(tmp_path)
    manager = DockerImageManager(repo_root)
    requirements = manager._read_requirements()
    actual = {
        "TENSORRT_VERSION": TENSORRT_VERSION,
        "TENSORRT_PYTHON_DISTRIBUTIONS": json.dumps(
            TENSORRT_DISTRIBUTIONS, separators=(",", ":"), sort_keys=True
        ),
        "TENSORRT_APT_PACKAGES": json.dumps(
            TENSORRT_APT_PACKAGES, separators=(",", ":"), sort_keys=True
        ),
        "TENSORRT_HEADER_VERSION": TENSORRT_VERSION,
        "TENSORRT_OVERLAY_FILES": "present",
        "TENSORRT_NATIVE_VERSION": TENSORRT_VERSION,
        "MODELOPT_VERSION": "0.44.0",
        "NLOHMANN_JSON_HEADER": "present",
        "NEMO_PROMPT_RNNT": "available",
        "PYTHON_PROFILES": "demo,reference_common",
    }
    mismatches = (
        ("TENSORRT_PYTHON_DISTRIBUTIONS", "{}", "Python distribution versions"),
        ("TENSORRT_APT_PACKAGES", "{}", "APT package versions"),
        ("TENSORRT_HEADER_VERSION", "11.1.0.105", "C++ header version"),
        ("TENSORRT_OVERLAY_FILES", "missing", "header or native library is missing"),
        ("TENSORRT_NATIVE_VERSION", "11.1.0.105", "native runtime version"),
    )

    for key, wrong_value, reason in mismatches:
        observed = {**actual, key: wrong_value}
        assert any(reason in item for item in manager._version_mismatches(observed, requirements))


def test_source_contract_describes_parameterized_tensorrt_overlay(tmp_path: Path) -> None:
    __import__("tensorrt_model_connect.python_profiles")
    repo_root, _, _ = _write_profile_fingerprint_repo(tmp_path)
    manager = DockerImageManager(repo_root)

    default_contract = manager.source_contract()
    selected_json = manager.source_contract_json(
        tensorrt_version="11.2.1.2",
        tensorrt_apt_version="11.2.1.2-1+cuda13.3",
    )
    selected_contract = json.loads(selected_json)

    assert selected_contract == manager.source_contract(
        tensorrt_version="11.2.1.2",
        tensorrt_apt_version="11.2.1.2-1+cuda13.3",
    )
    assert selected_contract["schema_version"] == 1
    assert selected_contract["environment_contract_version"] == 2
    assert (
        selected_contract["common_input_fingerprint"]
        == default_contract["common_input_fingerprint"]
    )
    assert selected_contract["input_fingerprint"] != default_contract["input_fingerprint"]
    assert selected_contract["python_profiles"] == ["demo", "reference_common"]
    assert selected_contract["tensorrt"] == {
        "version": "11.2.1.2",
        "apt_version": "11.2.1.2-1+cuda13.3",
        "python_distributions": {name: "11.2.1.2" for name in TENSORRT_DISTRIBUTIONS},
        "apt_packages": {name: "11.2.1.2-1+cuda13.3" for name in TENSORRT_APT_PACKAGES},
        "headers": ["NvInferVersion.h", "NvOnnxParser.h"],
        "header_version": "11.2.1.2",
        "native_libraries": [
            "libnvinfer.so",
            "libnvinfer.so.11",
            "libnvonnxparser.so",
            "libnvonnxparser.so.11",
            "libnvinfer_builder_resource_sm110.so.*",
        ],
        "native_library_distribution": "tensorrt_cu13_libs",
        "native_runtime_version": "11.2.1.2",
    }


def test_source_contract_loads_profiles_without_ambient_pythonpath(tmp_path: Path) -> None:
    repo_root, _, _ = _write_profile_fingerprint_repo(tmp_path)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; "
            "from tools.ci.docker_image import DockerImageManager; "
            "print(','.join(DockerImageManager(Path.cwd(), {}).source_contract()"
            "['python_profiles']))",
        ],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "demo,reference_common"


def test_image_contract_cli_emits_canonical_contract_json(tmp_path: Path) -> None:
    repo_root, _, _ = _write_profile_fingerprint_repo(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.ci",
            "image",
            "contract",
            "--tensorrt-version",
            "11.2.1.2",
            "--tensorrt-apt-version",
            "11.2.1.2-1+cuda13.3",
        ],
        cwd=repo_root,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    contract = json.loads(result.stdout)
    assert contract["environment_contract_version"] == 2
    assert contract["tensorrt"]["version"] == "11.2.1.2"
    assert contract["tensorrt"]["apt_version"] == "11.2.1.2-1+cuda13.3"


def test_validate_image_contract_returns_the_verified_source_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root, _, _ = _write_profile_fingerprint_repo(tmp_path)
    manager = DockerImageManager(repo_root)
    expected = manager._read_requirements()
    actual = {
        "TENSORRT_VERSION": expected.tensorrt,
        "TENSORRT_PYTHON_DISTRIBUTIONS": json.dumps(
            expected.python_distributions, separators=(",", ":"), sort_keys=True
        ),
        "TENSORRT_APT_PACKAGES": json.dumps(
            expected.apt_packages, separators=(",", ":"), sort_keys=True
        ),
        "TENSORRT_OVERLAY_FILES": "present",
        "TENSORRT_HEADER_VERSION": expected.tensorrt,
        "TENSORRT_NATIVE_VERSION": expected.tensorrt,
        "MODELOPT_VERSION": expected.modelopt,
        "NLOHMANN_JSON_HEADER": "present",
        "NEMO_PROMPT_RNNT": "available",
        "PYTHON_PROFILES": expected.python_profiles,
    }
    monkeypatch.setattr(manager, "_query_fingerprint", lambda _: expected.fingerprint)
    monkeypatch.setattr(manager, "_query_versions", lambda _: actual)

    assert manager.validate_image_contract("example.invalid/runtime@sha256:test") == (
        expected.contract()
    )


def test_validate_image_contract_fails_closed_on_a_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root, _, _ = _write_profile_fingerprint_repo(tmp_path)
    manager = DockerImageManager(repo_root)
    expected = manager._read_requirements()
    monkeypatch.setattr(manager, "_query_fingerprint", lambda _: expected.fingerprint)
    monkeypatch.setattr(manager, "_query_versions", lambda _: {})

    with pytest.raises(CiError, match="TensorRT version mismatch"):
        manager.validate_image_contract("example.invalid/runtime@sha256:test")


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("tensorrt_version", "11.2", "TENSORRT_VERSION must be an exact four-part version"),
        ("tensorrt_version", "", "TENSORRT_VERSION must be an exact four-part version"),
        (
            "tensorrt_apt_version",
            "11.2.*",
            "TENSORRT_APT_VERSION must be an exact package version",
        ),
        (
            "tensorrt_apt_version",
            "",
            "TENSORRT_APT_VERSION must be an exact package version",
        ),
    ),
)
def test_source_contract_rejects_non_exact_overlay_versions(
    tmp_path: Path,
    keyword: str,
    value: str,
    message: str,
) -> None:
    repo_root, _, _ = _write_profile_fingerprint_repo(tmp_path)
    with pytest.raises(CiError, match=message):
        DockerImageManager(repo_root).source_contract(**{keyword: value})


def test_source_contract_rejects_mixed_tensorrt_overlay_versions(tmp_path: Path) -> None:
    repo_root, _, _ = _write_profile_fingerprint_repo(tmp_path)
    with pytest.raises(CiError, match="must select the same TensorRT version"):
        DockerImageManager(repo_root).source_contract(
            tensorrt_version="11.2.1.2",
            tensorrt_apt_version=TENSORRT_APT_VERSION,
        )


def test_source_contract_rechecks_profile_asset_containment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root, _, _ = _write_profile_fingerprint_repo(tmp_path)
    outside = tmp_path / "outside.lock.txt"
    outside.write_text("demo==1.0\n", encoding="utf-8")
    manager = DockerImageManager(repo_root)
    monkeypatch.setattr(
        manager,
        "_load_profile_registry",
        lambda: (
            {
                "version": 1,
                "profiles": {
                    "demo": {
                        "kind": "venv",
                        "prebuild": True,
                        "requirements": str(outside),
                    }
                },
            },
            ("demo",),
        ),
    )

    with pytest.raises(CiError, match="unsafe requirements path"):
        manager.source_contract()


def test_profile_sources_are_fingerprinted_and_repo_is_the_build_context() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "class DockerImageManager" in script
    assert "semantic_fingerprint" in script
    assert 'b"python-profile-registry\\0"' in script
    assert "assets: set[Path]" in script
    assert 'Path("tools/ci/process.py")' not in script
    assert 'package_root / "python_profiles.py"' in script
    assert '"-f"' in script
    assert "str(self.config.dockerfile)" in script
    assert '"."' in script
    assert "profile builder source leaked into the runtime image" in script
    assert '"--user"' in script
    assert '"65534:65534"' in script
    assert '"--read-only"' in script
    assert '"tensorrt_cu13_bindings"' in script
    assert '"libnvonnxparsers-dev"' in script
    assert "getInferLibBuildVersion" in script
    assert "source_contract_json" in script


def test_profile_fingerprint_ignores_manifest_comments_and_ownership_fields(
    tmp_path: Path,
) -> None:
    repo_root, manifest, _ = _write_profile_fingerprint_repo(tmp_path)
    baseline = _resolved_image_for_repo(tmp_path / "baseline", repo_root)

    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            "# Synthetic family used only by the image-fingerprint tests.",
            "# An unrelated ownership comment changed.",
        ),
        encoding="utf-8",
    )
    comment_changed = _resolved_image_for_repo(tmp_path / "comment-change", repo_root)
    assert comment_changed == baseline

    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            'aliases = ["demo"]',
            'aliases = ["demo", "demo-alias"]',
        ),
        encoding="utf-8",
    )
    ownership_changed = _resolved_image_for_repo(tmp_path / "ownership-change", repo_root)
    assert ownership_changed == baseline


def test_profile_fingerprint_ignores_unrelated_family_loader_changes(
    tmp_path: Path,
) -> None:
    repo_root, _, _ = _write_profile_fingerprint_repo(tmp_path)
    baseline = _resolved_image_for_repo(tmp_path / "baseline", repo_root)

    family_loader = (
        repo_root / "python" / "tensorrt_model_connect" / "families" / "__init__.py"
    )
    family_loader.write_text(
        family_loader.read_text(encoding="utf-8")
        + "\n# Unrelated application-only family parsing change.\n",
        encoding="utf-8",
    )

    changed = _resolved_image_for_repo(tmp_path / "family-loader-change", repo_root)
    assert changed == baseline


def test_source_contract_does_not_execute_or_fingerprint_package_init(
    tmp_path: Path,
) -> None:
    repo_root, _, _ = _write_profile_fingerprint_repo(tmp_path)
    baseline = DockerImageManager(repo_root).source_contract()
    package_init = repo_root / "python" / "tensorrt_model_connect" / "__init__.py"
    package_init.write_text(
        "raise RuntimeError('package metadata must not execute')\n",
        encoding="utf-8",
    )

    changed = DockerImageManager(repo_root).source_contract()

    assert changed["common_input_fingerprint"] == baseline["common_input_fingerprint"]
    assert changed["input_fingerprint"] == baseline["input_fingerprint"]


def test_profile_builder_does_not_execute_package_init(tmp_path: Path) -> None:
    package_root = tmp_path / "tensorrt_model_connect"
    package_root.mkdir()
    (package_root / "__init__.py").write_text(
        "raise RuntimeError('package init must not execute')\n",
        encoding="utf-8",
    )
    (package_root / "python_profiles.py").write_text(
        "def load_python_profile_registry(): return {}\n"
        "def prebuilt_python_profile_names(registry): return ()\n"
        "def profile_root(): raise AssertionError('not reached')\n"
        "def resolve_profile_python(name, base_python): raise AssertionError('not reached')\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["TRTMC_PYTHON_PROFILE_SOURCE"] = str(package_root)
    env["PYTHONPATH"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / ".github/scripts/build-python-profiles.py")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "no prebuilt Python profiles were declared" in result.stderr
    assert "package init must not execute" not in result.stderr


def test_source_contract_does_not_execute_or_fingerprint_family_loader(
    tmp_path: Path,
) -> None:
    repo_root, _, _ = _write_profile_fingerprint_repo(tmp_path)
    baseline = DockerImageManager(repo_root).source_contract()
    family_loader = (
        repo_root / "python" / "tensorrt_model_connect" / "families" / "__init__.py"
    )
    family_loader.write_text(
        "raise RuntimeError('family loader must not execute')\n",
        encoding="utf-8",
    )

    changed = DockerImageManager(repo_root).source_contract()

    assert changed["common_input_fingerprint"] == baseline["common_input_fingerprint"]
    assert changed["input_fingerprint"] == baseline["input_fingerprint"]


def test_profile_fingerprint_ignores_registry_comments(tmp_path: Path) -> None:
    repo_root, _, _ = _write_profile_fingerprint_repo(tmp_path)
    baseline = _resolved_image_for_repo(tmp_path / "baseline", repo_root)

    registry = repo_root / "python" / "tensorrt_model_connect" / "python_profiles.toml"
    registry.write_text(
        "# Review-only comment.\n" + registry.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    changed = _resolved_image_for_repo(tmp_path / "registry-comment", repo_root)
    assert changed == baseline


def test_profile_fingerprint_ignores_lazy_profile_declarations(tmp_path: Path) -> None:
    repo_root, manifest, _ = _write_profile_fingerprint_repo(tmp_path)
    baseline = _resolved_image_for_repo(tmp_path / "baseline", repo_root)

    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            "lazy_demo|families/demo/requirements.lock.txt",
            "renamed_lazy_demo|families/demo/requirements.lock.txt",
        ),
        encoding="utf-8",
    )

    changed = _resolved_image_for_repo(tmp_path / "lazy-profile-change", repo_root)
    assert changed == baseline


def test_profile_fingerprint_changes_for_semantic_profile_declaration(
    tmp_path: Path,
) -> None:
    repo_root, manifest, _ = _write_profile_fingerprint_repo(tmp_path)
    baseline = _resolved_image_for_repo(tmp_path / "baseline", repo_root)

    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            "demo|families/demo/requirements.lock.txt|families/demo/verify.py|true",
            "demo|families/demo/requirements.lock.txt|families/demo/verify.py|false",
            1,
        ),
        encoding="utf-8",
    )

    changed = _resolved_image_for_repo(tmp_path / "profile-change", repo_root)
    assert changed != baseline


def test_profile_fingerprint_changes_for_referenced_profile_asset_content(
    tmp_path: Path,
) -> None:
    repo_root, _, requirements = _write_profile_fingerprint_repo(tmp_path)
    baseline = _resolved_image_for_repo(tmp_path / "baseline", repo_root)

    requirements.write_text("demo-package==1.0.1\n", encoding="utf-8")

    changed = _resolved_image_for_repo(tmp_path / "asset-change", repo_root)
    assert changed != baseline


@pytest.mark.parametrize(
    "relative_path",
    (
        Path("Dockerfile"),
        Path(".github/scripts/build-python-profiles.py"),
        Path("python/tensorrt_model_connect/python_profiles.py"),
        Path("python/tensorrt_model_connect/families/demo/verify.py"),
    ),
)
def test_profile_fingerprint_changes_for_every_baked_recipe_input(
    tmp_path: Path,
    relative_path: Path,
) -> None:
    repo_root, _, _ = _write_profile_fingerprint_repo(tmp_path)
    baseline = _resolved_image_for_repo(tmp_path / "baseline", repo_root)
    target = repo_root / relative_path
    target.write_text(
        target.read_text(encoding="utf-8") + "\n# Environment-affecting recipe change.\n",
        encoding="utf-8",
    )

    changed = _resolved_image_for_repo(tmp_path / "recipe-change", repo_root)

    assert changed != baseline


@pytest.mark.parametrize(
    "relative_path",
    (
        Path("tools/__init__.py"),
        Path("tools/ci/__init__.py"),
        Path("tools/ci/__main__.py"),
        Path("tools/ci/docker_image.py"),
        Path("tools/ci/process.py"),
    ),
)
def test_profile_fingerprint_ignores_contract_producer_code(
    tmp_path: Path,
    relative_path: Path,
) -> None:
    repo_root, _, _ = _write_profile_fingerprint_repo(tmp_path)
    baseline = _resolved_image_for_repo(tmp_path / "baseline", repo_root)
    target = repo_root / relative_path
    target.write_text(
        target.read_text(encoding="utf-8") + "\n# Control-plane-only change.\n",
        encoding="utf-8",
    )

    changed = _resolved_image_for_repo(tmp_path / "producer-change", repo_root)

    assert changed == baseline
