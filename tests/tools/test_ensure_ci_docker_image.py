# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for CI Docker image resolution and validation."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / ".github" / "scripts" / "ensure-ci-docker-image.sh"


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
  profiles="${FAKE_DOCKER_PROFILES-chronos,deepseek_ocr,elf_flow,internlm,magpie_tts_reference,nemotron_h_reference,phi4_multimodal}"
  if [ -f "$FAKE_DOCKER_REBUILT" ]; then
    capability="available"
    profiles="chronos,deepseek_ocr,elf_flow,internlm,magpie_tts_reference,nemotron_h_reference,phi4_multimodal"
  fi
  cat <<'EOF'
TENSORRT_VERSION=11.0.0.114
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
    profiles: str = (
        "chronos,deepseek_ocr,elf_flow,internlm,magpie_tts_reference,"
        "nemotron_h_reference,phi4_multimodal"
    ),
    changed_paths: tuple[str, ...] = (),
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
            "FAKE_DOCKER_REBUILT": str(tmp_path / "rebuilt"),
            "GITHUB_ENV": str(github_env),
            "RUNNER_TEMP": str(tmp_path / "runner-temp"),
            "TRTMC_CI_IMAGE": "trtmc-dev-gb300:manylinux_2_39",
            "CI_BASE_REF": "fake-base" if changed_paths else "",
            "FAKE_GIT_CHANGED_PATHS": "\n".join(changed_paths),
        }
    )
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    github_env_text = github_env.read_text() if github_env.exists() else ""
    return result, github_env_text, log_path.read_text()


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
    resolved_image = re.search(
        r"^TRTMC_CI_IMAGE=(.+)$", bootstrap_env, re.MULTILINE
    ).group(1)
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
    resolved_image = re.search(
        r"^TRTMC_CI_IMAGE=(.+)$", bootstrap_env, re.MULTILINE
    ).group(1)
    fingerprint = re.search(
        r"--label org\.nvidia\.trtmc\.ci-input-fingerprint=([0-9a-f]{64})",
        bootstrap_log,
    ).group(1)

    result, _, docker_log = _run_ensure_script(
        tmp_path / "matching",
        existing_images={resolved_image: fingerprint},
        changed_paths=(".github/scripts/ensure-ci-docker-image.sh",),
    )

    assert result.returncode == 0, result.stderr
    assert "build " not in docker_log
    assert f"CI Docker image '{resolved_image}' already matches" in result.stdout


def test_missing_prebuilt_profiles_rebuilds_the_image(tmp_path: Path) -> None:
    bootstrap_result, bootstrap_env, bootstrap_log = _run_ensure_script(
        tmp_path / "bootstrap",
        existing_images={},
    )
    assert bootstrap_result.returncode == 0, bootstrap_result.stderr
    resolved_image = re.search(
        r"^TRTMC_CI_IMAGE=(.+)$", bootstrap_env, re.MULTILINE
    ).group(1)
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


def test_profile_sources_are_fingerprinted_and_repo_is_the_build_context() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "family_model_manifests" in script
    assert "declared_profile_assets" in script
    assert 'python/tensorrt_model_connect/python_profiles.py' in script
    assert '-f "$dockerfile" \\\n    .' in script
    assert "trtmc-empty-docker-context" not in script
    assert "profile builder source leaked into the runtime image" in script
