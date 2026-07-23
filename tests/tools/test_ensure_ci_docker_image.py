# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for CI Docker image resolution and validation."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from tools.ci.docker_image import WorkflowImageLock


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "tools" / "ci" / "docker_image.py"
DEFAULT_PROFILES = (
    "chronos,deepseek_ocr,elf_flow,internlm,lance_reference,magpie_tts_reference,"
    "nemotron_h_reference,personaplex_reference,phi4_multimodal"
)


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
  profiles="${FAKE_DOCKER_PROFILES-chronos,deepseek_ocr,elf_flow,internlm,lance_reference,magpie_tts_reference,nemotron_h_reference,personaplex_reference,phi4_multimodal}"
  if [ -f "$FAKE_DOCKER_REBUILT" ]; then
    capability="available"
    profiles="$FAKE_DOCKER_REBUILT_PROFILES"
  fi
  cat <<'EOF'
TENSORRT_VERSION=11.2.0.113
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
        REPO_ROOT / "python" / "tensorrt_model_connect" / "python_profiles.toml",
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
python_profile_specs = [
  "demo|families/demo/requirements.lock.txt|families/demo/verify.py|true",
]
default_execution_profiles = ["reference|demo"]
""",
        encoding="utf-8",
    )
    requirements = demo_root / "requirements.lock.txt"
    requirements.write_text("demo-package==1.0.0\n", encoding="utf-8")
    (demo_root / "verify.py").write_text("import demo_package\n", encoding="utf-8")

    (repo_root / "Dockerfile").write_text(
        "ARG TENSORRT_VERSION=11.2.0.113\nARG MODELOPT_VERSION=0.44.0\n",
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
        profiles="demo",
        rebuilt_profiles="demo",
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


def test_profile_sources_are_fingerprinted_and_repo_is_the_build_context() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "class DockerImageManager" in script
    assert "semantic_fingerprint" in script
    assert 'b"python-profile-registry\\0"' in script
    assert "assets: set[Path]" in script
    assert 'package_root / "python_profiles.py"' in script
    assert '"-f"' in script
    assert "str(self.config.dockerfile)" in script
    assert '"."' in script
    assert "profile builder source leaked into the runtime image" in script
    assert '"--user"' in script
    assert '"65534:65534"' in script
    assert '"--read-only"' in script


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


def test_profile_fingerprint_changes_for_semantic_profile_declaration(
    tmp_path: Path,
) -> None:
    repo_root, manifest, _ = _write_profile_fingerprint_repo(tmp_path)
    baseline = _resolved_image_for_repo(tmp_path / "baseline", repo_root)

    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            "families/demo/verify.py|true",
            "families/demo/verify.py|false",
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
