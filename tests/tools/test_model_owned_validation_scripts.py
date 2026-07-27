# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for model-owned validation entrypoints.

Trace: ARCH-MODPLUG-001
Intent: keep developer validation scripts aligned with model-local E2E tests.
Preconditions: validation scripts are present in the repository.
Postconditions: family validation runs model-owned pytest nodes and exposes
isolated model-plugin validation.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _render_autopilot_prompt(filename: str) -> str:
    module_path = REPO_ROOT / "scripts" / "autopilot" / filename
    spec = importlib.util.spec_from_file_location(
        f"test_autopilot_{module_path.stem}",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_prompt(
        {
            "model_type": "unit",
            "hf_id": "org/unit-model",
            "family_name": "unit_family",
        },
        "agent-9",
    )


def test_validate_family_uses_model_owned_e2e_entrypoint() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prevent validate_family.sh from scheduling the shared E2E test node.
    Preconditions: scripts/validate_family.sh exists.
    Postconditions: the script builds tests/e2e/models/<family> node ids.
    """
    text = (REPO_ROOT / "scripts" / "validate_family.sh").read_text(encoding="utf-8")

    assert "tests/test_e2e.py::test_e2e" not in text
    assert "tests/e2e/models/${E2E_FAMILY}/test_${E2E_FAMILY}_e2e.py" in text
    assert "--model-plugin-dir" in text
    assert "--isolate-model-plugin" in text
    assert 'export TRTMC_MODEL_PLUGIN_DIR="$MODEL_PLUGIN_DIR"' in text
    assert "export TRTMC_MODEL_PLUGIN_STRICT=1" in text


def test_validate_family_forwards_trust_remote_code_to_bundle_build() -> None:
    """Remote-code models must receive the flag before any downstream checks."""
    text = (REPO_ROOT / "scripts" / "validate_family.sh").read_text(encoding="utf-8")

    assert "TRUST_REMOTE_CODE_ARGS=()" in text
    assert "TRUST_REMOTE_CODE_ARGS+=(--trust-remote-code)" in text
    assert "BUILD_ARGS=(" in text
    assert '"${TRUST_REMOTE_CODE_ARGS[@]}"' in text
    assert 'run_step "Build bundle" build_candidate_bundle' in text
    assert '-o "$CANDIDATE_BUNDLE_PATH"' in text


def test_validate_family_build_invocation_includes_trust_remote_code(
    tmp_path: Path,
) -> None:
    """Exercise argument forwarding without requiring a real model build."""
    argument_log = tmp_path / "build-arguments.txt"
    fake_binary = tmp_path / "fake-trtmc"
    fake_binary.write_text(
        """#!/usr/bin/env bash
if [[ "$1" == "build" ]]; then
    : > "$ARGUMENT_LOG"
    printf '%s\\n' "$@" >> "$ARGUMENT_LOG"
fi
exit 1
""",
        encoding="utf-8",
    )
    fake_binary.chmod(0o755)
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/usr/bin/env bash
if [[ "$1" == "-" ]]; then
    cat >/dev/null
    exit 1
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "validate_family.sh"),
            "org/definitely-not-a-model",
            "--binary",
            str(fake_binary),
            "--bundle-dir",
            str(tmp_path),
            "--trust-remote-code",
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "ARGUMENT_LOG": str(argument_log),
            "HF_PYTHON": str(fake_python),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 1
    arguments = argument_log.read_text(encoding="utf-8").splitlines()
    assert arguments[:3] == [
        "build",
        "org/definitely-not-a-model",
        "-o",
    ]
    candidate = Path(arguments[3])
    assert candidate.name == "org_definitely-not-a-model.trtfb"
    assert candidate.parent.parent == tmp_path
    assert candidate.parent.name.startswith(".validate-family-build.")
    assert candidate != tmp_path / "org_definitely-not-a-model.trtfb"
    assert arguments[4:] == [
        "--max-cache-length",
        "256",
        "--trust-remote-code",
    ]


def test_validate_family_fails_when_non_decoder_has_no_e2e_manifest(
    tmp_path: Path,
) -> None:
    """A successful build is insufficient when no model-owned E2E gate exists."""
    project_dir = tmp_path / "repo"
    scripts_dir = project_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (project_dir / "tests" / "e2e" / "models").mkdir(parents=True)
    script = scripts_dir / "validate_family.sh"
    script.write_text(
        (REPO_ROOT / "scripts" / "validate_family.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    fake_binary = tmp_path / "fake-trtmc"
    fake_binary.write_text(
        """#!/usr/bin/env bash
if [[ "$1" == "build" ]]; then
    output=""
    previous=""
    for argument in "$@"; do
        if [[ "$previous" == "-o" ]]; then
            output="$argument"
        fi
        previous="$argument"
    done
    mkdir -p "$(dirname "$output")"
    printf '%s' "built" > "$output"
    exit 0
fi
if [[ "$1" == "inspect" ]]; then
    echo "Runtime strategy: unit_non_decoder"
    exit 0
fi
exit 1
""",
        encoding="utf-8",
    )
    fake_binary.chmod(0o755)
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/usr/bin/env bash
if [[ "$1" == "-" ]]; then
    cat >/dev/null
    exit 1
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            str(script),
            "org/no-e2e-manifest",
            "--binary",
            str(fake_binary),
            "--bundle-dir",
            str(tmp_path),
        ],
        cwd=project_dir,
        env={**os.environ, "HF_PYTHON": str(fake_python)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 1
    assert "PASS  Build bundle" in result.stdout
    assert "SKIP  diff_logits (non-decoder: unit_non_decoder)" in result.stdout
    assert "FAIL  E2E pytest (no manifest -- create one)" in result.stdout
    assert "1 passed, 1 failed" in result.stdout


def test_validate_family_e2e_consumes_only_the_current_bundle(
    tmp_path: Path,
) -> None:
    """Exact or explicit manifest selection must never rebuild the canonical ID."""
    project_dir = tmp_path / "repo"
    scripts_dir = project_dir / "scripts"
    manifest_dir = project_dir / "tests" / "e2e" / "models" / "unit" / "manifests"
    scripts_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    script = scripts_dir / "validate_family.sh"
    script.write_text(
        (REPO_ROOT / "scripts" / "validate_family.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (manifest_dir.parent / "MODEL.toml").write_text(
        'id = "unit"\ntest_manifests = ["manifests/unit-model.json"]\n',
        encoding="utf-8",
    )
    (manifest_dir / "unit-model.json").write_text(
        json.dumps(
            {
                "name": "unit-model",
                "hf_id": "org/unit-model",
                "bundle": "unit-model.trtfb",
            }
        ),
        encoding="utf-8",
    )

    fake_binary = tmp_path / "fake-trtmc"
    fake_binary.write_text(
        """#!/usr/bin/env bash
if [[ "$1" == "build" ]]; then
    if [[ "${NOOP_BUILD:-}" == "1" ]]; then
        exit 0
    fi
    output=""
    previous=""
    for argument in "$@"; do
        if [[ "$previous" == "-o" ]]; then
            output="$argument"
        fi
        previous="$argument"
    done
    mkdir -p "$(dirname "$output")"
    if [[ "${CANDIDATE_KIND:-}" == "symlink" ]]; then
        ln -s "$CANDIDATE_TARGET" "$output"
    elif [[ "${CANDIDATE_KIND:-}" == "directory" ]]; then
        mkdir -p "$output"
        printf '%s' "$2" > "$output/nested-bundle"
    else
        printf '%s' "$2" > "$output"
    fi
    exit 0
fi
if [[ "$1" == "inspect" ]]; then
    echo "Runtime strategy: unit_non_decoder"
    exit 0
fi
exit 1
""",
        encoding="utf-8",
    )
    fake_binary.chmod(0o755)

    pytest_argument_log = tmp_path / "pytest-arguments.txt"
    proven_bundle_log = tmp_path / "proven-bundle.txt"
    python_wrapper = tmp_path / "python-wrapper"
    python_wrapper.write_text(
        """#!/usr/bin/env bash
if [[ "$1" == "-m" && "$2" == "pytest" ]]; then
    printf '%s\\n' "$@" > "$PYTEST_ARGUMENT_LOG"
    engine_dir=""
    previous=""
    for argument in "$@"; do
        if [[ "$previous" == "--engine-dir" ]]; then
            engine_dir="$argument"
        fi
        if [[ "$argument" == "--rebuild-engines" ]]; then
            exit 91
        fi
        previous="$argument"
    done
    [[ -L "$engine_dir/unit-model.trtfb" ]] || exit 92
    cat "$engine_dir/unit-model.trtfb" > "$PROVEN_BUNDLE_LOG"
    exit 0
fi
exec "$REAL_PYTHON" "$@"
""",
        encoding="utf-8",
    )
    python_wrapper.chmod(0o755)
    bundle_dir = tmp_path / "bundles"
    engine_dir = tmp_path / "engines"
    common_command = [
        "bash",
        str(script),
        "--binary",
        str(fake_binary),
        "--bundle-dir",
        str(bundle_dir),
        "--engine-dir",
        str(engine_dir),
    ]
    environment = {
        **os.environ,
        "HF_PYTHON": str(python_wrapper),
        "REAL_PYTHON": sys.executable,
        "PYTEST_ARGUMENT_LOG": str(pytest_argument_log),
        "PROVEN_BUNDLE_LOG": str(proven_bundle_log),
    }

    bundle_dir.mkdir(parents=True)
    final_bundle = bundle_dir / "org_unit-model.trtfb"
    final_bundle.write_text("STALE-BUNDLE", encoding="utf-8")
    stale = subprocess.run(
        [*common_command[:2], "org/unit-model", *common_command[2:]],
        cwd=project_dir,
        env={**environment, "NOOP_BUILD": "1"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert stale.returncode == 1
    assert "FAIL  Build bundle" in stale.stdout
    assert final_bundle.read_text(encoding="utf-8") == "STALE-BUNDLE"
    assert not pytest_argument_log.exists()
    assert not list(bundle_dir.glob(".validate-family-build.*"))

    candidate_target = tmp_path / "candidate-target.trtfb"
    candidate_target.write_text("NONEMPTY-SYMLINK-TARGET", encoding="utf-8")
    symlink_candidate = subprocess.run(
        [*common_command[:2], "org/unit-model", *common_command[2:]],
        cwd=project_dir,
        env={
            **environment,
            "CANDIDATE_KIND": "symlink",
            "CANDIDATE_TARGET": str(candidate_target),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert symlink_candidate.returncode == 1
    assert "non-empty regular, non-symlink bundle" in symlink_candidate.stderr
    assert final_bundle.read_text(encoding="utf-8") == "STALE-BUNDLE"
    assert not pytest_argument_log.exists()
    assert not list(bundle_dir.glob(".validate-family-build.*"))

    directory_candidate = subprocess.run(
        [*common_command[:2], "org/unit-model", *common_command[2:]],
        cwd=project_dir,
        env={**environment, "CANDIDATE_KIND": "directory"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert directory_candidate.returncode == 1
    assert "non-empty regular, non-symlink bundle" in directory_candidate.stderr
    assert final_bundle.read_text(encoding="utf-8") == "STALE-BUNDLE"
    assert not pytest_argument_log.exists()
    assert not list(bundle_dir.glob(".validate-family-build.*"))

    final_bundle.unlink()
    final_bundle.mkdir()
    directory_destination = subprocess.run(
        [*common_command[:2], "org/unit-model", *common_command[2:]],
        cwd=project_dir,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert directory_destination.returncode == 1
    assert "bundle destination exists but is not a regular file" in (
        directory_destination.stderr
    )
    assert final_bundle.is_dir()
    assert not any(final_bundle.iterdir())
    assert not pytest_argument_log.exists()
    assert not list(bundle_dir.glob(".validate-family-build.*"))

    final_bundle.rmdir()
    directory_target = tmp_path / "bundle-directory-target"
    directory_target.mkdir()
    final_bundle.symlink_to(directory_target, target_is_directory=True)
    symlink_destination = subprocess.run(
        [*common_command[:2], "org/unit-model", *common_command[2:]],
        cwd=project_dir,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert symlink_destination.returncode == 1
    assert "bundle destination exists but is not a regular file" in (
        symlink_destination.stderr
    )
    assert final_bundle.is_symlink()
    assert not any(directory_target.iterdir())
    assert not pytest_argument_log.exists()
    assert not list(bundle_dir.glob(".validate-family-build.*"))

    final_bundle.unlink()
    directory_target.rmdir()
    exact = subprocess.run(
        [*common_command[:2], "org/unit-model", *common_command[2:]],
        cwd=project_dir,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert exact.returncode == 0, exact.stdout + exact.stderr
    assert proven_bundle_log.read_text(encoding="utf-8") == "org/unit-model"
    assert final_bundle.read_text(encoding="utf-8") == "org/unit-model"
    assert "--rebuild-engines" not in pytest_argument_log.read_text(encoding="utf-8")
    assert not list(bundle_dir.glob(".validate-family-build.*"))

    pytest_argument_log.unlink()
    local_model = tmp_path / "unit-model-local-checkpoint"
    local_model.mkdir()
    implicit_local = subprocess.run(
        [*common_command[:2], str(local_model), *common_command[2:]],
        cwd=project_dir,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert implicit_local.returncode == 1
    assert "local checkpoints require --e2e-model MANIFEST_NAME" in implicit_local.stderr
    assert not pytest_argument_log.exists()

    explicit_local = subprocess.run(
        [
            *common_command[:2],
            str(local_model),
            *common_command[2:],
            "--e2e-model",
            "unit-model",
        ],
        cwd=project_dir,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert explicit_local.returncode == 0, explicit_local.stdout + explicit_local.stderr
    assert proven_bundle_log.read_text(encoding="utf-8") == str(local_model)
    assert "--rebuild-engines" not in pytest_argument_log.read_text(encoding="utf-8")


def test_autopilot_prompt_uses_model_owned_e2e_entrypoint() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep generated autopilot instructions on model-local E2E tests.
    Preconditions: scripts/autopilot/autorun.py exists.
    Postconditions: the final E2E command points at tests/e2e/models/<family>.
    """
    text = (REPO_ROOT / "scripts" / "autopilot" / "autorun.py").read_text(encoding="utf-8")

    assert "tests/test_e2e.py::test_e2e[{family_name}]" not in text
    assert "tests/e2e/models/{family_name}/test_{family_name}_e2e.py" in text
    assert "src/runtime/plugins/" not in text
    assert "REGISTER_PIPELINE_PLUGIN_WITH_FORCE_LINK" not in text
    assert "RUNTIME_TO_TASK_STRATEGY" not in text
    assert "tools/check_runtime_strategy_matrix.py" in text
    assert "src/runtime/models/{family_name}/MODEL.toml" in text


def test_dispatch_prompt_uses_complete_model_owned_capsule() -> None:
    text = (REPO_ROOT / "scripts" / "autopilot" / "dispatch.py").read_text(encoding="utf-8")

    assert '"runtime_strategy": "decoder_kv_cache"' not in text
    assert "python/tensorrt_model_connect/families/{family_name}.py" not in text
    assert "python/tensorrt_model_connect/families/{family_name}/MODEL.toml" in text
    assert "src/runtime/models/{family_name}/MODEL.toml" in text
    assert "tests/runtime_strategy_matrix.yaml" in text
    assert '"reference_family": "causal_base_continuation"' in text
    assert '"user_contract": "continuation_parity"' in text
    assert '"reference_family": "causal_lm"' not in text
    assert '"user_contract": "text_generation"' not in text


def test_autopilot_final_gates_follow_manifest_and_precede_submission() -> None:
    """Final evidence must cover the complete capsule before commit or PASS."""
    for filename in ("dispatch.py", "autorun.py"):
        prompt = _render_autopilot_prompt(filename)
        manifest = "tests/e2e/models/unit_family/manifests/unit_family.json"
        required_manifest = f"test -f \\\n    {manifest}"
        prepare_engine_dir = "mkdir -p \\\n    /tmp/trtmc-engines/unit_family"
        validate_family = "./scripts/validate_family.sh org/unit-model"
        model_ci = "python3 tools/model_ci.py validate"
        test_impact = "python3 tools/test_impact.py --validate"
        runtime_matrix = "python3 tools/check_runtime_strategy_matrix.py"
        final_e2e = "tests/e2e/models/unit_family/test_unit_family_e2e.py -v"
        submit = "git fetch github main"

        assert prompt.index(manifest) < prompt.index(required_manifest)
        assert prompt.index(required_manifest) < prompt.index(prepare_engine_dir)
        assert prompt.index(prepare_engine_dir) < prompt.index(validate_family)
        assert prompt.index(validate_family) < prompt.index(model_ci)
        assert prompt.index(model_ci) < prompt.index(test_impact)
        assert prompt.index(test_impact) < prompt.index(runtime_matrix)
        assert prompt.index(runtime_matrix) < prompt.index(final_e2e)
        assert prompt.index(final_e2e) < prompt.index(submit)

        validate_block = prompt[prompt.index(validate_family) : prompt.index(model_ci)]
        assert "--bundle-dir /tmp/trtmc-engines/unit_family" in validate_block
        assert "--engine-dir /tmp/trtmc-engines/unit_family" in validate_block
        assert "--isolate-model-plugin" in validate_block

        for descriptor in (
            "python/tensorrt_model_connect/families/unit_family/MODEL.toml",
            "src/runtime/models/unit_family/MODEL.toml",
            "tests/e2e/models/unit_family/MODEL.toml",
        ):
            assert f"test -f \\\n    {descriptor}" in prompt

        if filename == "dispatch.py":
            assert prompt.index(runtime_matrix) < prompt.index('"status": "PASS"')
