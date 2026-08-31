# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contract tests for the repository-local TRTMC devToolkit."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from tools.ci.package import WheelArchiveValidator, WheelPackageManager


REPO_ROOT = Path(__file__).resolve().parents[2]
DEVTOOLKIT_ROOT = REPO_ROOT / "scripts" / "devToolkit"
sys.path.insert(0, str(DEVTOOLKIT_ROOT))

from trtmc_devtoolkit import (  # noqa: E402
    DevToolkit,
    DockerTarget,
    PrepareRequest,
    validation_handoff,
)
from trtmc_devtoolkit.cohorts import CohortRegistry, normalize_architecture  # noqa: E402
from trtmc_devtoolkit.models import DevToolkitError  # noqa: E402
from trtmc_devtoolkit.planner import image_fingerprint  # noqa: E402


class RecordingRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def run(
        self,
        command,
        *,
        cwd: Path,
        env=None,
        check: bool = True,
        capture_output: bool = False,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env, capture_output, timeout
        arguments = [str(item) for item in command]
        self.commands.append(arguments)
        output = ""
        returncode = 0
        if arguments[:3] in (
            ["docker", "image", "inspect"],
            ["docker", "container", "inspect"],
        ) and "--format" not in arguments:
            returncode = 1
        elif arguments[0] == "nvidia-smi":
            output = "NVIDIA GB300, GPU-uuid, 595.58.03, 10.0, 191000\n"
        elif arguments[:2] == ["docker", "version"]:
            output = "28.0.0\n"
        elif arguments[:2] == ["docker", "exec"]:
            if "import ctypes, tensorrt" in " ".join(arguments):
                output = "11.1.0.106 11.1.0.106\n"
            elif "--query-gpu=compute_cap" in arguments:
                output = "10.0\n"
            elif arguments[-3:-1] == ["sh", "-c"]:
                output = "100"
        result = subprocess.CompletedProcess(arguments, returncode, output, "")
        if check and returncode:
            raise DevToolkitError(f"fake command failed: {arguments}")
        return result


def _minimal_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repo"
    cohort_dir = repository / "configs" / "environment-cohorts"
    cohort_dir.mkdir(parents=True)
    shutil.copy(
        REPO_ROOT / "configs" / "environment-cohorts" / "trt111-cu133.json",
        cohort_dir / "trt111-cu133.json",
    )
    shutil.copy(REPO_ROOT / "Dockerfile.dev.aarch64", repository / "Dockerfile.dev.aarch64")
    requirements = repository / "requirements"
    requirements.mkdir()
    (requirements / "community-ci.txt").write_text("pytest==8.4.2\n", encoding="utf-8")
    return repository


def test_resolves_exact_supported_cohort() -> None:
    registry = CohortRegistry(REPO_ROOT / "configs" / "environment-cohorts")

    cohort = registry.resolve(
        tensorrt="11.1.0.106",
        cuda="13.3",
        architecture="aarch64",
        python_version="3.12",
        allow_experimental=False,
    )

    assert cohort.id == "trt111-cu133"
    assert cohort.architectures["aarch64"].docker_context == "requirements"


def test_checked_in_cohorts_match_schema_and_package_default() -> None:
    root = REPO_ROOT / "configs" / "environment-cohorts"
    schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
    cohorts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("*.json"))
        if path.name != "schema.json"
    ]
    for cohort in cohorts:
        jsonschema.validate(cohort, schema)
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        package = tomllib.load(stream)["tool"]["tensorrt-model-connect"]["package"]

    supported = [cohort for cohort in cohorts if cohort["status"] == "supported"]
    assert len(supported) == 1
    assert supported[0]["tensorrt"]["version"] == package["default-tensorrt-version"]
    for architecture in ("x86_64", "aarch64"):
        dockerfile = REPO_ROOT / supported[0]["architectures"][architecture]["dockerfile"]
        assert f"tensorrt.__version__ == '{package['default-tensorrt-version']}'" in (
            dockerfile.read_text(encoding="utf-8")
        )


def test_rejects_nearest_or_partial_version_match() -> None:
    registry = CohortRegistry(REPO_ROOT / "configs" / "environment-cohorts")

    with pytest.raises(DevToolkitError, match="No exact environment cohort"):
        registry.resolve(
            tensorrt="11.1",
            cuda="13.3",
            architecture="aarch64",
            python_version="3.12",
            allow_experimental=False,
        )


@pytest.mark.parametrize(
    ("raw", "expected"),
    (("amd64", "x86_64"), ("arm64", "aarch64"), ("aarch64", "aarch64")),
)
def test_normalizes_architecture_aliases(raw: str, expected: str) -> None:
    assert normalize_architecture(raw) == expected


def test_image_fingerprint_tracks_dockerfile_and_context(tmp_path: Path) -> None:
    repository = _minimal_repository(tmp_path)
    cohort = CohortRegistry(repository / "configs" / "environment-cohorts").load_all()[0]
    contract = cohort.architectures["aarch64"]

    initial = image_fingerprint(repository, cohort.source, contract)
    (repository / "requirements" / "community-ci.txt").write_text(
        "pytest==8.4.3\n", encoding="utf-8"
    )

    assert image_fingerprint(repository, cohort.source, contract) != initial


def test_plan_is_read_only_and_apply_prepares_owned_container(
    tmp_path: Path,
) -> None:
    repository = _minimal_repository(tmp_path)
    runner = RecordingRunner()
    state_root = tmp_path / "runs"
    toolkit = DevToolkit.from_checkout(
        repository,
        state_root=state_root,
        source_revision_override="a" * 40,
        runner=runner,
    )
    request = PrepareRequest(
        tensorrt="11.1.0.106",
        cuda="13.3",
        architecture="aarch64",
        target=DockerTarget(gpu="0", container_name="trtmc-dev-gb300-test"),
    )

    plan = toolkit.plan(request)

    assert not plan.state_dir.exists()
    assert plan.state_dir.parent == state_root
    assert [step.id for step in plan.steps] == [
        "doctor",
        "provision",
        "build-install",
        "verify-install",
        "receipt",
    ]

    result = toolkit.apply(plan)

    assert result.receipt.is_file()
    receipt = json.loads(result.receipt.read_text(encoding="utf-8"))
    assert receipt["status"] == "ready"
    assert receipt["environment"]["container_name"] == "trtmc-dev-gb300-test"
    docker_build = next(command for command in runner.commands if command[:2] == ["docker", "build"])
    assert docker_build[-1] == str(repository / "requirements")
    docker_run = next(command for command in runner.commands if command[:2] == ["docker", "run"])
    assert f"org.nvidia.trtmc.devtoolkit-run={plan.run_id}" in docker_run
    assert ["--gpus", "device=0"] == docker_run[
        docker_run.index("--gpus") : docker_run.index("--gpus") + 2
    ]

    handoff = validation_handoff(
        result,
        model="qwen3-0.6b",
        workload="qwen.generate",
        bundle=plan.state_dir / "qwen.bundle",
        output=plan.state_dir / "validation",
    )
    assert handoff.command[:3] == ("docker", "exec", "--env")
    assert "tools/trtmc_validate.py" in handoff.command
    assert "/trtmc-devtoolkit-run/qwen.bundle" in handoff.command


def test_rejects_invalid_source_revision_override(tmp_path: Path) -> None:
    repository = _minimal_repository(tmp_path)

    with pytest.raises(DevToolkitError, match="source_revision_override"):
        DevToolkit.from_checkout(repository, source_revision_override="working-tree")


def test_x86_64_wheel_platform_is_accepted_and_selected(tmp_path: Path) -> None:
    wheel = tmp_path / "dist" / "tensorrt_model_connect-0.1.0-py312-none-manylinux_2_39_x86_64.whl"
    wheel.parent.mkdir()
    wheel.touch()
    context = SimpleNamespace(
        repository=tmp_path,
        env={"TRTMC_PACKAGE_WHEEL_ARCH": "manylinux_2_39_x86_64"},
    )

    validator = WheelArchiveValidator(context, "manylinux_2_39_x86_64")

    assert validator.architecture == "x86_64"
    assert WheelPackageManager(context).select_wheel("py312") == wheel
