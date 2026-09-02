# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy downstream handoff recipes retained outside the capability core."""

from __future__ import annotations

from pathlib import Path

from ..models import DevToolkitError, HandoffPlan, PrepareResult


def _target_path(result: PrepareResult, path: Path) -> str:
    resolved = path.resolve()
    repository = result.plan.repository.resolve()
    if result.environment.kind == "local":
        return str(resolved)
    try:
        relative = resolved.relative_to(result.plan.state_dir.resolve())
        return str(Path("/trtmc-devtoolkit-run") / relative)
    except ValueError:
        pass
    try:
        relative = resolved.relative_to(repository)
    except ValueError as error:
        raise DevToolkitError(
            f"Docker handoff path must be inside the mounted checkout: {resolved}"
        ) from error
    return str(Path("/workspace/tensorrt-model-connect") / relative)


def _wrap(result: PrepareResult, name: str, command: list[str]) -> HandoffPlan:
    environment = dict(result.environment.environment)
    if result.environment.kind == "docker":
        container = result.environment.container_name
        if not container:
            raise DevToolkitError("Docker environment handle has no container name")
        wrapped = ["docker", "exec"]
        for variable, value in sorted(environment.items()):
            wrapped.extend(["--env", f"{variable}={value}"])
        wrapped.append(container)
        wrapped.extend(command)
        return HandoffPlan(name, tuple(wrapped), {})
    return HandoffPlan(name, tuple(command), environment)


def validation_handoff(
    result: PrepareResult,
    *,
    model: str,
    workload: str,
    bundle: Path,
    output: Path,
) -> HandoffPlan:
    return _wrap(
        result,
        "validation",
        [
            result.environment.python,
            "tools/trtmc_validate.py",
            model,
            workload,
            "--bundle",
            _target_path(result, bundle),
            "--output",
            _target_path(result, output),
        ],
    )


def profiling_handoff(
    result: PrepareResult,
    *,
    model: str,
    bundle: Path,
    output: Path,
) -> HandoffPlan:
    return _wrap(
        result,
        "profiling",
        [
            result.environment.python,
            "tools/trtmc_profile.py",
            "--model",
            model,
            "--bundle",
            _target_path(result, bundle),
            "--trtmc-binary",
            result.environment.trtmc,
            "--output-dir",
            _target_path(result, output),
        ],
    )


def performance_handoff(
    result: PrepareResult,
    *,
    suite: Path,
    environment: Path,
    entry: str,
) -> HandoffPlan:
    return _wrap(
        result,
        "performance",
        [
            result.environment.python,
            "tools/perf_matrix.py",
            "run",
            _target_path(result, suite),
            "--environment",
            _target_path(result, environment),
            "--entry",
            entry,
        ],
    )
