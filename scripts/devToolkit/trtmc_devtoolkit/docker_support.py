# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared, model-agnostic Docker transport mechanics."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from .models import DevToolkitError
from .runner import Runner, command_output


_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def docker_command(context: str, *arguments: str) -> list[str]:
    return ["docker", "--context", context, *arguments]


def docker_daemon_id(runner: Runner, repository: Path, context: str) -> str:
    value = command_output(
        runner,
        docker_command(context, "info", "--format", "{{.ID}}"),
        cwd=repository,
        timeout=30,
    )
    if not value:
        raise DevToolkitError(f"Docker context {context!r} has no daemon identity")
    return value


def require_docker_client_version(
    runner: Runner,
    repository: Path,
    context: str,
) -> None:
    output = command_output(
        runner,
        docker_command(context, "version", "--format", "{{.Client.Version}}"),
        cwd=repository,
        timeout=30,
    )
    match = re.match(r"([0-9]+)\.([0-9]+)", output)
    if match is None or tuple(int(value) for value in match.groups()) < (20, 10):
        raise DevToolkitError(f"Docker CLI 20.10 or newer is required; found {output}")


def _inspect_docker_object(
    runner: Runner,
    repository: Path,
    command: Sequence[str],
    description: str,
) -> dict[str, object] | None:
    result = runner.run(
        command,
        cwd=repository,
        check=False,
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if any(marker in detail.lower() for marker in ("no such", "not found")):
            return None
        raise DevToolkitError(
            f"Could not inspect {description}: {detail or f'Docker exited {result.returncode}'}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise DevToolkitError(f"Could not inspect {description}: {error}") from error
    if not isinstance(payload, dict):
        raise DevToolkitError(f"Docker inspect returned invalid data for {description}")
    return payload


def inspect_docker_container(
    runner: Runner,
    repository: Path,
    context: str,
    container: str,
) -> dict[str, object] | None:
    return _inspect_docker_object(
        runner,
        repository,
        docker_command(
            context,
            "inspect",
            "--type",
            "container",
            "--format",
            "{{json .}}",
            container,
        ),
        f"Docker container {container}",
    )


def inspect_docker_image(
    runner: Runner,
    repository: Path,
    context: str,
    reference: str,
) -> dict[str, object] | None:
    return _inspect_docker_object(
        runner,
        repository,
        docker_command(context, "image", "inspect", "--format", "{{json .}}", reference),
        f"Docker image {reference}",
    )


@contextmanager
def docker_environment_file(
    state_dir: Path,
    environment: Mapping[str, str],
) -> Iterator[Path | None]:
    """Expose values to Docker through a short-lived mode-0600 env file."""
    if not environment:
        yield None
        return
    for name, value in environment.items():
        if _ENVIRONMENT_NAME.fullmatch(name) is None:
            raise DevToolkitError(f"Invalid Docker environment name: {name!r}")
        if not isinstance(value, str) or any(character in value for character in "\r\n\0"):
            raise DevToolkitError(
                f"Docker environment value for {name!r} must be a single text line"
            )
    secret_dir = state_dir / ".secrets"
    secret_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(secret_dir, 0o700)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=secret_dir,
            prefix="docker-environment-",
            suffix=".list",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            for name, value in sorted(environment.items()):
                stream.write(f"{name}={value}\n")
            stream.flush()
            os.fsync(stream.fileno())
        yield temporary
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
