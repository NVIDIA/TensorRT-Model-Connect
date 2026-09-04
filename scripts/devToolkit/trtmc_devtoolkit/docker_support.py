# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared, model-agnostic Docker transport mechanics."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from .models import DevToolkitError


_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


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
