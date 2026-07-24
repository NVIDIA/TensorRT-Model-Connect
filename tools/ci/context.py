# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provide shared repository state and external-command access to CI classes.

Boundary: filesystem and process mechanics only; this module contains no stage policy.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from .process import CiError, CommandRunner, ObservedProcessResult


class CiContext:
    """Own the repository, environment, subprocess runner, and CI state files."""

    def __init__(self, repository: Path | None = None, env: Mapping[str, str] | None = None):
        self.repository = (repository or Path.cwd()).resolve()
        self.env = dict(env or os.environ)
        self.commands = CommandRunner(cwd=self.repository, env=self.env)
        self.state_dir = self.repository / self.env.get("TRTMC_CI_STATE_DIR", ".ci")

    def prepare_shared_directories(self) -> None:
        for name in (
            "ENGINE_DIR",
            "HF_HOME",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "HF_MODULES_CACHE",
        ):
            value = self.env.get(name, "")
            if value:
                Path(value).mkdir(parents=True, exist_ok=True)

    def run(
        self,
        command: Sequence[str | Path],
        *,
        limit: str | None = None,
        updates: Mapping[str, str] | None = None,
        unset: Sequence[str] = (),
        check: bool = True,
        capture_output: bool = False,
        cwd: Path | None = None,
    ):
        environment = dict(self.env)
        environment.update(updates or {})
        for name in unset:
            environment.pop(name, None)
        arguments = [str(item) for item in command]
        if limit:
            arguments = ["timeout", "--kill-after=2m", limit, *arguments]
        return self.commands.run(
            arguments,
            check=check,
            capture_output=capture_output,
            env=environment,
            cwd=cwd,
        )

    def output(
        self,
        command: Sequence[str | Path],
        *,
        updates: Mapping[str, str] | None = None,
        unset: Sequence[str] = (),
        check: bool = True,
        cwd: Path | None = None,
    ) -> str:
        return self.run(
            command,
            updates=updates,
            unset=unset,
            check=check,
            capture_output=True,
            cwd=cwd,
        ).stdout.strip()

    def run_observed(
        self,
        command: Sequence[str | Path],
        *,
        limit: str | None = None,
        updates: Mapping[str, str] | None = None,
        unset: Sequence[str] = (),
        check: bool = True,
        cwd: Path | None = None,
    ) -> ObservedProcessResult:
        environment = dict(self.env)
        environment.update(updates or {})
        for name in unset:
            environment.pop(name, None)
        return self.commands.run_observed(
            [str(item) for item in command],
            check=check,
            env=environment,
            timeout=self._duration_seconds(limit),
            cwd=cwd,
        )

    @staticmethod
    def _duration_seconds(value: str | None) -> int | None:
        if not value:
            return None
        match = re.fullmatch(r"([1-9][0-9]*)([smhd]?)", value)
        if match is None:
            raise CiError(f"command timeout must be a positive duration like 30s or 45m: {value}")
        scale = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[match.group(2)]
        return int(match.group(1)) * scale

    def executable(self, name: str) -> str:
        path = shutil.which(name, path=self.env.get("PATH"))
        if not path:
            raise CiError(f"Required executable was not found on PATH: {name}")
        return path

    def read_json(self, path: str | Path) -> dict[str, object]:
        return json.loads((self.repository / path).read_text(encoding="utf-8"))

    def write_json(self, path: str | Path, value: object) -> None:
        destination = self.repository / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def write_state(self, name: str, value: Mapping[str, str]) -> Path:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        destination = self.state_dir / name
        destination.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination

    def read_state(self, name: str) -> dict[str, str]:
        path = self.state_dir / name
        if not path.is_file():
            raise CiError(f"Reusable CI state is missing: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(item, str) for key, item in value.items()
        ):
            raise CiError(f"Reusable CI state is invalid: {path}")
        return value

    @staticmethod
    def positive_integer(value: str, name: str) -> int:
        if not value.isdigit() or int(value) < 1:
            raise CiError(f"{name} must be a positive integer")
        return int(value)

    def remove(self, *paths: str | Path) -> None:
        for value in paths:
            path = self.repository / value
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
