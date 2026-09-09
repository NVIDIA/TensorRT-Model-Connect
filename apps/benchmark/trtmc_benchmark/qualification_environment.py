# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare family-owned Python environments without importing model dependencies."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence


_PROBE = """
import importlib.metadata, json, platform, sys
print(json.dumps({
    "executable": sys.executable,
    "prefix": sys.prefix,
    "python": platform.python_version(),
    "packages": sorted((d.metadata['Name'], d.version)
                       for d in importlib.metadata.distributions()),
}))
"""


def process_environment() -> dict[str, str]:
    """Keep runner device/cache allocation, but do not inject another Python's packages."""
    environment = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def python_command(python: str, *arguments: str) -> list[str]:
    return [str(Path(python).expanduser().absolute()), *arguments]


def inspect_python(python: str, timeout: int) -> dict[str, Any]:
    completed = subprocess.run(
        python_command(python, "-c", _PROBE),
        env=process_environment(),
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )
    evidence = json.loads(completed.stdout)
    if not isinstance(evidence, dict) or not evidence.get("python"):
        raise ValueError("Python environment probe returned invalid evidence")
    return evidence


def prepare_family_environment(
    *,
    family_root: Path,
    cases: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any],
    common_python: Path,
    directory: Path,
    timeout: int,
    reuse: bool,
) -> dict[str, Any]:
    """Resolve once per family; a reused receipt must still describe the actual packages."""
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "environment.json"
    if reuse and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for role, python in receipt["interpreters"].items():
            if inspect_python(python, timeout) != receipt["evidence"][role]:
                raise ValueError(
                    f"{family_root.name}: {role} environment changed; prepare a new run"
                )
        return receipt

    tools = environment.get("tools", {})
    python = str(Path(str(tools.get("python", common_python))).expanduser().absolute())
    interpreters = {"python": python, "reference_python": python}
    hook = family_root / "tests" / "qualification" / "prepare_environment.py"
    command: list[str] = []
    if hook.is_file():
        if hook.is_symlink():
            raise ValueError("family environment preparation must be a regular local file")
        root = Path(environment.get("storage", {}).get("environment_root", directory / "python"))
        request_path, output_path = directory / "request.json", directory / "resolved.json"
        request = {
            "common_python": python,
            "family_root": str(family_root),
            "cases": list(cases),
            "environment_directory": str(root.expanduser().absolute() / family_root.name),
            "allow_create": environment.get("execution", {}).get(
                "allow_environment_creation", False
            ),
        }
        request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
        command = python_command(
            python, str(hook), "--request", str(request_path), "--output", str(output_path)
        )
        before = inspect_python(python, timeout)
        try:
            with (
                (directory / "stdout.log").open("w") as stdout,
                (directory / "stderr.log").open("w") as stderr,
            ):
                subprocess.run(
                    command,
                    cwd=directory,
                    env=process_environment(),
                    stdout=stdout,
                    stderr=stderr,
                    check=True,
                    timeout=timeout,
                )
        finally:
            if inspect_python(python, timeout) != before:
                raise ValueError("family preparation modified the common Python environment")
        resolved = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(resolved, dict) or not isinstance(resolved.get("python"), str):
            raise ValueError("family preparation must return a python path")
        interpreters = {
            "python": resolved["python"],
            "reference_python": resolved.get("reference_python", resolved["python"]),
        }
    for role, value in interpreters.items():
        if not isinstance(value, str) or not Path(value).is_absolute() or not Path(value).is_file():
            raise ValueError(f"{role} must be an existing absolute interpreter path")
    receipt = {
        "interpreters": interpreters,
        "evidence": {role: inspect_python(value, timeout) for role, value in interpreters.items()},
        "command": command,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt
