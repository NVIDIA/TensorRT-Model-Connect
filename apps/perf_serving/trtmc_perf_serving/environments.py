# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference Python environments: a requirements file layered on the serving interpreter.

A model whose native reference needs packages the serving interpreter lacks (remote code, a
family's own dependencies) runs in a virtual environment that sees this interpreter's packages and
adds the requirements on top. Environments are cached by the digest of the requirements, the
interpreter, and the install options, so each is created once per host. Each records its ``pip freeze``
and is reused only while that is unchanged; a changed one is left in place and a fresh one created.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import site
import subprocess
import sys
from pathlib import Path

PIP_TIMEOUT_S = 7200  # native extensions (for example FlashAttention) build for a long time on aarch64


def _digest(requirements: Path, build_isolation: bool, prepare: Path | None = None) -> str:
    digest = hashlib.sha256(b"trtmc-perf-serve-reference-env-v1\0")
    digest.update(requirements.read_bytes())
    if prepare is not None:
        digest.update(b"\0prepare\0" + prepare.read_bytes())
    digest.update(f"{sys.executable}\0{sys.version}\0build-isolation={build_isolation}".encode())
    return digest.hexdigest()


def _inherit_site_packages(environment: Path) -> None:
    """Make the serving interpreter's packages visible: ``--system-site-packages`` of a venv created
    from another venv only reaches the base interpreter."""
    children = sorted(environment.glob("lib/python*/site-packages"))
    if len(children) != 1:
        raise RuntimeError(f"no unambiguous site-packages directory in {environment}")
    parents = sorted({str(Path(value).resolve()) for value in site.getsitepackages() if Path(value).is_dir()})
    (children[0] / "trtmc-serving-environment.pth").write_text("\n".join(parents) + "\n")


def _run(command: list[str], log: Path, timeout: int) -> None:
    env = {**os.environ, "MAX_JOBS": os.environ.get("MAX_JOBS", "4")}  # bounded extension builds
    with open(log, "a") as handle:
        handle.write(f"+ {' '.join(command)}\n")
        handle.flush()
        code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env, timeout=timeout).returncode
    if code != 0:
        raise RuntimeError(f"{' '.join(command[:4])} failed (exit {code}); see {log}")


def reference_python(requirements: Path, root: Path, *, build_isolation: bool = True,
                     prepare: Path | None = None) -> Path:
    """The interpreter of the environment for ``requirements`` (created on first use); ``prepare`` runs once
    after the install, in the environment's directory (for example an upstream checkout)."""
    requirements = requirements.resolve()
    if not requirements.is_file():
        raise RuntimeError(f"requirements file not found: {requirements}")
    if prepare is not None and not prepare.resolve().is_file():
        raise RuntimeError(f"preparation script not found: {prepare}")
    expected = _digest(requirements, build_isolation, prepare.resolve() if prepare else None)
    name = requirements.parent.name if requirements.name == "requirements.txt" else requirements.stem
    for attempt in itertools.count():
        environment = root.resolve() / (f"{name}-{expected[:12]}" + (f"-r{attempt}" if attempt else ""))
        python, stamp, frozen = environment / "bin/python", environment / ".requirements.sha256", environment / ".freeze"
        if not environment.exists():
            break  # created here
        if python.is_file() and stamp.is_file() and stamp.read_text().strip() == expected and frozen.is_file():
            try:
                if _freeze(python) == frozen.read_text():
                    return python
            except (OSError, subprocess.SubprocessError):  # a broken interpreter: replaced like a changed one
                pass
        # Unfinished, changed since its creation, or without its freeze record: left as it is, and the next
        # directory is created afresh.
    environment.parent.mkdir(parents=True, exist_ok=True)
    log = environment.parent / f"{environment.name}.setup.log"
    _run([sys.executable, "-m", "venv", "--system-site-packages", str(environment)], log, timeout=600)
    _inherit_site_packages(environment)
    install = [str(python), "-m", "pip", "install", "--disable-pip-version-check"]
    if not build_isolation:
        install.append("--no-build-isolation")
    _run([*install, "-r", str(requirements)], log, timeout=PIP_TIMEOUT_S)
    if prepare is not None:
        _run([str(python), str(prepare.resolve()), str(environment)], log, timeout=PIP_TIMEOUT_S)
    frozen.write_text(_freeze(python))
    stamp.write_text(expected + "\n")
    return python


def _freeze(python: Path) -> str:
    """The environment's resolved packages (``pip freeze --all``), recorded at creation, verified at reuse."""
    return subprocess.run([str(python), "-m", "pip", "freeze", "--all"], capture_output=True, text=True,
                          check=True, timeout=300).stdout
