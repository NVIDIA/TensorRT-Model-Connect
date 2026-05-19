"""Pip entry point for the packaged native ``trtmc`` executable."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Iterable


def _existing_executable(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _native_binary_candidates() -> list[Path]:
    candidates: list[Path] = []
    override = os.environ.get("TRTMC_NATIVE_BIN")
    if override:
        candidates.append(Path(override).expanduser())

    try:
        candidates.append(
            Path(resources.files("tensorrt_model_connect").joinpath("bin", "trtmc"))
        )
    except (FileNotFoundError, ModuleNotFoundError):
        pass

    package_file = Path(__file__).resolve()
    candidates.append(package_file.parents[2] / "build" / "trtmc")
    return candidates


def _missing_binary_message() -> str:
    return (
        "trtmc native executable was not found. Install a wheel built with "
        "TRTMC_NATIVE_BIN=/path/to/trtmc, or set TRTMC_NATIVE_BIN to an existing "
        "native trtmc executable."
    )


def main() -> int:
    binary = _existing_executable(_native_binary_candidates())
    if binary is None:
        print(_missing_binary_message(), file=sys.stderr)
        return 127

    os.environ.setdefault("TRTMC_PYTHON", sys.executable)
    os.environ.setdefault("TRTMC_DISABLE_SOURCE_PYTHONPATH", "1")
    argv = [str(binary), *sys.argv[1:]]

    if os.name == "posix":
        os.execv(str(binary), argv)
        raise AssertionError("os.execv returned unexpectedly")

    return subprocess.call(argv)


if __name__ == "__main__":
    raise SystemExit(main())
