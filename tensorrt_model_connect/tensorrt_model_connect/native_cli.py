"""Pip entry point for the packaged native ``trtmc`` executable."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib import util as importlib_util
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
        "trtmc native executable was not found. Install a release wheel that "
        "includes native artifacts, build ./build/trtmc from source, or set "
        "TRTMC_NATIVE_BIN to an existing native trtmc executable."
    )


def _tensorrt_library_dir() -> Path | None:
    spec = importlib_util.find_spec("tensorrt_libs")
    if spec is None:
        return None
    locations = spec.submodule_search_locations
    if locations:
        path = Path(next(iter(locations)))
        return path if path.is_dir() else None
    if spec.origin:
        path = Path(spec.origin).parent
        return path if path.is_dir() else None
    return None


def _configure_runtime_environment() -> None:
    os.environ.setdefault("TRTMC_PYTHON", sys.executable)
    os.environ.setdefault("TRTMC_DISABLE_SOURCE_PYTHONPATH", "1")
    if sys.prefix != sys.base_prefix:
        os.environ.setdefault("VIRTUAL_ENV", sys.prefix)

    trt_lib_dir = _tensorrt_library_dir()
    if trt_lib_dir is not None:
        os.environ.setdefault("TRTMC_TRT_LIBRARY_DIR", str(trt_lib_dir))


def main() -> int:
    binary = _existing_executable(_native_binary_candidates())
    if binary is None:
        print(_missing_binary_message(), file=sys.stderr)
        return 127

    _configure_runtime_environment()
    argv = [str(binary), *sys.argv[1:]]

    if os.name == "posix":
        os.execv(str(binary), argv)
        raise AssertionError("os.execv returned unexpectedly")

    return subprocess.call(argv)


if __name__ == "__main__":
    raise SystemExit(main())
