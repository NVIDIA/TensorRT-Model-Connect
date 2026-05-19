from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

from setuptools import Distribution
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except Exception:  # pragma: no cover - older setuptools falls back to wheel.
    try:
        from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    except Exception:
        _bdist_wheel = None


PACKAGE_NAME = "tensorrt_model_connect"
MANYLINUX_AARCH64_PLATFORM = "manylinux_2_35_aarch64"
NATIVE_LIB_PATTERNS = (
    "libtrtmc*.so",
    "libtrtmc*.so.*",
    "libtrtmc*.dylib",
    "trtmc*.dll",
)
NATIVE_BACKEND_PATTERNS = (
    "libtrtmc_backend*.so",
    "libtrtmc_backend*.so.*",
    "libtrtmc_backend*.dylib",
    "trtmc_backend*.dll",
)


def _truthy(value: str | None) -> bool:
    return value not in (None, "", "0", "false", "False", "no", "No")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _native_binary_candidates() -> list[Path]:
    candidates: list[Path] = []
    override = os.environ.get("TRTMC_NATIVE_BIN")
    if override:
        candidates.append(Path(override).expanduser())
    candidates.append(_project_root() / "build" / "trtmc")
    return candidates


def _find_native_binary() -> Path | None:
    for candidate in _native_binary_candidates():
        if candidate.is_file():
            return candidate.resolve()
    return None


def _native_library_dirs(binary: Path) -> list[Path]:
    dirs: list[Path] = []
    override = os.environ.get("TRTMC_NATIVE_LIB_DIR")
    if override:
        dirs.append(Path(override).expanduser())
    dirs.append(binary.parent)
    return dirs


class build_py(_build_py):
    def run(self) -> None:
        super().run()
        self._native_outputs: list[str] = []
        self._copy_native_artifacts()

    def get_outputs(self, include_bytecode: int = 1) -> list[str]:
        outputs = super().get_outputs(include_bytecode)
        return [*outputs, *getattr(self, "_native_outputs", [])]

    def _copy_native_artifacts(self) -> None:
        binary = _find_native_binary()
        if binary is None:
            if _truthy(os.environ.get("TRTMC_REQUIRE_NATIVE_BIN")):
                candidates = ", ".join(str(path) for path in _native_binary_candidates())
                raise RuntimeError(f"TRTMC native binary was not found; checked: {candidates}")
            return

        dest_dir = Path(self.build_lib) / PACKAGE_NAME / "bin"
        dest_dir.mkdir(parents=True, exist_ok=True)
        self._copy_executable(binary, dest_dir / "trtmc")
        seen = {binary.resolve()}
        copied_backends: list[Path] = []

        for lib_dir in _native_library_dirs(binary):
            if not lib_dir.is_dir():
                continue
            for pattern in NATIVE_LIB_PATTERNS:
                for source in sorted(lib_dir.glob(pattern)):
                    resolved = source.resolve()
                    if resolved in seen or not source.is_file():
                        continue
                    seen.add(resolved)
                    self._copy_file(source, dest_dir / source.name)
                    if _matches_any(source.name, NATIVE_BACKEND_PATTERNS):
                        copied_backends.append(source)

        if _truthy(os.environ.get("TRTMC_REQUIRE_NATIVE_LIBS")) and not copied_backends:
            dirs = ", ".join(str(path) for path in _native_library_dirs(binary))
            raise RuntimeError(
                "TRTMC native backend libraries were not found; "
                f"checked {dirs}. Build the TensorRT backend DSO before packaging."
            )

    def _copy_file(self, source: Path, destination: Path) -> None:
        shutil.copy2(source, destination)
        self._native_outputs.append(str(destination))

    def _copy_executable(self, source: Path, destination: Path) -> None:
        self._copy_file(source, destination)
        mode = destination.stat().st_mode
        destination.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    return any(Path(name).match(pattern) for pattern in patterns)


def _wheel_python_tag(default: str) -> str:
    override = os.environ.get("TRTMC_WHEEL_PYTHON_TAG")
    if not override:
        return default

    aliases = {
        "py10": "py310",
        "py12": "py312",
        "3.10": "py310",
        "3.12": "py312",
    }
    tag = aliases.get(override, override)
    if tag in {"py3", "py310", "py312"}:
        return tag
    raise RuntimeError(
        f"invalid TRTMC_WHEEL_PYTHON_TAG={override!r}; expected py310, py312, or py3"
    )


def _wheel_platform_tag(default: str) -> str:
    override = os.environ.get("TRTMC_WHEEL_PLATFORM_TAG")
    if override:
        aliases = {
            "aarch64": MANYLINUX_AARCH64_PLATFORM,
            "linux_aarch64": MANYLINUX_AARCH64_PLATFORM,
            "manylinux": MANYLINUX_AARCH64_PLATFORM,
            "manylinux_2_35": MANYLINUX_AARCH64_PLATFORM,
        }
        return aliases.get(override, override)

    if default == "linux_aarch64":
        return MANYLINUX_AARCH64_PLATFORM
    return default


class BinaryDistribution(Distribution):
    def has_ext_modules(self) -> bool:
        return True

    def is_pure(self) -> bool:
        return False


cmdclass = {"build_py": build_py}

if _bdist_wheel is not None:

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self) -> None:
            super().finalize_options()
            self.root_is_pure = False

        def get_tag(self) -> tuple[str, str, str]:
            _python, _abi, platform = super().get_tag()
            return _wheel_python_tag("py3"), "none", _wheel_platform_tag(platform)

    cmdclass["bdist_wheel"] = bdist_wheel


setup(cmdclass=cmdclass, distclass=BinaryDistribution)
