# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and verify the one native wheel produced by the new architecture."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

from .context import CiContext
from .process import CiError


WHEEL_STATE = "wheel.json"


def family_ids(repository: Path) -> tuple[str, ...]:
    families = tuple(
        sorted(path.parent.name for path in (repository / "families").glob("*/model.py"))
    )
    if not families:
        raise CiError("repository has no Python model families")
    return families


class SourceArchiveValidator:
    """Require the source archive to carry the physical dependency declarations."""

    def __init__(self, context: CiContext):
        self.context = context

    def validate(self, archives: list[Path]) -> None:
        if len(archives) != 1:
            raise CiError(f"expected one source archive, found {len(archives)}")
        archive = archives[0]
        with tarfile.open(archive, "r:gz") as source:
            members = [Path(member.name) for member in source.getmembers() if member.isfile()]
        if not members or any(path.is_absolute() or ".." in path.parts for path in members):
            raise CiError(f"{archive}: source archive contains an unsafe path")
        roots = {path.parts[0] for path in members if path.parts}
        if len(roots) != 1:
            raise CiError(f"{archive}: source archive must have one root directory")
        packaged = {Path(*path.parts[1:]).as_posix() for path in members if len(path.parts) > 1}
        expected = {"requirements/base.txt"} | {
            path.relative_to(self.context.repository).as_posix()
            for path in (self.context.repository / "families").glob("*/requirements.txt")
        }
        missing = sorted(expected - packaged)
        if missing:
            raise CiError(f"{archive}: dependency declarations are missing: {missing}")
        print(f"validated source archive={archive} family_requirements={len(expected) - 1}")


def load_native_libraries(bin_dir: Path, families: tuple[str, ...]) -> None:
    """Load every installed DSO now so unresolved symbols fail packaging."""

    libraries = [
        bin_dir / "libtrtmc_core.so",
        bin_dir / "libtrtmc_runtime.so",
        bin_dir / "libtrtmc_backend_trt.so",
        bin_dir / "libtrtmc_byok_tvm_ffi.so",
        *(bin_dir / f"libtrtmc_model_{family}.so" for family in families),
    ]
    script = """
import ctypes
import os
import sys

ctypes.CDLL(sys.argv[1], mode=os.RTLD_NOW | ctypes.RTLD_GLOBAL)
for path in sys.argv[2:]:
    ctypes.CDLL(path, mode=os.RTLD_NOW | ctypes.RTLD_LOCAL)
"""
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(bin_dir), environment.get("LD_LIBRARY_PATH", "")) if value
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, *(str(path) for path in libraries)],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path("/tmp"),
        env=environment,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise CiError(f"installed native library failed RTLD_NOW: {detail}")


class WheelArchiveValidator:
    """Require only the Python packages and native artifacts users execute."""

    def __init__(self, context: CiContext, _platform: str | None = None):
        self.context = context

    def validate(self, wheels: list[Path]) -> None:
        if len(wheels) != 1:
            raise CiError(f"expected one wheel, found {len(wheels)}")
        wheel = wheels[0]
        expected_families = family_ids(self.context.repository)
        with zipfile.ZipFile(wheel) as archive:
            names = tuple(name for name in archive.namelist() if not name.endswith("/"))
            generated = [
                name for name in names if "__pycache__" in Path(name).parts or name.endswith(".pyc")
            ]
            if generated:
                raise CiError(f"{wheel}: generated Python cache files are packaged")
            if "tensorrt_model_connect/__init__.py" not in names:
                raise CiError(f"{wheel}: Python core package is missing")
            if "trtmc_benchmark/__init__.py" not in names:
                raise CiError(f"{wheel}: Python benchmark application is missing")
            source_suffixes = {
                ".c",
                ".cc",
                ".cpp",
                ".cu",
                ".cuh",
                ".h",
                ".hpp",
                ".py",
                ".pyc",
            }
            expected_catalog = {
                "trtmc_benchmark/_catalog/"
                + path.relative_to(self.context.repository / "families").as_posix()
                for path in (self.context.repository / "families").glob("*/tests/**/*")
                if path.is_file()
                and path.suffix not in source_suffixes
                and "__pycache__" not in path.parts
            }
            missing_catalog = sorted(expected_catalog - set(names))
            if missing_catalog:
                raise CiError(f"{wheel}: benchmark catalog is missing: {missing_catalog}")
            if "families/__init__.py" not in names:
                raise CiError(f"{wheel}: Python families package is missing")
            metadata_files = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(metadata_files) != 1:
                raise CiError(f"{wheel}: package metadata is missing")
            metadata = archive.read(metadata_files[0]).decode("utf-8")
            extras = sorted(
                line.partition(":")[2].strip()
                for line in metadata.splitlines()
                if line.startswith("Provides-Extra:")
            )
            if extras != ["cutedsl", "test"]:
                raise CiError(f"{wheel}: expected only application extras, found {extras}")
            packaged_python = tuple(
                sorted(
                    Path(name).parts[1]
                    for name in names
                    if len(Path(name).parts) == 3
                    and Path(name).parts[0] == "families"
                    and Path(name).parts[2] == "model.py"
                )
            )
            if packaged_python != expected_families:
                raise CiError(f"{wheel}: Python family set is incomplete")
            packaged_names = set(names)
            expected_python = {
                path.relative_to(self.context.repository).as_posix()
                for family in expected_families
                for path in (self.context.repository / "families" / family).rglob("*.py")
                if "tests"
                not in path.relative_to(self.context.repository / "families" / family).parts
                and "__pycache__" not in path.parts
            }
            missing_python = sorted(expected_python - packaged_names)
            if missing_python:
                raise CiError(f"{wheel}: family Python files are missing: {missing_python}")
            mismatched_python = sorted(
                name
                for name in expected_python
                if archive.read(name)
                != (self.context.repository / name).read_bytes()
            )
            if mismatched_python:
                raise CiError(
                    f"{wheel}: family Python files differ from Source: {mismatched_python}"
                )
            invalid_python = []
            for name in sorted(expected_python):
                try:
                    compile(archive.read(name), name, "exec")
                except (SyntaxError, UnicodeError):
                    invalid_python.append(name)
            if invalid_python:
                raise CiError(
                    f"{wheel}: family Python files do not compile: {invalid_python}"
                )

            expected_family_data = {
                path.relative_to(self.context.repository).as_posix()
                for family in expected_families
                for path in (self.context.repository / "families" / family).rglob("*")
                if path.is_file()
                and path.suffix != ".py"
                and not {
                    "tests",
                    "runtime",
                    "native_plugins",
                    "__pycache__",
                }
                & set(path.relative_to(self.context.repository / "families" / family).parts)
            }
            missing_family_data = sorted(expected_family_data - packaged_names)
            if missing_family_data:
                raise CiError(f"{wheel}: family build data is missing: {missing_family_data}")

            expected_plugin_sources = {
                path.relative_to(self.context.repository).as_posix()
                for directory in (self.context.repository / "families").rglob("native_plugins")
                if directory.is_dir()
                for path in directory.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            }
            missing_plugin_sources = sorted(expected_plugin_sources - packaged_names)
            if missing_plugin_sources:
                raise CiError(
                    f"{wheel}: installed family plugin builder sources are missing: "
                    f"{missing_plugin_sources}"
                )

            module_bins = {
                Path(name).name for name in names if name.startswith("tensorrt_model_connect/bin/")
            }
            required = {
                "trtmc",
                "libtrtmc_core.so",
                "libtrtmc_runtime.so",
                "libtrtmc_backend_trt.so",
                "libtrtmc_byok_tvm_ffi.so",
                "trtmc_benchmark_worker",
                "trtmc_dataset_benchmark",
            }
            missing = sorted(required - module_bins)
            if missing:
                raise CiError(f"{wheel}: native package is missing {', '.join(missing)}")
            family_dsos = tuple(
                sorted(
                    name.removeprefix("libtrtmc_model_").removesuffix(".so")
                    for name in module_bins
                    if name.startswith("libtrtmc_model_") and name.endswith(".so")
                )
            )
            if family_dsos != expected_families:
                raise CiError(f"{wheel}: family DSO set is incomplete")
            backend_dsos = sorted(
                name for name in module_bins if name.startswith("libtrtmc_backend_")
            )
            if backend_dsos != ["libtrtmc_backend_trt.so"]:
                raise CiError(f"{wheel}: expected one unaliased TensorRT backend DSO")
            scripts = [name for name in names if name.endswith(".data/scripts/trtmc")]
            script_cores = [
                name for name in names if name.endswith(".data/scripts/libtrtmc_core.so")
            ]
            script_runtimes = [
                name for name in names if name.endswith(".data/scripts/libtrtmc_runtime.so")
            ]
            if len(scripts) != 1 or len(script_cores) != 1 or len(script_runtimes) != 1:
                raise CiError(f"{wheel}: installed CLI payload is incomplete")
            entry_points = [
                name for name in names if name.endswith(".dist-info/entry_points.txt")
            ]
            if len(entry_points) != 1 or "trtmc-bench" not in archive.read(
                entry_points[0]
            ).decode("utf-8"):
                raise CiError(f"{wheel}: trtmc-bench console entrypoint is missing")
        print(f"validated wheel={wheel} families={len(expected_families)}")


class InstalledWheelValidator:
    """Verify imports and exact native files after installation."""

    def __init__(self, repository: Path):
        self.repository = repository

    def validate(self, wheel: Path) -> None:
        script = """
import importlib
import json
from pathlib import Path
import sysconfig

core = importlib.import_module("tensorrt_model_connect")
families = importlib.import_module("families")
print(json.dumps({
    "core": str(Path(core.__file__).resolve()),
    "families": str(Path(families.__file__).resolve()),
    "family_requirements": sorted(
        path.parent.name for path in Path(families.__file__).resolve().parent.glob("*/requirements.txt")
    ),
    "bin": str(Path(core.__file__).resolve().parent / "bin"),
    "scripts": sysconfig.get_path("scripts"),
}))
"""
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path("/tmp"),
            env=environment,
        )
        payload = json.loads(completed.stdout)
        imported = (Path(payload["core"]), Path(payload["families"]))
        if any(path.is_relative_to(self.repository.resolve()) for path in imported):
            raise CiError("installed wheel validation imported the source checkout")
        bin_dir = Path(payload["bin"])
        expected = set(family_ids(self.repository))
        expected_requirements = {
            path.parent.name
            for path in (self.repository / "families").glob("*/requirements.txt")
        }
        if set(payload["family_requirements"]) != expected_requirements:
            raise CiError(f"installed family dependency declarations are incomplete: {wheel}")
        packaged = {
            path.name.removeprefix("libtrtmc_model_").removesuffix(".so")
            for path in bin_dir.glob("libtrtmc_model_*.so")
        }
        required = (
            bin_dir / "trtmc",
            bin_dir / "libtrtmc_core.so",
            bin_dir / "libtrtmc_runtime.so",
            bin_dir / "libtrtmc_backend_trt.so",
            bin_dir / "libtrtmc_byok_tvm_ffi.so",
            bin_dir / "trtmc_benchmark_worker",
            bin_dir / "trtmc_dataset_benchmark",
        )
        if not all(path.is_file() for path in required) or packaged != expected:
            raise CiError(f"installed wheel is incomplete: {wheel}")
        load_native_libraries(bin_dir, tuple(sorted(expected)))
        executable = Path(payload["scripts"]) / "trtmc"
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise CiError(f"installed trtmc CLI is missing: {executable}")
        if executable.resolve().is_relative_to(self.repository.resolve()):
            raise CiError("installed trtmc CLI resolves into the source checkout")
        benchmark_cli = Path(payload["scripts"]) / "trtmc-bench"
        if not benchmark_cli.is_file() or not os.access(benchmark_cli, os.X_OK):
            raise CiError(f"installed trtmc-bench CLI is missing: {benchmark_cli}")
        benchmark_help = subprocess.run(
            [benchmark_cli, "--help"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path("/tmp"),
            env=environment,
        )
        if "Task API benchmarks" not in benchmark_help.stdout:
            raise CiError("installed trtmc-bench CLI returned invalid help")
        version = subprocess.run(
            [executable, "version"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path("/tmp"),
            env=environment,
        )
        if not version.stdout.startswith("trtmc "):
            raise CiError("installed trtmc CLI returned an invalid version")
        with tempfile.TemporaryDirectory(prefix="trtmc-installed-wheel-") as directory:
            bundle = Path(directory) / "inspect.bundle"
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; "
                    "from tensorrt_model_connect.bundle_writer import BundleWriter; "
                    "writer = BundleWriter(Path(__import__('sys').argv[1])); "
                    "writer.set_header(family='inspect', task='text_generation', backend='trt'); "
                    "writer.finish()",
                    bundle,
                ],
                check=True,
                cwd=Path("/tmp"),
                env=environment,
            )
            inspected = subprocess.run(
                [executable, "inspect", bundle],
                check=True,
                capture_output=True,
                text=True,
                cwd=Path("/tmp"),
                env=environment,
            )
            metadata = json.loads(inspected.stdout)
            if metadata.get("family") != "inspect" or metadata.get("backend") != "trt":
                raise CiError("installed trtmc CLI failed bundle inspection")
        print(f"installed wheel={wheel} trtmc={executable} families={len(packaged)}")


class WheelPackageManager:
    """Build, install, and validate the one native wheel."""

    def __init__(self, context: CiContext):
        self.context = context

    def preflight(self) -> None:
        families = family_ids(self.context.repository)
        self.context.run(["python", "tools/model_ci.py", "validate"])
        print(f"package preflight families={len(families)}")

    def build(self) -> None:
        self.preflight()
        self.context.remove("dist")
        (self.context.repository / "dist").mkdir(parents=True, exist_ok=True)
        self.context.run(
            [
                "python",
                "-m",
                "build",
                "--no-isolation",
                "--sdist",
                "--outdir",
                "dist",
                ".",
            ]
        )
        archives = sorted((self.context.repository / "dist").glob("*.tar.gz"))
        SourceArchiveValidator(self.context).validate(archives)
        self.context.run(
            [
                "python",
                "-m",
                "build",
                "--no-isolation",
                "--wheel",
                "--outdir",
                "dist",
                ".",
            ]
        )
        wheels = sorted((self.context.repository / "dist").glob("*.whl"))
        WheelArchiveValidator(self.context).validate(wheels)
        self.context.write_state(WHEEL_STATE, {"wheel": str(wheels[0])})

    def install_once(self) -> Path:
        state = self.context.state_dir / WHEEL_STATE
        if not state.is_file():
            wheel = self.select_compatible_wheel()
            WheelArchiveValidator(self.context).validate([wheel])
            self.context.write_state(WHEEL_STATE, {"wheel": str(wheel)})
        wheel = Path(self.context.read_state(WHEEL_STATE)["wheel"])
        self.context.run(
            [
                "python",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--force-reinstall",
                "--no-deps",
                wheel,
            ]
        )
        InstalledWheelValidator(self.context.repository).validate(wheel)
        return wheel

    def verify_installed(self) -> None:
        wheel = Path(self.context.read_state(WHEEL_STATE)["wheel"])
        InstalledWheelValidator(self.context.repository).validate(wheel)

    def select_compatible_wheel(self, directory: str = "dist") -> Path:
        wheels = sorted((self.context.repository / directory).glob("*.whl"))
        if len(wheels) != 1:
            raise CiError(f"expected one wheel under {directory}, found {len(wheels)}")
        return wheels[0]
