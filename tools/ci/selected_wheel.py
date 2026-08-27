# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install one explicitly selected release wheel for nightly validation.

Boundary: Python consumer isolation only; package certification remains in ``package``.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .context import CiContext
from .process import CiError


SELECTED_WHEEL_DIR_ENV = "TRTMC_SELECTED_WHEEL_DIR"
SELECTED_WHEEL_PYTHON_TAG_ENV = "TRTMC_SELECTED_WHEEL_PYTHON_TAG"
SELECTED_WHEEL_TENSORRT_VERSION_ENV = "TRTMC_SELECTED_WHEEL_TENSORRT_VERSION"
SELECTED_WHEEL_ENVIRONMENT = (
    SELECTED_WHEEL_DIR_ENV,
    SELECTED_WHEEL_PYTHON_TAG_ENV,
    SELECTED_WHEEL_TENSORRT_VERSION_ENV,
)
_EXACT_TENSORRT_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+")
_PYTHON_TAG = re.compile(r"py[0-9]+")


@dataclass(frozen=True)
class SelectedWheelContract:
    """Validated host or container view of the selected wheel artifact."""

    directory: Path
    python_tag: str
    tensorrt_version: str

    @classmethod
    def from_context(cls, context: CiContext) -> SelectedWheelContract | None:
        configured = context.env.get(SELECTED_WHEEL_DIR_ENV, "").strip()
        companions = {
            name: context.env.get(name, "").strip() for name in SELECTED_WHEEL_ENVIRONMENT[1:]
        }
        if not configured:
            unexpected = [name for name, value in companions.items() if value]
            if unexpected:
                raise CiError(
                    f"{SELECTED_WHEEL_DIR_ENV} is required when {', '.join(unexpected)} is set"
                )
            return None

        directory_input = Path(configured)
        if not directory_input.is_absolute():
            raise CiError(f"{SELECTED_WHEEL_DIR_ENV} must be an absolute path")
        try:
            directory = directory_input.resolve(strict=True)
        except OSError as error:
            raise CiError(f"selected wheel directory is unavailable: {directory_input}") from error
        if not directory.is_dir() or directory in {Path("/"), context.repository.resolve()}:
            raise CiError(f"unsafe selected wheel directory: {directory}")

        python_tag = companions[SELECTED_WHEEL_PYTHON_TAG_ENV]
        if not _PYTHON_TAG.fullmatch(python_tag):
            raise CiError(f"{SELECTED_WHEEL_PYTHON_TAG_ENV} must be an exact pyNNN tag")
        tensorrt_version = companions[SELECTED_WHEEL_TENSORRT_VERSION_ENV]
        if not _EXACT_TENSORRT_VERSION.fullmatch(tensorrt_version):
            raise CiError(
                f"{SELECTED_WHEEL_TENSORRT_VERSION_ENV} must be an exact four-part version"
            )
        return cls(directory, python_tag, tensorrt_version)


@dataclass(frozen=True)
class SelectedWheelRuntime:
    """Paths and safe provenance for one isolated target-installed wheel."""

    wheel: Path
    site_packages: Path
    python: Path
    trtmc: Path
    python_tag: str
    tensorrt_version: str
    package_version: str
    provenance: Path

    @classmethod
    def prepare(
        cls,
        context: CiContext,
        work: Path,
        provenance: Path,
        *,
        base_python: str | Path | None = None,
    ) -> SelectedWheelRuntime | None:
        contract = SelectedWheelContract.from_context(context)
        if contract is None:
            return None
        wheel = cls._select_wheel(contract)

        site_packages = work / "site-packages"
        if site_packages.exists():
            shutil.rmtree(site_packages)
        work.mkdir(parents=True, exist_ok=True)
        interpreter = base_python or (
            "/opt/venv/bin/python"
            if Path("/opt/venv/bin/python").is_file()
            else shutil.which("python") or "python"
        )
        # Preserve the virtualenv entrypoint. Resolving its symlink would invoke
        # the system interpreter and lose the base runtime's site-packages.
        python = Path(interpreter)
        context.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--force-reinstall",
                "--no-deps",
                "--target",
                site_packages,
                wheel,
            ],
            unset=("PYTHONPATH", "PYTHONHOME"),
        )
        trtmc = site_packages / "tensorrt_model_connect/bin/trtmc"
        package_version = cls._validate_install(
            context,
            contract,
            site_packages,
            python,
            trtmc,
        )

        payload = {
            "schema_version": 1,
            "wheel": wheel.name,
            "python_tag": contract.python_tag,
            "package_version": package_version,
            "tensorrt_version": contract.tensorrt_version,
            "environment": "isolated-pip-target",
            "import_source": "selected-wheel",
            "cli_source": "selected-wheel",
        }
        provenance.parent.mkdir(parents=True, exist_ok=True)
        provenance.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload, sort_keys=True))
        return cls(
            wheel=wheel,
            site_packages=site_packages,
            python=python,
            trtmc=trtmc,
            python_tag=contract.python_tag,
            tensorrt_version=contract.tensorrt_version,
            package_version=package_version,
            provenance=provenance,
        )

    @staticmethod
    def _select_wheel(contract: SelectedWheelContract) -> Path:
        patterns = (
            f"*-{contract.python_tag}-none-manylinux_*_aarch64.whl",
            f"*-{contract.python_tag}-none-linux_aarch64.whl",
        )
        candidates = sorted(
            {
                path.resolve()
                for pattern in patterns
                for path in contract.directory.glob(pattern)
                if path.is_file() and not path.is_symlink()
            }
        )
        if len(candidates) != 1:
            raise CiError(
                f"expected exactly one {contract.python_tag}-compatible Linux aarch64 wheel "
                f"under {contract.directory}, found {len(candidates)}: "
                f"{[path.name for path in candidates]}"
            )
        wheel = candidates[0]
        if not wheel.is_relative_to(contract.directory):
            raise CiError("selected wheel escapes its mounted artifact directory")
        return wheel

    @staticmethod
    def _validate_install(
        context: CiContext,
        contract: SelectedWheelContract,
        site_packages: Path,
        python: Path,
        trtmc: Path,
    ) -> str:
        probe = r"""
import importlib.metadata
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import tensorrt
import tensorrt_model_connect
import trtmc_server

print(json.dumps({
    "python": str(Path(sys.executable).resolve()),
    "python_tag": f"py{sys.version_info.major}{sys.version_info.minor}",
    "package_file": str(Path(tensorrt_model_connect.__file__).resolve()),
    "server_package_file": str(Path(trtmc_server.__file__).resolve()),
    "legacy_server_present": importlib.util.find_spec("tensorrt_model_connect.serve") is not None,
    "package_version": importlib.metadata.version("tensorrt-model-connect"),
    "tensorrt_distribution_version": importlib.metadata.version("tensorrt"),
    "tensorrt_runtime_version": tensorrt.__version__,
    "trtmc": str(Path(shutil.which("trtmc") or "").resolve()),
}, sort_keys=True))
"""
        target = site_packages.resolve()
        environment = {
            "PATH": f"{trtmc.parent}:{context.env.get('PATH', '')}",
            "PYTHONPATH": str(target),
        }
        try:
            payload = json.loads(
                context.output(
                    [python, "-c", probe],
                    updates=environment,
                    unset=("PYTHONHOME",),
                )
            )
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise CiError("selected wheel runtime probe returned invalid metadata") from error
        expected = {
            "python": str(python.resolve()),
            "python_tag": contract.python_tag,
            "tensorrt_distribution_version": contract.tensorrt_version,
            "tensorrt_runtime_version": contract.tensorrt_version,
            "trtmc": str(trtmc.resolve()),
            "legacy_server_present": False,
        }
        for name, value in expected.items():
            if payload.get(name) != value:
                raise CiError(
                    f"selected wheel runtime {name} mismatch: expected {value}, "
                    f"found {payload.get(name)}"
                )
        try:
            package_file = Path(str(payload["package_file"])).resolve(strict=True)
        except (KeyError, OSError) as error:
            raise CiError("selected wheel package import path is unavailable") from error
        if not package_file.is_relative_to(target):
            raise CiError(f"selected wheel package imported outside its target: {package_file}")
        try:
            server_package_file = Path(str(payload["server_package_file"])).resolve(strict=True)
        except (KeyError, OSError) as error:
            raise CiError("selected wheel server import path is unavailable") from error
        if not server_package_file.is_relative_to(target):
            raise CiError(
                f"selected wheel server imported outside its target: {server_package_file}"
            )
        if not trtmc.is_file() or trtmc.read_bytes()[:4] != b"\x7fELF":
            raise CiError(f"selected wheel did not install the native trtmc CLI: {trtmc}")
        package_version = payload.get("package_version")
        if not isinstance(package_version, str) or not package_version:
            raise CiError("selected wheel installed package version is missing")
        return package_version

    def environment(self, source: Path, base: dict[str, str]) -> dict[str, str]:
        """Return the Python-only environment that prevents source-package shadowing."""
        return {
            "PYTHONPATH": f"{self.site_packages}:{source}",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PATH": f"{self.trtmc.parent}:{base.get('PATH', '')}",
            "TRTMC_BINARY": str(self.trtmc),
            "TRTMC_HF_PYTHON": str(self.python),
            "TRTMC_TEST_INSTALLED_WHEEL": "1",
        }
