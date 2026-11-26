# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI environment forwarding and opt-in isolated reference preparation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import sysconfig

from .context import CiContext
from .process import CiError


COMMON_ENVIRONMENT = (
    "CI_BASE_REF",
    "GITHUB_RUN_ID",
    "GITHUB_RUN_ATTEMPT",
    "SOURCE_QUALITY_TIMEOUT",
    "PYTHON_UNIT_TIMEOUT",
    "CPP_BUILD_TIMEOUT",
    "CPP_UNIT_TIMEOUT",
)

TRUSTED_ENVIRONMENT = COMMON_ENVIRONMENT + (
    "MODEL",
    "TRTMC_BINARY",
    "TRTMC_RUNTIME_ROOT",
    "TRTMC_NATIVE_BUILD_DIR",
    "TRTMC_E2E_TIMEOUT",
    "TRTMC_E2E_STOP_AFTER",
    "TRTMC_CI_STATE_DIR",
    "TRT_ROOT",
    "CMAKE_CUDA_ARCHITECTURES",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_MODULES_CACHE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)

OPTIONAL_HUGGING_FACE_ENVIRONMENT = (
    "HF_HUB_DISABLE_XET",
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "HF_HUB_ETAG_TIMEOUT",
)


def forwarded_environment(names: tuple[str, ...], env: dict[str, str]) -> tuple[str, ...]:
    model_directories = sorted(
        name
        for name, value in env.items()
        if name.startswith("TRTMC_") and name.endswith("_MODEL_DIR") and value
    )
    return tuple(dict.fromkeys((*names, *model_directories)))


def environment_directory(repository: Path, family: str, env: dict[str, str]) -> Path:
    if re.fullmatch(r"[a-z][a-z0-9_]*", family) is None:
        raise CiError(f"Invalid reference family: {family!r}")
    return repository / (env.get("TRTMC_CI_STATE_DIR") or ".ci") / "reference" / family


def prepared_python(repository: Path, family: str, env: dict[str, str]) -> Path:
    directory = environment_directory(repository, family, env)
    python = directory / "bin/python"
    receipt = directory / "ready.json"
    requirements = repository / "families" / family / "tests/requirements.txt"
    try:
        ready = json.loads(receipt.read_text(encoding="utf-8"))
        current = requirements.read_text(encoding="utf-8")
    except (OSError, ValueError) as error:
        raise CiError(
            f"Reference environment is not prepared; run MODEL={family} "
            "python -m tools.ci pipeline setup before offline execution"
        ) from error
    if (
        not isinstance(ready, dict)
        or ready.get("requirements") != current
        or not python.is_file()
        or not os.access(python, os.X_OK)
    ):
        raise CiError(f"Reference environment is incomplete or stale: {directory}")
    # Keep the venv interpreter path: resolving its symlink loses the environment.
    return python.absolute()


def _base_site_packages() -> list[str]:
    return sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})


class ReferenceEnvironment:
    def __init__(self, context: CiContext):
        self.context = context

    def prepare_selected(self) -> None:
        family = self.context.env.get("MODEL", "")
        if not family:
            return
        directory = environment_directory(self.context.repository, family, self.context.env)
        tests = self.context.repository / "families" / family / "tests"
        declarations = [
            json.loads(path.read_text(encoding="utf-8")).get("reference_environment")
            for path in sorted((tests / "manifests").glob("*.json"))
        ]
        if any(value not in (None, "isolated") for value in declarations):
            raise CiError(f"Unsupported reference environment declaration for {family}")
        if "isolated" not in declarations:
            return
        requirements = tests / "requirements.txt"
        if not requirements.is_file():
            raise CiError(f"Reference requirements are missing: {requirements}")
        contents = requirements.read_text(encoding="utf-8")
        receipt = directory / "ready.json"
        receipt.unlink(missing_ok=True)
        self.context.run([sys.executable, "-m", "venv", directory])
        python = directory / "bin/python"
        result = self.context.run(
            [python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            capture_output=True,
        )
        site = Path(result.stdout.strip())
        if not site.is_relative_to(directory.absolute()):
            raise CiError(f"Reference interpreter returned an invalid package directory: {site}")
        # A nested venv does not inherit its parent venv's site-packages.
        # Append them after its own packages to reuse the installed CUDA/Torch
        # stack while reference package overrides stay private to this venv.
        (site / "trtmc_base_packages.pth").write_text(
            "\n".join(_base_site_packages()) + "\n", encoding="utf-8"
        )
        self.context.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--requirement",
                requirements,
            ],
            unset=("PYTHONPATH", "PYTHONHOME", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"),
        )
        receipt.write_text(json.dumps({"requirements": contents}) + "\n", encoding="utf-8")
        print(
            f"Prepared isolated reference interpreter: {prepared_python(self.context.repository, family, self.context.env)}"
        )
