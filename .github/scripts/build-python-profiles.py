# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize every declared Python profile during the CI image build."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path


def _load_profile_api():
    """Load the builder API without executing package application metadata."""
    package_name = "tensorrt_model_connect"
    package_root = Path(
        os.environ.get(
            "TRTMC_PYTHON_PROFILE_SOURCE",
            "/opt/trtmc-profile-source/tensorrt_model_connect",
        )
    ).resolve()
    module_name = f"{package_name}.python_profiles"
    package = types.ModuleType(package_name)
    package.__package__ = package_name
    package.__path__ = [str(package_root)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(
        module_name,
        package_root / "python_profiles.py",
    )
    if spec is None or spec.loader is None:
        raise SystemExit("unable to load the Python profile builder API")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    profile_api = _load_profile_api()
    names = profile_api.prebuilt_python_profile_names(
        profile_api.load_python_profile_registry()
    )
    if not names:
        raise SystemExit("no prebuilt Python profiles were declared")

    base_python = os.environ.get("TRTMC_BASE_PYTHON", "/opt/venv/bin/python")
    resolved: dict[str, dict[str, str]] = {}
    for name in names:
        python = profile_api.resolve_profile_python(name, base_python)
        ready = Path(python).parent.parent / ".ready"
        if not ready.is_file():
            raise SystemExit(f"profile {name!r} was not marked ready: {ready}")
        resolved[name] = {"python": python, "ready": str(ready)}

    manifest = {
        "schema_version": 1,
        "profiles": resolved,
    }
    root = profile_api.profile_root()
    (root / ".image-ready.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("prebuilt_python_profiles=" + ",".join(names))


if __name__ == "__main__":
    main()
