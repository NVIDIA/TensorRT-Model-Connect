# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-flight checks of an environment file on its machine (``doctor --environment``)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .config import Environment

# key -> kind: existing directory, executable file, or a directory created on first use.
PATHS = {"repo": "dir", "data_root": "dir", "runtime_root": "dir", "worker": "exe", "serve_python": "exe",
         "aiperf": "exe", "bundle_root": "created", "golden_store.root": "created",
         "hf_datasets_cache": "created", "reference_env_root": "created"}
SERVE_PACKAGES = ("torch", "transformers", "fastapi", "uvicorn", "multipart", "soundfile", "PIL", "numpy")


def _value(values: Mapping[str, Any], key: str) -> Any:
    for part in key.split("."):
        if not isinstance(values, Mapping) or part not in values:
            return None
        values = values[part]
    return values


def _writable_parent(path: Path) -> bool:
    parent = next((item for item in [path, *path.parents] if item.exists()), None)
    return parent is not None and os.access(parent, os.W_OK)


def check_paths(environment: Environment) -> dict[str, str]:
    checks = {}
    for key, kind in PATHS.items():
        value = _value(environment.values, key)
        if value is None:
            checks[key] = "missing key"
            continue
        path = Path(str(value))
        if kind == "exe":
            checks[key] = "ok" if path.is_file() and os.access(path, os.X_OK) else f"missing executable {path}"
        elif path.is_dir():
            checks[key] = "ok"
        elif kind == "created" and _writable_parent(path):
            checks[key] = "ok (created on first use)"
        else:
            checks[key] = f"missing directory {path}"
    return checks


def check_serving(environment: Environment) -> dict[str, str]:
    """Import the serving packages in serve_python and read its GPU fingerprint."""
    from .services import _serve_env

    python = str(environment["serve_python"])
    probe = ("import importlib, json; missing = [name for name in %r if importlib.util.find_spec(name) is None]; "
             "print(json.dumps(missing))" % (SERVE_PACKAGES,))
    completed = subprocess.run([python, "-c", "import importlib.util; " + probe], capture_output=True, text=True,
                               env=_serve_env(environment), timeout=300)
    missing = json.loads(completed.stdout.strip() or "null") if completed.returncode == 0 else None
    checks = {"serve_packages": "ok" if missing == [] else f"missing {missing or completed.stderr[-300:]}"}
    completed = subprocess.run([python, "-m", "trtmc_perf_serving", "platform"], capture_output=True, text=True,
                               env=_serve_env(environment), cwd=environment.path("repo"), timeout=300)
    if completed.returncode == 0:
        fingerprint = json.loads(completed.stdout.strip().splitlines()[-1])["fingerprint"]
        checks["gpu"] = f"ok ({fingerprint.get('gpu_arch')}, torch {fingerprint.get('packages', {}).get('torch')})"
    else:
        checks["gpu"] = f"unavailable: {completed.stderr.strip()[-300:]}"
    return checks


def problems(checks: Mapping[str, str]) -> list[str]:
    return [key for key, value in checks.items() if not value.startswith("ok")]
