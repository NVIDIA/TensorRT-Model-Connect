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
         "aiperf": "exe", "bundle_root": "created", "hf_datasets_cache": "created",
         "reference_env_root": "created"}
# The TRTMC runtime (torch, transformers) and apps/perf_serving/requirements.txt: timm and diffusers serve the generic
# timm/ classifier and image-generation references, which run without a family environment.
SERVE_PACKAGES = ("torch", "transformers", "accelerate", "fastapi", "uvicorn", "multipart", "soundfile", "PIL", "numpy",
                  "timm", "diffusers")
# Imports each package named on its command line (an installed package can still fail to import: a missing or
# mismatched dependency) and prints, as its last line, the JSON list of those that fail, each with its error.
IMPORT_PROBE = """\
import importlib, json, sys
failed = []
for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
    except Exception as error:  # an ImportError, or the OSError of a missing shared library
        failed.append(f"{name} ({type(error).__name__}: {error})"[:300])
print(json.dumps(failed))
"""
# One cuDNN convolution: an interpreter that mixes cuDNN versions imports fine but fails here, as would every
# native reference with a convolution (vision, audio).
CONVOLUTION_PROBE = ("import torch; torch.nn.functional.conv2d(torch.ones(1, 3, 8, 8, device='cuda'), "
                     "torch.ones(4, 3, 3, 3, device='cuda')); print(torch.backends.cudnn.version())")


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


def missing_serve_packages(environment: Environment) -> list[str] | str:
    """The serving packages serve_python cannot import, each with its error ([] when it imports them all), or why
    the probe itself failed (never empty, so a caller cannot read a failed probe as success)."""
    from .services import _serve_env

    completed = subprocess.run([str(environment["serve_python"]), "-c", IMPORT_PROBE, *SERVE_PACKAGES],
                               capture_output=True, text=True, env=_serve_env(environment), timeout=300)
    lines = completed.stdout.strip().splitlines()
    try:
        missing = json.loads(lines[-1]) if completed.returncode == 0 and lines else None
    except ValueError:
        missing = None
    if isinstance(missing, list):
        return missing
    return completed.stderr.strip()[-300:] or f"the import probe failed (exit status {completed.returncode})"


def check_serving(environment: Environment) -> dict[str, str]:
    """Import the serving packages in serve_python, read its GPU fingerprint, and run one cuDNN convolution."""
    from .services import _serve_env

    python = str(environment["serve_python"])
    missing = missing_serve_packages(environment)
    checks = {"serve_packages": "ok" if missing == [] else f"missing {missing}"}
    completed = subprocess.run([python, "-m", "trtmc_perf_serving", "platform"], capture_output=True, text=True,
                               env=_serve_env(environment), cwd=environment.path("repo"), timeout=300)
    if completed.returncode == 0:
        fingerprint = json.loads(completed.stdout.strip().splitlines()[-1])["fingerprint"]
        checks["gpu"] = f"ok ({fingerprint.get('gpu_arch')}, torch {fingerprint.get('packages', {}).get('torch')})"
    else:
        checks["gpu"] = f"unavailable: {completed.stderr.strip()[-300:]}"
        return checks
    completed = subprocess.run([python, "-c", CONVOLUTION_PROBE], capture_output=True, text=True,
                               env=_serve_env(environment), timeout=300)
    checks["cudnn"] = (f"ok (cuDNN {(completed.stdout.split() or ['?'])[-1]})" if completed.returncode == 0 else
                       "convolution failed (Jetson Thor: serve_env in config/environments/example.yaml): "
                       + completed.stderr.strip()[-300:])
    return checks


def problems(checks: Mapping[str, str]) -> list[str]:
    return [key for key, value in checks.items() if not value.startswith("ok")]
