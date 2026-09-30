# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Platform identity of a serving environment.

``fingerprint`` holds what changes reference numerics (GPU architecture, framework
and CUDA versions) and keys reference goldens; ``host_details`` is recorded for
traceability only.
"""

from __future__ import annotations

import importlib.metadata
import platform as _platform
import socket
from typing import Any

# Libraries whose kernels or defaults affect reference outputs; absent ones are skipped.
NUMERIC_PACKAGES = ("torch", "transformers", "diffusers", "timm", "nemo_toolkit", "accelerate")


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def fingerprint() -> dict[str, Any]:
    import torch

    value: dict[str, Any] = {"packages": {name: _version(name) for name in NUMERIC_PACKAGES if _version(name)},
                             "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        value["gpu_arch"] = f"sm{major}{minor}"
    else:
        value["gpu_arch"] = "cpu"
    return value


def host_details() -> dict[str, Any]:
    import torch

    details: dict[str, Any] = {"hostname": socket.gethostname(), "python": _platform.python_version(),
                               "machine": _platform.machine()}
    if torch.cuda.is_available():
        details["gpu_name"] = torch.cuda.get_device_name(0)
        try:
            import pynvml

            pynvml.nvmlInit()
            details["driver"] = pynvml.nvmlSystemGetDriverVersion()
        except Exception:  # noqa: BLE001 - driver version is informational
            pass
    return details


def apply_deterministic_numerics() -> dict[str, Any]:
    """Settings used when generating reference goldens: no TF32, deterministic kernels."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {"tf32": False, "cudnn_deterministic": True, "deterministic_algorithms": "warn_only",
            "float32_matmul_precision": "highest"}
