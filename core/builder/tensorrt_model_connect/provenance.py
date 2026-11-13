# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort, model-agnostic identity of the inputs that produced a bundle."""

from __future__ import annotations

import importlib.metadata
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

CORE_SECTION_PREFIX = "trtmc."
PROVENANCE_SECTION = "trtmc.provenance.json"
PROVENANCE_VERSION = 1

_DISTRIBUTION = "tensorrt-model-connect"
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
# Shared request fields that change the serialized engines. Family-specific
# inputs stay in family-owned sections.
_BUILD_FIELDS = (
    "precision",
    "max_sequence_length",
    "max_batch_size",
    "image_height",
    "image_width",
    "video_num_frames",
    "tensor_parallel_size",
    "context_parallel_size",
    "quantization",
    "fp32_layers",
    "dynamic_kv_cache",
    "mtp_seq_len",
)


def collect(request: Any) -> dict[str, Any]:
    """Describe a build request without paths, host names, or timestamps.

    Every field is best effort: an unknown value is ``None`` and never fails
    the build.
    """

    return {
        "version": PROVENANCE_VERSION,
        "producer": _producer(),
        "checkpoint": checkpoint_identity(getattr(request, "model_dir", None)),
        "build": _build_inputs(request),
        "toolchain": _toolchain(),
    }


def checkpoint_identity(model_dir: object) -> dict[str, str] | None:
    """Return the Hugging Face repository and commit of a cache snapshot."""

    if model_dir is None:
        return None
    try:
        path = Path(model_dir).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    repository = path.parent.parent.name
    if (
        path.parent.name != "snapshots"
        or _COMMIT.fullmatch(path.name) is None
        or not repository.startswith("models--")
    ):
        return None
    return {
        "hf_id": repository.removeprefix("models--").replace("--", "/"),
        "revision": path.name,
    }


def _build_inputs(request: Any) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for name in _BUILD_FIELDS:
        if not hasattr(request, name):
            continue
        value = getattr(request, name)
        if isinstance(value, (tuple, list)):
            value = list(value)
            if all(_is_scalar(item) for item in value):
                inputs[name] = value
        elif _is_scalar(value):
            inputs[name] = value
    # Some family requests pass the transform separately; omit what is unknown.
    if hasattr(request, "graph_transform"):
        inputs["graph_transform"] = request.graph_transform is not None
    return inputs


def _is_scalar(value: object) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _producer() -> dict[str, Any]:
    try:
        version = importlib.metadata.version(_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        version = None
    revision, dirty = source_state(Path(__file__).resolve().parent)
    return {
        "name": _DISTRIBUTION,
        "version": version,
        "source_revision": revision,
        "source_dirty": dirty,
    }


def source_state(package_dir: Path) -> tuple[str | None, bool | None]:
    """Return the checkout commit and tracked-file dirtiness of this package.

    An installed package inside an unrelated Git work tree, such as a virtual
    environment within another checkout, reports no revision.
    """

    toplevel_and_head = _git(package_dir, "rev-parse", "--show-toplevel", "HEAD")
    if toplevel_and_head is None:
        return None, None
    lines = toplevel_and_head.splitlines()
    if len(lines) != 2 or _COMMIT.fullmatch(lines[1]) is None:
        return None, None
    toplevel, revision = lines
    try:
        owned = (Path(toplevel) / "core/builder/tensorrt_model_connect").resolve() == package_dir
    except (OSError, RuntimeError):
        owned = False
    if not owned:
        return None, None
    status = _git(Path(toplevel), "status", "--porcelain", "--untracked-files=no")
    return revision, None if status is None else bool(status.strip())


def _git(directory: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(directory), *arguments],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _toolchain() -> dict[str, Any]:
    # A family that never imported TensorRT serialized no engine, so there is
    # no builder identity to record and no reason to initialize CUDA here.
    tensorrt = sys.modules.get("tensorrt")
    version = getattr(tensorrt, "__version__", None)
    return {
        "tensorrt_version": version if isinstance(version, str) else None,
        "sm": _device_sm() if tensorrt is not None else None,
        "cpu_arch": platform.machine() or None,
    }


def _device_sm() -> int | None:
    try:
        from cuda.bindings import runtime

        status, device = runtime.cudaGetDevice()
        if int(status) != 0:
            return None
        status, properties = runtime.cudaGetDeviceProperties(device)
        if int(status) != 0:
            return None
        return int(properties.major) * 10 + int(properties.minor)
    except Exception:  # noqa: BLE001 - provenance must never fail a build.
        return None
