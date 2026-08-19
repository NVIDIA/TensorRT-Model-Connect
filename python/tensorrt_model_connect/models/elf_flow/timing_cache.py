# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict, model-owned TensorRT timing cache for ELF replay parity."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from tensorrt_model_connect import trt_compat


_CACHE_PATH_ENV = "TRTMC_ELF_TIMING_CACHE_PATH"
_METADATA_PATH_ENV = "TRTMC_ELF_TIMING_CACHE_METADATA_PATH"
_GENERATE_ENV = "TRTMC_ELF_TIMING_CACHE_GENERATE"
_GENERIC_CACHE_ENVS = (
    "TRTMC_TRT_TIMING_CACHE_PATH",
    "TRTMC_TRT_TIMING_CACHE_DIR",
)


@dataclass(frozen=True)
class TimingCacheState:
    cache: Any
    cache_path: Path
    metadata_path: Path
    generate: bool


def _runtime_gpu_metadata() -> tuple[str, str]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - CI profiles provide torch
        raise RuntimeError("ELF timing-cache validation requires PyTorch GPU metadata") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("ELF timing cache requires an available CUDA device")
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    return torch.cuda.get_device_name(device), f"{major}.{minor}"


def _optimization_level() -> int:
    value = os.environ.get("TRTMC_BUILDER_OPTIMIZATION_LEVEL", "").strip()
    if not value:
        raise RuntimeError("TRTMC_BUILDER_OPTIMIZATION_LEVEL is required for an ELF timing cache")
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError("TRTMC_BUILDER_OPTIMIZATION_LEVEL must be an integer") from exc


def _generation_requested() -> bool:
    value = os.environ.get(_GENERATE_ENV, "").strip()
    if value in {"", "0"}:
        return False
    if value == "1":
        return True
    raise RuntimeError(f"{_GENERATE_ENV} must be 0 or 1")


def _configured_paths() -> tuple[Path, Path] | None:
    cache_value = os.environ.get(_CACHE_PATH_ENV, "").strip()
    if not cache_value:
        if _generation_requested():
            raise RuntimeError(f"{_CACHE_PATH_ENV} is required for cache generation")
        return None
    metadata_value = os.environ.get(_METADATA_PATH_ENV, "").strip()
    if not metadata_value:
        raise RuntimeError(f"{_METADATA_PATH_ENV} is required")
    enabled_generic = [name for name in _GENERIC_CACHE_ENVS if os.environ.get(name, "").strip()]
    if enabled_generic:
        raise RuntimeError(
            "ELF model timing cache cannot be combined with generic cache settings: "
            + ", ".join(enabled_generic)
        )
    return Path(cache_value), Path(metadata_value)


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"ELF timing-cache metadata is unreadable: {path}") from exc
    if not isinstance(metadata, dict):
        raise RuntimeError(f"ELF timing-cache metadata must be an object: {path}")
    return metadata


def _validated_payload(cache_path: Path, metadata_path: Path) -> bytes:
    metadata = _load_metadata(metadata_path)
    required = {
        "builder_optimization_level",
        "compute_capability",
        "gpu",
        "path",
        "schema_version",
        "sha256",
        "tensorrt_version",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise RuntimeError(f"ELF timing-cache metadata is missing fields: {missing}")
    try:
        payload = cache_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"ELF timing cache is unreadable: {cache_path}") from exc
    if not payload:
        raise RuntimeError(f"ELF timing cache is empty: {cache_path}")

    checks = {
        "schema_version": (metadata["schema_version"], 1),
        "path": (metadata["path"], cache_path.name),
        "sha256": (metadata["sha256"], hashlib.sha256(payload).hexdigest()),
        "tensorrt_version": (
            metadata["tensorrt_version"],
            trt_compat.tensorrt_version(),
        ),
        "builder_optimization_level": (
            metadata["builder_optimization_level"],
            _optimization_level(),
        ),
    }
    gpu, compute_capability = _runtime_gpu_metadata()
    checks["gpu"] = (metadata["gpu"], gpu)
    checks["compute_capability"] = (
        metadata["compute_capability"],
        compute_capability,
    )
    mismatches = [
        f"{name}: cache={actual!r}, runtime={expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if mismatches:
        raise RuntimeError("ELF timing-cache metadata mismatch: " + "; ".join(mismatches))
    return payload


def attach_model_timing_cache(builder_config: Any, trt_module: Any) -> TimingCacheState | None:
    paths = _configured_paths()
    if paths is None:
        return None
    cache_path, metadata_path = paths
    generate = _generation_requested()
    payload = b"" if generate else _validated_payload(cache_path, metadata_path)
    cache = builder_config.create_timing_cache(payload)
    if not builder_config.set_timing_cache(cache, False):
        raise RuntimeError(f"ELF timing cache is incompatible: {cache_path}")
    if not generate:
        flag = getattr(trt_module.BuilderFlag, "ERROR_ON_TIMING_CACHE_MISS", None)
        if flag is None:
            raise RuntimeError("TensorRT does not support strict timing-cache misses")
        builder_config.set_flag(flag)
    return TimingCacheState(cache, cache_path, metadata_path, generate)


def persist_generated_model_timing_cache(
    builder_config: Any,
    state: TimingCacheState | None,
) -> None:
    if state is None or not state.generate:
        return
    cache = (
        builder_config.get_timing_cache()
        if hasattr(builder_config, "get_timing_cache")
        else state.cache
    )
    if cache is None or not hasattr(cache, "serialize"):
        raise RuntimeError("TensorRT did not expose the generated ELF timing cache")
    payload = bytes(cache.serialize())
    if not payload:
        raise RuntimeError("TensorRT generated an empty ELF timing cache")

    state.cache_path.parent.mkdir(parents=True, exist_ok=True)
    state.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state.cache_path.with_name(f".{state.cache_path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, state.cache_path)

    gpu, compute_capability = _runtime_gpu_metadata()
    metadata = {
        "builder_optimization_level": _optimization_level(),
        "compute_capability": compute_capability,
        "gpu": gpu,
        "path": state.cache_path.name,
        "schema_version": 1,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "tensorrt_version": trt_compat.tensorrt_version(),
    }
    temporary = state.metadata_path.with_name(f".{state.metadata_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, state.metadata_path)
