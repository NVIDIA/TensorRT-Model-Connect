# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict timing-cache replay for deterministic Bark engine builds."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import Any

from tensorrt_model_connect import trt_compat


_MODE_ENV = "TRTMC_BARK_TIMING_CACHE_MODE"
_PATH_ENV = "TRTMC_BARK_TIMING_CACHE_PATH"
_SHA256_ENV = "TRTMC_BARK_TIMING_CACHE_SHA256"
_BUILDER_INT_ENVS = (
    ("builder_optimization_level", "TRTMC_BUILDER_OPTIMIZATION_LEVEL"),
    ("max_num_tactics", "TRTMC_MAX_NUM_TACTICS"),
    ("avg_timing_iterations", "TRTMC_AVG_TIMING_ITERATIONS"),
)


@dataclass(frozen=True)
class _CacheState:
    mode: str
    path: Path
    cache: Any
    original_sha256: str | None


def _mode() -> str:
    mode = os.environ.get(_MODE_ENV, "off").strip().lower() or "off"
    if mode not in {"off", "record", "verified"}:
        raise ValueError(f"{_MODE_ENV} must be off, record, or verified, got {mode!r}")
    return mode


def _set_required_flag(config: Any, flag_name: str) -> None:
    builder_flag = getattr(trt_compat.get_trt(), "BuilderFlag", None)
    flag = getattr(builder_flag, flag_name, None)
    if flag is None or not hasattr(config, "set_flag"):
        raise RuntimeError(f"TensorRT does not support required builder flag {flag_name}")
    config.set_flag(flag)


def _apply_builder_int_envs(config: Any) -> None:
    for attribute, env_name in _BUILDER_INT_ENVS:
        value = os.environ.get(env_name)
        if value is None or not value.strip() or not hasattr(config, attribute):
            continue
        try:
            setattr(config, attribute, int(value))
        except ValueError as exc:
            raise ValueError(f"{env_name} must be an integer, got {value!r}") from exc


def _serialize(cache: Any, path: Path) -> bytes:
    if cache is None or not hasattr(cache, "serialize"):
        raise RuntimeError(f"failed to serialize Bark timing cache: {path}")
    payload = cache.serialize()
    if payload is None:
        raise RuntimeError(f"failed to serialize Bark timing cache: {path}")
    return bytes(payload)


def _expected_sha256(path: Path, payload: bytes) -> str:
    expected = os.environ.get(_SHA256_ENV, "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise RuntimeError(f"{_SHA256_ENV} must contain the verified cache SHA-256")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"verified Bark timing cache SHA-256 mismatch for {path}: "
            f"expected {expected}, got {actual}"
        )
    return actual


def _attach(config: Any, mode: str) -> _CacheState:
    path_text = os.environ.get(_PATH_ENV, "").strip()
    if not path_text:
        raise RuntimeError(f"{_PATH_ENV} is required in {mode} mode")
    path = Path(path_text)
    if mode == "verified" and not path.is_file():
        raise RuntimeError(f"verified Bark timing cache does not exist: {path}")
    if not hasattr(config, "create_timing_cache") or not hasattr(config, "set_timing_cache"):
        raise RuntimeError("TensorRT does not support an attached timing cache")

    try:
        payload = path.read_bytes() if path.is_file() else b""
    except OSError as exc:
        raise RuntimeError(f"failed to read Bark timing cache {path}: {exc}") from exc
    if mode == "verified" and not payload:
        raise RuntimeError(f"verified Bark timing cache is empty: {path}")
    original_sha256 = _expected_sha256(path, payload) if mode == "verified" else None

    for flag_name in ("EDITABLE_TIMING_CACHE", "DISABLE_COMPILATION_CACHE"):
        _set_required_flag(config, flag_name)
    if mode == "verified":
        _set_required_flag(config, "ERROR_ON_TIMING_CACHE_MISS")

    try:
        cache = config.create_timing_cache(payload)
        accepted = config.set_timing_cache(cache, False)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"failed to attach {mode} Bark timing cache {path}: {exc}") from exc
    if not accepted:
        raise RuntimeError(f"TensorRT rejected {mode} Bark timing cache {path}")
    return _CacheState(mode, path, cache, original_sha256)


def _active_cache(config: Any, state: _CacheState) -> Any:
    if hasattr(config, "get_timing_cache"):
        cache = config.get_timing_cache()
        if cache is not None:
            return cache
    return state.cache


def _verify_unchanged(config: Any, state: _CacheState) -> None:
    payload = _serialize(_active_cache(config, state), state.path)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != state.original_sha256:
        raise RuntimeError(
            f"verified Bark timing cache changed during build: {state.path}; "
            "a timing-cache miss or tactic update occurred"
        )


def _save_recording(config: Any, state: _CacheState) -> None:
    payload = _serialize(_active_cache(config, state), state.path)
    try:
        state.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = state.path.with_name(f".{state.path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, state.path)
    except OSError as exc:
        raise RuntimeError(f"failed to save Bark timing cache {state.path}: {exc}") from exc


def build_bark_serialized_network(builder: Any, network: Any, config: Any) -> Any:
    """Build with an optional record-only or verified Bark timing cache."""
    mode = _mode()
    if mode == "off":
        return builder.build_serialized_network(network, config)

    _apply_builder_int_envs(config)
    state = _attach(config, mode)
    raw_builder = trt_compat.unwrap(builder)
    plan = raw_builder.build_serialized_network(
        trt_compat.unwrap(network), trt_compat.unwrap(config)
    )
    if mode == "verified":
        _verify_unchanged(config, state)
    elif plan is not None:
        _save_recording(config, state)
    return plan
