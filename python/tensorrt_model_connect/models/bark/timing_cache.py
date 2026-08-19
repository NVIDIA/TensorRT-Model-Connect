# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict timing-cache replay for deterministic Bark engine builds."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
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
    tactic_hashes: dict[str, int] | None


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


def _query_tactic_hashes(cache: Any, path: Path) -> dict[str, int]:
    for method in ("queryKeys", "query"):
        if not hasattr(cache, method):
            raise RuntimeError(f"TensorRT does not support verified Bark tactic queries ({method})")
    try:
        keys = list(cache.queryKeys())
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"failed to query verified Bark timing cache {path}: {exc}") from exc
    if not keys:
        raise RuntimeError(f"verified Bark timing cache has no tactic entries: {path}")

    tactics: dict[str, int] = {}
    try:
        for key in keys:
            key_text = str(key)
            value = cache.query(key)
            tactic_hash = int(value.tacticHash)
            timing_msec = float(value.timingMSec)
            if tactic_hash == (1 << 64) - 1 or not math.isfinite(timing_msec) or timing_msec < 0:
                raise ValueError(f"invalid tactic value for {key_text}")
            if key_text in tactics:
                raise ValueError(f"duplicate timing-cache key {key_text}")
            tactics[key_text] = tactic_hash
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"failed to inspect verified Bark timing cache {path}: {exc}") from exc
    return tactics


def _replay_tactics(cache: Any, path: Path) -> dict[str, int]:
    if not hasattr(cache, "update"):
        raise RuntimeError("TensorRT does not support verified Bark tactic replay (update)")
    expected = _query_tactic_hashes(cache, path)
    try:
        for key in cache.queryKeys():
            if not cache.update(key, cache.query(key)):
                raise RuntimeError(f"TensorRT rejected tactic {key}")
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"failed to replay verified Bark tactics from {path}: {exc}") from exc
    return expected


def _tactic_delta(expected: dict[str, int], actual: dict[str, int]) -> str:
    added = sorted(actual.keys() - expected.keys())
    removed = sorted(expected.keys() - actual.keys())
    changed = sorted(key for key in expected.keys() & actual.keys() if expected[key] != actual[key])
    samples = [f"added {key}={actual[key]}" for key in added[:2]]
    samples.extend(f"removed {key}={expected[key]}" for key in removed[:2])
    samples.extend(f"changed {key}={expected[key]}->{actual[key]}" for key in changed[:2])
    detail = "; ".join(samples) if samples else "no key-level difference found"
    return f"added={len(added)}, removed={len(removed)}, changed={len(changed)}; {detail}"


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
    if mode == "verified":
        _expected_sha256(path, payload)

    for flag_name in ("EDITABLE_TIMING_CACHE", "DISABLE_COMPILATION_CACHE"):
        _set_required_flag(config, flag_name)
    if mode == "verified":
        _set_required_flag(config, "ERROR_ON_TIMING_CACHE_MISS")

    try:
        cache = config.create_timing_cache(payload)
        tactic_hashes = _replay_tactics(cache, path) if mode == "verified" else None
        accepted = config.set_timing_cache(cache, mode == "verified")
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"failed to attach {mode} Bark timing cache {path}: {exc}") from exc
    if not accepted:
        raise RuntimeError(f"TensorRT rejected {mode} Bark timing cache {path}")
    return _CacheState(mode, path, cache, tactic_hashes)


def _active_cache(config: Any, state: _CacheState) -> Any:
    if hasattr(config, "get_timing_cache"):
        cache = config.get_timing_cache()
        if cache is not None:
            return cache
    return state.cache


def _verify_tactics_unchanged(config: Any, state: _CacheState) -> None:
    if state.tactic_hashes is None:
        raise RuntimeError("verified Bark timing cache is missing its tactic fingerprint")
    actual = _query_tactic_hashes(_active_cache(config, state), state.path)
    if actual != state.tactic_hashes:
        raise RuntimeError(
            f"verified Bark tactic selection changed during build: {state.path}; "
            f"{_tactic_delta(state.tactic_hashes, actual)}"
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
        _verify_tactics_unchanged(config, state)
    elif plan is not None:
        _save_recording(config, state)
    return plan
