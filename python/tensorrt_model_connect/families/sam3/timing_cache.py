# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict, model-owned tactic replay for accuracy-sensitive SAM3 engines."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterator, Literal, Mapping

from tensorrt_model_connect import trt_compat


_SCHEMA_VERSION = 3
_ENGINE_KINDS = frozenset({"text-encoder", "vision-encoder", "core"})
_GRAPH_CONTRACT_ALGORITHM = "sha256-source-graph-profile-v1"
_GRAPH_SOURCE_FILES: Mapping[str, tuple[str, ...]] = {
    "core": ("core_builder.py", "graph_ops.py"),
    "text-encoder": ("graph_ops.py", "text_encoder_builder.py"),
    "vision-encoder": ("graph_ops.py", "vision_encoder_builder.py"),
}
_CACHE_FILE_NAME = re.compile(r"[a-z0-9][a-z0-9.-]*\.cache")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class _RuntimeMetadata:
    gpu_name: str
    gpu_architecture: str
    tensorrt_version: str
    cuda_runtime_version: int
    cuda_driver_version: int

    @property
    def signature(self) -> tuple[str, str, str, int, int]:
        return (
            self.gpu_name,
            self.gpu_architecture,
            self.tensorrt_version,
            self.cuda_runtime_version,
            self.cuda_driver_version,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "cuda_driver_version": self.cuda_driver_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "gpu_architecture": self.gpu_architecture,
            "gpu_name": self.gpu_name,
            "tensorrt_version": self.tensorrt_version,
        }


# An auto-selected cache is deliberately narrower than TensorRT's own cache
# compatibility header. Tactics recorded on a different GPU or software stack
# must never become a silent default.
_BUILTIN_TARGETS: Mapping[tuple[str, str, str, int, int], str] = {
    (
        "NVIDIA L4",
        "sm89",
        "10.15.1.29",
        13010,
        13020,
    ): "l4-sm89-trt10.15.1.29-cuda13010-driver13020",
}
# Built-in replay is enabled only after an engine kind has independent L4
# qualification.  Vision recording/replay is supported programmatically, but
# remains an ordinary TensorRT build until a qualified payload is packaged.
_BUILTIN_ENGINE_KINDS: Mapping[str, frozenset[str]] = {
    "l4-sm89-trt10.15.1.29-cuda13010-driver13020": frozenset(
        {"core", "text-encoder"}
    ),
}
_BUILTIN_GRAPH_FINGERPRINTS: Mapping[str, Mapping[str, str]] = {
    "l4-sm89-trt10.15.1.29-cuda13010-driver13020": {
        "core": "9ab98f95e1be5ab592d492c5ab936690ad07d118c7142cfdb74aee663afdc997",
        "text-encoder": "6509eacf17797ae93c328e401f33e6974e2eaac0f471b88748fdd605697d54b5",
    },
}


class _GraphContractMismatch(RuntimeError):
    """The cache is well-formed but was recorded for another SAM3 graph."""


@dataclass(frozen=True)
class Sam3TimingCachePolicy:
    """Programmatic SAM3 timing-cache selection; the default is automatic."""

    mode: Literal["auto", "off", "verified"] = "auto"
    directory: Path | None = None

    def validate(self) -> None:
        if self.mode not in {"auto", "off", "verified"}:
            raise ValueError(f"unknown SAM3 timing-cache mode: {self.mode!r}")
        if self.mode == "verified" and self.directory is None:
            raise ValueError("verified SAM3 timing-cache mode requires a directory")
        if self.mode != "verified" and self.directory is not None:
            raise ValueError(f"SAM3 timing-cache mode {self.mode!r} does not accept a directory")


_POLICY: ContextVar[Sam3TimingCachePolicy] = ContextVar(
    "sam3_timing_cache_policy", default=Sam3TimingCachePolicy()
)


@contextmanager
def use_sam3_timing_cache(policy: Sam3TimingCachePolicy) -> Iterator[None]:
    """Temporarily override cache selection without process environment state."""

    policy.validate()
    token = _POLICY.set(policy)
    try:
        yield
    finally:
        _POLICY.reset(token)


def _graph_contract_fingerprint(engine_kind: str, graph_profile: Mapping[str, Any]) -> str:
    """Bind a cache to exact graph source bytes and effective build parameters."""

    if engine_kind not in _ENGINE_KINDS:
        raise ValueError(f"unsupported SAM3 timing-cache engine kind: {engine_kind!r}")
    if not isinstance(graph_profile, Mapping) or not graph_profile:
        raise ValueError(f"SAM3 {engine_kind} graph profile must be a nonempty mapping")
    source_root = Path(__file__).resolve().parent
    sources: dict[str, str] = {}
    try:
        for file_name in _GRAPH_SOURCE_FILES[engine_kind]:
            sources[file_name] = hashlib.sha256((source_root / file_name).read_bytes()).hexdigest()
        payload = json.dumps(
            {
                "algorithm": _GRAPH_CONTRACT_ALGORITHM,
                "engine_kind": engine_kind,
                "graph_profile": graph_profile,
                "sources": sources,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(f"could not fingerprint the SAM3 {engine_kind} graph: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _runtime_metadata() -> _RuntimeMetadata:
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        try:
            from cuda import cudart  # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeError("SAM3 timing-cache target validation requires CUDA Python") from exc

    success = getattr(getattr(cudart, "cudaError_t", None), "cudaSuccess", 0)
    try:
        status, device = cudart.cudaGetDevice()
        if status not in (success, 0):
            raise RuntimeError(f"cudaGetDevice failed with status {status}")
        status, properties = cudart.cudaGetDeviceProperties(int(device))
        if status not in (success, 0):
            raise RuntimeError(f"cudaGetDeviceProperties failed with status {status}")
        status, cuda_runtime_version = cudart.cudaRuntimeGetVersion()
        if status not in (success, 0):
            raise RuntimeError(f"cudaRuntimeGetVersion failed with status {status}")
        status, cuda_driver_version = cudart.cudaDriverGetVersion()
        if status not in (success, 0):
            raise RuntimeError(f"cudaDriverGetVersion failed with status {status}")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"could not query the active CUDA device for SAM3: {exc}") from exc

    gpu_name = properties.name
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode("utf-8", errors="replace").rstrip("\x00")
    gpu_name = str(gpu_name).strip()
    major = int(properties.major)
    minor = int(properties.minor)
    if (
        not gpu_name
        or major <= 0
        or minor < 0
        or int(cuda_runtime_version) <= 0
        or int(cuda_driver_version) <= 0
    ):
        raise RuntimeError("CUDA returned invalid target metadata for the SAM3 timing cache")
    return _RuntimeMetadata(
        gpu_name=gpu_name,
        gpu_architecture=f"sm{major}{minor}",
        tensorrt_version=trt_compat.tensorrt_version(),
        cuda_runtime_version=int(cuda_runtime_version),
        cuda_driver_version=int(cuda_driver_version),
    )


def _validate_graph_contract(
    value: Any,
    *,
    engine_kind: str,
    expected_fingerprint: str,
    expected_inventory: frozenset[str],
    source: str,
) -> None:
    if not isinstance(value, dict) or set(value) != {"algorithm", "engines"}:
        raise RuntimeError(f"SAM3 timing-cache graph contract is invalid: {source}")
    if value["algorithm"] != _GRAPH_CONTRACT_ALGORITHM:
        raise RuntimeError(f"SAM3 timing-cache graph algorithm mismatch: {source}")
    engines = value["engines"]
    if not isinstance(engines, dict) or set(engines) != expected_inventory:
        raise RuntimeError(f"SAM3 timing-cache graph inventory is invalid: {source}")
    if any(not isinstance(item, str) or _SHA256.fullmatch(item) is None for item in engines.values()):
        raise RuntimeError(f"SAM3 timing-cache graph fingerprint is invalid: {source}")
    if engines[engine_kind] != expected_fingerprint:
        raise _GraphContractMismatch(
            f"SAM3 {engine_kind} graph contract mismatch: "
            f"cache={engines[engine_kind]!r}, runtime={expected_fingerprint!r}"
        )


def _load_manifest(
    directory: Path,
    runtime: _RuntimeMetadata,
    *,
    engine_kind: str,
    graph_fingerprint: str,
) -> dict[str, Any]:
    path = directory / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"SAM3 timing-cache manifest is unreadable: {path}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"SAM3 timing-cache manifest must be an object: {path}")
    required = {"engines", "graph_contract", "schema_version", "target"}
    if set(manifest) != required:
        raise RuntimeError(f"SAM3 timing-cache manifest fields are invalid: {path}")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != _SCHEMA_VERSION:
        raise RuntimeError(
            f"SAM3 timing-cache schema mismatch: cache={manifest['schema_version']!r}, "
            f"runtime={_SCHEMA_VERSION!r}"
        )
    if manifest["target"] != runtime.as_dict():
        raise RuntimeError(
            f"SAM3 timing-cache target mismatch: "
            f"cache={manifest['target']!r}, runtime={runtime.as_dict()!r}"
        )
    engines = manifest["engines"]
    if not isinstance(engines, dict) or not engines or not set(engines) <= _ENGINE_KINDS:
        raise RuntimeError(f"SAM3 timing-cache engine inventory is invalid: {path}")
    inventory = frozenset(engines)
    if engine_kind not in inventory:
        raise RuntimeError(f"SAM3 timing-cache engine {engine_kind!r} is missing: {path}")
    for kind, entry in engines.items():
        if not isinstance(entry, dict) or set(entry) != {
            "file",
            "sha256",
            "tactic_count",
            "tactic_sha256",
        }:
            raise RuntimeError(f"SAM3 timing-cache entry is invalid for {kind}")
        if (
            not isinstance(entry["file"], str)
            or _CACHE_FILE_NAME.fullmatch(entry["file"]) is None
        ):
            raise RuntimeError(f"SAM3 timing-cache filename is invalid for {kind}")
        _validate_entry_fingerprint(entry, kind)
    _validate_graph_contract(
        manifest["graph_contract"],
        engine_kind=engine_kind,
        expected_fingerprint=graph_fingerprint,
        expected_inventory=inventory,
        source=str(path),
    )
    return manifest


def _validate_entry_fingerprint(entry: Any, engine_kind: str) -> None:
    required = {"sha256", "tactic_count", "tactic_sha256"}
    if not isinstance(entry, dict) or not required <= set(entry):
        raise RuntimeError(f"SAM3 timing-cache entry is invalid for {engine_kind}")
    if not isinstance(entry["sha256"], str) or _SHA256.fullmatch(entry["sha256"]) is None:
        raise RuntimeError(f"SAM3 timing-cache SHA-256 is invalid for {engine_kind}")
    if (
        type(entry["tactic_count"]) is not int
        or entry["tactic_count"] <= 0
        or not isinstance(entry["tactic_sha256"], str)
        or _SHA256.fullmatch(entry["tactic_sha256"]) is None
    ):
        raise RuntimeError(f"SAM3 timing-cache tactic fingerprint is invalid for {engine_kind}")


def _load_payload(directory: Path, entry: Any, engine_kind: str) -> bytes:
    required = {"file", "sha256", "tactic_count", "tactic_sha256"}
    if not isinstance(entry, dict) or set(entry) != required:
        raise RuntimeError(f"SAM3 timing-cache entry is invalid for {engine_kind}")
    file_name = entry["file"]
    if not isinstance(file_name, str) or _CACHE_FILE_NAME.fullmatch(file_name) is None:
        raise RuntimeError(f"SAM3 timing-cache filename is invalid for {engine_kind}")
    _validate_entry_fingerprint(entry, engine_kind)
    cache_path = directory / file_name
    try:
        payload = cache_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"SAM3 timing cache is unreadable: {cache_path}") from exc
    if not payload:
        raise RuntimeError(f"SAM3 timing cache is empty: {cache_path}")
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if entry["sha256"] != actual_sha256:
        raise RuntimeError(
            f"SAM3 timing-cache SHA-256 mismatch for {cache_path}: "
            f"expected {entry['sha256']!r}, got {actual_sha256!r}"
        )
    return payload


def _load_builtin_payload(
    contract_id: str,
    runtime: _RuntimeMetadata,
    engine_kind: str,
    graph_fingerprint: str,
) -> tuple[dict[str, Any], bytes]:
    expected_inventory = _BUILTIN_ENGINE_KINDS.get(contract_id)
    if expected_inventory is None or not expected_inventory:
        raise RuntimeError(f"packaged SAM3 timing-cache inventory is missing: {contract_id}")
    if engine_kind not in expected_inventory:
        raise RuntimeError(
            f"packaged SAM3 timing cache is not qualified for {engine_kind}: {contract_id}"
        )
    try:
        from .timing_cache_data import CONTRACTS

        contract = CONTRACTS[contract_id]
    except (ImportError, KeyError) as exc:
        raise RuntimeError(f"packaged SAM3 timing-cache contract is missing: {contract_id}") from exc
    if not isinstance(contract, dict) or set(contract) != {"manifest", "payloads"}:
        raise RuntimeError(f"packaged SAM3 timing-cache contract is invalid: {contract_id}")
    manifest = contract["manifest"]
    if not isinstance(manifest, dict):
        raise RuntimeError(f"packaged SAM3 timing-cache manifest is invalid: {contract_id}")
    required = {"engines", "graph_contract", "schema_version", "target"}
    if set(manifest) != required:
        raise RuntimeError(f"packaged SAM3 timing-cache manifest fields are invalid: {contract_id}")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != _SCHEMA_VERSION:
        raise RuntimeError(f"packaged SAM3 timing-cache schema mismatch: {contract_id}")
    _validate_graph_contract(
        manifest["graph_contract"],
        engine_kind=engine_kind,
        expected_fingerprint=graph_fingerprint,
        expected_inventory=expected_inventory,
        source=contract_id,
    )
    if manifest["target"] != runtime.as_dict():
        raise RuntimeError(f"packaged SAM3 timing-cache target mismatch: {contract_id}")
    engines = manifest["engines"]
    payloads = contract["payloads"]
    if (
        not isinstance(engines, dict)
        or set(engines) != expected_inventory
        or not isinstance(payloads, dict)
        or set(payloads) != expected_inventory
    ):
        raise RuntimeError(f"packaged SAM3 timing-cache inventory is invalid: {contract_id}")
    entry = engines[engine_kind]
    if not isinstance(entry, dict) or set(entry) != {
        "sha256",
        "tactic_count",
        "tactic_sha256",
    }:
        raise RuntimeError(f"packaged SAM3 timing-cache entry is invalid for {engine_kind}")
    _validate_entry_fingerprint(entry, engine_kind)
    encoded = payloads[engine_kind]
    if not isinstance(encoded, str) or not encoded:
        raise RuntimeError(f"packaged SAM3 timing-cache payload is invalid for {engine_kind}")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"packaged SAM3 timing-cache payload is corrupt for {engine_kind}") from exc
    if not payload:
        raise RuntimeError(f"packaged SAM3 timing cache is empty for {engine_kind}")
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if entry["sha256"] != actual_sha256:
        raise RuntimeError(
            f"packaged SAM3 timing-cache SHA-256 mismatch for {engine_kind}: "
            f"expected {entry['sha256']!r}, got {actual_sha256!r}"
        )
    return entry, payload


def _set_required_flag(config: Any, name: str) -> None:
    flag = getattr(getattr(trt_compat.get_trt(), "BuilderFlag", None), name, None)
    if flag is None or not hasattr(config, "set_flag"):
        raise RuntimeError(f"TensorRT does not support required builder flag {name}")
    config.set_flag(flag)


def _query_tactics(cache: Any) -> dict[str, int]:
    if not hasattr(cache, "queryKeys") or not hasattr(cache, "query"):
        raise RuntimeError("TensorRT does not support editable timing-cache queries")
    tactics: dict[str, int] = {}
    try:
        keys = list(cache.queryKeys())
        for key in keys:
            key_text = str(key)
            value = cache.query(key)
            tactic_hash = int(value.tacticHash)
            timing_msec = float(value.timingMSec)
            if (
                tactic_hash == (1 << 64) - 1
                or not math.isfinite(timing_msec)
                or timing_msec < 0
            ):
                raise ValueError(f"invalid tactic value for {key_text}")
            if key_text in tactics:
                raise ValueError(f"duplicate timing-cache key {key_text}")
            tactics[key_text] = tactic_hash
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"failed to inspect the verified SAM3 timing cache: {exc}") from exc
    if not tactics:
        raise RuntimeError("verified SAM3 timing cache has no tactic entries")
    return tactics


def _tactic_sha256(tactics: Mapping[str, int]) -> str:
    payload = json.dumps(tactics, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _replay_tactics(cache: Any) -> dict[str, int]:
    if not hasattr(cache, "update"):
        raise RuntimeError("TensorRT does not support editable timing-cache replay")
    expected = _query_tactics(cache)
    try:
        for key in cache.queryKeys():
            if not cache.update(key, cache.query(key)):
                raise RuntimeError(f"TensorRT rejected cached SAM3 tactic {key}")
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"failed to replay verified SAM3 tactics: {exc}") from exc
    return expected


def _validate_tactic_fingerprint(entry: Mapping[str, Any], tactics: Mapping[str, int]) -> None:
    actual_count = len(tactics)
    actual_sha256 = _tactic_sha256(tactics)
    if entry["tactic_count"] != actual_count or entry["tactic_sha256"] != actual_sha256:
        raise RuntimeError(
            "SAM3 timing-cache tactic fingerprint mismatch: "
            f"count cache={entry['tactic_count']!r}, actual={actual_count!r}; "
            f"sha256 cache={entry['tactic_sha256']!r}, actual={actual_sha256!r}"
        )


def _tactic_delta(expected: Mapping[str, int], actual: Mapping[str, int]) -> str:
    expected_keys = set(expected)
    actual_keys = set(actual)
    added = sorted(actual_keys - expected_keys)
    removed = sorted(expected_keys - actual_keys)
    changed = sorted(key for key in expected_keys & actual_keys if expected[key] != actual[key])
    return f"added={len(added)}, removed={len(removed)}, changed={len(changed)}"


def build_sam3_serialized_network(
    builder: Any,
    network: Any,
    config: Any,
    *,
    engine_kind: str,
    graph_profile: Mapping[str, Any],
) -> Any:
    """Build one SAM3 engine, strictly replaying its cache on a pinned target."""

    if engine_kind not in _ENGINE_KINDS:
        raise ValueError(f"unsupported SAM3 timing-cache engine kind: {engine_kind!r}")
    policy = _POLICY.get()
    policy.validate()
    if policy.mode == "off":
        return builder.build_serialized_network(network, config)
    runtime = _runtime_metadata()
    if policy.mode == "auto":
        contract_id = _BUILTIN_TARGETS.get(runtime.signature)
        if contract_id is None:
            return builder.build_serialized_network(network, config)
        builtin_engine_kinds = _BUILTIN_ENGINE_KINDS.get(contract_id)
        if (
            builtin_engine_kinds is None
            or not builtin_engine_kinds
            or not builtin_engine_kinds <= _ENGINE_KINDS
        ):
            raise RuntimeError(f"packaged SAM3 engine inventory is missing: {contract_id}")
        if engine_kind not in builtin_engine_kinds:
            return builder.build_serialized_network(network, config)
        graph_fingerprint = _graph_contract_fingerprint(engine_kind, graph_profile)
        canonical_graphs = _BUILTIN_GRAPH_FINGERPRINTS.get(contract_id)
        if canonical_graphs is None or set(canonical_graphs) != builtin_engine_kinds:
            raise RuntimeError(f"packaged SAM3 graph inventory is missing: {contract_id}")
        if graph_fingerprint != canonical_graphs[engine_kind]:
            return builder.build_serialized_network(network, config)
    else:
        graph_fingerprint = _graph_contract_fingerprint(engine_kind, graph_profile)
    if policy.mode == "verified":
        assert policy.directory is not None
        manifest = _load_manifest(
            policy.directory,
            runtime,
            engine_kind=engine_kind,
            graph_fingerprint=graph_fingerprint,
        )
        entry = manifest["engines"][engine_kind]
        payload = _load_payload(policy.directory, entry, engine_kind)
    else:
        assert policy.mode == "auto"
        # For a supported exact target, missing packaged data is an
        # installation error, not permission to retime nondeterministically.
        entry, payload = _load_builtin_payload(
            contract_id,
            runtime,
            engine_kind,
            graph_fingerprint,
        )
    raw_builder = trt_compat.unwrap(builder)
    raw_network = trt_compat.unwrap(network)
    raw_config = trt_compat.unwrap(config)
    for flag_name in (
        "EDITABLE_TIMING_CACHE",
        "DISABLE_COMPILATION_CACHE",
        "ERROR_ON_TIMING_CACHE_MISS",
    ):
        _set_required_flag(raw_config, flag_name)

    try:
        cache = raw_config.create_timing_cache(payload)
        expected = _replay_tactics(cache)
        _validate_tactic_fingerprint(entry, expected)
        # Preserve TensorRT's native cache-header compatibility check in
        # addition to the explicit model-owned target and graph contracts.
        accepted = raw_config.set_timing_cache(cache, False)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"failed to attach verified SAM3 {engine_kind} timing cache: {exc}") from exc
    if not accepted:
        raise RuntimeError(f"TensorRT rejected verified SAM3 {engine_kind} timing cache")

    plan = raw_builder.build_serialized_network(raw_network, raw_config)
    active_cache = raw_config.get_timing_cache() if hasattr(raw_config, "get_timing_cache") else cache
    actual = _query_tactics(active_cache)
    if actual != expected:
        raise RuntimeError(
            f"verified SAM3 {engine_kind} tactic selection changed during build: "
            f"{_tactic_delta(expected, actual)}"
        )
    return plan
