# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact qualification and bundle contract for runtime-owned native KV memory.

The first contract is deliberately narrow.  A family name is not sufficient
evidence: model identity, pinned revision, graph config, TensorRT version, and
GPU architecture must all match one family-owned qualification record.
"""

from __future__ import annotations

import hashlib
import os
import re
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping


class DynamicMemoryContractError(ValueError):
    """A candidate qualification or runtime-memory contract is invalid."""


@dataclass(frozen=True)
class BuildTarget:
    """Facts that affect native dynamic-memory qualification."""

    trt_version: str
    gpu_architecture: str
    cuda_runtime: str
    cudnn_backend: str
    cudnn_frontend_revision: str
    nvrtc: str
    driver: str

    def runtime_stack(self) -> dict[str, str]:
        """Return the independently detected stack using contract field names."""

        return {
            "sm": self.gpu_architecture.strip().lower(),
            "tensorrt": self.trt_version.strip(),
            "cuda_runtime": self.cuda_runtime.strip(),
            "cudnn_backend": self.cudnn_backend.strip(),
            "cudnn_frontend_revision":
                self.cudnn_frontend_revision.strip().lower(),
            "nvrtc": self.nvrtc.strip(),
            "driver": self.driver.strip(),
        }


@dataclass(frozen=True)
class DynamicMemoryQualification:
    """One exact model/revision/config/target qualification record."""

    family: str
    qualified_model_id: str
    qualified_model_revision: str
    qualified_config_sha256: str
    qualified_target: str
    minimum_trt_version: str
    gpu_architecture: str
    cuda_runtime: str
    cudnn_backend: str
    cudnn_frontend_revision: str
    nvrtc: str
    driver: str
    precision: str
    native_kv_plugin_abi: int
    model_context_limit: int
    prefill_chunk_limit: int
    kv_layout: str
    kv_dtype: str
    active_kv_profile_limits: tuple[int, ...]
    runtime_owned: bool

    def matches_target(self, target: BuildTarget) -> bool:
        return target.runtime_stack() == self.qualified_runtime_stack()

    def qualified_runtime_stack(self) -> dict[str, str]:
        """Return the exact live stack required by this qualification."""

        return {
            "sm": self.gpu_architecture,
            "tensorrt": self.minimum_trt_version,
            "cuda_runtime": self.cuda_runtime,
            "cudnn_backend": self.cudnn_backend,
            "cudnn_frontend_revision": self.cudnn_frontend_revision,
            "nvrtc": self.nvrtc,
            "driver": self.driver,
        }

    def runtime_memory_contract(
        self,
        *,
        model_dir: str | Path,
        config: Any,
        precision: str,
    ) -> dict[str, Any]:
        """Build the static bundle contract from the actual parsed config."""

        _require_config_fingerprint(
            Path(model_dir), self.qualified_config_sha256
        )
        actual_precision = str(precision or "").strip().lower()
        if actual_precision != self.precision:
            raise DynamicMemoryContractError(
                "Qualified dynamic-memory build precision mismatch: "
                f"expected {self.precision}, got {actual_precision or '<empty>'}"
            )

        model_limit = _positive_int(
            getattr(config, "max_position_embeddings", 0),
            "config.max_position_embeddings",
        )
        if model_limit != self.model_context_limit:
            raise DynamicMemoryContractError(
                "Qualified model context limit mismatch: "
                f"expected {self.model_context_limit}, got {model_limit}"
            )

        dtype_bytes = _dtype_bytes(self.kv_dtype)
        layers = _positive_int(
            getattr(config, "num_hidden_layers", 0),
            "config.num_hidden_layers",
        )
        kv_heads = _positive_int(
            getattr(config, "num_key_value_heads", 0),
            "config.num_key_value_heads",
        )
        head_dim = _positive_int(
            getattr(config, "head_dim", 0),
            "config.head_dim",
        )
        kv_bytes_per_token = (
            2 * layers * kv_heads * head_dim * dtype_bytes
        )

        return validate_runtime_memory_contract(
            {
                "contract_version": 1,
                "qualified_model_id": self.qualified_model_id,
                "qualified_model_revision": self.qualified_model_revision,
                "qualified_config_sha256": self.qualified_config_sha256,
                "qualified_target": self.qualified_target,
                "qualified_runtime_stack": {
                    "sm": self.gpu_architecture,
                    "tensorrt": self.minimum_trt_version,
                    "cuda_runtime": self.cuda_runtime,
                    "cudnn_backend": self.cudnn_backend,
                    "cudnn_frontend_revision":
                        self.cudnn_frontend_revision,
                    "nvrtc": self.nvrtc,
                    "driver": self.driver,
                },
                "native_kv_plugin_abi": self.native_kv_plugin_abi,
                "model_context_limit": self.model_context_limit,
                "prefill_chunk_limit": self.prefill_chunk_limit,
                "kv_layout": self.kv_layout,
                "kv_dtype": self.kv_dtype,
                "kv_bytes_per_token": kv_bytes_per_token,
                "active_kv_profile_limits": list(
                    self.active_kv_profile_limits
                ),
                "runtime_owned": self.runtime_owned,
            }
        )


@dataclass(frozen=True)
class ResolvedDynamicMemoryQualification:
    """A qualification whose snapshot, fingerprint, and target were verified."""

    qualification: DynamicMemoryQualification
    model_dir: Path
    target: BuildTarget

    @property
    def family(self) -> str:
        return self.qualification.family

    @property
    def precision(self) -> str:
        return self.qualification.precision

    @property
    def model_context_limit(self) -> int:
        return self.qualification.model_context_limit

    @property
    def active_kv_profile_limits(self) -> tuple[int, ...]:
        return self.qualification.active_kv_profile_limits

    @property
    def qualified_model_id(self) -> str:
        return self.qualification.qualified_model_id

    @property
    def qualified_model_revision(self) -> str:
        return self.qualification.qualified_model_revision

    def runtime_memory_contract(
        self, *, config: Any, precision: str
    ) -> dict[str, Any]:
        return self.qualification.runtime_memory_contract(
            model_dir=self.model_dir,
            config=config,
            precision=precision,
        )


_QUALIFICATION_FAMILIES = ("qwen", "llama")
DEVELOPER_CHUNK_VARIANT_ENV = "TRTMC_DEVELOPER_CHUNK_VARIANT"
DEVELOPER_CHUNK_VARIANT_VALUE = "C/2"
_DEVELOPER_CHUNK_VARIANT_MODEL_IDS = frozenset(
    {
        "Qwen/Qwen3-0.6B",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_VERSION_RE = re.compile(r"(\d+)\.(\d+)")
_RUNTIME_STACK_KEYS = frozenset(
    {
        "sm",
        "tensorrt",
        "cuda_runtime",
        "cudnn_backend",
        "cudnn_frontend_revision",
        "nvrtc",
        "driver",
    }
)
_RUNTIME_MEMORY_KEYS = frozenset(
    {
        "contract_version",
        "qualified_model_id",
        "qualified_model_revision",
        "qualified_config_sha256",
        "qualified_target",
        "qualified_runtime_stack",
        "native_kv_plugin_abi",
        "model_context_limit",
        "prefill_chunk_limit",
        "kv_layout",
        "kv_dtype",
        "kv_bytes_per_token",
        "active_kv_profile_limits",
        "runtime_owned",
    }
)


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicMemoryContractError(
            f"runtime_memory.{field} must be a positive integer"
        )
    return value


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DynamicMemoryContractError(
            f"runtime_memory.{field} must be a non-empty string"
        )
    return value.strip()


def _dtype_bytes(dtype: str) -> int:
    sizes = {
        "bfloat16": 2,
        "float16": 2,
        "float32": 4,
    }
    try:
        return sizes[dtype]
    except KeyError as exc:
        raise DynamicMemoryContractError(
            f"Unsupported qualified KV dtype: {dtype!r}"
        ) from exc


def _validate_runtime_stack(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise DynamicMemoryContractError(
            "runtime_memory.qualified_runtime_stack must be a JSON object"
        )
    unknown = sorted(set(value) - _RUNTIME_STACK_KEYS)
    missing = sorted(_RUNTIME_STACK_KEYS - set(value))
    if unknown:
        raise DynamicMemoryContractError(
            "runtime_memory.qualified_runtime_stack has unsupported "
            f"field(s): {unknown}"
        )
    if missing:
        raise DynamicMemoryContractError(
            "runtime_memory.qualified_runtime_stack is missing required "
            f"field(s): {missing}"
        )

    normalized = {
        key: _nonempty_string(
            value[key], f"qualified_runtime_stack.{key}"
        )
        for key in _RUNTIME_STACK_KEYS
    }
    normalized["sm"] = normalized["sm"].lower()
    if not re.fullmatch(r"sm\d+", normalized["sm"]):
        raise DynamicMemoryContractError(
            "runtime_memory.qualified_runtime_stack.sm must use the "
            "smNNN spelling"
        )
    if not _version_tuple(normalized["tensorrt"]):
        raise DynamicMemoryContractError(
            "runtime_memory.qualified_runtime_stack.tensorrt must contain "
            "major.minor"
        )
    if not _version_tuple(normalized["cuda_runtime"]):
        raise DynamicMemoryContractError(
            "runtime_memory.qualified_runtime_stack.cuda_runtime must "
            "contain major.minor"
        )
    if not _version_tuple(normalized["cudnn_backend"]):
        raise DynamicMemoryContractError(
            "runtime_memory.qualified_runtime_stack.cudnn_backend must "
            "contain major.minor"
        )
    normalized["cudnn_frontend_revision"] = normalized[
        "cudnn_frontend_revision"
    ].lower()
    if not _REVISION_RE.fullmatch(
        normalized["cudnn_frontend_revision"]
    ):
        raise DynamicMemoryContractError(
            "runtime_memory.qualified_runtime_stack."
            "cudnn_frontend_revision must be a 40-character lowercase "
            "commit SHA"
        )
    if not _version_tuple(normalized["nvrtc"]):
        raise DynamicMemoryContractError(
            "runtime_memory.qualified_runtime_stack.nvrtc must contain "
            "major.minor"
        )
    if not re.fullmatch(r"\d+(?:\.\d+)+", normalized["driver"]):
        raise DynamicMemoryContractError(
            "runtime_memory.qualified_runtime_stack.driver must be an "
            "exact dotted numeric driver release"
        )
    return {
        key: normalized[key]
        for key in (
            "sm",
            "tensorrt",
            "cuda_runtime",
            "cudnn_backend",
            "cudnn_frontend_revision",
            "nvrtc",
            "driver",
        )
    }


def validate_runtime_memory_contract(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize the version-1 bundle header object."""

    if not isinstance(value, Mapping):
        raise DynamicMemoryContractError(
            "runtime_memory must be a JSON object"
        )
    unknown = sorted(set(value) - _RUNTIME_MEMORY_KEYS)
    missing = sorted(_RUNTIME_MEMORY_KEYS - set(value))
    if unknown:
        raise DynamicMemoryContractError(
            f"runtime_memory has unsupported field(s): {unknown}"
        )
    if missing:
        raise DynamicMemoryContractError(
            f"runtime_memory is missing required field(s): {missing}"
        )

    version = _positive_int(value["contract_version"], "contract_version")
    if version != 1:
        raise DynamicMemoryContractError(
            f"Unsupported runtime_memory contract_version: {version}"
        )

    model_id = _nonempty_string(
        value["qualified_model_id"], "qualified_model_id"
    )
    revision = _nonempty_string(
        value["qualified_model_revision"], "qualified_model_revision"
    ).lower()
    if not _REVISION_RE.fullmatch(revision):
        raise DynamicMemoryContractError(
            "runtime_memory.qualified_model_revision must be a 40-character "
            "lowercase commit SHA"
        )
    fingerprint = _nonempty_string(
        value["qualified_config_sha256"], "qualified_config_sha256"
    ).lower()
    if not _SHA256_RE.fullmatch(fingerprint):
        raise DynamicMemoryContractError(
            "runtime_memory.qualified_config_sha256 must be a 64-character "
            "lowercase SHA-256"
        )
    target = _nonempty_string(
        value["qualified_target"], "qualified_target"
    )
    runtime_stack = _validate_runtime_stack(
        value["qualified_runtime_stack"]
    )
    plugin_abi = _positive_int(
        value["native_kv_plugin_abi"], "native_kv_plugin_abi"
    )
    model_limit = _positive_int(
        value["model_context_limit"], "model_context_limit"
    )
    chunk_limit = _positive_int(
        value["prefill_chunk_limit"], "prefill_chunk_limit"
    )
    if chunk_limit > model_limit:
        raise DynamicMemoryContractError(
            "runtime_memory.prefill_chunk_limit cannot exceed "
            "model_context_limit"
        )
    layout = _nonempty_string(value["kv_layout"], "kv_layout")
    dtype = _nonempty_string(value["kv_dtype"], "kv_dtype")
    _dtype_bytes(dtype)
    bytes_per_token = _positive_int(
        value["kv_bytes_per_token"], "kv_bytes_per_token"
    )

    raw_buckets = value["active_kv_profile_limits"]
    if (
        not isinstance(raw_buckets, (list, tuple))
        or not raw_buckets
    ):
        raise DynamicMemoryContractError(
            "runtime_memory.active_kv_profile_limits must be a non-empty array"
        )
    buckets = [
        _positive_int(bucket, "active_kv_profile_limits[]")
        for bucket in raw_buckets
    ]
    if buckets != sorted(set(buckets)):
        raise DynamicMemoryContractError(
            "runtime_memory.active_kv_profile_limits must be strictly "
            "increasing and unique"
        )
    if buckets[-1] != model_limit:
        raise DynamicMemoryContractError(
            "runtime_memory.active_kv_profile_limits must end at "
            "model_context_limit"
        )
    if chunk_limit not in buckets:
        raise DynamicMemoryContractError(
            "runtime_memory.prefill_chunk_limit must be one of "
            "active_kv_profile_limits"
        )
    if value["runtime_owned"] is not True:
        raise DynamicMemoryContractError(
            "runtime_memory.runtime_owned must be true"
        )

    return {
        "contract_version": version,
        "qualified_model_id": model_id,
        "qualified_model_revision": revision,
        "qualified_config_sha256": fingerprint,
        "qualified_target": target,
        "qualified_runtime_stack": runtime_stack,
        "native_kv_plugin_abi": plugin_abi,
        "model_context_limit": model_limit,
        "prefill_chunk_limit": chunk_limit,
        "kv_layout": layout,
        "kv_dtype": dtype,
        "kv_bytes_per_token": bytes_per_token,
        "active_kv_profile_limits": buckets,
        "runtime_owned": True,
    }


def _version_tuple(value: str) -> tuple[int, int] | None:
    match = _VERSION_RE.search(str(value or ""))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _qualification_path(family: str) -> Path:
    return Path(__file__).resolve().parent / "families" / family / "MODEL.toml"


def _load_qualification(
    family: str, raw: Mapping[str, Any]
) -> DynamicMemoryQualification:
    try:
        buckets = tuple(
            _positive_int(value, "active_kv_profile_limits[]")
            for value in raw["active_kv_profile_limits"]
        )
        qualification = DynamicMemoryQualification(
            family=family,
            qualified_model_id=_nonempty_string(
                raw["model_id"], "qualified_model_id"
            ),
            qualified_model_revision=_nonempty_string(
                raw["revision"], "qualified_model_revision"
            ).lower(),
            qualified_config_sha256=_nonempty_string(
                raw["config_sha256"], "qualified_config_sha256"
            ).lower(),
            qualified_target=_nonempty_string(
                raw["target"], "qualified_target"
            ),
            minimum_trt_version=_nonempty_string(
                raw["minimum_trt_version"], "minimum_trt_version"
            ),
            gpu_architecture=_nonempty_string(
                raw["gpu_architecture"], "gpu_architecture"
            ).lower(),
            cuda_runtime=_nonempty_string(
                raw["cuda_runtime"], "cuda_runtime"
            ),
            cudnn_backend=_nonempty_string(
                raw["cudnn_backend"], "cudnn_backend"
            ),
            cudnn_frontend_revision=_nonempty_string(
                raw["cudnn_frontend_revision"],
                "cudnn_frontend_revision",
            ).lower(),
            nvrtc=_nonempty_string(raw["nvrtc"], "nvrtc"),
            driver=_nonempty_string(raw["driver"], "driver"),
            precision=_nonempty_string(
                raw["precision"], "precision"
            ).lower(),
            native_kv_plugin_abi=_positive_int(
                raw["native_kv_plugin_abi"], "native_kv_plugin_abi"
            ),
            model_context_limit=_positive_int(
                raw["model_context_limit"], "model_context_limit"
            ),
            prefill_chunk_limit=_positive_int(
                raw["prefill_chunk_limit"], "prefill_chunk_limit"
            ),
            kv_layout=_nonempty_string(raw["kv_layout"], "kv_layout"),
            kv_dtype=_nonempty_string(raw["kv_dtype"], "kv_dtype"),
            active_kv_profile_limits=buckets,
            runtime_owned=raw["runtime_owned"] is True,
        )
    except (KeyError, TypeError) as exc:
        raise DynamicMemoryContractError(
            f"Invalid dynamic_memory_qualification in "
            f"{_qualification_path(family)}: {exc}"
        ) from exc

    validate_runtime_memory_contract(
        {
            "contract_version": 1,
            "qualified_model_id": qualification.qualified_model_id,
            "qualified_model_revision": qualification.qualified_model_revision,
            "qualified_config_sha256": qualification.qualified_config_sha256,
            "qualified_target": qualification.qualified_target,
            "qualified_runtime_stack": {
                "sm": qualification.gpu_architecture,
                "tensorrt": qualification.minimum_trt_version,
                "cuda_runtime": qualification.cuda_runtime,
                "cudnn_backend": qualification.cudnn_backend,
                "cudnn_frontend_revision":
                    qualification.cudnn_frontend_revision,
                "nvrtc": qualification.nvrtc,
                "driver": qualification.driver,
            },
            "native_kv_plugin_abi": qualification.native_kv_plugin_abi,
            "model_context_limit": qualification.model_context_limit,
            "prefill_chunk_limit": qualification.prefill_chunk_limit,
            "kv_layout": qualification.kv_layout,
            "kv_dtype": qualification.kv_dtype,
            # Structural validation only. Actual B is derived from config.
            "kv_bytes_per_token": 1,
            "active_kv_profile_limits": list(
                qualification.active_kv_profile_limits
            ),
            "runtime_owned": qualification.runtime_owned,
        }
    )
    if not _version_tuple(qualification.minimum_trt_version):
        raise DynamicMemoryContractError(
            "minimum_trt_version must contain major.minor"
        )
    if not re.fullmatch(r"sm\d+", qualification.gpu_architecture):
        raise DynamicMemoryContractError(
            "gpu_architecture must use the smNNN spelling"
        )
    if qualification.precision != "bf16":
        raise DynamicMemoryContractError(
            "Initial dynamic-memory qualifications must use bf16"
        )
    return qualification


def load_dynamic_memory_qualifications(
) -> tuple[DynamicMemoryQualification, ...]:
    """Load exact qualification records owned by the two initial families."""

    records: list[DynamicMemoryQualification] = []
    for family in _QUALIFICATION_FAMILIES:
        path = _qualification_path(family)
        with path.open("rb") as source:
            raw = tomllib.load(source)
        entries = raw.get("dynamic_memory_qualification", [])
        if not isinstance(entries, list):
            raise DynamicMemoryContractError(
                f"{path}: dynamic_memory_qualification must be an array of tables"
            )
        records.extend(_load_qualification(family, entry) for entry in entries)

    identities = [
        (record.qualified_model_id, record.qualified_model_revision)
        for record in records
    ]
    if len(identities) != len(set(identities)):
        raise DynamicMemoryContractError(
            "Duplicate dynamic-memory qualification identity"
        )
    return tuple(records)


def require_developer_chunk_variant_opt_in(
    environment: Mapping[str, str] | None = None,
) -> None:
    """Require the exact developer-only opt-in for a C/2 build or run."""

    active_environment = os.environ if environment is None else environment
    actual = active_environment.get(DEVELOPER_CHUNK_VARIANT_ENV)
    if actual != DEVELOPER_CHUNK_VARIANT_VALUE:
        raise DynamicMemoryContractError(
            "Developer C/2 qualification requires the explicit opt-in "
            f"{DEVELOPER_CHUNK_VARIANT_ENV}="
            f"{DEVELOPER_CHUNK_VARIANT_VALUE}; got "
            f"{actual!r}"
        )


def _declared_qualification(
    qualification: DynamicMemoryQualification,
) -> DynamicMemoryQualification:
    matches = [
        record
        for record in load_dynamic_memory_qualifications()
        if (
            record.qualified_model_id
            == qualification.qualified_model_id
            and record.qualified_model_revision
            == qualification.qualified_model_revision
        )
    ]
    if len(matches) != 1:
        raise DynamicMemoryContractError(
            "Developer chunk variants require one exact declared "
            "dynamic-memory qualification"
        )
    return matches[0]


def _c_div_2_record(
    base: DynamicMemoryQualification,
) -> DynamicMemoryQualification:
    if (
        base.qualified_model_id
        not in _DEVELOPER_CHUNK_VARIANT_MODEL_IDS
    ):
        raise DynamicMemoryContractError(
            "Developer C/2 qualification is limited to the two initial "
            "qualified models"
        )
    chunk_limit = base.prefill_chunk_limit
    if chunk_limit % 2 != 0:
        raise DynamicMemoryContractError(
            "Developer C/2 qualification requires an even default "
            "prefill_chunk_limit"
        )
    variant_chunk_limit = chunk_limit // 2
    variant_buckets = tuple(
        sorted(
            {
                *base.active_kv_profile_limits,
                variant_chunk_limit,
            }
        )
    )
    return replace(
        base,
        prefill_chunk_limit=variant_chunk_limit,
        active_kv_profile_limits=variant_buckets,
    )


def derive_developer_chunk_variant_qualification(
    resolved: ResolvedDynamicMemoryQualification,
    *,
    environment: Mapping[str, str] | None = None,
) -> ResolvedDynamicMemoryQualification:
    """Derive the one legal C/2 variant from an exact default record."""

    require_developer_chunk_variant_opt_in(environment)
    declared = _declared_qualification(resolved.qualification)
    if resolved.qualification != declared:
        raise DynamicMemoryContractError(
            "Developer C/2 qualification must be derived from the exact "
            "default qualification record"
        )
    if not declared.matches_target(resolved.target):
        raise DynamicMemoryContractError(
            "Developer C/2 qualification target does not match the "
            "declared qualification"
        )
    return replace(
        resolved,
        qualification=_c_div_2_record(declared),
    )


def validate_qualified_native_build(
    resolved: ResolvedDynamicMemoryQualification,
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Accept only the declared build or its explicitly opted-in C/2 variant."""

    if not isinstance(resolved, ResolvedDynamicMemoryQualification):
        raise DynamicMemoryContractError(
            "Qualified native builds require a resolved exact qualification"
        )
    declared = _declared_qualification(resolved.qualification)
    if not declared.matches_target(resolved.target):
        raise DynamicMemoryContractError(
            "Qualified native build target does not match its declaration"
        )
    expected_identity = (
        declared.qualified_model_id,
        declared.qualified_model_revision,
    )
    if _snapshot_identity(resolved.model_dir) != expected_identity:
        raise DynamicMemoryContractError(
            "Qualified native builds require the exact declared HF snapshot"
        )
    _require_config_fingerprint(
        resolved.model_dir,
        declared.qualified_config_sha256,
    )
    if resolved.qualification == declared:
        return "default"

    require_developer_chunk_variant_opt_in(environment)
    if resolved.qualification != _c_div_2_record(declared):
        raise DynamicMemoryContractError(
            "Qualified native build rejected a non-default variant; only "
            "the exact developer C/2 qualification is allowed"
        )
    return "developer_c_div_2"


def _snapshot_identity(path: Path) -> tuple[str, str] | None:
    """Recover ``org/repo`` and revision from an HF cache snapshot path."""

    absolute = path.resolve()
    parts = absolute.parts
    for index, part in enumerate(parts):
        if part != "snapshots" or index < 1 or index + 1 >= len(parts):
            continue
        if index + 2 != len(parts):
            continue
        cache_name = parts[index - 1]
        if not cache_name.startswith("models--"):
            continue
        identity_parts = cache_name.removeprefix("models--").split("--", 1)
        if len(identity_parts) != 2 or not all(identity_parts):
            continue
        return "/".join(identity_parts), parts[index + 1]
    return None


def qualification_for_model_ref(
    model_ref: str,
    *,
    requested_revision: str | None = None,
) -> DynamicMemoryQualification | None:
    """Return a candidate record without probing TensorRT or CUDA."""

    records = load_dynamic_memory_qualifications()
    normalized_ref = str(model_ref or "").strip()
    local = Path(normalized_ref)
    identity = (
        _snapshot_identity(local)
        if local.is_dir()
        else None
    )
    if identity is None:
        model_id = normalized_ref
        revision = requested_revision
    else:
        model_id, revision = identity
        if requested_revision and requested_revision != revision:
            if any(
                record.qualified_model_id == model_id
                for record in records
            ):
                raise DynamicMemoryContractError(
                    "Recognized runtime-memory-qualified HF snapshot revision "
                    "conflicts with the explicitly requested revision: "
                    f"snapshot {revision!r}, requested {requested_revision!r}"
                )
            return None

    for record in records:
        if record.qualified_model_id != model_id:
            continue
        if revision and record.qualified_model_revision != revision:
            raise DynamicMemoryContractError(
                "Recognized runtime-memory-qualified model revision mismatch: "
                f"{model_id} requires {record.qualified_model_revision}, "
                f"got {revision}"
            )
        return record
    return None


def probe_build_target() -> BuildTarget:
    """Probe the complete live stack from the native runtime-KV plugin."""

    # Keep this import local: trt_plugins imports trt_compat and is also used by
    # the qualified builders after this candidate gate has succeeded.
    from .trt_plugins import query_runtime_kv_plugin_stack

    stack = query_runtime_kv_plugin_stack()
    return BuildTarget(
        trt_version=stack["tensorrt"],
        gpu_architecture=stack["sm"],
        cuda_runtime=stack["cuda_runtime"],
        cudnn_backend=stack["cudnn_backend"],
        cudnn_frontend_revision=stack["cudnn_frontend_revision"],
        nvrtc=stack["nvrtc"],
        driver=stack["driver"],
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_config_fingerprint(
    model_dir: Path, expected_sha256: str
) -> None:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise DynamicMemoryContractError(
            f"Qualified model is missing config.json: {model_dir}"
        )
    actual = _sha256_file(config_path)
    if actual != expected_sha256:
        raise DynamicMemoryContractError(
            "Qualified model config fingerprint mismatch: "
            f"expected {expected_sha256}, got {actual}"
        )


def resolve_model_only_qualification(
    model_ref: str,
    *,
    requested_revision: str | None,
    resolve_model: Callable[..., str],
) -> ResolvedDynamicMemoryQualification | None:
    """Resolve and verify the exact no-build-flag qualification.

    Unknown local directories and unqualified targets are not applicable and
    return ``None``.  Once an exact identity is recognized, a corrupt or stale
    config is an invalid candidate and raises rather than silently opting in.
    """

    qualification = qualification_for_model_ref(
        model_ref,
        requested_revision=requested_revision,
    )
    if qualification is None:
        return None

    try:
        target = probe_build_target()
    except Exception as exc:
        raise DynamicMemoryContractError(
            "Recognized runtime-memory-qualified model, but the active "
            "TensorRT/GPU target could not be verified"
        ) from exc
    if not qualification.matches_target(target):
        expected_stack = qualification.qualified_runtime_stack()
        actual_stack = target.runtime_stack()
        mismatches = [
            name
            for name in expected_stack
            if expected_stack[name] != actual_stack[name]
        ]
        raise DynamicMemoryContractError(
            "Recognized runtime-memory-qualified model, but this build target "
            "is not qualified: mismatched runtime-stack field(s) "
            f"{mismatches}; expected {expected_stack}, got {actual_stack}"
        )

    local = Path(str(model_ref))
    if local.is_dir():
        resolved = Path(resolve_model(str(local)))
        resolved_identity = _snapshot_identity(resolved)
        expected_identity = (
            qualification.qualified_model_id,
            qualification.qualified_model_revision,
        )
        if resolved_identity != expected_identity:
            raise DynamicMemoryContractError(
                "Recognized runtime-memory-qualified local model resolved to "
                "a different HF snapshot: "
                f"expected {expected_identity}, got {resolved_identity}"
            )
    else:
        resolved = Path(
            resolve_model(
                qualification.qualified_model_id,
                revision=qualification.qualified_model_revision,
            )
        )

    _require_config_fingerprint(
        resolved, qualification.qualified_config_sha256
    )
    return ResolvedDynamicMemoryQualification(
        qualification=qualification,
        model_dir=resolved,
        target=target,
    )
