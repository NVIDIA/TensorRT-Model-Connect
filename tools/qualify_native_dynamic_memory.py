#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic long-context qualification for native runtime-memory bundles.

The runner consumes token IDs rather than text, loads each real bundle through
the normal C++ factory, and writes complete final-position/decode logits.  This
script compares those rows with the pinned Hugging Face checkpoint and enforces
the model's existing family thresholds.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import struct
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from tensorrt_model_connect.dynamic_memory_contract import (  # noqa: E402
    DEVELOPER_CHUNK_VARIANT_ENV,
    DEVELOPER_CHUNK_VARIANT_VALUE,
    require_developer_chunk_variant_opt_in,
)


BUNDLE_MAGIC = b"TRTFB\x00\x01\x00"
LOGITS_MAGIC = b"TRTMCQL1"
LOGITS_HEADER = struct.Struct("<8sIIQQ")
CHUNK_VARIANT_BUILD_SCHEMA = "trtmc.native-dynamic-memory-chunk-variant-build/v2"
BUILD_MANIFEST_SCHEMA = "trtmc.dynamic-memory-test-manifest/v2"
BASE_ARTIFACT_BINDING_SCHEMA = (
    "trtmc.native-dynamic-memory-base-artifact-binding/v1"
)
RUNTIME_KV_PLUGIN_BINDING_SCHEMA = (
    "trtmc.native-dynamic-memory-runtime-kv-plugin-binding/v1"
)
RUNTIME_KV_PLUGIN_ENV = "TRTMC_TRT_PLUGIN_LIBRARY"
RUNTIME_KV_PLUGIN_ABI_SYMBOL = (
    "trtmc_runtime_kv_plugin_abi_version"
)
_BINARY_IDENTITY_FIELDS = (
    "path",
    "device",
    "inode",
    "size_bytes",
    "mtime_ns",
    "ctime_ns",
    "sha256",
)
_BASE_ARTIFACT_BINDING_FIELDS = {
    "schema_version",
    "build_manifest",
    "base_build_receipt",
    "bundle",
    "qualifier_runner",
    "benchmark_worker",
    "core",
    "trt_backend",
    "model_plugin",
    "runtime_kv_plugin",
    "source",
    "qualified_model",
}
_RUNTIME_KV_PLUGIN_BINDING_FIELDS = {
    "schema_version",
    "environment",
    "environment_was_set",
    "preload_mapping",
    "selected",
    "loaded_mapping",
}
_KV_DTYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
}
_GRAPH_MODEL_CONTRACT_FIELDS = {
    "model_context_limit",
    "prefill_chunk_limit",
    "active_kv_profile_limits",
    "num_layers",
    "vocab_size",
    "kv_dtype",
    "kv_bytes_per_token",
    "kv_width",
}
_LIFETIME_PROTOCOL = {
    "schema_version": 1,
    "execution_order": ["warmup", "measured"],
    "warmup_count": 1,
    "measured_count": 1,
}
_RUNTIME_PHASES_AFTER_BASELINE = (
    "before runtime KV planning",
    "after shared context and output allocation",
    "after runtime KV allocation",
    "after successful runtime-memory request completion",
)
_MEMORY_ATTRIBUTION_FLOOR_BYTES = 64 * 1024 * 1024
_COLD_START_RETENTION_FLOOR_BYTES = 512 * 1024 * 1024
_COLD_START_RETENTION_WEIGHT_FRACTION = 0.05
_COLD_START_PERSISTENT_DRIVER_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
_COLD_WARM_OUTPUT_EQUIVALENCE_FIELDS = {
    "schema_version",
    "warmup_execution_ordinal",
    "measured_execution_ordinal",
    "prompt_tokens_equal",
    "prefill_launches_equal",
    "decode_launches_equal",
    "final_kv_position_equal",
    "selected_token_ids_equal",
    "step_top1_token_ids_equal",
    "full_float32_logits_bitwise_equal",
    "passed",
}
RUNNER_CAPTURE_SCHEMA = "trtmc.native-dynamic-memory-runner-capture/v1"
_FULL_GPU_UUID_PATTERN = re.compile(
    r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    context_limit: int
    chunk_limit: int
    buckets: tuple[int, ...]
    num_layers: int
    vocab_size: int
    kv_dtype: str
    kv_bytes_per_token: int
    num_query_heads: int
    threshold_path: str


@dataclass(frozen=True)
class TrustedRuntimeGeometry:
    model_context_limit: int
    prefill_chunk_limit: int
    kv_bytes_per_token: int

    def __post_init__(self) -> None:
        for name, value in (
            ("model_context_limit", self.model_context_limit),
            ("prefill_chunk_limit", self.prefill_chunk_limit),
            ("kv_bytes_per_token", self.kv_bytes_per_token),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"trusted runtime geometry {name} must be positive")
        if self.prefill_chunk_limit > self.model_context_limit:
            raise ValueError(
                "trusted prefill chunk limit exceeds the model context limit"
            )


@dataclass(frozen=True)
class SamplerTrustAnchor:
    pid: int
    cuda_logical_device_index: int
    physical_device_index: int
    pci_bus_id: str
    gpu_uuid: str

    def __post_init__(self) -> None:
        if type(self.pid) is not int or self.pid <= 0:
            raise ValueError("trusted sampler child PID must be positive")
        if (
            type(self.cuda_logical_device_index) is not int
            or self.cuda_logical_device_index < 0
            or type(self.physical_device_index) is not int
            or self.physical_device_index < 0
        ):
            raise ValueError("trusted sampler CUDA/NVML indices are invalid")
        if (
            not isinstance(self.pci_bus_id, str)
            or not self.pci_bus_id
            or not isinstance(self.gpu_uuid, str)
            or not self.gpu_uuid
        ):
            raise ValueError("trusted sampler PCI/UUID identity is invalid")


def trusted_runtime_geometry(
    spec: ModelSpec,
    *,
    prefill_chunk_limit: int | None = None,
) -> TrustedRuntimeGeometry:
    return TrustedRuntimeGeometry(
        model_context_limit=spec.context_limit,
        prefill_chunk_limit=(
            spec.chunk_limit
            if prefill_chunk_limit is None
            else prefill_chunk_limit
        ),
        kv_bytes_per_token=spec.kv_bytes_per_token,
    )


SPECS = {
    "Qwen/Qwen3-0.6B": ModelSpec(
        model_id="Qwen/Qwen3-0.6B",
        context_limit=40_960,
        chunk_limit=1_024,
        buckets=(128, 256, 512, 1_024, 2_048, 8_192, 32_768, 40_960),
        num_layers=28,
        vocab_size=151_936,
        kv_dtype="bfloat16",
        kv_bytes_per_token=114_688,
        num_query_heads=16,
        threshold_path="tests/e2e/models/qwen/thresholds/qwen3-0.6b-topp.json",
    ),
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": ModelSpec(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        context_limit=2_048,
        chunk_limit=512,
        buckets=(128, 256, 512, 2_048),
        num_layers=22,
        vocab_size=32_000,
        kv_dtype="bfloat16",
        kv_bytes_per_token=22_528,
        num_query_heads=32,
        threshold_path="tests/e2e/models/llama/thresholds/tinyllama-1.1b.json",
    ),
}


@dataclass(frozen=True)
class Case:
    name: str
    prompt_tokens: int
    decode_tokens: int
    expect_admission_rejection: bool = False
    expected_decode_profile_ids: tuple[int, ...] = ()
    expected_decode_bucket_limits: tuple[int, ...] = ()


def _read_bundle_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as bundle:
        if bundle.read(8) != BUNDLE_MAGIC:
            raise ValueError(f"{path} is not a TRTMC bundle")
        raw_length = bundle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"{path} has a truncated header length")
        header_length = struct.unpack("<Q", raw_length)[0]
        payload = bundle.read(header_length)
        if len(payload) != header_length:
            raise ValueError(f"{path} has a truncated JSON header")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("bundle JSON header is not an object")
    return parsed


def _resolve_spec(header: dict[str, Any]) -> ModelSpec:
    contract = header.get("runtime_memory")
    if not isinstance(contract, dict):
        raise ValueError("bundle has no runtime_memory contract")
    model_id = contract.get("qualified_model_id")
    try:
        spec = SPECS[str(model_id)]
    except KeyError as exc:
        raise ValueError(
            f"bundle model {model_id!r} is not one of the two qualified models"
        ) from exc
    expected = {
        "contract_version": 1,
        "model_context_limit": spec.context_limit,
        "prefill_chunk_limit": spec.chunk_limit,
        "active_kv_profile_limits": list(spec.buckets),
        "kv_bytes_per_token": spec.kv_bytes_per_token,
        "runtime_owned": True,
    }
    mismatches = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise ValueError(f"bundle qualification contract mismatch: {mismatches}")
    return spec


def _read_bundle_section(
    path: Path,
    header: Mapping[str, Any],
    section_name: str,
) -> bytes:
    sections = header.get("sections")
    if not isinstance(sections, Mapping):
        raise RuntimeError("qualified bundle has no section table")
    section = sections.get(section_name)
    if not isinstance(section, Mapping):
        raise RuntimeError(f"qualified bundle is missing {section_name}")
    offset = section.get("offset")
    size = section.get("size")
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < 0
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
    ):
        raise RuntimeError(f"qualified bundle has invalid {section_name} offset/size")
    with path.open("rb") as bundle:
        if bundle.read(8) != BUNDLE_MAGIC:
            raise RuntimeError(f"{path} is not a TRTMC bundle")
        raw_length = bundle.read(8)
        if len(raw_length) != 8:
            raise RuntimeError(f"{path} has a truncated header length")
        header_length = struct.unpack("<Q", raw_length)[0]
        data_offset = 16 + header_length
        file_size = path.stat().st_size
        if (
            data_offset > file_size
            or offset > file_size - data_offset
            or size > file_size - data_offset - offset
        ):
            raise RuntimeError(f"qualified bundle has truncated {section_name} bytes")
        bundle.seek(data_offset + offset)
        plan = bundle.read(size)
    if len(plan) != size:
        raise RuntimeError(f"qualified bundle has truncated {section_name} bytes")
    return plan


def _shape_list(shape: Any) -> list[int]:
    try:
        return [int(value) for value in shape]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"TensorRT inspector returned invalid shape {shape!r}") from exc


def _positive_graph_contract_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"qualified engine model_contract {field} is not a positive integer")
    return value


def _derive_graph_kv_width(
    *,
    kv_dtype: Any,
    kv_bytes_per_token: Any,
    num_layers: Any,
) -> int:
    if not isinstance(kv_dtype, str) or kv_dtype not in _KV_DTYPE_BYTES:
        raise RuntimeError(
            f"qualified engine model_contract has unsupported KV dtype {kv_dtype!r}"
        )
    bytes_per_token = _positive_graph_contract_int(
        kv_bytes_per_token,
        "kv_bytes_per_token",
    )
    layers = _positive_graph_contract_int(num_layers, "num_layers")
    divisor = 2 * layers * _KV_DTYPE_BYTES[kv_dtype]
    if bytes_per_token % divisor != 0:
        raise RuntimeError(
            "qualified engine model_contract kv_bytes_per_token is not exactly "
            "divisible by 2*num_layers*kv_dtype_bytes"
        )
    width = bytes_per_token // divisor
    if width <= 0:
        raise RuntimeError("qualified engine model_contract derived KV width is not positive")
    return width


def _graph_model_contract_from_bundle_header(header: Mapping[str, Any]) -> dict[str, Any]:
    runtime_memory = header.get("runtime_memory")
    if not isinstance(runtime_memory, Mapping):
        raise RuntimeError("qualified bundle has no runtime_memory contract")
    result = {
        "model_context_limit": runtime_memory.get("model_context_limit"),
        "prefill_chunk_limit": runtime_memory.get("prefill_chunk_limit"),
        "active_kv_profile_limits": runtime_memory.get("active_kv_profile_limits"),
        "num_layers": header.get("num_layers"),
        "vocab_size": header.get("vocab_size"),
        "kv_dtype": runtime_memory.get("kv_dtype"),
        "kv_bytes_per_token": runtime_memory.get("kv_bytes_per_token"),
    }
    result["kv_width"] = _derive_graph_kv_width(
        kv_dtype=result["kv_dtype"],
        kv_bytes_per_token=result["kv_bytes_per_token"],
        num_layers=result["num_layers"],
    )
    return result


def _validate_graph_model_contract(
    value: Any,
    spec: ModelSpec,
    *,
    num_layers: int,
    chunk_limit: int,
    buckets: tuple[int, ...],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _GRAPH_MODEL_CONTRACT_FIELDS:
        actual_fields = sorted(value) if isinstance(value, Mapping) else value
        raise RuntimeError(
            "qualified engine model_contract fields mismatch: "
            f"expected {sorted(_GRAPH_MODEL_CONTRACT_FIELDS)}, got {actual_fields!r}"
        )
    if num_layers != spec.num_layers:
        raise RuntimeError(
            "qualified bundle num_layers does not match the pinned model: "
            f"expected {spec.num_layers}, got {num_layers}"
        )
    _positive_graph_contract_int(value.get("model_context_limit"), "model_context_limit")
    _positive_graph_contract_int(value.get("prefill_chunk_limit"), "prefill_chunk_limit")
    _positive_graph_contract_int(value.get("vocab_size"), "vocab_size")
    derived_width = _derive_graph_kv_width(
        kv_dtype=value.get("kv_dtype"),
        kv_bytes_per_token=value.get("kv_bytes_per_token"),
        num_layers=value.get("num_layers"),
    )
    if value.get("kv_width") != derived_width:
        raise RuntimeError(
            "qualified engine model_contract KV width does not match "
            f"the exact B/(2*num_layers*dtype_bytes) derivation: "
            f"expected {derived_width}, got {value.get('kv_width')!r}"
        )
    expected = {
        "model_context_limit": spec.context_limit,
        "prefill_chunk_limit": chunk_limit,
        "active_kv_profile_limits": list(buckets),
        "num_layers": spec.num_layers,
        "vocab_size": spec.vocab_size,
        "kv_dtype": spec.kv_dtype,
        "kv_bytes_per_token": spec.kv_bytes_per_token,
        "kv_width": derived_width,
    }
    mismatches = {
        field: {"expected": expected[field], "actual": value.get(field)}
        for field in sorted(expected)
        if value.get(field) != expected[field]
    }
    if mismatches:
        raise RuntimeError(f"qualified engine model_contract mismatch: {mismatches}")
    return expected


def _walk_json_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for nested in value.values():
            yield from _walk_json_mappings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_json_mappings(nested)


def _cache_concat_layers(inspector_json: Any) -> list[str]:
    matches: list[str] = []
    for record in _walk_json_mappings(inspector_json):
        layer_descriptor = " ".join(
            str(record.get(key, ""))
            for key in (
                "Name",
                "name",
                "LayerType",
                "layer_type",
                "Type",
                "type",
            )
        ).lower()
        if "concat" not in layer_descriptor:
            continue
        serialized = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
        )
        lowered = serialized.lower()
        if re.search(r"cache_[kv]_[0-9]+", lowered) is not None:
            name = record.get("Name", record.get("name", "<unnamed>"))
            matches.append(str(name))
    return sorted(set(matches))


def _dense_attention_layers(inspector_json: Any) -> list[str]:
    matches: list[str] = []
    dense_markers = (
        "attention_mask",
        "attention.mask",
        "causal_mask",
        "causal.mask",
        "dense_attention",
        "dense.attention",
        "attention_scores",
        "attention.scores",
    )
    for record in _walk_json_mappings(inspector_json):
        layer_descriptor = " ".join(
            str(record.get(key, ""))
            for key in (
                "Name",
                "name",
                "LayerType",
                "layer_type",
                "Type",
                "type",
            )
        ).lower()
        if not any(marker in layer_descriptor for marker in dense_markers):
            continue
        name = record.get("Name", record.get("name", "<unnamed>"))
        matches.append(str(name))
    return sorted(set(matches))


def _validate_qualified_engine_graph_evidence(
    evidence: Mapping[str, Any],
    spec: ModelSpec,
    *,
    num_layers: int,
    chunk_limit: int | None = None,
    buckets: tuple[int, ...] | None = None,
    expected_runtime_stack: Mapping[str, Any],
) -> dict[str, Any]:
    if num_layers <= 0:
        raise RuntimeError("qualified bundle has no model layers")
    runtime_stack = evidence.get("runtime_stack")
    if not isinstance(runtime_stack, Mapping) or not runtime_stack:
        raise RuntimeError("qualified engine evidence has no exact live runtime stack")
    if not expected_runtime_stack or dict(runtime_stack) != dict(expected_runtime_stack):
        raise RuntimeError(
            "qualified engine evidence runtime stack does not match the expected live stack"
        )
    expected_chunk = chunk_limit if chunk_limit is not None else spec.chunk_limit
    expected_buckets = buckets if buckets is not None else spec.buckets
    model_contract = _validate_graph_model_contract(
        evidence.get("model_contract"),
        spec,
        num_layers=num_layers,
        chunk_limit=expected_chunk,
        buckets=expected_buckets,
    )
    expected_kv_width = model_contract["kv_width"]
    expected_vocab_size = model_contract["vocab_size"]
    sections = evidence.get("engine_sections")
    if not isinstance(sections, Mapping) or set(sections) != {
        "prefill_engine_plan",
        "engine_plan",
    }:
        raise RuntimeError(
            "qualified engine evidence must contain distinct prefill/decode sections"
        )

    expected_inputs = {
        "token_id",
        "position_id",
        "history_length",
        *{f"cache_k_{layer}" for layer in range(num_layers)},
        *{f"cache_v_{layer}" for layer in range(num_layers)},
    }
    expected_outputs = {
        "logits",
        *{f"present_k_{layer}" for layer in range(num_layers)},
        *{f"present_v_{layer}" for layer in range(num_layers)},
    }
    summary: dict[str, Any] = {
        "passed": True,
        "runtime_stack": dict(runtime_stack),
        "model_contract": model_contract,
        "gates": {
            "actual_split_engine_sections": True,
            "distinct_prefill_decode_plans": True,
            "no_attention_mask_input": True,
            "current_rows_only_present_outputs": True,
            "native_segmented_attention_covers_full_model": True,
            "no_dense_attention_mask_or_scores": True,
            "no_cache_concat_fallback": True,
        },
        "engine_sections": {},
    }

    for section_name, expected_role in (
        ("prefill_engine_plan", "prefill"),
        ("engine_plan", "decode"),
    ):
        raw_section = sections[section_name]
        if not isinstance(raw_section, Mapping):
            raise RuntimeError(f"{section_name} inspector evidence is not an object")
        inputs = raw_section.get("inputs")
        outputs = raw_section.get("outputs")
        if isinstance(inputs, Mapping) and "attention_mask" in inputs:
            raise RuntimeError(f"{section_name} exposes forbidden attention_mask input")
        if not isinstance(inputs, Mapping) or set(inputs) != expected_inputs:
            raise RuntimeError(
                f"{section_name} input contract mismatch: expected "
                f"{sorted(expected_inputs)}, got "
                f"{sorted(inputs) if isinstance(inputs, Mapping) else inputs!r}"
            )
        if not isinstance(outputs, Mapping) or set(outputs) != expected_outputs:
            raise RuntimeError(
                f"{section_name} output contract mismatch: expected "
                f"{sorted(expected_outputs)}, got "
                f"{sorted(outputs) if isinstance(outputs, Mapping) else outputs!r}"
            )
        profile_count = raw_section.get("num_optimization_profiles")
        expected_profile_count = 1 if expected_role == "prefill" else len(expected_buckets)
        if profile_count != expected_profile_count:
            raise RuntimeError(
                f"{section_name} has {profile_count!r} profiles, expected {expected_profile_count}"
            )
        plugin_layers = raw_section.get("native_contiguous_attention_layer_indices")
        if plugin_layers != list(range(num_layers)):
            raise RuntimeError(
                f"{section_name} does not expose one "
                "NativeContiguousAttentionV2 layer per model layer"
            )
        dense_attention_layers = raw_section.get("dense_attention_layers")
        if dense_attention_layers != []:
            raise RuntimeError(
                f"{section_name} contains a dense attention mask/score path: "
                f"{dense_attention_layers!r}"
            )
        concat_layers = raw_section.get("cache_concat_layers")
        if concat_layers != []:
            raise RuntimeError(
                f"{section_name} contains a full-history cache concat fallback: {concat_layers!r}"
            )
        inspector_size = raw_section.get("inspector_size_bytes")
        inspector_sha = raw_section.get("inspector_sha256")
        engine_sha = raw_section.get("engine_sha256")
        if (
            isinstance(inspector_size, bool)
            or not isinstance(inspector_size, int)
            or inspector_size <= 0
            or not isinstance(inspector_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", inspector_sha) is None
            or not isinstance(engine_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", engine_sha) is None
        ):
            raise RuntimeError(f"{section_name} has incomplete source-bound inspector identity")

        for tensor_name, tensor_record in (*inputs.items(), *outputs.items()):
            if not isinstance(tensor_record, Mapping):
                raise RuntimeError(
                    f"{section_name} tensor {tensor_name!r} inspector evidence is not an object"
                )

        expected_token_shape = [-1] if expected_role == "prefill" else [1]
        token_shape = inputs["token_id"].get("shape")
        position_shape = inputs["position_id"].get("shape")
        if token_shape != expected_token_shape:
            raise RuntimeError(
                f"{section_name} token_id shape is {token_shape!r}, "
                f"expected {expected_token_shape}"
            )
        if position_shape != expected_token_shape:
            raise RuntimeError(
                f"{section_name} position_id shape is {position_shape!r}, "
                f"expected {expected_token_shape}"
            )
        token_profiles = inputs["token_id"].get("profiles")
        if not isinstance(token_profiles, list) or len(token_profiles) != expected_profile_count:
            raise RuntimeError(f"{section_name} has incomplete token profile evidence")
        if inputs["position_id"].get("profiles") != token_profiles:
            raise RuntimeError(
                f"{section_name} token_id and position_id profiles are not identical"
            )
        for profile_index, profile in enumerate(token_profiles):
            if not isinstance(profile, Mapping):
                raise RuntimeError(f"{section_name} token profile {profile_index} is invalid")
            minimum = profile.get("min")
            optimum = profile.get("opt")
            maximum = profile.get("max")
            if expected_role == "decode":
                if minimum != [1] or optimum != [1] or maximum != [1]:
                    raise RuntimeError(
                        f"{section_name} decode profile {profile_index} is not fixed to Sq=1"
                    )
            elif (
                minimum != [1]
                or not isinstance(optimum, list)
                or len(optimum) != 1
                or not 1 <= optimum[0] <= expected_chunk
                or maximum != [expected_chunk]
            ):
                raise RuntimeError(
                    f"{section_name} prefill profile does not cover Sq=[1,{expected_chunk}]"
                )

        history = inputs["history_length"]
        if history.get("shape") != [1]:
            raise RuntimeError(f"{section_name} history_length is not a scalar [1] tensor")
        history_profiles = history.get("profiles")
        expected_history_profile = {"min": [1], "opt": [1], "max": [1]}
        if not isinstance(history_profiles, list) or len(history_profiles) != expected_profile_count:
            raise RuntimeError(f"{section_name} has incomplete history_length profile evidence")
        for profile_index, profile in enumerate(history_profiles):
            if profile != expected_history_profile:
                raise RuntimeError(
                    f"{section_name} history_length profile {profile_index} "
                    "is not fixed to scalar [1]"
                )

        logits_shape = outputs["logits"].get("shape")
        if logits_shape != [1, expected_vocab_size]:
            raise RuntimeError(
                f"{section_name} logits shape is {logits_shape!r}, "
                f"expected [1, {expected_vocab_size}]"
            )

        for layer in range(num_layers):
            for value_name in ("k", "v"):
                cache_name = f"cache_{value_name}_{layer}"
                present_name = f"present_{value_name}_{layer}"
                cache_shape = inputs[cache_name].get("shape")
                present_shape = outputs[present_name].get("shape")
                if cache_shape != [-1, expected_kv_width]:
                    raise RuntimeError(
                        f"{section_name} {cache_name} does not use the "
                        f"source-bound KV width {expected_kv_width}"
                    )
                expected_present_rows = -1 if expected_role == "prefill" else 1
                if present_shape != [
                    expected_present_rows,
                    expected_kv_width,
                ]:
                    raise RuntimeError(
                        f"{section_name} {present_name} is not exact-Sq current-row "
                        f"output at source-bound KV width {expected_kv_width}"
                    )
                cache_profiles = inputs[cache_name].get("profiles")
                if (
                    not isinstance(cache_profiles, list)
                    or len(cache_profiles) != expected_profile_count
                ):
                    raise RuntimeError(
                        f"{section_name} {cache_name} profile evidence is incomplete"
                    )
                for profile_index, profile in enumerate(cache_profiles):
                    if not isinstance(profile, Mapping):
                        raise RuntimeError(
                            f"{section_name} {cache_name} profile {profile_index} is invalid"
                        )
                    minimum = profile.get("min")
                    optimum = profile.get("opt")
                    maximum = profile.get("max")
                    width = expected_kv_width
                    if expected_role == "decode":
                        bucket = expected_buckets[profile_index]
                        if (
                            minimum != [1, width]
                            or optimum != [bucket, width]
                            or maximum != [bucket, width]
                        ):
                            raise RuntimeError(
                                f"{section_name} {cache_name} profile "
                                f"{profile_index} does not bind bucket {bucket}"
                            )
                    elif (
                        minimum != [1, width]
                        or not isinstance(optimum, list)
                        or len(optimum) != 2
                        or not 1 <= optimum[0] <= spec.context_limit
                        or optimum[1] != width
                        or maximum != [spec.context_limit, width]
                    ):
                        raise RuntimeError(
                            f"{section_name} {cache_name} does not cover the "
                            f"model history limit {spec.context_limit}"
                        )

        summary["engine_sections"][section_name] = dict(raw_section)
    if sections["prefill_engine_plan"].get("engine_sha256") == sections["engine_plan"].get(
        "engine_sha256"
    ):
        raise RuntimeError(
            "prefill_engine_plan and engine_plan have the same serialized engine identity"
        )
    return summary


def inspect_qualified_bundle_engines(
    bundle: Path,
    header: Mapping[str, Any],
    spec: ModelSpec,
    output_dir: Path,
    *,
    artifact_prefix: str = "",
    chunk_limit: int | None = None,
    buckets: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Deserialize and inspect the exact qualified bundle plans.

    Runtime-stack validation happens before either plan is deserialized. The
    resulting evidence is tied to the serialized section SHA, contains actual
    TensorRT I/O/profile metadata, and retains the engine-inspector JSON.
    """

    from tensorrt_model_connect import trt_compat
    from tensorrt_model_connect.trt_plugins import (
        query_runtime_kv_plugin_stack,
    )

    contract = header.get("runtime_memory")
    if not isinstance(contract, Mapping):
        raise RuntimeError("qualified bundle has no runtime_memory contract")
    expected_stack = contract.get("qualified_runtime_stack")
    actual_stack = query_runtime_kv_plugin_stack()
    if not isinstance(expected_stack, Mapping) or actual_stack != dict(expected_stack):
        raise RuntimeError(
            "live runtime stack does not match the qualified bundle before "
            f"engine inspection: expected={expected_stack!r}, "
            f"actual={actual_stack!r}"
        )

    num_layers = header.get("num_layers")
    if isinstance(num_layers, bool) or not isinstance(num_layers, int):
        raise RuntimeError("qualified bundle num_layers is invalid")
    model_contract = _graph_model_contract_from_bundle_header(header)
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    if runtime is None:
        raise RuntimeError("TensorRT runtime creation failed for engine inspection")

    collected: dict[str, Any] = {
        "runtime_stack": actual_stack,
        "model_contract": model_contract,
        "engine_sections": {},
    }
    try:
        for section_name in ("prefill_engine_plan", "engine_plan"):
            plan = _read_bundle_section(bundle, header, section_name)
            engine = runtime.deserialize_cuda_engine(plan)
            if engine is None:
                raise RuntimeError(
                    f"TensorRT failed to deserialize {section_name} for qualified engine inspection"
                )
            inputs: dict[str, Any] = {}
            outputs: dict[str, Any] = {}
            for tensor_index in range(int(engine.num_io_tensors)):
                name = str(engine.get_tensor_name(tensor_index))
                mode = engine.get_tensor_mode(name)
                record: dict[str, Any] = {
                    "shape": _shape_list(engine.get_tensor_shape(name)),
                }
                if mode == trt.TensorIOMode.INPUT:
                    profiles = []
                    for profile_index in range(int(engine.num_optimization_profiles)):
                        minimum, optimum, maximum = engine.get_tensor_profile_shape(
                            name,
                            profile_index,
                        )
                        profiles.append(
                            {
                                "min": _shape_list(minimum),
                                "opt": _shape_list(optimum),
                                "max": _shape_list(maximum),
                            }
                        )
                    record["profiles"] = profiles
                    inputs[name] = record
                elif mode == trt.TensorIOMode.OUTPUT:
                    outputs[name] = record
                else:
                    raise RuntimeError(f"{section_name} tensor {name!r} has unknown I/O mode")

            inspector = engine.create_engine_inspector()
            if inspector is None:
                raise RuntimeError(f"TensorRT did not create an inspector for {section_name}")
            raw_inspector = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
            if not isinstance(raw_inspector, str) or not raw_inspector.strip():
                raise RuntimeError(f"TensorRT returned no inspector JSON for {section_name}")
            try:
                inspector_json = json.loads(raw_inspector)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"TensorRT returned invalid inspector JSON for {section_name}: {exc}"
                ) from exc
            inspector_path = output_dir / (f"{artifact_prefix}{section_name}.engine-inspector.json")
            inspector_path.write_text(
                raw_inspector.rstrip() + "\n",
                encoding="utf-8",
            )
            inspector_bytes = inspector_path.read_bytes()
            layer_indices = sorted(
                {
                    int(index)
                    for index in re.findall(
                        r"layer\.(\d+)\.attn\."
                        r"NativeContiguousAttentionV2",
                        raw_inspector,
                    )
                }
            )
            collected["engine_sections"][section_name] = {
                "engine_sha256": hashlib.sha256(plan).hexdigest(),
                "num_optimization_profiles": int(engine.num_optimization_profiles),
                "inputs": inputs,
                "outputs": outputs,
                "native_contiguous_attention_layer_indices": layer_indices,
                "dense_attention_layers": _dense_attention_layers(inspector_json),
                "cache_concat_layers": _cache_concat_layers(inspector_json),
                "inspector_path": str(inspector_path),
                "inspector_size_bytes": len(inspector_bytes),
                "inspector_sha256": hashlib.sha256(inspector_bytes).hexdigest(),
            }
            del inspector
            del engine
    finally:
        del runtime
        gc.collect()

    return _validate_qualified_engine_graph_evidence(
        collected,
        spec,
        num_layers=num_layers,
        chunk_limit=chunk_limit,
        buckets=buckets,
        expected_runtime_stack=expected_stack,
    )


def _validate_chunk_variant(
    base_header: dict[str, Any],
    variant_header: dict[str, Any],
    spec: ModelSpec,
) -> int:
    base = base_header["runtime_memory"]
    variant = variant_header.get("runtime_memory")
    if not isinstance(variant, dict):
        raise ValueError("developer chunk-variant bundle has no runtime_memory contract")
    invariant_fields = (
        "contract_version",
        "qualified_model_id",
        "qualified_model_revision",
        "qualified_config_sha256",
        "qualified_target",
        "qualified_runtime_stack",
        "native_kv_plugin_abi",
        "model_context_limit",
        "kv_layout",
        "kv_dtype",
        "kv_bytes_per_token",
        "runtime_owned",
    )
    mismatches = {
        field: {"base": base.get(field), "variant": variant.get(field)}
        for field in invariant_fields
        if base.get(field) != variant.get(field)
    }
    if base_header.get("vocab_size") != variant_header.get("vocab_size"):
        mismatches["vocab_size"] = {
            "base": base_header.get("vocab_size"),
            "variant": variant_header.get("vocab_size"),
        }
    if mismatches:
        raise ValueError(
            f"developer chunk-variant bundle changes qualified model facts: {mismatches}"
        )
    expected_chunk = spec.chunk_limit // 2
    actual_chunk = variant.get("prefill_chunk_limit")
    if actual_chunk != expected_chunk:
        raise ValueError(
            "developer chunk-variant bundle must use exactly C/2: "
            f"expected {expected_chunk}, got {actual_chunk!r}"
        )
    buckets = variant.get("active_kv_profile_limits")
    expected_buckets = sorted({*spec.buckets, expected_chunk})
    if buckets != expected_buckets:
        raise ValueError(
            "developer chunk-variant active_kv_profile_limits must equal "
            f"the canonical C/2 buckets: expected {expected_buckets}, "
            f"got {buckets!r}"
        )
    return expected_chunk


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _validate_file_identity(
    identity: Any,
    expected_path: Path,
    *,
    label: str,
    require_binary_metadata: bool = False,
) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise ValueError(f"{label} identity is missing")
    raw_path = identity.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} path identity is missing")
    declared_path = Path(raw_path).expanduser()
    if not declared_path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    actual_path = declared_path.resolve()
    if str(actual_path) != raw_path:
        raise ValueError(f"{label} path must be canonical")
    expected_path = expected_path.expanduser().resolve()
    if actual_path != expected_path:
        raise ValueError(f"{label} path mismatch: expected {expected_path}, got {actual_path}")
    if not actual_path.is_file():
        raise ValueError(f"{label} does not exist: {actual_path}")

    try:
        fd = os.open(
            actual_path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as exc:
        raise ValueError(f"{label} cannot be reopened: {exc}") from exc
    try:
        before = os.fstat(fd)
        digest = hashlib.sha256()
        offset = 0
        while True:
            chunk = os.pread(fd, 8 * 1024 * 1024, offset)
            if not chunk:
                break
            digest.update(chunk)
            offset += len(chunk)
        after = os.fstat(fd)
        endpoint = actual_path.stat()
    finally:
        os.close(fd)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(before, field) != getattr(after, field)
        or getattr(before, field) != getattr(endpoint, field)
        for field in stable_fields
    ):
        raise ValueError(f"{label} changed while its identity was reopened")

    actual_size = before.st_size
    if (
        isinstance(identity.get("size_bytes"), bool)
        or identity.get("size_bytes") != actual_size
        or actual_size <= 0
    ):
        raise ValueError(f"{label} size identity mismatch")
    actual_sha = digest.hexdigest()
    if identity.get("sha256") != actual_sha:
        raise ValueError(f"{label} SHA-256 identity mismatch")
    validated = {
        "path": str(actual_path),
        "size_bytes": actual_size,
        "sha256": actual_sha,
    }
    if require_binary_metadata:
        observed = {
            "device": before.st_dev,
            "inode": before.st_ino,
            "mtime_ns": before.st_mtime_ns,
            "ctime_ns": before.st_ctime_ns,
        }
        for field, value in observed.items():
            declared = identity.get(field)
            if (
                isinstance(declared, bool)
                or not isinstance(declared, int)
                or declared != value
            ):
                raise ValueError(f"{label} {field} identity mismatch")
        validated = {
            "path": str(actual_path),
            "device": before.st_dev,
            "inode": before.st_ino,
            "size_bytes": actual_size,
            "mtime_ns": before.st_mtime_ns,
            "ctime_ns": before.st_ctime_ns,
            "sha256": actual_sha,
        }
        if set(validated) != set(_BINARY_IDENTITY_FIELDS):
            raise AssertionError("binary identity field definition drifted")
    return validated


def _validate_runtime_kv_plugin_mapping(
    evidence: Any,
    plugin_identity: Mapping[str, Any],
) -> dict[str, Any]:
    label = "developer chunk-variant runtime-KV plugin mapping"
    if not isinstance(evidence, Mapping):
        raise ValueError(f"{label} evidence is missing")
    expected_mapping = {
        "path": plugin_identity["path"],
        "device": plugin_identity["device"],
        "inode": plugin_identity["inode"],
    }
    expected = {
        "schema_version": 1,
        "source": "/proc/self/maps",
        "selection_rule": (
            "selected_path_or_same_basename_or_exported_abi_symbol"
        ),
        "abi_symbol": RUNTIME_KV_PLUGIN_ABI_SYMBOL,
        "candidate_count": 1,
        "deleted_candidate_count": 0,
        "selected": dict(plugin_identity),
        "candidate_mappings": [expected_mapping],
    }
    pid = evidence.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError(f"{label} PID is invalid")
    mismatches = {
        field: {"expected": value, "actual": evidence.get(field)}
        for field, value in expected.items()
        if evidence.get(field) != value
    }
    if mismatches:
        raise ValueError(f"{label} does not prove one exact DSO: {mismatches}")
    return {
        **expected,
        "pid": pid,
    }


def _validate_chunk_variant_build_receipt(
    *,
    receipt_path: Path,
    variant_bundle: Path,
    base_header: Mapping[str, Any],
    variant_header: Mapping[str, Any],
    spec: ModelSpec,
    source_state: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    receipt = _read_json_object(receipt_path, "developer chunk-variant build receipt")
    if receipt.get("schema_version") != CHUNK_VARIANT_BUILD_SCHEMA:
        raise ValueError("developer chunk-variant build receipt has an unexpected schema")
    required_flags = {
        "developer_only": True,
        "fresh_build": True,
        "artifact_reused": False,
        "source_state_unchanged": True,
    }
    for field, expected in required_flags.items():
        if receipt.get(field) is not expected:
            raise ValueError(
                f"developer chunk-variant build receipt does not prove {field}={expected}"
            )
    if receipt.get("opt_in") != {
        "environment": DEVELOPER_CHUNK_VARIANT_ENV,
        "value": DEVELOPER_CHUNK_VARIANT_VALUE,
    }:
        raise ValueError("developer chunk-variant build receipt has the wrong opt-in")
    if receipt.get("builder_entrypoint") != (
        "tensorrt_model_connect.engine_builder._build_native_impl_qualified"
    ):
        raise ValueError("developer chunk-variant build receipt used an unexpected builder")

    bundle_identity = _validate_file_identity(
        receipt.get("bundle"),
        variant_bundle,
        label="developer chunk-variant bundle",
    )
    producer_path = REPO_ROOT / "tools" / "build_native_dynamic_memory_chunk_variant.py"
    producer_identity = _validate_file_identity(
        receipt.get("producer"),
        producer_path,
        label="developer chunk-variant producer",
    )
    runtime_kv_plugin = receipt.get("runtime_kv_plugin")
    if not isinstance(runtime_kv_plugin, Mapping):
        raise ValueError(
            "developer chunk-variant runtime-KV plugin identity is missing"
        )
    runtime_kv_plugin_path = Path(
        str(runtime_kv_plugin.get("path", ""))
    )
    runtime_kv_plugin_identity = _validate_file_identity(
        runtime_kv_plugin,
        runtime_kv_plugin_path,
        label="developer chunk-variant runtime-KV plugin",
        require_binary_metadata=True,
    )
    runtime_kv_plugin_mapping = _validate_runtime_kv_plugin_mapping(
        receipt.get("runtime_kv_plugin_mapping"),
        runtime_kv_plugin_identity,
    )
    build_manifest = receipt.get("build_manifest")
    expected_manifest_fields = {
        "path",
        "sha256",
        "schema_version",
        "git_head",
        "source_state_sha256",
        "build_artifacts_sha256",
    }
    if (
        not isinstance(build_manifest, Mapping)
        or set(build_manifest) != expected_manifest_fields
    ):
        raise ValueError(
            "developer chunk-variant build manifest binding is required"
        )
    manifest_path = Path(str(build_manifest.get("path", "")))
    try:
        canonical_manifest = manifest_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"developer chunk-variant build manifest is unreadable: {exc}"
        ) from exc
    if canonical_manifest != manifest_path:
        raise ValueError(
            "developer chunk-variant build manifest path must be canonical"
        )
    manifest_tool = REPO_ROOT / "tools" / (
        "capture_dynamic_memory_test_manifest.py"
    )
    spec_loader = importlib.util.spec_from_file_location(
        "_trtmc_chunk_variant_manifest_replay", manifest_tool
    )
    if spec_loader is None or spec_loader.loader is None:
        raise ValueError("cannot load exact-head build manifest validator")
    manifest_module = importlib.util.module_from_spec(spec_loader)
    sys.modules[spec_loader.name] = manifest_module
    spec_loader.loader.exec_module(manifest_module)
    try:
        manifest = manifest_module.load_and_validate_build_manifest(
            canonical_manifest
        )
    except Exception as exc:
        if exc.__class__.__name__ != "ManifestError":
            raise
        raise ValueError(
            f"developer chunk-variant build manifest replay failed: {exc}"
        ) from exc
    if Path(str(manifest.get("repo_root", ""))).resolve() != REPO_ROOT:
        raise ValueError(
            "developer chunk-variant build manifest belongs to a different "
            "source tree"
        )
    source_pre = manifest["source_state_pre"]
    actual_manifest_binding = {
        "path": str(canonical_manifest),
        "sha256": _sha256(canonical_manifest),
        "schema_version": BUILD_MANIFEST_SCHEMA,
        "git_head": source_pre["git_head"],
        "source_state_sha256": source_pre["source_state_sha256"],
        "build_artifacts_sha256": manifest[
            "build_artifacts_sha256"
        ],
    }
    if dict(build_manifest) != actual_manifest_binding:
        raise ValueError(
            "developer chunk-variant build manifest binding changed"
        )
    manifest_plugin = manifest["build_artifacts"]["runtime_kv_plugin"]
    manifest_plugin_identity = {
        "path": manifest_plugin["path"],
        "device": manifest_plugin["st_dev"],
        "inode": manifest_plugin["st_ino"],
        "size_bytes": manifest_plugin["size_bytes"],
        "mtime_ns": manifest_plugin["mtime_ns"],
        "ctime_ns": manifest_plugin["ctime_ns"],
        "sha256": manifest_plugin["sha256"],
    }
    if manifest_plugin_identity != runtime_kv_plugin_identity:
        raise ValueError(
            "developer chunk-variant runtime-KV plugin does not match the "
            "exact-head build manifest"
        )
    build_manifest_identity = actual_manifest_binding
    build_timing = receipt.get("build_timing")
    if not isinstance(build_timing, Mapping):
        raise ValueError("developer chunk-variant build timing identity is missing")
    timing_path = Path(str(build_timing.get("path", "")))
    timing_identity = _validate_file_identity(
        build_timing,
        timing_path,
        label="developer chunk-variant build timing",
    )

    base_contract = base_header.get("runtime_memory")
    variant_contract = variant_header.get("runtime_memory")
    if not isinstance(base_contract, Mapping) or not isinstance(variant_contract, Mapping):
        raise ValueError("developer chunk-variant bundle contract is missing")
    if receipt.get("runtime_memory") != dict(variant_contract):
        raise ValueError("developer chunk-variant receipt contract does not match bundle")
    expected_qualified_model = {
        "model_id": variant_contract.get("qualified_model_id"),
        "revision": variant_contract.get("qualified_model_revision"),
        "config_sha256": variant_contract.get("qualified_config_sha256"),
        "target": variant_contract.get("qualified_target"),
    }
    qualified_model = receipt.get("qualified_model")
    if not isinstance(qualified_model, Mapping) or any(
        qualified_model.get(field) != value for field, value in expected_qualified_model.items()
    ):
        raise ValueError(
            "developer chunk-variant receipt qualified model does not match the bundle"
        )
    expected_default_policy = {
        "prefill_chunk_limit": spec.chunk_limit,
        "active_kv_profile_limits": list(spec.buckets),
    }
    expected_variant_policy = {
        "prefill_chunk_limit": spec.chunk_limit // 2,
        "active_kv_profile_limits": sorted({*spec.buckets, spec.chunk_limit // 2}),
    }
    if receipt.get("default_policy") != expected_default_policy:
        raise ValueError("developer chunk-variant receipt default policy mismatch")
    if receipt.get("variant_policy") != expected_variant_policy:
        raise ValueError("developer chunk-variant receipt variant policy mismatch")

    receipt_pre = receipt.get("source_state_pre")
    receipt_post = receipt.get("source_state_post")
    if not isinstance(receipt_pre, Mapping) or not isinstance(receipt_post, Mapping):
        raise ValueError("developer chunk-variant receipt source state is incomplete")
    expected_head = source_state.get("git_head")
    expected_source_sha = source_state.get("source_state_sha256")
    if not isinstance(expected_head, str) or not isinstance(expected_source_sha, str):
        raise ValueError("current qualification source state is incomplete")
    for label, snapshot in (
        ("prebuild", receipt_pre),
        ("postbuild", receipt_post),
    ):
        if (
            snapshot.get("git_head") != expected_head
            or snapshot.get("source_state_sha256") != expected_source_sha
        ):
            raise ValueError(
                f"developer chunk-variant {label} source state does not match qualification source"
            )
    if (
        build_manifest_identity["git_head"] != expected_head
        or build_manifest_identity["source_state_sha256"]
        != expected_source_sha
    ):
        raise ValueError(
            "developer chunk-variant build manifest source does not match "
            "qualification source"
        )

    return {
        "path": str(receipt_path),
        "size_bytes": receipt_path.stat().st_size,
        "sha256": _sha256(receipt_path),
        "schema_version": receipt["schema_version"],
        "bundle": bundle_identity,
        "producer": producer_identity,
        "runtime_kv_plugin": runtime_kv_plugin_identity,
        "runtime_kv_plugin_mapping": runtime_kv_plugin_mapping,
        "build_manifest": build_manifest_identity,
        "build_timing": timing_identity,
        "source_state_sha256": expected_source_sha,
        "git_head": expected_head,
    }


def _load_perf_provenance_module() -> Any:
    """Load the shared strict build-manifest/build-receipt validators."""

    module_name = "_trtmc_correctness_base_artifact_provenance"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = Path(__file__).with_name(
        "capture_native_dynamic_memory_perf.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(
            f"cannot load base artifact provenance validator: {path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _validate_base_artifact_binding(
    *,
    build_manifest_path: Path,
    base_build_receipt_path: Path,
    bundle: Path,
    runner: Path,
    spec: ModelSpec,
    source_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind canonical correctness to one clean exact-head binary/bundle set.

    The manifest validator owns the complete binary identity set and the
    performance capture validator owns the fresh no-flag bundle receipt. This
    qualifier composes those existing strict validators and only adds the
    correctness-specific runner/model selection.
    """

    if (
        source_state.get("git_dirty") is not False
        or source_state.get("exact_head_gate_satisfied") is not True
    ):
        raise ValueError(
            "base artifact binding requires a clean exact qualification HEAD"
        )
    perf = _load_perf_provenance_module()
    try:
        manifest_binding, manifest_artifacts = perf._read_build_manifest(
            build_manifest_path
        )
        receipt_path = base_build_receipt_path.expanduser().resolve(
            strict=True
        )
        receipt = perf._read_object(
            receipt_path,
            "base native-dynamic build receipt",
        )
        plugin_path = Path(
            manifest_artifacts["runtime_kv_plugin"]["path"]
        )
        receipt_boundaries, _ = perf._validate_build_receipt(
            receipt,
            bundle=bundle.expanduser().resolve(strict=True),
            role="native-dynamic",
            source_state=source_state,
            plugin_library=plugin_path,
        )
    except Exception as exc:
        if exc.__class__.__name__ not in {
            "CaptureError",
            "ManifestError",
        } and not isinstance(
            exc,
            (KeyError, OSError, TypeError, ValueError),
        ):
            raise
        raise ValueError(
            f"invalid canonical base artifact provenance: {exc}"
        ) from exc
    for boundary_name, boundary_state in receipt_boundaries.items():
        if (
            boundary_state.get("git_dirty") is not False
            or boundary_state.get("exact_head_gate_satisfied") is not True
            or boundary_state.get("git_head")
            != source_state.get("git_head")
            or boundary_state.get("source_state_sha256")
            != source_state.get("source_state_sha256")
        ):
            raise ValueError(
                "base build receipt "
                f"{boundary_name} is not the current clean exact HEAD"
            )

    if receipt.get("build_manifest") != manifest_binding:
        raise ValueError(
            "base build receipt and --build-manifest identify different "
            "exact-head builds"
        )
    manifest_document = _read_json_object(
        Path(manifest_binding["path"]),
        "validated exact-head build manifest",
    )
    raw_artifacts = manifest_document.get("build_artifacts")
    raw_backend = (
        raw_artifacts.get("trt_backend")
        if isinstance(raw_artifacts, Mapping)
        else None
    )
    backend_relative = (
        raw_backend.get("relative_path")
        if isinstance(raw_backend, Mapping)
        else None
    )
    if (
        not isinstance(backend_relative, str)
        or re.fullmatch(
            r"libtrtmc_backend_trt_[0-9]+_[0-9]+\.so",
            backend_relative,
        )
        is None
    ):
        raise ValueError(
            "build manifest does not identify an active versioned TensorRT "
            "backend"
        )
    try:
        active_backend_path = (
            Path(str(manifest_document["build_dir"]))
            / backend_relative
        ).absolute()
        active_backend_stat = active_backend_path.stat()
        canonical_backend = Path(
            manifest_artifacts["trt_backend"]["path"]
        )
        canonical_backend_stat = canonical_backend.stat()
    except (KeyError, OSError, TypeError) as exc:
        raise ValueError(
            f"active versioned TensorRT backend cannot be reopened: {exc}"
        ) from exc
    if (
        active_backend_path.resolve(strict=True) != canonical_backend
        or active_backend_stat.st_dev != canonical_backend_stat.st_dev
        or active_backend_stat.st_ino != canonical_backend_stat.st_ino
    ):
        raise ValueError(
            "active versioned TensorRT backend no longer resolves to the "
            "manifest identity"
        )

    bundle = bundle.expanduser().resolve(strict=True)
    runner = runner.expanduser().resolve(strict=True)
    header = _read_bundle_header(bundle)
    resolved_spec = _resolve_spec(header)
    if resolved_spec != spec:
        raise ValueError(
            "base build receipt is being applied to a different qualified model"
        )
    contract = header["runtime_memory"]
    expected_model = {
        "model_id": spec.model_id,
        "model_revision": contract.get("qualified_model_revision"),
        "precision": header.get("precision"),
        "target": contract.get("qualified_target"),
        "bundle_build_id": receipt.get("bundle_build_id"),
    }
    for field in (
        "model_revision",
        "precision",
        "target",
        "bundle_build_id",
    ):
        value = expected_model[field]
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"qualified base bundle/receipt has no valid {field}"
            )
    receipt_model = {
        field: receipt.get(field)
        for field in expected_model
    }
    if receipt_model != expected_model:
        raise ValueError(
            "base build receipt model tuple does not match the supplied "
            f"bundle: expected={expected_model!r}, actual={receipt_model!r}"
        )
    if header.get("model_id") != spec.model_id:
        raise ValueError(
            "base bundle top-level model_id does not match its qualified "
            "runtime-memory contract"
        )

    runner_identity = perf._file_identity(
        runner,
        label="dynamic-memory qualifier runner",
    )
    try:
        perf._require_manifest_artifact_match(
            manifest_artifacts,
            "qualify",
            runner_identity,
            where="dynamic-memory qualifier runner",
        )
    except Exception as exc:
        if exc.__class__.__name__ != "CaptureError":
            raise
        raise ValueError(
            f"invalid canonical base artifact provenance: {exc}"
        ) from exc

    model_key = (
        "model_qwen"
        if spec.model_id == "Qwen/Qwen3-0.6B"
        else "model_llama"
    )
    selected_artifacts = {
        "benchmark_worker": manifest_artifacts["benchmark_worker"],
        "core": manifest_artifacts["core"],
        "trt_backend": manifest_artifacts["trt_backend"],
        "runtime_kv_plugin": manifest_artifacts[
            "runtime_kv_plugin"
        ],
    }
    return {
        "schema_version": BASE_ARTIFACT_BINDING_SCHEMA,
        "build_manifest": manifest_binding,
        "base_build_receipt": perf._file_identity(
            receipt_path,
            label="base native-dynamic build receipt",
        ),
        "bundle": perf._file_identity(
            bundle,
            label="base native-dynamic bundle",
        ),
        "qualifier_runner": runner_identity,
        "benchmark_worker": selected_artifacts["benchmark_worker"],
        "core": selected_artifacts["core"],
        "trt_backend": {
            "active_versioned_path": str(active_backend_path),
            "identity": selected_artifacts["trt_backend"],
        },
        "model_plugin": {
            "artifact_key": model_key,
            "identity": manifest_artifacts[model_key],
        },
        "runtime_kv_plugin": selected_artifacts[
            "runtime_kv_plugin"
        ],
        "source": {
            "git_head": source_state["git_head"],
            "source_state_sha256": source_state[
                "source_state_sha256"
            ],
            "git_dirty": False,
            "exact_head_gate_satisfied": True,
        },
        "qualified_model": expected_model,
    }


def _base_artifact_binding_passed(
    evidence: Any,
    *,
    bundle: Path,
    runner: Path,
    spec: ModelSpec,
    source_state: Mapping[str, Any],
) -> bool:
    """Reopen the base manifest/receipt and all bound artifacts at promotion."""

    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != _BASE_ARTIFACT_BINDING_FIELDS
        or evidence.get("schema_version")
        != BASE_ARTIFACT_BINDING_SCHEMA
    ):
        return False
    manifest = evidence.get("build_manifest")
    receipt = evidence.get("base_build_receipt")
    if not isinstance(manifest, Mapping) or not isinstance(
        receipt, Mapping
    ):
        return False
    try:
        replayed = _validate_base_artifact_binding(
            build_manifest_path=Path(str(manifest.get("path", ""))),
            base_build_receipt_path=Path(
                str(receipt.get("path", ""))
            ),
            bundle=bundle,
            runner=runner,
            spec=spec,
            source_state=source_state,
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return replayed == dict(evidence)


def _bind_runtime_kv_plugin_from_base_artifacts(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Select the manifest DSO before Python imports any TRT plugin loader."""

    if "tensorrt_model_connect.trt_plugins" in sys.modules:
        raise ValueError(
            "runtime-KV plugin loader was imported before canonical DSO "
            "binding"
        )
    plugin = evidence.get("runtime_kv_plugin")
    if not isinstance(plugin, Mapping):
        raise ValueError(
            "base artifact binding has no runtime-KV plugin identity"
        )
    selected_path = Path(str(plugin.get("path", "")))
    selected = _validate_file_identity(
        plugin,
        selected_path,
        label="base artifact runtime-KV plugin",
        require_binary_metadata=True,
    )
    perf = _load_perf_provenance_module()
    try:
        mapped_candidates = perf._runtime_kv_mapping_candidates(
            perf._mapped_library_records(os.getpid()),
            selected=selected,
        )
    except Exception as exc:
        if exc.__class__.__name__ != "CaptureError":
            raise
        raise ValueError(
            f"cannot validate preloaded runtime-KV plugin mappings: {exc}"
        ) from exc
    if mapped_candidates:
        if len(mapped_candidates) != 1:
            raise ValueError(
                "process already maps multiple runtime-KV plugin DSOs "
                f"before canonical binding: {mapped_candidates!r}"
            )
        mapped = mapped_candidates[0]
        if (
            mapped.get("deleted") is not False
            or mapped.get("path") != selected["path"]
            or mapped.get("device") != selected["device"]
            or mapped.get("inode") != selected["inode"]
        ):
            raise ValueError(
                "process already maps a different runtime-KV plugin than "
                "the exact build manifest"
            )
        preload_mapping: dict[str, Any] | None = dict(mapped)
    else:
        preload_mapping = None
    environment_was_set = RUNTIME_KV_PLUGIN_ENV in os.environ
    if environment_was_set:
        raw = os.environ[RUNTIME_KV_PLUGIN_ENV]
        if not raw:
            raise ValueError(
                f"{RUNTIME_KV_PLUGIN_ENV} was explicitly set but is empty"
            )
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        try:
            candidate = candidate.resolve(strict=True)
        except OSError as exc:
            raise ValueError(
                f"{RUNTIME_KV_PLUGIN_ENV} cannot be resolved: {exc}"
            ) from exc
        if candidate != Path(selected["path"]):
            raise ValueError(
                f"{RUNTIME_KV_PLUGIN_ENV} selects a different runtime-KV "
                "plugin than the exact build manifest"
            )
        _validate_file_identity(
            selected,
            candidate,
            label=f"{RUNTIME_KV_PLUGIN_ENV} runtime-KV plugin",
            require_binary_metadata=True,
        )
    else:
        os.environ[RUNTIME_KV_PLUGIN_ENV] = selected["path"]
    return {
        "schema_version": RUNTIME_KV_PLUGIN_BINDING_SCHEMA,
        "environment": RUNTIME_KV_PLUGIN_ENV,
        "environment_was_set": environment_was_set,
        "preload_mapping": preload_mapping,
        "selected": selected,
        "loaded_mapping": None,
    }


def _finalize_runtime_kv_plugin_binding(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the loader mapped the selected inode after stack inspection."""

    if (
        set(evidence) != _RUNTIME_KV_PLUGIN_BINDING_FIELDS
        or evidence.get("schema_version")
        != RUNTIME_KV_PLUGIN_BINDING_SCHEMA
        or evidence.get("loaded_mapping") is not None
    ):
        raise ValueError(
            "runtime-KV plugin pre-load binding evidence is malformed"
        )
    selected = evidence.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError(
            "runtime-KV plugin pre-load binding has no selected identity"
        )
    perf = _load_perf_provenance_module()
    try:
        loaded_mapping = perf._validate_exact_plugin_mapping(
            perf._mapped_library_records(os.getpid()),
            selected=selected,
            where="dynamic-memory qualifier process",
        )
    except Exception as exc:
        if exc.__class__.__name__ != "CaptureError":
            raise
        raise ValueError(
            f"runtime-KV plugin loaded mapping is not exact: {exc}"
        ) from exc
    reopened = _validate_file_identity(
        selected,
        Path(str(selected.get("path", ""))),
        label="loaded runtime-KV plugin",
        require_binary_metadata=True,
    )
    if reopened != dict(selected):
        raise ValueError(
            "runtime-KV plugin changed after it was loaded"
        )
    return {
        **dict(evidence),
        "loaded_mapping": loaded_mapping,
    }


def _persisted_runtime_kv_plugin_binding_passed(
    evidence: Any,
    *,
    base_artifact_binding: Mapping[str, Any] | None,
) -> bool:
    """Validate historical mapping evidence without trusting its gate bool."""

    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != _RUNTIME_KV_PLUGIN_BINDING_FIELDS
        or evidence.get("schema_version")
        != RUNTIME_KV_PLUGIN_BINDING_SCHEMA
        or evidence.get("environment") != RUNTIME_KV_PLUGIN_ENV
        or not isinstance(
            evidence.get("environment_was_set"), bool
        )
        or not isinstance(base_artifact_binding, Mapping)
    ):
        return False
    selected = evidence.get("selected")
    if (
        not isinstance(selected, Mapping)
        or selected
        != base_artifact_binding.get("runtime_kv_plugin")
    ):
        return False
    loaded = evidence.get("loaded_mapping")
    if (
        not isinstance(loaded, Mapping)
        or set(loaded)
        != {
            "path",
            "device",
            "inode",
            "deleted",
            "identity_sha256",
        }
        or loaded.get("deleted") is not False
        or any(
            loaded.get(field) != selected.get(field)
            for field in ("path", "device", "inode")
        )
    ):
        return False
    perf = _load_perf_provenance_module()
    if loaded.get("identity_sha256") != perf._canonical_sha(
        selected
    ):
        return False
    try:
        if (
            _validate_file_identity(
                selected,
                Path(str(selected.get("path", ""))),
                label="persisted runtime-KV plugin",
                require_binary_metadata=True,
            )
            != dict(selected)
        ):
            return False
    except (OSError, TypeError, ValueError):
        return False
    preload = evidence.get("preload_mapping")
    if preload is not None:
        if (
            not isinstance(preload, Mapping)
            or set(preload)
            != {"path", "device", "inode", "deleted"}
            or preload.get("deleted") is not False
            or any(
                preload.get(field) != selected.get(field)
                for field in ("path", "device", "inode")
            )
        ):
            return False
    return True


def _runtime_kv_plugin_binding_passed(
    evidence: Any,
    *,
    base_artifact_binding: Mapping[str, Any] | None,
) -> bool:
    """Replay the environment, file identity, and unique live DSO mapping."""

    if not _persisted_runtime_kv_plugin_binding_passed(
        evidence,
        base_artifact_binding=base_artifact_binding,
    ):
        return False
    assert isinstance(evidence, Mapping)
    selected = evidence["selected"]
    assert isinstance(selected, Mapping)
    raw_environment = os.environ.get(RUNTIME_KV_PLUGIN_ENV)
    if not raw_environment:
        return False
    try:
        environment_path = Path(raw_environment).expanduser()
        if not environment_path.is_absolute():
            environment_path = Path.cwd() / environment_path
        if (
            environment_path.resolve(strict=True)
            != Path(str(selected.get("path", "")))
        ):
            return False
        if (
            _validate_file_identity(
                selected,
                environment_path,
                label="replayed runtime-KV plugin",
                require_binary_metadata=True,
            )
            != dict(selected)
        ):
            return False
        perf = _load_perf_provenance_module()
        replayed_mapping = perf._validate_exact_plugin_mapping(
            perf._mapped_library_records(os.getpid()),
            selected=selected,
            where="replayed dynamic-memory qualifier process",
        )
    except Exception as exc:
        if exc.__class__.__name__ not in {
            "CaptureError",
        } and not isinstance(
            exc,
            (KeyError, OSError, TypeError, ValueError),
        ):
            raise
        return False
    if replayed_mapping != evidence.get("loaded_mapping"):
        return False
    return True


def _decode_profile_index(spec: ModelSpec, history_tokens: int) -> int:
    for index, bucket in enumerate(spec.buckets):
        if bucket >= history_tokens:
            return index
    raise ValueError(
        f"history length {history_tokens} exceeds the terminal decode bucket {spec.buckets[-1]}"
    )


def _split_decode_boundary_cases(spec: ModelSpec) -> tuple[Case, ...]:
    cases: list[Case] = []
    for bucket in spec.buckets[:-1]:
        for label, prompt_tokens in (
            ("p-minus-1", bucket - 1),
            ("p", bucket),
            ("p-plus-1", bucket + 1),
        ):
            profile_index = _decode_profile_index(spec, prompt_tokens)
            cases.append(
                Case(
                    f"decode-bucket-{bucket}-{label}",
                    prompt_tokens,
                    1,
                    expected_decode_profile_ids=(profile_index,),
                    expected_decode_bucket_limits=(spec.buckets[profile_index],),
                )
            )
    return tuple(cases)


def _profile_crossing_cases(spec: ModelSpec) -> tuple[Case, ...]:
    return tuple(
        Case(
            f"profile-crossing-{bucket}",
            bucket,
            2,
            expected_decode_profile_ids=(index, index + 1),
            expected_decode_bucket_limits=(bucket, spec.buckets[index + 1]),
        )
        for index, bucket in enumerate(spec.buckets[:-1])
    )


def _qwen_cases(spec: ModelSpec) -> tuple[Case, ...]:
    c = spec.chunk_limit
    m = spec.context_limit
    base = (
        Case("c-minus-1", c - 1, 0),
        Case("c", c, 0),
        Case("c-plus-1", c + 1, 0),
        Case("two-c-plus-17", 2 * c + 17, 0),
        Case("total-32768", 32_760, 8),
        Case("total-model-limit", m - 8, 8),
        Case("prefill-last-position", m, 0),
        Case("model-limit-plus-1", m + 1, 0, True),
    )
    return base + _split_decode_boundary_cases(spec) + _profile_crossing_cases(spec)


def _tiny_cases(spec: ModelSpec) -> tuple[Case, ...]:
    lengths: list[int] = []
    for bucket in spec.buckets:
        for length in (bucket - 1, bucket, bucket + 1):
            if length > 0 and length not in lengths:
                lengths.append(length)
    cases = [
        Case(f"bucket-boundary-{length}", length, 0)
        for length in lengths
        if length <= spec.context_limit
    ]
    cases.extend(
        (
            Case("total-model-limit", spec.context_limit - 8, 8),
            Case("prefill-last-position", spec.context_limit, 0),
            Case("model-limit-plus-1", spec.context_limit + 1, 0, True),
        )
    )
    cases.extend(_split_decode_boundary_cases(spec))
    cases.extend(_profile_crossing_cases(spec))
    # P+1 for the final bucket is exactly this rejection; keep one canonical
    # name in reports rather than executing the same case twice.
    return tuple(cases)


def _cases_for(spec: ModelSpec) -> tuple[Case, ...]:
    if spec.model_id.startswith("Qwen/"):
        return _qwen_cases(spec)
    return _tiny_cases(spec)


def deterministic_token_ids(length: int, vocab_size: int) -> np.ndarray:
    if length <= 0 or vocab_size <= 1:
        raise ValueError("length and vocab_size must be positive")
    # Integer-only arithmetic makes every prefix byte-identical across
    # machines and across the C/C+1/M cases. Avoid token zero so synthetic
    # padding conventions cannot accidentally affect a future graph.
    positions = np.arange(length, dtype=np.uint64)
    values = (
        positions * np.uint64(48_271)
        + (positions >> np.uint64(3)) * np.uint64(69_621)
        + np.uint64(17)
    )
    return (values % np.uint64(vocab_size - 1) + 1).astype(np.int32)


def read_logits_artifact(path: Path) -> np.ndarray:
    with path.open("rb") as artifact:
        raw_header = artifact.read(LOGITS_HEADER.size)
        if len(raw_header) != LOGITS_HEADER.size:
            raise ValueError("qualification logits artifact has a truncated header")
        magic, version, dtype, rows, columns = LOGITS_HEADER.unpack(raw_header)
        if magic != LOGITS_MAGIC or version != 1 or dtype != 1:
            raise ValueError(
                "qualification logits artifact has an unsupported format "
                f"(magic={magic!r}, version={version}, dtype={dtype})"
            )
        expected_bytes = rows * columns * np.dtype(np.float32).itemsize
        payload = artifact.read()
        if len(payload) != expected_bytes:
            raise ValueError(
                "qualification logits artifact payload size mismatch: "
                f"expected {expected_bytes}, got {len(payload)}"
            )
    return np.frombuffer(payload, dtype="<f4").reshape((rows, columns)).copy()


def _validate_logits_artifact_metadata(
    artifact: Any,
    *,
    role: str,
    expected_rows: int,
    expected_columns: int,
) -> Path:
    if not isinstance(artifact, Mapping):
        raise RuntimeError(f"{role} logits artifact metadata is missing")
    expected = {
        "format": "trtmc-qualification-logits-v1",
        "dtype": "float32",
        "rows": expected_rows,
        "vocab_size": expected_columns,
    }
    for field, value in expected.items():
        if artifact.get(field) != value:
            raise RuntimeError(
                f"{role} logits artifact {field}={artifact.get(field)!r}, "
                f"expected {value!r}"
            )
    raw_path = artifact.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(f"{role} logits artifact has no path")
    path = Path(raw_path)
    if not path.is_absolute() or not path.is_file():
        raise RuntimeError(f"{role} logits artifact path is not an existing absolute file")
    return path


def _validate_admission_trace(trace: Mapping[str, Any], *, label: str) -> None:
    expected = {
        "status": "rejected",
        "error_type": "admission",
        "stage": "before_attention",
        "attention_started": False,
        "prefill_launches": 0,
        "decode_launches": 0,
        "final_kv_position": 0,
        "invocations": [],
        "selected_token_ids": [],
        "step_top1_token_ids": [],
    }
    for key, value in expected.items():
        if trace.get(key) != value:
            raise RuntimeError(
                f"{label}: rejection trace {key}={trace.get(key)!r}, expected {value!r}"
            )
    ledger = trace.get("attention_execution_ledger")
    ledger_fields = {
        "source",
        "available",
        "module_count",
        "before",
        "after",
        "delta",
    }
    if not isinstance(ledger, Mapping) or set(ledger) != ledger_fields:
        raise RuntimeError(
            f"{label}: rejection trace has no exact attention execution ledger"
        )
    module_count = ledger.get("module_count")
    before = ledger.get("before")
    after = ledger.get("after")
    delta = ledger.get("delta")
    if (
        ledger.get("source")
        != "runtime_memory_transfer_snapshot_v1.execution_attempt_events"
        or ledger.get("available") is not True
        or type(module_count) is not int
        or module_count <= 0
        or type(before) is not int
        or before < 0
        or type(after) is not int
        or after < before
        or type(delta) is not int
        or delta != after - before
        or delta != 0
    ):
        raise RuntimeError(
            f"{label}: rejection was not causally proven before any "
            "attention execution attempt"
        )
    forbidden = {
        "lifetime_protocol",
        "load_cycle_warmup",
        "load_cycles",
        "logits_artifact",
        "cold_start_logits_artifact",
        "cold_warm_output_equivalence",
        "runtime_memory_receipt",
        "kv_allocation_id",
    }
    present = sorted(forbidden.intersection(trace))
    if present:
        raise RuntimeError(
            f"{label}: rejection trace contains post-admission evidence: {present}"
        )


def _admission_trace_passed(trace: Any, *, label: str) -> bool:
    if not isinstance(trace, Mapping):
        return False
    try:
        _validate_admission_trace(trace, label=label)
    except RuntimeError:
        return False
    return True


def _write_tokens(path: Path, tokens: np.ndarray) -> None:
    path.write_text("\n".join(str(int(token)) for token in tokens) + "\n", encoding="utf-8")


def _parse_runner_json(stdout: str) -> dict[str, Any]:
    candidates = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not candidates:
        raise RuntimeError("qualification runner produced no JSON output")
    try:
        payload = json.loads(candidates[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"qualification runner's final line is not JSON: {candidates[-1]!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("qualification runner JSON is not an object")
    return payload


def _run_captured_command(
    command: list[str],
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], int]:
    process = subprocess.Popen(
        command,
        env=(dict(environment) if environment is not None else None),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    return (
        subprocess.CompletedProcess(
            command,
            process.returncode,
            stdout=stdout,
            stderr=stderr,
        ),
        process.pid,
    )


def _validate_runner_cuda_visible_device(value: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value
        or (
            re.fullmatch(r"[0-9]+", value) is None
            and _FULL_GPU_UUID_PATTERN.fullmatch(value) is None
        )
    ):
        raise ValueError(
            "runner CUDA-visible device must be one numeric physical GPU "
            "index or one full GPU UUID"
        )
    return value


def _runner_cuda_visible_device_arg(value: str) -> str:
    try:
        return _validate_runner_cuda_visible_device(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _runner_child_environment(
    cuda_visible_device: str,
    *,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    selector = _validate_runner_cuda_visible_device(cuda_visible_device)
    environment = dict(
        os.environ if base_environment is None else base_environment
    )
    environment["CUDA_VISIBLE_DEVICES"] = selector
    return environment


def _normalize_pci_bus_id(value: str) -> str:
    match = re.fullmatch(
        r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])",
        value.strip(),
    )
    if match is None:
        raise RuntimeError(f"invalid NVIDIA PCI bus identity: {value!r}")
    return (
        f"{match.group(1)[-4:].lower()}:{match.group(2).lower()}:"
        f"{match.group(3).lower()}.{match.group(4)}"
    )


def _sampler_trust_anchor(
    *,
    child_pid: int,
    cuda_logical_device_index: int = 0,
    cuda_visible_device: str | None = None,
) -> SamplerTrustAnchor:
    if type(child_pid) is not int or child_pid <= 0:
        raise RuntimeError("qualification producer has no child PID trust anchor")
    inventory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,pci.bus_id,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows: list[tuple[int, str, str]] = []
    for raw_line in inventory.stdout.splitlines():
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) != 3:
            raise RuntimeError(
                f"unexpected nvidia-smi GPU identity row: {raw_line!r}"
            )
        try:
            physical_index = int(fields[0])
        except ValueError as exc:
            raise RuntimeError(
                f"invalid nvidia-smi physical GPU index: {fields[0]!r}"
            ) from exc
        rows.append(
            (
                physical_index,
                _normalize_pci_bus_id(fields[1]),
                fields[2],
            )
        )
    if not rows:
        raise RuntimeError("nvidia-smi reported no full GPU identities")

    explicit_child_selector = cuda_visible_device is not None
    if not explicit_child_selector:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        selectors = (
            [part.strip() for part in visible.split(",") if part.strip()]
            if visible is not None and visible.strip()
            else [str(row[0]) for row in rows]
        )
    else:
        selectors = [
            _validate_runner_cuda_visible_device(cuda_visible_device)
        ]
    if cuda_logical_device_index >= len(selectors):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES does not expose the qualification logical GPU"
        )
    selector = selectors[cuda_logical_device_index]
    selected: tuple[int, str, str] | None = None
    if selector.isdigit():
        requested_index = int(selector)
        selected = next(
            (row for row in rows if row[0] == requested_index),
            None,
        )
    else:
        if explicit_child_selector:
            candidates = [row for row in rows if row[2] == selector]
        else:
            uuid_selector = selector.removeprefix("GPU-")
            candidates = [
                row
                for row in rows
                if row[2] == selector
                or row[2].removeprefix("GPU-").startswith(uuid_selector)
            ]
        if len(candidates) == 1:
            selected = candidates[0]
    if selected is None:
        raise RuntimeError(
            f"cannot resolve CUDA-visible GPU selector {selector!r} "
            "to an independent nvidia-smi identity"
        )
    return SamplerTrustAnchor(
        pid=child_pid,
        cuda_logical_device_index=cuda_logical_device_index,
        physical_device_index=selected[0],
        pci_bus_id=selected[1],
        gpu_uuid=selected[2],
    )


def _sampler_anchor_json(anchor: SamplerTrustAnchor) -> dict[str, Any]:
    return {
        "pid": anchor.pid,
        "cuda_logical_device_index": anchor.cuda_logical_device_index,
        "physical_device_index": anchor.physical_device_index,
        "pci_bus_id": anchor.pci_bus_id,
        "gpu_uuid": anchor.gpu_uuid,
    }


def _capture_file_receipt(path: Path, *, root: Path) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(
            f"runner capture artifact escapes its evidence directory: {resolved}"
        ) from exc
    if not resolved.is_file():
        raise RuntimeError(f"runner capture artifact is missing: {resolved}")
    return {
        "path": relative.as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _write_runner_capture_manifest(
    evidence_dir: Path,
    *,
    child_pid: int,
    sampler_anchor: SamplerTrustAnchor,
    include_logits: bool,
) -> dict[str, Any]:
    artifact_paths = {
        "command": evidence_dir / "command.json",
        "returncode": evidence_dir / "returncode.txt",
        "tokens": evidence_dir / "tokens.txt",
        "runner_stdout": evidence_dir / "runner.stdout.log",
        "runner_stderr": evidence_dir / "runner.stderr.log",
        "normalized_trace": evidence_dir / "runner-trace.json",
    }
    if include_logits:
        artifact_paths.update(
            {
                "measured_logits": evidence_dir / "runner-logits.bin",
                "cold_start_logits": (
                    evidence_dir / "runner-logits.bin.cold-start.bin"
                ),
            }
        )
    manifest = {
        "schema": RUNNER_CAPTURE_SCHEMA,
        "runner_pid": child_pid,
        "sampler_trust_anchor": _sampler_anchor_json(sampler_anchor),
        "artifacts": {
            name: _capture_file_receipt(path, root=evidence_dir)
            for name, path in artifact_paths.items()
        },
    }
    (evidence_dir / "capture-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _read_capture_file(
    evidence_dir: Path,
    receipt: Any,
    *,
    name: str,
) -> tuple[Path, bytes]:
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise RuntimeError(f"runner capture {name} receipt has an invalid schema")
    relative = receipt.get("path")
    size_bytes = receipt.get("size_bytes")
    sha256 = receipt.get("sha256")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or type(size_bytes) is not int
        or size_bytes < 0
        or not isinstance(sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
    ):
        raise RuntimeError(f"runner capture {name} receipt is invalid")
    root = evidence_dir.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"runner capture {name} path escapes its root") from exc
    if not path.is_file():
        raise RuntimeError(f"runner capture {name} file is missing")
    payload = path.read_bytes()
    if (
        len(payload) != size_bytes
        or hashlib.sha256(payload).hexdigest() != sha256
    ):
        raise RuntimeError(f"runner capture {name} file does not match its receipt")
    return path, payload


def _runner_command(
    *,
    runner: Path,
    bundle: Path,
    token_path: Path,
    logits_path: Path,
    case: Case,
    context_limit: int,
) -> list[str]:
    command = [
        str(runner),
        "--bundle",
        str(bundle),
        "--tokens",
        str(token_path),
        "--logits",
        str(logits_path),
        "--max-new-tokens",
        str(case.decode_tokens),
        "--max-sequence-length",
        str(context_limit),
    ]
    if not case.expect_admission_rejection:
        command.append("--warmup-load-cycle")
    return command


def run_trt_case(
    *,
    runner: Path,
    bundle: Path,
    tokens: np.ndarray,
    case: Case,
    context_limit: int,
    evidence_dir: Path,
    runner_cuda_visible_device: str,
) -> tuple[
    dict[str, Any],
    np.ndarray | None,
    str,
    SamplerTrustAnchor,
]:
    evidence_dir.mkdir(parents=True, exist_ok=False)
    token_path = evidence_dir / "tokens.txt"
    logits_path = evidence_dir / "runner-logits.bin"
    _write_tokens(token_path, tokens)
    command = _runner_command(
        runner=runner,
        bundle=bundle,
        token_path=token_path,
        logits_path=logits_path,
        case=case,
        context_limit=context_limit,
    )
    (evidence_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n",
        encoding="utf-8",
    )
    runner_environment = _runner_child_environment(
        runner_cuda_visible_device
    )
    completed, child_pid = _run_captured_command(
        command,
        environment=runner_environment,
    )
    sampler_anchor = _sampler_trust_anchor(
        child_pid=child_pid,
        cuda_logical_device_index=0,
        cuda_visible_device=runner_cuda_visible_device,
    )
    (evidence_dir / "runner.stdout.log").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (evidence_dir / "runner.stderr.log").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    (evidence_dir / "returncode.txt").write_text(
        f"{completed.returncode}\n",
        encoding="utf-8",
    )
    trace = _parse_runner_json(completed.stdout)
    if case.expect_admission_rejection:
        if completed.returncode != 3:
            raise RuntimeError(
                f"{case.name}: expected admission exit 3, got {completed.returncode}; "
                f"trace={trace}; stderr={completed.stderr[-4000:]}"
            )
        _validate_admission_trace(trace, label=case.name)
        if logits_path.exists():
            raise RuntimeError(f"{case.name}: rejected request wrote a logits artifact")
        cold_logits_path = Path(f"{logits_path}.cold-start.bin")
        if cold_logits_path.exists():
            raise RuntimeError(
                f"{case.name}: rejected request wrote a cold-start logits artifact"
            )
        (evidence_dir / "runner-trace.json").write_text(
            json.dumps(trace, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_runner_capture_manifest(
            evidence_dir,
            child_pid=child_pid,
            sampler_anchor=sampler_anchor,
            include_logits=False,
        )
        return trace, None, completed.stderr, sampler_anchor
    if completed.returncode != 0 or trace.get("status") != "ok":
        raise RuntimeError(
            f"{case.name}: runner failed ({completed.returncode}); trace={trace}; "
            f"stderr={completed.stderr[-4000:]}"
        )
    measured_logits = read_logits_artifact(logits_path)
    measured_metadata = trace.get("logits_artifact")
    measured_source = _validate_logits_artifact_metadata(
        measured_metadata,
        role=f"{case.name} measured",
        expected_rows=measured_logits.shape[0],
        expected_columns=measured_logits.shape[1],
    )
    if measured_source.resolve() != logits_path.resolve():
        raise RuntimeError(
            f"{case.name}: measured logits metadata does not bind the requested output path"
        )

    cold_logits_path = Path(f"{logits_path}.cold-start.bin")
    cold_logits = read_logits_artifact(cold_logits_path)
    cold_metadata = trace.get("cold_start_logits_artifact")
    cold_source = _validate_logits_artifact_metadata(
        cold_metadata,
        role=f"{case.name} cold-start",
        expected_rows=cold_logits.shape[0],
        expected_columns=cold_logits.shape[1],
    )
    if cold_source.resolve() != cold_logits_path.resolve():
        raise RuntimeError(
            f"{case.name}: cold-start logits metadata does not bind the requested output path"
        )
    if cold_source.resolve() == measured_source.resolve():
        raise RuntimeError(f"{case.name}: cold and measured logits artifacts are not distinct")
    (evidence_dir / "runner-trace.json").write_text(
        json.dumps(trace, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_runner_capture_manifest(
        evidence_dir,
        child_pid=child_pid,
        sampler_anchor=sampler_anchor,
        include_logits=True,
    )
    return trace, measured_logits, completed.stderr, sampler_anchor


def replay_runner_capture(
    evidence_dir: Path,
    *,
    expected_command: list[str],
    expected_tokens: np.ndarray,
    expected_returncode: int,
    expected_trace: Any,
    case: Case,
    model_spec: ModelSpec,
    trusted_geometry: TrustedRuntimeGeometry,
    expected_sampler: SamplerTrustAnchor,
    expected_lifetime_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    evidence_dir = evidence_dir.resolve()
    manifest_path = evidence_dir / "capture-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("runner capture manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_artifact_names = {
        "command",
        "returncode",
        "tokens",
        "runner_stdout",
        "runner_stderr",
        "normalized_trace",
    }
    if not case.expect_admission_rejection:
        expected_artifact_names.update(
            {"measured_logits", "cold_start_logits"}
        )
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != {
            "schema",
            "runner_pid",
            "sampler_trust_anchor",
            "artifacts",
        }
        or manifest.get("schema") != RUNNER_CAPTURE_SCHEMA
        or type(manifest.get("runner_pid")) is not int
        or manifest.get("runner_pid") != expected_sampler.pid
        or manifest.get("sampler_trust_anchor")
        != _sampler_anchor_json(expected_sampler)
        or not isinstance(manifest.get("artifacts"), Mapping)
        or set(manifest["artifacts"]) != expected_artifact_names
    ):
        raise RuntimeError("runner capture manifest does not bind the trusted run")

    files: dict[str, tuple[Path, bytes]] = {}
    for name in sorted(expected_artifact_names):
        files[name] = _read_capture_file(
            evidence_dir,
            manifest["artifacts"][name],
            name=name,
        )
    try:
        command = json.loads(files["command"][1])
    except json.JSONDecodeError as exc:
        raise RuntimeError("runner capture command is not valid JSON") from exc
    if command != expected_command:
        raise RuntimeError("runner capture command does not match the trusted case")
    expected_returncode_text = f"{expected_returncode}\n".encode()
    if files["returncode"][1] != expected_returncode_text:
        raise RuntimeError("runner capture return code does not match the trusted case")
    expected_token_text = (
        "\n".join(str(int(token)) for token in expected_tokens) + "\n"
    ).encode()
    if files["tokens"][1] != expected_token_text:
        raise RuntimeError("runner capture tokens do not match the trusted case")

    stdout_text = files["runner_stdout"][1].decode("utf-8")
    stdout_trace = _parse_runner_json(stdout_text)
    try:
        normalized_trace = json.loads(files["normalized_trace"][1])
    except json.JSONDecodeError as exc:
        raise RuntimeError("runner normalized trace is not valid JSON") from exc
    if (
        stdout_trace != normalized_trace
        or not isinstance(expected_trace, Mapping)
        or dict(expected_trace) != normalized_trace
    ):
        raise RuntimeError(
            "runner stdout, normalized trace, and report trace are not identical"
        )

    if case.expect_admission_rejection:
        _validate_admission_trace(normalized_trace, label=case.name)
        return {
            "trace": normalized_trace,
            "logits": None,
            "validation_evidence": None,
            "runner_stderr_sha256": hashlib.sha256(
                files["runner_stderr"][1]
            ).hexdigest(),
        }

    measured_path = files["measured_logits"][0]
    cold_path = files["cold_start_logits"][0]
    measured_metadata = normalized_trace.get("logits_artifact")
    cold_metadata = normalized_trace.get("cold_start_logits_artifact")
    if (
        not isinstance(measured_metadata, Mapping)
        or measured_metadata.get("path") != str(measured_path)
        or not isinstance(cold_metadata, Mapping)
        or cold_metadata.get("path") != str(cold_path)
    ):
        raise RuntimeError(
            "runner normalized trace does not bind the captured logits paths"
        )
    measured_logits = read_logits_artifact(measured_path)
    validation_evidence = _validate_trace(
        case,
        model_spec,
        normalized_trace,
        measured_logits,
        expected_chunk_limit=trusted_geometry.prefill_chunk_limit,
        expected_effective_request_limit=int(
            normalized_trace["runtime_memory_receipt"][
                "runtime_kv_capacity_tokens"
            ]
        ),
        expected_lifetime_policy=expected_lifetime_policy,
        trusted_geometry=trusted_geometry,
        expected_sampler=expected_sampler,
        require_nvml_reconciliation=True,
    )
    if validation_evidence is None:
        raise RuntimeError("runner capture replay produced no validation evidence")
    return {
        "trace": normalized_trace,
        "logits": measured_logits,
        "validation_evidence": validation_evidence,
        "runner_stderr_sha256": hashlib.sha256(
            files["runner_stderr"][1]
        ).hexdigest(),
    }


def _read_report_artifact(
    raw_path: Any,
    raw_sha256: Any,
    *,
    output_dir: Path,
    label: str,
) -> tuple[Path, bytes]:
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not isinstance(raw_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", raw_sha256) is None
    ):
        raise RuntimeError(f"{label} report artifact receipt is invalid")
    root = output_dir.resolve()
    path = Path(raw_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} report artifact escapes the output directory") from exc
    if not path.is_file():
        raise RuntimeError(f"{label} report artifact is missing")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != raw_sha256:
        raise RuntimeError(f"{label} report artifact SHA-256 does not match")
    return path, payload


def _load_thresholds(repo_root: Path, spec: ModelSpec) -> dict[str, float]:
    data = json.loads((repo_root / spec.threshold_path).read_text(encoding="utf-8"))
    raw = data.get("threshold_overrides")
    if not isinstance(raw, dict):
        raise ValueError(f"{spec.threshold_path} has no threshold_overrides")
    required = (
        "logit_atol",
        "logit_cosine_p5",
        "logit_rel_l2_p95",
        "stable_margin",
        "stable_top1_match_rate",
        "token_agreement_rate",
        "unstable_topk_hit_rate",
    )
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"{spec.threshold_path} misses thresholds: {missing}")
    return {name: float(raw[name]) for name in required}


def _hf_dtype_name(contract_dtype: str) -> str:
    if contract_dtype == "bfloat16":
        return "bfloat16"
    if contract_dtype == "float16":
        return "float16"
    if contract_dtype == "float32":
        return "float32"
    raise ValueError(f"unsupported runtime KV dtype for HF qualification: {contract_dtype}")


def load_hf_model(
    model_path: str,
    dtype_name: str,
    device: str,
    *,
    revision: str | None,
):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, dtype_name)
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "local_files_only": os.path.isdir(model_path),
    }
    if revision is not None and not os.path.isdir(model_path):
        kwargs["revision"] = revision
    # Prefer the memory-efficient official SDPA implementation. Some older
    # Transformers versions do not accept this keyword, so retry without it.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, attn_implementation="sdpa", **kwargs
        )
    except (TypeError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.to(device)
    model.eval()
    return model


def run_hf_reference(
    model,
    tokens: np.ndarray,
    selected_token_ids: Iterable[int],
    device: str,
) -> np.ndarray:
    import torch

    rows: list[np.ndarray] = []
    input_ids = torch.from_numpy(tokens.astype(np.int64, copy=False)).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True)
        logits = output.logits[:, -1, :]
        rows.append(logits.float().cpu().numpy()[0])
        past = output.past_key_values
        for token in selected_token_ids:
            step = torch.tensor([[int(token)]], dtype=torch.long, device=device)
            output = model(input_ids=step, past_key_values=past, use_cache=True)
            logits = output.logits[:, -1, :]
            rows.append(logits.float().cpu().numpy()[0])
            past = output.past_key_values
    return np.stack(rows)


def compare_logits(
    trt_logits: np.ndarray,
    hf_logits: np.ndarray,
    selected_token_ids: Iterable[int],
    thresholds: dict[str, float],
    *,
    unstable_top_k: int = 5,
) -> dict[str, Any]:
    if trt_logits.shape != hf_logits.shape:
        raise ValueError(f"logit shape mismatch: TRT {trt_logits.shape}, HF {hf_logits.shape}")
    diff = np.abs(trt_logits.astype(np.float64) - hf_logits.astype(np.float64))
    max_abs = diff.max(axis=1)
    mean_abs = diff.mean(axis=1)
    dots = np.sum(trt_logits.astype(np.float64) * hf_logits.astype(np.float64), axis=1)
    norms = np.linalg.norm(trt_logits.astype(np.float64), axis=1) * np.linalg.norm(
        hf_logits.astype(np.float64), axis=1
    )
    cosine = np.divide(dots, norms, out=np.zeros_like(dots), where=norms != 0)
    rel_l2 = np.linalg.norm(diff, axis=1) / np.maximum(
        np.linalg.norm(hf_logits.astype(np.float64), axis=1), 1e-12
    )

    trt_top1 = np.argmax(trt_logits, axis=1)
    hf_top1 = np.argmax(hf_logits, axis=1)
    partitioned = np.partition(hf_logits, kth=-2, axis=1)
    hf_margin = partitioned[:, -1] - partitioned[:, -2]
    stable = hf_margin >= thresholds["stable_margin"]
    stable_rate = float(np.mean(trt_top1[stable] == hf_top1[stable])) if np.any(stable) else 1.0
    unstable = ~stable
    top_k = max(1, min(int(unstable_top_k), hf_logits.shape[1]))
    hf_topk = np.argpartition(hf_logits, kth=-top_k, axis=1)[:, -top_k:]
    unstable_topk_rate = (
        float(
            np.mean(
                np.any(
                    hf_topk[unstable] == trt_top1[unstable, np.newaxis],
                    axis=1,
                )
            )
        )
        if np.any(unstable)
        else 1.0
    )
    selected = np.asarray(tuple(selected_token_ids), dtype=np.int64)
    token_rate = (
        float(np.mean(selected == hf_top1[: selected.size]))
        if selected.size
        else float(trt_top1[0] == hf_top1[0])
    )
    metrics = {
        "max_abs_logit_diff": float(np.max(max_abs)),
        "mean_abs_logit_diff": float(np.mean(mean_abs)),
        "logit_cosine_p5": float(np.percentile(cosine, 5)),
        "logit_rel_l2_p95": float(np.percentile(rel_l2, 95)),
        "stable_top1_match_rate": stable_rate,
        "unstable_topk_hit_rate": unstable_topk_rate,
        "unstable_top_k": top_k,
        "token_agreement_rate": token_rate,
        "trt_top1_token_ids": trt_top1.astype(int).tolist(),
        "hf_top1_token_ids": hf_top1.astype(int).tolist(),
        "hf_top1_margins": hf_margin.astype(float).tolist(),
    }
    gates = {
        "logit_atol": metrics["max_abs_logit_diff"] <= thresholds["logit_atol"],
        "logit_cosine_p5": (metrics["logit_cosine_p5"] >= thresholds["logit_cosine_p5"]),
        "logit_rel_l2_p95": (metrics["logit_rel_l2_p95"] <= thresholds["logit_rel_l2_p95"]),
        "stable_top1_match_rate": (
            metrics["stable_top1_match_rate"] >= thresholds["stable_top1_match_rate"]
        ),
        "unstable_topk_hit_rate": (
            metrics["unstable_topk_hit_rate"] >= thresholds["unstable_topk_hit_rate"]
        ),
        "token_agreement_rate": (
            metrics["token_agreement_rate"] >= thresholds["token_agreement_rate"]
        ),
    }
    # Match the existing Qwen/Llama family comparator: max-absolute error is
    # retained as a diagnostic, not promoted into a new standalone hard gate.
    # Numerical parity is composite (cosine OR relative-L2), then combined
    # with the token-level gates.
    composite_gates = {
        "numerical": gates["logit_cosine_p5"] or gates["logit_rel_l2_p95"],
        "token_level": (
            gates["token_agreement_rate"]
            or (gates["stable_top1_match_rate"] and gates["unstable_topk_hit_rate"])
        ),
    }
    return {
        "passed": all(composite_gates.values()),
        "metrics": metrics,
        "thresholds": thresholds,
        "gates": gates,
        "composite_gates": composite_gates,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_state_provenance(
    repo_root: Path,
    tool_path: Path,
    artifact_dir: Path,
    *,
    label: str,
) -> dict[str, Any]:
    """Record complete dirty-source evidence without calling it exact HEAD."""
    git = ["git", "-c", f"safe.directory={repo_root.resolve()}"]
    exclusions = (
        ":(exclude)artifacts",
        ":(exclude)build",
        ":(exclude)build-*",
    )
    head = subprocess.run(
        [*git, "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    status_bytes = subprocess.run(
        [
            *git,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--",
            ".",
            *exclusions,
        ],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    status = [
        entry.decode("utf-8", errors="surrogateescape")
        for entry in status_bytes.split(b"\0")
        if entry
    ]
    staged_diff = subprocess.run(
        [*git, "diff", "--cached", "--binary", "--", ".", *exclusions],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    unstaged_diff = subprocess.run(
        [*git, "diff", "--binary", "--", ".", *exclusions],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    untracked_paths = [
        entry.decode("utf-8", errors="surrogateescape")
        for entry in subprocess.run(
            [
                *git,
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                ".",
                *exclusions,
            ],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout.split(b"\0")
        if entry
    ]
    untracked_files: list[dict[str, Any]] = []
    for relative in sorted(untracked_paths):
        path = repo_root / relative
        stat = path.lstat()
        if path.is_symlink():
            payload = os.readlink(path).encode("utf-8", errors="surrogateescape")
            kind = "symlink"
        elif path.is_file():
            payload = path.read_bytes()
            kind = "file"
        else:
            raise ValueError(f"untracked source entry is not a file or symlink: {relative}")
        untracked_files.append(
            {
                "path": relative,
                "kind": kind,
                "size_bytes": stat.st_size,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )

    snapshot_dir = artifact_dir / "source-state"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    staged_path = snapshot_dir / f"{label}-staged.patch"
    unstaged_path = snapshot_dir / f"{label}-unstaged.patch"
    untracked_path = snapshot_dir / f"{label}-untracked-files.json"
    staged_path.write_bytes(staged_diff)
    unstaged_path.write_bytes(unstaged_diff)
    untracked_path.write_text(
        json.dumps(untracked_files, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    state_identity = {
        "git_head": head,
        "status": status,
        "staged_diff_sha256": hashlib.sha256(staged_diff).hexdigest(),
        "unstaged_diff_sha256": hashlib.sha256(unstaged_diff).hexdigest(),
        "untracked_files": untracked_files,
    }
    state_sha256 = hashlib.sha256(
        json.dumps(
            state_identity,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "git_head": head,
        "git_dirty": bool(status),
        "source_state_sha256": state_sha256,
        "staged_diff_sha256": state_identity["staged_diff_sha256"],
        "unstaged_diff_sha256": state_identity["unstaged_diff_sha256"],
        "untracked_files": untracked_files,
        "status": status,
        "artifacts": {
            "staged_patch": {
                "path": str(staged_path),
                "sha256": _sha256(staged_path),
                "size_bytes": staged_path.stat().st_size,
            },
            "unstaged_patch": {
                "path": str(unstaged_path),
                "sha256": _sha256(unstaged_path),
                "size_bytes": unstaged_path.stat().st_size,
            },
            "untracked_manifest": {
                "path": str(untracked_path),
                "sha256": _sha256(untracked_path),
                "size_bytes": untracked_path.stat().st_size,
            },
        },
        "qualification_tool": str(tool_path.resolve()),
        "qualification_tool_sha256": _sha256(tool_path.resolve()),
        "exact_head_gate_satisfied": not status,
        "note": (
            "Dirty source is recorded but is not an exact-head promotion "
            "receipt; rebuild after the final commit."
            if status
            else "Qualification ran from a clean exact HEAD."
        ),
    }


def _hf_cache_snapshot_identity(path: Path) -> tuple[str, str] | None:
    absolute = path.resolve()
    parts = absolute.parts
    for index, part in enumerate(parts):
        if part != "snapshots" or index < 1 or index + 2 != len(parts):
            continue
        cache_name = parts[index - 1]
        if not cache_name.startswith("models--"):
            continue
        identity = cache_name.removeprefix("models--").split("--", 1)
        if len(identity) != 2 or not all(identity):
            continue
        return "/".join(identity), parts[index + 1]
    return None


def verify_hf_reference(
    model_ref: str,
    contract: dict[str, Any],
    *,
    remote_revision: str | None,
) -> dict[str, str]:
    expected_id = str(contract["qualified_model_id"])
    expected_revision = str(contract["qualified_model_revision"])
    expected_config_sha = str(contract["qualified_config_sha256"])
    local = Path(model_ref)
    if local.is_dir():
        identity = _hf_cache_snapshot_identity(local)
        if identity != (expected_id, expected_revision):
            raise ValueError(
                "local HF reference must be the exact qualified cache snapshot: "
                f"expected {(expected_id, expected_revision)!r}, got {identity!r}"
            )
        if remote_revision is not None and remote_revision != expected_revision:
            raise ValueError(
                "--model-revision contradicts the local qualified snapshot: "
                f"expected {expected_revision}, got {remote_revision}"
            )
        config_path = local / "config.json"
        if not config_path.is_file():
            raise ValueError(f"qualified HF snapshot is missing config.json: {local}")
        actual_config_sha = _sha256(config_path)
        if actual_config_sha != expected_config_sha:
            raise ValueError(
                "qualified HF config fingerprint mismatch: "
                f"expected {expected_config_sha}, got {actual_config_sha}"
            )
        return {
            "kind": "hf_cache_snapshot",
            "model_id": expected_id,
            "revision": expected_revision,
            "config_sha256": actual_config_sha,
            "path": str(local.resolve()),
        }

    if model_ref != expected_id:
        raise ValueError(f"remote HF reference must be {expected_id!r}, got {model_ref!r}")
    if remote_revision != expected_revision:
        raise ValueError(
            "remote HF qualification requires an explicit immutable "
            f"--model-revision {expected_revision}"
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to verify a remote HF config") from exc
    config_path = Path(
        hf_hub_download(
            repo_id=expected_id,
            filename="config.json",
            revision=expected_revision,
        )
    )
    actual_config_sha = _sha256(config_path)
    if actual_config_sha != expected_config_sha:
        raise ValueError(
            "remote qualified HF config fingerprint mismatch: "
            f"expected {expected_config_sha}, got {actual_config_sha}"
        )
    return {
        "kind": "remote_immutable_revision",
        "model_id": expected_id,
        "revision": expected_revision,
        "config_sha256": actual_config_sha,
    }


def _select_cases(cases: tuple[Case, ...], requested: list[str]) -> tuple[Case, ...]:
    if not requested:
        return cases
    by_name = {case.name: case for case in cases}
    unknown = sorted(set(requested) - by_name.keys())
    if unknown:
        raise ValueError(f"unknown case(s) {unknown}; choices are {sorted(by_name)}")
    return tuple(by_name[name] for name in requested)


def _validate_trace(
    case: Case,
    spec: ModelSpec,
    trace: dict[str, Any],
    logits: np.ndarray,
    *,
    expected_chunk_limit: int | None = None,
    expected_effective_request_limit: int | None = None,
    expected_lifetime_policy: Mapping[str, Any] | None = None,
    trusted_geometry: TrustedRuntimeGeometry | None = None,
    expected_sampler: SamplerTrustAnchor | None = None,
    require_nvml_reconciliation: bool = False,
) -> dict[str, Any] | None:
    geometry = (
        trusted_runtime_geometry(
            spec,
            prefill_chunk_limit=expected_chunk_limit,
        )
        if trusted_geometry is None
        else trusted_geometry
    )
    if (
        geometry.model_context_limit != spec.context_limit
        or geometry.kv_bytes_per_token != spec.kv_bytes_per_token
        or (
            expected_chunk_limit is not None
            and geometry.prefill_chunk_limit != expected_chunk_limit
        )
    ):
        raise RuntimeError(f"{case.name}: trusted runtime geometry is inconsistent")
    chunk_limit = geometry.prefill_chunk_limit
    receipt = trace.get("runtime_memory_receipt")
    if not isinstance(receipt, dict) or int(receipt.get("kv_allocation_id", 0)) <= 0:
        raise RuntimeError(f"{case.name}: missing KV allocation trace")
    allocation_id = int(receipt["kv_allocation_id"])
    reserved = int(receipt.get("runtime_kv_capacity_tokens", 0))
    bytes_per_token = int(receipt.get("kv_bytes_per_token", 0))
    if reserved <= 0 or bytes_per_token <= 0:
        raise RuntimeError(f"{case.name}: receipt is missing R/B accounting")
    effective_request_limit = (
        expected_effective_request_limit
        if expected_effective_request_limit is not None
        else reserved
    )
    expected_launches = math.ceil(case.prompt_tokens / chunk_limit)
    expected = {
        "prompt_tokens": case.prompt_tokens,
        "prefill_chunk_limit": chunk_limit,
        "prefill_launches": expected_launches,
        "decode_launches": case.decode_tokens,
        "final_kv_position": case.prompt_tokens + case.decode_tokens,
        "effective_request_limit": effective_request_limit,
    }
    for name, value in expected.items():
        if trace.get(name) != value:
            raise RuntimeError(f"{case.name}: trace {name}={trace.get(name)!r}, expected {value!r}")
    if logits.shape[0] != case.decode_tokens + 1:
        raise RuntimeError(
            f"{case.name}: expected {case.decode_tokens + 1} logit rows, got {logits.shape[0]}"
        )
    if (
        trace.get("runtime_kv_capacity_tokens") != reserved
        or trace.get("kv_allocation_id") != allocation_id
    ):
        raise RuntimeError(
            f"{case.name}: top-level R/allocation identity does not bind the receipt"
        )
    lifetime_policy = _validate_lifetime_policy(
        {
            "kind": "max_sequence_length",
            "requested_tokens": spec.context_limit,
        }
        if expected_lifetime_policy is None
        else expected_lifetime_policy
    )
    _validate_receipt_policy_binding(
        lifetime_policy,
        receipt,
        trusted_geometry=geometry,
        expected_capacity_tokens=reserved,
        expected_effective_request_limit=effective_request_limit,
    )
    shared_context_bytes = int(receipt.get("context_device_memory_bytes", 0))
    if shared_context_bytes <= 0:
        raise RuntimeError(f"{case.name}: receipt is missing actual-shape context accounting")
    peak_device_bytes = receipt.get("peak_device_bytes")
    if not isinstance(peak_device_bytes, int) or peak_device_bytes < 0:
        raise RuntimeError(f"{case.name}: receipt is missing the sampled device high-water")
    if receipt.get("peak_device_bytes_scope") != "device_wide":
        raise RuntimeError(f"{case.name}: peak_device_bytes is not marked device-wide")
    peak_boundaries = receipt.get("peak_device_sample_boundaries")
    if not isinstance(peak_boundaries, list) or {
        "after_runtime_kv_allocation",
        "after_successful_request_completion",
    } - set(peak_boundaries):
        raise RuntimeError(f"{case.name}: peak receipt is missing load/request sample boundaries")
    if int(receipt.get("peak_device_sample_count", 0)) < 2:
        raise RuntimeError(f"{case.name}: peak receipt has fewer than two lifetime samples")
    validation_evidence: dict[str, Any] | None = None
    if require_nvml_reconciliation:
        lifetime_evidence = validate_warmup_evidence(
            trace,
            measured_logits=logits,
            trusted_geometry=geometry,
            expected_sampler=expected_sampler,
            expected_lifetime_policy=lifetime_policy,
            expected_capacity_tokens=reserved,
            expected_final_kv_position=case.prompt_tokens + case.decode_tokens,
            expected_prompt_tokens=case.prompt_tokens,
            expected_prefill_launches=expected_launches,
            expected_decode_launches=case.decode_tokens,
        )
        validation_evidence = {
            "warmup_evidence": lifetime_evidence,
            "cold_start_evidence": lifetime_evidence,
            "peak_memory_reconciliation": lifetime_evidence[
                "peak_memory_reconciliation"
            ],
        }

    invocations = trace.get("invocations")
    if not isinstance(invocations, list):
        raise RuntimeError(f"{case.name}: missing per-invocation split trace")
    if len(invocations) != expected_launches + case.decode_tokens:
        raise RuntimeError(
            f"{case.name}: expected {expected_launches + case.decode_tokens} "
            f"invocations, got {len(invocations)}"
        )
    expected_history = 0
    total_append_bytes = 0
    stable_base_address: int | None = None
    trace_buckets = tuple(sorted(set((*spec.buckets, chunk_limit))))
    role_plan_ids: dict[str, str] = {}
    role_engine_identities: dict[str, int] = {}
    role_profile_plan_ids: dict[tuple[str, int], str] = {}
    for index, invocation in enumerate(invocations):
        if not isinstance(invocation, dict):
            raise RuntimeError(f"{case.name}: invocation {index} is not an object")
        role = "prefill" if index < expected_launches else "decode"
        begin = expected_history
        query = min(chunk_limit, case.prompt_tokens - begin) if role == "prefill" else 1
        end = begin + query
        active = end
        # T bounds only the read-only history prefix, not the current Sq
        # rows.  Cold prefill uses the explicit one-row sentinel.  Every
        # non-cold invocation binds the first history bucket at or above H.
        # Decode profile selection is based on history H, so it can cross a
        # profile boundary while the current row remains a separate Sq=1
        # segment.
        expected_bound = 1 if begin == 0 else reserved
        if begin != 0:
            for bucket in trace_buckets:
                if bucket >= begin:
                    expected_bound = min(bucket, reserved)
                    break
        expected_fields = {
            "invocation_index": index,
            "role": role,
            "chunk_range": [begin, end],
            "launch_count": 1,
            "kv_allocation_id": allocation_id,
            "H": begin,
            "A": active,
            "T": expected_bound,
            "R": reserved,
            "kv_device_to_host_bytes": 0,
            "kv_append_bytes": query * bytes_per_token,
            "full_history_device_to_device_bytes": 0,
        }
        for field, value in expected_fields.items():
            if invocation.get(field) != value:
                raise RuntimeError(
                    f"{case.name}: invocation {index} {field}="
                    f"{invocation.get(field)!r}, expected {value!r}"
                )
        if not isinstance(invocation.get("profile_id"), int) or int(invocation["profile_id"]) < 0:
            raise RuntimeError(f"{case.name}: invocation {index} has no TensorRT profile ID")
        profile_id = int(invocation["profile_id"])
        plan_id = invocation.get("plan_id")
        expected_section = "prefill_engine_plan" if role == "prefill" else "engine_plan"
        if not isinstance(plan_id, str):
            raise RuntimeError(f"{case.name}: invocation {index} has no TensorRT plan identity")
        match = re.fullmatch(
            rf"{re.escape(expected_section)}@engine=0x([0-9a-fA-F]+)",
            plan_id,
        )
        engine_identity = int(match.group(1), 16) if match is not None else 0
        if match is None or engine_identity == 0:
            raise RuntimeError(
                f"{case.name}: invocation {index} has invalid {role} plan identity {plan_id!r}"
            )
        prior_role_identity = role_engine_identities.setdefault(role, engine_identity)
        if prior_role_identity != engine_identity:
            raise RuntimeError(f"{case.name}: {role} invocations do not share one engine identity")
        prior_role_plan = role_plan_ids.setdefault(role, plan_id)
        if prior_role_plan != plan_id:
            raise RuntimeError(f"{case.name}: {role} invocations do not share one engine identity")
        role_profile = (role, profile_id)
        prior_profile_plan = role_profile_plan_ids.setdefault(role_profile, plan_id)
        if prior_profile_plan != plan_id:
            raise RuntimeError(f"{case.name}: {role} profile {profile_id} changed engine identity")
        if invocation.get("cuda_graph_status") not in {"uncaptured", "active"}:
            raise RuntimeError(f"{case.name}: invocation {index} has invalid CUDA graph status")
        base_address = invocation.get("kv_base_address")
        if not isinstance(base_address, int) or base_address <= 0 or (base_address % 256) != 0:
            raise RuntimeError(f"{case.name}: invocation {index} has an invalid KV base address")
        if stable_base_address is None:
            stable_base_address = base_address
        elif base_address != stable_base_address:
            raise RuntimeError(f"{case.name}: invocation {index} replaced the KV base address")
        invocation_context_bytes = invocation.get("context_device_memory_bytes")
        if (
            not isinstance(invocation_context_bytes, int)
            or invocation_context_bytes <= 0
            or invocation_context_bytes > shared_context_bytes
        ):
            raise RuntimeError(
                f"{case.name}: invocation {index} has invalid actual-shape context bytes"
            )
        expected_history = end
        total_append_bytes += int(invocation["kv_append_bytes"])
    if expected_history != case.prompt_tokens + case.decode_tokens:
        raise RuntimeError(f"{case.name}: invocation chunk ranges are not contiguous")
    expected_append_bytes = expected_history * bytes_per_token
    if total_append_bytes != expected_append_bytes:
        raise RuntimeError(
            f"{case.name}: append traffic {total_append_bytes}, expected {expected_append_bytes}"
        )
    if {
        "prefill",
        "decode",
    }.issubset(role_engine_identities) and role_engine_identities[
        "prefill"
    ] == role_engine_identities["decode"]:
        raise RuntimeError(f"{case.name}: prefill and decode report the same engine identity")
    if case.expected_decode_profile_ids:
        decode_invocations = [
            invocation for invocation in invocations if invocation.get("role") == "decode"
        ]
        actual_profile_ids = tuple(
            int(invocation["profile_id"]) for invocation in decode_invocations
        )
        if actual_profile_ids != case.expected_decode_profile_ids:
            raise RuntimeError(
                f"{case.name}: decode profiles {actual_profile_ids}, expected "
                f"{case.expected_decode_profile_ids} for bucket limits "
                f"{case.expected_decode_bucket_limits}"
            )
        if (
            len([invocation for invocation in invocations if invocation.get("role") == "prefill"])
            != expected_launches
        ):
            raise RuntimeError(
                f"{case.name}: decode bucket selection triggered an extra prefill invocation"
            )
    if case.name.startswith("profile-crossing-"):
        decode_invocations = [
            invocation for invocation in invocations if invocation.get("role") == "decode"
        ]
        if len(decode_invocations) != 2:
            raise RuntimeError(
                f"{case.name}: profile crossing requires exactly two decode invocations"
            )
        if decode_invocations[0]["profile_id"] == decode_invocations[1]["profile_id"]:
            raise RuntimeError(f"{case.name}: decoder profile did not switch across the bucket")
    return validation_evidence


def context_shape_sweep(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the measured actual-shape context points from one real request."""
    invocations = trace.get("invocations")
    if not isinstance(invocations, list):
        raise ValueError("trace has no per-invocation shape measurements")
    sweep: list[dict[str, Any]] = []
    for invocation in invocations:
        if not isinstance(invocation, dict):
            raise ValueError("trace invocation is not an object")
        chunk = invocation.get("chunk_range")
        if (
            not isinstance(chunk, list)
            or len(chunk) != 2
            or not all(isinstance(value, int) for value in chunk)
        ):
            raise ValueError("trace invocation has an invalid chunk range")
        sweep.append(
            {
                "role": invocation.get("role"),
                "Sq": int(chunk[1]) - int(chunk[0]),
                "H": invocation.get("H"),
                "A": invocation.get("A"),
                "T": invocation.get("T"),
                "R": invocation.get("R"),
                "context_device_memory_bytes": invocation.get("context_device_memory_bytes"),
            }
        )
    return sweep


def validate_context_memory_envelope(
    spec: ModelSpec,
    case_reports: Iterable[Mapping[str, Any]],
    *,
    require_full_coverage: bool,
) -> dict[str, Any]:
    """Prove measured context memory stays inside a fused O(C*A) envelope.

    ``updateDeviceMemorySizeForShapes()`` includes fixed TensorRT context
    storage, so each execution role starts from its measured smallest-A
    baseline. Growth above that baseline is bounded by two BF16
    ``[Hq, C, A]`` score-equivalent surfaces, allowing a fused implementation
    to retain one auxiliary workspace without weakening the asymptotic gate.
    This remains deliberately generous while rejecting profile-M full-score
    materialization and quadratic growth.
    """
    element_bytes = 2
    max_score_equivalent_surfaces = 2
    points: list[dict[str, Any]] = []
    for case_report in case_reports:
        case_name = str(case_report.get("name", ""))
        sweep = case_report.get("actual_shape_context_sweep")
        if sweep is None:
            continue
        if not isinstance(sweep, list):
            raise RuntimeError(f"{case_name}: actual_shape_context_sweep is not an array")
        for index, raw in enumerate(sweep):
            if not isinstance(raw, Mapping):
                raise RuntimeError(f"{case_name}: context sweep point {index} is not an object")
            role = raw.get("role")
            sq = raw.get("Sq")
            active = raw.get("A")
            history_bound = raw.get("T")
            capacity = raw.get("R")
            context_bytes = raw.get("context_device_memory_bytes")
            if (
                role not in {"prefill", "decode"}
                or isinstance(sq, bool)
                or not isinstance(sq, int)
                or sq <= 0
                or sq > spec.chunk_limit
                or isinstance(active, bool)
                or not isinstance(active, int)
                or active <= 0
                or active > spec.context_limit
                or isinstance(history_bound, bool)
                or not isinstance(history_bound, int)
                or history_bound <= 0
                or history_bound > spec.context_limit
                or isinstance(capacity, bool)
                or not isinstance(capacity, int)
                or capacity <= 0
                or capacity > spec.context_limit
                or active > capacity
                or history_bound > capacity
                or isinstance(context_bytes, bool)
                or not isinstance(context_bytes, int)
                or context_bytes <= 0
            ):
                raise RuntimeError(f"{case_name}: context sweep point {index} is invalid")
            points.append(
                {
                    "case": case_name,
                    "role": role,
                    "Sq": sq,
                    "A": active,
                    "T": history_bound,
                    "R": capacity,
                    "context_device_memory_bytes": context_bytes,
                }
            )

    if not points:
        return {
            "schema_version": 1,
            "status": ("failed" if require_full_coverage else "not_evaluated"),
            "passed": not require_full_coverage,
            "coverage_required": require_full_coverage,
            "reason": "no successful context-shape points were recorded",
            "points": [],
        }

    role_envelopes: dict[str, Any] = {}
    all_points_within_envelope = True
    for role in ("prefill", "decode"):
        role_points = [point for point in points if point["role"] == role]
        if not role_points:
            role_envelopes[role] = {
                "present": False,
                "passed": False,
                "points": [],
            }
            all_points_within_envelope = False
            continue
        baseline_active = min(point["A"] for point in role_points)
        baseline_bytes = max(
            point["context_device_memory_bytes"]
            for point in role_points
            if point["A"] == baseline_active
        )
        evaluated: list[dict[str, Any]] = []
        role_passed = True
        for point in sorted(
            role_points,
            key=lambda item: (
                item["A"],
                item["Sq"],
                item["T"],
                item["case"],
            ),
        ):
            growth_tokens = max(0, point["A"] - baseline_active)
            linear_growth_bound = (
                max_score_equivalent_surfaces
                * spec.chunk_limit
                * growth_tokens
                * spec.num_query_heads
                * element_bytes
            )
            allowed_bytes = baseline_bytes + linear_growth_bound
            within = point["context_device_memory_bytes"] <= allowed_bytes
            role_passed = role_passed and within
            evaluated.append(
                {
                    **point,
                    "baseline_active_tokens": baseline_active,
                    "baseline_context_bytes": baseline_bytes,
                    "linear_growth_bound_bytes": linear_growth_bound,
                    "allowed_context_bytes": allowed_bytes,
                    "within_o_c_times_a_envelope": within,
                }
            )
        role_envelopes[role] = {
            "present": True,
            "passed": role_passed,
            "baseline_active_tokens": baseline_active,
            "baseline_context_bytes": baseline_bytes,
            "points": evaluated,
        }
        all_points_within_envelope = all_points_within_envelope and role_passed

    active_values = sorted({point["A"] for point in points})
    full_context_points = [point for point in points if point["A"] == spec.context_limit]
    full_score_bytes = (
        spec.context_limit * spec.context_limit * spec.num_query_heads * element_bytes
    )
    # One eighth of a full score tensor still leaves a broad implementation
    # margin while making a full [Hq,M,M] score allocation impossible.
    full_score_rejection_bound = full_score_bytes // 8
    all_points_below_score_bound = all(
        point["context_device_memory_bytes"] < full_score_rejection_bound for point in points
    )
    coverage = {
        "has_prefill_and_decode": all(
            role_envelopes[role]["present"] for role in ("prefill", "decode")
        ),
        "reaches_model_context_limit": bool(full_context_points),
        "has_at_least_three_active_lengths": len(active_values) >= 3,
    }
    coverage_passed = all(coverage.values())
    gates = {
        "all_points_within_o_c_times_a_envelope": all_points_within_envelope,
        "all_points_below_materialized_score_bound": all_points_below_score_bound,
        "coverage": coverage,
    }
    passed = bool(
        all_points_within_envelope
        and all_points_below_score_bound
        and (coverage_passed or not require_full_coverage)
    )
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "coverage_required": require_full_coverage,
        "model_context_limit": spec.context_limit,
        "prefill_chunk_limit": spec.chunk_limit,
        "num_query_heads": spec.num_query_heads,
        "element_bytes": element_bytes,
        "max_score_equivalent_surfaces": max_score_equivalent_surfaces,
        "envelope": ("measured_role_baseline + 2 * C * (A - baseline_A) * Hq * sizeof(bfloat16)"),
        "materialized_full_score_bytes": full_score_bytes,
        "materialized_score_rejection_bound_bytes": full_score_rejection_bound,
        "active_lengths": active_values,
        "role_envelopes": role_envelopes,
        "gates": gates,
    }


def _parse_attributed_memory_sample(
    sample: Any,
    *,
    sampler_pid: int,
    sampler_device: int | None,
    require_phase: bool,
) -> dict[str, Any]:
    if not isinstance(sample, Mapping):
        raise RuntimeError("runtime memory sample is not an object")
    phase = sample.get("phase")
    device = sample.get("device")
    free_bytes = sample.get("free_bytes")
    total_bytes = sample.get("total_bytes")
    used_bytes = sample.get("used_bytes")
    process_used_bytes = sample.get("process_used_bytes")
    all_process_used_bytes = sample.get("all_compute_process_used_bytes")
    other_process_used_bytes = sample.get("other_compute_process_used_bytes")
    nvml_total_bytes = sample.get("nvml_device_total_bytes")
    nvml_reserved_bytes = sample.get("nvml_device_reserved_bytes")
    nvml_free_bytes = sample.get("nvml_device_free_bytes")
    nvml_used_bytes = sample.get("nvml_device_used_bytes")
    post_nvml_free_bytes = sample.get("post_nvml_free_bytes")
    post_nvml_total_bytes = sample.get("post_nvml_total_bytes")
    compute_processes = sample.get("compute_processes")
    if (
        (require_phase and (not isinstance(phase, str) or not phase))
        or (
            require_phase
            and (
                not isinstance(device, int)
                or device < 0
                or sampler_device is None
                or device != sampler_device
            )
        )
        or not isinstance(free_bytes, int)
        or not isinstance(total_bytes, int)
        or not isinstance(used_bytes, int)
        or not isinstance(process_used_bytes, int)
        or not isinstance(all_process_used_bytes, int)
        or not isinstance(other_process_used_bytes, int)
        or not isinstance(nvml_total_bytes, int)
        or not isinstance(nvml_reserved_bytes, int)
        or not isinstance(nvml_free_bytes, int)
        or not isinstance(nvml_used_bytes, int)
        or not isinstance(post_nvml_free_bytes, int)
        or not isinstance(post_nvml_total_bytes, int)
        or not isinstance(compute_processes, list)
        or free_bytes <= 0
        or total_bytes <= 0
        or free_bytes > total_bytes
        or used_bytes != total_bytes - free_bytes
        or process_used_bytes < 0
        or all_process_used_bytes < process_used_bytes
        or other_process_used_bytes != all_process_used_bytes - process_used_bytes
        or nvml_total_bytes <= 0
        or nvml_reserved_bytes < 0
        or nvml_free_bytes < 0
        or nvml_used_bytes < 0
        or nvml_reserved_bytes + nvml_free_bytes + nvml_used_bytes != nvml_total_bytes
        or nvml_total_bytes != total_bytes + nvml_reserved_bytes
        or post_nvml_free_bytes <= 0
        or post_nvml_total_bytes != total_bytes
        or post_nvml_free_bytes > post_nvml_total_bytes
    ):
        raise RuntimeError("runtime memory sample is invalid")

    process_sum = 0
    current_process_sum = 0
    parsed_processes: list[dict[str, int]] = []
    observed_pids: set[int] = set()
    for process in compute_processes:
        if not isinstance(process, Mapping):
            raise RuntimeError("runtime memory NVML process row is not an object")
        pid = process.get("pid")
        process_bytes = process.get("used_bytes")
        if (
            not isinstance(pid, int)
            or pid <= 0
            or pid in observed_pids
            or not isinstance(process_bytes, int)
            or process_bytes < 0
        ):
            raise RuntimeError("runtime memory NVML process row is invalid")
        observed_pids.add(pid)
        process_sum += process_bytes
        if pid == sampler_pid:
            current_process_sum += process_bytes
        parsed_processes.append({"pid": pid, "used_bytes": process_bytes})
    if (
        sampler_pid not in observed_pids
        or process_sum != all_process_used_bytes
        or current_process_sum != process_used_bytes
    ):
        raise RuntimeError("runtime memory NVML process ledger disagrees with its aggregate fields")

    parsed = {
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "used_bytes": used_bytes,
        "process_used_bytes": process_used_bytes,
        "all_compute_process_used_bytes": all_process_used_bytes,
        "other_compute_process_used_bytes": other_process_used_bytes,
        "nvml_device_total_bytes": nvml_total_bytes,
        "nvml_device_reserved_bytes": nvml_reserved_bytes,
        "nvml_device_free_bytes": nvml_free_bytes,
        "nvml_device_used_bytes": nvml_used_bytes,
        "post_nvml_free_bytes": post_nvml_free_bytes,
        "post_nvml_total_bytes": post_nvml_total_bytes,
        "compute_processes": parsed_processes,
    }
    if require_phase:
        parsed["phase"] = phase
        parsed["device"] = device
    return parsed


def _signed_memory_attribution(
    baseline: Mapping[str, Any],
    boundary: Mapping[str, Any],
    *,
    positive_unlisted_external_limit_bytes: int | None = None,
    negative_unlisted_external_limit_bytes: int | None = None,
) -> dict[str, Any]:
    if baseline["total_bytes"] != boundary["total_bytes"]:
        raise RuntimeError("memory attribution samples disagree on CUDA device total")
    device_growth = int(baseline["free_bytes"]) - int(boundary["free_bytes"])
    process_growth = int(boundary["process_used_bytes"]) - int(
        baseline["process_used_bytes"]
    )
    visible_other_process_growth = int(
        boundary["other_compute_process_used_bytes"]
    ) - int(baseline["other_compute_process_used_bytes"])
    baseline_non_current = int(baseline["nvml_device_used_bytes"]) - int(
        baseline["process_used_bytes"]
    )
    boundary_non_current = int(boundary["nvml_device_used_bytes"]) - int(
        boundary["process_used_bytes"]
    )
    external_growth = boundary_non_current - baseline_non_current
    unexplained_growth = device_growth - process_growth - external_growth
    unlisted_external_growth = external_growth - visible_other_process_growth
    baseline_bracket_difference = abs(
        int(baseline["post_nvml_free_bytes"]) - int(baseline["free_bytes"])
    )
    boundary_bracket_difference = abs(
        int(boundary["post_nvml_free_bytes"]) - int(boundary["free_bytes"])
    )
    tolerance = max(
        _MEMORY_ATTRIBUTION_FLOOR_BYTES,
        math.ceil(
            0.02
            * max(
                abs(device_growth),
                abs(process_growth),
                abs(external_growth),
                1,
            )
        ),
    )
    positive_unlisted_limit = (
        tolerance
        if positive_unlisted_external_limit_bytes is None
        else max(tolerance, positive_unlisted_external_limit_bytes)
    )
    negative_unlisted_limit = (
        tolerance
        if negative_unlisted_external_limit_bytes is None
        else max(tolerance, negative_unlisted_external_limit_bytes)
    )
    passed = bool(
        baseline_bracket_difference <= tolerance
        and boundary_bracket_difference <= tolerance
        and abs(unexplained_growth) <= tolerance
        and -negative_unlisted_limit
        <= unlisted_external_growth
        <= positive_unlisted_limit
    )
    return {
        "cuda_device_growth_bytes": device_growth,
        "nvml_current_process_growth_bytes": process_growth,
        "nvml_visible_other_process_growth_bytes": visible_other_process_growth,
        "nvml_non_current_device_growth_bytes": external_growth,
        "unlisted_external_growth_bytes": unlisted_external_growth,
        "positive_unlisted_external_limit_bytes": positive_unlisted_limit,
        "negative_unlisted_external_limit_bytes": negative_unlisted_limit,
        "unexplained_growth_bytes": unexplained_growth,
        "baseline_cuda_nvml_bracket_difference_bytes": baseline_bracket_difference,
        "boundary_cuda_nvml_bracket_difference_bytes": boundary_bracket_difference,
        "tolerance_bytes": tolerance,
        "tolerance_rule": "max(64MiB,2pct)",
        "reconciliation_formula": "U = D - P - X",
        "passed": passed,
    }


def _validate_lifetime_phase_samples(
    lifetime: Mapping[str, Any],
    *,
    sampler_identity: Mapping[str, Any],
    role: str,
) -> list[dict[str, Any]]:
    samples = lifetime.get("runtime_phase_memory_samples")
    if not isinstance(samples, list):
        raise RuntimeError(f"{role} lifetime has no synchronized runtime phase samples")
    if len(samples) != 5:
        raise RuntimeError(f"{role} lifetime must have exactly five runtime phase samples")
    parsed = [
        _parse_attributed_memory_sample(
            sample,
            sampler_pid=int(sampler_identity["pid"]),
            sampler_device=int(sampler_identity["cuda_logical_device_index"]),
            require_phase=True,
        )
        for sample in samples
    ]
    baseline_phase = str(parsed[0]["phase"])
    if not (
        baseline_phase.startswith("before runtime-memory ")
        and baseline_phase.endswith(" engine deserialization")
    ):
        raise RuntimeError(f"{role} lifetime has an invalid pre-engine baseline phase")
    phases = tuple(str(sample["phase"]) for sample in parsed[1:])
    if phases != _RUNTIME_PHASES_AFTER_BASELINE:
        raise RuntimeError(
            f"{role} lifetime runtime phases are incomplete or out of order: {phases}"
        )
    return parsed


def _validate_lifetime_endpoint_bindings(
    lifetime: Mapping[str, Any],
    phase_samples: list[dict[str, Any]],
    *,
    sampler_identity: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    before_load = _parse_attributed_memory_sample(
        lifetime.get("before_load"),
        sampler_pid=int(sampler_identity["pid"]),
        sampler_device=None,
        require_phase=False,
    )
    after_requests = _parse_attributed_memory_sample(
        lifetime.get("after_requests"),
        sampler_pid=int(sampler_identity["pid"]),
        sampler_device=None,
        require_phase=False,
    )
    before_binding = _signed_memory_attribution(before_load, phase_samples[0])
    after_binding = _signed_memory_attribution(phase_samples[-1], after_requests)

    def endpoint_passed(attribution: Mapping[str, Any]) -> bool:
        bounded_fields = (
            "cuda_device_growth_bytes",
            "nvml_current_process_growth_bytes",
            "unlisted_external_growth_bytes",
            "unexplained_growth_bytes",
        )
        return bool(
            attribution.get("passed") is True
            and all(
                abs(int(attribution[field])) <= _MEMORY_ATTRIBUTION_FLOOR_BYTES
                for field in bounded_fields
            )
        )

    before_passed = endpoint_passed(before_binding)
    after_passed = endpoint_passed(after_binding)
    expected_process_growth = int(after_requests["process_used_bytes"]) - int(
        before_load["process_used_bytes"]
    )
    expected_device_growth = int(after_requests["used_bytes"]) - int(
        before_load["used_bytes"]
    )
    if (
        lifetime.get("process_growth_bytes") != expected_process_growth
        or lifetime.get("device_wide_growth_bytes") != expected_device_growth
    ):
        raise RuntimeError(f"{role} lifetime growth accounting is inconsistent")
    if not before_passed or not after_passed:
        raise RuntimeError(
            f"{role} lifetime endpoints do not bind the synchronized phase samples"
        )
    return {
        "schema_version": 1,
        "before_load_to_pre_engine_baseline": {
            **before_binding,
            "passed": before_passed,
        },
        "request_completion_to_after_requests": {
            **after_binding,
            "passed": after_passed,
        },
        "passed": True,
    }


def _validate_lifetime_policy(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, Mapping):
        raise RuntimeError("qualification lifetime policy is not an object")
    value = dict(policy)
    kind = value.get("kind")
    if kind == "auto":
        valid = value == {"kind": "auto"}
    elif kind == "fraction":
        fraction = value.get("requested_fraction")
        valid = (
            set(value) == {"kind", "requested_fraction"}
            and type(fraction) is float
            and math.isfinite(fraction)
            and 0.0 < fraction <= 1.0
        )
    elif kind == "bytes":
        requested_bytes = value.get("requested_bytes")
        valid = (
            set(value) == {"kind", "requested_bytes"}
            and type(requested_bytes) is int
            and requested_bytes > 0
        )
    elif kind == "max_sequence_length":
        requested_tokens = value.get("requested_tokens")
        valid = (
            set(value) == {"kind", "requested_tokens"}
            and type(requested_tokens) is int
            and requested_tokens > 0
        )
    else:
        valid = False
    if not valid:
        raise RuntimeError(f"qualification lifetime policy is invalid: {value!r}")
    return value


def _receipt_policy_for_lifetime_policy(policy: Mapping[str, Any]) -> str:
    kind = policy["kind"]
    return "auto" if kind == "max_sequence_length" else str(kind)


def _binary64_fraction_floor(fraction: float, value: int) -> int:
    """Floor the exact rational value represented by a binary64 fraction."""

    if (
        type(fraction) is not float
        or not math.isfinite(fraction)
        or fraction <= 0.0
        or fraction > 1.0
        or type(value) is not int
        or value < 0
    ):
        raise ValueError("binary64 fraction budget inputs are invalid")
    numerator, denominator = fraction.as_integer_ratio()
    return numerator * value // denominator


def _validate_receipt_policy_binding(
    policy: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    trusted_geometry: TrustedRuntimeGeometry,
    expected_capacity_tokens: int,
    expected_effective_request_limit: int,
) -> None:
    """Bind the typed user request to the receipt's resolved policy contract."""

    kind = str(policy["kind"])
    expected_receipt_policy = _receipt_policy_for_lifetime_policy(policy)
    expected_fraction = (
        float(policy["requested_fraction"])
        if kind == "fraction"
        else 0.0
        if kind == "bytes"
        else 0.9
    )
    expected_requested_bytes = (
        int(policy["requested_bytes"]) if kind == "bytes" else 0
    )
    expected_request_limit = (
        int(policy["requested_tokens"])
        if kind == "max_sequence_length"
        else 0
    )

    policy_fraction = receipt.get("policy_fraction")
    requested_kv_bytes = receipt.get("requested_kv_bytes")
    request_context_limit = receipt.get("request_context_limit")
    model_context_limit = receipt.get("model_context_limit")
    prefill_chunk_limit = receipt.get("prefill_chunk_limit")
    capacity_tokens = receipt.get("runtime_kv_capacity_tokens")
    effective_request_limit = receipt.get("effective_request_limit")
    bytes_per_token = receipt.get("kv_bytes_per_token")
    kv_budget_bytes = receipt.get("kv_budget_bytes")
    kv_reserved_bytes = receipt.get("kv_reserved_bytes")
    kv_committed_bytes = receipt.get("kv_committed_bytes")
    context_device_memory_bytes = receipt.get(
        "context_device_memory_bytes"
    )
    ordinary_device_input_bytes = receipt.get(
        "ordinary_device_input_bytes"
    )
    ordinary_device_output_bytes = receipt.get(
        "ordinary_device_output_bytes"
    )
    external_device_output_bytes = receipt.get(
        "external_device_output_bytes"
    )
    graph_private_device_bytes = receipt.get(
        "graph_private_device_bytes"
    )
    if (
        type(receipt.get("receipt_schema_version")) is not int
        or receipt.get("receipt_schema_version") != 3
        or receipt.get("policy") != expected_receipt_policy
        or type(policy_fraction) not in {int, float}
        or not math.isfinite(float(policy_fraction))
        or float(policy_fraction) != expected_fraction
        or type(requested_kv_bytes) is not int
        or requested_kv_bytes != expected_requested_bytes
        or type(request_context_limit) is not int
        or request_context_limit != expected_request_limit
        or type(model_context_limit) is not int
        or model_context_limit != trusted_geometry.model_context_limit
        or type(prefill_chunk_limit) is not int
        or prefill_chunk_limit != trusted_geometry.prefill_chunk_limit
        or type(capacity_tokens) is not int
        or capacity_tokens != expected_capacity_tokens
        or capacity_tokens <= 0
        or capacity_tokens > model_context_limit
        or type(effective_request_limit) is not int
        or effective_request_limit != expected_effective_request_limit
        or effective_request_limit != capacity_tokens
        or type(bytes_per_token) is not int
        or bytes_per_token != trusted_geometry.kv_bytes_per_token
        or type(kv_reserved_bytes) is not int
        or type(kv_committed_bytes) is not int
        or kv_reserved_bytes != capacity_tokens * bytes_per_token
        or kv_committed_bytes != kv_reserved_bytes
        or type(context_device_memory_bytes) is not int
        or context_device_memory_bytes <= 0
        or type(ordinary_device_input_bytes) is not int
        or ordinary_device_input_bytes < 0
        or type(ordinary_device_output_bytes) is not int
        or ordinary_device_output_bytes < 0
        or type(external_device_output_bytes) is not int
        or external_device_output_bytes < 0
        or type(graph_private_device_bytes) is not int
        or graph_private_device_bytes < 0
    ):
        raise RuntimeError(
            "runtime-memory receipt does not bind the typed request policy, "
            "resolved R/effective limit, exact contiguous KV ledger, and "
            "typed non-KV device accounting"
        )

    if expected_request_limit > trusted_geometry.model_context_limit:
        raise RuntimeError(
            "runtime-memory receipt accepted a max-sequence request above the "
            "model context limit"
        )
    if expected_request_limit != 0 and capacity_tokens > expected_request_limit:
        raise RuntimeError(
            "runtime-memory receipt capacity exceeds the typed max-sequence request"
        )
    capacity_decision_free_bytes = receipt.get(
        "capacity_decision_free_bytes"
    )
    capacity_decision_total_bytes = receipt.get(
        "capacity_decision_total_bytes"
    )
    capacity_decision_used_bytes = receipt.get(
        "capacity_decision_device_used_bytes"
    )
    settled_free_bytes = receipt.get("settled_free_bytes")
    settled_total_bytes = receipt.get("settled_total_bytes")
    settled_used_bytes = receipt.get("settled_device_used_bytes")
    safety_reserve_bytes = receipt.get("safety_reserve_bytes")
    if (
        type(kv_budget_bytes) is not int
        or kv_budget_bytes <= 0
        or type(capacity_decision_free_bytes) is not int
        or capacity_decision_free_bytes <= 0
        or type(capacity_decision_total_bytes) is not int
        or capacity_decision_total_bytes <= 0
        or capacity_decision_free_bytes > capacity_decision_total_bytes
        or type(capacity_decision_used_bytes) is not int
        or capacity_decision_used_bytes
        != capacity_decision_total_bytes - capacity_decision_free_bytes
        or receipt.get("final_free_bytes")
        != capacity_decision_free_bytes
        or receipt.get("final_total_bytes")
        != capacity_decision_total_bytes
        or receipt.get("final_device_used_bytes")
        != capacity_decision_used_bytes
        or type(settled_free_bytes) is not int
        or settled_free_bytes <= 0
        or type(settled_total_bytes) is not int
        or settled_total_bytes != capacity_decision_total_bytes
        or settled_free_bytes > settled_total_bytes
        or type(settled_used_bytes) is not int
        or settled_used_bytes != settled_total_bytes - settled_free_bytes
        or receipt.get("settled_snapshot_unavailable_reason") is not None
        or type(safety_reserve_bytes) is not int
        or safety_reserve_bytes < 0
    ):
        raise RuntimeError(
            "runtime-memory receipt has no exact schema-v3 capacity-decision, "
            "settled, and deprecated-final snapshot binding"
        )
    safely_available_bytes = max(
        0,
        capacity_decision_free_bytes - safety_reserve_bytes,
    )
    if kind == "bytes":
        expected_budget_bytes = expected_requested_bytes
        if kv_reserved_bytes > safely_available_bytes:
            raise RuntimeError(
                "runtime-memory receipt claims an explicit byte request that "
                "did not fit the capacity-decision memory snapshot"
            )
    else:
        # Treat the requested binary64 fraction as its exact rational value.
        # A rounded binary64 multiplication can exceed the policy by bytes at
        # large capacities before floor() is applied.  Do not recompute from
        # settled_*: those fields describe post-allocation residency and may
        # show more free memory after tentative overhead is resized.
        expected_budget_bytes = _binary64_fraction_floor(
            expected_fraction,
            safely_available_bytes,
        )
    if kv_budget_bytes != expected_budget_bytes:
        raise RuntimeError(
            "runtime-memory receipt KV budget does not exactly resolve the "
            "typed policy from the capacity-decision sample"
        )

    semantic_limit = min(
        trusted_geometry.model_context_limit,
        expected_request_limit
        if expected_request_limit != 0
        else trusted_geometry.model_context_limit,
    )
    budget_rows = kv_budget_bytes // trusted_geometry.kv_bytes_per_token
    expected_resolved_capacity = min(semantic_limit, budget_rows)
    if (
        capacity_tokens != expected_resolved_capacity
        or kv_reserved_bytes > kv_budget_bytes
        or capacity_tokens * trusted_geometry.kv_bytes_per_token
        > kv_budget_bytes
        or receipt.get("capped_by_model")
        is not (capacity_tokens == trusted_geometry.model_context_limit)
        or receipt.get("capped_by_request_limit")
        is not (
            expected_request_limit != 0
            and capacity_tokens == expected_request_limit
        )
    ):
        raise RuntimeError(
            "runtime-memory receipt R does not exactly equal the runtime "
            "solver's semantic/capacity-decision-budget row minimum"
        )


def _validate_memory_sampler(
    trace: Mapping[str, Any],
    *,
    expected_sampler: SamplerTrustAnchor | None = None,
) -> dict[str, Any]:
    sampler = trace.get("memory_sampler")
    if not isinstance(sampler, Mapping) or sampler.get("source") != (
        "nvmlDeviceGetComputeRunningProcesses_v3"
    ):
        raise RuntimeError("qualification requires independent NVML process memory sampling")
    if sampler.get("captures_all_compute_processes") is not True:
        raise RuntimeError("qualification requires the complete NVML compute-process ledger")
    if sampler.get("device_memory_source") != "nvmlDeviceGetMemoryInfo_v2":
        raise RuntimeError("qualification requires independent NVML device memory sampling")
    sampler_pid = sampler.get("pid")
    logical_device = sampler.get("cuda_logical_device_index")
    physical_device = sampler.get("physical_device_index")
    pci_bus_id = sampler.get("pci_bus_id")
    gpu_uuid = sampler.get("gpu_uuid")
    if (
        type(sampler_pid) is not int
        or sampler_pid <= 0
        or type(logical_device) is not int
        or logical_device < 0
        or type(physical_device) is not int
        or physical_device < 0
        or not isinstance(pci_bus_id, str)
        or re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", pci_bus_id)
        is None
        or not isinstance(gpu_uuid, str)
        or re.fullmatch(r"GPU-[0-9a-fA-F-]{16,}", gpu_uuid) is None
    ):
        raise RuntimeError("qualification NVML sampler has an invalid GPU or process identity")
    if expected_sampler is not None and (
        sampler_pid != expected_sampler.pid
        or logical_device != expected_sampler.cuda_logical_device_index
        or physical_device != expected_sampler.physical_device_index
        or str(pci_bus_id).lower() != expected_sampler.pci_bus_id.lower()
        or gpu_uuid != expected_sampler.gpu_uuid
    ):
        raise RuntimeError(
            "qualification NVML sampler does not match the producer-trusted "
            "child/CUDA-visible GPU identity"
        )
    return {
        "pid": sampler_pid,
        "cuda_logical_device_index": logical_device,
        "physical_device_index": physical_device,
        "pci_bus_id": pci_bus_id,
        "gpu_uuid": gpu_uuid,
    }


def _reconcile_lifetime_peak_with_nvml(
    lifetime: Mapping[str, Any],
    *,
    sampler_identity: Mapping[str, Any],
    role: str,
    expected_persistent_unlisted_external_growth_bytes: int = 0,
    persistent_unlisted_external_limit_bytes: int | None = None,
    negative_persistent_unlisted_external_limit_bytes: int | None = None,
) -> dict[str, Any]:
    receipt = lifetime.get("runtime_memory_receipt")
    if not isinstance(receipt, Mapping):
        raise RuntimeError(f"{role} lifetime has no runtime memory receipt")
    device_peak = receipt.get("peak_device_bytes")
    if not isinstance(device_peak, int) or device_peak < 0:
        raise RuntimeError(f"{role} lifetime receipt has no sampled device-wide peak")

    parsed_samples = _validate_lifetime_phase_samples(
        lifetime,
        sampler_identity=sampler_identity,
        role=role,
    )
    baseline_sample = parsed_samples[0]
    required_boundaries = (
        "after runtime KV allocation",
        "after successful runtime-memory request completion",
    )
    observed_boundaries = [
        sample for sample in parsed_samples if sample["phase"] in required_boundaries
    ]
    if (
        len(observed_boundaries) != len(required_boundaries)
        or tuple(str(sample["phase"]) for sample in observed_boundaries)
        != required_boundaries
    ):
        raise RuntimeError(
            f"{role} lifetime must have exactly one synchronized sample for each peak boundary"
        )

    receipt_total = receipt.get("pre_load_total_bytes")
    if not isinstance(receipt_total, int) or any(
        sample["total_bytes"] != receipt_total
        for sample in [baseline_sample, *observed_boundaries]
    ):
        raise RuntimeError(
            f"{role} synchronized runtime phase samples disagree on CUDA device total"
        )
    baseline_free = int(baseline_sample["free_bytes"])
    synchronized_device_peak = max(
        0,
        max(
            baseline_free - int(sample["free_bytes"])
            for sample in observed_boundaries
        ),
    )
    if synchronized_device_peak != device_peak:
        raise RuntimeError(
            f"{role} receipt peak does not match synchronized runtime phase samples: "
            f"receipt={device_peak}, synchronized={synchronized_device_peak}"
        )
    expected_snapshot_fields = (
        ("pre_load_free_bytes", parsed_samples[0]["free_bytes"]),
        ("pre_load_total_bytes", parsed_samples[0]["total_bytes"]),
        ("post_load_free_bytes", parsed_samples[1]["free_bytes"]),
        ("post_load_total_bytes", parsed_samples[1]["total_bytes"]),
        (
            "post_load_device_used_bytes",
            parsed_samples[1]["total_bytes"]
            - parsed_samples[1]["free_bytes"],
        ),
        (
            "capacity_decision_free_bytes",
            parsed_samples[2]["free_bytes"],
        ),
        (
            "capacity_decision_total_bytes",
            parsed_samples[2]["total_bytes"],
        ),
        (
            "capacity_decision_device_used_bytes",
            parsed_samples[2]["total_bytes"]
            - parsed_samples[2]["free_bytes"],
        ),
        ("final_free_bytes", parsed_samples[2]["free_bytes"]),
        ("final_total_bytes", parsed_samples[2]["total_bytes"]),
        (
            "final_device_used_bytes",
            parsed_samples[2]["total_bytes"]
            - parsed_samples[2]["free_bytes"],
        ),
        ("settled_free_bytes", parsed_samples[3]["free_bytes"]),
        ("settled_total_bytes", parsed_samples[3]["total_bytes"]),
        (
            "settled_device_used_bytes",
            parsed_samples[3]["total_bytes"]
            - parsed_samples[3]["free_bytes"],
        ),
    )
    for field, expected in expected_snapshot_fields:
        if receipt.get(field) != expected:
            raise RuntimeError(
                f"{role} receipt {field} does not bind its synchronized runtime phase"
            )

    boundary_reconciliation: list[dict[str, Any]] = []
    for sample in observed_boundaries:
        attribution = _signed_memory_attribution(
            baseline_sample,
            sample,
            positive_unlisted_external_limit_bytes=(
                persistent_unlisted_external_limit_bytes
            ),
            negative_unlisted_external_limit_bytes=(
                negative_persistent_unlisted_external_limit_bytes
            ),
        )
        persistent_delta_matches = bool(
            abs(
                int(attribution["unlisted_external_growth_bytes"])
                - expected_persistent_unlisted_external_growth_bytes
            )
            <= int(attribution["tolerance_bytes"])
        )
        boundary_reconciliation.append(
            {
                "phase": str(sample["phase"]),
                **attribution,
                "expected_persistent_unlisted_external_growth_bytes": (
                    expected_persistent_unlisted_external_growth_bytes
                ),
                "persistent_unlisted_external_growth_matches": (
                    persistent_delta_matches
                ),
                "passed": bool(attribution["passed"] and persistent_delta_matches),
                "sample": sample,
            }
        )

    process_peak = max(
        0,
        max(
            int(row["nvml_current_process_growth_bytes"])
            for row in boundary_reconciliation
        ),
    )
    maximum_tolerance = max(
        int(row["tolerance_bytes"]) for row in boundary_reconciliation
    )
    passed = all(bool(row["passed"]) for row in boundary_reconciliation)
    result = {
        "schema_version": 1,
        "role": role,
        "device_wide_peak_bytes": device_peak,
        "nvml_process_peak_bytes": process_peak,
        "absolute_difference_bytes": abs(device_peak - process_peak),
        "maximum_unexplained_difference_bytes": max(
            abs(int(row["unexplained_growth_bytes"]))
            for row in boundary_reconciliation
        ),
        "maximum_unlisted_external_difference_bytes": max(
            abs(int(row["unlisted_external_growth_bytes"]))
            for row in boundary_reconciliation
        ),
        "tolerance_bytes": maximum_tolerance,
        "tolerance_rule": "max(64MiB,2pct)",
        "device_scope": "cudaMemGetInfo-device-wide",
        "process_scope": "nvml-current-process",
        "external_scope": (
            "nvml-device-used-minus-current-process-with-visible-process-ledger"
        ),
        "reconciliation_formula": "U = D - P - X",
        "sample_boundaries": [
            str(baseline_sample["phase"]),
            *required_boundaries,
        ],
        "synchronized_cuda_peak_bytes": synchronized_device_peak,
        "baseline_sample": baseline_sample,
        "peak_boundary_samples": observed_boundaries,
        "boundary_reconciliation": boundary_reconciliation,
        "passed": passed,
    }
    if not passed:
        failed_rows = [
            {
                "phase": row["phase"],
                "unexplained_growth_bytes": row["unexplained_growth_bytes"],
                "unlisted_external_growth_bytes": row[
                    "unlisted_external_growth_bytes"
                ],
                "baseline_bracket_bytes": row[
                    "baseline_cuda_nvml_bracket_difference_bytes"
                ],
                "boundary_bracket_bytes": row[
                    "boundary_cuda_nvml_bracket_difference_bytes"
                ],
                "tolerance_bytes": row["tolerance_bytes"],
            }
            for row in boundary_reconciliation
            if not row["passed"]
        ]
        raise RuntimeError(
            f"{role} CUDA/NVML memory scopes do not reconcile after independent "
            f"external attribution: {failed_rows}"
        )
    return result


def _validate_cold_warm_output_equivalence(
    trace: Mapping[str, Any],
    measured_logits: np.ndarray,
) -> dict[str, Any]:
    evidence = trace.get("cold_warm_output_equivalence")
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != _COLD_WARM_OUTPUT_EQUIVALENCE_FIELDS
        or type(evidence.get("schema_version")) is not int
        or evidence.get("schema_version") != 1
        or type(evidence.get("warmup_execution_ordinal")) is not int
        or evidence.get("warmup_execution_ordinal") != 0
        or type(evidence.get("measured_execution_ordinal")) is not int
        or evidence.get("measured_execution_ordinal") != 1
    ):
        raise RuntimeError("cold/warm output equivalence evidence has an invalid schema")
    boolean_gates = _COLD_WARM_OUTPUT_EQUIVALENCE_FIELDS - {
        "schema_version",
        "warmup_execution_ordinal",
        "measured_execution_ordinal",
    }
    if any(evidence.get(name) is not True for name in boolean_gates):
        raise RuntimeError(
            "cold/warm execution outputs are not exactly equivalent: "
            f"{dict(evidence)!r}"
        )
    if measured_logits.ndim != 2 or measured_logits.dtype != np.float32:
        raise RuntimeError("measured logits are not a two-dimensional float32 artifact")
    measured_path = _validate_logits_artifact_metadata(
        trace.get("logits_artifact"),
        role="measured",
        expected_rows=measured_logits.shape[0],
        expected_columns=measured_logits.shape[1],
    )
    cold_path = _validate_logits_artifact_metadata(
        trace.get("cold_start_logits_artifact"),
        role="cold-start",
        expected_rows=measured_logits.shape[0],
        expected_columns=measured_logits.shape[1],
    )
    if cold_path.resolve() == measured_path.resolve():
        raise RuntimeError("cold-start and measured logits artifacts must be distinct")
    persisted_measured_logits = read_logits_artifact(measured_path)
    cold_logits = read_logits_artifact(cold_path)
    measured_bytes = measured_logits.tobytes(order="C")
    persisted_measured_bytes = persisted_measured_logits.tobytes(order="C")
    cold_bytes = cold_logits.tobytes(order="C")
    if persisted_measured_bytes != measured_bytes:
        raise RuntimeError("measured logits payload is not bound to its persisted artifact")
    python_bitwise_equal = cold_bytes == measured_bytes
    top1 = np.argmax(measured_logits, axis=1).astype(int).tolist()
    cold_top1 = np.argmax(cold_logits, axis=1).astype(int).tolist()
    selected = trace.get("selected_token_ids")
    selected_count = len(selected) if isinstance(selected, list) else -1
    python_ids_equal = bool(
        cold_top1 == top1
        and trace.get("step_top1_token_ids") == top1
        and isinstance(selected, list)
        and selected == top1[:selected_count]
        and selected == cold_top1[:selected_count]
    )
    if not python_bitwise_equal or not python_ids_equal:
        raise RuntimeError(
            "cold/warm logits artifacts or independently derived token IDs differ"
        )
    return {
        **dict(evidence),
        "python_full_float32_logits_bitwise_equal": python_bitwise_equal,
        "python_selected_and_top1_token_ids_equal": python_ids_equal,
        "cold_start_logits_sha256": hashlib.sha256(cold_bytes).hexdigest(),
        "measured_logits_sha256": hashlib.sha256(measured_bytes).hexdigest(),
        "cold_start_logits_artifact": str(cold_path),
        "measured_logits_artifact": str(measured_path),
    }


def validate_warmup_evidence(
    trace: dict[str, Any],
    *,
    trusted_geometry: TrustedRuntimeGeometry,
    expected_sampler: SamplerTrustAnchor | None = None,
    measured_logits: np.ndarray | None = None,
    expected_lifetime_policy: Mapping[str, Any] | None = None,
    expected_capacity_tokens: int | None = None,
    expected_final_kv_position: int | None = None,
    expected_prompt_tokens: int | None = None,
    expected_prefill_launches: int | None = None,
    expected_decode_launches: int | None = None,
) -> dict[str, Any]:
    """Validate one fully-gated cold lifetime followed by one measured lifetime."""

    protocol = trace.get("lifetime_protocol")
    if (
        not isinstance(protocol, dict)
        or set(protocol) != set(_LIFETIME_PROTOCOL)
        or type(protocol.get("schema_version")) is not int
        or type(protocol.get("warmup_count")) is not int
        or type(protocol.get("measured_count")) is not int
        or protocol != _LIFETIME_PROTOCOL
    ):
        raise RuntimeError(
            "qualification lifetime_protocol must exactly describe one warmup "
            "followed by one measured lifetime"
        )
    warmup = trace.get("load_cycle_warmup")
    cycles = trace.get("load_cycles")
    if not isinstance(warmup, Mapping):
        raise RuntimeError("qualification trace is missing the warmup load lifetime")
    if type(trace.get("load_cycle_count")) is not int or trace.get(
        "load_cycle_count"
    ) != 1:
        raise RuntimeError("qualification trace must report exactly one measured load cycle")
    if (
        not isinstance(cycles, list)
        or len(cycles) != 1
        or not isinstance(cycles[0], Mapping)
    ):
        raise RuntimeError("qualification trace must contain exactly one measured load lifetime")
    measured = cycles[0]
    if measured_logits is None:
        measured_metadata = trace.get("logits_artifact")
        if not isinstance(measured_metadata, Mapping):
            raise RuntimeError("qualification trace has no measured logits artifact")
        rows = measured_metadata.get("rows")
        columns = measured_metadata.get("vocab_size")
        if type(rows) is not int or rows <= 0 or type(columns) is not int or columns <= 0:
            raise RuntimeError("measured logits artifact shape metadata is invalid")
        measured_path = _validate_logits_artifact_metadata(
            measured_metadata,
            role="measured",
            expected_rows=rows,
            expected_columns=columns,
        )
        measured_logits = read_logits_artifact(measured_path)

    sampler_identity = _validate_memory_sampler(
        trace,
        expected_sampler=expected_sampler,
    )
    receipt = trace.get("runtime_memory_receipt")
    if not isinstance(receipt, Mapping):
        raise RuntimeError("qualification trace has no runtime memory receipt")
    receipt_capacity = receipt.get("runtime_kv_capacity_tokens")
    receipt_allocation_id = receipt.get("kv_allocation_id")
    receipt_chunk_limit = receipt.get("prefill_chunk_limit")
    if (
        type(receipt_capacity) is not int
        or receipt_capacity <= 0
        or trace.get("runtime_kv_capacity_tokens") != receipt_capacity
        or type(receipt_allocation_id) is not int
        or receipt_allocation_id <= 0
        or trace.get("kv_allocation_id") != receipt_allocation_id
        or type(receipt_chunk_limit) is not int
        or receipt_chunk_limit <= 0
        or trace.get("prefill_chunk_limit") != receipt_chunk_limit
        or trace.get("effective_request_limit")
        != receipt.get("effective_request_limit")
    ):
        raise RuntimeError(
            "qualification trace top-level R/allocation/chunk/effective fields "
            "do not bind the measured runtime-memory receipt"
        )

    if expected_lifetime_policy is None:
        expected_lifetime_policy = measured.get("policy")
    expected_policy = _validate_lifetime_policy(expected_lifetime_policy)
    expected_receipt_policy = _receipt_policy_for_lifetime_policy(expected_policy)
    if expected_capacity_tokens is None:
        expected_capacity_tokens = receipt.get("runtime_kv_capacity_tokens")
    if expected_final_kv_position is None:
        expected_final_kv_position = trace.get("final_kv_position")
    if expected_prompt_tokens is None:
        expected_prompt_tokens = trace.get("prompt_tokens")
    if expected_prefill_launches is None:
        expected_prefill_launches = trace.get("prefill_launches")
    if expected_decode_launches is None:
        expected_decode_launches = trace.get("decode_launches")
    expected_workload = {
        "prompt_tokens": expected_prompt_tokens,
        "prefill_launches": expected_prefill_launches,
        "decode_launches": expected_decode_launches,
        "final_kv_position": expected_final_kv_position,
    }
    if (
        type(expected_capacity_tokens) is not int
        or expected_capacity_tokens <= 0
        or any(type(value) is not int or value < 0 for value in expected_workload.values())
        or expected_prompt_tokens <= 0
        or expected_final_kv_position <= 0
        or expected_final_kv_position
        != expected_prompt_tokens + expected_decode_launches
        or expected_prefill_launches
        != math.ceil(expected_prompt_tokens / receipt_chunk_limit)
    ):
        raise RuntimeError("qualification cold/measured expectations are invalid")
    for name, expected in expected_workload.items():
        if type(trace.get(name)) is not int or trace.get(name) != expected:
            raise RuntimeError(
                f"qualification trace {name}={trace.get(name)!r}, "
                f"expected {expected!r}"
            )
    expected_effective_request_limit = trace.get("effective_request_limit")
    if (
        type(expected_effective_request_limit) is not int
        or expected_effective_request_limit <= 0
    ):
        raise RuntimeError(
            "qualification trace has no typed effective request limit"
        )
    if expected_final_kv_position > expected_effective_request_limit:
        raise RuntimeError(
            "qualification workload exceeds the resolved runtime KV/effective "
            "request limit"
        )
    _validate_receipt_policy_binding(
        expected_policy,
        receipt,
        trusted_geometry=trusted_geometry,
        expected_capacity_tokens=expected_capacity_tokens,
        expected_effective_request_limit=expected_effective_request_limit,
    )

    expected_rows = (
        (warmup, 0, "warmup", False, "unmeasured-load-cycle-warmup"),
        (measured, 1, "measured", True, "measured-load-cycle"),
    )
    parsed_phase_rows: dict[str, list[dict[str, Any]]] = {}
    parsed_boundaries: dict[str, dict[str, dict[str, Any]]] = {}
    endpoint_bindings: dict[str, dict[str, Any]] = {}
    for row, ordinal, role, measured_marker, label in expected_rows:
        expected_markers = {
            "execution_ordinal": ordinal,
            "role": role,
            "measured": measured_marker,
            "label": label,
        }
        for name, expected in expected_markers.items():
            actual = row.get(name)
            marker_matches = (
                actual is expected
                if name == "measured"
                else type(actual) is int and actual == expected
                if name == "execution_ordinal"
                else actual == expected
            )
            if not marker_matches:
                raise RuntimeError(
                    f"{role} lifetime {name}={actual!r}, expected {expected!r}"
                )
        actual_policy = _validate_lifetime_policy(row.get("policy"))
        if actual_policy != expected_policy:
            raise RuntimeError(
                f"{role} lifetime request policy does not match the measured request"
            )
        if (
            type(row.get("runtime_kv_capacity_tokens")) is not int
            or row.get("runtime_kv_capacity_tokens") != expected_capacity_tokens
        ):
            raise RuntimeError(
                f"{role} lifetime capacity does not match the measured request"
            )
        for name, expected in expected_workload.items():
            if type(row.get(name)) is not int or row.get(name) != expected:
                raise RuntimeError(
                    f"{role} lifetime {name}={row.get(name)!r}, expected {expected!r}"
                )
        row_receipt = row.get("runtime_memory_receipt")
        if (
            not isinstance(row_receipt, Mapping)
            or row_receipt.get("runtime_kv_capacity_tokens")
            != expected_capacity_tokens
            or row_receipt.get("effective_request_limit")
            != trace.get("effective_request_limit")
            or row_receipt.get("policy") != expected_receipt_policy
            or row.get("kv_allocation_id") != row_receipt.get("kv_allocation_id")
        ):
            raise RuntimeError(
                f"{role} lifetime runtime-memory receipt is inconsistent"
            )
        _validate_receipt_policy_binding(
            expected_policy,
            row_receipt,
            trusted_geometry=trusted_geometry,
            expected_capacity_tokens=expected_capacity_tokens,
            expected_effective_request_limit=expected_effective_request_limit,
        )
        if role == "measured" and dict(row_receipt) != dict(receipt):
            raise RuntimeError(
                "measured lifetime receipt does not match the top-level receipt"
            )
        if row.get("selected_token_ids") != trace.get("selected_token_ids"):
            raise RuntimeError(f"{role} lifetime selected token IDs are inconsistent")
        if row.get("step_top1_token_ids") != trace.get("step_top1_token_ids"):
            raise RuntimeError(f"{role} lifetime top-1 token IDs are inconsistent")

        parsed_phase_rows[role] = _validate_lifetime_phase_samples(
            row,
            sampler_identity=sampler_identity,
            role=role,
        )
        endpoint_bindings[role] = _validate_lifetime_endpoint_bindings(
            row,
            parsed_phase_rows[role],
            sampler_identity=sampler_identity,
            role=role,
        )
        before_load = _parse_attributed_memory_sample(
            row.get("before_load"),
            sampler_pid=int(sampler_identity["pid"]),
            sampler_device=None,
            require_phase=False,
        )
        after_unload = _parse_attributed_memory_sample(
            row.get("after_unload"),
            sampler_pid=int(sampler_identity["pid"]),
            sampler_device=None,
            require_phase=False,
        )
        retained_process = int(after_unload["process_used_bytes"]) - int(
            before_load["process_used_bytes"]
        )
        retained_device = int(after_unload["used_bytes"]) - int(
            before_load["used_bytes"]
        )
        if (
            row.get("retained_bytes") != retained_process
            or row.get("device_wide_retained_bytes") != retained_device
        ):
            raise RuntimeError(
                f"{role} lifetime retained-memory accounting is inconsistent"
            )
        parsed_boundaries[role] = {
            "before_load": before_load,
            "after_unload": after_unload,
        }

    stable_receipt_fields = (
        "policy",
        "policy_fraction",
        "requested_kv_bytes",
        "safety_reserve_bytes",
        "model_context_limit",
        "prefill_chunk_limit",
        "request_context_limit",
        "runtime_kv_capacity_tokens",
        "effective_request_limit",
        "kv_bytes_per_token",
        "serialized_plan_bytes",
        "resident_weight_bytes",
        "engine_weight_bytes",
        "context_device_memory_bytes",
        "ordinary_device_input_bytes",
        "ordinary_device_output_bytes",
        "external_device_output_bytes",
        "graph_private_device_bytes",
        "kv_reserved_bytes",
        "kv_committed_bytes",
    )
    cold_receipt = warmup["runtime_memory_receipt"]
    for field in stable_receipt_fields:
        if field not in cold_receipt or field not in receipt:
            raise RuntimeError(
                f"cold/measured receipts are missing stable contract field {field}"
            )
        if cold_receipt[field] != receipt[field]:
            raise RuntimeError(
                f"cold/measured receipts disagree on stable contract field {field}"
            )

    output_equivalence = _validate_cold_warm_output_equivalence(
        trace,
        measured_logits,
    )

    cold_recovery = _signed_memory_attribution(
        parsed_boundaries["warmup"]["before_load"],
        parsed_boundaries["warmup"]["after_unload"],
        positive_unlisted_external_limit_bytes=(
            _COLD_START_PERSISTENT_DRIVER_LIMIT_BYTES
        ),
    )
    cold_driver_retained_bytes = max(
        0,
        int(cold_recovery["unlisted_external_growth_bytes"]),
    )
    cold_exclusive_compute_ledger = all(
        int(sample["other_compute_process_used_bytes"]) == 0
        for sample in (
            parsed_boundaries["warmup"]["before_load"],
            *parsed_phase_rows["warmup"],
            parsed_boundaries["warmup"]["after_unload"],
        )
    )
    cold_driver_retention_passed = bool(
        cold_exclusive_compute_ledger
        and cold_driver_retained_bytes
        <= _COLD_START_PERSISTENT_DRIVER_LIMIT_BYTES
    )
    if not cold_recovery["passed"] or not cold_driver_retention_passed:
        raise RuntimeError(
            "cold-start unload memory does not reconcile within the explicit "
            "process/driver bounds: "
            f"attribution={cold_recovery}, "
            f"exclusive_compute_ledger={cold_exclusive_compute_ledger}"
        )

    measured_recovery = _signed_memory_attribution(
        parsed_boundaries["measured"]["before_load"],
        parsed_boundaries["measured"]["after_unload"],
        negative_unlisted_external_limit_bytes=cold_driver_retained_bytes,
    )
    if not measured_recovery["passed"]:
        raise RuntimeError(
            "measured unload memory does not reconcile: "
            f"{measured_recovery}"
        )
    measured_driver_release_bytes = max(
        0,
        -int(measured_recovery["unlisted_external_growth_bytes"]),
    )

    cold_peak = _reconcile_lifetime_peak_with_nvml(
        warmup,
        sampler_identity=sampler_identity,
        role="cold_start",
        expected_persistent_unlisted_external_growth_bytes=(
            cold_driver_retained_bytes
        ),
        persistent_unlisted_external_limit_bytes=(
            _COLD_START_PERSISTENT_DRIVER_LIMIT_BYTES
        ),
    )
    measured_peak = _reconcile_lifetime_peak_with_nvml(
        measured,
        sampler_identity=sampler_identity,
        role="measured",
        expected_persistent_unlisted_external_growth_bytes=(
            -measured_driver_release_bytes
        ),
        negative_persistent_unlisted_external_limit_bytes=(
            cold_driver_retained_bytes
        ),
    )

    resident_weight_bytes = cold_receipt.get("resident_weight_bytes")
    if type(resident_weight_bytes) is not int or resident_weight_bytes <= 0:
        raise RuntimeError("cold-start receipt has no resident-weight accounting")
    cold_retention_limit = max(
        _COLD_START_RETENTION_FLOOR_BYTES,
        math.ceil(
            _COLD_START_RETENTION_WEIGHT_FRACTION * resident_weight_bytes
        ),
    )
    cold_retained_process = int(warmup["retained_bytes"])
    cold_retained_device = int(warmup["device_wide_retained_bytes"])
    measured_retained_process = int(measured["retained_bytes"])
    measured_retained_device = int(measured["device_wide_retained_bytes"])
    cold_retention_passed = bool(
        abs(cold_retained_process) <= cold_retention_limit
    )
    measured_retention_passed = bool(
        abs(measured_retained_process)
        <= _MEMORY_ATTRIBUTION_FLOOR_BYTES
        and measured_driver_release_bytes
        <= cold_driver_retained_bytes + int(measured_recovery["tolerance_bytes"])
    )
    if not cold_retention_passed:
        raise RuntimeError(
            "cold-start retained memory exceeds its explicit bound: "
            f"process={cold_retained_process}, "
            f"limit={cold_retention_limit}"
        )
    if not measured_retention_passed:
        raise RuntimeError(
            "measured lifetime retained memory exceeds its explicit bound: "
            f"process={measured_retained_process}, "
            f"driver_release={measured_driver_release_bytes}, "
            f"limit={_MEMORY_ATTRIBUTION_FLOOR_BYTES}"
        )

    continuity = _signed_memory_attribution(
        parsed_boundaries["warmup"]["after_unload"],
        parsed_boundaries["measured"]["before_load"],
    )
    if not continuity["passed"]:
        raise RuntimeError(
            "warmup unload to measured before-load memory continuity does not reconcile: "
            f"{continuity}"
        )

    peak_memory_reconciliation = {
        **measured_peak,
        "schema_version": 2,
        "reconciliation_basis": "cold_start_and_measured_lifetimes",
        "cold_start_execution_ordinal": 0,
        "measured_execution_ordinal": 1,
        "warmup_excluded_from_measured_peak": True,
        "warmup_independently_hard_gated": True,
        "cold_start_reconciliation": cold_peak,
        "measured_reconciliation": measured_peak,
        "warmup_continuity_reconciliation": {
            **continuity,
            "baseline_sample": parsed_boundaries["warmup"]["after_unload"],
            "boundary_sample": parsed_boundaries["measured"]["before_load"],
        },
        "passed": bool(cold_peak["passed"] and measured_peak["passed"]),
    }
    return {
        "schema_version": 2,
        "status": "passed",
        "passed": True,
        "lifetime_protocol": dict(_LIFETIME_PROTOCOL),
        "typed_policy": expected_policy,
        "runtime_kv_capacity_tokens": expected_capacity_tokens,
        **expected_workload,
        "reconciliation_basis": "cold_start_and_measured_lifetimes",
        "warmup_excluded_from_measured_peak": True,
        "warmup_independently_hard_gated": True,
        "cold_start_output_equivalence": output_equivalence,
        "sampler_identity": sampler_identity,
        "lifetime_endpoint_bindings": endpoint_bindings,
        "cold_start_peak_reconciliation": cold_peak,
        "measured_peak_reconciliation": measured_peak,
        "peak_memory_reconciliation": peak_memory_reconciliation,
        "cold_start_retention_gate": {
            "process_retained_bytes": cold_retained_process,
            "device_wide_retained_bytes": cold_retained_device,
            "limit_bytes": cold_retention_limit,
            "limit_rule": "max(512MiB,5pct_resident_weights)",
            "attribution": cold_recovery,
            "passed": cold_retention_passed,
        },
        "cold_start_persistent_driver_gate": {
            "unlisted_driver_retained_bytes": cold_driver_retained_bytes,
            "limit_bytes": _COLD_START_PERSISTENT_DRIVER_LIMIT_BYTES,
            "limit_rule": "2GiB",
            "requires_exclusive_compute_ledger": True,
            "exclusive_compute_ledger": cold_exclusive_compute_ledger,
            "peak_and_unload_delta_bound": True,
            "passed": cold_driver_retention_passed,
        },
        "measured_retention_gate": {
            "process_retained_bytes": measured_retained_process,
            "device_wide_retained_bytes": measured_retained_device,
            "cold_driver_release_bytes": measured_driver_release_bytes,
            "limit_bytes": _MEMORY_ATTRIBUTION_FLOOR_BYTES,
            "limit_rule": (
                "abs(process)<=64MiB and negative unlisted release "
                "cannot exceed cold driver retention"
            ),
            "attribution": measured_recovery,
            "passed": measured_retention_passed,
        },
        "warmup_phase_order": [
            sample["phase"] for sample in parsed_phase_rows["warmup"]
        ],
        "measured_phase_order": [
            sample["phase"] for sample in parsed_phase_rows["measured"]
        ],
        "continuity_reconciliation": {
            **continuity,
            "baseline_sample": parsed_boundaries["warmup"]["after_unload"],
            "boundary_sample": parsed_boundaries["measured"]["before_load"],
        },
    }


def reconcile_device_peak_with_nvml(
    trace: dict[str, Any],
    *,
    trusted_geometry: TrustedRuntimeGeometry,
    expected_sampler: SamplerTrustAnchor | None = None,
) -> dict[str, Any]:
    """Validate both cold and measured lifetime peaks and return combined evidence."""

    return validate_warmup_evidence(
        trace,
        trusted_geometry=trusted_geometry,
        expected_sampler=expected_sampler,
    )["peak_memory_reconciliation"]


def _all_gate_values_true(gates: Any) -> bool:
    if not isinstance(gates, Mapping) or not gates:
        return False
    for value in gates.values():
        if isinstance(value, Mapping):
            if not _all_gate_values_true(value):
                return False
        elif value is not True:
            return False
    return True


def _chunk_variant_producer_receipt_passed(
    evidence: Any,
    *,
    variant_bundle: Path | None,
    source_state: Mapping[str, Any],
) -> bool:
    """Reopen the validated C/2 producer summary at promotion time."""

    fields = {
        "path",
        "size_bytes",
        "sha256",
        "schema_version",
        "bundle",
        "producer",
        "runtime_kv_plugin",
        "runtime_kv_plugin_mapping",
        "build_manifest",
        "build_timing",
        "source_state_sha256",
        "git_head",
    }
    if (
        variant_bundle is None
        or not isinstance(evidence, Mapping)
        or set(evidence) != fields
        or evidence.get("schema_version") != CHUNK_VARIANT_BUILD_SCHEMA
        or evidence.get("git_head") != source_state.get("git_head")
        or evidence.get("source_state_sha256")
        != source_state.get("source_state_sha256")
    ):
        return False
    try:
        receipt_path = Path(str(evidence["path"]))
        _validate_file_identity(
            evidence,
            receipt_path,
            label="promoted chunk-variant build receipt",
        )
        _validate_file_identity(
            evidence["bundle"],
            variant_bundle,
            label="promoted chunk-variant bundle",
        )
        _validate_file_identity(
            evidence["producer"],
            REPO_ROOT
            / "tools"
            / "build_native_dynamic_memory_chunk_variant.py",
            label="promoted chunk-variant producer",
        )
        plugin = evidence["runtime_kv_plugin"]
        if not isinstance(plugin, Mapping):
            return False
        validated_plugin = _validate_file_identity(
            plugin,
            Path(str(plugin.get("path", ""))),
            label="promoted chunk-variant runtime-KV plugin",
            require_binary_metadata=True,
        )
        timing = evidence["build_timing"]
        if not isinstance(timing, Mapping):
            return False
        _validate_file_identity(
            timing,
            Path(str(timing.get("path", ""))),
            label="promoted chunk-variant build timing",
        )
        mapping = evidence["runtime_kv_plugin_mapping"]
        if (
            not isinstance(mapping, Mapping)
            or mapping.get("candidate_count") != 1
            or mapping.get("deleted_candidate_count") != 0
            or mapping.get("selected") != validated_plugin
        ):
            return False
        manifest = evidence["build_manifest"]
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("schema_version") != BUILD_MANIFEST_SCHEMA
            or manifest.get("git_head") != evidence.get("git_head")
            or manifest.get("source_state_sha256")
            != evidence.get("source_state_sha256")
        ):
            return False
        manifest_path = Path(str(manifest.get("path", "")))
        if (
            not manifest_path.is_absolute()
            or manifest_path.resolve(strict=True) != manifest_path
            or _sha256(manifest_path) != manifest.get("sha256")
        ):
            return False
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return True


def _persisted_warmup_evidence_passed(
    evidence: Any,
    *,
    trace: Any,
    trusted_geometry: TrustedRuntimeGeometry | None,
    expected_sampler: SamplerTrustAnchor | None,
    expected_lifetime_policy: Mapping[str, Any],
    expected_capacity_tokens: int,
    expected_final_kv_position: int,
    expected_prompt_tokens: int,
    expected_prefill_launches: int,
    expected_decode_launches: int,
) -> bool:
    """Replay raw trace/artifacts and require an exact derived-evidence match.

    Persisted ``passed`` booleans are never inputs to promotion.  The replay
    reopens both independent logits artifacts through
    :func:`validate_warmup_evidence`, revalidates every phase/receipt/ledger
    invariant, and then requires the complete normalized evidence payload to
    equal what was persisted in the report.
    """

    if (
        not isinstance(evidence, Mapping)
        or not isinstance(trace, Mapping)
        or not isinstance(trusted_geometry, TrustedRuntimeGeometry)
        or not isinstance(expected_sampler, SamplerTrustAnchor)
    ):
        return False
    try:
        replayed = validate_warmup_evidence(
            dict(trace),
            trusted_geometry=trusted_geometry,
            expected_sampler=expected_sampler,
            expected_lifetime_policy=expected_lifetime_policy,
            expected_capacity_tokens=expected_capacity_tokens,
            expected_final_kv_position=expected_final_kv_position,
            expected_prompt_tokens=expected_prompt_tokens,
            expected_prefill_launches=expected_prefill_launches,
            expected_decode_launches=expected_decode_launches,
        )
    except Exception:
        return False
    return dict(evidence) == replayed


def _persisted_case_warmup_evidence_passed(
    evidence: Any,
    *,
    trace: Any,
    case: Case,
    trusted_geometry: TrustedRuntimeGeometry | None,
    expected_sampler: SamplerTrustAnchor | None,
    expected_lifetime_policy: Mapping[str, Any] | None = None,
) -> bool:
    if (
        not isinstance(trace, Mapping)
        or not isinstance(trusted_geometry, TrustedRuntimeGeometry)
        or not isinstance(expected_sampler, SamplerTrustAnchor)
    ):
        return False
    receipt = trace.get("runtime_memory_receipt")
    chunk_limit = trace.get("prefill_chunk_limit")
    if (
        not isinstance(receipt, Mapping)
        or type(receipt.get("runtime_kv_capacity_tokens")) is not int
        or type(receipt.get("model_context_limit")) is not int
        or type(chunk_limit) is not int
        or chunk_limit <= 0
    ):
        return False
    if expected_lifetime_policy is None:
        cycles = trace.get("load_cycles")
        if (
            not isinstance(cycles, list)
            or len(cycles) != 1
            or not isinstance(cycles[0], Mapping)
            or not isinstance(cycles[0].get("policy"), Mapping)
        ):
            return False
        expected_lifetime_policy = cycles[0]["policy"]
    return _persisted_warmup_evidence_passed(
        evidence,
        trace=trace,
        trusted_geometry=trusted_geometry,
        expected_sampler=expected_sampler,
        expected_lifetime_policy=expected_lifetime_policy,
        expected_capacity_tokens=receipt["runtime_kv_capacity_tokens"],
        expected_final_kv_position=case.prompt_tokens + case.decode_tokens,
        expected_prompt_tokens=case.prompt_tokens,
        expected_prefill_launches=math.ceil(case.prompt_tokens / chunk_limit),
        expected_decode_launches=case.decode_tokens,
    )


def evaluate_qualification_outcome(
    *,
    canonical_cases: tuple[Case, ...],
    selected_cases: tuple[Case, ...],
    case_reports: Iterable[Mapping[str, Any]],
    skip_hf: bool,
    case_filter_used: bool,
    source_state_pre: Mapping[str, Any],
    source_state_post: Mapping[str, Any],
    context_memory_envelope: Mapping[str, Any],
    qualified_engine_graph: Mapping[str, Any],
    model_spec: ModelSpec,
    runner: Path,
    bundle: Path,
    runner_evidence_root: Path,
    thresholds: Mapping[str, float],
    trusted_geometry: TrustedRuntimeGeometry,
    sampler_anchors: Mapping[str, SamplerTrustAnchor],
    base_artifact_binding: Mapping[str, Any] | None = None,
    runtime_kv_plugin_binding: Mapping[str, Any] | None = None,
    variant_bundle: Path | None = None,
    trusted_variant_geometry: TrustedRuntimeGeometry | None = None,
    variant_build_receipt: Mapping[str, Any] | None = None,
    qualified_variant_engine_graph: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Separate canonical qualification from developer diagnostics.

    This function is intentionally pure so report promotion cannot depend on
    CLI return-code conventions or incremental report writes.
    """
    reports = tuple(case_reports)
    canonical_names = tuple(case.name for case in canonical_cases)
    selected_names = tuple(case.name for case in selected_cases)
    reported_names = tuple(str(report.get("name", "")) for report in reports)
    canonical_matrix_complete = (
        selected_names == canonical_names and reported_names == canonical_names
    )
    case_execution_passed = (
        len(reports) == len(selected_cases)
        and reported_names == selected_names
        and all(report.get("execution_passed") is True for report in reports)
    )

    report_by_name = {str(report.get("name", "")): report for report in reports}
    parity_states: dict[str, str] = {}
    warmup_states: dict[str, str] = {}
    admission_states: dict[str, str] = {}
    raw_capture_states: dict[str, str] = {}
    selected_diagnostics_passed = case_execution_passed
    for case in selected_cases:
        report = report_by_name.get(case.name, {})
        expected_tokens = deterministic_token_ids(
            case.prompt_tokens,
            model_spec.vocab_size,
        )
        base_evidence_dir = (
            runner_evidence_root.resolve() / case.name / "base"
        )
        base_replay: dict[str, Any] | None = None
        variant_replay: dict[str, Any] | None = None
        try:
            runner_evidence = report.get("runner_evidence")
            if (
                not isinstance(runner_evidence, Mapping)
                or runner_evidence.get("base") != str(base_evidence_dir)
            ):
                raise RuntimeError(
                    f"{case.name}: report does not bind the trusted base capture path"
                )
            base_replay = replay_runner_capture(
                base_evidence_dir,
                expected_command=_runner_command(
                    runner=runner,
                    bundle=bundle,
                    token_path=base_evidence_dir / "tokens.txt",
                    logits_path=base_evidence_dir / "runner-logits.bin",
                    case=case,
                    context_limit=trusted_geometry.model_context_limit,
                ),
                expected_tokens=expected_tokens,
                expected_returncode=(
                    3 if case.expect_admission_rejection else 0
                ),
                expected_trace=report.get("trace"),
                case=case,
                model_spec=model_spec,
                trusted_geometry=trusted_geometry,
                expected_sampler=sampler_anchors[f"{case.name}/base"],
            )
            if variant_bundle is not None:
                if trusted_variant_geometry is None:
                    raise RuntimeError("trusted chunk-variant geometry is missing")
                variant_evidence_dir = (
                    runner_evidence_root.resolve()
                    / case.name
                    / "chunk-variant"
                )
                if runner_evidence.get("chunk_variant") != str(
                    variant_evidence_dir
                ):
                    raise RuntimeError(
                        f"{case.name}: report does not bind the trusted "
                        "chunk-variant capture path"
                    )
                chunk_variant_report = report.get("chunk_variant")
                if not isinstance(chunk_variant_report, Mapping):
                    raise RuntimeError(
                        f"{case.name}: chunk-variant report is missing"
                    )
                variant_replay = replay_runner_capture(
                    variant_evidence_dir,
                    expected_command=_runner_command(
                        runner=runner,
                        bundle=variant_bundle,
                        token_path=variant_evidence_dir / "tokens.txt",
                        logits_path=(
                            variant_evidence_dir / "runner-logits.bin"
                        ),
                        case=case,
                        context_limit=(
                            trusted_variant_geometry.model_context_limit
                        ),
                    ),
                    expected_tokens=expected_tokens,
                    expected_returncode=(
                        3 if case.expect_admission_rejection else 0
                    ),
                    expected_trace=chunk_variant_report.get("trace"),
                    case=case,
                    model_spec=model_spec,
                    trusted_geometry=trusted_variant_geometry,
                    expected_sampler=sampler_anchors[
                        f"{case.name}/chunk-variant"
                    ],
                )
            raw_capture_states[case.name] = "passed"
        except (KeyError, RuntimeError, ValueError, OSError):
            raw_capture_states[case.name] = "failed"
            selected_diagnostics_passed = False
        if case.expect_admission_rejection:
            parity_states[case.name] = "not_applicable"
            warmup = report.get("warmup_evidence")
            warmup_states[case.name] = (
                "not_applicable"
                if isinstance(warmup, Mapping)
                and warmup.get("status") == "not_applicable"
                else "failed"
            )
            trace = report.get("trace")
            admission_passed = bool(
                base_replay is not None
                and
                report.get("admission_rejected_before_attention") is True
                and _admission_trace_passed(trace, label=case.name)
                and (
                    variant_bundle is None
                    or variant_replay is not None
                )
            )
            chunk_variant = report.get("chunk_variant")
            if isinstance(chunk_variant, Mapping):
                variant_trace = chunk_variant.get("trace")
                admission_passed = bool(
                    admission_passed
                    and chunk_variant.get("admission_rejected_before_attention")
                    is True
                    and _admission_trace_passed(
                        variant_trace,
                        label=f"{case.name} chunk variant",
                    )
                )
            admission_states[case.name] = (
                "passed" if admission_passed else "failed"
            )
            if (
                warmup_states[case.name] != "not_applicable"
                or not admission_passed
            ):
                selected_diagnostics_passed = False
        else:
            admission_states[case.name] = "not_applicable"
            parity = report.get("parity")
            warmup = report.get("warmup_evidence")
            chunk_variant = report.get("chunk_variant")
            warmup_passed = False
            parity_state = "failed"
            try:
                if base_replay is None:
                    raise RuntimeError(f"{case.name}: base raw replay failed")
                base_validation = base_replay["validation_evidence"]
                base_logits = base_replay["logits"]
                if (
                    not isinstance(base_validation, Mapping)
                    or not isinstance(base_logits, np.ndarray)
                    or not isinstance(warmup, Mapping)
                    or warmup.get("status") != "passed"
                    or warmup.get("passed") is not True
                    or warmup.get("base")
                    != base_validation.get("warmup_evidence")
                ):
                    raise RuntimeError(
                        f"{case.name}: persisted warmup evidence differs from replay"
                    )
                output_dir = runner_evidence_root.resolve().parent
                trt_path, _ = _read_report_artifact(
                    report.get("trt_logits_artifact"),
                    report.get("trt_logits_sha256"),
                    output_dir=output_dir,
                    label=f"{case.name} TRT logits",
                )
                report_trt_logits = read_logits_artifact(trt_path)
                if not np.array_equal(report_trt_logits, base_logits):
                    raise RuntimeError(
                        f"{case.name}: report TRT logits differ from raw capture"
                    )

                if skip_hf:
                    if (
                        not isinstance(parity, Mapping)
                        or parity.get("status") != "not_run"
                        or "passed" in parity
                    ):
                        raise RuntimeError(
                            f"{case.name}: skipped parity summary is invalid"
                        )
                    parity_state = "not_run"
                else:
                    hf_path, _ = _read_report_artifact(
                        report.get("hf_logits_artifact"),
                        report.get("hf_logits_sha256"),
                        output_dir=output_dir,
                        label=f"{case.name} HF logits",
                    )
                    hf_logits = np.load(hf_path, allow_pickle=False)
                    replayed_parity = compare_logits(
                        base_logits,
                        hf_logits,
                        base_replay["trace"]["selected_token_ids"],
                        dict(thresholds),
                    )
                    replayed_parity["status"] = (
                        "passed" if replayed_parity["passed"] else "failed"
                    )
                    if (
                        not isinstance(parity, Mapping)
                        or dict(parity) != replayed_parity
                    ):
                        raise RuntimeError(
                            f"{case.name}: parity summary differs from artifact replay"
                        )
                    parity_state = (
                        "passed"
                        if replayed_parity["passed"] is True
                        else "failed"
                    )

                if variant_bundle is not None:
                    if (
                        variant_replay is None
                        or not isinstance(chunk_variant, Mapping)
                        or not isinstance(
                            variant_replay.get("validation_evidence"),
                            Mapping,
                        )
                        or chunk_variant.get("warmup_evidence")
                        != variant_replay["validation_evidence"].get(
                            "warmup_evidence"
                        )
                    ):
                        raise RuntimeError(
                            f"{case.name}: chunk-variant warmup replay differs"
                        )
                    variant_logits = variant_replay.get("logits")
                    if not isinstance(variant_logits, np.ndarray):
                        raise RuntimeError(
                            f"{case.name}: chunk-variant logits are missing"
                        )
                    variant_path, _ = _read_report_artifact(
                        chunk_variant.get("trt_logits_artifact"),
                        chunk_variant.get("trt_logits_sha256"),
                        output_dir=output_dir,
                        label=f"{case.name} chunk-variant TRT logits",
                    )
                    if not np.array_equal(
                        read_logits_artifact(variant_path),
                        variant_logits,
                    ):
                        raise RuntimeError(
                            f"{case.name}: chunk-variant report logits differ"
                        )
                    replayed_chunk_parity = compare_logits(
                        variant_logits,
                        base_logits,
                        variant_replay["trace"]["selected_token_ids"],
                        dict(thresholds),
                    )
                    if (
                        chunk_variant.get("base_vs_variant_parity")
                        != replayed_chunk_parity
                        or chunk_variant.get("passed")
                        is not replayed_chunk_parity["passed"]
                    ):
                        raise RuntimeError(
                            f"{case.name}: chunk parity summary differs from replay"
                        )
                warmup_passed = True
            except (RuntimeError, ValueError, TypeError, OSError):
                selected_diagnostics_passed = False
            parity_states[case.name] = parity_state
            warmup_states[case.name] = "passed" if warmup_passed else "failed"
            if not warmup_passed:
                selected_diagnostics_passed = False
        chunk_variant = report.get("chunk_variant")
        if isinstance(chunk_variant, Mapping) and chunk_variant.get("passed") is not True:
            selected_diagnostics_passed = False

    hf_parity_executed_and_passed = bool(
        not skip_hf
        and canonical_matrix_complete
        and all(
            case.expect_admission_rejection or parity_states.get(case.name) == "passed"
            for case in canonical_cases
        )
    )

    pre_head = source_state_pre.get("git_head")
    post_head = source_state_post.get("git_head")
    pre_source = source_state_pre.get("source_state_sha256")
    post_source = source_state_post.get("source_state_sha256")
    source_state_unchanged = bool(
        isinstance(pre_head, str)
        and pre_head
        and pre_head == post_head
        and isinstance(pre_source, str)
        and pre_source
        and pre_source == post_source
    )
    source_clean_exact_head = bool(
        source_state_unchanged
        and source_state_pre.get("git_dirty") is False
        and source_state_post.get("git_dirty") is False
        and source_state_pre.get("exact_head_gate_satisfied") is True
        and source_state_post.get("exact_head_gate_satisfied") is True
    )
    base_artifact_binding_passed = _base_artifact_binding_passed(
        base_artifact_binding,
        bundle=bundle,
        runner=runner,
        spec=model_spec,
        source_state=source_state_pre,
    )
    runtime_kv_plugin_binding_passed = (
        _runtime_kv_plugin_binding_passed(
            runtime_kv_plugin_binding,
            base_artifact_binding=base_artifact_binding,
        )
    )

    context_gates = context_memory_envelope.get("gates")
    full_context_memory_coverage = bool(
        context_memory_envelope.get("status") == "passed"
        and context_memory_envelope.get("passed") is True
        and context_memory_envelope.get("coverage_required") is True
        and _all_gate_values_true(context_gates)
    )
    diagnostic_context_memory_passed = bool(context_memory_envelope.get("passed") is True)

    runtime_stack = qualified_engine_graph.get("runtime_stack")
    qualified_engine_graph_passed = bool(
        qualified_engine_graph.get("passed") is True
        and isinstance(runtime_stack, Mapping)
        and bool(runtime_stack)
        and _all_gate_values_true(qualified_engine_graph.get("gates"))
    )
    variant_runtime_stack = (
        qualified_variant_engine_graph.get("runtime_stack")
        if isinstance(qualified_variant_engine_graph, Mapping)
        else None
    )
    c_div_2_variant_engine_graph_passed = bool(
        variant_bundle is not None
        and trusted_variant_geometry is not None
        and isinstance(qualified_variant_engine_graph, Mapping)
        and qualified_variant_engine_graph.get("passed") is True
        and isinstance(variant_runtime_stack, Mapping)
        and bool(variant_runtime_stack)
        and _all_gate_values_true(
            qualified_variant_engine_graph.get("gates")
        )
    )
    c_div_2_variant_producer_receipt_passed = (
        _chunk_variant_producer_receipt_passed(
            variant_build_receipt,
            variant_bundle=variant_bundle,
            source_state=source_state_pre,
        )
    )
    warmup_evidence_passed = bool(
        len(reports) == len(selected_cases)
        and all(
            warmup_states.get(case.name)
            == ("not_applicable" if case.expect_admission_rejection else "passed")
            for case in selected_cases
        )
    )
    admission_rejection_evidence_passed = bool(
        len(reports) == len(selected_cases)
        and all(
            admission_states.get(case.name)
            == ("passed" if case.expect_admission_rejection else "not_applicable")
            for case in selected_cases
        )
    )
    raw_runner_evidence_passed = bool(
        len(reports) == len(selected_cases)
        and all(
            raw_capture_states.get(case.name) == "passed"
            for case in selected_cases
        )
    )

    qualification_gates = {
        "canonical_matrix_complete": canonical_matrix_complete,
        "case_filter_not_used": not case_filter_used,
        "case_execution_passed": case_execution_passed,
        "raw_runner_evidence_passed": raw_runner_evidence_passed,
        "hf_parity_executed_and_passed": hf_parity_executed_and_passed,
        "source_state_unchanged": source_state_unchanged,
        "source_clean_exact_head": source_clean_exact_head,
        "base_artifact_binding_passed": (
            base_artifact_binding_passed
        ),
        "runtime_kv_plugin_binding_passed": (
            runtime_kv_plugin_binding_passed
        ),
        "full_context_memory_coverage": full_context_memory_coverage,
        "qualified_engine_graph_passed": qualified_engine_graph_passed,
        "c_div_2_variant_engine_graph_passed": (
            c_div_2_variant_engine_graph_passed
        ),
        "c_div_2_variant_producer_receipt_passed": (
            c_div_2_variant_producer_receipt_passed
        ),
        "warmup_evidence_passed": warmup_evidence_passed,
        "admission_rejection_evidence_passed": (
            admission_rejection_evidence_passed
        ),
    }
    passed = all(qualification_gates.values())
    execution_passed = bool(
        case_execution_passed
        and raw_runner_evidence_passed
        and source_state_unchanged
        and diagnostic_context_memory_passed
        and qualified_engine_graph_passed
    )
    diagnostic_passed = bool(execution_passed and selected_diagnostics_passed)
    status = "passed" if passed else "diagnostic_passed" if diagnostic_passed else "failed"
    return {
        "passed": passed,
        "promotion_eligible": passed,
        "diagnostic_passed": diagnostic_passed,
        "execution_passed": execution_passed,
        "status": status,
        "qualification_gates": qualification_gates,
        "qualification_blockers": [
            name for name, value in qualification_gates.items() if not value
        ],
        "parity_execution": parity_states,
        "raw_runner_evidence": {
            "status": "passed" if raw_runner_evidence_passed else "failed",
            "passed": raw_runner_evidence_passed,
            "case_states": raw_capture_states,
        },
        "warmup_evidence": {
            "status": "passed" if warmup_evidence_passed else "failed",
            "passed": warmup_evidence_passed,
            "case_states": warmup_states,
        },
        "admission_rejection_evidence": {
            "status": (
                "passed" if admission_rejection_evidence_passed else "failed"
            ),
            "passed": admission_rejection_evidence_passed,
            "case_states": admission_states,
        },
    }


def _write_qualification_report(report_path: Path, report: Mapping[str, Any]) -> None:
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@contextlib.contextmanager
def qualification_failure_checkpoint(
    *,
    report: dict[str, Any],
    report_path: Path,
    repo_root: Path,
    output_dir: Path,
):
    """Persist a complete failed checkpoint with the copied runner evidence."""
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - evidence must survive every producer failure.
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "stage": report.get("stage", "unknown"),
        }
        cases = report.get("cases")
        if isinstance(cases, list) and cases:
            active_case = cases[-1]
            if isinstance(active_case, dict) and active_case.get("status") == "running":
                failure["stage"] = active_case.get("stage", failure["stage"])
                active_case.update(
                    {
                        "status": "failed",
                        "passed": False,
                        "diagnostic_passed": False,
                        "qualification_passed": False,
                        "execution_passed": False,
                        "failure": failure,
                    }
                )
        report.update(
            {
                "status": "failed",
                "passed": False,
                "promotion_eligible": False,
                "diagnostic_passed": False,
                "execution_passed": False,
                "case_diagnostics_passed_so_far": False,
                "qualification_blockers": ["execution_error"],
                "failure": failure,
            }
        )
        try:
            source_state_post = source_state_provenance(
                repo_root,
                Path(__file__),
                output_dir,
                label="post-failure",
            )
            report["source_state_post"] = source_state_post
            source_state_pre = report.get("source_state_pre")
            report["source_state_unchanged"] = bool(
                isinstance(source_state_pre, Mapping)
                and source_state_pre.get("git_head") == source_state_post.get("git_head")
                and source_state_pre.get("source_state_sha256")
                == source_state_post.get("source_state_sha256")
            )
        except Exception as source_exc:  # noqa: BLE001
            report["source_state_post_error"] = {
                "type": type(source_exc).__name__,
                "message": str(source_exc),
            }
            report["source_state_unchanged"] = False
        _write_qualification_report(report_path, report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--model", required=True, help="Pinned HF snapshot path used to build the bundle"
    )
    parser.add_argument(
        "--model-revision",
        help=(
            "Required immutable commit SHA when --model is a remote model ID; "
            "must equal the bundle qualification"
        ),
    )
    parser.add_argument(
        "--runner", type=Path, default=Path("build-dynkv/trtmc_dynamic_memory_qualify")
    )
    parser.add_argument(
        "--build-manifest",
        type=Path,
        help=(
            "Passed trtmc.dynamic-memory-test-manifest/v2 from the exact "
            "clean HEAD that built the qualifier and runtime DSOs. Required "
            "with --base-build-receipt for canonical qualification; omission "
            "limits the result to diagnostic-only."
        ),
    )
    parser.add_argument(
        "--base-build-receipt",
        type=Path,
        help=(
            "Fresh native-dynamic bundle build receipt emitted by "
            "capture_native_dynamic_memory_perf.py. Required with "
            "--build-manifest for canonical qualification."
        ),
    )
    parser.add_argument(
        "--chunk-variant-bundle",
        type=Path,
        help=(
            "Bundle built from the same qualified tuple with prefill C/2. "
            "Required, with its producer receipt, for canonical qualification; "
            "omission limits the result to diagnostic-only."
        ),
    )
    parser.add_argument(
        "--chunk-variant-build-receipt",
        type=Path,
        help=(
            "Source-bound build receipt emitted by "
            "build_native_dynamic_memory_chunk_variant.py. Required with "
            "--chunk-variant-bundle and for canonical qualification."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run one named matrix case (repeatable); default is all",
    )
    parser.add_argument(
        "--runner-cuda-visible-device",
        required=True,
        type=_runner_cuda_visible_device_arg,
        metavar="SELECTOR",
        help=(
            "Expose exactly one numeric physical GPU index or full GPU UUID "
            "to each qualification runner child. The producer keeps the "
            "incoming CUDA_VISIBLE_DEVICES."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--skip-hf", action="store_true", help="Run boundary/trace checks without claiming parity"
    )
    args = parser.parse_args()

    repo_root = REPO_ROOT
    bundle = args.bundle.resolve()
    runner = args.runner.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    header = _read_bundle_header(bundle)
    spec = _resolve_spec(header)
    variant_bundle = (
        args.chunk_variant_bundle.resolve() if args.chunk_variant_bundle is not None else None
    )
    variant_build_receipt_path = (
        args.chunk_variant_build_receipt.resolve()
        if args.chunk_variant_build_receipt is not None
        else None
    )
    if (variant_bundle is None) != (variant_build_receipt_path is None):
        raise ValueError(
            "--chunk-variant-bundle and --chunk-variant-build-receipt must be provided together"
        )
    variant_chunk_limit = None
    variant_header = None
    if variant_bundle is not None:
        require_developer_chunk_variant_opt_in()
        variant_header = _read_bundle_header(variant_bundle)
        variant_chunk_limit = _validate_chunk_variant(header, variant_header, spec)
    canonical_cases = _cases_for(spec)
    cases = _select_cases(canonical_cases, args.case)
    vocab_size = int(header.get("vocab_size", 0))
    if vocab_size <= 1:
        raise ValueError("bundle header has no valid vocab_size")
    contract = header["runtime_memory"]
    hf_reference_proof = verify_hf_reference(
        args.model,
        contract,
        remote_revision=args.model_revision,
    )
    thresholds = _load_thresholds(repo_root, spec)
    source_state_pre = source_state_provenance(
        repo_root,
        Path(__file__),
        output_dir,
        label="pre",
    )
    if (args.build_manifest is None) != (
        args.base_build_receipt is None
    ):
        raise ValueError(
            "--build-manifest and --base-build-receipt must be provided "
            "together"
        )
    base_artifact_binding = None
    runtime_kv_plugin_binding = None
    if args.build_manifest is not None:
        assert args.base_build_receipt is not None
        base_artifact_binding = _validate_base_artifact_binding(
            build_manifest_path=args.build_manifest,
            base_build_receipt_path=args.base_build_receipt,
            bundle=bundle,
            runner=runner,
            spec=spec,
            source_state=source_state_pre,
        )
        runtime_kv_plugin_binding = (
            _bind_runtime_kv_plugin_from_base_artifacts(
                base_artifact_binding
            )
        )
    variant_build_receipt = None
    if variant_bundle is not None:
        assert variant_header is not None
        assert variant_build_receipt_path is not None
        variant_build_receipt = _validate_chunk_variant_build_receipt(
            receipt_path=variant_build_receipt_path,
            variant_bundle=variant_bundle,
            base_header=header,
            variant_header=variant_header,
            spec=spec,
            source_state=source_state_pre,
        )
    qualified_engine_graph = inspect_qualified_bundle_engines(
        bundle,
        header,
        spec,
        output_dir,
        artifact_prefix="base.",
    )
    if runtime_kv_plugin_binding is not None:
        runtime_kv_plugin_binding = (
            _finalize_runtime_kv_plugin_binding(
                runtime_kv_plugin_binding
            )
        )
    variant_engine_graph = None
    if variant_bundle is not None:
        assert variant_header is not None
        assert variant_chunk_limit is not None
        variant_contract = variant_header["runtime_memory"]
        variant_engine_graph = inspect_qualified_bundle_engines(
            variant_bundle,
            variant_header,
            spec,
            output_dir,
            artifact_prefix="c-div-2.",
            chunk_limit=variant_chunk_limit,
            buckets=tuple(int(value) for value in variant_contract["active_kv_profile_limits"]),
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "model_id": spec.model_id,
        "bundle": str(bundle),
        "bundle_sha256": _sha256(bundle),
        "runner": str(runner),
        "runner_sha256": _sha256(runner),
        "hf_reference": hf_reference_proof,
        "model_context_limit": spec.context_limit,
        "prefill_chunk_limit": spec.chunk_limit,
        "vocab_size": vocab_size,
        "threshold_source": spec.threshold_path,
        "source_state": source_state_pre,
        "source_state_pre": source_state_pre,
        "base_artifact_binding": base_artifact_binding,
        "runtime_kv_plugin_binding": runtime_kv_plugin_binding,
        "qualified_engine_graph": qualified_engine_graph,
        "canonical_case_names": [case.name for case in canonical_cases],
        "selected_case_names": [case.name for case in cases],
        "case_filter_used": bool(args.case),
        "hf_parity_requested": not args.skip_hf,
        "passed": False,
        "promotion_eligible": False,
        "diagnostic_passed": False,
        "execution_passed": False,
        "status": "running",
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "producer_CUDA_VISIBLE_DEVICES": os.environ.get(
                "CUDA_VISIBLE_DEVICES"
            ),
            "runner_CUDA_VISIBLE_DEVICES": (
                args.runner_cuda_visible_device
            ),
            DEVELOPER_CHUNK_VARIANT_ENV: os.environ.get(DEVELOPER_CHUNK_VARIANT_ENV),
        },
        "cases": [],
    }
    if variant_bundle is not None:
        assert variant_build_receipt is not None
        assert variant_engine_graph is not None
        report["developer_chunk_variant"] = {
            "bundle": str(variant_bundle),
            "bundle_sha256": _sha256(variant_bundle),
            "prefill_chunk_limit": variant_chunk_limit,
            "comparison": "base-C versus developer-C/2",
            "build_receipt": variant_build_receipt,
            "qualified_engine_graph": variant_engine_graph,
        }

    hf_model = None
    if not args.skip_hf and any(not case.expect_admission_rejection for case in cases):
        hf_model = load_hf_model(
            args.model,
            _hf_dtype_name(str(contract["kv_dtype"])),
            args.device,
            revision=args.model_revision,
        )

    report_path = output_dir / "qualification-report.json"
    _write_qualification_report(report_path, report)
    all_case_diagnostics_passed = True
    sampler_anchors: dict[str, SamplerTrustAnchor] = {}
    runner_evidence_root = output_dir / "runner-evidence"
    with qualification_failure_checkpoint(
        report=report,
        report_path=report_path,
        repo_root=repo_root,
        output_dir=output_dir,
    ):
        runner_evidence_root.mkdir(exist_ok=False)
        for case in cases:
            print(
                f"[qualification] {case.name}: prompt={case.prompt_tokens} "
                f"decode={case.decode_tokens}",
                file=sys.stderr,
                flush=True,
            )
            tokens = deterministic_token_ids(case.prompt_tokens, vocab_size)
            case_report: dict[str, Any] = {
                "name": case.name,
                "prompt_tokens": case.prompt_tokens,
                "decode_tokens": case.decode_tokens,
                "expect_admission_rejection": (case.expect_admission_rejection),
                "input_token_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
                "status": "running",
                "stage": "base_runner",
                "execution_passed": False,
                "runner_evidence": {
                    "base": str(runner_evidence_root / case.name / "base"),
                },
            }
            report["cases"].append(case_report)
            _write_qualification_report(report_path, report)
            trace, trt_logits, runner_stderr, base_sampler_anchor = run_trt_case(
                runner=runner,
                bundle=bundle,
                tokens=tokens,
                case=case,
                context_limit=spec.context_limit,
                evidence_dir=runner_evidence_root / case.name / "base",
                runner_cuda_visible_device=(
                    args.runner_cuda_visible_device
                ),
            )
            case_report["trace"] = trace
            sampler_anchors[f"{case.name}/base"] = base_sampler_anchor
            case_report["sampler_trust_anchor"] = {
                "pid": base_sampler_anchor.pid,
                "cuda_logical_device_index": (
                    base_sampler_anchor.cuda_logical_device_index
                ),
                "physical_device_index": (
                    base_sampler_anchor.physical_device_index
                ),
                "pci_bus_id": base_sampler_anchor.pci_bus_id,
                "gpu_uuid": base_sampler_anchor.gpu_uuid,
            }
            case_report["execution_passed"] = True
            case_report["stage"] = "chunk_variant_runner"
            variant_trace: dict[str, Any] | None = None
            variant_logits: np.ndarray | None = None
            variant_sampler_anchor: SamplerTrustAnchor | None = None
            variant_stderr = ""
            if variant_bundle is not None:
                case_report["runner_evidence"]["chunk_variant"] = str(
                    runner_evidence_root / case.name / "chunk-variant"
                )
                (
                    variant_trace,
                    variant_logits,
                    variant_stderr,
                    variant_sampler_anchor,
                ) = run_trt_case(
                    runner=runner,
                    bundle=variant_bundle,
                    tokens=tokens,
                    case=case,
                    context_limit=spec.context_limit,
                    evidence_dir=(runner_evidence_root / case.name / "chunk-variant"),
                    runner_cuda_visible_device=(
                        args.runner_cuda_visible_device
                    ),
                )
                assert variant_sampler_anchor is not None
                sampler_anchors[
                    f"{case.name}/chunk-variant"
                ] = variant_sampler_anchor
            case_report["stage"] = "validation"
            if case.expect_admission_rejection:
                case_report["admission_rejected_before_attention"] = True
                case_report["cold_start_evidence"] = {
                    "status": "not_applicable",
                    "reason": "cold/measured protocol is disabled for admission rejection",
                }
                case_report["warmup_evidence"] = {
                    "status": "not_applicable",
                    "reason": "warmup protocol is disabled for admission rejection",
                }
                case_report["parity"] = {
                    "status": "not_applicable",
                    "reason": "admission rejection has no logits parity",
                }
                if variant_trace is not None:
                    case_report["chunk_variant"] = {
                        "passed": True,
                        "execution_passed": True,
                        "trace": variant_trace,
                        "admission_rejected_before_attention": True,
                        "cold_start_evidence": {
                            "status": "not_applicable",
                            "reason": (
                                "cold/measured protocol is disabled for "
                                "admission rejection"
                            ),
                        },
                        "warmup_evidence": {
                            "status": "not_applicable",
                            "reason": "warmup protocol is disabled for admission rejection",
                        },
                    }
            else:
                assert trt_logits is not None
                case_report["stage"] = "base_validation"
                base_validation = _validate_trace(
                    case,
                    spec,
                    trace,
                    trt_logits,
                    expected_sampler=base_sampler_anchor,
                    require_nvml_reconciliation=True,
                )
                assert base_validation is not None
                case_report["cold_start_evidence"] = base_validation[
                    "cold_start_evidence"
                ]
                case_report["warmup_evidence"] = {
                    "status": "passed",
                    "passed": True,
                    "base": base_validation["warmup_evidence"],
                }
                case_report["actual_shape_context_sweep"] = context_shape_sweep(trace)
                case_report["peak_memory_reconciliation"] = base_validation[
                    "peak_memory_reconciliation"
                ]
                logits_copy = output_dir / f"{case.name}.trt-logits.bin"
                source = Path(trace["logits_artifact"]["path"])
                logits_copy.write_bytes(source.read_bytes())
                case_report["trt_logits_artifact"] = str(logits_copy)
                case_report["trt_logits_sha256"] = _sha256(logits_copy)
                if variant_trace is not None:
                    assert variant_logits is not None
                    assert variant_chunk_limit is not None
                    assert variant_sampler_anchor is not None
                    case_report["stage"] = "chunk_variant_validation"
                    variant_validation = _validate_trace(
                        case,
                        spec,
                        variant_trace,
                        variant_logits,
                        expected_chunk_limit=variant_chunk_limit,
                        expected_sampler=variant_sampler_anchor,
                        require_nvml_reconciliation=True,
                    )
                    assert variant_validation is not None
                    case_report["warmup_evidence"]["chunk_variant"] = variant_validation[
                        "warmup_evidence"
                    ]
                    variant_copy = output_dir / f"{case.name}.c-div-2.trt-logits.bin"
                    variant_source = Path(variant_trace["logits_artifact"]["path"])
                    variant_copy.write_bytes(variant_source.read_bytes())
                    chunk_parity = compare_logits(
                        variant_logits,
                        trt_logits,
                        variant_trace["selected_token_ids"],
                        thresholds,
                    )
                    case_report["chunk_variant"] = {
                        "passed": bool(chunk_parity["passed"]),
                        "execution_passed": True,
                        "trace": variant_trace,
                        "sampler_trust_anchor": {
                            "pid": variant_sampler_anchor.pid,
                            "cuda_logical_device_index": (
                                variant_sampler_anchor.cuda_logical_device_index
                            ),
                            "physical_device_index": (
                                variant_sampler_anchor.physical_device_index
                            ),
                            "pci_bus_id": variant_sampler_anchor.pci_bus_id,
                            "gpu_uuid": variant_sampler_anchor.gpu_uuid,
                        },
                        "cold_start_evidence": variant_validation[
                            "cold_start_evidence"
                        ],
                        "warmup_evidence": variant_validation["warmup_evidence"],
                        "actual_shape_context_sweep": context_shape_sweep(variant_trace),
                        "peak_memory_reconciliation": variant_validation[
                            "peak_memory_reconciliation"
                        ],
                        "base_vs_variant_parity": chunk_parity,
                        "trt_logits_artifact": str(variant_copy),
                        "trt_logits_sha256": _sha256(variant_copy),
                    }
                if hf_model is None:
                    case_report["parity"] = {
                        "status": "not_run",
                        "reason": "--skip-hf was requested",
                    }
                else:
                    case_report["stage"] = "hf_reference"
                    hf_logits = run_hf_reference(
                        hf_model,
                        tokens,
                        trace["selected_token_ids"],
                        args.device,
                    )
                    case_report["stage"] = "hf_parity"
                    parity = compare_logits(
                        trt_logits,
                        hf_logits,
                        trace["selected_token_ids"],
                        thresholds,
                    )
                    parity["status"] = "passed" if parity["passed"] else "failed"
                    case_report["parity"] = parity
                    hf_path = output_dir / f"{case.name}.hf-logits.npy"
                    np.save(hf_path, hf_logits)
                    case_report["hf_logits_artifact"] = str(hf_path)
                    case_report["hf_logits_sha256"] = _sha256(hf_path)
            variant_passed = bool(
                "chunk_variant" not in case_report
                or case_report["chunk_variant"].get("passed") is True
            )
            parity = case_report["parity"]
            parity_status = parity.get("status")
            warmup_passed = bool(
                case.expect_admission_rejection
                or (
                    case_report["warmup_evidence"].get("passed") is True
                    and _persisted_case_warmup_evidence_passed(
                        case_report["warmup_evidence"]["base"],
                        trace=trace,
                        case=case,
                        trusted_geometry=trusted_runtime_geometry(spec),
                        expected_sampler=base_sampler_anchor,
                    )
                    and (
                        variant_trace is None
                        or _persisted_case_warmup_evidence_passed(
                            case_report["chunk_variant"]["warmup_evidence"],
                            trace=variant_trace,
                            case=case,
                            trusted_geometry=trusted_runtime_geometry(
                                spec,
                                prefill_chunk_limit=variant_chunk_limit,
                            ),
                            expected_sampler=variant_sampler_anchor,
                        )
                    )
                )
            )
            if case.expect_admission_rejection:
                qualification_case_passed = variant_passed
                diagnostic_case_passed = variant_passed
            else:
                qualification_case_passed = bool(
                    parity_status == "passed"
                    and parity.get("passed") is True
                    and variant_passed
                    and warmup_passed
                )
                diagnostic_case_passed = bool(
                    parity_status in {"passed", "not_run"}
                    and (parity_status == "not_run" or parity.get("passed") is True)
                    and variant_passed
                    and warmup_passed
                )
            case_report["qualification_passed"] = qualification_case_passed
            case_report["diagnostic_passed"] = diagnostic_case_passed
            case_report["passed"] = qualification_case_passed
            case_report["status"] = (
                "passed"
                if qualification_case_passed
                else "diagnostic_passed"
                if diagnostic_case_passed
                else "failed"
            )
            case_report["stage"] = "completed"
            if runner_stderr:
                stderr_path = output_dir / f"{case.name}.runner.stderr.log"
                stderr_path.write_text(runner_stderr, encoding="utf-8")
                case_report["runner_stderr"] = str(stderr_path)
            if variant_stderr:
                variant_stderr_path = output_dir / f"{case.name}.c-div-2.runner.stderr.log"
                variant_stderr_path.write_text(variant_stderr, encoding="utf-8")
                case_report.setdefault("chunk_variant", {})["runner_stderr"] = str(
                    variant_stderr_path
                )
            all_case_diagnostics_passed = bool(
                all_case_diagnostics_passed and diagnostic_case_passed
            )
            report["case_diagnostics_passed_so_far"] = all_case_diagnostics_passed
            _write_qualification_report(report_path, report)

    if report.get("failure") is not None:
        print(
            json.dumps(
                {
                    "passed": False,
                    "promotion_eligible": False,
                    "diagnostic_passed": False,
                    "execution_passed": False,
                    "status": "failed",
                    "qualification_blockers": report["qualification_blockers"],
                    "report": str(report_path),
                    "bundle_sha256": report["bundle_sha256"],
                    "cases": len(report["cases"]),
                    "failure": report["failure"],
                },
                sort_keys=True,
            )
        )
        return 1

    with qualification_failure_checkpoint(
        report=report,
        report_path=report_path,
        repo_root=repo_root,
        output_dir=output_dir,
    ):
        report["stage"] = "context_memory_envelope"
        context_memory_envelope = validate_context_memory_envelope(
            spec,
            report["cases"],
            require_full_coverage=not bool(args.case),
        )
        report["context_memory_envelope"] = context_memory_envelope
        report["stage"] = "source_state_post"
        source_state_post = source_state_provenance(
            repo_root,
            Path(__file__),
            output_dir,
            label="post",
        )
        source_state_unchanged = (
            source_state_pre["source_state_sha256"]
            == source_state_post["source_state_sha256"]
        )
        report["source_state_post"] = source_state_post
        report["source_state_unchanged"] = source_state_unchanged
        report["stage"] = "qualification_outcome"
        outcome = evaluate_qualification_outcome(
            canonical_cases=canonical_cases,
            selected_cases=cases,
            case_reports=report["cases"],
            skip_hf=args.skip_hf,
            case_filter_used=bool(args.case),
            source_state_pre=source_state_pre,
            source_state_post=source_state_post,
            context_memory_envelope=context_memory_envelope,
            qualified_engine_graph=qualified_engine_graph,
            model_spec=spec,
            runner=runner,
            bundle=bundle,
            runner_evidence_root=runner_evidence_root,
            thresholds=thresholds,
            trusted_geometry=trusted_runtime_geometry(spec),
            sampler_anchors=sampler_anchors,
            base_artifact_binding=base_artifact_binding,
            runtime_kv_plugin_binding=runtime_kv_plugin_binding,
            variant_bundle=variant_bundle,
            trusted_variant_geometry=(
                trusted_runtime_geometry(
                    spec,
                    prefill_chunk_limit=variant_chunk_limit,
                )
                if variant_chunk_limit is not None
                else None
            ),
            variant_build_receipt=variant_build_receipt,
            qualified_variant_engine_graph=variant_engine_graph,
        )
        report.update(outcome)
        report["stage"] = "completed"
        _write_qualification_report(report_path, report)

    if report.get("failure") is not None:
        print(
            json.dumps(
                {
                    "passed": False,
                    "promotion_eligible": False,
                    "diagnostic_passed": False,
                    "execution_passed": False,
                    "status": "failed",
                    "qualification_blockers": report["qualification_blockers"],
                    "report": str(report_path),
                    "bundle_sha256": report["bundle_sha256"],
                    "cases": len(report["cases"]),
                    "failure": report["failure"],
                },
                sort_keys=True,
            )
        )
        return 1

    print(
        json.dumps(
            {
                "passed": report["passed"],
                "promotion_eligible": report[
                    "promotion_eligible"
                ],
                "diagnostic_passed": report["diagnostic_passed"],
                "execution_passed": report["execution_passed"],
                "status": report["status"],
                "qualification_blockers": report["qualification_blockers"],
                "report": str(report_path),
                "bundle_sha256": report["bundle_sha256"],
                "cases": len(report["cases"]),
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] or report["diagnostic_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
