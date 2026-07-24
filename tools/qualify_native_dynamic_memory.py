#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic long-context qualification for native runtime-memory bundles.

The runner consumes token IDs rather than text, loads each real bundle through
the normal C++ factory, and writes complete final-position/decode logits.  This
script compares those rows with the pinned Hugging Face checkpoint and enforces
the model's existing family thresholds.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
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
CHUNK_VARIANT_BUILD_SCHEMA = "trtmc.native-dynamic-memory-chunk-variant-build/v1"
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
) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise ValueError(f"{label} identity is missing")
    actual_path = Path(str(identity.get("path", ""))).expanduser().resolve()
    expected_path = expected_path.expanduser().resolve()
    if actual_path != expected_path:
        raise ValueError(f"{label} path mismatch: expected {expected_path}, got {actual_path}")
    if not actual_path.is_file():
        raise ValueError(f"{label} does not exist: {actual_path}")
    actual_size = actual_path.stat().st_size
    if (
        isinstance(identity.get("size_bytes"), bool)
        or identity.get("size_bytes") != actual_size
        or actual_size <= 0
    ):
        raise ValueError(f"{label} size identity mismatch")
    actual_sha = _sha256(actual_path)
    if identity.get("sha256") != actual_sha:
        raise ValueError(f"{label} SHA-256 identity mismatch")
    return {
        "path": str(actual_path),
        "size_bytes": actual_size,
        "sha256": actual_sha,
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

    return {
        "path": str(receipt_path),
        "size_bytes": receipt_path.stat().st_size,
        "sha256": _sha256(receipt_path),
        "schema_version": receipt["schema_version"],
        "bundle": bundle_identity,
        "producer": producer_identity,
        "build_timing": timing_identity,
        "source_state_sha256": expected_source_sha,
        "git_head": expected_head,
    }


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


def run_trt_case(
    *,
    runner: Path,
    bundle: Path,
    tokens: np.ndarray,
    case: Case,
    context_limit: int,
    work_dir: Path,
) -> tuple[dict[str, Any], np.ndarray | None, str]:
    token_path = work_dir / f"{case.name}.tokens.txt"
    logits_path = work_dir / f"{case.name}.trt-logits.bin"
    _write_tokens(token_path, tokens)
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
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    trace = _parse_runner_json(completed.stdout)
    if case.expect_admission_rejection:
        if completed.returncode != 3:
            raise RuntimeError(
                f"{case.name}: expected admission exit 3, got {completed.returncode}; "
                f"trace={trace}; stderr={completed.stderr[-4000:]}"
            )
        expected = {
            "status": "rejected",
            "error_type": "admission",
            "stage": "before_attention",
            "prefill_launches": 0,
            "decode_launches": 0,
        }
        for key, value in expected.items():
            if trace.get(key) != value:
                raise RuntimeError(
                    f"{case.name}: rejection trace {key}={trace.get(key)!r}, expected {value!r}"
                )
        if logits_path.exists():
            raise RuntimeError(f"{case.name}: rejected request wrote a logits artifact")
        return trace, None, completed.stderr
    if completed.returncode != 0 or trace.get("status") != "ok":
        raise RuntimeError(
            f"{case.name}: runner failed ({completed.returncode}); trace={trace}; "
            f"stderr={completed.stderr[-4000:]}"
        )
    return trace, read_logits_artifact(logits_path), completed.stderr


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
    require_nvml_reconciliation: bool = False,
) -> None:
    chunk_limit = expected_chunk_limit or spec.chunk_limit
    effective_request_limit = (
        expected_effective_request_limit
        if expected_effective_request_limit is not None
        else spec.context_limit
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
    receipt = trace.get("runtime_memory_receipt")
    if not isinstance(receipt, dict) or int(receipt.get("kv_allocation_id", 0)) <= 0:
        raise RuntimeError(f"{case.name}: missing KV allocation trace")
    allocation_id = int(receipt["kv_allocation_id"])
    reserved = int(receipt.get("runtime_kv_capacity_tokens", 0))
    bytes_per_token = int(receipt.get("kv_bytes_per_token", 0))
    if reserved <= 0 or bytes_per_token <= 0:
        raise RuntimeError(f"{case.name}: receipt is missing R/B accounting")
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
    if require_nvml_reconciliation:
        reconcile_device_peak_with_nvml(trace)

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


def reconcile_device_peak_with_nvml(trace: dict[str, Any]) -> dict[str, Any]:
    """Reconcile device-wide cudaMemGetInfo high-water with process NVML.

    The qualification-only runtime observer records NVML process usage beside
    the exact cudaMemGetInfo samples used by the receipt. The samplers have
    different scope, so the gate allows the larger of 64 MiB and two percent
    while preserving every raw value.
    """
    sampler = trace.get("memory_sampler")
    if not isinstance(sampler, dict) or sampler.get("source") != (
        "nvmlDeviceGetComputeRunningProcesses_v3"
    ):
        raise RuntimeError("qualification requires independent NVML process memory sampling")
    receipt = trace.get("runtime_memory_receipt")
    if not isinstance(receipt, dict):
        raise RuntimeError("qualification trace has no runtime memory receipt")
    device_peak = receipt.get("peak_device_bytes")
    if not isinstance(device_peak, int) or device_peak < 0:
        raise RuntimeError("receipt has no sampled device-wide peak")

    load_cycles = trace.get("load_cycles")
    if not isinstance(load_cycles, list) or not load_cycles:
        raise RuntimeError("qualification trace has no measured load lifetime")
    lifetime = load_cycles[-1]
    if not isinstance(lifetime, dict):
        raise RuntimeError("qualification load lifetime is not an object")
    phase_samples = lifetime.get("runtime_phase_memory_samples")
    if not isinstance(phase_samples, list) or not phase_samples:
        raise RuntimeError("qualification lifetime has no synchronized runtime phase samples")
    parsed_samples: list[dict[str, int | str]] = []
    for sample in phase_samples:
        if not isinstance(sample, dict):
            raise RuntimeError("runtime phase memory sample is not an object")
        phase = sample.get("phase")
        free_bytes = sample.get("free_bytes")
        total_bytes = sample.get("total_bytes")
        process_used_bytes = sample.get("process_used_bytes")
        if (
            not isinstance(phase, str)
            or not isinstance(free_bytes, int)
            or not isinstance(total_bytes, int)
            or not isinstance(process_used_bytes, int)
            or free_bytes <= 0
            or total_bytes <= 0
            or free_bytes > total_bytes
            or process_used_bytes < 0
        ):
            raise RuntimeError("runtime phase memory sample is invalid")
        parsed_samples.append(
            {
                "phase": phase,
                "free_bytes": free_bytes,
                "total_bytes": total_bytes,
                "process_used_bytes": process_used_bytes,
            }
        )

    baseline_samples = [
        sample
        for sample in parsed_samples
        if str(sample["phase"]).startswith("before runtime-memory ")
        and str(sample["phase"]).endswith(" engine deserialization")
    ]
    if len(baseline_samples) != 1:
        raise RuntimeError(
            "qualification lifetime must have exactly one synchronized pre-engine baseline"
        )
    baseline_sample = baseline_samples[0]
    observed_boundaries = [
        sample
        for sample in parsed_samples
        if sample["phase"]
        in {
            "after runtime KV allocation",
            "after successful runtime-memory request completion",
        }
    ]
    required_boundaries = {
        "after runtime KV allocation",
        "after successful runtime-memory request completion",
    }
    observed_phase_names = {str(sample["phase"]) for sample in observed_boundaries}
    if not required_boundaries.issubset(observed_phase_names):
        missing = sorted(required_boundaries - observed_phase_names)
        raise RuntimeError(
            "qualification lifetime is missing synchronized peak boundaries: " + ", ".join(missing)
        )

    receipt_total = receipt.get("pre_load_total_bytes")
    if not isinstance(receipt_total, int) or any(
        sample["total_bytes"] != receipt_total for sample in [baseline_sample, *observed_boundaries]
    ):
        raise RuntimeError("synchronized runtime phase samples disagree on CUDA device total")
    baseline_free = int(baseline_sample["free_bytes"])
    synchronized_device_peak = max(
        0,
        max(baseline_free - int(sample["free_bytes"]) for sample in observed_boundaries),
    )
    if synchronized_device_peak != device_peak:
        raise RuntimeError(
            "receipt peak does not match synchronized runtime phase samples: "
            f"receipt={device_peak}, synchronized={synchronized_device_peak}"
        )

    baseline = int(baseline_sample["process_used_bytes"])
    process_peak = max(
        0,
        max(int(sample["process_used_bytes"]) - baseline for sample in observed_boundaries),
    )
    difference = abs(device_peak - process_peak)
    tolerance = max(
        64 * 1024 * 1024,
        math.ceil(0.02 * max(device_peak, process_peak, 1)),
    )
    result = {
        "device_wide_peak_bytes": device_peak,
        "nvml_process_peak_bytes": process_peak,
        "absolute_difference_bytes": difference,
        "tolerance_bytes": tolerance,
        "tolerance_rule": "max(64MiB,2pct)",
        "device_scope": "cudaMemGetInfo-device-wide",
        "process_scope": "nvml-current-process",
        "sample_boundaries": [
            str(baseline_sample["phase"]),
            "after runtime KV allocation",
            "after successful runtime-memory request completion",
        ],
        "synchronized_cuda_peak_bytes": synchronized_device_peak,
        "baseline_sample": baseline_sample,
        "peak_boundary_samples": observed_boundaries,
        "passed": difference <= tolerance,
    }
    if not result["passed"]:
        raise RuntimeError(
            "device-wide peak and NVML process peak do not reconcile: "
            f"device={device_peak}, process={process_peak}, "
            f"difference={difference}, tolerance={tolerance}"
        )
    return result


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
    selected_diagnostics_passed = case_execution_passed
    for case in selected_cases:
        report = report_by_name.get(case.name, {})
        if case.expect_admission_rejection:
            parity_states[case.name] = "not_applicable"
        else:
            parity = report.get("parity")
            if (
                isinstance(parity, Mapping)
                and parity.get("status") == "passed"
                and parity.get("passed") is True
            ):
                parity_states[case.name] = "passed"
            elif (
                isinstance(parity, Mapping)
                and parity.get("status") == "not_run"
                and "passed" not in parity
            ):
                parity_states[case.name] = "not_run"
            else:
                parity_states[case.name] = "failed"
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

    qualification_gates = {
        "canonical_matrix_complete": canonical_matrix_complete,
        "case_filter_not_used": not case_filter_used,
        "case_execution_passed": case_execution_passed,
        "hf_parity_executed_and_passed": hf_parity_executed_and_passed,
        "source_state_unchanged": source_state_unchanged,
        "source_clean_exact_head": source_clean_exact_head,
        "full_context_memory_coverage": full_context_memory_coverage,
        "qualified_engine_graph_passed": qualified_engine_graph_passed,
    }
    passed = all(qualification_gates.values())
    execution_passed = bool(
        case_execution_passed
        and source_state_unchanged
        and diagnostic_context_memory_passed
        and qualified_engine_graph_passed
    )
    diagnostic_passed = bool(execution_passed and selected_diagnostics_passed)
    status = "passed" if passed else "diagnostic_passed" if diagnostic_passed else "failed"
    return {
        "passed": passed,
        "diagnostic_passed": diagnostic_passed,
        "execution_passed": execution_passed,
        "status": status,
        "qualification_gates": qualification_gates,
        "qualification_blockers": [
            name for name, value in qualification_gates.items() if not value
        ],
        "parity_execution": parity_states,
    }


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
        "--chunk-variant-bundle",
        type=Path,
        help=(
            "Developer-only bundle built from the same qualified tuple with "
            "prefill C/2; compare it against the normal C bundle"
        ),
    )
    parser.add_argument(
        "--chunk-variant-build-receipt",
        type=Path,
        help=(
            "Source-bound build receipt emitted by "
            "build_native_dynamic_memory_chunk_variant.py; required with "
            "--chunk-variant-bundle"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run one named matrix case (repeatable); default is all",
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
        "qualified_engine_graph": qualified_engine_graph,
        "canonical_case_names": [case.name for case in canonical_cases],
        "selected_case_names": [case.name for case in cases],
        "case_filter_used": bool(args.case),
        "hf_parity_requested": not args.skip_hf,
        "passed": False,
        "diagnostic_passed": False,
        "execution_passed": False,
        "status": "running",
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
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

    all_case_diagnostics_passed = True
    with tempfile.TemporaryDirectory(
        prefix="trtmc-native-memory-qualification-", dir=output_dir
    ) as temporary:
        work_dir = Path(temporary)
        variant_work_dir = work_dir / "chunk-variant"
        if variant_bundle is not None:
            variant_work_dir.mkdir()
        for case in cases:
            print(
                f"[qualification] {case.name}: prompt={case.prompt_tokens} "
                f"decode={case.decode_tokens}",
                file=sys.stderr,
                flush=True,
            )
            tokens = deterministic_token_ids(case.prompt_tokens, vocab_size)
            trace, trt_logits, runner_stderr = run_trt_case(
                runner=runner,
                bundle=bundle,
                tokens=tokens,
                case=case,
                context_limit=spec.context_limit,
                work_dir=work_dir,
            )
            case_report: dict[str, Any] = {
                "name": case.name,
                "prompt_tokens": case.prompt_tokens,
                "decode_tokens": case.decode_tokens,
                "expect_admission_rejection": (case.expect_admission_rejection),
                "input_token_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
                "trace": trace,
                "execution_passed": True,
            }
            variant_trace: dict[str, Any] | None = None
            variant_logits: np.ndarray | None = None
            variant_stderr = ""
            if variant_bundle is not None:
                variant_trace, variant_logits, variant_stderr = run_trt_case(
                    runner=runner,
                    bundle=variant_bundle,
                    tokens=tokens,
                    case=case,
                    context_limit=spec.context_limit,
                    work_dir=variant_work_dir,
                )
            if case.expect_admission_rejection:
                case_report["admission_rejected_before_attention"] = True
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
                    }
            else:
                assert trt_logits is not None
                _validate_trace(
                    case,
                    spec,
                    trace,
                    trt_logits,
                    require_nvml_reconciliation=True,
                )
                case_report["actual_shape_context_sweep"] = context_shape_sweep(trace)
                case_report["peak_memory_reconciliation"] = reconcile_device_peak_with_nvml(trace)
                logits_copy = output_dir / f"{case.name}.trt-logits.bin"
                source = Path(trace["logits_artifact"]["path"])
                logits_copy.write_bytes(source.read_bytes())
                case_report["trt_logits_artifact"] = str(logits_copy)
                case_report["trt_logits_sha256"] = _sha256(logits_copy)
                if variant_trace is not None:
                    assert variant_logits is not None
                    assert variant_chunk_limit is not None
                    _validate_trace(
                        case,
                        spec,
                        variant_trace,
                        variant_logits,
                        expected_chunk_limit=variant_chunk_limit,
                        require_nvml_reconciliation=True,
                    )
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
                        "actual_shape_context_sweep": context_shape_sweep(variant_trace),
                        "peak_memory_reconciliation": (
                            reconcile_device_peak_with_nvml(variant_trace)
                        ),
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
                    hf_logits = run_hf_reference(
                        hf_model,
                        tokens,
                        trace["selected_token_ids"],
                        args.device,
                    )
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
            if case.expect_admission_rejection:
                qualification_case_passed = variant_passed
                diagnostic_case_passed = variant_passed
            else:
                qualification_case_passed = bool(
                    parity_status == "passed" and parity.get("passed") is True and variant_passed
                )
                diagnostic_case_passed = bool(
                    parity_status in {"passed", "not_run"}
                    and (parity_status == "not_run" or parity.get("passed") is True)
                    and variant_passed
                )
            case_report["qualification_passed"] = qualification_case_passed
            case_report["diagnostic_passed"] = diagnostic_case_passed
            case_report["passed"] = qualification_case_passed
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
            report["cases"].append(case_report)
            report_path = output_dir / "qualification-report.json"
            report["case_diagnostics_passed_so_far"] = all_case_diagnostics_passed
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    context_memory_envelope = validate_context_memory_envelope(
        spec,
        report["cases"],
        require_full_coverage=not bool(args.case),
    )
    report["context_memory_envelope"] = context_memory_envelope
    source_state_post = source_state_provenance(
        repo_root,
        Path(__file__),
        output_dir,
        label="post",
    )
    source_state_unchanged = (
        source_state_pre["source_state_sha256"] == source_state_post["source_state_sha256"]
    )
    report["source_state_post"] = source_state_post
    report["source_state_unchanged"] = source_state_unchanged
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
    )
    report.update(outcome)
    report_path = output_dir / "qualification-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "passed": report["passed"],
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
