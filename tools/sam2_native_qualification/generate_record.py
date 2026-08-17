#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a non-authorizing SAM2 qualification record from immutable artifacts.

This tool has no production-pin mutation path.  It authenticates one bundle,
the exclusive Q3 receipt emitted before W3, and the linked regular receipt
emitted after W3/N100/Q1.  A later reviewed source change must activate the
record hash in the production-only pin provider.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import struct
import sys
from typing import Any


BUNDLE_MAGIC = b"BUNDLE\x01\x00"
RECORD_SCHEMA_VERSION = 2
BENCHMARK_RECEIPT_SCHEMA_VERSION = 2
RECORD_ARTIFACT_TYPE = "sam2_native_qualification_record"
AUDIT_ARTIFACT_TYPE = "sam2_native_qualification_audit_manifest"
POLICY_ID = "sam2_semantic_accuracy_v1"
FAMILY = "sam2"
MODEL_ID = "sam2.1-hiera-small-bbox"
WORKLOAD = "sam2.1-hiera-small-bbox-five-frame"
ENGINE_CONTRACT_VERSION = 5
STRATEGY = "sam2_bbox_video_tracking"
PRECISION = "mixed_bf16_fp32"
GPU_NAME = "NVIDIA L4"
COMPUTE_CAPABILITY = "8.9"
TENSORRT_VERSION = "11.1.0.106"
TENSORRT_ABI = "11.1"
BUILD_RECEIPT_SCHEMA_VERSION = 2
BUILDER_OPTIMIZATION_LEVEL = 3
CHECKPOINT_SHA256 = "89fd676560809c8504411b574cea305c86db1f65bda790ec7fe16cedc6c6ff73"
CONFIG_SHA256 = "59488bb78c7cc48aaaebd966ea9d054014f683459d062b7a959a4aa501342656"
GOLDEN_MANIFEST_SHA256 = "c25251ee27da05afd75adc3c6869cbc2944b80c05c5d6e703b6ebbbba697a4f0"
GOLDEN_MASKS_SHA256 = "1c7830b37739e409fbb8dab2b81c31c63b3379e6c10ae9e6b4ca2cc48a656094"
BASELINE_RECEIPT_SHA256 = "af85ed2de2143db6fdf3af40d3621e7f33b97c632904950ee79f3d5f7219f028"
BASELINE_CAPTURE_SHA256 = "ba57e65432945f3d6fdafae564a3901f0dd5479ba73e400a5b0a88aa595c30b4"
PLAN_SECTIONS = (
    "sam2_image_engine_plan",
    "sam2_prompt_engine_plan",
    "sam2_recurrent_h1_engine_plan",
    "sam2_recurrent_h2_engine_plan",
    "sam2_recurrent_h3_engine_plan",
    "sam2_recurrent_h4_engine_plan",
)
ALL_SECTIONS = PLAN_SECTIONS + ("config.json", "sam2_build_receipt.json")
ENCODED_JPEG_SHA256 = (
    "8a398f40747d5053cfc0d47d45090f2070a10afa4722e7d5b827a6ad0825a5aa",
    "2871555bca47da7473762ca87314b17bd55d100a0f982f78d6449080ff86856f",
    "5594181db7dd1c5da3ce05b945f74e66a5d8d098d71a7cb9e5e43834a393bbe2",
    "c3abc03371458939d09faf331749c2a87cc6fc91128eaab3901b179adb096a35",
    "3d8ea6042c82e7b340277c00666c4c2cefbae5de265ef06a71fe964905ed720b",
)
DECODED_JPEG_SHA256 = (
    "0bcadde0e5a6f8ba04f79c44f064c5b00d3cd1b250e2f2f3bbf10ef0630a9ce9",
    "0abfd57f9e3886a8c3068bf6bcc353b26d1e3a8a43819a80dfeb00f309b24ec3",
    "9166cc263c3edb262065fa3b98ee062cbf6d781dd656bae13def7f4141b7d025",
    "77525faadfc8a607e4e1556135887caaddd0b64d7cd677fcf47c38ecf9e25a4f",
    "cb0801b490ba13dfb6d36aeef06b049ff67ff11864ef62ccd858a0096d97c6af",
)
THRESHOLDS = {
    "every_frame_iou_min": 0.98,
    "macro_iou_min": 0.99,
    "global_iou_min": 0.99,
    "bbox_iou_min": 0.995,
    "bbox_max_coordinate_error_max": 0.5,
    "bbox_score_error_max": 0.01,
    "bbox_label_exact": True,
}
REPEAT_CONTRACT = (
    "each reset-separated replay independently passes the semantic mask and bbox gates; "
    "hashes and foreground counts are informational"
)
BENCHMARK_SOURCE_CLOSURE_ROLE = (
    "run-time snapshot of declared repository source and build-control inputs; executable "
    "SHA-256 is authoritative for the binary actually run"
)
COMMON_PROCESS_MODEL = {
    "tensorrt_iattention_v2_image_attention": True,
    "external_attention_dso_loaded": False,
    "bundle_build_count": 1,
    "expected_sha256_bundle_load_count": 1,
    "builder_returned_full_bundle_sha256": True,
    "loader_sealed_snapshot_sha256_bound_before_deserialization": True,
    "receipt_and_plan_evidence_from_builder_not_path_rereads": True,
    "engine_deserialization_count": 6,
    "shared_nonblocking_cuda_stream": True,
    "processor": "makeNativeDeviceVideoProcessor",
    "checked_enqueue_v3_adapter": True,
}
Q3_PROCESS_MODEL = {
    **COMMON_PROCESS_MODEL,
    "checkpoint_graph_build_before_replays": True,
    "six_plan_deserializations_before_replays": True,
}
REGULAR_PROCESS_MODEL = {
    **COMMON_PROCESS_MODEL,
    "checkpoint_graph_build_outside_timing": True,
    "six_plan_deserializations_outside_timing": True,
}
TIMING_BOUNDARIES = {
    "clock": "std::chrono::steady_clock synchronized wall time",
    "reset": (
        "t0 -> processor.reset (clear run state, drain workspace, invalidate the completed "
        "run, invoke the reset_execution_context hook on six stable modules without "
        "context recreation, validate device graph, transition to idle) -> "
        "cudaStreamSynchronize; included in native prefill"
    ),
    "native_prefill": (
        "t0 -> processor.reset -> cudaStreamSynchronize -> decodeSam2JpegBytes for all "
        "five retained authenticated byte vectors -> copy decoded RGB8 bytes into five "
        "stable HWC RGB8 frame buffers -> run_bbox_prompt (frame 0: same-stream RGB8 "
        "H2D -> CUDA Pillow horizontal uint8 pass -> CUDA Pillow vertical uint8 pass plus "
        "FP32 NCHW normalization -> image and prompt enqueue) -> cudaStreamSynchronize -> "
        "t1"
    ),
    "native_tracker": (
        "t1 -> propagate frames 1 through 4 (each frame: same-stream RGB8 H2D -> CUDA "
        "Pillow horizontal uint8 pass -> CUDA Pillow vertical uint8 pass plus FP32 NCHW "
        "normalization -> image and recurrent enqueue) -> cudaStreamSynchronize -> t2"
    ),
    "closest_envelope_total": "t0 -> t2",
    "jpeg_file_open_and_read_inside_timing": False,
    "encoded_input": (
        "pre-read authenticated immutable byte vectors; decodeSam2JpegBytes lvalue copy and "
        "JPEG decode plus byte-for-byte copy into stable HWC RGB8 frame storage are inside "
        "each native prefill; no host uint8-to-float conversion is performed"
    ),
    "accuracy_and_mask_download_inside_timing": False,
    "native_stage_split_comparable_to_delivered_baseline": False,
    "total_is_exact_apples_to_apples_with_delivered_lazy_loader": False,
    "comparison_scope": "closest five-frame end-to-end inference envelope only",
}
P90_METHOD = "sorted sample index round((n-1)*0.90), zero-based index 89 for n=100"
DELIVERED_BASELINE_REFERENCE = {
    "receipt_sha256": BASELINE_RECEIPT_SHA256,
    "baseline_receipt_contains_asset_hashes": False,
    "baseline_asset_binding": "external_reviewed_capture_evidence",
    "warmup_rows": 3,
    "measurement_rows": 100,
    "total_mean_ms": 257.1344714984298,
    "total_median_ms": 253.67085821926594,
    "total_p90_ms": 265.56191593408585,
    "comparison_warning": (
        "raw reference only; no speedup is computed because the delivered baseline uses a "
        "different lazy directory-loader and stage split, explicitly enables matmul and "
        "cuDNN TF32 while the native build disables TF32, and places release_state plus GC "
        "plus empty_cache outside timing while native reset and run invalidation are inside "
        "the closest envelope"
    ),
}
GRAPH_KINDS = ("image", "prompt", "recurrent", "recurrent", "recurrent", "recurrent")
GRAPH_HISTORY_FRAMES = (0, 0, 1, 2, 3, 4)
GRAPH_INPUTS = (1, 4, 5, 5, 5, 5)
GRAPH_OUTPUTS = (9, 3, 3, 3, 3, 3)
GRAPH_LAYERS = (1139, 882, 1630, 1652, 1674, 1696)
GRAPH_REFERENCED_TENSORS = (282, 185, 291, 291, 291, 291)
IMAGE_GRAPH_LAYER_COUNTS = {
    "convolution_layers": 23,
    "activation_layers": 28,
    "pooling_layers": 6,
    "element_wise_layers": 130,
    "shuffle_layers": 313,
    "constant_layers": 216,
    "slice_layers": 67,
    "resize_layers": 2,
    "normalization_layers": 32,
    "cast_layers": 223,
    "matrix_multiply_layers": 67,
    "softmax_layers": 0,
    "plugin_v3_layers": 0,
    "attention_input_layers": 16,
    "attention_output_layers": 16,
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
AUTHORITY_RE = re.compile(r"[a-z][a-z0-9._-]{0,127}\Z")


class EvidenceError(RuntimeError):
    """An input artifact failed closed validation."""


@dataclasses.dataclass(frozen=True)
class _RawJsonNumber:
    """A syntactically validated JSON number whose original spelling is retained."""

    lexeme: str


@dataclasses.dataclass(frozen=True)
class Snapshot:
    path: Path
    data: bytes
    sha256: str
    size_bytes: int


@dataclasses.dataclass(frozen=True)
class BundleSnapshot:
    path: Path
    sha256: str
    size_bytes: int
    header: dict[str, Any]
    section_sha256: dict[str, str]
    section_size: dict[str, int]
    config: dict[str, Any]
    build_receipt: dict[str, Any]


@dataclasses.dataclass
class _PublishedFile:
    """An open inode handle for one durably published output."""

    path: Path
    label: str
    device: int
    inode: int
    descriptor: int

    def close(self) -> None:
        if self.descriptor >= 0:
            try:
                os.close(self.descriptor)
            finally:
                self.descriptor = -1


def _fail(message: str) -> None:
    raise EvidenceError(f"SAM2 qualification evidence: {message}")


def _canonical_path(path: Path, label: str) -> Path:
    if not path.is_absolute() or Path(os.path.normpath(str(path))) != path:
        _fail(f"{label} path must be absolute and lexically canonical")
    return path


def _stable_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_regular(path: Path, maximum_size: int, label: str) -> tuple[int, os.stat_result]:
    path = _canonical_path(path, label)
    flags = os.O_RDONLY | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        _fail(f"cannot open no-follow {label}: {error.strerror}")
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum_size
        ):
            _fail(f"{label} is not a unique supported regular file")
        return descriptor, before
    except Exception:
        os.close(descriptor)
        raise


def _pread_all(descriptor: int, size: int, offset: int = 0) -> bytes:
    result = bytearray()
    while len(result) != size:
        block = os.pread(descriptor, min(1024 * 1024, size - len(result)), offset + len(result))
        if not block:
            _fail("stable snapshot read made no progress")
        result.extend(block)
    return bytes(result)


def _hash_fd_range(descriptor: int, offset: int, size: int) -> str:
    digest = hashlib.sha256()
    consumed = 0
    while consumed != size:
        block = os.pread(descriptor, min(8 * 1024 * 1024, size - consumed), offset + consumed)
        if not block:
            _fail("stable hash read made no progress")
        digest.update(block)
        consumed += len(block)
    return digest.hexdigest()


def _snapshot_regular_file(path: Path, maximum_size: int, label: str) -> Snapshot:
    descriptor, before = _open_regular(path, maximum_size, label)
    try:
        data = _pread_all(descriptor, before.st_size)
        after = os.fstat(descriptor)
        if _stable_identity(before) != _stable_identity(after):
            _fail(f"{label} changed while its snapshot was captured")
        return Snapshot(path, data, hashlib.sha256(data).hexdigest(), len(data))
    finally:
        os.close(descriptor)


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _encode_canonical_raw_json(value: Any) -> str:
    if isinstance(value, dict):
        members = (
            json.dumps(key, ensure_ascii=False, separators=(",", ":"))
            + ":"
            + _encode_canonical_raw_json(item)
            for key, item in value.items()
        )
        return "{" + ",".join(members) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_encode_canonical_raw_json(item) for item in value) + "]"
    if isinstance(value, _RawJsonNumber):
        return value.lexeme
    if value is None or isinstance(value, (bool, str)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    _fail("raw JSON parser produced an unsupported value")


def _reject_nonfinite_numbers(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail(f"{label} contains a non-finite number")
    if isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite_numbers(item, label)
    elif isinstance(value, list):
        for item in value:
            _reject_nonfinite_numbers(item, label)


def _parse_json(data: bytes, label: str, *, trailing_newline: bool) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        _fail(f"{label} is not UTF-8")
    try:
        raw_value = json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_int=_RawJsonNumber,
            parse_float=_RawJsonNumber,
            parse_constant=lambda token: _fail(f"{label} contains non-finite number {token}"),
        )
        canonical = _encode_canonical_raw_json(raw_value)
        value = json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=lambda token: _fail(f"{label} contains non-finite number {token}"),
        )
    except EvidenceError:
        raise
    except (json.JSONDecodeError, RecursionError, UnicodeError, ValueError) as error:
        _fail(f"{label} is invalid JSON: {error}")
    _reject_nonfinite_numbers(value, label)
    if not isinstance(value, dict):
        _fail(f"{label} root must be an object")
    if trailing_newline:
        canonical += "\n"
    if canonical.encode("utf-8") != data:
        _fail(f"{label} is not canonical compact JSON")
    return value


def _exact_keys(value: Any, expected: set[str] | tuple[str, ...], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(expected):
        _fail(f"{label} field set drifted")
    if isinstance(expected, tuple) and tuple(value) != expected:
        _fail(f"{label} field order drifted")
    return value


def _exact_contract(value: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        value = _exact_keys(value, tuple(expected), label)
        for key, expected_value in expected.items():
            _exact_contract(value[key], expected_value, f"{label} {key}")
        return
    if isinstance(expected, list):
        if not isinstance(value, list) or len(value) != len(expected):
            _fail(f"{label} list shape drifted")
        for index, (item, expected_item) in enumerate(zip(value, expected, strict=True)):
            _exact_contract(item, expected_item, f"{label} item {index}")
        return
    if type(value) is not type(expected) or value != expected:
        _fail(f"{label} drifted")


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        _fail(f"{label} is not a canonical {'positive ' if positive else ''}integer")
    return value


def _number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        _fail(f"{label} is not a finite number")
    return float(value)


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} is not a canonical SHA-256")
    return value


def _timestamp(value: Any, label: str) -> dt.datetime:
    if not isinstance(value, str):
        _fail(f"{label} is not a timestamp")
    try:
        parsed = dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        _fail(f"{label} is not canonical UTC")
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        _fail(f"{label} is not canonical UTC")
    return parsed


def _snapshot_bundle(path: Path) -> BundleSnapshot:
    descriptor, before = _open_regular(path, 2 * 1024 * 1024 * 1024, "bundle")
    try:
        prefix = _pread_all(descriptor, 16)
        if prefix[:8] != BUNDLE_MAGIC:
            _fail("bundle magic is invalid")
        header_size = struct.unpack("<Q", prefix[8:])[0]
        if header_size == 0 or header_size > 100 * 1024 * 1024 or 16 + header_size > before.st_size:
            _fail("bundle header size is invalid")
        header_bytes = _pread_all(descriptor, header_size, 16)
        header = _parse_json(header_bytes, "bundle header", trailing_newline=False)
        _exact_keys(
            header,
            {
                "model_id",
                "model_type",
                "family",
                "precision",
                "trt_version",
                "trt_abi",
                "gpu_name",
                "created_at",
                "runtime_strategy",
                "sections",
            },
            "bundle header",
        )
        expected_header = {
            "model_id": MODEL_ID,
            "model_type": "sam2_video_tracking",
            "family": FAMILY,
            "precision": PRECISION,
            "trt_version": TENSORRT_VERSION,
            "trt_abi": TENSORRT_ABI,
            "gpu_name": GPU_NAME,
            "runtime_strategy": STRATEGY,
        }
        for key, expected in expected_header.items():
            if header[key] != expected:
                _fail(f"bundle header {key} drifted")
        _timestamp(header["created_at"], "bundle created_at")
        sections = _exact_keys(header["sections"], set(ALL_SECTIONS), "bundle sections")
        if tuple(sections) != ALL_SECTIONS:
            _fail("bundle section order drifted")
        data_start = 16 + header_size
        next_offset = 0
        section_sha256: dict[str, str] = {}
        section_size: dict[str, int] = {}
        section_json: dict[str, dict[str, Any]] = {}
        for name in ALL_SECTIONS:
            entry = _exact_keys(sections[name], {"offset", "size", "sha256"}, f"section {name}")
            offset = _integer(entry["offset"], f"section {name} offset")
            size = _integer(entry["size"], f"section {name} size", positive=True)
            expected_digest = _sha256(entry["sha256"], f"section {name} SHA-256")
            if offset != next_offset or data_start + offset + size > before.st_size:
                _fail(f"section {name} is not contiguous and in bounds")
            observed = _hash_fd_range(descriptor, data_start + offset, size)
            if observed != expected_digest:
                _fail(f"section {name} SHA-256 mismatch")
            section_sha256[name] = observed
            section_size[name] = size
            if name in ("config.json", "sam2_build_receipt.json"):
                if size > 4 * 1024 * 1024:
                    _fail(f"section {name} is too large")
                section_json[name] = _parse_json(
                    _pread_all(descriptor, size, data_start + offset),
                    name,
                    trailing_newline=False,
                )
            next_offset += size
        if data_start + next_offset != before.st_size:
            _fail("bundle has trailing or unreferenced bytes")
        bundle_sha256 = _hash_fd_range(descriptor, 0, before.st_size)
        after = os.fstat(descriptor)
        if _stable_identity(before) != _stable_identity(after):
            _fail("bundle changed while its snapshot was captured")
    finally:
        os.close(descriptor)

    result = BundleSnapshot(
        path,
        bundle_sha256,
        before.st_size,
        header,
        section_sha256,
        section_size,
        section_json["config.json"],
        section_json["sam2_build_receipt.json"],
    )
    _validate_bundle_metadata(result)
    return result


def _validate_bundle_metadata(bundle: BundleSnapshot) -> None:
    config = _exact_keys(
        bundle.config,
        {
            "schema_version",
            "family",
            "model_id",
            "engine_contract_version",
            "runtime_strategy",
            "precision",
            "checkpoint_sha256",
            "source_config_sha256",
            "golden_manifest_sha256",
            "frame_count",
            "selected_object_count",
            "model_image_size",
            "original_image_height",
            "original_image_width",
            "plan_sections",
            "qualification",
            "runtime_eligible",
        },
        "embedded config",
    )
    expected = {
        "schema_version": 1,
        "family": FAMILY,
        "model_id": MODEL_ID,
        "engine_contract_version": ENGINE_CONTRACT_VERSION,
        "runtime_strategy": STRATEGY,
        "precision": PRECISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_config_sha256": CONFIG_SHA256,
        "golden_manifest_sha256": GOLDEN_MANIFEST_SHA256,
        "frame_count": 5,
        "selected_object_count": 1,
        "model_image_size": 1024,
        "original_image_height": 1280,
        "original_image_width": 1088,
        "plan_sections": list(PLAN_SECTIONS),
        "qualification": "unqualified",
        "runtime_eligible": False,
    }
    _exact_contract(config, expected, "embedded config contract")

    receipt = _exact_keys(
        bundle.build_receipt,
        {
            "schema_version",
            "family",
            "model_id",
            "qualification",
            "assets",
            "build",
            "image_attention",
            "graphs",
        },
        "build receipt",
    )
    _exact_contract(
        receipt["schema_version"], BUILD_RECEIPT_SCHEMA_VERSION, "build receipt schema version"
    )
    _exact_contract(receipt["family"], FAMILY, "build receipt family")
    _exact_contract(receipt["model_id"], MODEL_ID, "build receipt model ID")
    _exact_contract(
        receipt["qualification"],
        {
            "state": "unqualified",
            "runtime_eligible": False,
            "golden_parity_verified": False,
        },
        "build receipt qualification facts",
    )
    _exact_contract(
        receipt["assets"],
        {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "source_config_sha256": CONFIG_SHA256,
            "golden_manifest_sha256": GOLDEN_MANIFEST_SHA256,
            "embedded_config_sha256": bundle.section_sha256["config.json"],
        },
        "build receipt asset binding",
    )
    build = _exact_keys(
        receipt["build"],
        {
            "created_at_utc",
            "workspace_bytes",
            "network_mode",
            "tf32_enabled",
            "builder_optimization_level",
            "plan_profiling_verbosity",
            "tensorrt_version",
            "tensorrt_abi",
            "cuda_runtime_version",
            "cuda_driver_version",
            "gpu",
        },
        "build facts",
    )
    _exact_contract(
        build["builder_optimization_level"],
        BUILDER_OPTIMIZATION_LEVEL,
        "builder optimization level",
    )
    if (
        build["created_at_utc"] != bundle.header["created_at"]
        or _integer(build["workspace_bytes"], "workspace bytes", positive=True) <= 0
        or build["network_mode"] != "strongly_typed"
        or build["tf32_enabled"] is not False
        or build["plan_profiling_verbosity"] != "detailed"
        or build["tensorrt_version"] != TENSORRT_VERSION
        or build["tensorrt_abi"] != TENSORRT_ABI
    ):
        _fail("build target facts drifted")
    for key in ("cuda_runtime_version", "cuda_driver_version"):
        if not isinstance(build[key], str) or not build[key]:
            _fail(f"build {key} is empty")
    gpu = _exact_keys(
        build["gpu"], {"device", "name", "compute_capability", "global_memory_bytes"}, "build GPU"
    )
    if gpu["name"] != GPU_NAME or gpu["compute_capability"] != COMPUTE_CAPABILITY:
        _fail("build GPU target drifted")
    _integer(gpu["device"], "build GPU device")
    _integer(gpu["global_memory_bytes"], "build GPU memory", positive=True)
    _validate_attention(receipt["image_attention"], "build receipt image attention")
    graphs = receipt["graphs"]
    if not isinstance(graphs, list) or len(graphs) != 6:
        _fail("build receipt graph inventory drifted")
    for index, graph in enumerate(graphs):
        common_keys = {
            "section",
            "kind",
            "history_frames",
            "inputs",
            "outputs",
            "layers",
            "referenced_checkpoint_tensors",
            "serialized_bytes",
            "serialized_sha256",
            "graph_complete",
        }
        graph = _exact_keys(
            graph,
            common_keys | (set(IMAGE_GRAPH_LAYER_COUNTS) if index == 0 else set()),
            f"build receipt graph {index}",
        )
        expected = {
            "section": PLAN_SECTIONS[index],
            "kind": GRAPH_KINDS[index],
            "history_frames": GRAPH_HISTORY_FRAMES[index],
            "inputs": GRAPH_INPUTS[index],
            "outputs": GRAPH_OUTPUTS[index],
            "layers": GRAPH_LAYERS[index],
        }
        if index == 0:
            expected.update(IMAGE_GRAPH_LAYER_COUNTS)
        expected.update(
            {
                "referenced_checkpoint_tensors": GRAPH_REFERENCED_TENSORS[index],
                "serialized_bytes": bundle.section_size[PLAN_SECTIONS[index]],
                "serialized_sha256": bundle.section_sha256[PLAN_SECTIONS[index]],
                "graph_complete": True,
            }
        )
        _exact_contract(graph, expected, f"build receipt graph {index} binding")


def _validate_attention(value: Any, label: str) -> None:
    attention = _exact_keys(
        value,
        {
            "implementation",
            "operator",
            "api",
            "block_count",
            "head_dimension",
            "query_form",
            "key_value_form",
            "output_form",
            "normalization",
            "causal_mask",
            "decomposable",
            "fused_kernel_intent",
            "metadata_prefix",
            "metadata_index_width",
            "q_scale_formula",
            "k_scale_formula",
            "effective_score_scale",
            "scale_dtype",
        },
        label,
    )
    expected = {
        "implementation": "tensorrt_iattention_v2",
        "operator": "IAttention",
        "api": "addAttentionV2",
        "block_count": 16,
        "head_dimension": 96,
        "query_form": "padded_bhnd",
        "key_value_form": "padded_bhnd",
        "output_form": "padded_bhnd",
        "normalization": "softmax",
        "causal_mask": "none",
        "decomposable": False,
        "fused_kernel_intent": True,
        "metadata_prefix": "trtmc.sam2.iattention.block.",
        "metadata_index_width": 2,
        "q_scale_formula": "1/sqrt(head_dimension)",
        "k_scale_formula": "none",
        "effective_score_scale": "1/sqrt(head_dimension)",
        "scale_dtype": "bf16",
    }
    _exact_contract(attention, expected, label)


def _validate_runtime(value: Any, label: str) -> dict[str, Any]:
    runtime = _exact_keys(
        value,
        {
            "gpu_device",
            "gpu_name",
            "compute_capability",
            "global_memory_bytes",
            "tensorrt_version",
            "tensorrt_abi",
            "cuda_runtime_version",
            "cuda_driver_version",
            "hostname",
            "started_at_utc",
            "ended_at_utc",
            "gpu_uuid",
            "pci_bus_id",
            "cxx_compiler_id",
            "cxx_compiler_version",
            "cxx_language_standard",
            "engine_profiling_verbosity",
            "execution_context_nvtx_verbosity",
        },
        label,
    )
    if (
        runtime["gpu_name"] != GPU_NAME
        or runtime["compute_capability"] != COMPUTE_CAPABILITY
        or runtime["tensorrt_version"] != TENSORRT_VERSION
        or runtime["tensorrt_abi"] != TENSORRT_ABI
        or runtime["engine_profiling_verbosity"] != "detailed"
        or runtime["execution_context_nvtx_verbosity"] != "none"
        or runtime["cxx_language_standard"] != 201703
    ):
        _fail(f"{label} target provenance drifted")
    _integer(runtime["gpu_device"], f"{label} GPU device")
    _integer(runtime["global_memory_bytes"], f"{label} GPU memory", positive=True)
    started_at = _timestamp(runtime["started_at_utc"], f"{label} start")
    ended_at = _timestamp(runtime["ended_at_utc"], f"{label} end")
    if started_at > ended_at:
        _fail(f"{label} ends before it starts")
    for key in (
        "cuda_runtime_version",
        "cuda_driver_version",
        "hostname",
        "gpu_uuid",
        "pci_bus_id",
        "cxx_compiler_id",
        "cxx_compiler_version",
    ):
        if not isinstance(runtime[key], str) or not runtime[key]:
            _fail(f"{label} {key} is empty")
    return runtime


def _validate_replay(value: Any, expected_index: int, label: str) -> None:
    replay = _exact_keys(
        value,
        {
            "index",
            "mask_sha256",
            "bbox_sha256",
            "foreground_pixels",
            "frame_iou",
            "macro_iou",
            "global_iou",
            "bbox_iou",
            "bbox_max_coordinate_error",
            "bbox_score_error",
            "bbox_label_exact",
            "candidate_bbox",
            "reference_bbox",
            "passes",
        },
        label,
    )
    if (
        type(replay["index"]) is not int
        or replay["index"] != expected_index
        or replay["passes"] is not True
    ):
        _fail(f"{label} status/index drifted")
    _sha256(replay["mask_sha256"], f"{label} mask")
    _sha256(replay["bbox_sha256"], f"{label} bbox")
    if not isinstance(replay["foreground_pixels"], list) or len(replay["foreground_pixels"]) != 5:
        _fail(f"{label} foreground inventory drifted")
    for index, count in enumerate(replay["foreground_pixels"]):
        _integer(count, f"{label} foreground {index}")
    frame_iou = replay["frame_iou"]
    if not isinstance(frame_iou, list) or len(frame_iou) != 5:
        _fail(f"{label} frame IoU inventory drifted")
    if any(not 0.98 <= _number(item, f"{label} frame IoU") <= 1.0 for item in frame_iou):
        _fail(f"{label} frame IoU gate failed")
    gates = (
        ("macro_iou", 0.99, 1.0),
        ("global_iou", 0.99, 1.0),
        ("bbox_iou", 0.995, 1.0),
        ("bbox_max_coordinate_error", 0.0, 0.5),
        ("bbox_score_error", 0.0, 0.01),
    )
    for key, minimum, maximum in gates:
        if not minimum <= _number(replay[key], f"{label} {key}") <= maximum:
            _fail(f"{label} {key} gate failed")
    if replay["bbox_label_exact"] is not True:
        _fail(f"{label} bbox label gate failed")
    for key in ("candidate_bbox", "reference_bbox"):
        bbox = _exact_keys(replay[key], {"label", "score", "original_image_xyxy"}, f"{label} {key}")
        if (
            type(bbox["label"]) is not int
            or not isinstance(bbox["original_image_xyxy"], list)
            or len(bbox["original_image_xyxy"]) != 4
        ):
            _fail(f"{label} {key} structure drifted")
        _number(bbox["score"], f"{label} {key} score")
        for coordinate in bbox["original_image_xyxy"]:
            _number(coordinate, f"{label} {key} coordinate")
    if replay["candidate_bbox"]["label"] != replay["reference_bbox"]["label"]:
        _fail(f"{label} claims exact labels but candidate/reference labels differ")


def _validate_common_receipt(receipt: dict[str, Any], bundle: BundleSnapshot, label: str) -> None:
    if (
        receipt.get("schema_version") != BENCHMARK_RECEIPT_SCHEMA_VERSION
        or receipt.get("family") != FAMILY
        or receipt.get("workload") != WORKLOAD
    ):
        _fail(f"{label} identity/schema drifted")
    status = _exact_keys(
        receipt.get("status"),
        {
            "accuracy_qualified_for_this_diagnostic",
            "runtime_eligible",
            "performance_claim",
            "timing_performed",
            "outlier_filtering",
        },
        f"{label} status",
    )
    if (
        status["accuracy_qualified_for_this_diagnostic"] is not True
        or status["runtime_eligible"] is not False
        or status["performance_claim"] is not False
        or status["outlier_filtering"] is not False
    ):
        _fail(f"{label} diagnostic claim boundary drifted")
    _validate_attention(receipt.get("image_attention"), f"{label} image attention")
    assets = receipt.get("assets")
    if not isinstance(assets, dict):
        _fail(f"{label} assets are missing")
    common_expected = {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_config_sha256": CONFIG_SHA256,
        "golden_manifest_sha256": GOLDEN_MANIFEST_SHA256,
        "golden_masks_sha256": GOLDEN_MASKS_SHA256,
        "encoded_jpeg_sha256": list(ENCODED_JPEG_SHA256),
        "decoded_rgb_sha256": list(DECODED_JPEG_SHA256),
        "native_bundle_sha256": bundle.sha256,
        "native_build_receipt_sha256": bundle.section_sha256["sam2_build_receipt.json"],
    }
    for key, expected in common_expected.items():
        if assets.get(key) != expected:
            _fail(f"{label} asset {key} drifted")
    plans = assets.get("native_plans")
    expected_plans = [
        {"section": name, "sha256": bundle.section_sha256[name]} for name in PLAN_SECTIONS
    ]
    if plans != expected_plans:
        _fail(f"{label} six-plan lineage drifted")
    for key in (
        "benchmark_executable_sha256",
        "benchmark_source_manifest_sha256",
        "benchmark_source_closure_sha256",
    ):
        _sha256(assets.get(key), f"{label} {key}")
    if assets.get("benchmark_source_closure_role") != BENCHMARK_SOURCE_CLOSURE_ROLE:
        _fail(f"{label} benchmark source closure role drifted")
    accuracy = receipt.get("accuracy")
    if not isinstance(accuracy, dict):
        _fail(f"{label} accuracy is missing")
    _exact_contract(accuracy.get("thresholds"), THRESHOLDS, f"{label} semantic thresholds")
    if (
        accuracy.get("repeat_hashes_exact") is not False
        or accuracy.get("foreground_counts_exact") is not False
    ):
        _fail(f"{label} improperly requires bitwise repeat identity")
    if accuracy.get("repeat_contract") != REPEAT_CONTRACT:
        _fail(f"{label} repeat contract drifted")
    _validate_runtime(receipt.get("runtime"), f"{label} runtime")


def _receipt_number(value: Fraction | float) -> float:
    as_double = float(value)
    if not math.isfinite(as_double) or as_double < 0.0:
        _fail("timing derivation produced a negative or non-finite number")
    return math.floor(as_double * 1_000_000.0 + 0.5) / 1_000_000.0


def _timing_row_nanoseconds(row: Any, expected_index: int, label: str) -> tuple[int, int, int]:
    row = _exact_keys(
        row,
        {
            "index",
            "native_prefill_ns",
            "native_tracker_ns",
            "closest_envelope_total_ns",
            "native_prefill_ms",
            "native_tracker_ms",
            "closest_envelope_total_ms",
        },
        label,
    )
    if type(row["index"]) is not int or row["index"] != expected_index:
        _fail(f"{label} index drifted")
    values = []
    for key in ("native_prefill_ns", "native_tracker_ns", "closest_envelope_total_ns"):
        value = _integer(row[key], f"{label} {key}", positive=True)
        if value > (1 << 64) - 1:
            _fail(f"{label} {key} exceeds uint64")
        values.append(value)
    prefill, tracker, total = values
    if prefill > (1 << 64) - 1 - tracker or total != prefill + tracker:
        _fail(f"{label} boundary drifted")
    for ns_key, ms_key in (
        ("native_prefill_ns", "native_prefill_ms"),
        ("native_tracker_ns", "native_tracker_ms"),
        ("closest_envelope_total_ns", "closest_envelope_total_ms"),
    ):
        expected_ms = _receipt_number(Fraction(row[ns_key], 1_000_000))
        if _number(row[ms_key], f"{label} {ms_key}") != expected_ms:
            _fail(f"{label} {ms_key} is not derived from {ns_key}")
    return prefill, tracker, total


def _metric_summary(values: list[int]) -> dict[str, float]:
    ordered = sorted(values)
    middle = len(ordered) // 2
    median_ns = Fraction(ordered[middle - 1] + ordered[middle], 2)
    return {
        "mean_ms": _receipt_number(Fraction(sum(values), len(values) * 1_000_000)),
        "median_ms": _receipt_number(median_ns / 1_000_000),
        "p90_ms": _receipt_number(Fraction(ordered[89], 1_000_000)),
        "min_ms": _receipt_number(Fraction(ordered[0], 1_000_000)),
        "max_ms": _receipt_number(Fraction(ordered[-1], 1_000_000)),
    }


def _validate_timing_summary(value: Any, rows: list[tuple[int, int, int]]) -> None:
    summary = _exact_keys(
        value,
        {"native_prefill", "native_tracker", "closest_envelope_total"},
        "regular timing summary",
    )
    for metric_index, name in enumerate(
        ("native_prefill", "native_tracker", "closest_envelope_total")
    ):
        metric = _exact_keys(
            summary[name],
            {"mean_ms", "median_ms", "p90_ms", "min_ms", "max_ms"},
            f"regular timing summary {name}",
        )
        expected = _metric_summary([row[metric_index] for row in rows])
        for key, expected_value in expected.items():
            if _number(metric[key], f"regular timing summary {name} {key}") != expected_value:
                _fail(f"regular timing summary {name} {key} drifted")


def _validate_receipts(
    bundle: BundleSnapshot, q3_snapshot: Snapshot, regular_snapshot: Snapshot
) -> tuple[dict[str, Any], dict[str, Any]]:
    q3 = _parse_json(q3_snapshot.data, "Q3 receipt", trailing_newline=True)
    regular = _parse_json(regular_snapshot.data, "regular receipt", trailing_newline=True)
    q3_keys = (
        "schema_version",
        "family",
        "workload",
        "mode",
        "accuracy_only",
        "timing_performed",
        "status",
        "process_model",
        "sequence",
        "assets",
        "runtime",
        "image_attention",
        "accuracy",
    )
    regular_keys = (
        "schema_version",
        "family",
        "workload",
        "mode",
        "accuracy_only",
        "timing_performed",
        "status",
        "process_model",
        "sequence",
        "timing_boundaries",
        "assets",
        "runtime",
        "image_attention",
        "accuracy",
        "timing",
        "delivered_baseline_reference",
    )
    _exact_keys(q3, q3_keys, "Q3 receipt")
    _exact_keys(regular, regular_keys, "regular receipt")
    _validate_common_receipt(q3, bundle, "Q3 receipt")
    _validate_common_receipt(regular, bundle, "regular receipt")
    _exact_contract(q3["process_model"], Q3_PROCESS_MODEL, "Q3 process model")
    _exact_contract(regular["process_model"], REGULAR_PROCESS_MODEL, "regular process model")
    if (
        q3["mode"] != "accuracy_only"
        or q3["accuracy_only"] is not True
        or q3["timing_performed"] is not False
        or q3["status"]["timing_performed"] is not False
    ):
        _fail("Q3 receipt is not accuracy-only")
    _exact_contract(
        q3["sequence"],
        {
            "accuracy_replays": 3,
            "frames_per_replay": 5,
            "reset_before_each_replay": True,
            "order": "Q3_only",
            "warmup_rows": 0,
            "measurement_rows": 0,
            "postqualification_replays": 0,
        },
        "Q3 sequence",
    )
    q3_accuracy = _exact_keys(
        q3["accuracy"],
        {
            "thresholds",
            "repeat_hashes_exact",
            "foreground_counts_exact",
            "repeat_contract",
            "replays",
        },
        "Q3 accuracy",
    )
    q3_replays = q3_accuracy["replays"]
    if not isinstance(q3_replays, list) or len(q3_replays) != 3:
        _fail("Q3 replay count drifted")
    for index, replay in enumerate(q3_replays):
        _validate_replay(replay, index, f"Q3 replay {index}")

    if (
        regular["mode"] != "diagnostic_benchmark"
        or regular["accuracy_only"] is not False
        or regular["timing_performed"] is not True
        or regular["status"]["timing_performed"] is not True
    ):
        _fail("regular receipt is not a timed diagnostic")
    _exact_contract(
        regular["sequence"],
        {
            "prequalification_replays": 3,
            "warmup_rows": 3,
            "measurement_rows": 100,
            "postqualification_replays": 1,
            "order": "Q3_then_W3_then_N100_then_Q1",
            "accuracy_materialization_between_timing_rows": False,
        },
        "regular sequence",
    )
    regular_accuracy = _exact_keys(
        regular["accuracy"],
        {
            "thresholds",
            "repeat_hashes_exact",
            "foreground_counts_exact",
            "repeat_contract",
            "prequalification",
            "postqualification",
        },
        "regular accuracy",
    )
    pre = regular_accuracy["prequalification"]
    post = regular_accuracy["postqualification"]
    if pre != q3_replays:
        _fail("regular prequalification is not the exact published Q3 evidence")
    if not isinstance(post, list) or len(post) != 1:
        _fail("regular postqualification count drifted")
    _validate_replay(post[0], 0, "regular Q1 replay")
    regular_assets = regular["assets"]
    if (
        regular_assets.get("q3_receipt_sha256") != q3_snapshot.sha256
        or regular_assets.get("q3_receipt_size_bytes") != q3_snapshot.size_bytes
        or regular_assets.get("q3_receipt_role")
        != "exclusive same-process same-bundle Q3 receipt published before W3"
        or regular_assets.get("baseline_receipt_sha256") != BASELINE_RECEIPT_SHA256
        or regular_assets.get("baseline_capture_script_sha256") != BASELINE_CAPTURE_SHA256
    ):
        _fail("regular receipt does not bind the exact Q3/baseline evidence")
    q3_assets = q3["assets"]
    for key in (
        "baseline_receipt_sha256",
        "baseline_capture_script_sha256",
        "q3_receipt_sha256",
        "q3_receipt_size_bytes",
        "q3_receipt_role",
    ):
        if key in q3_assets:
            _fail("Q3 receipt contains post-Q3 performance evidence")
    q3_asset_keys = {
        "checkpoint_sha256",
        "source_config_sha256",
        "golden_manifest_sha256",
        "golden_masks_sha256",
        "encoded_jpeg_sha256",
        "decoded_rgb_sha256",
        "native_bundle_sha256",
        "native_build_receipt_sha256",
        "native_plans",
        "benchmark_executable_sha256",
        "benchmark_source_manifest_sha256",
        "benchmark_source_closure_sha256",
        "benchmark_source_closure_role",
    }
    regular_asset_keys = q3_asset_keys | {
        "baseline_receipt_sha256",
        "baseline_capture_script_sha256",
        "q3_receipt_sha256",
        "q3_receipt_size_bytes",
        "q3_receipt_role",
    }
    _exact_keys(q3_assets, q3_asset_keys, "Q3 assets")
    _exact_keys(regular_assets, regular_asset_keys, "regular assets")
    for key in q3_asset_keys:
        if regular_assets[key] != q3_assets[key]:
            _fail(f"Q3/regular asset lineage differs at {key}")
    q3_runtime = dict(q3["runtime"])
    regular_runtime = dict(regular["runtime"])
    q3_end = q3_runtime.pop("ended_at_utc")
    regular_end = regular_runtime.pop("ended_at_utc")
    if q3_runtime != regular_runtime or _timestamp(q3_end, "Q3 runtime end") > _timestamp(
        regular_end, "regular runtime end"
    ):
        _fail("Q3/regular runtime lineage is not one ordered process run")
    _exact_contract(regular["timing_boundaries"], TIMING_BOUNDARIES, "regular timing boundaries")
    _exact_contract(
        regular["delivered_baseline_reference"],
        DELIVERED_BASELINE_REFERENCE,
        "regular delivered baseline reference",
    )
    timing = _exact_keys(
        regular["timing"],
        {
            "sample_count",
            "excluded_rows",
            "p90_method",
            "outlier_removal",
            "warmup_rows",
            "measurement_rows",
            "summary",
        },
        "regular timing",
    )
    if (
        _integer(timing["sample_count"], "regular timing sample count") != 100
        or _integer(timing["excluded_rows"], "regular timing excluded rows") != 0
        or timing["p90_method"] != P90_METHOD
        or timing["outlier_removal"] is not False
    ):
        _fail("regular timing sample contract drifted")
    measurement_rows: list[tuple[int, int, int]] = []
    for label, rows, count in (
        ("warmup", timing["warmup_rows"], 3),
        ("measurement", timing["measurement_rows"], 100),
    ):
        if not isinstance(rows, list) or len(rows) != count:
            _fail(f"regular {label} rows drifted")
        for index, row in enumerate(rows):
            values = _timing_row_nanoseconds(row, index, f"regular {label} row {index}")
            if label == "measurement":
                measurement_rows.append(values)
    _validate_timing_summary(timing["summary"], measurement_rows)
    return q3, regular


def _canonical_sorted(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _open_output_parent(path: Path, label: str) -> int:
    path = _canonical_path(path, label)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path.parent, flags)
    except OSError as error:
        _fail(f"cannot open no-follow {label} parent: {error.strerror}")
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            _fail(f"{label} parent is not a directory")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _lstat_at(directory: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _same_owned_regular(value: os.stat_result | None, device: int, inode: int) -> bool:
    return (
        value is not None
        and stat.S_ISREG(value.st_mode)
        and value.st_dev == device
        and value.st_ino == inode
    )


def _unlink_owned(
    directory: int,
    name: str,
    device: int,
    inode: int,
    *,
    role: str,
) -> str:
    """Remove only the named link that still identifies the inode we created."""

    try:
        before = _lstat_at(directory, name)
    except OSError as error:
        return f"{role}=lstat-error({error.strerror})"
    if before is None:
        return f"{role}=absent"
    if not _same_owned_regular(before, device, inode):
        return f"{role}=preserved-nonowned"
    try:
        os.unlink(name, dir_fd=directory)
    except OSError as error:
        return f"{role}=unlink-error({error.strerror})"
    try:
        after = _lstat_at(directory, name)
    except OSError as error:
        return f"{role}=removed-verify-error({error.strerror})"
    if _same_owned_regular(after, device, inode):
        return f"{role}=unlink-made-no-progress"
    if after is None:
        return f"{role}=removed"
    return f"{role}=removed-owned-preserved-replacement"


def _status_is_removed(status: str) -> bool:
    return status.endswith("=removed")


def _retry_owned_unlink(
    directory: int,
    name: str,
    device: int,
    inode: int,
    *,
    role: str,
) -> list[str]:
    statuses = [_unlink_owned(directory, name, device, inode, role=role)]
    try:
        still_owned = _same_owned_regular(_lstat_at(directory, name), device, inode)
    except OSError as error:
        statuses.append(f"{role}-retry=lstat-error({error.strerror})")
        return statuses
    if still_owned:
        statuses.append(_unlink_owned(directory, name, device, inode, role=f"{role}-retry"))
    return statuses


def _temp_absent_or_nonowned(
    directory: int, name: str, device: int, inode: int
) -> tuple[bool, str]:
    try:
        value = _lstat_at(directory, name)
    except OSError as error:
        return False, f"temp=lstat-error({error.strerror})"
    if value is None:
        return True, "temp=absent"
    if not _same_owned_regular(value, device, inode):
        return False, "temp=preserved-nonowned"
    return False, "temp=owned-present"


def _cleanup_failed_publication(
    *,
    directory: int,
    temp_name: str | None,
    final_name: str,
    descriptor: int,
    published: bool,
) -> str:
    """Best-effort exact cleanup; never remove a path bound to another inode."""

    statuses: list[str] = []
    if descriptor < 0:
        return "no-owned-inode"
    try:
        identity = os.fstat(descriptor)
    except OSError as error:
        return f"owned-inode-fstat=error({error.strerror})"
    device, inode = identity.st_dev, identity.st_ino
    temp_absent = True
    if temp_name is not None:
        status = _unlink_owned(directory, temp_name, device, inode, role="temp")
        statuses.append(status)
        temp_absent, temp_check = _temp_absent_or_nonowned(directory, temp_name, device, inode)
        if not temp_absent:
            # A one-shot unlink fault is recoverable.  Authenticate the inode again
            # before retrying so a concurrent replacement is never removed.
            retry = _unlink_owned(directory, temp_name, device, inode, role="temp-retry")
            statuses.extend((temp_check, retry))
            temp_absent, temp_check = _temp_absent_or_nonowned(directory, temp_name, device, inode)
            if not temp_absent:
                statuses.append(temp_check)
    if published:
        statuses.extend(_retry_owned_unlink(directory, final_name, device, inode, role="final"))
    if temp_absent:
        try:
            os.fsync(directory)
            statuses.append("parent-fsync=ok")
        except OSError as error:
            statuses.append(f"parent-fsync=error({error.strerror})")
    else:
        statuses.append("parent-fsync=skipped-temp-present")
    return ", ".join(statuses)


def _require_output_absent(path: Path, label: str) -> None:
    path = _canonical_path(path, label)
    directory = _open_output_parent(path, label)
    try:
        try:
            value = _lstat_at(directory, path.name)
        except OSError as error:
            _fail(f"cannot inspect no-follow {label}: {error.strerror}")
        if value is not None:
            _fail(f"{label} must be absent")
    finally:
        os.close(directory)


def _write_exclusive(path: Path, data: bytes, label: str) -> _PublishedFile:
    """Durably publish complete bytes without ever replacing an existing target."""

    path = _canonical_path(path, label)
    directory = _open_output_parent(path, label)
    descriptor = -1
    temp_name: str | None = None
    published = False
    try:
        for _ in range(128):
            candidate = f".sam2-qualification-{secrets.token_hex(16)}.tmp"
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | os.O_CLOEXEC
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=directory,
                )
                temp_name = candidate
                break
            except FileExistsError:
                continue
        if descriptor < 0 or temp_name is None:
            raise OSError("cannot allocate a unique hidden temporary file")

        created = os.fstat(descriptor)
        if not stat.S_ISREG(created.st_mode) or created.st_nlink != 1:
            raise OSError("created temporary output is not a unique regular file")
        consumed = 0
        while consumed != len(data):
            written = os.write(descriptor, data[consumed:])
            if written <= 0:
                raise OSError("temporary output write made no progress")
            consumed += written
        os.fsync(descriptor)

        # link(2) is the no-replace atomic publication point.  The final name
        # cannot expose partial data because the source inode is already complete
        # and file-fsynced.
        os.link(
            temp_name,
            path.name,
            src_dir_fd=directory,
            dst_dir_fd=directory,
            follow_symlinks=False,
        )
        published = True
        final_stat = _lstat_at(directory, path.name)
        if not _same_owned_regular(final_stat, created.st_dev, created.st_ino):
            raise OSError("published target does not identify the prepared inode")

        temp_status = _unlink_owned(
            directory, temp_name, created.st_dev, created.st_ino, role="temp"
        )
        if not _status_is_removed(temp_status):
            raise OSError(f"cannot remove authenticated temporary link: {temp_status}")
        temp_name = None
        os.fsync(directory)
        result = _PublishedFile(
            path=path,
            label=label,
            device=created.st_dev,
            inode=created.st_ino,
            descriptor=descriptor,
        )
        descriptor = -1
        return result
    except (OSError, ValueError) as error:
        cleanup = _cleanup_failed_publication(
            directory=directory,
            temp_name=temp_name,
            final_name=path.name,
            descriptor=descriptor,
            published=published,
        )
        detail = error.strerror if isinstance(error, OSError) and error.strerror else str(error)
        _fail(f"cannot publish exclusive {label}: {detail}; cleanup: {cleanup}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory)


def _rollback_publication(publication: _PublishedFile) -> str:
    """Durably remove our published link if it still names our open inode."""

    directory = _open_output_parent(publication.path, publication.label)
    try:
        statuses = _retry_owned_unlink(
            directory,
            publication.path.name,
            publication.device,
            publication.inode,
            role="record",
        )
        try:
            os.fsync(directory)
            fsync_status = "parent-fsync=ok"
        except OSError as error:
            fsync_status = f"parent-fsync=error({error.strerror})"
        return f"{', '.join(statuses)}, {fsync_status}"
    finally:
        os.close(directory)


def generate(
    *,
    bundle_path: Path,
    q3_receipt_path: Path,
    regular_receipt_path: Path,
    record_output: Path,
    audit_output: Path,
    authority_id: str,
    authority_serial: int,
    generated_at_utc: str,
) -> tuple[bytes, bytes]:
    if AUTHORITY_RE.fullmatch(authority_id) is None:
        _fail("authority_id is not bounded canonical ASCII")
    _integer(authority_serial, "authority_serial", positive=True)
    _timestamp(generated_at_utc, "generated_at_utc")
    outputs = {
        _canonical_path(record_output, "record output"),
        _canonical_path(audit_output, "audit output"),
    }
    if len(outputs) != 2:
        _fail("record and audit outputs must differ")
    _require_output_absent(record_output, "record output")
    _require_output_absent(audit_output, "audit output")

    bundle = _snapshot_bundle(bundle_path)
    q3_snapshot = _snapshot_regular_file(q3_receipt_path, 16 * 1024 * 1024, "Q3 receipt")
    regular_snapshot = _snapshot_regular_file(
        regular_receipt_path, 64 * 1024 * 1024, "regular receipt"
    )
    q3, regular = _validate_receipts(bundle, q3_snapshot, regular_snapshot)
    runtime = q3["runtime"]
    assets = q3["assets"]
    record = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "artifact_type": RECORD_ARTIFACT_TYPE,
        "authority_id": authority_id,
        "authority_serial": authority_serial,
        "self_authorizing": False,
        "scope": {
            "family": bundle.header["family"],
            "model_id": bundle.header["model_id"],
            "engine_contract_version": bundle.config["engine_contract_version"],
            "runtime_strategy": bundle.header["runtime_strategy"],
            "precision": bundle.header["precision"],
            "gpu_name": runtime["gpu_name"],
            "compute_capability": runtime["compute_capability"],
            "tensorrt_version": runtime["tensorrt_version"],
            "tensorrt_abi": runtime["tensorrt_abi"],
        },
        "bundle": {
            "sha256": bundle.sha256,
            "size_bytes": bundle.size_bytes,
            "embedded_config_sha256": bundle.section_sha256["config.json"],
            "build_receipt_sha256": bundle.section_sha256["sam2_build_receipt.json"],
            "plans": [
                {"section": name, "sha256": bundle.section_sha256[name]} for name in PLAN_SECTIONS
            ],
        },
        "accuracy_evidence": {
            "receipt_sha256": q3_snapshot.sha256,
            "receipt_size_bytes": q3_snapshot.size_bytes,
            "regular_receipt_sha256": regular_snapshot.sha256,
            "regular_receipt_size_bytes": regular_snapshot.size_bytes,
            "mode": "accuracy_only",
            "policy_id": POLICY_ID,
            "replay_count": 3,
            "frames_per_replay": 5,
            "reset_before_each_replay": True,
            "all_semantic_gates_passed": True,
            "timing_performed": False,
            "golden_manifest_sha256": assets["golden_manifest_sha256"],
            "golden_masks_sha256": assets["golden_masks_sha256"],
            "benchmark_executable_sha256": assets["benchmark_executable_sha256"],
            "benchmark_source_manifest_sha256": assets["benchmark_source_manifest_sha256"],
            "benchmark_source_closure_sha256": assets["benchmark_source_closure_sha256"],
        },
        "generated_at_utc": generated_at_utc,
    }
    record_bytes = _canonical_sorted(record)
    record_sha256 = hashlib.sha256(record_bytes).hexdigest()
    generator_snapshot = _snapshot_regular_file(
        Path(__file__).resolve(), 4 * 1024 * 1024, "generator"
    )
    audit = {
        "schema_version": 1,
        "artifact_type": AUDIT_ARTIFACT_TYPE,
        "self_authorizing": False,
        "pin_mutation_supported": False,
        "generator": {
            "sha256": generator_snapshot.sha256,
            "size_bytes": generator_snapshot.size_bytes,
        },
        "inputs": {
            "bundle": {"sha256": bundle.sha256, "size_bytes": bundle.size_bytes},
            "q3_receipt": {"sha256": q3_snapshot.sha256, "size_bytes": q3_snapshot.size_bytes},
            "regular_receipt": {
                "sha256": regular_snapshot.sha256,
                "size_bytes": regular_snapshot.size_bytes,
            },
        },
        "record": {"sha256": record_sha256, "size_bytes": len(record_bytes)},
        "derived_gates": {
            "same_bundle": True,
            "same_process_runtime_lineage": True,
            "q3_published_before_w3": True,
            "q3_matches_regular_prequalification": True,
            "q3_all_semantic_gates_passed": True,
            "q1_all_semantic_gates_passed": True,
            "w3_rows": 3,
            "n100_rows": 100,
            "performance_claim": False,
            "runtime_eligible": False,
            "thresholds": THRESHOLDS,
        },
    }
    audit_bytes = _canonical_sorted(audit)
    record_publication = _write_exclusive(record_output, record_bytes, "record output")
    audit_publication: _PublishedFile | None = None
    try:
        audit_publication = _write_exclusive(audit_output, audit_bytes, "audit output")
    except EvidenceError as error:
        try:
            rollback = _rollback_publication(record_publication)
        except Exception as rollback_error:  # pragma: no cover - final fail-closed guard
            rollback = f"rollback-raised({rollback_error})"
        raise EvidenceError(f"{error}; record rollback: {rollback}") from error
    finally:
        record_publication.close()
        if audit_publication is not None:
            audit_publication.close()
    return record_bytes, audit_bytes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--q3-receipt", required=True, type=Path)
    parser.add_argument("--regular-receipt", required=True, type=Path)
    parser.add_argument("--record-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--authority-id", required=True)
    parser.add_argument("--authority-serial", required=True, type=int)
    parser.add_argument("--generated-at", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        record, audit = generate(
            bundle_path=arguments.bundle,
            q3_receipt_path=arguments.q3_receipt,
            regular_receipt_path=arguments.regular_receipt,
            record_output=arguments.record_output,
            audit_output=arguments.audit_output,
            authority_id=arguments.authority_id,
            authority_serial=arguments.authority_serial,
            generated_at_utc=arguments.generated_at,
        )
    except EvidenceError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"record_sha256={hashlib.sha256(record).hexdigest()}")
    print(f"audit_sha256={hashlib.sha256(audit).hexdigest()}")
    print("self_authorizing=false; production pin mutation is not supported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
