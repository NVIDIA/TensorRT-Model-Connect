# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import struct
import sys

import pytest

from tests.tools.dynamic_memory_manifest_fixture import (
    complete_command_receipts,
    load_manifest_module,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "qualify_native_dynamic_memory_perf.py"
SPEC = importlib.util.spec_from_file_location(
    "qualify_native_dynamic_memory_perf", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
perf = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = perf
SPEC.loader.exec_module(perf)

pytestmark = [pytest.mark.unit, pytest.mark.dynamic_memory]
SOURCE_SHA = "1" * 64
GIT_HEAD = "2" * 40
SHORT_REQUEST_SHA = "5" * 64
MEDIUM_REQUEST_SHA = "6" * 64
RUNTIME_STACK = {
    "schema": 1,
    "sm": "sm103",
    "tensorrt": "11.2.0.113",
    "cuda_runtime": "13.3",
    "cudnn_backend": "9.20.0",
    "cudnn_frontend_revision":
        "7b9b711c22b6823e87150213ecd8449260db8610",
    "nvrtc": "13.3",
    "driver": "580.105.08",
}
RUNTIME_PLAN = {
    "schema": 1,
    "device": 0,
    "role": "history",
    "hq": 16,
    "hkv": 8,
    "d": 128,
    "C": 128,
    "Sq": 1,
    "T": 512,
    "stats": "lse",
    "heur": "A",
    "plan": "eng10_k24=7",
    "workspace_bytes": 0,
    "cudnn_version": 92000,
}
TOKENIZER_CONTRACT = {
    "schema_version": perf.TOKENIZER_CONTRACT_SCHEMA,
    "tokenizer_json_sha256": "8" * 64,
    "tokenizer_json_bytes": 4096,
    "tokenizer_add_special_tokens": False,
    "tokenizer_special_prefix_ids": [],
    "tokenizer_special_suffix_ids": [],
}
PROFILE_LIMITS = (128, 256, 512, 1024, 2048, 8192, 32768, 40960)
MODULE_RESIDENCY_RESERVE_BYTES = 1024


def _dynamic_bundle_bytes() -> bytes:
    capture = perf._load_capture_module()
    boundary = capture._load_boundary_module()
    engine_plan = b"decode-plan"
    prefill_plan = b"prefill-plan"
    runtime_config = b"{}"
    plans = [
        {
            "section_name": "engine_plan",
            "section_sha256": hashlib.sha256(engine_plan).hexdigest(),
            "role": "decode",
            "optimization_profile_count": len(PROFILE_LIMITS),
        },
        {
            "section_name": "prefill_engine_plan",
            "section_sha256": hashlib.sha256(prefill_plan).hexdigest(),
            "role": "prefill",
            "optimization_profile_count": 1,
        },
    ]
    stack = {key: value for key, value in RUNTIME_STACK.items() if key != "schema"}
    contract = {
        "contract_version": 2,
        "qualified_model_id": "Qwen/Qwen3-0.6B",
        "qualified_model_revision": "1" * 40,
        "qualified_config_sha256": "2" * 64,
        "qualified_target": "sm103 + TensorRT 11.2.0.113",
        "qualified_runtime_stack": stack,
        "native_kv_plugin_abi": 2,
        "model_context_limit": 40960,
        "prefill_chunk_limit": 1024,
        "kv_layout": "contiguous_v1",
        "kv_dtype": "bfloat16",
        "kv_bytes_per_token": 114688,
        "active_kv_profile_limits": list(PROFILE_LIMITS),
        "runtime_owned": True,
        "runtime_config_sha256": hashlib.sha256(runtime_config).hexdigest(),
        "module_residency_calibration": {
            "schema_version": 1,
            "measurement_kind": "nvml_process_cumulative_first_use",
            "cuda_module_loading_mode": "lazy",
            "qualified_runtime_stack_sha256": (
                boundary.qualified_runtime_stack_sha256(stack)
            ),
            "plan_set_sha256": (
                boundary.module_residency_plan_set_sha256(plans)
            ),
            "plans": plans,
            "profile_reserves": [
                {
                    "covering_profile_limit": limit,
                    "cumulative_reserve_bytes": (
                        MODULE_RESIDENCY_RESERVE_BYTES
                    ),
                }
                for limit in PROFILE_LIMITS
            ],
            "evidence_sha256": "3" * 64,
        },
    }
    sections = {
        "engine_plan": {"offset": 0, "size": len(engine_plan)},
        "prefill_engine_plan": {
            "offset": len(engine_plan),
            "size": len(prefill_plan),
        },
        "config.json": {
            "offset": len(engine_plan) + len(prefill_plan),
            "size": len(runtime_config),
        },
    }
    header = json.dumps(
        {
            "model_id": "Qwen/Qwen3-0.6B",
            "runtime_memory": contract,
            "sections": sections,
        }
    ).encode("utf-8")
    return (
        capture.BUNDLE_MAGIC
        + struct.pack("<Q", len(header))
        + header
        + engine_plan
        + prefill_plan
        + runtime_config
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cuda_jit_cache() -> dict:
    empty_sha = perf._canonical_sha([])
    return {
        "path": "/tmp/trtmc-cuda-cache",
        "path_source": "CUDA_CACHE_PATH",
        "cuda_cache_path_env": "/tmp/trtmc-cuda-cache",
        "cuda_cache_disable_env": None,
        "enabled": True,
        "initial_state": "cold",
        "worker_started_ns": 200,
        "worker_finished_ns": 300,
        "before": {
            "captured_at_ns": 100,
            "exists": False,
            "is_directory": False,
            "entry_count": 0,
            "file_count": 0,
            "total_bytes": 0,
            "metadata_sha256": empty_sha,
        },
        "after": {
            "captured_at_ns": 400,
            "exists": True,
            "is_directory": True,
            "entry_count": 1,
            "file_count": 1,
            "total_bytes": 4096,
            "metadata_sha256": "7" * 64,
        },
    }


def _runtime_libraries(root: Path) -> dict:
    directory = root / "runtime-libraries"
    directory.mkdir(parents=True, exist_ok=True)
    nvrtc = directory / "libnvrtc.so.13.3.33"
    builtins = directory / "libnvrtc-builtins.so.13.3"
    if not nvrtc.exists():
        nvrtc.write_bytes(b"nvrtc")
    if not builtins.exists():
        builtins.write_bytes(b"nvrtc-builtins")
    return {
        "directory": str(directory.resolve()),
        "live_nvrtc_version": "13.3",
        "nvrtc": {
            "path": str(nvrtc.resolve()),
            "basename": nvrtc.name,
            "sha256": _sha256(nvrtc),
            "size_bytes": nvrtc.stat().st_size,
        },
        "nvrtc_builtins": {
            "path": str(builtins.resolve()),
            "basename": builtins.name,
            "sha256": _sha256(builtins),
            "size_bytes": builtins.stat().st_size,
        },
    }


def _binary_identity(path: Path) -> dict:
    metadata = path.stat()
    return {
        "path": str(path.resolve()),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "size_bytes": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
        "sha256": _sha256(path),
    }


def _canonical_identities(*paths: Path) -> list[dict]:
    return sorted(
        (_binary_identity(path) for path in paths),
        key=lambda row: (
            row["path"],
            row["device"],
            row["inode"],
        ),
    )


def _manifest_artifact_identity(
    path: Path, *, artifact_key: str, relative_path: str
) -> dict:
    metadata = path.stat()
    return {
        "artifact_key": artifact_key,
        "relative_path": relative_path,
        "path": str(path.resolve()),
        "st_dev": metadata.st_dev,
        "st_ino": metadata.st_ino,
        "size_bytes": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
        "mode": stat.S_IMODE(metadata.st_mode),
        "sha256": _sha256(path),
    }


def _qualification_context(
    root: Path, *, static_bundle: Path, dynamic_bundle: Path
) -> dict:
    build = root / "build"
    build.mkdir(parents=True, exist_ok=True)
    relative_paths = {
        "trtmc": "trtmc",
        "benchmark_worker": "trtmc_benchmark_worker",
        "core": "libtrtmc_core.so",
        "trt_backend": "libtrtmc_backend_trt.so",
        "runtime_kv_plugin": "libtrtmc_trt_plugins.so",
        "model_qwen": "models/qwen/libtrtmc_model_qwen.so",
        "model_llama": "models/llama/libtrtmc_model_llama.so",
        "qualify": "trtmc_dynamic_memory_qualify",
        "nvrtc_optional_output_regression": (
            "trtmc_nvrtc_optional_output_regression"
        ),
        "surfaces": "trtmc_dynamic_memory_surfaces",
    }
    for key, relative in relative_paths.items():
        artifact = build / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"qualification-{key}".encode())
        if key in (
            "trtmc",
            "benchmark_worker",
            "qualify",
            "nvrtc_optional_output_regression",
            "surfaces",
        ):
            artifact.chmod(artifact.stat().st_mode | stat.S_IXUSR)
    cache = build / "CMakeCache.txt"
    cache.write_text(
        (
            f"CMAKE_HOME_DIRECTORY:INTERNAL={root.resolve()}\n"
            "TRTMC_TRT_BACKEND_ABI:STRING=11_2\n"
        ),
        encoding="utf-8",
    )
    cmake_cache = _manifest_artifact_identity(
        cache, artifact_key="cmake_cache", relative_path="CMakeCache.txt"
    )
    cmake_cache["configured_source"] = str(root.resolve())
    active_backend = build / "libtrtmc_backend_trt_11_2.so"
    active_backend.symlink_to("libtrtmc_backend_trt.so")
    artifact_paths = dict(relative_paths)
    artifact_paths["trt_backend"] = "libtrtmc_backend_trt_11_2.so"
    artifacts = {
        key: _manifest_artifact_identity(
            build / relative,
            artifact_key=key,
            relative_path=relative,
        )
        for key, relative in artifact_paths.items()
    }
    manifest_module = load_manifest_module(REPO_ROOT)
    commands = complete_command_receipts(
        manifest_module,
        repo_root=root,
        build_dir=build,
        output_dir=build,
        python=Path(sys.executable),
    )
    source_state = {
        "git_head": GIT_HEAD,
        "source_state_sha256": SOURCE_SHA,
        "git_dirty": False,
        "exact_head_gate_satisfied": True,
    }
    manifest = {
        "schema_version": "trtmc.dynamic-memory-test-manifest/v2",
        "repo_root": str(root.resolve()),
        "build_dir": str(build.resolve()),
        "python": sys.executable,
        "source_state_pre": source_state,
        "commands": commands,
        "passed": True,
        "source_state_post": source_state,
        "source_state_unchanged": True,
        "cmake_cache": cmake_cache,
        "build_artifacts": artifacts,
        "build_artifacts_sha256": perf._canonical_sha(artifacts),
        "clean_build_command_sha256": perf._canonical_sha(commands[0]),
    }
    manifest_path = build / "build-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_binding = {
        "path": str(manifest_path.resolve()),
        "sha256": _sha256(manifest_path),
        "schema_version": "trtmc.dynamic-memory-test-manifest/v2",
        "git_head": GIT_HEAD,
        "source_state_sha256": SOURCE_SHA,
        "build_artifacts_sha256": manifest["build_artifacts_sha256"],
    }
    runtime_trtmc = {
        "model_id": "Qwen/Qwen3-0.6B",
        "model_family": "qwen",
        "core": _binary_identity(build / relative_paths["core"]),
        "trt_backend": _binary_identity(
            build / artifact_paths["trt_backend"]
        ),
        "runtime_kv_plugin": _binary_identity(
            build / relative_paths["runtime_kv_plugin"]
        ),
        "model": _binary_identity(build / relative_paths["model_qwen"]),
    }
    runtime_libraries = _runtime_libraries(root)
    unrelated_mapped_dso = root / "runtime-libraries" / "libunrelated.so"
    unrelated_mapped_dso.write_bytes(b"unrelated-mapped-dso")
    static_mapped_dso_identities = sorted(
        [
            *(
                runtime_trtmc[field]
                for field in (
                    "core",
                    "trt_backend",
                    "runtime_kv_plugin",
                    "model",
                )
            ),
            _binary_identity(unrelated_mapped_dso),
        ],
        key=lambda row: (
            row["path"],
            row["device"],
            row["inode"],
        ),
    )
    dynamic_mapped_dso_identities = sorted(
        [
            *static_mapped_dso_identities,
            *_canonical_identities(
                Path(runtime_libraries["nvrtc"]["path"]),
                Path(runtime_libraries["nvrtc_builtins"]["path"]),
            ),
        ],
        key=lambda row: (
            row["path"],
            row["device"],
            row["inode"],
        ),
    )
    worker_identity = _binary_identity(
        build / relative_paths["benchmark_worker"]
    )
    plugin_identity = runtime_trtmc["runtime_kv_plugin"]
    toolchain = {
        "worker": worker_identity,
        "plugin_library": plugin_identity,
        "runtime_trtmc_libraries": runtime_trtmc,
        "build_manifest": manifest_binding,
        "capture_tool": str(MODULE_PATH.resolve()),
        "capture_tool_sha256": _sha256(MODULE_PATH),
    }
    environment = {"target": "unit-test", "gpu": "synthetic"}

    receipts: dict[str, Path] = {}
    for role, bundle in (
        ("exact-head-static-split", static_bundle),
        ("native-dynamic", dynamic_bundle),
    ):
        stdout = build / f"{role}.stdout"
        stderr = build / f"{role}.stderr"
        stdout.write_text("build ok\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        command = [
            str((build / "trtmc").resolve()),
            "build",
            "Qwen/Qwen3-0.6B",
        ]
        plugin = plugin_identity if role == "native-dynamic" else None
        mapping = (
            {
                "path": plugin_identity["path"],
                "device": plugin_identity["device"],
                "inode": plugin_identity["inode"],
                "deleted": False,
                "identity_sha256": perf._canonical_sha(plugin_identity),
            }
            if plugin is not None
            else None
        )
        build_mapped_dso_identities = (
            [plugin_identity] if plugin is not None else []
        )
        bundle_mtime = bundle.stat().st_mtime_ns
        receipt = {
            "schema_version": "trtmc.native-dynamic-memory-perf-build/v2",
            "artifact_role": role,
            "model_id": "Qwen/Qwen3-0.6B",
            "model_revision": "revision-pinned",
            "precision": "bf16",
            "target": "sm103",
            "bundle_build_id": (
                "static-build"
                if role == "exact-head-static-split"
                else "dynamic-build"
            ),
            "fresh_build": True,
            "artifact_reused": False,
            "bundle": str(bundle.resolve()),
            "bundle_sha256": _sha256(bundle),
            "bundle_bytes": bundle.stat().st_size,
            "bundle_mtime_ns": bundle_mtime,
            "build_started_ns": bundle_mtime - 1,
            "build_finished_ns": bundle_mtime + 1,
            "command": command,
            "command_sha256": perf._canonical_sha(command),
            "resolved_command": command,
            "resolved_command_sha256": perf._canonical_sha(command),
            "trtmc_executable": _binary_identity(build / "trtmc"),
            "cwd": str(root.resolve()),
            "stdout": str(stdout.resolve()),
            "stdout_sha256": _sha256(stdout),
            "stderr": str(stderr.resolve()),
            "stderr_sha256": _sha256(stderr),
            "git_head": GIT_HEAD,
            "prebuild_source_state_sha256": SOURCE_SHA,
            "postbuild_source_state_sha256": SOURCE_SHA,
            "source_state_pre": source_state,
            "source_state_post": source_state,
            "build_manifest": manifest_binding,
            "runtime_kv_plugin": plugin,
            "runtime_kv_plugin_mapping": mapping,
            "mapped_dso_identities": build_mapped_dso_identities,
        }
        receipt_path = build / f"{role}-receipt.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        receipts[role] = receipt_path
    return {
        "build_manifest": manifest_binding,
        "runtime_trtmc_libraries": runtime_trtmc,
        "plugin_identity": plugin_identity,
        "toolchain": toolchain,
        "environment": environment,
        "runtime_libraries": runtime_libraries,
        "static_mapped_dso_identities": (
            static_mapped_dso_identities
        ),
        "dynamic_mapped_dso_identities": (
            dynamic_mapped_dso_identities
        ),
        "unrelated_mapped_dso": unrelated_mapped_dso,
        "receipts": receipts,
        "source_state": source_state,
    }


def _complete_dynamic_receipt(
    *,
    calibration: dict,
    serialized_plan_bytes: int,
    resident_weight_bytes: int,
    resident_weight_copy_count: int,
    weight_streaming_active: bool,
) -> dict:
    capacity = 128
    bytes_per_token = 114_688
    capacity_total = 8 * 1024 * 1024 * 1024
    capacity_free = 4 * 1024 * 1024 * 1024
    settled_free = capacity_free - capacity * bytes_per_token
    return {
        "receipt_schema_version": 4,
        "contract_version": 2,
        "policy": "auto",
        "policy_fraction": 0.9,
        "requested_kv_bytes": 0,
        "post_load_free_bytes": capacity_free,
        "safety_reserve_bytes": 64 * 1024 * 1024,
        "module_residency_reserve_bytes": (
            MODULE_RESIDENCY_RESERVE_BYTES
        ),
        "module_residency_reserve_profile_limit": 128,
        "module_residency_plan_set_sha256": calibration[
            "plan_set_sha256"
        ],
        "module_residency_evidence_sha256": calibration[
            "evidence_sha256"
        ],
        "module_residency_cuda_module_loading_mode": calibration[
            "cuda_module_loading_mode"
        ],
        "model_context_limit": 40_960,
        "prefill_chunk_limit": 1_024,
        "request_context_limit": 128,
        "runtime_kv_capacity_tokens": capacity,
        "effective_request_limit": capacity,
        "kv_bytes_per_token": bytes_per_token,
        "kv_budget_bytes": capacity * bytes_per_token,
        "pre_load_free_bytes": capacity_free,
        "pre_load_total_bytes": capacity_total,
        "serialized_plan_bytes": serialized_plan_bytes,
        "resident_weight_bytes": resident_weight_bytes,
        "resident_weight_copy_count": resident_weight_copy_count,
        "engine_weight_bytes": resident_weight_bytes,
        "weight_streaming_active": weight_streaming_active,
        "post_load_total_bytes": capacity_total,
        "post_load_device_used_bytes": capacity_total - capacity_free,
        "capacity_decision_free_bytes": capacity_free,
        "capacity_decision_total_bytes": capacity_total,
        "capacity_decision_device_used_bytes": capacity_total - capacity_free,
        "capacity_decision_resident_overhead_bytes": 100,
        "final_non_kv_overhead_delta_bytes": 50,
        "settled_free_bytes": settled_free,
        "settled_total_bytes": capacity_total,
        "settled_device_used_bytes": capacity_total - settled_free,
        "settled_snapshot_unavailable_reason": None,
        "final_free_bytes": capacity_free,
        "final_total_bytes": capacity_total,
        "final_device_used_bytes": capacity_total - capacity_free,
        "context_device_memory_bytes": 10,
        "ordinary_device_input_bytes": 20,
        "ordinary_device_output_bytes": 30,
        "external_device_output_bytes": 40,
        "host_staging_bytes": 60,
        "graph_private_device_bytes": 50,
        "kv_reserved_bytes": capacity * bytes_per_token,
        "kv_committed_bytes": capacity * bytes_per_token,
        "kv_metadata_bytes": 0,
        "peak_device_bytes": capacity * bytes_per_token,
        "peak_device_bytes_scope": "device_wide",
        "peak_device_bytes_baseline": (
            "cuda_mem_get_info_before_engine_deserialization_free"
        ),
        "peak_device_sample_count": 2,
        "peak_device_sample_boundaries": [
            "after_runtime_kv_allocation",
            "after_successful_request_completion",
        ],
        "backend_owned_cache_input_bytes": 0,
        "backend_owned_cache_output_bytes": 0,
        "kv_allocation_id": 1,
        "solve_iterations": 1,
        "capped_by_model": False,
        "capped_by_request_limit": True,
        "measurement_sources": dict(perf._MEASUREMENT_SOURCES),
    }


def _write_result(
    path: Path,
    *,
    role: str,
    bundle: Path,
    prompt_kind: str,
    prefill_ms: float = 10.0,
    decode_ms: float = 100.0,
    output_tokens: int = 64,
    serialized_plan_bytes: int = 1_000,
    resident_weight_bytes: int = 2_000,
    resident_weight_copy_count: int = 2,
    weight_streaming_active: bool = False,
    fresh_build: bool = True,
    artifact_reused: bool = False,
    qualification_context: dict,
) -> dict:
    request_sha = (
        SHORT_REQUEST_SHA if prompt_kind == "short" else MEDIUM_REQUEST_SHA
    )
    runtime_attention_plans = (
        [] if role == "exact-head-static-split" else [dict(RUNTIME_PLAN)]
    )
    runtime_stack = (
        None if role == "exact-head-static-split" else dict(RUNTIME_STACK)
    )
    runtime_libraries = (
        None
        if role == "exact-head-static-split"
        else qualification_context["runtime_libraries"]
    )
    build_runtime_kv_plugin = (
        None
        if role == "exact-head-static-split"
        else qualification_context["plugin_identity"]
    )
    runtime_trtmc_libraries = qualification_context[
        "runtime_trtmc_libraries"
    ]
    build_manifest = qualification_context["build_manifest"]
    mapped_dso_identities = (
        qualification_context["static_mapped_dso_identities"]
        if role == "exact-head-static-split"
        else qualification_context["dynamic_mapped_dso_identities"]
    )
    toolchain = qualification_context["toolchain"]
    cuda_jit_cache = _cuda_jit_cache()
    generated_stream = list(range(output_tokens))
    runtime_receipt = {
        "serialized_plan_bytes": serialized_plan_bytes,
        "resident_weight_bytes": resident_weight_bytes,
        "resident_weight_copy_count": resident_weight_copy_count,
        "weight_streaming_active": weight_streaming_active,
        "measurement_sources": dict(perf._MEASUREMENT_SOURCES),
    }
    runtime_memory_contract = None
    module_residency_receipt = None
    if role == "native-dynamic":
        capture = perf._load_capture_module()
        header, _ = capture._bundle_header(bundle)
        runtime_memory_contract = (
            capture._dynamic_bundle_runtime_memory_contract(
                bundle,
                header,
            )
        )
        calibration = runtime_memory_contract[
            "module_residency_calibration"
        ]
        runtime_receipt = _complete_dynamic_receipt(
            calibration=calibration,
            serialized_plan_bytes=serialized_plan_bytes,
            resident_weight_bytes=resident_weight_bytes,
            resident_weight_copy_count=resident_weight_copy_count,
            weight_streaming_active=weight_streaming_active,
        )
        module_residency_receipt = {
            field: runtime_receipt[field]
            for field in perf._MODULE_RESIDENCY_RECEIPT_FIELDS
        }
    structural_identity = {
        "operation": "generate",
        "prompt_sha256": (
            "9" * 64 if prompt_kind == "short" else "a" * 64
        ),
        "prompt_utf8_bytes": 16 if prompt_kind == "short" else 128,
        "generation": {
            "generation_mode": "ar",
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "num_samples": 1,
            "eos_token_id": 2147483647,
            "use_chat_template": False,
            "stop_on_boxed_answer": False,
            "capture_generated_token_ids": True,
            "max_new_tokens": output_tokens,
        },
        "measurement": {"warmup": 2, "iterations": 3},
    }
    generation_workload = {
        "schema_version": perf.GENERATION_WORKLOAD_SCHEMA,
        "kind": "fixed_length_greedy_ar",
        "structural_identity": structural_identity,
        "structural_identity_sha256": perf._canonical_sha(
            structural_identity
        ),
        "measured_generated_token_ids": [
            generated_stream,
            generated_stream,
            generated_stream,
        ],
        "measured_generated_token_ids_sha256": perf._canonical_sha(
            [generated_stream, generated_stream, generated_stream]
        ),
        "token_stream_repeatable_within_case": True,
    }
    payload = {
        "schema_version": perf.RESULT_SCHEMA,
        "status": "completed",
        "case_name": f"{role}-{prompt_kind}",
        "case_digest": hashlib.sha256(
            f"{role}-{prompt_kind}".encode()
        ).hexdigest(),
        "model_id": "Qwen/Qwen3-0.6B",
        "operation": "generate",
        "timing_scope": "public_pipeline_call_wall",
        "warmup": 2,
        "iterations": 3,
        "observations": [
            {
                "iteration": index,
                "runtime_e2e_wall_ms": prefill_ms + decode_ms,
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "output_tokens": output_tokens,
            }
            for index in range(3)
        ],
        "qualification_provenance": {
            "git_head": GIT_HEAD,
            "source_state_sha256": SOURCE_SHA,
            "source_state_pre_sha256": SOURCE_SHA,
            "source_state_post_sha256": SOURCE_SHA,
            "source_state_unchanged": True,
            "prebuild_source_state_sha256": SOURCE_SHA,
            "postbuild_source_state_sha256": SOURCE_SHA,
            "source_state_boundaries": {
                name: {
                    "git_head": GIT_HEAD,
                    "source_state_sha256": SOURCE_SHA,
                    "git_dirty": False,
                    "exact_head_gate_satisfied": True,
                }
                for name in perf._SOURCE_STATE_BOUNDARY_NAMES
            },
            "bundle_sha256": _sha256(bundle),
            "request_sha256": request_sha,
            "model_revision": "revision-pinned",
            "precision": "bf16",
            "target": "sm103",
            "toolchain_sha256": perf._canonical_sha(toolchain),
            "benchmark_environment_sha256": perf._canonical_sha(
                qualification_context["environment"]
            ),
            "bundle_build_id": (
                "static-build"
                if role == "exact-head-static-split"
                else "dynamic-build"
            ),
            "artifact_role": role,
            "fresh_build": fresh_build,
            "artifact_reused": artifact_reused,
            "runtime_attention_plans_sha256": perf._canonical_sha(
                runtime_attention_plans
            ),
            "runtime_stack_sha256": perf._canonical_sha(runtime_stack),
            "runtime_libraries_sha256": perf._canonical_sha(
                runtime_libraries
            ),
            "runtime_trtmc_libraries_sha256": perf._canonical_sha(
                runtime_trtmc_libraries
            ),
            "mapped_dso_identities_sha256": perf._canonical_sha(
                mapped_dso_identities
            ),
            "build_runtime_kv_plugin_sha256": perf._canonical_sha(
                build_runtime_kv_plugin
            ),
            "build_manifest_sha256": perf._canonical_sha(build_manifest),
            "cuda_jit_cache_sha256": perf._canonical_sha(cuda_jit_cache),
            "generation_workload_sha256": perf._canonical_sha(
                generation_workload
            ),
            "tokenizer_contract_sha256": perf._canonical_sha(
                TOKENIZER_CONTRACT
            ),
            "bundle_runtime_memory_contract_sha256": (
                perf._canonical_sha(runtime_memory_contract)
                if runtime_memory_contract is not None
                else None
            ),
            "runtime_memory_receipt_sha256": perf._canonical_sha(
                runtime_receipt
            ),
            "runtime_module_residency_receipt_sha256": (
                perf._canonical_sha(module_residency_receipt)
                if module_residency_receipt is not None
                else None
            ),
        },
        "runtime_memory_receipt": runtime_receipt,
        "bundle_runtime_memory_contract": runtime_memory_contract,
        "runtime_module_residency_receipt": module_residency_receipt,
        "runtime_attention_plans": runtime_attention_plans,
        "runtime_stack": runtime_stack,
        "runtime_libraries": runtime_libraries,
        "runtime_trtmc_libraries": runtime_trtmc_libraries,
        "mapped_dso_identities": mapped_dso_identities,
        "build_runtime_kv_plugin": build_runtime_kv_plugin,
        "cuda_jit_cache": cuda_jit_cache,
        "generation_workload": generation_workload,
        "tokenizer_contract": dict(TOKENIZER_CONTRACT),
    }
    request_file = path.with_suffix(".request.json")
    request_file.write_text(
        json.dumps({"prompt_kind": prompt_kind}), encoding="utf-8"
    )
    worker_stdout = path.with_suffix(".worker.stdout")
    worker_stderr = path.with_suffix(".worker.stderr")
    worker_stdout.write_text("worker ok\n", encoding="utf-8")
    worker_stderr.write_text("", encoding="utf-8")
    raw_output = path.with_suffix(".raw")
    worker_command = [
        toolchain["worker"]["path"],
        "--request",
        str(request_file.resolve()),
        "--output",
        str(raw_output.resolve()),
    ]
    build_receipt = qualification_context["receipts"][role]
    payload["qualification_evidence"] = {
        "build_receipt": str(build_receipt.resolve()),
        "build_receipt_sha256": _sha256(build_receipt),
        "request_file": str(request_file.resolve()),
        "request_file_sha256": _sha256(request_file),
        "comparison_sequence_limit": 128,
        "worker_command": worker_command,
        "worker_command_sha256": perf._canonical_sha(worker_command),
        "toolchain": toolchain,
        "environment": qualification_context["environment"],
        "runtime_trtmc_libraries": runtime_trtmc_libraries,
        "mapped_dso_identities": mapped_dso_identities,
        "runtime_libraries": runtime_libraries,
        "build_runtime_kv_plugin": build_runtime_kv_plugin,
        "build_manifest": build_manifest,
        "bundle_runtime_memory_contract": runtime_memory_contract,
        "runtime_memory_receipt": runtime_receipt,
        "runtime_module_residency_receipt": module_residency_receipt,
        "source_state_pre": qualification_context["source_state"],
        "worker_stderr": str(worker_stderr.resolve()),
        "worker_stderr_sha256": _sha256(worker_stderr),
        "worker_stdout": str(worker_stdout.resolve()),
        "worker_stdout_sha256": _sha256(worker_stdout),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _evidence(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    static_bundle = tmp_path / "static.trtfb"
    dynamic_bundle = tmp_path / "dynamic.trtfb"
    dynamic_payload = _dynamic_bundle_bytes()
    dynamic_bundle.write_bytes(dynamic_payload)
    static_bundle.write_bytes(b"s" * len(dynamic_payload))
    qualification_context = _qualification_context(
        tmp_path,
        static_bundle=static_bundle,
        dynamic_bundle=dynamic_bundle,
    )
    paths = {
        "static_bundle": static_bundle,
        "dynamic_bundle": dynamic_bundle,
    }
    for prompt_kind in ("short", "medium"):
        for kind, role, bundle in (
            ("static", "exact-head-static-split", static_bundle),
            ("dynamic", "native-dynamic", dynamic_bundle),
        ):
            path = tmp_path / f"{kind}-{prompt_kind}.json"
            _write_result(
                path,
                role=role,
                bundle=bundle,
                prompt_kind=prompt_kind,
                prefill_ms=10.0 if kind == "static" else 10.9,
                decode_ms=100.0 if kind == "static" else 104.0,
                serialized_plan_bytes=(
                    1_000 if kind == "static" else 1_040
                ),
                resident_weight_bytes=(
                    2_000 if kind == "static" else 2_080
                ),
                qualification_context=qualification_context,
            )
            paths[f"{kind}_{prompt_kind}"] = path
    return paths


def _qualify(paths: dict[str, Path]) -> dict:
    return perf.qualify(
        static_short=paths["static_short"],
        dynamic_short=paths["dynamic_short"],
        static_medium=paths["static_medium"],
        dynamic_medium=paths["dynamic_medium"],
        static_bundle=paths["static_bundle"],
        dynamic_bundle=paths["dynamic_bundle"],
    )


def _edit(path: Path, callback) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    callback(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _refresh_evidence_hash(payload: dict, field: str) -> None:
    payload["qualification_provenance"][f"{field}_sha256"] = (
        perf._canonical_sha(payload[field])
    )


def _sync_runtime_receipt_evidence(payload: dict) -> None:
    receipt = json.loads(
        json.dumps(payload["runtime_memory_receipt"])
    )
    payload["qualification_evidence"]["runtime_memory_receipt"] = receipt
    _refresh_evidence_hash(payload, "runtime_memory_receipt")


def test_passes_all_performance_packaging_and_provenance_gates(
    tmp_path: Path,
) -> None:
    report = _qualify(_evidence(tmp_path))

    assert report["status"] == "passed"
    assert report["gates"]["performance"]["short"][
        "decode_throughput_gte_95_percent_static"
    ]
    assert report["gates"]["performance"]["medium"][
        "prefill_proxy_regression_lte_10_percent"
    ]
    assert report["gates"]["packaging"][
        "bundle_bytes_lte_105_percent_static"
    ]
    assert report["gates"]["runtime_evidence_consistency"][
        "dynamic_runtime_attention_plans_present"
    ]
    assert report["gates"]["provenance"]["one_tokenizer_contract"]
    assert all(
        report["gates"]["provenance"][
            "all_source_boundaries_exact_head"
        ].values()
    )
    assert report["gates"]["performance"]["short"][
        "same_fixed_length_structural_workload"
    ]
    assert report["diagnostics"]["token_stream_equivalence"]["short"][
        "exact_generated_token_ids_match"
    ]
    assert not report["diagnostics"]["runtime_attention_plan_scope"][
        "per_invocation_H_A_profile_plan_proved"
    ]


def test_independent_sealed_contract_replay_rejects_plan_tamper(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "dynamic.trtfb"
    payload = bytearray(_dynamic_bundle_bytes())
    header_length = struct.unpack_from("<Q", payload, 8)[0]
    engine_plan_offset = 16 + header_length
    payload[engine_plan_offset] ^= 1
    bundle.write_bytes(payload)

    with pytest.raises(
        perf.QualificationError,
        match="independently replay the sealed v2 bundle contract",
    ):
        perf._replay_sealed_runtime_memory_contract(bundle)


def test_independent_sealed_contract_replay_rejects_runtime_config_tamper(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "dynamic.trtfb"
    payload = bytearray(_dynamic_bundle_bytes())
    payload[-1] ^= 1
    bundle.write_bytes(payload)

    with pytest.raises(
        perf.QualificationError,
        match="independently replay the sealed v2 bundle contract",
    ):
        perf._replay_sealed_runtime_memory_contract(bundle)


def test_reopens_every_mapped_dso_and_rejects_unrelated_path_swap(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)
    payload = json.loads(
        paths["static_short"].read_text(encoding="utf-8")
    )
    unrelated = next(
        Path(row["path"])
        for row in payload["mapped_dso_identities"]
        if Path(row["path"]).name == "libunrelated.so"
    )
    unrelated.write_bytes(b"path-swapped-after-capture")

    with pytest.raises(
        perf.QualificationError,
        match=(
            "mapped_dso_identities.*does not match the captured "
            "binary identity"
        ),
    ):
        _qualify(paths)


@pytest.mark.parametrize(
    ("source_drift", "expected_returncode"),
    [(False, 0), (True, 1)],
)
def test_main_samples_standalone_source_state_before_and_after(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_drift: bool,
    expected_returncode: int,
) -> None:
    paths = _evidence(tmp_path)
    output = tmp_path / "final-report.json"
    pre = {
        "git_head": GIT_HEAD,
        "source_state_sha256": SOURCE_SHA,
        "git_dirty": False,
        "exact_head_gate_satisfied": True,
    }
    post = {
        **pre,
        "source_state_sha256": (
            "f" * 64 if source_drift else SOURCE_SHA
        ),
    }
    snapshots = iter((pre, post))
    labels: list[str] = []

    def snapshot(
        _repo_root: Path,
        _artifact_dir: Path,
        *,
        label: str,
    ) -> dict:
        labels.append(label)
        return next(snapshots)

    monkeypatch.setattr(perf, "_source_state_snapshot", snapshot)
    returncode = perf.main(
        [
            "--static-short",
            str(paths["static_short"]),
            "--dynamic-short",
            str(paths["dynamic_short"]),
            "--static-medium",
            str(paths["static_medium"]),
            "--dynamic-medium",
            str(paths["dynamic_medium"]),
            "--static-bundle",
            str(paths["static_bundle"]),
            "--dynamic-bundle",
            str(paths["dynamic_bundle"]),
            "--repo-root",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert labels == ["final-perf-pre", "final-perf-post"]
    assert returncode == expected_returncode
    assert report["gates"]["standalone_source_state"]["passed"] is (
        not source_drift
    )


@pytest.mark.parametrize(
    "field",
    [
        "source_state_pre_sha256",
        "source_state_post_sha256",
        "source_state_unchanged",
        "source_state_boundaries",
    ],
)
def test_requires_complete_source_boundary_provenance(
    tmp_path: Path, field: str
) -> None:
    paths = _evidence(tmp_path)
    _edit(
        paths["dynamic_short"],
        lambda payload: payload["qualification_provenance"].pop(field),
    )

    with pytest.raises(
        perf.QualificationError,
        match=(
            "missing field: "
            f"dynamic-short.qualification_provenance.{field}"
        ),
    ):
        _qualify(paths)


@pytest.mark.parametrize(
    "field",
    ["git_dirty", "exact_head_gate_satisfied"],
)
def test_requires_each_source_boundary_gate_field(
    tmp_path: Path, field: str
) -> None:
    paths = _evidence(tmp_path)

    def remove(payload: dict) -> None:
        payload["qualification_provenance"]["source_state_boundaries"][
            "benchmark_post"
        ].pop(field)

    _edit(paths["dynamic_short"], remove)
    with pytest.raises(
        perf.QualificationError,
        match=rf"benchmark_post fields must match capture schema:.*{field}",
    ):
        _qualify(paths)


@pytest.mark.parametrize(
    ("field", "value", "gate"),
    [
        ("git_dirty", True, "all_source_boundaries_clean"),
        (
            "exact_head_gate_satisfied",
            False,
            "all_source_boundaries_exact_head",
        ),
    ],
)
def test_dirty_or_non_exact_capture_cannot_pass_qualification(
    tmp_path: Path,
    field: str,
    value: bool,
    gate: str,
) -> None:
    paths = _evidence(tmp_path)

    def tamper(payload: dict) -> None:
        payload["qualification_provenance"]["source_state_boundaries"][
            "benchmark_post"
        ][field] = value

    _edit(paths["dynamic_short"], tamper)
    report = _qualify(paths)

    assert report["status"] == "failed"
    assert not report["gates"]["provenance"][gate]["dynamic-short"]


@pytest.mark.parametrize(
    ("field", "value", "gate"),
    [
        (
            "git_head",
            "f" * 40,
            "all_source_boundaries_match_expected_commit",
        ),
        (
            "source_state_sha256",
            "e" * 64,
            "all_source_boundaries_match_source_state",
        ),
    ],
)
def test_tampered_source_boundary_identity_cannot_pass(
    tmp_path: Path,
    field: str,
    value: str,
    gate: str,
) -> None:
    paths = _evidence(tmp_path)

    def tamper(payload: dict) -> None:
        payload["qualification_provenance"]["source_state_boundaries"][
            "build_pre"
        ][field] = value

    _edit(paths["dynamic_short"], tamper)
    report = _qualify(paths)

    assert report["status"] == "failed"
    assert not report["gates"]["provenance"][gate]["dynamic-short"]


def test_declared_source_change_cannot_pass(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)
    _edit(
        paths["dynamic_short"],
        lambda payload: payload["qualification_provenance"].__setitem__(
            "source_state_unchanged", False
        ),
    )

    report = _qualify(paths)

    assert report["status"] == "failed"
    assert not report["gates"]["provenance"][
        "source_stable_prebuild_to_postbuild"
    ]["dynamic-short"]


@pytest.mark.parametrize(
    ("field", "error"),
    [
        (
            "runtime_attention_plans",
            "missing field: dynamic-short.runtime_attention_plans",
        ),
        ("runtime_stack", "missing field: dynamic-short.runtime_stack"),
        (
            "runtime_libraries",
            "missing field: dynamic-short.runtime_libraries",
        ),
        (
            "build_runtime_kv_plugin",
            "missing field: dynamic-short.build_runtime_kv_plugin",
        ),
        ("cuda_jit_cache", "missing field: dynamic-short.cuda_jit_cache"),
        (
            "generation_workload",
            "missing field: dynamic-short.generation_workload",
        ),
        (
            "tokenizer_contract",
            "missing field: dynamic-short.tokenizer_contract",
        ),
    ],
)
def test_requires_capture_produced_runtime_evidence(
    tmp_path: Path, field: str, error: str
) -> None:
    paths = _evidence(tmp_path)
    _edit(paths["dynamic_short"], lambda payload: payload.pop(field))

    with pytest.raises(perf.QualificationError, match=error):
        _qualify(paths)


@pytest.mark.parametrize(
    "field",
    [
        "runtime_attention_plans_sha256",
        "runtime_stack_sha256",
        "runtime_libraries_sha256",
        "build_runtime_kv_plugin_sha256",
        "cuda_jit_cache_sha256",
        "generation_workload_sha256",
        "tokenizer_contract_sha256",
    ],
)
def test_requires_runtime_evidence_provenance_hashes(
    tmp_path: Path, field: str
) -> None:
    paths = _evidence(tmp_path)
    _edit(
        paths["dynamic_short"],
        lambda payload: payload["qualification_provenance"].pop(field),
    )

    with pytest.raises(
        perf.QualificationError,
        match=(
            "missing field: "
            f"dynamic-short.qualification_provenance.{field}"
        ),
    ):
        _qualify(paths)


def test_rejects_runtime_evidence_hash_mismatch(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)
    _edit(
        paths["dynamic_short"],
        lambda payload: payload["runtime_attention_plans"][0].__setitem__(
            "T", 1024
        ),
    )

    with pytest.raises(
        perf.QualificationError,
        match="runtime_attention_plans_sha256 does not match captured evidence",
    ):
        _qualify(paths)


@pytest.mark.parametrize(
    ("case", "field", "replacement", "error"),
    [
        (
            "dynamic_short",
            "runtime_attention_plans",
            [],
            "must contain at least one dynamic LSE plan",
        ),
        (
            "static_short",
            "runtime_attention_plans",
            [RUNTIME_PLAN],
            r"must be \[\] for a static baseline",
        ),
        (
            "static_short",
            "runtime_stack",
            RUNTIME_STACK,
            "must be null for a static baseline",
        ),
        (
            "static_short",
            "runtime_libraries",
            {},
            "must be null for a static baseline",
        ),
        (
            "static_short",
            "build_runtime_kv_plugin",
            {},
            "must be null for the static baseline",
        ),
    ],
)
def test_enforces_dynamic_and_static_runtime_evidence_roles(
    tmp_path: Path,
    case: str,
    field: str,
    replacement: object,
    error: str,
) -> None:
    paths = _evidence(tmp_path)

    def replace(payload: dict) -> None:
        payload[field] = replacement
        _refresh_evidence_hash(payload, field)

    _edit(paths[case], replace)
    with pytest.raises(perf.QualificationError, match=error):
        _qualify(paths)


def test_requires_lse_plan_and_cudnn_stack_coherence(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)

    def replace_stats(payload: dict) -> None:
        payload["runtime_attention_plans"][0]["stats"] = "max_sum_exp"
        _refresh_evidence_hash(payload, "runtime_attention_plans")

    _edit(paths["dynamic_short"], replace_stats)
    with pytest.raises(
        perf.QualificationError,
        match=r"runtime_attention_plans\[0\]\.stats must be 'lse'",
    ):
        _qualify(paths)

    paths = _evidence(tmp_path / "cudnn")

    def disagree(payload: dict) -> None:
        payload["runtime_attention_plans"][0]["cudnn_version"] = 91900
        _refresh_evidence_hash(payload, "runtime_attention_plans")

    _edit(paths["dynamic_short"], disagree)
    with pytest.raises(
        perf.QualificationError,
        match="cudnn_version disagrees with runtime_stack.cudnn_backend",
    ):
        _qualify(paths)


def test_requires_complete_live_runtime_stack(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)

    def remove_nvrtc(payload: dict) -> None:
        payload["runtime_stack"].pop("nvrtc")
        _refresh_evidence_hash(payload, "runtime_stack")

    _edit(paths["dynamic_short"], remove_nvrtc)
    with pytest.raises(
        perf.QualificationError,
        match=r"runtime_stack fields must match capture schema:.*nvrtc",
    ):
        _qualify(paths)


def test_requires_runtime_library_paths_and_hashes_to_match_files(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)

    def corrupt_file_hash(payload: dict) -> None:
        payload["runtime_libraries"]["nvrtc"]["sha256"] = "0" * 64
        _refresh_evidence_hash(payload, "runtime_libraries")

    _edit(paths["dynamic_short"], corrupt_file_hash)
    with pytest.raises(
        perf.QualificationError,
        match="runtime_libraries.nvrtc does not match the captured file",
    ):
        _qualify(paths)


def test_requires_build_plugin_path_size_and_hash_to_match_file(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)
    payload = json.loads(
        paths["dynamic_short"].read_text(encoding="utf-8")
    )
    plugin = Path(payload["build_runtime_kv_plugin"]["path"])
    plugin.write_bytes(plugin.read_bytes() + b"tampered")

    with pytest.raises(
        perf.QualificationError,
        match=(
            "runtime_kv_plugin does not match the captured binary identity|"
            "artifact identity changed"
        ),
    ):
        _qualify(paths)


def test_reopens_and_rejects_tampered_build_receipt(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)
    payload = json.loads(
        paths["dynamic_short"].read_text(encoding="utf-8")
    )
    receipt = Path(
        payload["qualification_evidence"]["build_receipt"]
    )
    receipt.write_text(receipt.read_text() + "\n", encoding="utf-8")

    with pytest.raises(
        perf.QualificationError,
        match="build_receipt_sha256 no longer matches",
    ):
        _qualify(paths)


def test_replayed_v2_build_receipt_rejects_extra_field(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)
    result_path = paths["dynamic_short"]
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    receipt = Path(payload["qualification_evidence"]["build_receipt"])
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_payload["legacy_override"] = True
    receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
    payload["qualification_evidence"]["build_receipt_sha256"] = _sha256(
        receipt
    )
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        perf.QualificationError,
        match="exact v2 field set",
    ):
        _qualify(paths)


def test_build_receipt_bundle_path_cannot_be_substituted_by_equal_copy(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)
    copied = tmp_path / "copied-dynamic.trtfb"
    copied.write_bytes(paths["dynamic_bundle"].read_bytes())

    with pytest.raises(
        perf.QualificationError,
        match="bundle path does not match",
    ):
        perf._read_case(
            paths["dynamic_short"],
            "dynamic-short",
            "native-dynamic",
            copied,
        )


def test_runtime_model_dso_identity_is_reopened(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)
    payload = json.loads(
        paths["static_short"].read_text(encoding="utf-8")
    )
    model_dso = Path(
        payload["runtime_trtmc_libraries"]["model"]["path"]
    )
    model_dso.write_bytes(model_dso.read_bytes() + b"stale")

    with pytest.raises(
        perf.QualificationError,
        match="model does not match the captured binary identity",
    ):
        _qualify(paths)


def test_requires_coherent_cuda_jit_cache_evidence(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)

    def claim_warm(payload: dict) -> None:
        payload["cuda_jit_cache"]["initial_state"] = "warm"
        _refresh_evidence_hash(payload, "cuda_jit_cache")

    _edit(paths["dynamic_short"], claim_warm)
    with pytest.raises(
        perf.QualificationError,
        match="cuda_jit_cache.initial_state must be 'cold'",
    ):
        _qualify(paths)


def test_fails_when_dynamic_live_stacks_differ_across_prompts(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)

    def change_driver(payload: dict) -> None:
        payload["runtime_stack"]["driver"] = "580.105.09"
        _refresh_evidence_hash(payload, "runtime_stack")

    _edit(paths["dynamic_medium"], change_driver)
    with pytest.raises(
        perf.QualificationError,
        match="live runtime stack does not match.*bundle calibration",
    ):
        _qualify(paths)


def test_fails_when_dynamic_runtime_libraries_differ_across_prompts(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)
    alternate = _runtime_libraries(tmp_path / "alternate")

    def change_libraries(payload: dict) -> None:
        payload["runtime_libraries"] = alternate
        _refresh_evidence_hash(payload, "runtime_libraries")

    _edit(paths["dynamic_medium"], change_libraries)
    with pytest.raises(
        perf.QualificationError,
        match="mapped_dso_identities does not bind the live nvrtc DSO",
    ):
        _qualify(paths)


def test_generated_token_divergence_is_explicit_but_not_a_false_failure(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)

    def diverge(payload: dict) -> None:
        workload = payload["generation_workload"]
        for stream in workload["measured_generated_token_ids"]:
            stream[5] = 999
        workload["measured_generated_token_ids_sha256"] = (
            perf._canonical_sha(workload["measured_generated_token_ids"])
        )
        _refresh_evidence_hash(payload, "generation_workload")

    _edit(paths["dynamic_medium"], diverge)
    report = _qualify(paths)

    assert report["status"] == "passed"
    diagnostic = report["diagnostics"]["token_stream_equivalence"]["medium"]
    assert diagnostic["exact_generated_token_ids_match"] is False
    assert diagnostic["common_prefix_tokens"] == 5
    assert "diagnostic_only" in diagnostic["gate_effect"]


def test_structural_workload_mismatch_is_a_hard_failure(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)

    def change_prompt(payload: dict) -> None:
        workload = payload["generation_workload"]
        workload["structural_identity"]["prompt_sha256"] = "b" * 64
        workload["structural_identity_sha256"] = perf._canonical_sha(
            workload["structural_identity"]
        )
        _refresh_evidence_hash(payload, "generation_workload")

    _edit(paths["dynamic_medium"], change_prompt)
    report = _qualify(paths)

    assert report["status"] == "failed"
    assert not report["gates"]["performance"]["medium"][
        "same_fixed_length_structural_workload"
    ]


def test_rejects_non_repeatable_token_stream_inside_one_case(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)

    def change_one_iteration(payload: dict) -> None:
        workload = payload["generation_workload"]
        workload["measured_generated_token_ids"][1][0] = 999
        workload["measured_generated_token_ids_sha256"] = (
            perf._canonical_sha(workload["measured_generated_token_ids"])
        )
        workload["token_stream_repeatable_within_case"] = False
        _refresh_evidence_hash(payload, "generation_workload")

    _edit(paths["dynamic_short"], change_one_iteration)
    with pytest.raises(
        perf.QualificationError,
        match="does not prove a repeatable greedy token stream",
    ):
        _qualify(paths)


def test_fails_when_static_and_dynamic_tokenizer_contracts_differ(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)

    for prompt_kind in ("short", "medium"):
        def change_tokenizer(payload: dict) -> None:
            payload["tokenizer_contract"][
                "tokenizer_json_sha256"
            ] = "c" * 64
            _refresh_evidence_hash(payload, "tokenizer_contract")

        _edit(paths[f"dynamic_{prompt_kind}"], change_tokenizer)

    report = _qualify(paths)
    assert report["status"] == "failed"
    assert not report["gates"]["provenance"]["one_tokenizer_contract"]


@pytest.mark.parametrize(
    ("field", "value", "gate"),
    [
        ("decode_ms", 106.0, "decode_throughput_gte_95_percent_static"),
        ("prefill_ms", 11.1, "prefill_proxy_regression_lte_10_percent"),
    ],
)
def test_fails_short_or_medium_performance_threshold(
    tmp_path: Path,
    field: str,
    value: float,
    gate: str,
) -> None:
    paths = _evidence(tmp_path)

    def regress(payload: dict) -> None:
        for observation in payload["observations"]:
            observation[field] = value

    _edit(paths["dynamic_medium"], regress)
    report = _qualify(paths)

    assert report["status"] == "failed"
    assert not report["gates"]["performance"]["medium"][gate]


@pytest.mark.parametrize(
    ("receipt_field", "value", "gate"),
    [
        (
            "serialized_plan_bytes",
            1_051,
            "serialized_plan_bytes_lte_105_percent_static",
        ),
        (
            "resident_weight_bytes",
            2_101,
            "resident_weight_bytes_lte_105_percent_static",
        ),
    ],
)
def test_fails_mem13_size_thresholds(
    tmp_path: Path,
    receipt_field: str,
    value: int,
    gate: str,
) -> None:
    paths = _evidence(tmp_path)
    for prompt_kind in ("short", "medium"):
        def change_receipt(payload: dict) -> None:
            payload["runtime_memory_receipt"][receipt_field] = value
            _sync_runtime_receipt_evidence(payload)

        _edit(
            paths[f"dynamic_{prompt_kind}"],
            change_receipt,
        )

    report = _qualify(paths)

    assert report["status"] == "failed"
    assert not report["gates"]["packaging"][gate]


def test_fails_weight_copy_and_streaming_gates(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)
    for prompt_kind in ("short", "medium"):
        def break_receipt(payload: dict) -> None:
            payload["runtime_memory_receipt"][
                "resident_weight_copy_count"
            ] = 3
            payload["runtime_memory_receipt"]["weight_streaming_active"] = True
            _sync_runtime_receipt_evidence(payload)

        _edit(paths[f"dynamic_{prompt_kind}"], break_receipt)

    report = _qualify(paths)

    assert report["status"] == "failed"
    packaging = report["gates"]["packaging"]
    assert not packaging["resident_weight_copy_count_lte_2"]["dynamic"]
    assert not packaging["weight_streaming_disabled"]["dynamic"]


def test_rejects_bundle_changed_after_fresh_build_receipt(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)
    paths["dynamic_bundle"].write_bytes(b"d" * 1_051)

    with pytest.raises(
        perf.QualificationError,
        match="build_receipt replay failed: bundle SHA",
    ):
        _qualify(paths)


def test_fails_closed_on_old_worker_result_with_named_missing_field(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)
    _edit(paths["dynamic_short"], lambda payload: payload.pop(
        "qualification_provenance"
    ))
    output = tmp_path / "report.json"

    returncode = perf.main(
        [
            "--static-short",
            str(paths["static_short"]),
            "--dynamic-short",
            str(paths["dynamic_short"]),
            "--static-medium",
            str(paths["static_medium"]),
            "--dynamic-medium",
            str(paths["dynamic_medium"]),
            "--static-bundle",
            str(paths["static_bundle"]),
            "--dynamic-bundle",
            str(paths["dynamic_bundle"]),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert returncode == 1
    assert report["status"] == "failed"
    assert report["errors"] == [
        "missing field: dynamic-short.qualification_provenance"
    ]


def test_fails_source_request_bundle_and_freshness_provenance(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)

    def corrupt(payload: dict) -> None:
        provenance = payload["qualification_provenance"]
        provenance["source_state_sha256"] = "a" * 64
        provenance["request_sha256"] = "b" * 64
        provenance["bundle_sha256"] = "c" * 64
        provenance["fresh_build"] = False
        provenance["artifact_reused"] = True

    _edit(paths["dynamic_short"], corrupt)
    report = _qualify(paths)

    assert report["status"] == "failed"
    gates = report["gates"]["provenance"]
    assert not gates["shared_fields_match"]["source_state_sha256"]
    assert not gates["source_stable_prebuild_to_postbuild"]["dynamic-short"]
    assert not gates["dynamic_bundle_sha_matches_file"]
    assert not gates["short_request_sha_matches"]
    assert not gates["all_bundles_declared_fresh"]
    assert not gates["no_reused_artifacts"]


def test_fails_closed_on_unavailable_or_unproven_memory_measurement(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)
    _edit(
        paths["dynamic_short"],
        lambda payload: payload["runtime_memory_receipt"].__setitem__(
            "resident_weight_bytes", None
        ),
    )
    with pytest.raises(
        perf.QualificationError,
        match=(
            "dynamic-short.runtime_memory_receipt.resident_weight_bytes "
            "must be a positive integer"
        ),
    ):
        _qualify(paths)

    paths = _evidence(tmp_path / "second")
    _edit(
        paths["dynamic_short"],
        lambda payload: payload["runtime_memory_receipt"][
            "measurement_sources"
        ].pop("resident_weight_bytes"),
    )
    with pytest.raises(
        perf.QualificationError,
        match="missing field: .*measurement_sources.resident_weight_bytes",
    ):
        _qualify(paths)
