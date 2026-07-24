# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from tests.tools.dynamic_memory_manifest_fixture import (
    seed_manifest_test_modules,
)
from tests.tools.test_qualify_native_dynamic_memory import (
    _write_base_build_receipt,
    _write_exact_build_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT
    / "tools"
    / "capture_native_dynamic_memory_process_isolation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "capture_native_dynamic_memory_process_isolation", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
isolation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = isolation
SPEC.loader.exec_module(isolation)

pytestmark = [pytest.mark.unit, pytest.mark.dynamic_memory]
GPU_A = {
    "index": "0",
    "uuid": "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    "pci_bus_id": "00000000:01:00.0",
    "name": "GPU A",
}
GPU_B = {
    "index": "1",
    "uuid": "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    "pci_bus_id": "00000000:02:00.0",
    "name": "GPU B",
}
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "a" * 40
RUNTIME_STACK = {
    "schema": 1,
    "sm": "sm103",
    "tensorrt": "11.2.0.113",
    "cuda_runtime": "13.3",
    "cudnn_backend": "9.20.0",
    "cudnn_frontend_revision": "7b9b711c22b6823e87150213ecd8449260db8610",
    "nvrtc": "13.3",
    "driver": "580.105.08",
}


def _init_repo(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=path,
        check=True,
    )
    (path / "tracked.txt").write_text("stable\n", encoding="utf-8")
    capture_tool = (
        path / "tools" / "capture_native_dynamic_memory_perf.py"
    )
    capture_tool.parent.mkdir(parents=True)
    capture_tool.write_text(_fake_capture_source(), encoding="utf-8")
    capture_tool.with_name(
        "capture_dynamic_memory_test_manifest.py"
    ).write_text(
        "PINNED_BENCHMARK_SIBLING_MARKER = 'canonical-sibling'\n",
        encoding="utf-8",
    )
    seed_manifest_test_modules(path)
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=path, check=True)


def _inventory() -> dict:
    command = ["fake-nvidia-smi"]
    return {
        "command": command,
        "command_sha256": isolation._canonical_sha(command),
        "started_ns": 1,
        "finished_ns": 2,
        "returncode": 0,
        "stdout": "",
        "stdout_sha256": isolation._canonical_sha(""),
        "stderr": "",
        "stderr_sha256": isolation._canonical_sha(""),
        "gpus": [dict(GPU_A), dict(GPU_B)],
    }


def _fake_capture_source() -> str:
    return r'''#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import sys
import time

def canonical(value):
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")).hexdigest()

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def identity(path):
    path = path.resolve(strict=True)
    metadata = path.stat()
    return {
        "path": str(path),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "size_bytes": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
        "sha256": sha256(path),
    }

def argument(name):
    return sys.argv[sys.argv.index(name) + 1]

repo = Path(argument("--repo-root"))
output = Path(argument("--output"))
worker_stderr = Path(argument("--stderr-output"))
bundle = Path(argument("--bundle"))
request_path = Path(argument("--request")).resolve()
build_path = Path(argument("--build-receipt")).resolve()
worker_path = Path(argument("--worker")).resolve()
plugin_path = Path(argument("--plugin-library")).resolve()
request = json.loads(request_path.read_text(encoding="utf-8"))
build = json.loads(build_path.read_text(encoding="utf-8"))
label = output.parent.name
behavior = request.get("test_behavior", "")
cache_path = Path(os.environ["CUDA_CACHE_PATH"])
manifest_path = Path(__file__).with_name(
    "capture_dynamic_memory_test_manifest.py"
)
sibling_namespace = {}
exec(
    compile(manifest_path.read_bytes(), str(manifest_path), "exec"),
    sibling_namespace,
)
if (
    sibling_namespace["PINNED_BENCHMARK_SIBLING_MARKER"]
    != "canonical-sibling"
):
    raise RuntimeError("wrong sibling benchmark module")

def snapshot(path):
    captured = time.time_ns()
    files = sorted(item for item in path.rglob("*") if item.is_file()) if path.exists() else []
    metadata = [
        {"path": str(item.relative_to(path)), "bytes": item.read_bytes().hex()}
        for item in files
    ]
    return {
        "captured_at_ns": captured,
        "exists": path.exists(),
        "is_directory": path.is_dir(),
        "entry_count": len(files),
        "file_count": len(files),
        "total_bytes": sum(item.stat().st_size for item in files),
        "metadata_sha256": canonical(metadata),
    }

before = snapshot(cache_path)
initial_state = "warm" if before["file_count"] else "cold"
if behavior == "break_warm_continuity" and label == "gpu-a-warm":
    before["metadata_sha256"] = "9" * 64
cache_path.mkdir(parents=True, exist_ok=True)
jit = cache_path / "jit.bin"
if not jit.exists():
    jit.write_bytes(b"compiled")

worker_started = time.time_ns()
if behavior == "no_worker_overlap" and label == "gpu-a-concurrent":
    worker_started, worker_finished = 100, 200
elif behavior == "no_worker_overlap" and label == "gpu-b-concurrent":
    worker_started, worker_finished = 200, 300
else:
    time.sleep(0.12 if "concurrent" in label else 0.01)
    worker_finished = time.time_ns()
if behavior == "no_engine_load_overlap" and label == "gpu-a-concurrent":
    load_started, load_finished = 100, 200
elif behavior == "no_engine_load_overlap" and label == "gpu-b-concurrent":
    load_started, load_finished = 200, 300
elif behavior == "load_outside_worker" and label == "gpu-a-concurrent":
    load_started = worker_started - 1
    load_finished = worker_finished - 1
else:
    load_started = worker_started + 1
    load_finished = worker_finished - 1
after = snapshot(cache_path)

if behavior == "mutate_source" and label == "gpu-a-warm":
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")

visible = os.environ["CUDA_VISIBLE_DEVICES"]
if visible.startswith("GPU-aaaaaaaa"):
    uuid = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    pci = "00000000:01:00.0"
else:
    uuid = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    pci = "00000000:02:00.0"
token_ids = [11, 22, 33]
if behavior == "token_mismatch" and label == "gpu-b-concurrent":
    token_ids = [11, 22, 44]

runtime_stack = request["test_runtime_stack"]
runtime_libraries = request["test_runtime_libraries"]
runtime_trtmc = request["test_runtime_trtmc_libraries"]
if behavior == "runtime_stack_mismatch" and label == "gpu-b-concurrent":
    runtime_stack = dict(runtime_stack)
    runtime_stack["driver"] = "580.105.09"
if behavior == "runtime_library_mismatch" and label == "gpu-b-concurrent":
    runtime_libraries = json.loads(json.dumps(runtime_libraries))
    runtime_libraries["nvrtc"]["sha256"] = "8" * 64
cache = {
    "path": str(cache_path.resolve()),
    "path_source": "CUDA_CACHE_PATH",
    "cuda_cache_path_env": str(cache_path.resolve()),
    "cuda_cache_disable_env": os.environ["CUDA_CACHE_DISABLE"],
    "enabled": True,
    "initial_state": initial_state,
    "worker_started_ns": worker_started,
    "worker_finished_ns": worker_finished,
    "before": before,
    "after": after,
}
source_state = build["source_state_pre"]
source_sha = source_state["source_state_sha256"]
source_unchanged = not (
    behavior == "child_source_changed" and label == "gpu-a-warm"
)
output_summary = {
    "text": "deterministic",
    "text_truncated": False,
    "token_ids": token_ids,
}
iterations = request["measurement"]["iterations"]
worker_identity = identity(worker_path)
plugin_identity = identity(plugin_path)
capture_tool = Path(__file__).resolve()
build_manifest = build["build_manifest"]
toolchain = {
    "worker": worker_identity,
    "plugin_library": plugin_identity,
    "runtime_trtmc_libraries": runtime_trtmc,
    "build_manifest": build_manifest,
    "capture_tool": str(capture_tool),
    "capture_tool_sha256": sha256(capture_tool),
}
environment = {
    "cuda_visible_devices": visible,
    "cuda_logical_device": 0,
    "cuda_device_uuid": uuid,
    "cuda_pci_bus_id": pci,
    "cuda_compute_capability": "sm103",
}
worker_stdout = output.parent / "worker.stdout.log"
worker_stdout.write_text("", encoding="utf-8")
worker_stderr.write_text("worker diagnostic\n", encoding="utf-8")
raw_output = output.parent / "worker.raw.json"
worker_command = [
    str(worker_path),
    "--request",
    str(request_path),
    "--output",
    str(raw_output),
]
payload = {
    "schema_version": "trtmc.benchmark-worker-result/v1",
    "status": "completed",
    "case_name": request["case_name"],
    "case_digest": request["case_digest"],
    "model_id": "Qwen/Qwen3-0.6B",
    "operation": request["operation"],
    "timing_scope": "public_pipeline_call_wall",
    "load_ms": 1.0,
    "load_started_ns": load_started,
    "load_finished_ns": load_finished,
    "warmup": request["measurement"]["warmup"],
    "iterations": iterations,
    "observations": [
        {
            "iteration": index,
            "prefill_ms": 1.0,
            "decode_ms": 2.0,
            "output_tokens": len(token_ids),
        }
        for index in range(iterations)
    ],
    "output_summary": output_summary,
    "runtime_attention_plans": [{
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
        "plan": "plan",
        "workspace_bytes": 0,
        "cudnn_version": 92000,
    }],
    "runtime_stack": runtime_stack,
    "runtime_libraries": runtime_libraries,
    "runtime_trtmc_libraries": runtime_trtmc,
    "build_runtime_kv_plugin": plugin_identity,
    "cuda_jit_cache": cache,
    "qualification_provenance": {
        "git_head": build["git_head"],
        "source_state_sha256": source_sha,
        "source_state_pre_sha256": source_sha,
        "source_state_post_sha256": source_sha,
        "source_state_unchanged": source_unchanged,
        "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "request_sha256": "3" * 64,
        "model_revision": build["model_revision"],
        "artifact_role": "native-dynamic",
        "toolchain_sha256": canonical(toolchain),
        "benchmark_environment_sha256": canonical(environment),
        "runtime_stack_sha256": canonical(runtime_stack),
        "runtime_libraries_sha256": canonical(runtime_libraries),
        "runtime_trtmc_libraries_sha256": canonical(runtime_trtmc),
        "build_runtime_kv_plugin_sha256": canonical(plugin_identity),
        "build_manifest_sha256": canonical(build_manifest),
        "cuda_jit_cache_sha256": canonical(cache),
    },
    "qualification_evidence": {
        "build_receipt": str(build_path),
        "build_receipt_sha256": sha256(build_path),
        "request_file": str(request_path),
        "request_file_sha256": sha256(request_path),
        "worker_command": worker_command,
        "worker_command_sha256": canonical(worker_command),
        "toolchain": toolchain,
        "build_manifest": build_manifest,
        "runtime_trtmc_libraries": runtime_trtmc,
        "build_runtime_kv_plugin": plugin_identity,
        "source_state_pre": source_state,
        "source_state_post": source_state,
        "source_state_unchanged": source_unchanged,
        "worker_stdout": str(worker_stdout.resolve()),
        "worker_stdout_sha256": sha256(worker_stdout),
        "worker_stderr": str(worker_stderr.resolve()),
        "worker_stderr_sha256": sha256(worker_stderr),
        "environment": environment,
    },
}
if label == "gpu-a-warm":
    if behavior == "missing_child_build_receipt":
        payload["qualification_evidence"].pop("build_receipt")
    elif behavior == "missing_child_runtime_trtmc":
        payload.pop("runtime_trtmc_libraries")
    elif behavior == "missing_child_build_plugin":
        payload.pop("build_runtime_kv_plugin")
    elif behavior == "swap_child_worker_plugin":
        child_toolchain = payload["qualification_evidence"]["toolchain"]
        child_toolchain["worker"], child_toolchain["plugin_library"] = (
            child_toolchain["plugin_library"],
            child_toolchain["worker"],
        )
        payload["qualification_provenance"]["toolchain_sha256"] = canonical(
            child_toolchain
        )
    elif behavior == "swap_capture_tool_inode":
        replacement = capture_tool.with_name(
            "capture_native_dynamic_memory_perf.py.replacement"
        )
        replacement.write_bytes(capture_tool.read_bytes())
        os.replace(replacement, capture_tool)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload), encoding="utf-8")
print(json.dumps({"status": "completed", "output": str(output)}))
print("capture diagnostic", file=sys.stderr)
'''


def _runtime_stack_line(stack: dict = RUNTIME_STACK) -> str:
    return "[trtmc.runtime_stack] " + " ".join(
        f"{field}={stack[field]}"
        for field in isolation._RUNTIME_STACK_FIELDS
    )


def _runtime_libraries(inputs: Path) -> dict:
    library_dir = inputs / "runtime-libraries"
    library_dir.mkdir()
    nvrtc = library_dir / "libnvrtc.so.13.3"
    builtins = library_dir / "libnvrtc-builtins.so.13.3"
    nvrtc.write_bytes(b"nvrtc")
    builtins.write_bytes(b"nvrtc-builtins")

    def identity(path: Path) -> dict:
        return {
            "path": str(path.resolve()),
            "basename": path.name,
            "sha256": isolation._sha256(path),
            "size_bytes": path.stat().st_size,
        }

    return {
        "directory": str(library_dir.resolve()),
        "live_nvrtc_version": RUNTIME_STACK["nvrtc"],
        "nvrtc": identity(nvrtc),
        "nvrtc_builtins": identity(builtins),
    }


def _write_test_bundle(path: Path, spec) -> None:
    plans = {
        "prefill_engine_plan": b"qualified-prefill-engine-plan",
        "engine_plan": b"qualified-decode-engine-plan",
    }
    section_offset = 0
    sections = {}
    for name, plan in plans.items():
        sections[name] = {
            "offset": section_offset,
            "size": len(plan),
        }
        section_offset += len(plan)
    header = {
        "model_id": MODEL_ID,
        "vocab_size": spec.vocab_size,
        "hidden_size": 64,
        "num_layers": spec.num_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_cache_length": spec.context_limit,
        "precision": "bf16",
        "runtime_memory": {
            "contract_version": 1,
            "qualified_model_id": MODEL_ID,
            "qualified_model_revision": MODEL_REVISION,
            "qualified_config_sha256": "b" * 64,
            "qualified_target": "linux-x86_64-gb300",
            "qualified_runtime_stack": {
                key: value
                for key, value in RUNTIME_STACK.items()
                if key != "schema"
            },
            "native_kv_plugin_abi": 1,
            "model_context_limit": spec.context_limit,
            "prefill_chunk_limit": spec.chunk_limit,
            "kv_layout": "contiguous",
            "kv_dtype": spec.kv_dtype,
            "kv_bytes_per_token": spec.kv_bytes_per_token,
            "active_kv_profile_limits": list(spec.buckets),
            "runtime_owned": True,
        },
        "sections": sections,
    }
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(
        isolation.boundary.BUNDLE_MAGIC
        + struct.pack("<Q", len(header_bytes))
        + header_bytes
        + b"".join(plans.values())
    )


def _qualified_engine_graph(
    evidence_dir: Path,
    *,
    bundle: Path,
    spec,
) -> dict:
    header = isolation.boundary._read_bundle_header(bundle)
    contract = header["runtime_memory"]
    num_layers = header["num_layers"]
    vocab_size = header["vocab_size"]
    kv_width = (
        contract["kv_bytes_per_token"]
        // (2 * num_layers * 2)
    )
    sections = {}
    for section_name in isolation._QUALIFIED_ENGINE_SECTIONS:
        prefill = section_name == "prefill_engine_plan"
        if prefill:
            token_profiles = [
                {
                    "min": [1],
                    "opt": [spec.chunk_limit],
                    "max": [spec.chunk_limit],
                }
            ]
            cache_profiles = [
                {
                    "min": [1, kv_width],
                    "opt": [spec.chunk_limit, kv_width],
                    "max": [spec.context_limit, kv_width],
                }
            ]
        else:
            token_profiles = [
                {"min": [1], "opt": [1], "max": [1]}
                for _ in spec.buckets
            ]
            cache_profiles = [
                {
                    "min": [1, kv_width],
                    "opt": [bucket, kv_width],
                    "max": [bucket, kv_width],
                }
                for bucket in spec.buckets
            ]
        history_profiles = [
            {"min": [1], "opt": [1], "max": [1]}
            for _ in token_profiles
        ]
        token_shape = [-1] if prefill else [1]
        inputs = {
            "token_id": {
                "shape": token_shape,
                "profiles": token_profiles,
            },
            "position_id": {
                "shape": token_shape,
                "profiles": token_profiles,
            },
            "history_length": {
                "shape": [1],
                "profiles": history_profiles,
            },
        }
        outputs = {"logits": {"shape": [1, vocab_size]}}
        for layer in range(num_layers):
            for value_name in ("k", "v"):
                inputs[f"cache_{value_name}_{layer}"] = {
                    "shape": [-1, kv_width],
                    "profiles": cache_profiles,
                }
                outputs[f"present_{value_name}_{layer}"] = {
                    "shape": [-1 if prefill else 1, kv_width]
                }
        inspector = (
            evidence_dir
            / f"{section_name}.engine-inspector.json"
        )
        inspector.write_text(
            json.dumps(
                [
                    {
                        "Name": (
                            f"layer.{layer}.attn."
                            "NativeContiguousAttentionV2"
                        ),
                        "LayerType": "PluginV2",
                    }
                    for layer in range(num_layers)
                ]
            ),
            encoding="utf-8",
        )
        plan = isolation.boundary._read_bundle_section(
            bundle,
            header,
            section_name,
        )
        sections[section_name] = {
            "engine_sha256": hashlib.sha256(plan).hexdigest(),
            "num_optimization_profiles": len(token_profiles),
            "inputs": inputs,
            "outputs": outputs,
            "native_contiguous_attention_layer_indices": list(
                range(num_layers)
            ),
            "dense_attention_layers": [],
            "cache_concat_layers": [],
            "inspector_path": str(inspector),
            "inspector_size_bytes": inspector.stat().st_size,
            "inspector_sha256": isolation._sha256(inspector),
        }
    return {
        "passed": True,
        "gates": {
            name: True
            for name in isolation._QUALIFIED_ENGINE_GRAPH_GATES
        },
        "runtime_stack": dict(contract["qualified_runtime_stack"]),
        "model_contract": {
            "model_context_limit": contract["model_context_limit"],
            "prefill_chunk_limit": contract["prefill_chunk_limit"],
            "active_kv_profile_limits": contract[
                "active_kv_profile_limits"
            ],
            "num_layers": num_layers,
            "vocab_size": vocab_size,
            "kv_dtype": contract["kv_dtype"],
            "kv_bytes_per_token": contract["kv_bytes_per_token"],
            "kv_width": kv_width,
        },
        "engine_sections": sections,
    }


def _write_correctness_report(
    inputs: Path,
    *,
    repo: Path,
    bundle: Path,
    source: dict,
) -> Path:
    evidence_dir = inputs / "correctness"
    evidence_dir.mkdir()
    spec = isolation.boundary.SPECS[MODEL_ID]
    provenance_root = inputs / "correctness-base-provenance"
    manifest, _ = _write_exact_build_manifest(
        provenance_root,
        source_state=source,
        repo_root=repo,
    )
    bundle_header = isolation.boundary._read_bundle_header(bundle)
    base_build_receipt = _write_base_build_receipt(
        provenance_root,
        manifest_path=manifest,
        bundle=bundle,
        header=bundle_header,
        source_state=source,
    )
    runner = (
        provenance_root
        / "build"
        / "trtmc_dynamic_memory_qualify"
    )
    base_artifact_binding = (
        isolation.boundary._validate_base_artifact_binding(
            build_manifest_path=manifest,
            base_build_receipt_path=base_build_receipt,
            bundle=bundle,
            runner=runner,
            spec=spec,
            source_state=source,
        )
    )
    selected_plugin = base_artifact_binding["runtime_kv_plugin"]
    runtime_kv_plugin_binding = {
        "schema_version": (
            isolation.boundary.RUNTIME_KV_PLUGIN_BINDING_SCHEMA
        ),
        "environment": isolation.boundary.RUNTIME_KV_PLUGIN_ENV,
        "environment_was_set": False,
        "preload_mapping": None,
        "selected": selected_plugin,
        "loaded_mapping": {
            "path": selected_plugin["path"],
            "device": selected_plugin["device"],
            "inode": selected_plugin["inode"],
            "deleted": False,
            "identity_sha256": isolation._canonical_sha(
                selected_plugin
            ),
        },
    }
    cases = []
    for case_spec in isolation.boundary._cases_for(spec):
        safe_name = case_spec.name.replace("/", "-")
        stderr = evidence_dir / f"{safe_name}.stderr.log"
        stderr.write_text(
            _runtime_stack_line() + "\n", encoding="utf-8"
        )
        case = {
            "name": case_spec.name,
            "prompt_tokens": case_spec.prompt_tokens,
            "decode_tokens": case_spec.decode_tokens,
            "passed": True,
            "runner_stderr": str(stderr),
        }
        if case_spec.expect_admission_rejection:
            case.update(
                {
                    "admission_rejected_before_attention": True,
                    "trace": {
                        "status": "rejected",
                        "stage": "before_attention",
                        "prefill_launches": 0,
                        "decode_launches": 0,
                    },
                }
            )
        else:
            trt = evidence_dir / f"{safe_name}.trt-logits.bin"
            hf = evidence_dir / f"{safe_name}.hf-logits.npy"
            trt.write_bytes(f"trt-{case_spec.name}".encode())
            hf.write_bytes(f"hf-{case_spec.name}".encode())
            case.update(
                {
                    "trace": {"status": "ok"},
                    "parity": {
                        "passed": True,
                        "composite_gates": {
                            "numerical": True,
                            "token_level": True,
                        },
                    },
                    "trt_logits_artifact": str(trt),
                    "trt_logits_sha256": isolation._sha256(trt),
                    "hf_logits_artifact": str(hf),
                    "hf_logits_sha256": isolation._sha256(hf),
                }
            )
        cases.append(case)

    report = {
        "schema_version": 1,
        "passed": True,
        "promotion_eligible": True,
        "model_id": MODEL_ID,
        "bundle": str(bundle),
        "bundle_sha256": isolation._sha256(bundle),
        "runner": str(runner),
        "base_artifact_binding": base_artifact_binding,
        "runtime_kv_plugin_binding": runtime_kv_plugin_binding,
        "qualification_gates": {
            "base_artifact_binding_passed": True,
            "runtime_kv_plugin_binding_passed": True,
        },
        "model_context_limit": spec.context_limit,
        "prefill_chunk_limit": spec.chunk_limit,
        "hf_reference": {
            "kind": "hf_cache_snapshot",
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "config_sha256": "b" * 64,
            "path": str(inputs / "hf-snapshot"),
        },
        "source_state": source,
        "source_state_post": source,
        "source_state_unchanged": True,
        "qualified_engine_graph": _qualified_engine_graph(
            evidence_dir,
            bundle=bundle,
            spec=spec,
        ),
        "cases": cases,
        "context_memory_envelope": {
            "schema_version": 1,
            "status": "passed",
            "passed": True,
            "coverage_required": True,
            "gates": {
                "all_points_within_o_c_times_a_envelope": True,
                "full_context_below_materialized_score_bound": True,
                "coverage": {
                    "has_prefill_and_decode": True,
                    "reaches_model_context_limit": True,
                    "has_at_least_three_active_lengths": True,
                },
            },
        },
    }
    path = evidence_dir / "qualification-report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _performance_provenance(
    *,
    source: dict,
    bundle: Path,
    role: str,
    runtime_stack: dict | None,
    runtime_libraries: dict | None,
    runtime_trtmc: dict | None = None,
    build_plugin: dict | None = None,
    toolchain: dict | None = None,
    environment: dict | None = None,
    build_manifest: dict | None = None,
) -> dict:
    provenance = {
        "git_head": source["git_head"],
        "source_state_sha256": source["source_state_sha256"],
        "source_state_pre_sha256": source["source_state_sha256"],
        "source_state_post_sha256": source["source_state_sha256"],
        "source_state_unchanged": True,
        "prebuild_source_state_sha256": source["source_state_sha256"],
        "postbuild_source_state_sha256": source["source_state_sha256"],
        "bundle_sha256": isolation._sha256(bundle),
        "request_sha256": "3" * 64,
        "model_revision": MODEL_REVISION,
        "precision": "bfloat16",
        "target": "linux-x86_64-gb300",
        "toolchain_sha256": "4" * 64,
        "benchmark_environment_sha256": "5" * 64,
        "bundle_build_id": f"{role}-build",
        "artifact_role": role,
        "fresh_build": True,
        "artifact_reused": False,
        "runtime_attention_plans_sha256": "6" * 64,
        "runtime_stack_sha256": isolation._canonical_sha(runtime_stack),
        "runtime_libraries_sha256": isolation._canonical_sha(
            runtime_libraries
        ),
        "cuda_jit_cache_sha256": "7" * 64,
    }
    if role == "native-dynamic":
        assert runtime_trtmc is not None
        assert build_plugin is not None
        assert toolchain is not None
        assert environment is not None
        assert build_manifest is not None
        provenance.update(
            {
                "toolchain_sha256": isolation._canonical_sha(
                    toolchain
                ),
                "benchmark_environment_sha256": (
                    isolation._canonical_sha(environment)
                ),
                "runtime_trtmc_libraries_sha256": (
                    isolation._canonical_sha(runtime_trtmc)
                ),
                "build_runtime_kv_plugin_sha256": (
                    isolation._canonical_sha(build_plugin)
                ),
                "build_manifest_sha256": isolation._canonical_sha(
                    build_manifest
                ),
            }
        )
    return provenance


def _write_performance_report(
    inputs: Path,
    *,
    dynamic_bundle: Path,
    source: dict,
    runtime_libraries: dict,
    runtime_trtmc: dict,
    build_receipt: Path,
    worker: Path,
    plugin: Path,
    capture_tool: Path,
    request: Path,
) -> Path:
    evidence_dir = inputs / "performance"
    evidence_dir.mkdir()
    static_bundle = evidence_dir / "static.trtfb"
    static_bundle.write_bytes(b"static-bundle")
    build_document = json.loads(
        build_receipt.read_text(encoding="utf-8")
    )
    worker_identity = isolation._file_identity(worker)
    plugin_identity = isolation._file_identity(plugin)
    toolchain = {
        "worker": worker_identity,
        "plugin_library": plugin_identity,
        "runtime_trtmc_libraries": runtime_trtmc,
        "build_manifest": build_document["build_manifest"],
        "capture_tool": str(capture_tool.resolve()),
        "capture_tool_sha256": isolation._sha256(capture_tool),
    }
    cases = {}
    for case_name in (
        "static_short",
        "dynamic_short",
        "static_medium",
        "dynamic_medium",
    ):
        dynamic = case_name.startswith("dynamic_")
        bundle = dynamic_bundle if dynamic else static_bundle
        role = "native-dynamic" if dynamic else "exact-head-static-split"
        stack = RUNTIME_STACK if dynamic else None
        libraries = runtime_libraries if dynamic else None
        environment = (
            {"fixture": "performance-dynamic", "case": case_name}
            if dynamic
            else None
        )
        provenance = _performance_provenance(
            source=source,
            bundle=bundle,
            role=role,
            runtime_stack=stack,
            runtime_libraries=libraries,
            runtime_trtmc=runtime_trtmc if dynamic else None,
            build_plugin=plugin_identity if dynamic else None,
            toolchain=toolchain if dynamic else None,
            environment=environment,
            build_manifest=(
                build_document["build_manifest"]
                if dynamic
                else None
            ),
        )
        capture = {
            "schema_version": isolation.CAPTURE_RESULT_SCHEMA,
            "status": "completed",
            "model_id": MODEL_ID,
            "runtime_stack": stack,
            "runtime_libraries": libraries,
            "qualification_provenance": provenance,
        }
        if dynamic:
            case_request = evidence_dir / f"{case_name}.request.json"
            case_request_document = json.loads(
                request.read_text(encoding="utf-8")
            )
            case_request_document["case_name"] = case_name
            case_request.write_text(
                json.dumps(case_request_document),
                encoding="utf-8",
            )
            worker_stdout = evidence_dir / f"{case_name}.worker.stdout"
            worker_stderr = evidence_dir / f"{case_name}.worker.stderr"
            worker_stdout.write_text("", encoding="utf-8")
            worker_stderr.write_text("diagnostic\n", encoding="utf-8")
            raw_output = evidence_dir / f"{case_name}.raw.json"
            worker_command = [
                str(worker.resolve()),
                "--request",
                str(case_request.resolve()),
                "--output",
                str(raw_output.resolve()),
            ]
            capture.update(
                {
                    "runtime_trtmc_libraries": runtime_trtmc,
                    "build_runtime_kv_plugin": plugin_identity,
                    "qualification_evidence": {
                        "build_receipt": str(build_receipt.resolve()),
                        "build_receipt_sha256": isolation._sha256(
                            build_receipt
                        ),
                        "request_file": str(case_request.resolve()),
                        "request_file_sha256": isolation._sha256(
                            case_request
                        ),
                        "worker_command": worker_command,
                        "worker_command_sha256": (
                            isolation._canonical_sha(worker_command)
                        ),
                        "toolchain": toolchain,
                        "environment": environment,
                        "build_manifest": build_document[
                            "build_manifest"
                        ],
                        "runtime_trtmc_libraries": runtime_trtmc,
                        "build_runtime_kv_plugin": plugin_identity,
                        "source_state_pre": source,
                        "source_state_post": source,
                        "source_state_unchanged": True,
                        "worker_stdout": str(worker_stdout.resolve()),
                        "worker_stdout_sha256": isolation._sha256(
                            worker_stdout
                        ),
                        "worker_stderr": str(worker_stderr.resolve()),
                        "worker_stderr_sha256": isolation._sha256(
                            worker_stderr
                        ),
                    },
                }
            )
        capture_path = evidence_dir / f"{case_name}.json"
        capture_path.write_text(json.dumps(capture), encoding="utf-8")
        cases[case_name] = {
            "path": str(capture_path),
            "result_sha256": isolation._sha256(capture_path),
            "model_id": MODEL_ID,
            "iterations": 2,
            "warmup": 1,
            "mean_prefill_ms": 105.0 if dynamic else 100.0,
            "decode_tokens_per_second": 100.0,
            "total_output_tokens": 6,
            "qualification_provenance": provenance,
            "runtime_stack": stack,
            "runtime_libraries": libraries,
        }
    prompt_gate = {
        "static_decode_tokens_per_second": 100.0,
        "dynamic_decode_tokens_per_second": 100.0,
        "decode_throughput_ratio": 1.0,
        "static_mean_prefill_ms": 100.0,
        "dynamic_mean_prefill_ms": 105.0,
        "prefill_ratio": 1.05,
        "same_iterations": True,
        "same_warmup": True,
        "same_output_token_counts": True,
        "decode_throughput_gte_95_percent_static": True,
        "prefill_proxy_regression_lte_10_percent": True,
    }
    report = {
        "schema_version": isolation.PERFORMANCE_REPORT_SCHEMA,
        "status": "passed",
        "bundles": {
            "static": {
                "path": str(static_bundle),
                "sha256": isolation._sha256(static_bundle),
                "bytes": static_bundle.stat().st_size,
            },
            "dynamic": {
                "path": str(dynamic_bundle),
                "sha256": isolation._sha256(dynamic_bundle),
                "bytes": dynamic_bundle.stat().st_size,
            },
        },
        "cases": cases,
        "gates": {
            "provenance": {"all": True},
            "receipt_consistency": {"all": True},
            "runtime_evidence_consistency": {"all": True},
            "performance": {
                "short": dict(prompt_gate),
                "medium": dict(prompt_gate),
            },
            "packaging": {"all": True},
        },
    }
    path = evidence_dir / "performance-report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    behavior: str = "",
) -> argparse.Namespace:
    repo = tmp_path / "repo"
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _init_repo(repo)
    bundle = inputs / "model.trtfb"
    _write_test_bundle(
        bundle,
        isolation.boundary.SPECS[MODEL_ID],
    )
    fake_capture = (
        repo / "tools" / "capture_native_dynamic_memory_perf.py"
    )
    request = inputs / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_name": "isolation",
                "case_digest": "isolation-digest",
                "bundle": str(bundle),
                "operation": "generate",
                "runtime": {
                    "max_sequence_length": 128,
                    "runtime_kv_policy_requested": True,
                },
                "measurement": {"warmup": 1, "iterations": 2},
                "request": {
                    "prompt": "hello",
                    "max_new_tokens": 3,
                    "temperature": 0.0,
                    "top_k": 1,
                    "top_p": 1.0,
                    "seed": 12345,
                },
                "test_behavior": behavior,
            }
        ),
        encoding="utf-8",
    )
    precompute = repo / "artifacts" / "precompute"
    source = isolation._source_state_snapshot(
        repo, precompute, label="test-precompute"
    )
    runtime_libraries = _runtime_libraries(inputs)
    correctness_report = _write_correctness_report(
        inputs,
        repo=repo,
        bundle=bundle,
        source=source,
    )
    correctness = json.loads(
        correctness_report.read_text(encoding="utf-8")
    )
    base = correctness["base_artifact_binding"]
    build_receipt = Path(base["base_build_receipt"]["path"])
    worker = Path(base["benchmark_worker"]["path"])
    plugin = Path(base["runtime_kv_plugin"]["path"])
    runtime_trtmc = {
        "model_id": MODEL_ID,
        "model_family": "qwen",
        "core": base["core"],
        "trt_backend": base["trt_backend"]["identity"],
        "runtime_kv_plugin": base["runtime_kv_plugin"],
        "model": base["model_plugin"]["identity"],
    }
    request_document = json.loads(request.read_text(encoding="utf-8"))
    request_document["test_runtime_stack"] = RUNTIME_STACK
    request_document["test_runtime_libraries"] = runtime_libraries
    request_document["test_runtime_trtmc_libraries"] = runtime_trtmc
    request.write_text(json.dumps(request_document), encoding="utf-8")
    performance_report = _write_performance_report(
        inputs,
        dynamic_bundle=bundle,
        source=source,
        runtime_libraries=runtime_libraries,
        runtime_trtmc=runtime_trtmc,
        build_receipt=build_receipt,
        worker=worker,
        plugin=plugin,
        capture_tool=fake_capture,
        request=request,
    )
    monkeypatch.setattr(
        isolation.performance,
        "qualify",
        lambda **_kwargs: json.loads(
            performance_report.read_text(encoding="utf-8")
        ),
    )
    monkeypatch.setattr(isolation, "_gpu_inventory", _inventory)
    return argparse.Namespace(
        repo_root=repo,
        output_dir=repo / "artifacts" / "isolation",
        python=Path(sys.executable),
        capture_tool=fake_capture,
        bundle=bundle,
        build_receipt=build_receipt,
        request=request,
        correctness_report=correctness_report,
        performance_report=performance_report,
        worker=worker,
        plugin_library=plugin,
        comparison_sequence_limit=128,
        gpu_a="0",
        gpu_b="1",
    )


def _rewrite_json(path: Path, update) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    update(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def _dynamic_capture_path(
    performance_report: Path, case: str = "dynamic_short"
) -> Path:
    report = json.loads(performance_report.read_text(encoding="utf-8"))
    return Path(report["cases"][case]["path"])


def test_pinned_capture_trampoline_preserves_benchmark_sibling_lookup(
    tmp_path: Path,
) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    capture_tool = tools / "capture_native_dynamic_memory_perf.py"
    sibling = tools / "capture_dynamic_memory_test_manifest.py"
    sibling.write_text("MARKER = 'canonical-sibling'\n", encoding="utf-8")
    capture_tool.write_text(
        """
import json
from pathlib import Path
import sys

namespace = {}
exec(
    Path(__file__).with_name(
        "capture_dynamic_memory_test_manifest.py"
    ).read_text(encoding="utf-8"),
    namespace,
)
output = Path(sys.argv[sys.argv.index("--output") + 1])
output.write_text(
    json.dumps(
        {
            "file": __file__,
            "argv0": sys.argv[0],
            "marker": namespace["MARKER"],
            "benchmark": "benchmark" in sys.argv,
        }
    ),
    encoding="utf-8",
)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    result = tmp_path / "result.json"
    fd = os.open(capture_tool, os.O_RDONLY | os.O_CLOEXEC)
    try:
        argv = isolation._capture_argv(
            python=Path(sys.executable),
            capture_tool=capture_tool.resolve(),
            capture_tool_fd=fd,
            repo_root=tmp_path,
            bundle=tmp_path / "bundle",
            build_receipt=tmp_path / "build.json",
            request=tmp_path / "request.json",
            worker=tmp_path / "worker",
            plugin_library=tmp_path / "plugin.so",
            comparison_sequence_limit=128,
            result_path=result,
            worker_stderr_path=tmp_path / "worker.stderr",
        )
        completed = subprocess.run(
            argv,
            cwd=tmp_path,
            pass_fds=(fd,),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    finally:
        os.close(fd)

    assert completed.returncode == 0, completed.stderr
    observed = json.loads(result.read_text(encoding="utf-8"))
    assert observed == {
        "file": str(capture_tool.resolve()),
        "argv0": str(capture_tool.resolve()),
        "marker": "canonical-sibling",
        "benchmark": True,
    }


def test_produces_cold_warm_and_concurrent_isolation_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(tmp_path, monkeypatch)
    report = isolation.run_qualification(args)

    assert report["status"] == "passed"
    assert report["source_state_unchanged"] is True
    assert report["inputs"]["capture_tool"] == report["inputs"][
        "capture_tool_source_binding"
    ]["identity"]
    assert set(report["executions"]) == set(isolation._RUN_LABELS)
    assert report["concurrency"]["worker_overlap_ns"] > 0
    assert report["concurrency"]["engine_load_overlap_ns"] > 0
    assert report["gpu_a"]["uuid"] != report["gpu_b"]["uuid"]
    assert report["gpu_a"]["pci_bus_id"] != report["gpu_b"]["pci_bus_id"]
    assert report["child_results"]["gpu-a-cold"]["cuda_jit_cache"][
        "initial_state"
    ] == "cold"
    assert report["child_results"]["gpu-a-warm"]["cuda_jit_cache"][
        "initial_state"
    ] == "warm"
    assert report["gates"]["cache_paths"][
        "warm_cache_continues_from_cold_process"
    ]
    assert report["gates"]["shared_child_evidence"]["token_ids"]
    assert report["gates"]["companion_qualification_receipts"][
        "correctness_hf_logit_parity_passed"
    ]
    assert report["gates"]["companion_qualification_receipts"][
        "correctness_qualified_engine_graph_passed"
    ]
    assert report["gates"]["companion_qualification_receipts"][
        "performance_split_08_09_passed"
    ]
    graph = report["companion_qualification_evidence"]["correctness"][
        "qualified_engine_graph"
    ]
    assert graph["passed"] is True
    assert isolation._SHA256.fullmatch(graph["sha256"])
    assert graph["runtime_stack"] == RUNTIME_STACK
    assert graph["runtime_stack_sha256"] == isolation._canonical_sha(
        RUNTIME_STACK
    )
    assert set(graph["engine_sections"]) == set(
        isolation._QUALIFIED_ENGINE_SECTIONS
    )
    assert len(graph["engine_plan_identities"]) == 2
    bundle_header = isolation.boundary._read_bundle_header(args.bundle)
    for identity in graph["engine_plan_identities"]:
        plan = isolation.boundary._read_bundle_section(
            args.bundle,
            bundle_header,
            identity["section_name"],
        )
        assert identity["size_bytes"] == len(plan)
        assert identity["sha256"] == hashlib.sha256(plan).hexdigest()
    assert len(graph["inspector_artifacts"]) == 2
    assert all(
        isolation._file_identity_matches(identity)
        for identity in graph["inspector_artifacts"]
    )
    assert report["companion_qualification_evidence"]["correctness"][
        "runtime_stack_sha256"
    ] == report["companion_qualification_evidence"]["performance"][
        "runtime_stack_sha256"
    ]
    assert report["claim_scope"]["does_not_prove"] == [
        (
            "each cold/warm/concurrent child independently recomputed Hugging "
            "Face reference logits"
        ),
        (
            "SPLIT-08/SPLIT-09 timing was measured inside each "
            "cold/warm/concurrent child"
        ),
    ]
    for execution in report["executions"].values():
        assert execution["attempt_count"] == 1
        assert execution["retry_count"] == 0
        assert execution["locking"] == "none"
        assert not execution["ld_preload_set"]
        assert execution["environment"]["CUDA_CACHE_DISABLE"] == "0"
        assert execution["stdout"]["sha256"]
        assert execution["stderr"]["sha256"]
        assert execution["argv_sha256"] == isolation._canonical_sha(
            execution["argv"]
        )
        assert execution["environment_sha256"] == isolation._canonical_sha(
            execution["environment"]
        )
        assert execution["argv"][1:3] == [
            "-c",
            isolation._PINNED_CAPTURE_TRAMPOLINE,
        ]
        assert Path(execution["argv"][4]) == args.capture_tool.resolve()


def test_rejects_external_capture_tool_even_when_bytes_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(tmp_path, monkeypatch)
    external = tmp_path / "external-capture.py"
    external.write_bytes(args.capture_tool.read_bytes())
    args.capture_tool = external

    with pytest.raises(
        isolation.IsolationError,
        match="canonical current-source producer",
    ):
        isolation.run_qualification(args)


@pytest.mark.parametrize(
    ("argument_name", "expected_label"),
    [
        ("build_receipt", "build_receipt"),
        ("worker", "worker"),
        ("plugin_library", "plugin_library"),
    ],
)
def test_rejects_aggregate_same_bytes_on_different_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argument_name: str,
    expected_label: str,
) -> None:
    args = _prepare(tmp_path, monkeypatch)
    original = Path(getattr(args, argument_name))
    replacement = tmp_path / f"replacement-{original.name}"
    replacement.write_bytes(original.read_bytes())
    setattr(args, argument_name, replacement)

    with pytest.raises(
        isolation.IsolationError,
        match=(
            f"aggregate {expected_label} exact identity differs from "
            "correctness base artifacts"
        ),
    ):
        isolation.run_qualification(args)


def test_rejects_missing_aggregate_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(tmp_path, monkeypatch)
    args.worker = tmp_path / "missing-worker"

    with pytest.raises(
        isolation.IsolationError,
        match="required file does not exist",
    ):
        isolation.run_qualification(args)


def test_rejects_preexisting_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(tmp_path, monkeypatch)
    args.output_dir.mkdir(parents=True)

    with pytest.raises(isolation.IsolationError, match="must not already exist"):
        isolation.run_qualification(args)


def test_requires_complete_hf_correctness_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(tmp_path, monkeypatch)
    _rewrite_json(
        args.correctness_report,
        lambda report: report["cases"].pop(),
    )

    with pytest.raises(
        isolation.IsolationError, match="complete canonical case matrix"
    ):
        isolation.run_qualification(args)


def test_requires_persisted_base_artifact_binding_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(tmp_path, monkeypatch)
    _rewrite_json(
        args.correctness_report,
        lambda report: report["qualification_gates"].update(
            {"base_artifact_binding_passed": False}
        ),
    )

    with pytest.raises(
        isolation.IsolationError,
        match="did not persist the base artifact binding gate",
    ):
        isolation.run_qualification(args)


def test_replays_correctness_base_artifact_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(tmp_path, monkeypatch)

    def select_wrong_model_dso(report: dict) -> None:
        report["base_artifact_binding"]["model_plugin"][
            "artifact_key"
        ] = "model_llama"

    _rewrite_json(
        args.correctness_report,
        select_wrong_model_dso,
    )
    with pytest.raises(
        isolation.IsolationError,
        match="base artifact binding did not replay",
    ):
        isolation.run_qualification(args)


def test_replays_correctness_runtime_plugin_loaded_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(tmp_path, monkeypatch)

    def change_loaded_inode(report: dict) -> None:
        report["runtime_kv_plugin_binding"]["loaded_mapping"][
            "inode"
        ] += 1

    _rewrite_json(
        args.correctness_report,
        change_loaded_inode,
    )
    with pytest.raises(
        isolation.IsolationError,
        match="runtime-KV plugin binding did not replay",
    ):
        isolation.run_qualification(args)


def test_rejects_correctness_case_without_hf_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(tmp_path, monkeypatch)

    def fail_parity(report: dict) -> None:
        case = next(
            item for item in report["cases"] if "parity" in item
        )
        case["parity"]["passed"] = False

    _rewrite_json(args.correctness_report, fail_parity)
    with pytest.raises(isolation.IsolationError, match="did not pass HF parity"):
        isolation.run_qualification(args)


def test_requires_qualified_engine_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(tmp_path, monkeypatch)
    _rewrite_json(
        args.correctness_report,
        lambda report: report.pop("qualified_engine_graph"),
    )

    with pytest.raises(
        isolation.IsolationError,
        match="qualified_engine_graph must be a JSON object",
    ):
        isolation.run_qualification(args)


def test_rejects_failed_qualified_engine_graph_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(tmp_path, monkeypatch)

    def fail_graph_gate(report: dict) -> None:
        report["qualified_engine_graph"]["gates"][
            "no_dense_attention_mask_or_scores"
        ] = False

    _rewrite_json(args.correctness_report, fail_graph_gate)
    with pytest.raises(
        isolation.IsolationError,
        match="required true gates",
    ):
        isolation.run_qualification(args)


def test_requires_exact_qualified_engine_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(tmp_path, monkeypatch)

    def remove_engine_section(report: dict) -> None:
        report["qualified_engine_graph"]["engine_sections"].pop(
            "engine_plan"
        )

    _rewrite_json(args.correctness_report, remove_engine_section)
    with pytest.raises(
        isolation.IsolationError,
        match="must contain exactly prefill_engine_plan and engine_plan",
    ):
        isolation.run_qualification(args)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "required file does not exist"),
        ("tampered", "inspector artifact identity mismatch"),
    ],
)
def test_rejects_missing_or_tampered_engine_inspector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    args = _prepare(tmp_path, monkeypatch)
    report = json.loads(
        args.correctness_report.read_text(encoding="utf-8")
    )
    inspector = Path(
        report["qualified_engine_graph"]["engine_sections"][
            "prefill_engine_plan"
        ]["inspector_path"]
    )
    if mutation == "missing":
        inspector.unlink()
    else:
        inspector.write_text(
            inspector.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )

    with pytest.raises(isolation.IsolationError, match=message):
        isolation.run_qualification(args)


def test_recomputes_engine_plan_sha_from_bundle_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(tmp_path, monkeypatch)

    def replace_engine_sha(report: dict) -> None:
        report["qualified_engine_graph"]["engine_sections"][
            "prefill_engine_plan"
        ]["engine_sha256"] = "3" * 64

    _rewrite_json(args.correctness_report, replace_engine_sha)
    with pytest.raises(
        isolation.IsolationError,
        match="engine SHA does not match the aggregate bundle section",
    ):
        isolation.run_qualification(args)


@pytest.mark.parametrize(
    "mismatch",
    ["native_plugin_layers", "dense_attention", "cache_concat"],
)
def test_recomputes_layer_evidence_from_engine_inspector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    args = _prepare(tmp_path, monkeypatch)
    report = json.loads(
        args.correctness_report.read_text(encoding="utf-8")
    )
    section = report["qualified_engine_graph"]["engine_sections"][
        "prefill_engine_plan"
    ]
    inspector = Path(section["inspector_path"])
    inspector_json = json.loads(inspector.read_text(encoding="utf-8"))
    if mismatch == "native_plugin_layers":
        inspector_json = inspector_json[:1]
    elif mismatch == "dense_attention":
        inspector_json.append(
            {
                "Name": "attention_mask",
                "LayerType": "Constant",
            }
        )
    else:
        inspector_json.append(
            {
                "Name": "cache_k_0.concat",
                "LayerType": "Concatenation",
            }
        )
    inspector.write_text(
        json.dumps(inspector_json),
        encoding="utf-8",
    )
    section["inspector_size_bytes"] = inspector.stat().st_size
    section["inspector_sha256"] = isolation._sha256(inspector)
    args.correctness_report.write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    with pytest.raises(
        isolation.IsolationError,
        match="reported layer evidence does not match its inspector artifact",
    ):
        isolation.run_qualification(args)


def test_rejects_qualified_engine_runtime_stack_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(tmp_path, monkeypatch)

    def change_graph_stack(report: dict) -> None:
        report["qualified_engine_graph"]["runtime_stack"][
            "driver"
        ] = "580.105.09"

    _rewrite_json(args.correctness_report, change_graph_stack)
    with pytest.raises(
        isolation.IsolationError,
        match="does not match the correctness case runtime stack",
    ):
        isolation.run_qualification(args)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda graph: graph["engine_sections"][
                "prefill_engine_plan"
            ]["inputs"]["position_id"]["profiles"][0].update(
                opt=[1]
            ),
            "token_id and position_id profiles are not identical",
        ),
        (
            lambda graph: graph["engine_sections"][
                "engine_plan"
            ]["inputs"]["history_length"]["profiles"][0].update(
                max=[2]
            ),
            "history_length profile",
        ),
        (
            lambda graph: graph["engine_sections"][
                "prefill_engine_plan"
            ]["outputs"]["logits"].update(shape=[-1, 151_936]),
            "logits shape",
        ),
        (
            lambda graph: graph["engine_sections"][
                "engine_plan"
            ]["inputs"]["cache_k_0"].update(shape=[-1, 512]),
            "does not use the source-bound KV width",
        ),
        (
            lambda graph: graph["model_contract"].update(kv_width=512),
            "does not match the aggregate bundle header",
        ),
    ],
)
def test_rejects_incomplete_engine_shape_or_kv_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    args = _prepare(tmp_path, monkeypatch)

    def mutate_graph(report: dict) -> None:
        mutation(report["qualified_engine_graph"])

    _rewrite_json(args.correctness_report, mutate_graph)
    with pytest.raises(isolation.IsolationError, match=message):
        isolation.run_qualification(args)


def test_rejects_correctness_source_or_runtime_tuple_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(tmp_path, monkeypatch)
    report = json.loads(args.correctness_report.read_text(encoding="utf-8"))
    stderr = Path(report["cases"][0]["runner_stderr"])
    stderr.write_text(
        _runtime_stack_line({**RUNTIME_STACK, "driver": "580.105.09"})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        isolation.IsolationError,
        match="more than one live runtime stack",
    ):
        isolation.run_qualification(args)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report.update(status="failed"),
            "performance report is not passed",
        ),
        (
            lambda report: report["gates"]["performance"]["short"].update(
                decode_throughput_gte_95_percent_static=False
            ),
            "contains a failed gate",
        ),
        (
            lambda report: report["cases"]["dynamic_short"][
                "qualification_provenance"
            ].update(source_state_sha256="9" * 64),
            "source_state_sha256 does not match aggregate source",
        ),
    ],
)
def test_rejects_failed_or_unbound_performance_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    args = _prepare(tmp_path, monkeypatch)
    _rewrite_json(args.performance_report, mutation)

    with pytest.raises(isolation.IsolationError, match=message):
        isolation.run_qualification(args)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-build-receipt",
        "wrong-build-receipt-sha",
        "same-bytes-new-receipt-inode",
        "swap-worker-plugin",
        "swap-runtime-plugin",
        "swap-build-plugin",
    ],
)
def test_replays_complete_dynamic_performance_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    args = _prepare(tmp_path, monkeypatch)
    capture_path = _dynamic_capture_path(args.performance_report)
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    evidence = capture["qualification_evidence"]
    provenance = capture["qualification_provenance"]
    if mutation == "missing-build-receipt":
        evidence.pop("build_receipt")
    elif mutation == "wrong-build-receipt-sha":
        evidence["build_receipt_sha256"] = "0" * 64
    elif mutation == "same-bytes-new-receipt-inode":
        original = Path(evidence["build_receipt"])
        replacement = tmp_path / "equal-build-receipt.json"
        replacement.write_bytes(original.read_bytes())
        evidence["build_receipt"] = str(replacement.resolve())
        evidence["build_receipt_sha256"] = isolation._sha256(
            replacement
        )
    elif mutation == "swap-worker-plugin":
        toolchain = evidence["toolchain"]
        toolchain["worker"], toolchain["plugin_library"] = (
            toolchain["plugin_library"],
            toolchain["worker"],
        )
        provenance["toolchain_sha256"] = isolation._canonical_sha(
            toolchain
        )
    elif mutation == "swap-runtime-plugin":
        runtime_trtmc = json.loads(
            json.dumps(capture["runtime_trtmc_libraries"])
        )
        runtime_trtmc["runtime_kv_plugin"] = runtime_trtmc["core"]
        capture["runtime_trtmc_libraries"] = runtime_trtmc
        evidence["runtime_trtmc_libraries"] = runtime_trtmc
        evidence["toolchain"]["runtime_trtmc_libraries"] = runtime_trtmc
        provenance["runtime_trtmc_libraries_sha256"] = (
            isolation._canonical_sha(runtime_trtmc)
        )
        provenance["toolchain_sha256"] = isolation._canonical_sha(
            evidence["toolchain"]
        )
    else:
        wrong_plugin = evidence["toolchain"]["worker"]
        capture["build_runtime_kv_plugin"] = wrong_plugin
        evidence["build_runtime_kv_plugin"] = wrong_plugin
        provenance["build_runtime_kv_plugin_sha256"] = (
            isolation._canonical_sha(wrong_plugin)
        )
    capture_path.write_text(json.dumps(capture), encoding="utf-8")

    with pytest.raises(
        isolation.IsolationError,
        match=(
            "dynamic capture provenance replay failed|"
            "build receipt exact identity differs from aggregate"
        ),
    ):
        isolation.run_qualification(args)


@pytest.mark.parametrize(
    ("behavior", "failed_gate"),
    [
        ("token_mismatch", ("shared_child_evidence", "token_ids")),
        (
            "break_warm_continuity",
            ("cache_paths", "warm_cache_continues_from_cold_process"),
        ),
    ],
)
def test_fails_closed_on_token_or_cache_continuity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    behavior: str,
    failed_gate: tuple[str, str],
) -> None:
    args = _prepare(tmp_path, monkeypatch, behavior=behavior)
    report = isolation.run_qualification(args)

    assert report["status"] == "failed"
    group, gate = failed_gate
    assert not report["gates"][group][gate]


def test_requires_true_concurrent_worker_interval_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(tmp_path, monkeypatch, behavior="no_worker_overlap")
    report = isolation.run_qualification(args)

    assert report["status"] == "failed"
    assert report["concurrency"]["worker_overlap_ns"] == 0
    assert not report["gates"]["concurrent_worker_intervals_overlap"]


def test_requires_true_concurrent_engine_load_interval_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(
        tmp_path, monkeypatch, behavior="no_engine_load_overlap"
    )
    report = isolation.run_qualification(args)

    assert report["status"] == "failed"
    assert report["concurrency"]["engine_load_overlap_ns"] == 0
    assert not report["gates"][
        "concurrent_engine_load_intervals_overlap"
    ]


def test_requires_engine_load_interval_inside_worker_lifetime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(
        tmp_path, monkeypatch, behavior="load_outside_worker"
    )
    report = isolation.run_qualification(args)

    assert report["status"] == "failed"
    assert any(
        "engine-load interval is outside the worker lifetime" in error
        for error in report["errors"]
    )


def test_aggregate_source_state_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(tmp_path, monkeypatch, behavior="mutate_source")
    report = isolation.run_qualification(args)

    assert report["status"] == "failed"
    assert report["source_state_unchanged"] is False
    assert not report["gates"]["source_state_unchanged"]


def test_child_source_state_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(tmp_path, monkeypatch, behavior="child_source_changed")
    report = isolation.run_qualification(args)

    assert report["status"] == "failed"
    assert any(
        "child source_state_unchanged failed" in error
        for error in report["errors"]
    )


@pytest.mark.parametrize(
    ("behavior", "error"),
    [
        (
            "runtime_stack_mismatch",
            "runtime stack differs from companion receipts",
        ),
        (
            "runtime_library_mismatch",
            "runtime libraries differ from performance receipt",
        ),
    ],
)
def test_child_runtime_tuple_must_match_companion_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    behavior: str,
    error: str,
) -> None:
    args = _prepare(tmp_path, monkeypatch, behavior=behavior)
    report = isolation.run_qualification(args)

    assert report["status"] == "failed"
    assert any(
        error in observed_error for observed_error in report["errors"]
    )


@pytest.mark.parametrize(
    ("behavior", "error"),
    [
        (
            "missing_child_build_receipt",
            "dynamic capture provenance replay failed",
        ),
        (
            "swap_child_worker_plugin",
            "dynamic capture provenance replay failed",
        ),
        (
            "missing_child_runtime_trtmc",
            "dynamic capture provenance replay failed",
        ),
        (
            "missing_child_build_plugin",
            "dynamic capture provenance replay failed",
        ),
    ],
)
def test_replays_complete_child_binary_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    behavior: str,
    error: str,
) -> None:
    args = _prepare(tmp_path, monkeypatch, behavior=behavior)
    report = isolation.run_qualification(args)

    assert report["status"] == "failed"
    assert any(error in observed for observed in report["errors"])


def test_capture_tool_endpoint_swap_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(
        tmp_path,
        monkeypatch,
        behavior="swap_capture_tool_inode",
    )

    with pytest.raises(
        isolation.IsolationError,
        match="canonical capture tool provenance failed",
    ):
        isolation.run_qualification(args)


def test_requires_two_distinct_physical_gpu_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(tmp_path, monkeypatch)
    duplicate = _inventory()
    duplicate["gpus"][1]["uuid"] = duplicate["gpus"][0]["uuid"]
    monkeypatch.setattr(isolation, "_gpu_inventory", lambda: duplicate)

    with pytest.raises(isolation.IsolationError, match="different UUIDs"):
        isolation.run_qualification(args)
