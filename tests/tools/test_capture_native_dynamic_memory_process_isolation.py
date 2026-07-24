# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


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
    subprocess.run(["git", "add", "tracked.txt"], cwd=path, check=True)
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

def argument(name):
    return sys.argv[sys.argv.index(name) + 1]

repo = Path(argument("--repo-root"))
output = Path(argument("--output"))
worker_stderr = Path(argument("--stderr-output"))
bundle = Path(argument("--bundle"))
request = json.loads(Path(argument("--request")).read_text(encoding="utf-8"))
build = json.loads(Path(argument("--build-receipt")).read_text(encoding="utf-8"))
label = output.parent.name
behavior = request.get("test_behavior", "")
cache_path = Path(os.environ["CUDA_CACHE_PATH"])

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

runtime_stack = build["runtime_stack"]
runtime_libraries = build["runtime_libraries"]
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
source_sha = build["source_state_sha256"]
source_unchanged = not (
    behavior == "child_source_changed" and label == "gpu-a-warm"
)
output_summary = {
    "text": "deterministic",
    "text_truncated": False,
    "token_ids": token_ids,
}
iterations = request["measurement"]["iterations"]
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
        "runtime_stack_sha256": canonical(runtime_stack),
        "runtime_libraries_sha256": canonical(runtime_libraries),
        "cuda_jit_cache_sha256": canonical(cache),
    },
    "qualification_evidence": {
        "source_state_unchanged": source_unchanged,
        "environment": {
            "cuda_visible_devices": visible,
            "cuda_logical_device": 0,
            "cuda_device_uuid": uuid,
            "cuda_pci_bus_id": pci,
            "cuda_compute_capability": "sm103",
        },
    },
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload), encoding="utf-8")
worker_stderr.write_text("worker diagnostic\n", encoding="utf-8")
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


def _write_correctness_report(
    inputs: Path,
    *,
    bundle: Path,
    source: dict,
) -> Path:
    evidence_dir = inputs / "correctness"
    evidence_dir.mkdir()
    spec = isolation.boundary.SPECS[MODEL_ID]
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
        "model_id": MODEL_ID,
        "bundle": str(bundle),
        "bundle_sha256": isolation._sha256(bundle),
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
) -> dict:
    return {
        "git_head": source["git_head"],
        "source_state_sha256": source["source_state_sha256"],
        "source_state_pre_sha256": source["source_state_sha256"],
        "source_state_post_sha256": source["source_state_sha256"],
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


def _write_performance_report(
    inputs: Path,
    *,
    dynamic_bundle: Path,
    source: dict,
    runtime_libraries: dict,
) -> Path:
    evidence_dir = inputs / "performance"
    evidence_dir.mkdir()
    static_bundle = evidence_dir / "static.trtfb"
    static_bundle.write_bytes(b"static-bundle")
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
        provenance = _performance_provenance(
            source=source,
            bundle=bundle,
            role=role,
            runtime_stack=stack,
            runtime_libraries=libraries,
        )
        capture = {
            "schema_version": isolation.CAPTURE_RESULT_SCHEMA,
            "status": "completed",
            "model_id": MODEL_ID,
            "runtime_stack": stack,
            "runtime_libraries": libraries,
            "qualification_provenance": provenance,
        }
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
    bundle.write_bytes(b"bundle")
    worker = inputs / "worker"
    worker.write_bytes(b"worker")
    plugin = inputs / "plugin.so"
    plugin.write_bytes(b"plugin")
    fake_capture = inputs / "fake_capture.py"
    fake_capture.write_text(_fake_capture_source(), encoding="utf-8")
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
    build_receipt = inputs / "build-receipt.json"
    build_receipt.write_text(
        json.dumps(
            {
                "git_head": source["git_head"],
                "source_state_sha256": source["source_state_sha256"],
                "model_revision": MODEL_REVISION,
                "runtime_stack": RUNTIME_STACK,
                "runtime_libraries": runtime_libraries,
            }
        ),
        encoding="utf-8",
    )
    correctness_report = _write_correctness_report(
        inputs, bundle=bundle, source=source
    )
    performance_report = _write_performance_report(
        inputs,
        dynamic_bundle=bundle,
        source=source,
        runtime_libraries=runtime_libraries,
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


def test_produces_cold_warm_and_concurrent_isolation_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(tmp_path, monkeypatch)
    report = isolation.run_qualification(args)

    assert report["status"] == "passed"
    assert report["source_state_unchanged"] is True
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
        "performance_split_08_09_passed"
    ]
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


def test_requires_two_distinct_physical_gpu_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _prepare(tmp_path, monkeypatch)
    duplicate = _inventory()
    duplicate["gpus"][1]["uuid"] = duplicate["gpus"][0]["uuid"]
    monkeypatch.setattr(isolation, "_gpu_inventory", lambda: duplicate)

    with pytest.raises(isolation.IsolationError, match="different UUIDs"):
        isolation.run_qualification(args)
