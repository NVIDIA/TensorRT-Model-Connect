# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import stat
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "capture_native_dynamic_memory_perf.py"
SPEC = importlib.util.spec_from_file_location(
    "capture_native_dynamic_memory_perf", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
capture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capture
SPEC.loader.exec_module(capture)

pytestmark = [pytest.mark.unit, pytest.mark.dynamic_memory]

PLAN_T512 = (
    "[trtmc.runtime_kv.plan] schema=1 device=0 role=history "
    "hq=16 hkv=8 d=128 C=128 Sq=1 T=512 stats=lse heur=A "
    "plan=eng10_k24=7 workspace_bytes=0 cudnn_version=92000"
)
PLAN_T1024 = (
    "[trtmc.runtime_kv.plan] schema=1 device=0 role=history "
    "hq=16 hkv=8 d=128 C=128 Sq=1 T=1024 stats=lse heur=A "
    "plan=eng10_k24=7 workspace_bytes=0 cudnn_version=92000"
)
RUNTIME_STACK = (
    "[trtmc.runtime_stack] schema=1 sm=sm103 "
    "tensorrt=11.2.0.113 cuda_runtime=13.3 cudnn_backend=9.20.0 "
    "cudnn_frontend_revision="
    "7b9b711c22b6823e87150213ecd8449260db8610 "
    "nvrtc=13.3 driver=580.105.08"
)


def _bundle_bytes(model_id: str = "example/model") -> bytes:
    tokenizer = b'{"model":{"type":"BPE"}}'
    config = b"{}"
    sections = {
        "engine_plan": {"offset": 0, "size": 4},
        "prefill_engine_plan": {"offset": 4, "size": 5},
        "tokenizer.json": {"offset": 9, "size": len(tokenizer)},
        "config.json": {
            "offset": 9 + len(tokenizer),
            "size": len(config),
        },
    }
    header = json.dumps(
        {
            "model_id": model_id,
            "tokenizer_add_special_tokens": 0,
            "sections": sections,
        }
    ).encode("utf-8")
    return (
        capture.BUNDLE_MAGIC
        + struct.pack("<Q", len(header))
        + header
        + b"123456789"
        + tokenizer
        + config
    )


def _init_repo(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=path,
        check=True,
    )
    (path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=path, check=True)


def _run_fresh_build(repo: Path) -> tuple[Path, Path]:
    bundle = repo / "artifacts" / "fresh.trtfb"
    receipt = repo / "artifacts" / "fresh-build.json"
    source_dir = repo / "artifacts" / "fresh-source"
    payload = _bundle_bytes().hex()
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path;"
            f"Path({str(bundle)!r}).write_bytes(bytes.fromhex({payload!r}))"
        ),
    ]
    args = SimpleNamespace(
        repo_root=repo,
        bundle=bundle,
        receipt=receipt,
        source_artifact_dir=source_dir,
        stdout_output=repo / "artifacts" / "fresh.stdout",
        stderr_output=repo / "artifacts" / "fresh.stderr",
        cwd=repo,
        role="native-dynamic",
        model_id="example/model",
        model_revision="revision",
        precision="bf16",
        target="sm103|TensorRT 11.2.0.113",
        bundle_build_id="fresh-build",
        command=command,
    )
    assert capture._cmd_build(args) == 0
    return bundle, receipt


def _accounting() -> dict:
    return {
        "serialized_plan_bytes": 9,
        "resident_weight_bytes": 2000,
        "resident_weight_copy_count": 2,
        "weight_streaming_active": False,
        "measurement_sources": dict(capture.MEASUREMENT_SOURCES),
        "engine_sections": [],
        "tensorrt_version": "11.2.0.113",
    }


def _fake_runtime_libraries(repo: Path) -> tuple[Path, Path]:
    directory = repo / "artifacts" / "runtime-libraries"
    directory.mkdir(parents=True, exist_ok=True)
    nvrtc = directory / "libnvrtc.so.13.3.33"
    builtins = directory / "libnvrtc-builtins.so.13.3"
    nvrtc.write_bytes(b"nvrtc")
    builtins.write_bytes(b"nvrtc-builtins")
    return nvrtc, builtins


def _fixed_generation_request(max_new_tokens: int = 2) -> dict:
    return {
        "prompt": "hello",
        "max_new_tokens": max_new_tokens,
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
    }


def test_native_worker_records_each_measured_generated_token_stream() -> None:
    source = (
        REPO_ROOT / "examples" / "trtmc_benchmark_worker.cpp"
    ).read_text(encoding="utf-8")

    assert 'observation["generated_token_ids"] = last.token_ids' in source


def test_generation_workload_is_fixed_repeatable_and_fail_closed() -> None:
    result = {
        "observations": [
            {"output_tokens": 2, "generated_token_ids": [1, 2]},
            {"output_tokens": 2, "generated_token_ids": [1, 2]},
        ],
        "output_summary": {"token_ids": [1, 2]},
    }
    workload = capture._generation_workload(
        result,
        semantic_request=_fixed_generation_request(),
        measurement={"warmup": 1, "iterations": 2},
    )

    assert workload["kind"] == "fixed_length_greedy_ar"
    assert workload["token_stream_repeatable_within_case"] is True

    result["observations"][1]["generated_token_ids"] = [1, 3]
    result["output_summary"]["token_ids"] = [1, 3]
    with pytest.raises(capture.CaptureError, match="changed across"):
        capture._generation_workload(
            result,
            semantic_request=_fixed_generation_request(),
            measurement={"warmup": 1, "iterations": 2},
        )


def test_tokenizer_contract_is_bound_to_bundle_payload(tmp_path: Path) -> None:
    bundle = tmp_path / "model.trtfb"
    bundle.write_bytes(_bundle_bytes())
    header, payload_offset = capture._bundle_header(bundle)

    contract = capture._tokenizer_contract(
        bundle, header, payload_offset
    )

    assert contract["schema_version"] == capture.TOKENIZER_CONTRACT_SCHEMA
    assert contract["tokenizer_add_special_tokens"] is False
    assert contract["tokenizer_json_bytes"] > 0


def test_engine_section_selection_is_exact() -> None:
    header = {
        "sections": {
            "engine_plan": {"offset": 0, "size": 10},
            "prefill_engine_plan": {"offset": 10, "size": 11},
            "vision_plan": {"offset": 21, "size": 12},
            "tokenizer.json": {"offset": 33, "size": 13},
        }
    }
    assert capture._engine_sections(header) == [
        ("engine_plan", 0, 10),
        ("prefill_engine_plan", 10, 11),
    ]


def test_build_executes_command_and_proves_fresh_source_bound_artifact(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    bundle, receipt_path = _run_fresh_build(repo)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert receipt["schema_version"] == capture.BUILD_SCHEMA
    assert receipt["fresh_build"] is True
    assert receipt["artifact_reused"] is False
    assert receipt["bundle_sha256"] == capture._sha256(bundle)
    assert (
        receipt["prebuild_source_state_sha256"]
        == receipt["postbuild_source_state_sha256"]
    )
    assert receipt["source_state_pre"]["git_dirty"] is False
    assert receipt["source_state_post"]["exact_head_gate_satisfied"] is True


def test_build_preserves_dirty_source_as_diagnostic_evidence(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")

    _, receipt_path = _run_fresh_build(repo)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert receipt["source_state_pre"]["git_dirty"] is True
    assert receipt["source_state_post"]["git_dirty"] is True
    assert (
        receipt["source_state_pre"]["exact_head_gate_satisfied"] is False
    )
    assert (
        receipt["source_state_post"]["exact_head_gate_satisfied"] is False
    )


def test_build_rejects_preexisting_bundle(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    bundle, _ = _run_fresh_build(repo)
    args = SimpleNamespace(
        repo_root=repo,
        bundle=bundle,
        receipt=repo / "artifacts" / "second.json",
        source_artifact_dir=repo / "artifacts" / "second-source",
        stdout_output=repo / "artifacts" / "second.stdout",
        stderr_output=repo / "artifacts" / "second.stderr",
        cwd=repo,
        role="native-dynamic",
        model_id="example/model",
        model_revision="revision",
        precision="bf16",
        target="target",
        bundle_build_id="second",
        command=[sys.executable, "-c", "raise SystemExit(0)"],
    )
    with pytest.raises(capture.CaptureError, match="already exists"):
        capture._cmd_build(args)


def test_benchmark_runs_real_worker_and_enriches_dynamic_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    bundle, build_receipt = _run_fresh_build(repo)
    request = repo / "artifacts" / "request.json"
    request.write_text(
        json.dumps(
            {
                "case_name": "case",
                "bundle": str(bundle),
                "operation": "generate",
                "runtime": {"max_sequence_length": 128},
                "measurement": {"warmup": 1, "iterations": 1},
                "request": {
                    "prompt": "hello",
                    "max_new_tokens": 1,
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
                },
            }
        ),
        encoding="utf-8",
    )
    plugin = repo / "artifacts" / "plugin.so"
    plugin.write_bytes(b"plugin")
    output = repo / "artifacts" / "result.json"
    stderr = repo / "artifacts" / "result.stderr"
    worker = repo / "artifacts" / "worker.py"
    worker.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
payload = {
    "schema_version": "trtmc.benchmark-worker-result/v1",
    "status": "completed",
    "case_name": "case",
    "case_digest": "digest",
    "model_id": "example/model",
    "pipeline_type": "Example",
    "operation": "generate",
    "timing_scope": "public_pipeline_call_wall",
    "load_ms": 1.0,
    "warmup": 1,
    "iterations": 1,
    "observations": [{
        "iteration": 0,
        "runtime_e2e_wall_ms": 3.0,
        "prefill_ms": 1.0,
        "decode_ms": 2.0,
        "output_tokens": 1,
        "generated_token_ids": [1]
    }],
    "output_summary": {"token_ids": [1]},
    "runtime_memory_receipt": {
        "serialized_plan_bytes": 9,
        "resident_weight_bytes": 2000,
        "resident_weight_copy_count": 2,
        "weight_streaming_active": False,
        "measurement_sources": {
            "serialized_plan_bytes": "bundle_engine_section_sizes",
            "resident_weight_bytes": "tensorrt_total_weights_size_weight_streaming_disabled",
            "resident_weight_copy_count": "deduplicated_tensorrt_engine_identity"
        }
    }
}
open(args["--output"], "w", encoding="utf-8").write(json.dumps(payload))
cache = Path(os.environ["CUDA_CACHE_PATH"])
cache.mkdir(parents=True, exist_ok=True)
(cache / "jit.bin").write_bytes(b"jit")
"""
        + f"print({PLAN_T512!r}, file=sys.stderr)\n"
        + f"print({RUNTIME_STACK!r}, file=sys.stderr)\n",
        encoding="utf-8",
    )
    worker.chmod(worker.stat().st_mode | stat.S_IXUSR)
    cache_path = repo / "artifacts" / "cuda-cache"
    monkeypatch.setenv("CUDA_CACHE_PATH", str(cache_path))
    monkeypatch.setenv("CUDA_CACHE_DISABLE", "0")
    monkeypatch.setattr(capture, "_engine_accounting", lambda *_: _accounting())
    monkeypatch.setattr(
        capture,
        "_environment_identity",
        lambda: {"target": "unit-test"},
    )
    runtime_libraries = _fake_runtime_libraries(repo)
    monkeypatch.setattr(
        capture,
        "_mapped_library_paths",
        lambda _pid: runtime_libraries,
    )
    args = SimpleNamespace(
        repo_root=repo,
        bundle=bundle,
        build_receipt=build_receipt,
        request=request,
        worker=worker,
        plugin_library=plugin,
        output=output,
        stderr_output=stderr,
        comparison_sequence_limit=128,
        cwd=repo,
        role="native-dynamic",
    )

    assert capture._cmd_benchmark(args) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["runtime_memory_receipt"]["resident_weight_copy_count"] == 2
    assert result["qualification_provenance"]["fresh_build"] is True
    assert result["qualification_provenance"]["artifact_reused"] is False
    assert (
        result["qualification_provenance"]["bundle_sha256"]
        == capture._sha256(bundle)
    )
    assert result["runtime_attention_plans"] == [
        capture._parse_runtime_attention_plans(
            PLAN_T512, artifact_role="native-dynamic"
        )[0]
    ]
    assert result["runtime_stack"] == capture._parse_runtime_stack(
        RUNTIME_STACK, artifact_role="native-dynamic"
    )
    cache = result["qualification_evidence"]["cuda_jit_cache"]
    assert cache["path"] == str(cache_path.resolve())
    assert cache["initial_state"] == "cold"
    assert cache["before"]["file_count"] == 0
    assert cache["after"]["file_count"] == 1
    assert result["cuda_jit_cache"] == cache
    assert (
        result["qualification_provenance"]["cuda_jit_cache_sha256"]
        == capture._canonical_sha(result["cuda_jit_cache"])
    )
    assert (
        result["qualification_provenance"][
            "runtime_attention_plans_sha256"
        ]
        == capture._canonical_sha(result["runtime_attention_plans"])
    )
    assert result["qualification_provenance"]["source_state_unchanged"] is True
    assert (
        result["qualification_provenance"]["source_state_pre_sha256"]
        == result["qualification_provenance"]["source_state_post_sha256"]
    )
    assert result["qualification_evidence"]["source_state_unchanged"] is True
    boundaries = result["qualification_provenance"][
        "source_state_boundaries"
    ]
    assert set(boundaries) == set(capture.SOURCE_STATE_BOUNDARY_NAMES)
    assert all(not row["git_dirty"] for row in boundaries.values())
    assert all(
        row["exact_head_gate_satisfied"] for row in boundaries.values()
    )
    assert all(
        row["git_head"] == result["qualification_provenance"]["git_head"]
        for row in boundaries.values()
    )
    assert result["qualification_evidence"]["runtime_libraries"] == {
        "directory": str(runtime_libraries[0].parent),
        "live_nvrtc_version": "13.3",
        "nvrtc": {
            "basename": runtime_libraries[0].name,
            "path": str(runtime_libraries[0]),
            "sha256": capture._sha256(runtime_libraries[0]),
            "size_bytes": runtime_libraries[0].stat().st_size,
        },
        "nvrtc_builtins": {
            "basename": runtime_libraries[1].name,
            "path": str(runtime_libraries[1]),
            "sha256": capture._sha256(runtime_libraries[1]),
            "size_bytes": runtime_libraries[1].stat().st_size,
        },
    }
    assert result["runtime_libraries"] == result["qualification_evidence"][
        "runtime_libraries"
    ]
    assert result["generation_workload"]["kind"] == "fixed_length_greedy_ar"
    assert result["generation_workload"]["measured_generated_token_ids"] == [
        [1]
    ]
    assert result["tokenizer_contract"][
        "tokenizer_json_sha256"
    ] == capture._tokenizer_contract(
        bundle,
        capture._bundle_header(bundle)[0],
        capture._bundle_header(bundle)[1],
    )["tokenizer_json_sha256"]
    assert (
        result["qualification_provenance"]["runtime_libraries_sha256"]
        == capture._canonical_sha(result["runtime_libraries"])
    )


def test_dynamic_benchmark_rejects_disagreeing_runtime_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    bundle, build_receipt = _run_fresh_build(repo)
    request = repo / "artifacts" / "request.json"
    request.write_text(
        json.dumps(
            {
                "bundle": str(bundle),
                "operation": "generate",
                "runtime": {"max_sequence_length": 128},
                "measurement": {"warmup": 1, "iterations": 1},
                "request": {
                    "prompt": "hello",
                    "max_new_tokens": 1,
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
                },
            }
        ),
        encoding="utf-8",
    )
    plugin = repo / "artifacts" / "plugin.so"
    plugin.write_bytes(b"plugin")
    worker = repo / "artifacts" / "worker.py"
    worker.write_text(
        """#!/usr/bin/env python3
import json
import sys
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
json.dump({
    "schema_version": "trtmc.benchmark-worker-result/v1",
    "status": "completed",
    "observations": [{
        "output_tokens": 1,
        "generated_token_ids": [1]
    }],
    "output_summary": {"token_ids": [1]},
    "runtime_memory_receipt": {
        "serialized_plan_bytes": 10,
        "resident_weight_bytes": 2000,
        "resident_weight_copy_count": 2,
        "weight_streaming_active": False,
        "measurement_sources": {
            "serialized_plan_bytes": "bundle_engine_section_sizes",
            "resident_weight_bytes": "tensorrt_total_weights_size_weight_streaming_disabled",
            "resident_weight_copy_count": "deduplicated_tensorrt_engine_identity"
        }
    }
}, open(args["--output"], "w", encoding="utf-8"))
"""
        + f"print({PLAN_T512!r}, file=sys.stderr)\n"
        + f"print({RUNTIME_STACK!r}, file=sys.stderr)\n",
        encoding="utf-8",
    )
    worker.chmod(worker.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv(
        "CUDA_CACHE_PATH", str(repo / "artifacts" / "cuda-cache")
    )
    monkeypatch.setattr(capture, "_engine_accounting", lambda *_: _accounting())
    runtime_libraries = _fake_runtime_libraries(repo)
    monkeypatch.setattr(
        capture,
        "_mapped_library_paths",
        lambda _pid: runtime_libraries,
    )
    args = SimpleNamespace(
        repo_root=repo,
        bundle=bundle,
        build_receipt=build_receipt,
        request=request,
        worker=worker,
        plugin_library=plugin,
        output=repo / "artifacts" / "result.json",
        stderr_output=repo / "artifacts" / "result.stderr",
        comparison_sequence_limit=128,
        cwd=repo,
        role="native-dynamic",
    )
    with pytest.raises(capture.CaptureError, match="disagrees"):
        capture._cmd_benchmark(args)


def test_runtime_plan_parser_deduplicates_and_sorts_rows() -> None:
    plans = capture._parse_runtime_attention_plans(
        "\n".join((PLAN_T1024, PLAN_T512, PLAN_T512)),
        artifact_role="native-dynamic",
    )

    assert [(row["Sq"], row["T"]) for row in plans] == [
        (1, 512),
        (1, 1024),
    ]
    assert plans[0]["plan"] == "eng10_k24=7"
    assert plans[0]["workspace_bytes"] == 0


@pytest.mark.parametrize(
    ("stderr", "error"),
    [
        ("unrelated diagnostic", "did not emit"),
        (
            PLAN_T512.replace(" cudnn_version=92000", ""),
            "malformed runtime attention plan",
        ),
        (
            PLAN_T512.replace("stats=lse", "stats=max_sum_exp"),
            "stats=lse",
        ),
        (
            PLAN_T512 + "\nCUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED",
            "COMPILATION_FAILED",
        ),
        (
            PLAN_T512
            + "\n"
            + PLAN_T512.replace(
                "plan=eng10_k24=7", "plan=eng1_k24=35"
            ),
            "conflicting runtime attention plan rows",
        ),
    ],
)
def test_dynamic_runtime_plan_parser_fails_closed(
    stderr: str, error: str
) -> None:
    with pytest.raises(capture.CaptureError, match=error):
        capture._parse_runtime_attention_plans(
            stderr, artifact_role="native-dynamic"
        )


def test_static_runtime_plan_parser_rejects_any_runtime_plan() -> None:
    assert (
        capture._parse_runtime_attention_plans(
            "ordinary static diagnostic",
            artifact_role="exact-head-static-split",
        )
        == []
    )
    with pytest.raises(capture.CaptureError, match="static baseline"):
        capture._parse_runtime_attention_plans(
            PLAN_T512,
            artifact_role="exact-head-static-split",
        )


def test_runtime_stack_parser_requires_one_complete_live_tuple() -> None:
    expected = capture._parse_runtime_stack(
        RUNTIME_STACK, artifact_role="native-dynamic"
    )
    assert expected == {
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
    assert capture._encoded_cudnn_version("9.20.0") == 92000


@pytest.mark.parametrize(
    ("stderr", "error"),
    [
        ("ordinary diagnostic", "did not emit"),
        (
            RUNTIME_STACK.replace(" nvrtc=13.3", ""),
            "malformed runtime-stack",
        ),
        (
            RUNTIME_STACK.replace("nvrtc=13.3", "nvrtc=unavailable"),
            "unavailable",
        ),
        (
            RUNTIME_STACK
            + "\n"
            + RUNTIME_STACK.replace("nvrtc=13.3", "nvrtc=13.0"),
            "conflicting",
        ),
    ],
)
def test_runtime_stack_parser_fails_closed(
    stderr: str, error: str
) -> None:
    with pytest.raises(capture.CaptureError, match=error):
        capture._parse_runtime_stack(
            stderr, artifact_role="native-dynamic"
        )


def test_static_runtime_stack_parser_rejects_dynamic_evidence() -> None:
    assert (
        capture._parse_runtime_stack(
            "ordinary static diagnostic",
            artifact_role="exact-head-static-split",
        )
        is None
    )
    with pytest.raises(capture.CaptureError, match="static baseline"):
        capture._parse_runtime_stack(
            RUNTIME_STACK,
            artifact_role="exact-head-static-split",
        )


def test_runtime_library_provenance_requires_one_coherent_pair(
    tmp_path: Path,
) -> None:
    nvrtc, builtins = _fake_runtime_libraries(tmp_path)
    stack = capture._parse_runtime_stack(
        RUNTIME_STACK, artifact_role="native-dynamic"
    )

    result = capture._runtime_library_provenance(
        (nvrtc, builtins),
        artifact_role="native-dynamic",
        runtime_stack=stack,
    )
    assert result is not None
    assert result["directory"] == str(nvrtc.parent)
    assert result["nvrtc"]["sha256"] == capture._sha256(nvrtc)
    assert result["nvrtc_builtins"]["sha256"] == capture._sha256(builtins)

    duplicate = nvrtc.parent / "libnvrtc.so.13.3.99"
    duplicate.write_bytes(b"other")
    with pytest.raises(capture.CaptureError, match="exactly one nvrtc"):
        capture._runtime_library_provenance(
            (nvrtc, duplicate, builtins),
            artifact_role="native-dynamic",
            runtime_stack=stack,
        )

    other_directory = tmp_path / "other"
    other_directory.mkdir()
    other_builtins = other_directory / builtins.name
    other_builtins.write_bytes(b"other-builtins")
    with pytest.raises(capture.CaptureError, match="one directory"):
        capture._runtime_library_provenance(
            (nvrtc, other_builtins),
            artifact_role="native-dynamic",
            runtime_stack=stack,
        )


def test_static_runtime_library_provenance_is_not_required() -> None:
    assert (
        capture._runtime_library_provenance(
            (),
            artifact_role="exact-head-static-split",
            runtime_stack=None,
        )
        is None
    )


def test_cuda_cache_snapshot_distinguishes_cold_warm_and_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "relative-cache"
    monkeypatch.setenv("CUDA_CACHE_PATH", cache_path.name)
    monkeypatch.delenv("CUDA_CACHE_DISABLE", raising=False)
    configuration = capture._cuda_cache_configuration(tmp_path)

    assert configuration["path"] == str(cache_path.resolve())
    cold = capture._cuda_cache_snapshot(cache_path)
    assert capture._cuda_cache_initial_state(configuration, cold) == "cold"

    cache_path.mkdir()
    (cache_path / "compiled.bin").write_bytes(b"compiled")
    warm = capture._cuda_cache_snapshot(cache_path)
    assert warm["file_count"] == 1
    assert warm["total_bytes"] == len(b"compiled")
    assert capture._cuda_cache_initial_state(configuration, warm) == "warm"

    monkeypatch.setenv("CUDA_CACHE_DISABLE", "1")
    disabled = capture._cuda_cache_configuration(tmp_path)
    assert (
        capture._cuda_cache_initial_state(disabled, warm) == "disabled"
    )
