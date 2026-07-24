# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import _ctypes
import _ssl
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.tools.dynamic_memory_manifest_fixture import (
    complete_command_receipts,
    load_manifest_module,
    seed_manifest_test_modules,
)


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


@pytest.fixture(autouse=True)
def _isolate_runtime_kv_plugin_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep explicit-plugin tests independent of the manifest runner."""

    monkeypatch.delenv(capture.RUNTIME_KV_PLUGIN_ENV, raising=False)


PLAN_T512 = (
    "[trtmc.runtime_kv.plan] schema=1 device=0 role=history "
    "hq=16 hkv=8 d=128 C=128 Sq=1 T=512 stats=lse heur=A "
    "plan=eng10_k24=7 workspace_bytes=0 cudnn_version=92000"
)
MODEL_ID = "Qwen/Qwen3-0.6B"
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
RUNTIME_STACK_CONTRACT = {
    "sm": "sm103",
    "tensorrt": "11.2.0.113",
    "cuda_runtime": "13.3",
    "cudnn_backend": "9.20.0",
    "cudnn_frontend_revision": (
        "7b9b711c22b6823e87150213ecd8449260db8610"
    ),
    "nvrtc": "13.3",
    "driver": "580.105.08",
}
PROFILE_LIMITS = (128, 256, 512, 1024, 2048, 8192, 32768, 40960)
MODULE_RESIDENCY_RESERVE_BYTES = 1024


def _runtime_memory_contract(
    engine_plan: bytes,
    prefill_engine_plan: bytes,
) -> dict:
    boundary = capture._load_boundary_module()
    plans = [
        {
            "section_name": "engine_plan",
            "section_sha256": hashlib.sha256(engine_plan).hexdigest(),
            "role": "decode",
            "optimization_profile_count": len(PROFILE_LIMITS),
        },
        {
            "section_name": "prefill_engine_plan",
            "section_sha256": hashlib.sha256(
                prefill_engine_plan
            ).hexdigest(),
            "role": "prefill",
            "optimization_profile_count": 1,
        },
    ]
    return {
        "contract_version": 2,
        "qualified_model_id": MODEL_ID,
        "qualified_model_revision": "1" * 40,
        "qualified_config_sha256": "2" * 64,
        "qualified_target": "sm103 + TensorRT 11.2.0.113",
        "qualified_runtime_stack": dict(RUNTIME_STACK_CONTRACT),
        "native_kv_plugin_abi": 2,
        "model_context_limit": 40960,
        "prefill_chunk_limit": 1024,
        "kv_layout": "contiguous_v1",
        "kv_dtype": "bfloat16",
        "kv_bytes_per_token": 114688,
        "active_kv_profile_limits": list(PROFILE_LIMITS),
        "runtime_owned": True,
        "module_residency_calibration": {
            "schema_version": 1,
            "measurement_kind": "nvml_process_cumulative_first_use",
            "cuda_module_loading_mode": "lazy",
            "qualified_runtime_stack_sha256": (
                boundary.qualified_runtime_stack_sha256(
                    RUNTIME_STACK_CONTRACT
                )
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


def _bundle_bytes(model_id: str = MODEL_ID) -> bytes:
    engine_plan = b"1234"
    prefill_engine_plan = b"56789"
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
            "runtime_memory": _runtime_memory_contract(
                engine_plan,
                prefill_engine_plan,
            ),
            "sections": sections,
        }
    ).encode("utf-8")
    return (
        capture.BUNDLE_MAGIC
        + struct.pack("<Q", len(header))
        + header
        + engine_plan
        + prefill_engine_plan
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
    seed_manifest_test_modules(path)
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=path, check=True)


def _write_fake_trtmc(executable: Path, bundle: Path) -> None:
    payload = _bundle_bytes().hex()
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.write_text(
        (
            "#!/usr/bin/env python3\n"
            "import ctypes\n"
            "import os\n"
            "from pathlib import Path\n"
            "import time\n"
            f"_runtime_plugin = ctypes.CDLL(os.environ[{capture.RUNTIME_KV_PLUGIN_ENV!r}])\n"
            "time.sleep(0.1)\n"
            f"Path({str(bundle)!r}).write_bytes(bytes.fromhex({payload!r}))\n"
        ),
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)


def _manifest_identity(
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
        "sha256": capture._sha256(path),
    }


def _write_test_build_manifest(
    repo: Path,
    build_dir: Path,
    source_state: dict,
) -> Path:
    relative_paths = {
        "trtmc": "trtmc",
        "benchmark_worker": "trtmc_benchmark_worker",
        "core": "libtrtmc_core.so",
        "trt_backend": "libtrtmc_backend_trt.so",
        "runtime_kv_plugin": "libtrtmc_trt_plugins.so",
        "model_qwen": "models/qwen/libtrtmc_model_qwen.so",
        "model_llama": "models/llama/libtrtmc_model_llama.so",
        "qualify": "trtmc_dynamic_memory_qualify",
        "surfaces": "trtmc_dynamic_memory_surfaces",
        "nvrtc_optional_output_regression": (
            "trtmc_nvrtc_optional_output_regression"
        ),
    }
    for key, relative in relative_paths.items():
        artifact = build_dir / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if not artifact.exists():
            artifact.write_bytes(f"test-{key}".encode())
    cache = build_dir / "CMakeCache.txt"
    cache.write_text(
        (
            f"CMAKE_HOME_DIRECTORY:INTERNAL={repo.resolve()}\n"
            "TRTMC_TRT_BACKEND_ABI:STRING=11_2\n"
        ),
        encoding="utf-8",
    )
    cache_identity = _manifest_identity(
        cache, artifact_key="cmake_cache", relative_path="CMakeCache.txt"
    )
    cache_identity["configured_source"] = str(repo.resolve())
    active_backend = build_dir / "libtrtmc_backend_trt_11_2.so"
    active_backend.symlink_to("libtrtmc_backend_trt.so")
    artifact_paths = dict(relative_paths)
    artifact_paths["trt_backend"] = "libtrtmc_backend_trt_11_2.so"
    artifacts = {
        key: _manifest_identity(
            build_dir / relative,
            artifact_key=key,
            relative_path=relative,
        )
        for key, relative in artifact_paths.items()
    }
    manifest_module = load_manifest_module(REPO_ROOT)
    commands = complete_command_receipts(
        manifest_module,
        repo_root=repo,
        build_dir=build_dir,
        output_dir=build_dir,
        python=Path(sys.executable),
    )
    manifest = {
        "schema_version": capture.BUILD_MANIFEST_SCHEMA,
        "repo_root": str(repo.resolve()),
        "build_dir": str(build_dir.resolve()),
        "python": sys.executable,
        "source_state_pre": source_state,
        "commands": commands,
        "passed": True,
        "source_state_post": source_state,
        "source_state_unchanged": True,
        "cmake_cache": cache_identity,
        "build_artifacts": artifacts,
        "build_artifacts_sha256": capture._canonical_sha(artifacts),
        "clean_build_command_sha256": capture._canonical_sha(commands[0]),
    }
    path = build_dir / "build-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _run_fresh_build(
    repo: Path, *, worker_source: str | None = None
) -> tuple[Path, Path]:
    bundle = repo / "artifacts" / "fresh.trtfb"
    receipt = repo / "artifacts" / "fresh-build.json"
    source_dir = repo / "artifacts" / "fresh-source"
    plugin = repo / "artifacts" / capture.RUNTIME_KV_PLUGIN_DSO
    plugin.parent.mkdir(parents=True, exist_ok=True)
    plugin.write_bytes(Path(_ssl.__file__).read_bytes())
    trtmc = repo / "artifacts" / "trtmc"
    _write_fake_trtmc(trtmc, bundle)
    worker = repo / "artifacts" / "trtmc_benchmark_worker"
    worker.write_text(
        worker_source or "#!/bin/sh\nexit 0\n", encoding="utf-8"
    )
    worker.chmod(worker.stat().st_mode | stat.S_IXUSR)
    source_state = capture._source_state(
        repo, source_dir, label="manifest-source"
    )
    build_manifest = _write_test_build_manifest(
        repo, repo / "artifacts", source_state
    )
    command = [
        str(trtmc),
        "build",
        MODEL_ID,
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
        model_id=MODEL_ID,
        model_revision="revision",
        precision="bf16",
        target="sm103|TensorRT 11.2.0.113",
        bundle_build_id="fresh-build",
        plugin_library=plugin,
        build_manifest=build_manifest,
        command=command,
    )
    assert capture._cmd_build(args) == 0
    return bundle, receipt


def _fresh_build_args(
    repo: Path,
    *,
    bundle: Path,
    receipt: Path,
    trtmc: Path,
    plugin_library: Path | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        repo_root=repo,
        bundle=bundle,
        receipt=receipt,
        source_artifact_dir=repo / "artifacts" / f"{receipt.stem}-source",
        stdout_output=repo / "artifacts" / f"{receipt.stem}.stdout",
        stderr_output=repo / "artifacts" / f"{receipt.stem}.stderr",
        cwd=repo,
        role="native-dynamic",
        model_id=MODEL_ID,
        model_revision="revision",
        precision="bf16",
        target="sm103|TensorRT 11.2.0.113",
        bundle_build_id=receipt.stem,
        plugin_library=plugin_library,
        build_manifest=None,
        command=[str(trtmc), "build", MODEL_ID],
    )


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


def _mapping_record(path: Path) -> dict:
    canonical = path.resolve()
    metadata = canonical.stat()
    return {
        "path": str(canonical),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "deleted": False,
    }


def _mapped_identities(*paths: Path) -> list[dict]:
    return sorted(
        (
            capture._file_identity(path, label=f"mapped {path.name}")
            for path in paths
        ),
        key=lambda row: (
            row["path"],
            row["device"],
            row["inode"],
        ),
    )


def _worker_mapping_records(
    repo: Path, runtime_libraries: tuple[Path, Path]
) -> tuple[dict, ...]:
    build = repo / "artifacts"
    paths = (
        *runtime_libraries,
        build / "libtrtmc_core.so",
        build / "libtrtmc_backend_trt.so",
        build / capture.RUNTIME_KV_PLUGIN_DSO,
        build / "models/qwen/libtrtmc_model_qwen.so",
    )
    return tuple(_mapping_record(path) for path in paths)


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


def _complete_schema_v4_receipt(serialized_plan_bytes: int) -> dict:
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
        "module_residency_plan_set_sha256": (
            _runtime_memory_contract(b"1234", b"56789")[
                "module_residency_calibration"
            ]["plan_set_sha256"]
        ),
        "module_residency_evidence_sha256": "3" * 64,
        "module_residency_cuda_module_loading_mode": "lazy",
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
        "resident_weight_bytes": 2_000,
        "resident_weight_copy_count": 2,
        "engine_weight_bytes": 2_000,
        "weight_streaming_active": False,
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
        "measurement_sources": dict(capture.MEASUREMENT_SOURCES),
    }


def _benchmark_worker_source(serialized_plan_bytes: int) -> str:
    runtime_receipt = _complete_schema_v4_receipt(
        serialized_plan_bytes
    )
    return f"""#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
payload = {{
    "schema_version": "trtmc.benchmark-worker-result/v1",
    "status": "completed",
    "case_name": "case",
    "case_digest": "digest",
    "model_id": "{MODEL_ID}",
    "pipeline_type": "Example",
    "operation": "generate",
    "timing_scope": "public_pipeline_call_wall",
    "load_ms": 1.0,
    "warmup": 1,
    "iterations": 1,
    "observations": [{{
        "iteration": 0,
        "runtime_e2e_wall_ms": 3.0,
        "prefill_ms": 1.0,
        "decode_ms": 2.0,
        "output_tokens": 1,
        "generated_token_ids": [1]
    }}],
    "output_summary": {{"token_ids": [1]}},
    "runtime_memory_receipt": {runtime_receipt!r}
}}
open(args["--output"], "w", encoding="utf-8").write(json.dumps(payload))
cache_path = os.environ.get("CUDA_CACHE_PATH")
if cache_path:
    cache = Path(cache_path)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "jit.bin").write_bytes(b"jit")
print({PLAN_T512!r}, file=sys.stderr)
print({RUNTIME_STACK!r}, file=sys.stderr)
"""


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
    plugin = repo / "artifacts" / capture.RUNTIME_KV_PLUGIN_DSO
    assert receipt["runtime_kv_plugin"] == capture._file_identity(
        plugin,
        label="test plugin",
    )
    assert Path(receipt["runtime_kv_plugin"]["path"]).is_absolute()
    assert (
        receipt["prebuild_source_state_sha256"]
        == receipt["postbuild_source_state_sha256"]
    )
    assert receipt["source_state_pre"]["git_dirty"] is False
    assert receipt["source_state_post"]["exact_head_gate_satisfied"] is True


@pytest.mark.parametrize("layout", ("adjacent", "packaged"))
def test_no_flag_trtmc_build_selects_its_source_bound_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    bundle = repo / "artifacts" / f"{layout}.trtfb"
    receipt_path = repo / "artifacts" / f"{layout}.receipt.json"
    if layout == "adjacent":
        trtmc = repo / "build-dynkv" / "trtmc"
        plugin = trtmc.parent / capture.RUNTIME_KV_PLUGIN_DSO
    else:
        trtmc = repo / "venv" / "bin" / "trtmc"
        plugin = (
            repo
            / "venv"
            / "lib"
            / "python3.12"
            / "site-packages"
            / "tensorrt_model_connect"
            / "bin"
            / capture.RUNTIME_KV_PLUGIN_DSO
        )
    _write_fake_trtmc(trtmc, bundle)
    plugin.parent.mkdir(parents=True, exist_ok=True)
    plugin.write_bytes(Path(_ssl.__file__).read_bytes())
    monkeypatch.delenv(capture.RUNTIME_KV_PLUGIN_ENV, raising=False)

    args = _fresh_build_args(
        repo,
        bundle=bundle,
        receipt=receipt_path,
        trtmc=trtmc,
        plugin_library=None,
    )
    assert args.command == [str(trtmc), "build", MODEL_ID]
    args.build_manifest = repo / "artifacts" / "selection-manifest.json"
    source_state = capture._source_state(
        repo, args.source_artifact_dir, label="selection-manifest"
    )
    manifest_binding = {
        "path": str(args.build_manifest.resolve()),
        "sha256": "1" * 64,
        "schema_version": capture.BUILD_MANIFEST_SCHEMA,
        "git_head": source_state["git_head"],
        "source_state_sha256": source_state["source_state_sha256"],
        "build_artifacts_sha256": "2" * 64,
    }
    monkeypatch.setattr(
        capture,
        "_read_build_manifest",
        lambda _path: (
            manifest_binding,
            {
                "trtmc": capture._file_identity(
                    trtmc, label="selection trtmc"
                ),
                "runtime_kv_plugin": capture._file_identity(
                    plugin, label="selection plugin"
                ),
            },
        ),
    )
    assert capture._cmd_build(args) == 0

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["runtime_kv_plugin"] == capture._file_identity(
        plugin,
        label=f"{layout} plugin",
    )


def test_dynamic_build_rejects_non_trtmc_command_even_with_a_plugin(
    tmp_path: Path,
) -> None:
    plugin = tmp_path / "plugin.so"
    plugin.write_bytes(Path(_ssl.__file__).read_bytes())

    with pytest.raises(
        capture.CaptureError,
        match="must execute `trtmc build",
    ):
        capture._select_build_plugin(
            command=[sys.executable, "-c", "pass"],
            cwd=tmp_path,
            explicit_plugin=plugin,
        )


@pytest.mark.parametrize(
    "command_suffix",
    (
        ["--max-sequence-length", "4096"],
        ["--kv-cache-size", "8GiB"],
        ["--output", "other.trtfb"],
    ),
)
def test_dynamic_build_rejects_every_extra_model_build_flag(
    tmp_path: Path,
    command_suffix: list[str],
) -> None:
    trtmc = tmp_path / "trtmc"
    trtmc.write_text("#!/bin/sh\n", encoding="utf-8")
    trtmc.chmod(trtmc.stat().st_mode | stat.S_IXUSR)

    with pytest.raises(capture.CaptureError, match="argv must be exactly"):
        capture._resolve_command_executable(
            [str(trtmc), "build", MODEL_ID, *command_suffix],
            tmp_path,
            model_id=MODEL_ID,
        )


def test_dynamic_build_rejects_command_model_metadata_mismatch(
    tmp_path: Path,
) -> None:
    trtmc = tmp_path / "trtmc"
    trtmc.write_text("#!/bin/sh\n", encoding="utf-8")
    trtmc.chmod(trtmc.stat().st_mode | stat.S_IXUSR)

    with pytest.raises(capture.CaptureError, match="argv must be exactly"):
        capture._resolve_command_executable(
            [
                str(trtmc),
                "build",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            ],
            tmp_path,
            model_id=MODEL_ID,
        )


def test_plugin_mapping_rejects_duplicate_renamed_and_deleted_dsos(
    tmp_path: Path,
) -> None:
    selected = tmp_path / capture.RUNTIME_KV_PLUGIN_DSO
    selected.write_bytes(b"selected")
    identity = capture._file_identity(selected, label="selected plugin")
    selected_record = _mapping_record(selected)
    duplicate_dir = tmp_path / "duplicate"
    duplicate_dir.mkdir()
    duplicate = duplicate_dir / capture.RUNTIME_KV_PLUGIN_DSO
    duplicate.write_bytes(b"duplicate")
    with pytest.raises(capture.CaptureError, match="exactly one"):
        capture._validate_exact_plugin_mapping(
            (selected_record, _mapping_record(duplicate)),
            selected=identity,
            where="test",
        )

    renamed = duplicate_dir / "arbitrary-runtime-extension.so"
    renamed.write_bytes(capture.RUNTIME_KV_PLUGIN_ABI_SYMBOL)
    with pytest.raises(capture.CaptureError, match="exactly one"):
        capture._validate_exact_plugin_mapping(
            (selected_record, _mapping_record(renamed)),
            selected=identity,
            where="test",
        )

    deleted_record = {**selected_record, "deleted": True}
    with pytest.raises(capture.CaptureError, match="was deleted"):
        capture._validate_exact_plugin_mapping(
            (deleted_record,),
            selected=identity,
            where="test",
        )


def test_pinned_execution_uses_open_inode_after_path_swap(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "worker"
    executable.write_text(
        "#!/bin/sh\nprintf original\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    with capture._PinnedFile(executable, label="test executable") as pinned:
        executable.unlink()
        executable.write_text(
            "#!/bin/sh\nprintf replacement\n",
            encoding="utf-8",
        )
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        completed = subprocess.run(
            capture._pinned_execution_command(pinned, ()),
            check=True,
            pass_fds=(pinned.fd,),
            stdout=subprocess.PIPE,
            text=True,
        )
        assert completed.stdout == "original"
        with pytest.raises(capture.CaptureError, match="changed"):
            pinned.verify()


def test_pinned_elf_symbol_scan_has_no_path_or_basename_filter() -> None:
    extension = Path(_ctypes.__file__)
    with capture._PinnedFile(
        extension,
        label="arbitrary mapped extension",
    ) as pinned:
        assert pinned.exports_dynamic_symbol(b"PyInit__ctypes")
        assert not pinned.exports_dynamic_symbol(
            b"trtmc_symbol_that_does_not_exist"
        )


def test_build_receipt_reopens_plugin_and_rejects_dso_drift(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    bundle, receipt_path = _run_fresh_build(repo)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    plugin = repo / "artifacts" / capture.RUNTIME_KV_PLUGIN_DSO
    plugin.write_bytes(plugin.read_bytes() + b"tampered")

    with pytest.raises(
        capture.CaptureError,
        match=(
            "artifact identity changed|size_bytes no longer matches|"
            "sha256 no longer matches"
        ),
    ):
        capture._validate_build_receipt(
            receipt,
            bundle=bundle,
            role="native-dynamic",
            source_state=receipt["source_state_post"],
            plugin_library=plugin,
        )


def test_qualification_build_rejects_dirty_source_manifest(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(
        capture.CaptureError,
        match="clean exact HEAD|exact-head build manifest",
    ):
        _run_fresh_build(repo)


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
        model_id=MODEL_ID,
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
    bundle, build_receipt = _run_fresh_build(
        repo, worker_source=_benchmark_worker_source(9)
    )
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
    plugin = repo / "artifacts" / capture.RUNTIME_KV_PLUGIN_DSO
    output = repo / "artifacts" / "result.json"
    stderr = repo / "artifacts" / "result.stderr"
    worker = repo / "artifacts" / "trtmc_benchmark_worker"
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
        "_mapped_library_records",
        lambda _pid: _worker_mapping_records(repo, runtime_libraries),
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
    assert (
        result["runtime_memory_receipt"]
        == _complete_schema_v4_receipt(9)
    )
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
    assert result["build_runtime_kv_plugin"] == capture._file_identity(
        plugin,
        label="build runtime-KV plugin",
    )
    assert (
        result["qualification_provenance"][
            "build_runtime_kv_plugin_sha256"
        ]
        == capture._canonical_sha(result["build_runtime_kv_plugin"])
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
    expected_mapped_paths = {
        str(path.resolve())
        for path in (
            *runtime_libraries,
            repo / "artifacts" / "libtrtmc_core.so",
            repo / "artifacts" / "libtrtmc_backend_trt.so",
            plugin,
            (
                repo
                / "artifacts"
                / "models/qwen/libtrtmc_model_qwen.so"
            ),
        )
    }
    assert {
        row["path"] for row in result["mapped_dso_identities"]
    } == expected_mapped_paths
    assert result["qualification_evidence"]["mapped_dso_identities"] == (
        result["mapped_dso_identities"]
    )
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
    bundle, build_receipt = _run_fresh_build(
        repo, worker_source=_benchmark_worker_source(10)
    )
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
    plugin = repo / "artifacts" / capture.RUNTIME_KV_PLUGIN_DSO
    worker = repo / "artifacts" / "trtmc_benchmark_worker"
    monkeypatch.setenv(
        "CUDA_CACHE_PATH", str(repo / "artifacts" / "cuda-cache")
    )
    monkeypatch.setattr(capture, "_engine_accounting", lambda *_: _accounting())
    runtime_libraries = _fake_runtime_libraries(repo)
    monkeypatch.setattr(
        capture,
        "_mapped_library_records",
        lambda _pid: _worker_mapping_records(repo, runtime_libraries),
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
        (_mapping_record(nvrtc), _mapping_record(builtins)),
        artifact_role="native-dynamic",
        runtime_stack=stack,
        mapped_dso_identities=_mapped_identities(nvrtc, builtins),
    )
    assert result is not None
    assert result["directory"] == str(nvrtc.parent)
    assert result["nvrtc"]["sha256"] == capture._sha256(nvrtc)
    assert result["nvrtc_builtins"]["sha256"] == capture._sha256(builtins)

    duplicate = nvrtc.parent / "libnvrtc.so.13.3.99"
    duplicate.write_bytes(b"other")
    with pytest.raises(capture.CaptureError, match="exactly one nvrtc"):
        capture._runtime_library_provenance(
            tuple(
                _mapping_record(path)
                for path in (nvrtc, duplicate, builtins)
            ),
            artifact_role="native-dynamic",
            runtime_stack=stack,
            mapped_dso_identities=_mapped_identities(
                nvrtc,
                duplicate,
                builtins,
            ),
        )

    other_directory = tmp_path / "other"
    other_directory.mkdir()
    other_builtins = other_directory / builtins.name
    other_builtins.write_bytes(b"other-builtins")
    with pytest.raises(capture.CaptureError, match="one directory"):
        capture._runtime_library_provenance(
            (_mapping_record(nvrtc), _mapping_record(other_builtins)),
            artifact_role="native-dynamic",
            runtime_stack=stack,
            mapped_dso_identities=_mapped_identities(
                nvrtc,
                other_builtins,
            ),
        )


def test_static_runtime_library_provenance_is_not_required() -> None:
    assert (
        capture._runtime_library_provenance(
            (),
            artifact_role="exact-head-static-split",
            runtime_stack=None,
            mapped_dso_identities=[],
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
