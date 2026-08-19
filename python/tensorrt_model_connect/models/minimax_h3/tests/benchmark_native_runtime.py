# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark warm MiniMax-H3 requests and prove the observed residency boundary.

The shared benchmark worker keeps one ``IPipeline`` alive for warmup and measured
requests.  This harness parses H3 cache-hit markers instead of assuming that
TensorRT modules are resident.  Request-local modules yield per-request engine
timings; resident modules yield only a variant aggregate because the shared
runtime has no per-request timing snapshot API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from tensorrt_model_connect.models.minimax_h3.provenance import (
    CHECKPOINT_REVISION,
    atomic_write_json,
    file_identity,
    stable_file_record,
    validate_file_identity,
    validate_native_bundle_config,
    validate_source_revision,
)


STAGES = ("text_encoder", "adaln", "denoiser", "vae_decoder")
PERF_PATTERN = re.compile(
    r"\[minimax-h3\.perf\] text_encoder_ms=(?P<text_encoder>[0-9.]+) "
    r"adaln_ms=(?P<adaln>[0-9.]+) denoiser_ms=(?P<denoiser>[0-9.]+) "
    r"vae_decoder_ms=(?P<vae_decoder>[0-9.]+) total_ms=(?P<total>[0-9.]+)"
    r"(?P<annotations>[^\n]*)"
)
ENGINE_PATTERN = re.compile(
    r'\[trtmc\.engine_timing\] label="[^"]*" execute_ms=(?P<execute>[0-9.]+) '
    r"launches=(?P<launches>[0-9]+)"
)
BACKEND_PATTERN = re.compile(
    r"\[trtmc\] Backend loaded: [^\n]* \((?P<dso>libtrtmc_backend_[^)]+\.so)\)"
)
CUDA_GRAPH_FAILURE = "[cuda_graph] Capture failed"
ANNOTATION_PATTERN = re.compile(r"(?P<name>[a-z][a-z0-9_]*)=(?P<value>[^ ]+)")
CACHE_THRESHOLD_CONFIG_KEY = "minimax_h3.first_block_cache_threshold"


def evict_file_pages(path: Path) -> dict[str, bool | str]:
    """Best-effort eviction of clean cache pages for one file only."""

    posix_fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if posix_fadvise is None or dontneed is None:
        return {"supported": False, "attempted": False, "succeeded": False}

    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            posix_fadvise(descriptor, 0, 0, dontneed)
        finally:
            os.close(descriptor)
    except OSError as error:
        return {
            "supported": True,
            "attempted": True,
            "succeeded": False,
            "error": f"{type(error).__name__}: {error}",
        }
    return {"supported": True, "attempted": True, "succeeded": True}


def resolve_trt_backend_dso(executable: Path, bundle_config: dict) -> Path:
    """Resolve the exact adjacent backend DSO selected by the runtime loader."""

    if bundle_config.get("engine_backend") != "trt":
        raise ValueError("MiniMax-H3 native evidence requires engine_backend=trt")
    abi = bundle_config.get("trt_abi")
    match = re.fullmatch(r"(?P<major>[0-9]+)\.(?P<minor>[0-9]+)", str(abi))
    if match is None:
        raise ValueError("MiniMax-H3 bundle config has an invalid TensorRT ABI")
    names = (
        f"libtrtmc_backend_trt_{match.group('major')}_{match.group('minor')}.so",
        "libtrtmc_backend_trt.so",
    )
    for name in names:
        candidate = executable.parent / name
        if candidate.is_file():
            return candidate.resolve(strict=True)
    raise FileNotFoundError(
        "MiniMax-H3 could not bind the adjacent TensorRT backend DSO: "
        + ", ".join(str(executable.parent / name) for name in names)
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("MiniMax-H3 benchmark cannot summarize an empty sample set")
    return float(statistics.median(values))


def _summary(requests: list[dict[str, Any]]) -> dict[str, float | int]:
    summary: dict[str, float | int] = {
        "samples": len(requests),
        "median_public_pipeline_call_wall_ms": _median(
            [item["public_pipeline_call_wall_ms"] for item in requests]
        ),
        "median_pipeline_total_ms": _median([item["pipeline_total_ms"] for item in requests]),
        "median_host_prepost_and_output_assembly_ms": _median(
            [item["host_prepost_and_output_assembly_ms"] for item in requests]
        ),
    }
    if all("engine_execute" in item for item in requests):
        summary["median_engine_execute_ms"] = _median(
            [item["engine_execute"]["total_ms"] for item in requests]
        )
        summary["median_stage_non_engine_ms"] = _median(
            [item["stage_non_engine"]["total_ms"] for item in requests]
        )
    return summary


def _annotation_value(name: str, raw: str) -> bool | int | float | str:
    if name.endswith("_hit") and raw in {"0", "1"}:
        return raw == "1"
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def _perf_annotations(perf: dict[str, str]) -> dict[str, bool | int | float | str]:
    return {
        match.group("name"): _annotation_value(match.group("name"), match.group("value"))
        for match in ANNOTATION_PATTERN.finditer(perf.get("annotations", ""))
    }


def _request_timing(
    perf: dict[str, str],
    engine_group: list[dict[str, str]] | None,
    *,
    iteration: int,
    public_wall_ms: float | None,
) -> dict[str, Any]:
    stage_wall = {stage + "_ms": float(perf[stage]) for stage in STAGES}
    stage_wall["total_ms"] = sum(stage_wall.values())
    pipeline_total_ms = float(perf["total"])
    result: dict[str, Any] = {
        "iteration": iteration,
        "pipeline_total_ms": pipeline_total_ms,
        "stage_wall": stage_wall,
        "runtime_annotations": _perf_annotations(perf),
        # This is deliberately not called output-assembly time: it also contains
        # tokenization, RNG, patch/unpatch, scheduler, and other host work.
        "host_prepost_and_output_assembly_ms": pipeline_total_ms - stage_wall["total_ms"],
    }
    if engine_group is not None:
        engine_execute = {
            stage + "_ms": float(engine["execute"])
            for stage, engine in zip(STAGES, engine_group, strict=True)
        }
        engine_execute["total_ms"] = sum(engine_execute.values())
        result["engine_execute"] = engine_execute
        result["engine_launches"] = {
            stage: int(engine["launches"])
            for stage, engine in zip(STAGES, engine_group, strict=True)
        }
        stage_non_engine = {
            stage + "_ms": stage_wall[stage + "_ms"] - engine_execute[stage + "_ms"]
            for stage in STAGES
        }
        stage_non_engine["total_ms"] = sum(stage_non_engine.values())
        # Module creation/deserialization, transfers, and stage-local host work
        # cannot be split further with the current runtime timing surface.
        result["stage_non_engine"] = stage_non_engine
    if public_wall_ms is not None:
        result["public_pipeline_call_wall_ms"] = public_wall_ms
        result["worker_call_overhead_ms"] = public_wall_ms - pipeline_total_ms
    return result


def parse_worker_evidence(
    result: dict[str, Any],
    log_text: str,
    *,
    warmup: int,
    iterations: int,
    cuda_graphs: bool,
    cache_threshold: float | None = None,
    expected_backend_dso_name: str | None = None,
) -> dict[str, Any]:
    """Bind worker observations to H3 stage and TensorRT engine timings."""

    if result.get("status") != "completed":
        raise ValueError("MiniMax-H3 benchmark worker did not complete")
    if result.get("operation") != "generate_image":
        raise ValueError("MiniMax-H3 benchmark worker used the wrong operation")
    if result.get("warmup") != warmup or result.get("iterations") != iterations:
        raise ValueError("MiniMax-H3 benchmark worker changed the measurement count")
    observations = result.get("observations")
    if not isinstance(observations, list) or len(observations) != iterations:
        raise ValueError("MiniMax-H3 benchmark worker returned the wrong observation count")

    perf_matches = [match.groupdict() for match in PERF_PATTERN.finditer(log_text)]
    engine_matches = [match.groupdict() for match in ENGINE_PATTERN.finditer(log_text)]
    request_count = warmup + iterations
    if len(perf_matches) != request_count:
        raise ValueError(
            "MiniMax-H3 benchmark log has "
            f"{len(perf_matches)} pipeline timings for {request_count} requests"
        )
    if not engine_matches:
        raise ValueError("MiniMax-H3 benchmark log has no TensorRT engine timing evidence")
    loaded_backends = [match.group("dso") for match in BACKEND_PATTERN.finditer(log_text)]
    if expected_backend_dso_name is not None and loaded_backends != [expected_backend_dso_name]:
        raise ValueError(
            "MiniMax-H3 benchmark runtime did not load the provenance-bound TensorRT backend DSO"
        )
    capture_failures = log_text.count(CUDA_GRAPH_FAILURE)
    if cuda_graphs and capture_failures:
        raise ValueError(
            f"MiniMax-H3 CUDA graph capture failed {capture_failures} time(s); "
            "the requested variant is not a CUDA graph result"
        )

    per_request_engine_timings = len(engine_matches) == request_count * len(STAGES)
    all_requests = []
    for index, perf in enumerate(perf_matches):
        begin = index * len(STAGES)
        all_requests.append(
            _request_timing(
                perf,
                engine_matches[begin : begin + len(STAGES)] if per_request_engine_timings else None,
                iteration=index,
                public_wall_ms=None,
            )
        )

    measured_requests = []
    for iteration, (timing, observation) in enumerate(
        zip(all_requests[warmup:], observations, strict=True)
    ):
        if observation.get("iteration") != iteration:
            raise ValueError("MiniMax-H3 benchmark observation order is invalid")
        wall = observation.get("measured_wall_ms")
        if not isinstance(wall, (int, float)):
            raise ValueError("MiniMax-H3 benchmark observation has no measured wall time")
        item = dict(timing)
        item["iteration"] = iteration
        item["public_pipeline_call_wall_ms"] = float(wall)
        item["worker_call_overhead_ms"] = float(wall) - item["pipeline_total_ms"]
        measured_requests.append(item)

    first_request = all_requests[0]
    engine_timing_scope = "per_request"
    aggregate_engine = None
    if not per_request_engine_timings:
        engine_timing_scope = "variant_aggregate"
        aggregate_engine = {
            "entries": [
                {
                    "execute_ms": float(item["execute"]),
                    "launches": int(item["launches"]),
                }
                for item in engine_matches
            ],
            "total_ms": sum(float(item["execute"]) for item in engine_matches),
            "total_launches": sum(int(item["launches"]) for item in engine_matches),
            "requests_included": request_count,
            "per_request_values_available": False,
        }
    cache_fields = (
        "text_cache_hit",
        "adaln_cache_hit",
        "denoiser_resident_hit",
        "vae_resident_hit",
    )
    cache_markers_available = all(
        all(field in item["runtime_annotations"] for field in cache_fields) for item in all_requests
    )
    measured_cache_hits = (
        all(
            all(item["runtime_annotations"][field] is True for field in cache_fields)
            for item in all_requests[warmup:]
        )
        if cache_markers_available and warmup > 0
        else None
    )
    if cache_markers_available and warmup > 0 and not measured_cache_hits:
        raise ValueError("MiniMax-H3 measured warm requests did not hit all resident caches")
    if cache_threshold is not None:
        observed_thresholds = [
            item["runtime_annotations"].get("cache_threshold") for item in all_requests
        ]
        if any(not isinstance(value, (int, float)) for value in observed_thresholds):
            raise ValueError(
                "MiniMax-H3 benchmark requested a cache threshold but the runtime did not report it"
            )
        if any(
            not math.isclose(float(value), cache_threshold, rel_tol=0.0, abs_tol=1.0e-9)
            for value in observed_thresholds
        ):
            raise ValueError(
                "MiniMax-H3 runtime cache threshold does not match the requested override"
            )
    cold_start = {
        "pipeline_factory_load_ms": float(result["load_ms"]),
        "pipeline_factory_load_includes_plan_deserialization": False,
        "first_request": first_request,
        "plan_deserialization_separately_instrumented": False,
        "plan_deserialization_included_in": "first_request.stage_wall",
    }
    return {
        "cuda_graphs_requested": cuda_graphs,
        "cuda_graph_capture_failures_observed": capture_failures,
        "pipeline_factory_load_ms": float(result["load_ms"]),
        "pipeline_factory_load_includes_plan_deserialization": False,
        "warmup_requests": warmup,
        "measured_requests_are_pipeline_warm": warmup > 0,
        "engine_timing_scope": engine_timing_scope,
        "aggregate_engine_execute": aggregate_engine,
        "cache_markers_available": cache_markers_available,
        "measured_requests_hit_all_resident_caches": measured_cache_hits,
        "cache_threshold_override": cache_threshold,
        "loaded_backend_dso": loaded_backends[0] if loaded_backends else None,
        "cold_start": cold_start,
        "first_request": first_request,
        "requests": measured_requests,
        "summary": _summary(measured_requests),
        "output_summary": result.get("output_summary", {}),
    }


def build_worker_request(
    *,
    bundle: Path,
    plugin_dir: Path,
    prompt: str,
    seed: int,
    source_revision: str,
    cuda_graphs: bool,
    warmup: int,
    iterations: int,
    cache_threshold: float | None = None,
) -> dict[str, Any]:
    workload = {
        "prompt": prompt,
        "seed": seed,
        "height": 768,
        "width": 1344,
        "video_height": 768,
        "video_width": 1344,
        "video_num_frames": 124,
        "num_inference_steps": 50,
        "media_type": "video",
        "batch_size": 1,
    }
    runtime: dict[str, Any] = {
        "cuda_graphs": cuda_graphs,
        "model_plugin_search_paths": [str(plugin_dir)],
    }
    if cache_threshold is not None:
        runtime["config"] = {CACHE_THRESHOLD_CONFIG_KEY: cache_threshold}
    identity = {
        "source_revision": source_revision,
        "bundle_path": str(bundle),
        "workload": workload,
        "runtime": runtime,
        "measurement": {"warmup": warmup, "iterations": iterations},
    }
    return {
        "schema_version": 1,
        "case_name": f"minimax-h3-cuda-graphs-{'on' if cuda_graphs else 'off'}",
        "case_digest": _canonical_sha256(identity),
        "bundle": str(bundle),
        "operation": "generate_image",
        "request": workload,
        "runtime": runtime,
        "measurement": {
            "warmup": warmup,
            "iterations": iterations,
            "timing_scope": "public_pipeline_call_wall",
            "asset_loading_included": False,
        },
    }


def _worker_metadata(worker: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(worker), "--metadata"], capture_output=True, text=True, check=False, timeout=10
    )
    if completed.returncode:
        raise RuntimeError(f"MiniMax-H3 benchmark worker metadata failed: {completed.stderr}")
    try:
        metadata = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("MiniMax-H3 benchmark worker returned invalid metadata") from error
    if metadata.get("schema_version") != "trtmc.benchmark-worker-metadata/v1":
        raise RuntimeError("MiniMax-H3 benchmark worker metadata schema is unsupported")
    return metadata


def _run_variant(
    *,
    worker: Path,
    backend_dso_name: str,
    bundle: Path,
    plugin_dir: Path,
    prompt: str,
    seed: int,
    source_revision: str,
    cuda_graphs: bool,
    warmup: int,
    iterations: int,
    cache_threshold: float | None,
    timeout_s: int,
    output_dir: Path,
) -> dict[str, Any]:
    name = "cuda_graphs_on" if cuda_graphs else "cuda_graphs_off"
    variant_dir = output_dir / name
    variant_dir.mkdir()
    request = build_worker_request(
        bundle=bundle,
        plugin_dir=plugin_dir,
        prompt=prompt,
        seed=seed,
        source_revision=source_revision,
        cuda_graphs=cuda_graphs,
        warmup=warmup,
        iterations=iterations,
        cache_threshold=cache_threshold,
    )
    request_path = variant_dir / "worker_request.json"
    result_path = variant_dir / "worker_result.json"
    log_path = variant_dir / "worker.log"
    atomic_write_json(request_path, request)
    environment = os.environ.copy()
    environment.update(
        {
            "TRTMC_MODEL_PLUGIN_DIR": str(plugin_dir),
            "TRTMC_MODEL_PLUGIN_STRICT": "1",
            "WORLD_SIZE": "1",
            "RANK": "0",
        }
    )
    command = [str(worker), "--request", str(request_path), "--output", str(result_path)]
    bundle_page_cache_eviction = evict_file_pages(bundle)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout_s,
            env=environment,
        )
    process_wall_s = time.perf_counter() - started
    if not result_path.is_file():
        raise RuntimeError(
            f"MiniMax-H3 benchmark worker exited {completed.returncode} without a result; "
            f"see {log_path}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if completed.returncode or result.get("status") != "completed":
        raise RuntimeError(
            f"MiniMax-H3 benchmark worker failed: {result.get('error', completed.returncode)}; "
            f"see {log_path}"
        )
    if result.get("case_digest") != request["case_digest"]:
        raise RuntimeError("MiniMax-H3 benchmark worker returned the wrong case digest")
    evidence = parse_worker_evidence(
        result,
        log_path.read_text(encoding="utf-8", errors="replace"),
        warmup=warmup,
        iterations=iterations,
        cuda_graphs=cuda_graphs,
        cache_threshold=cache_threshold,
        expected_backend_dso_name=backend_dso_name,
    )
    evidence.update(
        {
            "process_wall_s": process_wall_s,
            "runtime_request": request["runtime"],
            "worker_request": stable_file_record(request_path, "worker request")[0],
            "worker_result": stable_file_record(result_path, "worker result")[0],
            "worker_log": stable_file_record(log_path, "worker log")[0],
            "bundle_page_cache_eviction": bundle_page_cache_eviction,
            "command": command,
        }
    )
    return evidence


def _variants(mode: str) -> tuple[bool, ...]:
    if mode == "off":
        return (False,)
    if mode == "on":
        return (True,)
    if mode == "both":
        return (False, True)
    raise ValueError(f"Unsupported CUDA graph mode: {mode}")


def _comparison(variants: list[dict[str, Any]]) -> dict[str, float] | None:
    by_mode = {item["cuda_graphs_requested"]: item for item in variants}
    if set(by_mode) != {False, True}:
        return None
    off = by_mode[False]["summary"]["median_public_pipeline_call_wall_ms"]
    on = by_mode[True]["summary"]["median_public_pipeline_call_wall_ms"]
    return {
        "cuda_graphs_off_median_ms": off,
        "cuda_graphs_on_median_ms": on,
        "cuda_graphs_on_minus_off_ms": on - off,
        "cuda_graphs_on_over_off": on / off,
    }


def _residency_contract(variants: list[dict[str, Any]]) -> dict[str, Any]:
    markers_available = all(item["cache_markers_available"] for item in variants)
    warm_hits = (
        all(item["measured_requests_hit_all_resident_caches"] is True for item in variants)
        if markers_available
        else None
    )
    return {
        "worker_process_reused_within_variant": True,
        "pipeline_instance_reused_across_requests": True,
        "cache_markers_available": markers_available,
        "measured_requests_hit_all_resident_caches": warm_hits,
        "engine_modules_reused_across_requests": warm_hits,
        "evidence": (
            "per-request text_cache_hit, adaln_cache_hit, denoiser_resident_hit, and "
            "vae_resident_hit markers"
            if markers_available
            else "runtime predates the explicit MiniMax-H3 cache-hit timing markers"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--plugin-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--cuda-graphs",
        nargs="?",
        const="on",
        default="both",
        choices=("off", "on", "both"),
        help="benchmark CUDA graphs off, on, or both (bare flag means on)",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument(
        "--cache-threshold",
        type=float,
        default=None,
        help=f"override runtime.config.{CACHE_THRESHOLD_CONFIG_KEY}",
    )
    parser.add_argument("--timeout-s", type=int, default=7200)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.timeout_s < 1:
        raise ValueError("warmup must be non-negative; iterations and timeout must be positive")
    if args.cache_threshold is not None and (
        not math.isfinite(args.cache_threshold) or args.cache_threshold <= 0.0
    ):
        raise ValueError("cache threshold must be a positive finite number")

    source_revision = validate_source_revision(args.source_revision)
    bundle = Path(args.bundle).resolve(strict=True)
    bundle_identity = file_identity(bundle)
    prompt_path = Path(args.prompt_file).resolve(strict=True)
    worker = Path(args.worker).resolve(strict=True)
    plugin_dir = Path(args.plugin_dir).resolve(strict=True)
    plugin = (plugin_dir / "libtrtmc_model_minimax_h3.so").resolve(strict=True)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"MiniMax-H3 benchmark output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    prompt_spec = json.loads(prompt_path.read_text(encoding="utf-8"))
    if not isinstance(prompt_spec.get("prompt"), str) or not prompt_spec["prompt"]:
        raise ValueError("MiniMax-H3 benchmark prompt must be a non-empty string")
    if not isinstance(prompt_spec.get("seed"), int) or isinstance(prompt_spec["seed"], bool):
        raise ValueError("MiniMax-H3 benchmark seed must be an integer")

    # Read the bundle's own pinned source first so already-qualified engines can
    # be benchmarked by a newer harness without misattributing their build.
    from tensorrt_model_connect.models.minimax_h3.provenance import load_bundle_config

    unvalidated_config = load_bundle_config(bundle)
    bundle_source_revision = validate_source_revision(unvalidated_config.get("source_revision", ""))
    bundle_config = validate_native_bundle_config(bundle, source_revision=bundle_source_revision)
    backend = resolve_trt_backend_dso(worker, bundle_config)
    prompt_record, prompt_identity = stable_file_record(prompt_path, "prompt file")
    worker_record, worker_identity = stable_file_record(worker, "benchmark worker")
    backend_record, backend_identity = stable_file_record(backend, "TensorRT backend")
    plugin_record, plugin_identity = stable_file_record(plugin, "MiniMax-H3 model plugin")
    script_path = Path(__file__).resolve()
    script_record, script_identity = stable_file_record(script_path, "benchmark harness")
    metadata = _worker_metadata(worker)

    variant_receipts = [
        _run_variant(
            worker=worker,
            backend_dso_name=backend.name,
            bundle=bundle,
            plugin_dir=plugin_dir,
            prompt=prompt_spec["prompt"],
            seed=int(prompt_spec["seed"]),
            source_revision=source_revision,
            cuda_graphs=enabled,
            warmup=args.warmup,
            iterations=args.iterations,
            cache_threshold=args.cache_threshold,
            timeout_s=args.timeout_s,
            output_dir=output_dir,
        )
        for enabled in _variants(args.cuda_graphs)
    ]

    validate_file_identity(bundle, bundle_identity, "native bundle")
    validate_file_identity(prompt_path, prompt_identity, "prompt file")
    validate_file_identity(worker, worker_identity, "benchmark worker")
    validate_file_identity(backend, backend_identity, "TensorRT backend")
    validate_file_identity(plugin, plugin_identity, "MiniMax-H3 model plugin")
    validate_file_identity(script_path, script_identity, "benchmark harness")
    receipt = {
        "schema_version": "trtmc.minimax-h3-native-runtime-benchmark/v1",
        "status": "passed",
        "backend": "tensorrt_native_single_device",
        "source_revision": source_revision,
        "bundle_source_revision": bundle_source_revision,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_inventory_sha256": bundle_config["checkpoint_inventory_sha256"],
        "builder_source_sha256": bundle_config["builder_source_sha256"],
        "plan_sha256": bundle_config["plan_sha256"],
        "workspace_limit_bytes": bundle_config["workspace_limit_bytes"],
        "inputs": {
            "bundle": {"path": str(bundle), **bundle_identity},
            "prompt_file": prompt_record,
            "worker": worker_record,
            "trt_backend": backend_record,
            "model_plugin": plugin_record,
            "benchmark_harness": script_record,
        },
        "worker_metadata": metadata,
        "workload": {
            "prompt": prompt_spec["prompt"],
            "seed": int(prompt_spec["seed"]),
            "height": 768,
            "width": 1344,
            "num_frames": 124,
            "num_inference_steps": 50,
            "batch_size": 1,
        },
        "measurement": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "timing_scope": "public_pipeline_call_wall",
            "variant_process_isolation": True,
            "variant_order": [item["cuda_graphs_requested"] for item in variant_receipts],
            "cache_threshold_override": args.cache_threshold,
        },
        "residency_contract": _residency_contract(variant_receipts),
        "timing_boundaries": {
            "pipeline_factory_load": (
                "creates the pipeline/tokenizer but does not deserialize the four lazy plans"
            ),
            "stage_wall": (
                "includes plan read/deserialization, engine execution, transfers, and "
                "stage-local host work"
            ),
            "engine_execute": "CUDA event time around TensorRT enqueue or CUDA graph launch",
            "stage_non_engine": (
                "available only for request-local engines; resident engine timing is emitted "
                "only as a whole-variant aggregate and is never divided into synthetic samples"
            ),
            "output_assembly": {
                "separately_instrumented": False,
                "included_in": "host_prepost_and_output_assembly_ms",
                "reason": (
                    "the current H3 pipeline exposes only four stage walls and total request wall; "
                    "exact output assembly requires pipeline/runtime instrumentation"
                ),
            },
            "png_encoding": "not executed by the benchmark worker",
        },
        "host": {"hostname": platform.node(), "platform": platform.platform()},
        "world_size": 1,
        "collective_transport": "none",
        "variants": variant_receipts,
        "comparison": _comparison(variant_receipts),
    }
    atomic_write_json(output_dir / "benchmark_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
