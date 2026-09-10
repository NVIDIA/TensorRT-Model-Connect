#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chronos-Bolt official PyTorch reference for Accuracy and Performance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Callable, Mapping, Sequence


REQUEST_SCHEMA = "trtmc.chronos-bolt-reference-request/v1"
ACCURACY_RESULT_SCHEMA = "trtmc.chronos-bolt-reference-result/v1"
PERFORMANCE_RESULT_SCHEMA = "trtmc.perf-baseline/v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"reference request must contain an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _device(torch_module: Any, requested: str) -> str:
    if requested == "auto":
        requested = "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("Chronos-Bolt reference requires an available CUDA device")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("reference device must be auto, cpu, or cuda")
    return requested


def _pipeline(request: Mapping[str, Any], device: str) -> Any:
    import torch
    from chronos import ChronosBoltPipeline

    if request.get("precision") != "fp32":
        raise ValueError("Chronos-Bolt qualification currently requires fp32 reference")
    return ChronosBoltPipeline.from_pretrained(
        str(request["checkpoint"]),
        device_map=device,
        dtype=torch.float32,
    )


def _predict(pipeline: Any, values: Sequence[float], device: str) -> Any:
    import torch

    context = torch.tensor([float(value) for value in values], dtype=torch.float32, device=device)
    with torch.inference_mode():
        return pipeline.predict(
            context,
            prediction_length=pipeline.model_prediction_length,
            limit_prediction_length=True,
        )


def _tensor_result(value: Any) -> dict[str, Any]:
    result = value.detach().float().cpu()
    return {
        "values": result.reshape(-1).tolist(),
        "shape": list(result.shape),
        "element_count": result.numel(),
    }


def _accuracy(request: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    device = _device(torch, str(request.get("device", "auto")))
    pipeline = _pipeline(request, device)
    raw_samples = request.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("Accuracy reference requires non-empty samples")
    samples = []
    for sample in raw_samples:
        if not isinstance(sample, Mapping):
            raise ValueError("Accuracy reference samples must be objects")
        values = sample.get("past_values")
        if not isinstance(values, list) or not values:
            raise ValueError("Accuracy reference sample requires past_values")
        samples.append(
            {
                "sample_id": str(sample["sample_id"]),
                **_tensor_result(_predict(pipeline, values, device)),
            }
        )
    return {
        "schema_version": ACCURACY_RESULT_SCHEMA,
        "backend": "official_pytorch",
        "model": request["model"],
        "revision": request["revision"],
        "precision": request["precision"],
        "device": device,
        "samples": samples,
    }


def _compile(pipeline: Any) -> dict[str, Any]:
    import torch
    from torch._dynamo.backends.registry import lookup_backend

    evidence = {"compiled_graph_count": 0}
    inductor = lookup_backend("inductor")

    def compile_graph(graph: Any, inputs: Any, **options: Any) -> Any:
        compiled = inductor(graph, inputs, **options)
        evidence["compiled_graph_count"] += 1
        return compiled

    # Each Performance case measures one fixed input shape. Specialization also
    # avoids Inductor's symbolic divisibility failure in Chronos patch padding.
    pipeline.model.forward = torch.compile(
        pipeline.model.forward,
        backend=compile_graph,
        fullgraph=False,
        dynamic=False,
    )
    evidence.update(
        {
            "api": "torch.compile",
            "target": "model.forward",
            "backend": "inductor",
            "mode": "default",
            "fullgraph": False,
            "dynamic": False,
            "applied": True,
        }
    )
    return evidence


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": max(values),
    }


def _measure(
    invoke: Callable[[], Any], warmup: int, iterations: int, compile_evidence: Mapping[str, Any]
) -> tuple[list[float], Any]:
    import torch

    if warmup <= 0 or iterations <= 0:
        raise ValueError("warmup and iterations must be positive")
    last = None
    for _ in range(warmup):
        last = invoke()
    torch.cuda.synchronize()
    compiled_graphs = int(compile_evidence["compiled_graph_count"])
    if compiled_graphs < 1:
        raise RuntimeError("warmup did not execute any compiled graphs")
    samples = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        last = invoke()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    if last is None or not all(math.isfinite(value) and value > 0.0 for value in samples):
        raise RuntimeError("reference produced no finite positive timing observations")
    if int(compile_evidence["compiled_graph_count"]) != compiled_graphs:
        raise RuntimeError("model compilation occurred inside timed samples")
    return samples, last


def _environment(torch_module: Any, transformers_module: Any) -> dict[str, Any]:
    import chronos

    device = torch_module.cuda.current_device()
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "python_executable": os.sys.executable,
        "gpu_uuid": str(torch_module.cuda.get_device_properties(device).uuid),
        "torch": torch_module.__version__,
        "transformers": transformers_module.__version__,
        "chronos": chronos.__version__,
        "cuda": torch_module.version.cuda,
        "gpu": torch_module.cuda.get_device_name(device),
        "gpu_capability": list(torch_module.cuda.get_device_capability(device)),
    }


def _performance(request: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    import transformers

    if str(request.get("mode")) != "torch-compile":
        raise ValueError("Chronos-Bolt Performance reference requires torch.compile")
    if str(request.get("compile_scope")) != "model.forward":
        raise ValueError("Chronos-Bolt Performance must compile model.forward")
    device = _device(torch, "cuda")
    pipeline = _pipeline(request, device)
    compile_evidence = _compile(pipeline)
    values = request.get("past_values")
    if not isinstance(values, list) or not values:
        raise ValueError("Performance reference requires past_values")
    context = torch.tensor([float(value) for value in values], dtype=torch.float32, device=device)

    def invoke() -> Any:
        with torch.inference_mode():
            return pipeline.predict(
                context,
                prediction_length=pipeline.model_prediction_length,
                limit_prediction_length=True,
            )

    samples, output = _measure(
        invoke,
        int(request["measurement"]["warmup"]),
        int(request["measurement"]["iterations"]),
        compile_evidence,
    )
    compile_evidence["warmup_completed"] = True
    compile_evidence["timed_callable_uses_compiled_target"] = True
    output_summary = _tensor_result(output)
    elapsed_seconds = sum(samples) / 1000.0
    return {
        "schema_version": PERFORMANCE_RESULT_SCHEMA,
        "status": "completed",
        "backend": "chronos-bolt",
        "mode": "torch-compile",
        "compile_scope": "model.forward",
        "compile_evidence": compile_evidence,
        "model": request["model"],
        "revision": request["revision"],
        "case_name": request["case_name"],
        "task": "time_series_forecast",
        "precision": "fp32",
        "measurement_policy": {
            "timing_scope": "task_model_call_wall",
            "input_preparation_included": False,
            "model_load_excluded": True,
            "compile_excluded": True,
            "warmup_excluded": True,
        },
        "samples_ms": samples,
        "metrics": {
            "latency_ms": _latency_summary(samples),
            "sample_count": len(samples),
            "request_throughput_per_s": len(samples) / elapsed_seconds,
            "forecast_elements_per_s": len(samples)
            * int(output_summary["element_count"])
            / elapsed_seconds,
        },
        "output_summary": output_summary,
        "environment": _environment(torch, transformers),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    request = _read_json(arguments.request)
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError("unsupported Chronos-Bolt reference request")
    mode = request.get("purpose")
    if mode == "accuracy":
        result = _accuracy(request)
    elif mode == "performance":
        result = _performance(request)
    else:
        raise ValueError("reference purpose must be accuracy or performance")
    _write_json(arguments.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
