# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-agnostic execution and reporting for family-owned Performance references."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

from .references.hf_transformers import flatten_config
from .references.timing_contracts import timing_contract


@dataclass(frozen=True)
class Session:
    """A loaded reference and the repeatable operation to measure."""

    invoke: Callable[[], Mapping[str, Any]]
    backend: str
    timing_scope: str = "task-model-call-wall"
    input_preparation_included: bool = False
    asset_loading_included: bool = False
    materialize: Callable[[dict[str, Any], Path], None] | None = None


@dataclass(frozen=True)
class Measurement:
    """Already measured family output using the shared result contract."""

    samples_ms: Sequence[float]
    output_summary: Mapping[str, Any]
    backend: str
    timing_scope: str = "task-model-call-wall"
    input_preparation_included: bool = False
    asset_loading_included: bool = False


def parser(description: str) -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=description)
    value.add_argument("--family", required=True)
    value.add_argument("--operation", required=True)
    value.add_argument("--manifest", required=True, type=Path)
    value.add_argument("--adapter-options-json", default="{}")
    value.add_argument("--timing-contract-json", required=True)
    value.add_argument("--padding", default="longest")
    value.add_argument("--model", required=True)
    value.add_argument("--revision")
    value.add_argument("--request-json", required=True)
    value.add_argument("--precision", required=True, choices=("fp16", "bf16", "fp32"))
    value.add_argument("--mode", required=True)
    value.add_argument("--warmup", required=True, type=int)
    value.add_argument("--iterations", required=True, type=int)
    value.add_argument("--case-name", required=True)
    value.add_argument("--testcase-name")
    value.add_argument("--output", required=True, type=Path)
    value.add_argument("--selected-task", required=True)
    value.add_argument("--trust-remote-code", action="store_true")
    value.add_argument("--local-files-only", action="store_true")
    return value


def json_object(raw: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def synchronize() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def measure(session: Session, warmup: int, iterations: int) -> tuple[list[float], dict[str, Any]]:
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    output: Mapping[str, Any] = {}
    for _ in range(warmup):
        output = session.invoke()
        synchronize()
    samples = []
    for _ in range(iterations):
        synchronize()
        started = time.perf_counter()
        output = session.invoke()
        synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    if not all(math.isfinite(value) and value > 0.0 for value in samples):
        raise RuntimeError("reference produced an invalid timing sample")
    return samples, dict(output)


def environment() -> dict[str, Any]:
    value: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import torch

        value["torch"] = str(torch.__version__)
        if torch.cuda.is_available():
            value["gpu"] = torch.cuda.get_device_name(0)
            value["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return value


def run(
    argv: Sequence[str],
    *,
    description: str,
    load: Callable[[argparse.Namespace, Mapping[str, Any], Mapping[str, Any]], Session],
) -> int:
    arguments = parser(description).parse_args(argv)
    request = flatten_config(json_object(arguments.request_json, "--request-json"))
    options = json_object(arguments.adapter_options_json, "--adapter-options-json")
    load_started = time.perf_counter()
    session = load(arguments, request, options)
    load_seconds = time.perf_counter() - load_started
    samples, output_summary = measure(session, arguments.warmup, arguments.iterations)
    if session.materialize is not None:
        session.materialize(output_summary, arguments.output)
    return _write(
        arguments,
        Measurement(
            samples,
            output_summary,
            session.backend,
            session.timing_scope,
            session.input_preparation_included,
            session.asset_loading_included,
        ),
        load_seconds=load_seconds,
    )


def run_premeasured(
    argv: Sequence[str],
    *,
    description: str,
    execute: Callable[
        [argparse.Namespace, Mapping[str, Any], Mapping[str, Any]], Measurement
    ],
) -> int:
    arguments = parser(description).parse_args(argv)
    request = flatten_config(json_object(arguments.request_json, "--request-json"))
    options = json_object(arguments.adapter_options_json, "--adapter-options-json")
    measurement = execute(arguments, request, options)
    return _write(arguments, measurement, load_seconds=None)


def _write(
    arguments: argparse.Namespace,
    measurement: Measurement,
    *,
    load_seconds: float | None,
) -> int:
    samples = [float(value) for value in measurement.samples_ms]
    if len(samples) != arguments.iterations or not all(
        math.isfinite(value) and value > 0.0 for value in samples
    ):
        raise RuntimeError("reference must return one finite positive sample per iteration")
    actual = {
        "timing_scope": measurement.timing_scope,
        "input_preparation_included": measurement.input_preparation_included,
        "asset_loading_included": measurement.asset_loading_included,
    }
    declared = json_object(arguments.timing_contract_json, "--timing-contract-json")
    expected = timing_contract(runner="task-reference", declared=declared) if declared else None
    if expected is not None and any(actual[name] != expected[name] for name in actual):
        raise RuntimeError(f"family reference timing drifted: actual={actual}, declared={expected}")
    result = {
        "schema_version": "trtmc.perf-baseline/v1",
        "status": "completed",
        "backend": measurement.backend,
        "framework": measurement.backend,
        "family": arguments.family,
        "operation": arguments.operation,
        "selected_task": arguments.selected_task,
        "mode": arguments.mode,
        "precision": arguments.precision,
        "padding": arguments.padding,
        "model": arguments.model,
        "case_name": arguments.case_name,
        "model_load_included": False,
        "model_load_seconds": load_seconds,
        **actual,
        "measurement": {"warmup": arguments.warmup, "iterations": arguments.iterations},
        "measurement_policy": {
            **actual,
            "model_load_excluded": True,
            "compile_excluded": True,
            "warmup_excluded": True,
            "output_materialization_included": True,
        },
        "samples_ms": samples,
        "metrics": {
            "sample_count": len(samples),
            "latency_ms": {
                "p50": statistics.median(samples),
                "min": min(samples),
                "max": max(samples),
                "mean": statistics.fmean(samples),
            },
        },
        "output_summary": dict(measurement.output_summary),
        "environment": environment(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0
