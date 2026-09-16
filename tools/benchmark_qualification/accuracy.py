# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal Accuracy qualification through the installed user benchmark."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from .catalog import QualificationCase, QualificationError, load_benchmark
from .datasets import Dataset, resolve_dataset
from .runtime import (
    RuntimeContext,
    prepare_bundle,
    reference_python,
    require_candidate,
    run_command,
    write_model_descriptor,
    write_result,
)


def run_accuracy(case: QualificationCase, context: RuntimeContext) -> dict[str, Any]:
    output = context.case_artifacts(case)
    output.mkdir(parents=True, exist_ok=True)
    definition = load_benchmark(context.repository, case)
    dataset = resolve_dataset(definition, context)
    metric = definition.get("metric")
    metric_name = metric.get("name") if isinstance(metric, Mapping) else None
    if metric_name == "exact_token_ids":
        result = _mmlu(case, context, dataset, output)
    elif metric_name == "forecast_tensor_parity":
        result = _etth1(case, context, definition, dataset, output)
    else:
        raise QualificationError(f"unsupported Accuracy metric {metric_name!r}")
    write_result(output, result)
    return result


def _mmlu(
    case: QualificationCase,
    context: RuntimeContext,
    dataset: Dataset,
    output: Path,
) -> dict[str, Any]:
    configured = case.values
    sample_limit = _positive_int(configured.get("samples"), "accuracy.samples")
    payload = json.loads(dataset.path.read_text(encoding="utf-8"))
    requests = payload.get("requests") if isinstance(payload, Mapping) else None
    if not isinstance(requests, list) or not requests:
        raise QualificationError("MMLU dataset must contain a non-empty requests list")
    selected = []
    for index, request in enumerate(requests[:sample_limit]):
        if not isinstance(request, Mapping):
            raise QualificationError(f"MMLU request {index} must be an object")
        selected.append(
            {
                "sample_id": str(request.get("id") or request.get("sample_id") or f"mmlu-{index}"),
                "prompt": _prompt(request),
            }
        )
    reference = configured.get("reference", {})
    candidate_request = configured.get("request", {})
    if not isinstance(reference, Mapping) or not isinstance(candidate_request, Mapping):
        raise QualificationError("MMLU reference and request must be objects")
    reference_request = {
        "model": str(case.candidate["checkpoint"]),
        "revision": case.candidate.get("revision"),
        "precision": str(reference.get("precision", "fp32")),
        "prompt_token_limit": int(configured.get("prompt_token_limit", 192)),
        "truncation_side": str(configured.get("truncation_side", "left")),
        "generation": dict(candidate_request),
        "samples": selected,
    }
    request_path = output / "reference-request.json"
    reference_path = output / "reference.json"
    _json(request_path, reference_request)
    runner = context.repository / "tools/benchmark_qualification/references/hf_text_generation.py"
    completed = run_command(
        [
            str(reference_python(case, context)),
            str(runner),
            "--request",
            str(request_path),
            "--output",
            str(reference_path),
        ],
        output,
        "reference",
        timeout=3600,
        verbose=context.verbose,
    )
    if completed.returncode != 0:
        raise QualificationError(f"HF Accuracy reference failed; see {output}")
    reference_samples = json.loads(reference_path.read_text(encoding="utf-8")).get("samples")
    if not isinstance(reference_samples, list) or len(reference_samples) != len(selected):
        raise QualificationError("HF Accuracy reference returned an invalid sample set")

    candidate_requests = [
        {
            "sample_id": sample["sample_id"],
            "request": {**dict(candidate_request), "prompt": sample["prompt"]},
        }
        for sample in reference_samples
    ]
    candidate, bundle = _candidate_outputs(case, context, output, "generate", candidate_requests)
    rows = []
    for actual, expected in zip(candidate, reference_samples, strict=True):
        actual_ids = actual.get("token_ids")
        expected_ids = expected.get("token_ids")
        matched = actual_ids == expected_ids and isinstance(actual_ids, list) and bool(actual_ids)
        rows.append(
            {
                "sample_id": expected["sample_id"],
                "passed": matched,
                "candidate_token_ids": actual_ids,
                "reference_token_ids": expected_ids,
            }
        )
    gates = configured.get("gate", {})
    if not isinstance(gates, Mapping):
        raise QualificationError("MMLU gate must be an object")
    minimum_rate = float(gates.get("min_pass_rate", 1.0))
    allowed_failures = int(gates.get("allowed_failures", 0))
    passed_count = sum(bool(row["passed"]) for row in rows)
    failed_count = len(rows) - passed_count
    pass_rate = passed_count / len(rows)
    passed = pass_rate >= minimum_rate and failed_count <= allowed_failures
    return {
        "schema_version": "trtmc.qualification-result/v1",
        "case": case.id,
        "kind": "accuracy",
        "status": "passed" if passed else "failed",
        "model": case.model,
        "benchmark": case.benchmark,
        "bundle": str(bundle),
        "dataset": dataset.receipt(),
        "metrics": {
            "samples": len(rows),
            "passed_samples": passed_count,
            "failed_samples": failed_count,
            "continuation_token_pass_rate": pass_rate,
        },
        "gate": {"min_pass_rate": minimum_rate, "allowed_failures": allowed_failures},
        "samples": rows,
    }


def _etth1(
    case: QualificationCase,
    context: RuntimeContext,
    definition: Mapping[str, Any],
    dataset: Dataset,
    output: Path,
) -> dict[str, Any]:
    configured = case.values
    window = configured.get("window")
    reference = configured.get("reference")
    if not isinstance(window, Mapping) or not isinstance(reference, Mapping):
        raise QualificationError("ETTh1 window and reference must be objects")
    samples = _etth1_windows(dataset.path, definition, window, int(configured.get("samples", 10)))
    runner_value = reference.get("command")
    if not isinstance(runner_value, str) or not runner_value:
        raise QualificationError("ETTh1 reference.command must be set")
    runner = (case.source.parent / runner_value).resolve()
    if not runner.is_file():
        raise QualificationError(f"ETTh1 reference runner does not exist: {runner}")
    request = {
        "model": str(case.candidate["checkpoint"]),
        "revision": case.candidate.get("revision"),
        "precision": str(reference.get("precision", "fp32")),
        "samples": samples,
    }
    request_path = output / "reference-request.json"
    reference_path = output / "reference.json"
    _json(request_path, request)
    completed = run_command(
        [
            str(reference_python(case, context)),
            str(runner),
            "--request",
            str(request_path),
            "--output",
            str(reference_path),
        ],
        output,
        "reference",
        timeout=3600,
        verbose=context.verbose,
    )
    if completed.returncode != 0:
        raise QualificationError(f"ETTh1 Accuracy reference failed; see {output}")
    expected = json.loads(reference_path.read_text(encoding="utf-8")).get("samples")
    if not isinstance(expected, list) or len(expected) != len(samples):
        raise QualificationError("ETTh1 reference returned an invalid sample set")
    candidate_requests = [
        {
            "sample_id": sample["sample_id"],
            "request": {
                "past_values": [float(value) for value in sample["past_values"]],
                "observed_mask": [1.0] * len(sample["past_values"]),
                "frequency": int(sample.get("frequency", 0)),
            },
        }
        for sample in samples
    ]
    actual, bundle = _candidate_outputs(case, context, output, "solve", candidate_requests)
    gate = configured.get("gate", {})
    if not isinstance(gate, Mapping):
        raise QualificationError("ETTh1 gate must be an object")
    relative_limit = float(gate["max_relative_l2"])
    absolute_limit = float(gate["max_absolute_error"])
    rows = []
    for left, right in zip(actual, expected, strict=True):
        left_values = [float(value) for value in left["values"]]
        right_values = [float(value) for value in right["values"]]
        shape_match = left.get("shape") == right.get("shape") and len(left_values) == len(
            right_values
        )
        if shape_match:
            difference = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(left_values, right_values, strict=True))
            )
            norm = math.sqrt(sum(value**2 for value in right_values))
            relative_l2 = difference / max(norm, 1.0e-12)
            maximum = max(abs(a - b) for a, b in zip(left_values, right_values, strict=True))
        else:
            relative_l2 = math.inf
            maximum = math.inf
        rows.append(
            {
                "sample_id": right["sample_id"],
                "passed": shape_match
                and relative_l2 <= relative_limit
                and maximum <= absolute_limit,
                "shape_match": shape_match,
                "relative_l2": relative_l2,
                "max_absolute_error": maximum,
            }
        )
    passed = all(row["passed"] for row in rows)
    return {
        "schema_version": "trtmc.qualification-result/v1",
        "case": case.id,
        "kind": "accuracy",
        "status": "passed" if passed else "failed",
        "model": case.model,
        "benchmark": case.benchmark,
        "bundle": str(bundle),
        "dataset": dataset.receipt(),
        "metrics": {
            "samples": len(rows),
            "failed_samples": sum(not row["passed"] for row in rows),
            "max_relative_l2": max(row["relative_l2"] for row in rows),
            "max_absolute_error": max(row["max_absolute_error"] for row in rows),
        },
        "gate": {
            "max_relative_l2": relative_limit,
            "max_absolute_error": absolute_limit,
        },
        "samples": rows,
    }


def _candidate_outputs(
    case: QualificationCase,
    context: RuntimeContext,
    output: Path,
    operation: str,
    requests: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Path]:
    if not requests:
        raise QualificationError("Accuracy candidate has no requests")
    worker, runtime_root = require_candidate(context)
    first_request = requests[0].get("request")
    if not isinstance(first_request, Mapping):
        raise QualificationError("Accuracy candidate request must be an object")
    descriptor = write_model_descriptor(case, output, first_request)
    bundle = prepare_bundle(case, context, output, descriptor)
    data_path = output / "candidate-inputs.json"
    _json(data_path, list(requests))
    candidate_output = output / "candidate-run"
    command = [
        str(context.trtmc_bench),
        "run",
        "--model",
        str(descriptor),
        "--case",
        case.name,
        "--operation",
        operation,
        "--bundle",
        str(bundle),
        "--data",
        str(data_path),
        "--runtime-root",
        str(runtime_root),
        "--worker",
        str(worker),
        "--warmup",
        "0",
        "--iterations",
        "1",
        "--telemetry",
        "off",
        "--no-build",
        "--output",
        str(candidate_output),
    ]
    completed = run_command(
        command,
        output,
        "candidate",
        timeout=7200,
        verbose=context.verbose,
    )
    if completed.returncode != 0:
        raise QualificationError(f"TRTMC Accuracy candidate failed; see {output}")
    result = json.loads((candidate_output / "result.json").read_text(encoding="utf-8"))
    cells = result.get("cells")
    if not isinstance(cells, list) or len(cells) != len(requests):
        raise QualificationError("trtmc-bench returned an invalid Accuracy result set")
    outputs = []
    for cell in cells:
        if not isinstance(cell, Mapping) or cell.get("status") != "completed":
            raise QualificationError("trtmc-bench reported a failed Accuracy request")
        summary = cell.get("output_summary")
        if not isinstance(summary, Mapping):
            raise QualificationError("trtmc-bench omitted an Accuracy output")
        outputs.append(dict(summary))
    return outputs, bundle


def _etth1_windows(
    path: Path,
    definition: Mapping[str, Any],
    configured: Mapping[str, Any],
    count: int,
) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    columns = configured.get("columns", ["OT"])
    if not isinstance(columns, list) or not columns:
        raise QualificationError("ETTh1 window.columns must be a non-empty list")
    context = int(configured.get("context_length", 512))
    prediction = int(configured.get("prediction_length", 64))
    start = int(configured.get("test_target_start", 11520))
    end = int(configured.get("test_end", 14400))
    stride = int(configured.get("stride", 24))
    starts = list(range(start - context, end - context - prediction + 1, stride))
    seed = int(definition.get("selection", {}).get("seed", 20260715))
    random.Random(seed).shuffle(starts)
    if count < 1 or len(starts) < count or len(rows) < end:
        raise QualificationError("ETTh1 dataset cannot satisfy the configured windows")
    return [
        {
            "sample_id": f"etth1-{index:04d}",
            "past_values": [
                float(row[column])
                for row in rows[window_start : window_start + context]
                for column in columns
            ],
            "frequency": int(configured.get("frequency", 0)),
        }
        for index, window_start in enumerate(starts[:count])
    ]


def _prompt(request: Mapping[str, Any]) -> str:
    messages = request.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if (
                isinstance(message, Mapping)
                and message.get("role") == "user"
                and isinstance(message.get("content"), str)
                and message["content"].strip()
            ):
                return str(message["content"])
    prompt = request.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    raise QualificationError("MMLU request has neither a prompt nor a user message")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise QualificationError(f"{name} must be a positive integer")
    return value


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
