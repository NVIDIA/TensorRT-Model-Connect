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
        result = _text_generation_parity(case, context, dataset, output)
    elif metric_name == "embedding_vector_parity":
        result = _encoder_embedding_parity(case, context, dataset, output)
    elif metric_name == "forecast_tensor_parity":
        result = _etth1(case, context, definition, dataset, output)
    else:
        raise QualificationError(f"unsupported Accuracy metric {metric_name!r}")
    write_result(output, result)
    return result


def _text_generation_parity(
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
        raise QualificationError("text-generation dataset must contain a non-empty requests list")
    selected = []
    for index, request in enumerate(requests[:sample_limit]):
        if not isinstance(request, Mapping):
            raise QualificationError(f"text-generation request {index} must be an object")
        selected.append(
            {
                "sample_id": str(request.get("id") or request.get("sample_id") or f"sample-{index}"),
                "prompt": _prompt(request),
            }
        )
    reference = configured.get("reference", {})
    candidate_request = configured.get("request", {})
    if not isinstance(reference, Mapping) or not isinstance(candidate_request, Mapping):
        raise QualificationError("Accuracy reference and request must be objects")
    reference_task = str(reference.get("task", "causal-lm"))
    if reference_task not in {"causal-lm", "seq2seq-lm"}:
        raise QualificationError(f"unsupported Accuracy reference task {reference_task!r}")
    output_token_policy = str(reference.get("output_token_policy", "new-tokens"))
    if output_token_policy not in {"new-tokens", "strip-start", "strip-start-and-eos"}:
        raise QualificationError(
            f"unsupported Accuracy output token policy {output_token_policy!r}"
        )
    reference_generation = dict(candidate_request)
    source_language_placement = reference.get("source_language_placement")
    if source_language_placement is not None:
        reference_generation["source_language_placement"] = source_language_placement
    reference_request = {
        "model": str(case.candidate["checkpoint"]),
        "revision": case.candidate.get("revision"),
        "trust_remote_code": bool(case.candidate.get("trust_remote_code", False)),
        "precision": str(reference.get("precision", "fp32")),
        "task": reference_task,
        "output_token_policy": output_token_policy,
        "prompt_token_limit": int(configured.get("prompt_token_limit", 192)),
        "truncation_side": str(configured.get("truncation_side", "left")),
        "generation": reference_generation,
        "samples": selected,
    }
    experts_implementation = reference.get("experts_implementation")
    if experts_implementation is not None:
        if not isinstance(experts_implementation, str) or not experts_implementation:
            raise QualificationError(
                "accuracy.reference.experts_implementation must be a non-empty string"
            )
        reference_request["experts_implementation"] = experts_implementation
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
        raise QualificationError("Accuracy gate must be an object")
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


def _encoder_embedding_parity(
    case: QualificationCase,
    context: RuntimeContext,
    dataset: Dataset,
    output: Path,
) -> dict[str, Any]:
    configured = case.values
    sample_limit = _positive_int(configured.get("samples"), "accuracy.samples")
    prompt_prefix = configured.get("prompt_prefix", "")
    if not isinstance(prompt_prefix, str):
        raise QualificationError("accuracy.prompt_prefix must be a string")
    samples = _sts_samples(dataset.path, sample_limit, prompt_prefix)
    reference = configured.get("reference", {})
    gates = configured.get("gate", {})
    if not isinstance(reference, Mapping) or not isinstance(gates, Mapping):
        raise QualificationError("encoder Accuracy reference and gate must be objects")
    task = str(case.candidate["task"])
    try:
        mode, operation = {
            "encoding": ("cls", "encode"),
            "embedding": ("embedding", "embed"),
        }[task]
    except KeyError as error:
        raise QualificationError(
            f"embedding vector parity does not support candidate task {task!r}"
        ) from error
    model_class = reference.get("model_class", "auto")
    tokenizer_class = reference.get("tokenizer_class", "auto")
    if not all(
        isinstance(value, str) and value
        for value in (model_class, tokenizer_class)
    ):
        raise QualificationError("encoder reference classes must be non-empty strings")
    reference_request = {
        "model": str(case.candidate["checkpoint"]),
        "revision": case.candidate.get("revision"),
        "trust_remote_code": bool(case.candidate.get("trust_remote_code", False)),
        "precision": str(reference.get("precision", "fp32")),
        "mode": mode,
        "model_class": model_class,
        "tokenizer_class": tokenizer_class,
        "max_length": int(case.candidate.get("build", {}).get("max_sequence_length", 512)),
        "samples": samples,
    }
    request_path = output / "reference-request.json"
    reference_path = output / "reference.json"
    output.mkdir(parents=True, exist_ok=True)
    _json(request_path, reference_request)
    runner = context.repository / "tools/benchmark_qualification/references/hf_encoder.py"
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
        raise QualificationError(f"HF encoder Accuracy reference failed; see {output}")
    reference_samples = json.loads(reference_path.read_text(encoding="utf-8")).get("samples")
    if not isinstance(reference_samples, list) or len(reference_samples) != len(samples):
        raise QualificationError("HF encoder Accuracy reference returned an invalid sample set")
    candidate_requests = [
        {
            "sample_id": sample["sample_id"],
            "request": {"prompt": sample["prompt"]},
        }
        for sample in samples
    ]
    candidate, bundle = _candidate_outputs(
        case, context, output, operation, candidate_requests
    )
    compared = _compare_encoder_embeddings(reference_samples, candidate, gates)
    return {
        "schema_version": "trtmc.qualification-result/v1",
        "case": case.id,
        "kind": "accuracy",
        "status": compared["status"],
        "model": case.model,
        "benchmark": case.benchmark,
        "bundle": str(bundle),
        "dataset": dataset.receipt(),
        "metrics": compared["metrics"],
        "gate": compared["gate"],
        "samples": compared["samples"],
        "pairs": compared["pairs"],
    }


def _sts_samples(path: Path, count: int, prefix: str) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for dataset_index, line in enumerate(stream):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise QualificationError(
                    f"STSBenchmark row {dataset_index} is not valid JSON"
                ) from error
            if not isinstance(row, Mapping):
                raise QualificationError(f"STSBenchmark row {dataset_index} must be an object")
            sentence1 = row.get("sentence1")
            sentence2 = row.get("sentence2")
            if not all(isinstance(value, str) and value.strip() for value in (sentence1, sentence2)):
                raise QualificationError(
                    f"STSBenchmark row {dataset_index} must contain two sentences"
                )
            try:
                score = float(row["score"])
            except (KeyError, TypeError, ValueError) as error:
                raise QualificationError(
                    f"STSBenchmark row {dataset_index} has an invalid score"
                ) from error
            if not math.isfinite(score):
                raise QualificationError(
                    f"STSBenchmark row {dataset_index} has a non-finite score"
                )
            pair_id = f"stsbenchmark-{dataset_index:06d}"
            for suffix, side, sentence in (
                ("a", "sentence1", sentence1),
                ("b", "sentence2", sentence2),
            ):
                pairs.append(
                    {
                        "sample_id": f"{pair_id}-{suffix}",
                        "pair_id": pair_id,
                        "pair_side": side,
                        "score": score,
                        "prompt": prefix + str(sentence).strip(),
                    }
                )
            if len(pairs) == count * 2:
                break
    if len(pairs) != count * 2:
        raise QualificationError(
            f"STSBenchmark contains {len(pairs) // 2} usable pairs; {count} required"
        )
    return pairs


def _compare_encoder_embeddings(
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    if len(reference) != len(candidate) or not reference:
        raise QualificationError("encoder reference and candidate sample counts differ")
    minimum_cosine = float(gates.get("min_vector_cosine", 0.999))
    minimum_rate = float(gates.get("min_vector_pass_rate", 1.0))
    maximum_pair_delta = float(gates.get("max_pair_cosine_abs_delta", 0.02))
    if not -1.0 <= minimum_cosine <= 1.0:
        raise QualificationError("min_vector_cosine must be in [-1, 1]")
    if not 0.0 <= minimum_rate <= 1.0:
        raise QualificationError("min_vector_pass_rate must be in [0, 1]")
    if maximum_pair_delta < 0.0 or not math.isfinite(maximum_pair_delta):
        raise QualificationError("max_pair_cosine_abs_delta must be finite and nonnegative")

    sample_rows: list[dict[str, Any]] = []
    vector_cosines: list[float] = []
    reference_pairs: dict[str, dict[str, tuple[Mapping[str, Any], list[float]]]] = {}
    candidate_pairs: dict[str, dict[str, list[float]]] = {}
    for index, (expected, actual) in enumerate(zip(reference, candidate, strict=True)):
        expected_id = str(expected.get("sample_id", index))
        vector = expected.get("vector")
        if not isinstance(vector, list):
            raise QualificationError(f"HF encoder sample {expected_id!r} has no vector")
        reference_vector = _finite_vector(vector, f"HF encoder sample {expected_id!r}")
        candidate_vector = _candidate_vector(actual, expected_id)
        cosine = _vector_cosine(reference_vector, candidate_vector)
        pair_id = str(expected.get("pair_id", ""))
        pair_side = str(expected.get("pair_side", ""))
        if not pair_id or pair_side not in {"sentence1", "sentence2"}:
            raise QualificationError(f"HF encoder sample {expected_id!r} has invalid pair metadata")
        reference_pairs.setdefault(pair_id, {})[pair_side] = (expected, reference_vector)
        candidate_pairs.setdefault(pair_id, {})[pair_side] = candidate_vector
        vector_cosines.append(cosine)
        sample_rows.append(
            {
                "sample_id": expected_id,
                "pair_id": pair_id,
                "pair_side": pair_side,
                "vector_dim": len(reference_vector),
                "vector_cosine": cosine,
                "passed": cosine >= minimum_cosine,
            }
        )

    pair_rows: list[dict[str, Any]] = []
    pair_deltas: list[float] = []
    scores: list[float] = []
    reference_similarities: list[float] = []
    candidate_similarities: list[float] = []
    for pair_id, expected in reference_pairs.items():
        actual = candidate_pairs.get(pair_id, {})
        if set(expected) != {"sentence1", "sentence2"} or set(actual) != {
            "sentence1",
            "sentence2",
        }:
            raise QualificationError(f"encoder pair {pair_id!r} is incomplete")
        reference_similarity = _vector_cosine(
            expected["sentence1"][1], expected["sentence2"][1]
        )
        candidate_similarity = _vector_cosine(actual["sentence1"], actual["sentence2"])
        delta = abs(candidate_similarity - reference_similarity)
        score = float(expected["sentence1"][0]["score"])
        pair_deltas.append(delta)
        scores.append(score)
        reference_similarities.append(reference_similarity)
        candidate_similarities.append(candidate_similarity)
        pair_rows.append(
            {
                "pair_id": pair_id,
                "score": score,
                "hf_cosine": reference_similarity,
                "candidate_cosine": candidate_similarity,
                "cosine_abs_delta": delta,
                "passed": delta <= maximum_pair_delta,
            }
        )

    passed_vectors = sum(value >= minimum_cosine for value in vector_cosines)
    pass_rate = passed_vectors / len(vector_cosines)
    max_pair_delta = max(pair_deltas)
    status = (
        "passed"
        if pass_rate >= minimum_rate and max_pair_delta <= maximum_pair_delta
        else "failed"
    )
    return {
        "status": status,
        "metrics": {
            "samples": len(vector_cosines),
            "pairs": len(pair_rows),
            "passed_vectors": passed_vectors,
            "vector_pass_rate": pass_rate,
            "mean_vector_cosine": math.fsum(vector_cosines) / len(vector_cosines),
            "min_vector_cosine": min(vector_cosines),
            "mean_pair_cosine_abs_delta": math.fsum(pair_deltas) / len(pair_deltas),
            "max_pair_cosine_abs_delta": max_pair_delta,
            "hf_sts_spearman": _spearman(scores, reference_similarities),
            "candidate_sts_spearman": _spearman(scores, candidate_similarities),
        },
        "gate": {
            "min_vector_cosine": minimum_cosine,
            "min_vector_pass_rate": minimum_rate,
            "max_pair_cosine_abs_delta": maximum_pair_delta,
        },
        "samples": sample_rows,
        "pairs": pair_rows,
    }


def _candidate_vector(summary: Mapping[str, Any], sample_id: str) -> list[float]:
    values = summary.get("values")
    if not isinstance(values, list):
        raise QualificationError(f"TRTMC encoder sample {sample_id!r} has no vector values")
    vector = _finite_vector(values, f"TRTMC encoder sample {sample_id!r}")
    dim = summary.get("dim")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
        dim = len(vector)
    if summary.get("feature_kind") == "token":
        if len(vector) < dim:
            raise QualificationError(f"TRTMC encoder sample {sample_id!r} is shorter than dim")
        vector = vector[:dim]
    elif len(vector) != dim:
        raise QualificationError(
            f"TRTMC encoder sample {sample_id!r} has {len(vector)} values for dim {dim}"
        )
    return vector


def _finite_vector(values: Sequence[Any], label: str) -> list[float]:
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise QualificationError(f"{label} contains a non-numeric value") from error
    if not vector or any(not math.isfinite(value) for value in vector):
        raise QualificationError(f"{label} must be non-empty and finite")
    return vector


def _vector_cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise QualificationError(
            f"encoder vector dimensions differ: {len(left)} != {len(right)}"
        )
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise QualificationError("encoder vector must have a nonzero norm")
    cosine = math.fsum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
    return max(-1.0, min(1.0, cosine))


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return _pearson(_ranks(left), _ranks(right))


def _ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[indexed[position][0]] = rank
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    centered_left = [value - left_mean for value in left]
    centered_right = [value - right_mean for value in right]
    denominator = math.sqrt(
        math.fsum(value * value for value in centered_left)
        * math.fsum(value * value for value in centered_right)
    )
    if denominator <= 0.0:
        return None
    return math.fsum(
        a * b for a, b in zip(centered_left, centered_right, strict=True)
    ) / denominator


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
    inputs = request.get("inputs")
    if isinstance(inputs, Mapping):
        prompt = inputs.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt
    raise QualificationError("text-generation request has neither a prompt nor a user message")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise QualificationError(f"{name} must be a positive integer")
    return value


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
