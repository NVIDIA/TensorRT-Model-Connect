#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chronos-Bolt-owned ETTh1 Accuracy and compiled-reference Performance executor."""

from __future__ import annotations

import argparse
from array import array
import csv
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import uuid
from typing import Any, Iterable, Mapping, Sequence


REQUEST_SCHEMA = "trtmc.qualification-executor-request/v1"
RESULT_SCHEMA = "trtmc.qualification-result/v1"
REFERENCE_REQUEST_SCHEMA = "trtmc.chronos-bolt-reference-request/v1"
REFERENCE_RESULT_SCHEMA = "trtmc.chronos-bolt-reference-result/v1"


class ChronosQualificationError(RuntimeError):
    """Chronos-Bolt qualification cannot produce valid evidence."""


class PerformanceReferenceError(ChronosQualificationError):
    """One Performance reference mode cannot produce valid comparison evidence."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        request = _read_json(arguments.request, "executor request")
        item = _item(request)
        result = _execute(request, item, arguments.output.parent.resolve())
        _write_json(arguments.output, result)
        return 0 if result["execution"] == "completed" else 1
    except Exception as error:
        try:
            request = _read_json(arguments.request, "executor request")
            item = _item(request)
            _write_json(
                arguments.output,
                _error_result(item, str(error), arguments.output.parent.resolve()),
            )
        except Exception:
            print(
                f"Chronos-Bolt qualification failed before result identity was available: {error}",
                file=sys.stderr,
            )
        return 1


def _execute(request: Mapping[str, Any], item: Mapping[str, Any], item_dir: Path) -> dict[str, Any]:
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise ChronosQualificationError(f"request schema_version must be {REQUEST_SCHEMA}")
    if item.get("family") != "chronos_bolt" or item.get("kind") not in {
        "accuracy",
        "performance",
    }:
        raise ChronosQualificationError(
            "the Chronos-Bolt executor accepts only chronos_bolt Accuracy or Performance"
        )
    definition = _mapping(item.get("definition"), "suite definition")
    case = _mapping(item.get("case"), "case")
    environment = _mapping(request.get("environment"), "environment")
    if environment.get("schema_version") != "trtmc.qualification-environment/v1":
        raise ChronosQualificationError("unsupported qualification environment")
    manifest = _read_json(Path(str(item["manifest_path"])), "model manifest")
    if manifest.get("name") != item.get("model") or manifest.get("family") != "chronos_bolt":
        raise ChronosQualificationError("plan item and Chronos-Bolt manifest do not match")

    phase = request.get("phase")
    if phase == "prepare":
        return _prepare_case(request, item, manifest, definition, case, environment, item_dir)
    prepared = _mapping(request.get("preparation", {}), "preparation")
    _validate_preparation(item, prepared, require_samples=item["kind"] == "accuracy")
    manifest = {**manifest, "hf_revision": prepared["hf_revision"]}
    if phase == "check":
        return {
            **_identity(item),
            "schema_version": RESULT_SCHEMA,
            "execution": "completed",
            "verdict": "pass" if item["gate_policy"] == "blocking" else None,
            "details": {"prepared": dict(prepared)},
            "artifacts": [],
        }
    if phase != "run":
        raise ChronosQualificationError("executor phase must be prepare, check, or run")
    if item["kind"] == "accuracy":
        if definition.get("implementation") != "etth1_time_series_parity":
            raise ChronosQualificationError("unsupported Chronos-Bolt Accuracy implementation")
        return _execute_accuracy(
            request, item, manifest, definition, case, environment, prepared, item_dir
        )
    if definition.get("implementation") != "time_series_performance":
        raise ChronosQualificationError("unsupported Chronos-Bolt Performance implementation")
    return _execute_performance(
        request, item, manifest, definition, case, environment, prepared, item_dir
    )


def _prepare_case(
    request: Mapping[str, Any],
    item: Mapping[str, Any],
    manifest: Mapping[str, Any],
    definition: Mapping[str, Any],
    case: Mapping[str, Any],
    environment: Mapping[str, Any],
    item_dir: Path,
) -> dict[str, Any]:
    sample_snapshot = None
    if item["kind"] == "accuracy":
        samples, dataset_path = _load_samples(definition, case, environment, request.get("dataset"))
        sample_snapshot = item_dir / "selected-samples.json"
        _write_json(
            sample_snapshot,
            {
                "dataset_path": str(dataset_path),
                "dataset_version": _mapping(definition.get("dataset"), "dataset").get("version"),
                "samples": samples,
            },
        )
    execution = _mapping(environment.get("execution", {}), "environment execution")
    if not execution.get("allow_build", False):
        raise ChronosQualificationError(
            "a new Chronos-Bolt run requires allow_build; resume prepared work to reuse it"
        )
    checkpoint = _resolve_checkpoint(manifest, environment, item_dir)
    spec = {
        "models": [
            {
                "model": item["model"],
                "cases": [
                    {
                        "name": "prepare",
                        "testcase": _string(
                            _mapping(case.get("candidate"), "case candidate").get("testcase"),
                            "candidate.testcase",
                        ),
                        "measurement": {"warmup": 0, "iterations": 1},
                    }
                ],
            }
        ]
    }
    prepared_environment = {
        **environment,
        "storage": {
            **_mapping(environment.get("storage"), "environment storage"),
            "bundle_cache": str(item_dir / "bundles"),
            "bundle_roots": [],
        },
    }
    build = _run_benchmark(
        request={**request, "preparation": checkpoint},
        environment=prepared_environment,
        item_dir=item_dir,
        spec=spec,
        label="Chronos-Bolt bundle preparation",
        prepare_only=True,
    )
    bundles = build.get("bundles")
    if not isinstance(bundles, list) or len(bundles) != 1 or bundles[0].get("status") != "built":
        raise ChronosQualificationError(
            "preparation must build exactly one fresh Chronos-Bolt bundle"
        )
    bundle = str(Path(str(bundles[0]["bundle"])).resolve())
    checkpoint_path = Path(str(checkpoint["checkpoint"]))
    input_files = [bundle]
    if sample_snapshot is not None:
        input_files.append(str(sample_snapshot))
    input_files.extend(
        str(path)
        for path in sorted(checkpoint_path.rglob("*"))
        if path.is_file() and path.suffix in {".bin", ".json", ".model", ".safetensors", ".txt"}
    )
    prepared = {
        **checkpoint,
        "bundle": bundle,
        "build": bundles[0],
        "plan_item_id": item["id"],
        "sample_snapshot": str(sample_snapshot) if sample_snapshot else None,
        "input_files": input_files,
    }
    _write_json(item_dir / "bundle-receipt.json", prepared)
    return {
        **_identity(item),
        "schema_version": RESULT_SCHEMA,
        "execution": "completed",
        "verdict": "pass" if item["gate_policy"] == "blocking" else None,
        "details": {"prepared": prepared},
        "artifacts": [
            _artifact("bundle receipt", "bundle-receipt.json"),
            _artifact("build request", "candidate-spec.json"),
            _artifact("build command", "candidate-command.json"),
            _artifact("checkpoint command", "checkpoint-command.json"),
        ],
    }


def _resolve_checkpoint(
    manifest: Mapping[str, Any], environment: Mapping[str, Any], item_dir: Path
) -> dict[str, str]:
    tools = _mapping(environment.get("tools"), "environment tools")
    execution = _mapping(environment.get("execution", {}), "environment execution")
    python = _python_path(tools.get("reference_python"), "tools.reference_python")
    payload = {
        "repo_id": _string(manifest.get("hf_id"), "manifest hf_id"),
        "revision": manifest.get("hf_revision"),
        "local_files_only": bool(execution.get("local_files_only", False)),
    }
    code = (
        "import json, sys; from pathlib import Path; "
        "from huggingface_hub import snapshot_download; "
        "from chronos import ChronosBoltPipeline; "
        "r=json.loads(sys.argv[1]); p=Path(snapshot_download(**r)); "
        "print(json.dumps({'checkpoint':str(p),'hf_revision':p.name}))"
    )
    command = [str(python), "-c", code, json.dumps(payload)]
    _write_json(item_dir / "checkpoint-command.json", {"argv": command, "cwd": str(item_dir)})
    stdout_path = item_dir / "checkpoint.stdout.log"
    stderr_path = item_dir / "checkpoint.stderr.log"
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            command,
            cwd=item_dir,
            env=_reference_environment(),
            stdout=stdout,
            stderr=stderr,
            check=False,
            timeout=_timeout(environment),
        )
    if completed.returncode != 0:
        raise ChronosQualificationError(
            f"Chronos-Bolt checkpoint resolution exited {completed.returncode}; "
            f"see {stderr_path.name}"
        )
    result = _read_json(stdout_path, "checkpoint receipt")
    revision = result.get("hf_revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ChronosQualificationError("HF checkpoint must resolve to an immutable revision")
    checkpoint = Path(_string(result.get("checkpoint"), "checkpoint path"))
    if not checkpoint.is_dir():
        raise ChronosQualificationError("resolved HF checkpoint directory is missing")
    return {"checkpoint": str(checkpoint.resolve()), "hf_revision": revision}


def _validate_preparation(
    item: Mapping[str, Any], prepared: Mapping[str, Any], *, require_samples: bool
) -> None:
    if prepared.get("plan_item_id") != item["id"]:
        raise ChronosQualificationError("preparation does not match the selected case")
    if not Path(str(prepared.get("checkpoint", ""))).is_dir():
        raise ChronosQualificationError("prepared checkpoint is missing")
    if not Path(str(prepared.get("bundle", ""))).is_file():
        raise ChronosQualificationError("prepared bundle is missing")
    revision = prepared.get("hf_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ChronosQualificationError("prepared HF revision is invalid")
    if require_samples:
        _prepared_samples(prepared)


def _load_samples(
    definition: Mapping[str, Any],
    case: Mapping[str, Any],
    environment: Mapping[str, Any],
    dataset_evidence: Any = None,
) -> tuple[list[dict[str, Any]], Path]:
    storage = _mapping(environment.get("storage"), "environment storage")
    data_root = _path(storage.get("data_root"), "storage.data_root")
    dataset = _mapping(definition.get("dataset"), "suite dataset")
    relative = Path(_string(dataset.get("relative_path"), "dataset.relative_path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ChronosQualificationError("dataset.relative_path must stay below data_root")
    dataset_path = (data_root / relative).resolve()
    try:
        dataset_path.relative_to(data_root)
    except ValueError as error:
        raise ChronosQualificationError("dataset path escapes data_root") from error
    if not dataset_path.is_file():
        raise ChronosQualificationError(f"ETTh1 dataset does not exist: {dataset_path}")
    evidence = _mapping(dataset_evidence, "verified dataset evidence")
    if Path(_string(evidence.get("path"), "verified dataset path")) != dataset_path:
        raise ChronosQualificationError("verified dataset path does not match the ETTh1 suite")
    window = _mapping(case.get("window"), "case window")
    columns = window.get("input_columns")
    if not isinstance(columns, list) or not columns or not all(isinstance(v, str) for v in columns):
        raise ChronosQualificationError("window.input_columns must be a non-empty string list")
    with dataset_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(columns) <= set(reader.fieldnames or ()):
            raise ChronosQualificationError("ETTh1 dataset is missing configured input columns")
        rows = list(reader)
    context = _positive_integer(window.get("context_length"), "window.context_length")
    prediction = _positive_integer(window.get("prediction_length"), "window.prediction_length")
    stride = _positive_integer(window.get("stride"), "window.stride")
    target_start = _positive_integer(window.get("test_target_start"), "window.test_target_start")
    test_end = _positive_integer(window.get("test_end"), "window.test_end")
    if len(rows) < test_end or target_start < context or target_start + prediction > test_end:
        raise ChronosQualificationError("ETTh1 window bounds are inconsistent with the dataset")
    starts = list(range(target_start - context, test_end - context - prediction + 1, stride))
    selection = _mapping(definition.get("selection"), "suite selection")
    if selection.get("method") != "seeded_windows":
        raise ChronosQualificationError("unsupported ETTh1 sample selection")
    seed = selection.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ChronosQualificationError("seeded ETTh1 selection requires an integer seed")
    random.Random(seed).shuffle(starts)
    limit = case.get("sample_limit")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit == 0 or limit < -1:
        raise ChronosQualificationError("case sample_limit must be -1 or positive")
    if limit > 0:
        starts = starts[:limit]
    if not starts:
        raise ChronosQualificationError("ETTh1 selection contains no windows")
    frequency = int(window.get("frequency", 0))
    samples = []
    for start in starts:
        values = [float(row[column]) for row in rows[start : start + context] for column in columns]
        samples.append(
            {
                "sample_id": f"etth1_{start + context:06d}",
                "dataset_index": start + context,
                "past_values": values,
                "frequency": frequency,
            }
        )
    return samples, dataset_path


def _prepared_samples(prepared: Mapping[str, Any]) -> tuple[list[dict[str, Any]], Path, str]:
    snapshot = prepared.get("sample_snapshot")
    if not isinstance(snapshot, str) or not snapshot:
        raise ChronosQualificationError("Accuracy requires prepared ETTh1 samples")
    selected = _read_json(Path(snapshot), "prepared ETTh1 samples")
    samples = selected.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ChronosQualificationError("prepared ETTh1 samples must be non-empty")
    ids = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ChronosQualificationError("prepared ETTh1 sample must be an object")
        sample_id = _string(sample.get("sample_id"), "prepared sample id")
        values = sample.get("past_values")
        if not isinstance(values, list) or not values:
            raise ChronosQualificationError("prepared ETTh1 sample requires past_values")
        if sample_id in ids:
            raise ChronosQualificationError("prepared ETTh1 sample ids must be unique")
        ids.add(sample_id)
    return (
        samples,
        Path(_string(selected.get("dataset_path"), "prepared dataset path")),
        _string(selected.get("dataset_version"), "prepared dataset version"),
    )


def _execute_accuracy(
    request: Mapping[str, Any],
    item: Mapping[str, Any],
    manifest: Mapping[str, Any],
    definition: Mapping[str, Any],
    case: Mapping[str, Any],
    environment: Mapping[str, Any],
    prepared: Mapping[str, Any],
    item_dir: Path,
) -> dict[str, Any]:
    scoring = _mapping(definition.get("scoring"), "Accuracy scoring")
    if scoring.get("implementation") != "time_series_tensor_parity":
        raise ChronosQualificationError("unsupported Chronos-Bolt Accuracy scoring")
    samples, dataset_path, dataset_version = _prepared_samples(prepared)
    _write_jsonl(item_dir / "selected-samples.jsonl", samples)
    reference = _run_reference(
        purpose="accuracy",
        manifest=manifest,
        case=case,
        samples=samples,
        environment=environment,
        prepared=prepared,
        item_dir=item_dir,
    )
    candidate = _run_candidate_forecasts(
        samples=samples,
        environment=environment,
        prepared=prepared,
        item_dir=item_dir,
    )
    comparison = _compare_samples(reference.get("samples"), candidate, case)
    _write_jsonl(item_dir / "samples.jsonl", comparison["samples"])
    _write_jsonl(item_dir / "disagreements.jsonl", comparison["disagreements"])
    return {
        **_identity(item),
        "schema_version": RESULT_SCHEMA,
        "execution": "completed",
        "verdict": comparison["verdict"],
        "details": {
            "gate_policy": "blocking",
            "dataset": {"path": str(dataset_path), "version": dataset_version},
            "reference": {
                "backend": reference["backend"],
                "model": reference["model"],
                "revision": reference["revision"],
                "precision": reference["precision"],
                "device": reference["device"],
            },
            "candidate": _conversion_evidence(item, prepared),
            "comparison": {
                "reference": "official_pytorch",
                "candidate": "converted_tensorrt_bundle",
                "output_contract": "time_series_tensor_parity",
            },
            "actual_sample_count": comparison["actual_sample_count"],
            "passed_sample_count": comparison["passed_sample_count"],
            "failed_sample_count": comparison["failed_sample_count"],
            "metrics": comparison["metrics"],
            "gate_evaluations": comparison["gate_evaluations"],
        },
        "artifacts": [
            _artifact("selected samples", "selected-samples.jsonl"),
            _artifact("sample comparisons", "samples.jsonl"),
            _artifact("disagreements", "disagreements.jsonl"),
            _artifact("TensorRT forecast commands", "candidate-commands.jsonl"),
            _artifact("TensorRT forecasts", "candidate-forecasts.json"),
            _artifact("reference request", "reference-request.json"),
            _artifact("reference command", "reference-command.json"),
            _artifact("reference result", "reference.json"),
            _artifact("reference stdout", "reference.stdout.log"),
            _artifact("reference stderr", "reference.stderr.log"),
        ],
    }


def _run_reference(
    *,
    purpose: str,
    manifest: Mapping[str, Any],
    case: Mapping[str, Any],
    environment: Mapping[str, Any],
    prepared: Mapping[str, Any],
    item_dir: Path,
    samples: Sequence[Mapping[str, Any]] = (),
    performance_mode: str | None = None,
) -> dict[str, Any]:
    tools = _mapping(environment.get("tools"), "environment tools")
    execution = _mapping(environment.get("execution", {}), "environment execution")
    python = _python_path(tools.get("reference_python"), "tools.reference_python")
    configured = _mapping(case.get("reference"), "case reference")
    if configured.get("implementation") != "chronos_bolt":
        raise ChronosQualificationError("unsupported Chronos-Bolt reference implementation")
    payload: dict[str, Any] = {
        "schema_version": REFERENCE_REQUEST_SCHEMA,
        "purpose": purpose,
        "model": _string(manifest.get("hf_id"), "manifest hf_id"),
        "checkpoint": _string(prepared.get("checkpoint"), "prepared checkpoint"),
        "revision": _string(prepared.get("hf_revision"), "prepared HF revision"),
        "precision": _string(configured.get("precision"), "reference precision"),
    }
    if purpose == "accuracy":
        payload.update(
            {
                "device": str(execution.get("reference_device", "auto")),
                "samples": [dict(sample) for sample in samples],
            }
        )
        result_name = "reference.json"
    else:
        candidate = _mapping(case.get("candidate"), "case candidate")
        benchmark_request = _mapping(candidate.get("request"), "candidate request")
        mode = performance_mode or str(configured.get("mode"))
        payload.update(
            {
                "case_name": case["id"],
                "mode": mode,
                "compile_scope": (
                    configured.get("compile_scope") if mode == "torch-compile" else None
                ),
                "past_values": benchmark_request.get("past_values"),
                "frequency": benchmark_request.get("frequency", 0),
                "measurement": dict(
                    _mapping(candidate.get("measurement"), "candidate measurement")
                ),
            }
        )
        result_name = "reference-performance.json"
    request_path = item_dir / (
        "reference-request.json" if purpose == "accuracy" else "reference-performance-request.json"
    )
    output_path = item_dir / result_name
    stdout_path = item_dir / (
        "reference.stdout.log" if purpose == "accuracy" else "reference-performance.stdout.log"
    )
    stderr_path = item_dir / (
        "reference.stderr.log" if purpose == "accuracy" else "reference-performance.stderr.log"
    )
    command_path = item_dir / (
        "reference-command.json" if purpose == "accuracy" else "reference-performance-command.json"
    )
    _write_json(request_path, payload)
    runner = Path(__file__).with_name("reference.py")
    command = [
        str(python),
        str(runner),
        "--request",
        str(request_path),
        "--output",
        str(output_path),
    ]
    _write_json(command_path, {"argv": command, "cwd": str(item_dir)})
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            command,
            cwd=item_dir,
            env=_reference_environment(),
            stdout=stdout,
            stderr=stderr,
            check=False,
            timeout=_timeout(environment),
        )
    if completed.returncode != 0:
        raise ChronosQualificationError(
            f"Chronos-Bolt {purpose} reference exited {completed.returncode}; "
            f"see {stderr_path.name}"
        )
    result = _read_json(output_path, f"Chronos-Bolt {purpose} reference")
    expected = REFERENCE_RESULT_SCHEMA if purpose == "accuracy" else "trtmc.perf-baseline/v1"
    if result.get("schema_version") != expected:
        raise ChronosQualificationError("Chronos-Bolt reference returned an unsupported result")
    if result.get("revision") != prepared.get("hf_revision"):
        raise ChronosQualificationError("reference checkpoint revision differs from preparation")
    return result


def _run_candidate_forecasts(
    *,
    samples: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any],
    prepared: Mapping[str, Any],
    item_dir: Path,
) -> list[dict[str, Any]]:
    runtime_root = _path(
        _mapping(environment.get("storage"), "environment storage").get("runtime_root"),
        "storage.runtime_root",
    )
    tools = _mapping(environment.get("tools"), "environment tools")
    binary_value = tools.get("trtmc") or runtime_root / "trtmc"
    binary = _path(str(binary_value), "tools.trtmc")
    if not binary.is_file():
        raise ChronosQualificationError(f"TensorRT CLI does not exist: {binary}")
    bundle = _path(prepared.get("bundle"), "prepared bundle")
    commands = []
    outputs = []
    inputs_root = item_dir / "candidate-inputs"
    inputs_root.mkdir(exist_ok=True)
    environment_values = dict(os.environ)
    environment_values["LD_LIBRARY_PATH"] = ":".join(
        value
        for value in (str(runtime_root), environment_values.get("LD_LIBRARY_PATH", ""))
        if value
    )
    for index, sample in enumerate(samples):
        sample_id = _string(sample.get("sample_id"), "sample id")
        values = sample.get("past_values")
        if not isinstance(values, list) or not values:
            raise ChronosQualificationError("candidate sample requires past_values")
        input_path = inputs_root / f"{index:04d}.values.f32"
        mask_path = inputs_root / f"{index:04d}.mask.f32"
        with input_path.open("wb") as stream:
            array("f", (float(value) for value in values)).tofile(stream)
        with mask_path.open("wb") as stream:
            array("f", (1.0 for _ in values)).tofile(stream)
        command = [
            str(binary),
            "forecast",
            str(bundle),
            "--runtime-root",
            str(runtime_root),
            "--input",
            str(input_path),
            "--mask",
            str(mask_path),
            "--frequency",
            str(int(sample.get("frequency", 0))),
        ]
        completed = subprocess.run(
            command,
            cwd=item_dir,
            env=environment_values,
            capture_output=True,
            text=True,
            check=False,
            timeout=_timeout(environment),
        )
        commands.append(
            {
                "sample_id": sample_id,
                "argv": command,
                "returncode": completed.returncode,
                "stderr": completed.stderr,
            }
        )
        if completed.returncode != 0:
            _write_jsonl(item_dir / "candidate-commands.jsonl", commands)
            raise ChronosQualificationError(
                f"TensorRT forecast failed for {sample_id} with rc={completed.returncode}"
            )
        payload = _command_json(completed.stdout, sample_id)
        outputs.append({"sample_id": sample_id, **dict(payload)})
    _write_jsonl(item_dir / "candidate-commands.jsonl", commands)
    _write_json(item_dir / "candidate-forecasts.json", {"samples": outputs})
    return outputs


def _command_json(stdout: str, sample_id: str) -> Mapping[str, Any]:
    payloads = []
    for line in stdout.splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            value = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            payloads.append(dict(value))
    if not payloads:
        raise ChronosQualificationError(
            f"TensorRT forecast returned no JSON object for {sample_id}"
        )
    if any(payload != payloads[0] for payload in payloads[1:]):
        raise ChronosQualificationError(
            f"TensorRT forecast returned inconsistent rank outputs for {sample_id}"
        )
    return payloads[0]


def _compare_samples(
    reference: Any, candidate: Any, case: Mapping[str, Any], *, parity_field: str = "gate"
) -> dict[str, Any]:
    if not isinstance(reference, list) or not reference:
        raise ChronosQualificationError("reference produced no time-series samples")
    if not isinstance(candidate, list) or len(candidate) != len(reference):
        raise ChronosQualificationError("candidate and reference sample counts differ")
    gates = _mapping(case.get(parity_field), f"case {parity_field}")
    relative_limit = _positive_number(gates.get("max_relative_l2"), "max_relative_l2")
    absolute_limit = _positive_number(gates.get("max_absolute_error"), "max_absolute_error")
    rows = []
    for expected, actual in zip(reference, candidate, strict=True):
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
            raise ChronosQualificationError("time-series samples must be objects")
        sample_id = _string(expected.get("sample_id"), "reference sample id")
        if actual.get("sample_id") != sample_id:
            raise ChronosQualificationError("candidate and reference sample ids differ")
        expected_values = _numeric_list(expected.get("values"), "reference values")
        actual_values = _numeric_list(actual.get("values"), "candidate values")
        expected_shape = _shape(expected.get("shape"), "reference shape")
        actual_shape = _shape(actual.get("shape"), "candidate shape")
        shape_matches = (
            expected_shape == actual_shape
            and math.prod(expected_shape) == len(expected_values)
            and math.prod(actual_shape) == len(actual_values)
        )
        if not shape_matches:
            relative_l2 = None
            max_error = None
        else:
            difference = math.sqrt(
                sum((left - right) ** 2 for left, right in zip(actual_values, expected_values))
            )
            reference_norm = math.sqrt(sum(value**2 for value in expected_values))
            relative_l2 = difference / max(reference_norm, 1.0e-12)
            max_error = max(
                abs(left - right) for left, right in zip(actual_values, expected_values)
            )
        passed = bool(
            shape_matches
            and relative_l2 is not None
            and relative_l2 <= relative_limit
            and max_error is not None
            and max_error <= absolute_limit
        )
        rows.append(
            {
                "sample_id": sample_id,
                "passed": passed,
                "candidate_shape": list(actual_shape),
                "reference_shape": list(expected_shape),
                "shape_match": shape_matches,
                "relative_l2": relative_l2,
                "max_absolute_error": max_error,
            }
        )
    failed = sum(not row["passed"] for row in rows)
    max_relative = max(
        (float(row["relative_l2"]) for row in rows if row["relative_l2"] is not None),
        default=0.0,
    )
    max_absolute = max(
        (float(row["max_absolute_error"]) for row in rows if row["max_absolute_error"] is not None),
        default=0.0,
    )
    matching_shapes = sum(bool(row["shape_match"]) for row in rows)
    passed = failed == 0
    return {
        "verdict": "pass" if passed else "fail",
        "actual_sample_count": len(rows),
        "passed_sample_count": len(rows) - failed,
        "failed_sample_count": failed,
        "metrics": {
            "sample_agreement_rate": (len(rows) - failed) / len(rows),
            "shape_match_rate": matching_shapes / len(rows),
            "max_relative_l2": max_relative,
            "max_absolute_error": max_absolute,
        },
        "gate_evaluations": [
            {
                "name": "all_sample_shapes",
                "operator": "==",
                "threshold": len(rows),
                "actual": matching_shapes,
                "passed": matching_shapes == len(rows),
            },
            {
                "name": "all_samples_relative_l2",
                "operator": "<=",
                "threshold": relative_limit,
                "actual": max_relative,
                "passed": max_relative <= relative_limit,
            },
            {
                "name": "all_samples_absolute_error",
                "operator": "<=",
                "threshold": absolute_limit,
                "actual": max_absolute,
                "passed": max_absolute <= absolute_limit,
            },
        ],
        "samples": rows,
        "disagreements": [row for row in rows if not row["passed"]],
    }


def measurement_stability(samples: Sequence[float]) -> dict[str, Any]:
    if len(samples) != 10 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for value in samples
    ):
        raise ValueError("stability requires ten finite positive latency samples")
    median = statistics.median(samples)
    first = statistics.median(samples[:5])
    last = statistics.median(samples[5:])
    drift = abs(last - first) / first
    close = sum(abs(sample - median) / median <= 0.05 for sample in samples)
    return {
        "stable": drift <= 0.05 and close >= 8,
        "first_half_median_ms": first,
        "last_half_median_ms": last,
        "median_drift": drift,
        "samples_within_five_percent": close,
    }


def _execute_performance(
    request: Mapping[str, Any],
    item: Mapping[str, Any],
    manifest: Mapping[str, Any],
    definition: Mapping[str, Any],
    case: Mapping[str, Any],
    environment: Mapping[str, Any],
    prepared: Mapping[str, Any],
    item_dir: Path,
) -> dict[str, Any]:
    expected_stability = {
        "samples": 10,
        "median_drift_limit": 0.05,
        "within_median_fraction": 0.05,
        "minimum_close_samples": 8,
        "retries": 1,
    }
    if definition.get("stability") != expected_stability:
        raise ChronosQualificationError("unsupported Performance stability protocol")
    preferred_mode, fallback_mode = _performance_reference_modes(case)
    attempts = []
    artifacts = []
    for mode in (preferred_mode, fallback_mode):
        directory_name = "compiled" if mode == "torch-compile" else "eager"
        for attempt in range(2):
            relative_root = f"{directory_name}/attempt-{attempt + 1}/"
            directory = item_dir / relative_root
            directory.mkdir(parents=True, exist_ok=True)
            try:
                result = _performance_attempt(
                    request,
                    item,
                    manifest,
                    definition,
                    case,
                    environment,
                    prepared,
                    directory,
                    reference_mode=mode,
                )
            except PerformanceReferenceError as error:
                failed = {
                    "mode": mode,
                    "attempt": attempt + 1,
                    "execution": "error",
                    "error": str(error),
                }
                parity_path = directory / "reference-parity.json"
                if parity_path.is_file():
                    failed["output_parity"] = _read_json(parity_path, "Performance output parity")
                attempts.append(failed)
                artifacts.extend(
                    {**artifact, "path": relative_root + artifact["path"]}
                    for artifact in _existing_artifacts(directory)
                )
                break

            details = _mapping(result.get("details"), "Performance details")
            stability = {
                side: measurement_stability(details[side]["samples_ms"])
                for side in ("candidate", "reference")
            }
            attempts.append(
                {
                    "mode": mode,
                    "attempt": attempt + 1,
                    "execution": "completed",
                    "stability": stability,
                    "candidate": details["candidate"],
                    "reference": details["reference"],
                }
            )
            artifacts.extend(
                {**artifact, "path": relative_root + artifact["path"]}
                for artifact in result["artifacts"]
            )
            if not all(value["stable"] for value in stability.values()):
                continue

            result["artifacts"] = artifacts
            details["measurement_attempts"] = attempts
            details["measurement_stability"] = stability
            details["comparison_valid"] = True
            details["reference_selection"] = {
                "policy": "prefer_torch_compile_then_eager",
                "preferred_mode": preferred_mode,
                "fallback_mode": fallback_mode,
                "selected_mode": mode,
                "fallback_used": mode == fallback_mode,
            }
            return result

    return {
        **_identity(item),
        "schema_version": RESULT_SCHEMA,
        "execution": "error",
        "verdict": None,
        "details": {
            "error": "all_performance_references_failed",
            "measurement_attempts": attempts,
            "reference_selection": {
                "policy": "prefer_torch_compile_then_eager",
                "preferred_mode": preferred_mode,
                "fallback_mode": fallback_mode,
                "selected_mode": None,
                "fallback_used": False,
            },
        },
        "artifacts": artifacts,
    }


def _performance_reference_modes(case: Mapping[str, Any]) -> tuple[str, str]:
    configured = _mapping(case.get("reference"), "case reference")
    preferred = configured.get("mode")
    fallback = configured.get("fallback")
    if preferred != "torch-compile" or fallback != "eager":
        raise ChronosQualificationError(
            "Chronos-Bolt Performance requires torch-compile with eager fallback"
        )
    return str(preferred), str(fallback)


def _performance_attempt(
    request: Mapping[str, Any],
    item: Mapping[str, Any],
    manifest: Mapping[str, Any],
    definition: Mapping[str, Any],
    case: Mapping[str, Any],
    environment: Mapping[str, Any],
    prepared: Mapping[str, Any],
    item_dir: Path,
    *,
    reference_mode: str,
) -> dict[str, Any]:
    if item.get("gate_policy") != "observation_only":
        raise ChronosQualificationError(
            "Chronos-Bolt Performance requires observation_only until a device run owns its gate"
        )
    comparison_policy = _mapping(definition.get("comparison"), "Performance comparison")
    if comparison_policy != {
        "primary_metric": "latency_ms.p50",
        "output_contract": "time_series_tensor_parity",
    }:
        raise ChronosQualificationError("unsupported Chronos-Bolt Performance comparison")
    candidate_config = _mapping(case.get("candidate"), "case candidate")
    benchmark_request = dict(_mapping(candidate_config.get("request"), "candidate request"))
    spec = {
        "models": [
            {
                "model": item["model"],
                "cases": [
                    {
                        "name": item["case_id"],
                        "testcase": _string(candidate_config.get("testcase"), "candidate.testcase"),
                        "request": benchmark_request,
                        "measurement": dict(
                            _mapping(candidate_config.get("measurement"), "candidate measurement")
                        ),
                        "telemetry": dict(
                            _mapping(
                                candidate_config.get("telemetry", {"gpu": "auto"}),
                                "candidate telemetry",
                            )
                        ),
                    }
                ],
            }
        ]
    }
    candidate = _run_benchmark(
        request={**request, "preparation": prepared},
        environment=environment,
        item_dir=item_dir,
        spec=spec,
        label="Chronos-Bolt Performance candidate",
    )
    cells = candidate.get("cells")
    if not isinstance(cells, list) or len(cells) != 1 or cells[0].get("status") != "completed":
        raise ChronosQualificationError("Chronos-Bolt Performance candidate did not complete")
    cell = _mapping(cells[0], "Performance benchmark cell")
    metrics = _mapping(cell.get("metrics"), "Performance metrics")
    _validate_performance_metrics(definition, metrics)
    policy = _mapping(candidate.get("measurement_policy"), "candidate measurement policy")
    _validate_timing(definition, policy, "candidate")
    measurement = _mapping(candidate_config.get("measurement"), "candidate measurement")
    iterations = int(measurement.get("iterations"))
    candidate_samples = _performance_samples(cell.get("samples_ms"), iterations, "TensorRT")
    direct_candidate = _run_candidate_forecasts(
        samples=[
            {
                "sample_id": item["case_id"],
                "past_values": benchmark_request["past_values"],
                "frequency": benchmark_request.get("frequency", 0),
            }
        ],
        environment=environment,
        prepared=prepared,
        item_dir=item_dir,
    )
    try:
        reference = _run_reference(
            purpose="performance",
            manifest=manifest,
            case=case,
            environment=environment,
            prepared=prepared,
            item_dir=item_dir,
            performance_mode=reference_mode,
        )
        reference_metrics = _validate_performance_reference(
            definition,
            case,
            manifest,
            reference,
            iterations,
            expected_mode=reference_mode,
        )
        parity = _compare_samples(
            [{"sample_id": item["case_id"], **reference["output_summary"]}],
            direct_candidate,
            case,
            parity_field="parity",
        )
        _write_json(item_dir / "reference-parity.json", parity)
        if parity["verdict"] != "pass":
            raise ChronosQualificationError(
                f"TensorRT and {reference_mode} reference outputs differ"
            )
        _validate_performance_device(
            _mapping(candidate.get("environment"), "candidate environment"),
            _mapping(reference.get("environment"), "reference environment"),
        )
    except (ChronosQualificationError, subprocess.TimeoutExpired) as error:
        raise PerformanceReferenceError(f"{reference_mode} reference failed: {error}") from error
    candidate_p50 = float(_mapping(metrics.get("latency_ms"), "candidate latency")["p50"])
    reference_p50 = float(_mapping(reference_metrics.get("latency_ms"), "reference latency")["p50"])
    ratio = reference_p50 / candidate_p50
    conversion = _conversion_evidence(item, prepared)
    return {
        **_identity(item),
        "schema_version": RESULT_SCHEMA,
        "execution": "completed",
        "verdict": None,
        "details": {
            "gate_policy": "observation_only",
            "candidate": {
                "backend": "tensorrt",
                "conversion": conversion,
                "metrics": dict(metrics),
                "samples_ms": candidate_samples,
                "measurement_policy": dict(policy),
                "runtime_environment": dict(candidate["environment"]),
            },
            "reference": {
                "backend": "official_pytorch",
                "mode": reference["mode"],
                "compile_scope": reference.get("compile_scope"),
                "compile_evidence": dict(reference["compile_evidence"]),
                "model": reference["model"],
                "revision": reference["revision"],
                "precision": reference["precision"],
                "metrics": reference_metrics,
                "samples_ms": list(reference["samples_ms"]),
                "measurement_policy": dict(reference["measurement_policy"]),
                "runtime_environment": dict(reference["environment"]),
            },
            "comparison": {
                "output_contract": "time_series_tensor_parity",
                "output_match": True,
                "output_metrics": parity["metrics"],
                "reference_mode": reference_mode,
                "candidate_p50_ms": candidate_p50,
                "reference_p50_ms": reference_p50,
                "reference_over_candidate_p50": ratio,
            },
            "metrics": {
                "candidate_latency_ms_p50": candidate_p50,
                "reference_latency_ms_p50": reference_p50,
                "reference_over_candidate_p50": ratio,
            },
        },
        "artifacts": [
            _artifact("TensorRT performance request", "candidate-spec.json"),
            _artifact("TensorRT command", "candidate-command.json"),
            _artifact("TensorRT performance result", "candidate/result.json"),
            _artifact("TensorRT performance report", "candidate/report.html"),
            _artifact("TensorRT performance stdout", "candidate.stdout.log"),
            _artifact("TensorRT performance stderr", "candidate.stderr.log"),
            _artifact("TensorRT parity command", "candidate-commands.jsonl"),
            _artifact("TensorRT parity output", "candidate-forecasts.json"),
            _artifact("reference output parity", "reference-parity.json"),
            _artifact("reference request", "reference-performance-request.json"),
            _artifact("reference command", "reference-performance-command.json"),
            _artifact("reference result", "reference-performance.json"),
            _artifact("reference stdout", "reference-performance.stdout.log"),
            _artifact("reference stderr", "reference-performance.stderr.log"),
        ],
    }


def _validate_performance_reference(
    definition: Mapping[str, Any],
    case: Mapping[str, Any],
    manifest: Mapping[str, Any],
    reference: Mapping[str, Any],
    iterations: int,
    *,
    expected_mode: str,
) -> dict[str, Any]:
    if reference.get("status") != "completed" or reference.get("backend") != "chronos-bolt":
        raise ChronosQualificationError("Chronos-Bolt Performance reference did not complete")
    configured = _mapping(case.get("reference"), "case reference")
    if expected_mode not in _performance_reference_modes(case):
        raise ChronosQualificationError("Performance reference mode is not configured")
    if reference.get("mode") != expected_mode:
        raise ChronosQualificationError("Performance reference used the wrong mode")
    if reference.get("model") != manifest.get("hf_id"):
        raise ChronosQualificationError("Performance reference used the wrong HF model")
    if reference.get("precision") != configured.get("precision"):
        raise ChronosQualificationError("Performance reference precision differs from the case")
    evidence = _mapping(reference.get("compile_evidence"), "compile evidence")
    if expected_mode == "torch-compile":
        if reference.get("compile_scope") != configured.get("compile_scope"):
            raise ChronosQualificationError("Performance reference compiled the wrong callable")
        required = {
            "api": "torch.compile",
            "target": "model.forward",
            "backend": "inductor",
            "dynamic": False,
            "applied": True,
            "warmup_completed": True,
            "timed_callable_uses_compiled_target": True,
        }
    else:
        if reference.get("compile_scope") is not None:
            raise ChronosQualificationError("eager Performance reference has a compile scope")
        required = {
            "api": "none",
            "applied": False,
            "compiled_graph_count": 0,
            "warmup_completed": True,
            "timed_callable_uses_compiled_target": False,
        }
    for field, expected in required.items():
        if evidence.get(field) != expected:
            raise ChronosQualificationError(
                f"Performance reference has invalid compile evidence field {field!r}"
            )
    if expected_mode == "torch-compile":
        graphs = evidence.get("compiled_graph_count")
        if isinstance(graphs, bool) or not isinstance(graphs, int) or graphs < 1:
            raise ChronosQualificationError("Performance reference compiled no graphs")
    policy = _mapping(reference.get("measurement_policy"), "reference measurement policy")
    _validate_timing(definition, policy, "reference")
    samples = _performance_samples(reference.get("samples_ms"), iterations, expected_mode)
    metrics = dict(_mapping(reference.get("metrics"), "reference metrics"))
    _validate_performance_metrics(definition, metrics)
    if int(metrics.get("sample_count", -1)) != len(samples):
        raise ChronosQualificationError("reference sample_count differs from raw samples")
    return metrics


def _validate_performance_metrics(
    definition: Mapping[str, Any], metrics: Mapping[str, Any]
) -> None:
    required = definition.get("metrics")
    if not isinstance(required, list) or not required:
        raise ChronosQualificationError("Performance suite metrics must be a non-empty list")
    for field in required:
        value: Any = metrics
        for name in str(field).split("."):
            if not isinstance(value, Mapping) or name not in value:
                raise ChronosQualificationError(f"Performance metric {field!r} is missing")
            value = value[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ChronosQualificationError(f"Performance metric {field!r} must be finite numeric")


def _validate_timing(definition: Mapping[str, Any], actual: Mapping[str, Any], side: str) -> None:
    timing = _mapping(definition.get("timing"), "Performance timing")
    expected = _mapping(timing.get(side), f"Performance {side} timing")
    for field, value in expected.items():
        if actual.get(field) != value:
            raise ChronosQualificationError(
                f"Performance {side} timing does not match suite field {field!r}"
            )


def _validate_performance_device(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> None:
    gpus = candidate.get("gpus", [])
    allocation = candidate.get("cuda_visible_devices")
    if allocation and "," in str(allocation):
        raise ChronosQualificationError("Chronos-Bolt Performance requires one GPU allocation")
    if isinstance(gpus, list) and len(gpus) > 1:
        gpus = [
            gpu for gpu in gpus if str(allocation) in {str(gpu.get("index")), str(gpu.get("uuid"))}
        ]
    if (
        not isinstance(gpus, list)
        or len(gpus) != 1
        or not gpus[0].get("uuid")
        or not reference.get("gpu_uuid")
    ):
        raise ChronosQualificationError("Performance GPU identity is missing or ambiguous")
    left = str(gpus[0]["uuid"]).removeprefix("GPU-").lower()
    right = str(reference["gpu_uuid"]).removeprefix("GPU-").lower()
    if left != right:
        raise ChronosQualificationError("candidate and reference used different GPUs")


def _performance_samples(value: Any, expected: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != expected:
        raise ChronosQualificationError(
            f"{label} latency samples do not match requested iterations"
        )
    samples = []
    for sample in value:
        if (
            isinstance(sample, bool)
            or not isinstance(sample, (int, float))
            or not math.isfinite(float(sample))
            or float(sample) <= 0.0
        ):
            raise ChronosQualificationError(f"{label} latency samples must be finite positive")
        samples.append(float(sample))
    return samples


def _conversion_evidence(item: Mapping[str, Any], prepared: Mapping[str, Any]) -> dict[str, Any]:
    build = _mapping(prepared.get("build"), "prepared build")
    if build.get("model") != item.get("model") or build.get("status") != "built":
        raise ChronosQualificationError("prepared TensorRT bundle identity is invalid")
    return {
        "backend": "tensorrt",
        "source": "converted_bundle",
        "model": item["model"],
        "manifest": item["manifest_path"],
        "bundle": _string(prepared.get("bundle"), "prepared bundle"),
        "bundle_status": "reused",
    }


def _run_benchmark(
    *,
    request: Mapping[str, Any],
    environment: Mapping[str, Any],
    item_dir: Path,
    spec: Mapping[str, Any],
    label: str,
    prepare_only: bool = False,
) -> dict[str, Any]:
    tools = _mapping(environment.get("tools"), "environment tools")
    storage = _mapping(environment.get("storage"), "environment storage")
    execution = _mapping(environment.get("execution", {}), "environment execution")
    runtime_root = _path(storage.get("runtime_root"), "storage.runtime_root")
    spec_path = item_dir / "candidate-spec.json"
    output_dir = item_dir / "candidate"
    stdout_path = item_dir / "candidate.stdout.log"
    stderr_path = item_dir / "candidate.stderr.log"
    _write_json(spec_path, spec)
    command = [
        str(tools.get("python") or sys.executable),
        "-m",
        "trtmc_benchmark",
        "run",
        str(spec_path),
        "--manifest-root",
        str(request["families_root"]),
        "--runtime-root",
        str(runtime_root),
        "-o",
        str(output_dir),
    ]
    worker = tools.get("trtmc_worker")
    if worker:
        command.extend(("--worker", str(_path(worker, "tools.trtmc_worker"))))
    bundle_cache = storage.get("bundle_cache")
    if bundle_cache:
        command.extend(("--bundle-cache", str(_path(bundle_cache, "storage.bundle_cache"))))
    roots = storage.get("bundle_roots", [])
    if not isinstance(roots, list) or not all(isinstance(value, str) for value in roots):
        raise ChronosQualificationError("storage.bundle_roots must be a path list")
    for root in roots:
        command.extend(("--bundle-root", str(_path(root, "storage.bundle_roots"))))
    if not execution.get("allow_build", False):
        command.append("--no-build")
    prepared = _mapping(request.get("preparation", {}), "benchmark preparation")
    if prepared.get("checkpoint"):
        command.extend(("--model-dir", f"{spec['models'][0]['model']}={prepared['checkpoint']}"))
    if prepare_only:
        command.append("--prepare-only")
    elif prepared.get("bundle"):
        command.extend(("--bundle", str(prepared["bundle"]), "--no-build"))
    _write_json(item_dir / "candidate-command.json", {"argv": command, "cwd": str(item_dir)})
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            command,
            cwd=item_dir,
            stdout=stdout,
            stderr=stderr,
            check=False,
            timeout=_timeout(environment),
        )
    if completed.returncode != 0:
        raise ChronosQualificationError(
            f"{label} exited {completed.returncode}; see {stderr_path.name}"
        )
    if prepare_only:
        return _read_json(stdout_path, "bundle preparation")
    result = _read_json(output_dir / "result.json", f"{label} result")
    if (
        result.get("schema_version") != "trtmc.benchmark-run/v2"
        or result.get("status") != "completed"
    ):
        raise ChronosQualificationError(f"{label} returned an incomplete result")
    bundles = result.get("preparation", {}).get("bundles", [])
    if (
        len(bundles) != 1
        or Path(str(bundles[0].get("bundle", ""))).resolve()
        != Path(str(prepared["bundle"])).resolve()
    ):
        raise ChronosQualificationError("benchmark did not use the prepared bundle")
    return result


def _item(request: Mapping[str, Any]) -> dict[str, Any]:
    item = request.get("plan_item")
    if not isinstance(item, dict):
        raise ChronosQualificationError("executor request requires a plan_item object")
    required = (
        "id",
        "family",
        "model",
        "kind",
        "suite_id",
        "case_id",
        "gate_policy",
        "manifest_path",
    )
    if any(not isinstance(item.get(name), str) or not item[name] for name in required):
        raise ChronosQualificationError("plan item identity is incomplete")
    return item


def _identity(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "plan_item_id": item["id"],
        "family": item["family"],
        "model": item["model"],
        "kind": item["kind"],
        "suite_id": item["suite_id"],
        "case_id": item["case_id"],
    }


def _error_result(
    item: Mapping[str, Any], message: str, item_dir: Path | None = None
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA,
        **_identity(item),
        "execution": "error",
        "verdict": None,
        "details": {"error": message},
        "artifacts": _existing_artifacts(item_dir) if item_dir else [],
    }


def _existing_artifacts(item_dir: Path) -> list[dict[str, str]]:
    known = (
        ("selected samples", "selected-samples.jsonl"),
        ("sample comparisons", "samples.jsonl"),
        ("disagreements", "disagreements.jsonl"),
        ("candidate request", "candidate-spec.json"),
        ("candidate command", "candidate-command.json"),
        ("candidate forecasts", "candidate-forecasts.json"),
        ("candidate forecast commands", "candidate-commands.jsonl"),
        ("candidate result", "candidate/result.json"),
        ("candidate report", "candidate/report.html"),
        ("candidate stdout", "candidate.stdout.log"),
        ("candidate stderr", "candidate.stderr.log"),
        ("reference output parity", "reference-parity.json"),
        ("reference request", "reference-request.json"),
        ("reference command", "reference-command.json"),
        ("reference result", "reference.json"),
        ("reference stdout", "reference.stdout.log"),
        ("reference stderr", "reference.stderr.log"),
        ("Performance reference request", "reference-performance-request.json"),
        ("Performance reference command", "reference-performance-command.json"),
        ("Performance reference result", "reference-performance.json"),
        ("Performance reference stdout", "reference-performance.stdout.log"),
        ("Performance reference stderr", "reference-performance.stderr.log"),
    )
    prefixes = (
        "",
        "retry/",
        "compiled/attempt-1/",
        "compiled/attempt-2/",
        "eager/attempt-1/",
        "eager/attempt-2/",
    )
    return [
        _artifact(prefix + label, prefix + relative)
        for prefix in prefixes
        for label, relative in known
        if not (item_dir / (prefix + relative)).is_symlink()
        and (item_dir / (prefix + relative)).is_file()
    ]


def _artifact(label: str, path: str) -> dict[str, str]:
    return {"label": label, "path": path}


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ChronosQualificationError(f"{field} must be an object")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ChronosQualificationError(f"{field} must be a non-empty string")
    return value


def _path(value: Any, field: str) -> Path:
    return (
        Path(_string(str(value) if isinstance(value, Path) else value, field))
        .expanduser()
        .resolve()
    )


def _python_path(value: Any, field: str) -> Path:
    path = Path(_string(value, field)).expanduser().absolute()
    if not path.is_file():
        raise ChronosQualificationError(f"{field} does not exist: {path}")
    return path


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ChronosQualificationError(f"{field} must be a positive integer")
    return value


def _positive_number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ChronosQualificationError(f"{field} must be finite and positive")
    return float(value)


def _numeric_list(value: Any, field: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ChronosQualificationError(f"{field} must be a non-empty list")
    result = []
    for entry in value:
        if (
            isinstance(entry, bool)
            or not isinstance(entry, (int, float))
            or not math.isfinite(float(entry))
        ):
            raise ChronosQualificationError(f"{field} must contain finite numbers")
        result.append(float(entry))
    return result


def _shape(value: Any, field: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(entry, bool) or not isinstance(entry, int) or entry < 1 for entry in value
        )
    ):
        raise ChronosQualificationError(f"{field} must contain positive dimensions")
    return tuple(value)


def _reference_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _timeout(environment: Mapping[str, Any]) -> int:
    execution = _mapping(environment.get("execution", {}), "environment execution")
    value = execution.get("timeout_seconds", 7200)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ChronosQualificationError("execution.timeout_seconds must be positive")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ChronosQualificationError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ChronosQualificationError(f"{label} must contain an object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
