# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal Accuracy qualification through the installed user benchmark."""

from __future__ import annotations

from array import array
from collections import defaultdict
import csv
from itertools import permutations
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from trtmc_benchmark.artifact_metrics import (
    generated_audio_metrics,
    generated_audio_passes,
    generated_media_metrics,
    generated_media_passes,
    metric_geometry_metrics,
    metric_geometry_passes,
    robot_action_metrics,
    robot_action_passes,
)
from qualification_tests.benchmark_qualification.performance import reference_protocol

from .catalog import QualificationCase, QualificationError, load_benchmark
from .datasets import Dataset, resolve_dataset
from .runtime import (
    RuntimeContext,
    prepare_bundle,
    reference_environment_options,
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
    metric = definition.get("metric")
    metric_name = metric.get("name") if isinstance(metric, Mapping) else None
    if metric_name == "task_output_parity":
        result = _task_output_parity(case, context, output)
    else:
        dataset = resolve_dataset(definition, context, case.source)
    if metric_name == "exact_token_ids":
        result = _text_generation_parity(case, context, definition, dataset, output)
    elif metric_name == "embedding_vector_parity":
        result = _encoder_embedding_parity(case, context, dataset, output)
    elif metric_name == "reranking_score_parity":
        result = _reranking_score_parity(case, context, dataset, output)
    elif metric_name == "forecast_tensor_parity":
        result = _etth1(case, context, definition, dataset, output)
    elif metric_name == "image_classification_top1_parity":
        result = _image_classification_parity(case, context, dataset, output)
    elif metric_name == "speech_transcription_wer_parity":
        result = _speech_transcription_parity(case, context, definition, dataset, output)
    elif metric_name == "image_feature_knn_parity":
        result = _image_feature_knn_parity(case, context, dataset, output)
    elif metric_name == "coco_object_detection_accuracy":
        result = _coco_object_detection_accuracy(case, context, dataset, output)
    elif metric_name == "prompted_segmentation_parity":
        result = _prompted_segmentation_parity(case, context, dataset, output)
    elif metric_name == "text_prompted_instance_segmentation_parity":
        result = _text_prompted_instance_segmentation_parity(case, context, dataset, output)
    elif metric_name == "semantic_segmentation_parity":
        result = _semantic_segmentation_parity(case, context, dataset, output)
    elif metric_name == "vision_language_text_parity":
        result = _vision_language_text_parity(case, context, dataset, output)
    elif metric_name == "ocr_text_parity":
        result = _ocr_text_parity(case, context, definition, dataset, output)
    elif metric_name == "localization_text_parity":
        result = _localization_text_parity(case, context, dataset, output)
    elif metric_name == "stereo_disparity_parity":
        result = _stereo_disparity_parity(case, context, dataset, output)
    elif metric_name == "metric_geometry_parity":
        result = _metric_geometry_parity(case, context, dataset, output)
    elif metric_name == "robot_action_parity":
        result = _robot_action_parity(case, context, dataset, output)
    elif metric_name == "task_output_parity":
        pass
    else:
        raise QualificationError(f"unsupported Accuracy metric {metric_name!r}")
    write_result(output, result)
    return result


def _task_output_parity(
    case: QualificationCase, context: RuntimeContext, output: Path
) -> dict[str, Any]:
    configured = case.values
    request = configured.get("request", {})
    reference = configured.get("reference", {})
    gate = configured.get("gate", {})
    if not all(isinstance(value, Mapping) for value in (request, reference, gate)):
        raise QualificationError("task-output request, reference, and gate must be objects")
    request = _resolve_task_assets(case, request)
    descriptor = write_model_descriptor(case, output, request, context=context)
    descriptor_value = json.loads(descriptor.read_text(encoding="utf-8"))
    adapter_value = reference.get("adapter")
    script_value = reference.get("script")
    family_root = case.source.parents[2].resolve()
    if (adapter_value is None) == (script_value is None):
        raise QualificationError("task-output reference requires exactly one adapter or script")
    adapter = str(adapter_value) if adapter_value is not None else None
    script = None
    if script_value is not None:
        relative = Path(str(script_value))
        script = (family_root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not script.is_relative_to(family_root)
            or not script.is_file()
        ):
            raise QualificationError(f"task-output reference script is unavailable: {script_value}")
    adapter_options = dict(reference.get("adapter_options", {}))
    for name, value in tuple(adapter_options.items()):
        if not name.endswith("_path") or not isinstance(value, str):
            continue
        path = Path(value)
        if not path.is_absolute():
            path = (case.source.parent / path).resolve()
        if not path.is_relative_to(family_root) or not path.is_file():
            raise QualificationError(f"task-output reference asset {name!r} is unavailable: {path}")
        adapter_options[name] = str(path)
    adapter_options = reference_environment_options(case, context, adapter_options)
    reference_path = output / "reference-result.json"
    revision = reference.get("revision", descriptor_value.get("hf_revision"))
    selected_task = case.candidate.get("selected_task")
    command = reference_protocol.command(
        python=reference_python(case, context),
        generic_runner=(
            context.repository
            / "qualification_tests/benchmark_qualification/performance/references/generic_reference.py"
        ),
        script=script,
        family=case.family,
        operation=str(configured["operation"]),
        manifest=descriptor,
        selected_task=str(selected_task or case.candidate["task"]) if script else (
            str(selected_task) if selected_task else None
        ),
        testcase_name=case.name,
        adapter=adapter,
        adapter_options=adapter_options,
        timing_contract={},
        padding="longest",
        model=str(descriptor_value["hf_id"]),
        revision=str(revision) if revision else None,
        request=request,
        precision=str(reference.get("precision", case.candidate["precision"])),
        mode=str(reference.get("mode", "hf-eager")),
        warmup=0,
        iterations=1,
        case_name=case.name,
        output=reference_path,
        trust_remote_code=bool(case.candidate.get("trust_remote_code", False)),
        local_files_only=os.environ.get("TRTMC_QUALIFICATION_LOCAL_FILES_ONLY") == "1",
    )
    completed = run_command(command, output, "reference", timeout=7200, verbose=context.verbose)
    if completed.returncode != 0:
        raise QualificationError(f"task-output Accuracy reference failed; see {output}")
    try:
        reference_result = json.loads(reference_path.read_text(encoding="utf-8"))
        expected = reference_result["output_summary"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise QualificationError("task-output Accuracy reference returned no output") from error
    actual_values, bundle = _candidate_outputs(
        case,
        context,
        output,
        str(configured["operation"]),
        [{"sample_id": case.name, "request": request}],
    )
    actual = actual_values[0]
    contract = str(reference.get("output_contract", ""))
    if not isinstance(expected, Mapping):
        raise QualificationError("task-output Accuracy reference returned an invalid output")
    try:
        if contract == "media-artifact-parity":
            metrics = generated_media_metrics(actual, expected)
            passed = generated_media_passes(metrics, gate)
        elif contract == "audio-artifact-parity":
            metrics = generated_audio_metrics(actual, expected)
            passed = generated_audio_passes(metrics, gate)
        elif contract == "normalized-text":
            candidate_text = _normalized_answer(actual.get("text"))
            reference_text = _normalized_answer(expected.get("text"))
            distance = _normalized_edit_distance(candidate_text, reference_text)
            metrics = {"normalized_edit_distance": distance}
            passed = bool(candidate_text) and distance <= float(
                gate["max_normalized_edit_distance"]
            )
        else:
            raise QualificationError(f"unsupported task-output contract {contract!r}")
    except (KeyError, TypeError, ValueError) as error:
        raise QualificationError(f"task-output Accuracy comparison failed: {error}") from error
    return {
        "schema_version": "trtmc.qualification-result/v1",
        "case": case.id,
        "kind": "accuracy",
        "status": "passed" if passed else "failed",
        "model": case.model,
        "benchmark": case.benchmark,
        "bundle": str(bundle),
        "reference": {
            "adapter": adapter,
            "script": str(script) if script else None,
            "model": reference_result.get("model"),
        },
        "metrics": metrics,
        "gate": dict(gate),
    }


def _resolve_task_assets(case: QualificationCase, request: Mapping[str, Any]) -> dict[str, Any]:
    resolved = dict(request)
    family_root = case.source.parents[2].resolve()
    for key, value in request.items():
        if not key.endswith("_path") or not isinstance(value, str):
            continue
        path = Path(value)
        if path.is_absolute():
            continue
        path = (case.source.parent / path).resolve()
        if not path.is_relative_to(family_root) or not path.is_file():
            raise QualificationError(f"task-output asset {key!r} is unavailable: {path}")
        if key == "prompt_path":
            payload = (
                json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else None
            )
            prompt = (
                payload.get("prompt")
                if isinstance(payload, Mapping)
                else path.read_text(encoding="utf-8").strip()
            )
            if not isinstance(prompt, str) or not prompt:
                raise QualificationError(f"task-output prompt is unavailable: {path}")
            resolved.pop(key, None)
            resolved["prompt"] = prompt
        else:
            resolved[key] = str(path)
    return resolved


def _robot_control_samples(dataset: Dataset, sample_limit: int) -> list[dict[str, Any]]:
    try:
        payload = json.loads(dataset.path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise QualificationError("robot-control dataset is not valid JSON") from error
    requests = payload.get("requests") if isinstance(payload, Mapping) else None
    if not isinstance(requests, list) or len(requests) < sample_limit:
        raise QualificationError(f"robot-control dataset requires at least {sample_limit} requests")
    root = dataset.path.parent.resolve()
    selected = []
    for index, request in enumerate(requests[:sample_limit]):
        if not isinstance(request, Mapping):
            raise QualificationError(f"robot-control request {index} must be an object")
        sample = {"sample_id": str(request.get("id") or f"sample-{index}")}
        for source, target in (("image", "image_path"), ("state", "state_path")):
            relative = request.get(source)
            if not isinstance(relative, str) or not relative:
                raise QualificationError(f"robot-control request {index} has no {source}")
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise QualificationError(
                    f"robot-control request {index} {source} is unavailable: {path}"
                )
            sample[target] = str(path)
        selected.append(sample)
    return selected


def _robot_action_parity(
    case: QualificationCase,
    context: RuntimeContext,
    dataset: Dataset,
    output: Path,
) -> dict[str, Any]:
    configured = case.values
    sample_limit = _positive_int(configured.get("samples"), "accuracy.samples")
    selected = _robot_control_samples(dataset, sample_limit)
    expected = _family_image_reference(case, context, output, selected)
    actual, bundle = _candidate_outputs(
        case,
        context,
        output,
        "control",
        [
            {
                "sample_id": sample["sample_id"],
                "request": {
                    "image_path": sample["image_path"],
                    "state_path": sample["state_path"],
                },
            }
            for sample in selected
        ],
    )
    gate = configured.get("gate", {})
    if not isinstance(gate, Mapping):
        raise QualificationError("robot-action gate must be an object")
    minimum_sample_rate = float(gate.get("min_sample_pass_rate", 1.0))
    if not 0.0 <= minimum_sample_rate <= 1.0:
        raise QualificationError("robot-action min_sample_pass_rate must be in [0, 1]")
    thresholds = {key: value for key, value in gate.items() if key != "min_sample_pass_rate"}
    rows = []
    for sample, candidate_sample, reference_sample in zip(selected, actual, expected, strict=True):
        try:
            metrics = robot_action_metrics(candidate_sample, reference_sample)
            passed = robot_action_passes(metrics, thresholds)
        except ValueError as error:
            raise QualificationError(
                f"robot-action sample {sample['sample_id']} is invalid: {error}"
            ) from error
        rows.append({"sample_id": sample["sample_id"], "passed": passed, **metrics})
    pass_rate = sum(bool(row["passed"]) for row in rows) / len(rows)
    return {
        "schema_version": "trtmc.qualification-result/v1",
        "case": case.id,
        "kind": "accuracy",
        "status": "passed" if pass_rate >= minimum_sample_rate else "failed",
        "model": case.model,
        "benchmark": case.benchmark,
        "bundle": str(bundle),
        "dataset": dataset.receipt(),
        "metrics": {
            "samples": len(rows),
            "sample_pass_rate": pass_rate,
            **{name: max(float(row[name]) for row in rows) for name in thresholds},
        },
        "gate": dict(gate),
        "samples": rows,
    }


def _metric_geometry_parity(
    case: QualificationCase,
    context: RuntimeContext,
    dataset: Dataset,
    output: Path,
) -> dict[str, Any]:
    configured = case.values
    sample_limit = _positive_int(configured.get("samples"), "accuracy.samples")
    selected = _image_parity_samples(dataset, sample_limit, "metric-geometry")
    expected = _family_image_reference(case, context, output, selected)
    request = configured.get("request", {})
    if not isinstance(request, Mapping):
        raise QualificationError("metric-geometry request must be an object")
    candidate_requests = [
        {
            "sample_id": sample["sample_id"],
            "request": {**dict(request), "image_path": sample["image_path"]},
        }
        for sample in selected
    ]
    actual, bundle = _candidate_outputs(case, context, output, "geometry", candidate_requests)
    gate = configured.get("gate", {})
    if not isinstance(gate, Mapping):
        raise QualificationError("metric-geometry gate must be an object")
    minimum_sample_rate = float(gate.get("min_sample_pass_rate", 1.0))
    if not 0.0 <= minimum_sample_rate <= 1.0:
        raise QualificationError("metric-geometry min_sample_pass_rate must be in [0, 1]")
    thresholds = {key: value for key, value in gate.items() if key != "min_sample_pass_rate"}
    rows = []
    for sample, candidate_sample, reference_sample in zip(selected, actual, expected, strict=True):
        try:
            metrics = metric_geometry_metrics(candidate_sample, reference_sample)
            passed = metric_geometry_passes(metrics, thresholds)
        except (OSError, ValueError) as error:
            raise QualificationError(
                f"metric-geometry sample {sample['sample_id']} is invalid: {error}"
            ) from error
        rows.append({"sample_id": sample["sample_id"], "passed": passed, **metrics})
    pass_rate = sum(bool(row["passed"]) for row in rows) / len(rows)
    metric_names = tuple(thresholds)
    aggregate = {
        name: (
            min(float(row[name]) for row in rows)
            if name in {"mask_iou", "points_cosine"}
            else max(float(row[name]) for row in rows)
        )
        for name in metric_names
    }
    return {
        "schema_version": "trtmc.qualification-result/v1",
        "case": case.id,
        "kind": "accuracy",
        "status": "passed" if pass_rate >= minimum_sample_rate else "failed",
        "model": case.model,
        "benchmark": case.benchmark,
        "bundle": str(bundle),
        "dataset": dataset.receipt(),
        "metrics": {"samples": len(rows), "sample_pass_rate": pass_rate, **aggregate},
        "gate": dict(gate),
        "samples": rows,
    }


def _stereo_samples(dataset: Dataset, sample_limit: int) -> list[dict[str, Any]]:
    try:
        payload = json.loads(dataset.path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise QualificationError("stereo dataset is not valid JSON") from error
    requests = payload.get("requests") if isinstance(payload, Mapping) else None
    if not isinstance(requests, list) or len(requests) < sample_limit:
        raise QualificationError(f"stereo dataset requires at least {sample_limit} requests")
    root = dataset.path.parent.resolve()
    selected = []
    for index, request in enumerate(requests[:sample_limit]):
        if not isinstance(request, Mapping):
            raise QualificationError(f"stereo request {index} must be an object")
        sample = {"sample_id": str(request.get("id") or f"sample-{index}")}
        for source, target in (
            ("left_image", "left_image_path"),
            ("right_image", "right_image_path"),
        ):
            relative = request.get(source)
            if not isinstance(relative, str) or not relative:
                raise QualificationError(f"stereo request {index} has no {source}")
            image = (root / relative).resolve()
            if not image.is_relative_to(root) or not image.is_file():
                raise QualificationError(f"stereo request {index} {source} is unavailable: {image}")
            sample[target] = str(image)
        selected.append(sample)
    return selected


def _disparity_values(value: Mapping[str, Any], label: str) -> list[float]:
    height = value.get("height")
    width = value.get("width")
    count = value.get("element_count", value.get("disparity_pixels"))
    artifact = value.get("disparity_artifact")
    if (
        isinstance(height, bool)
        or not isinstance(height, int)
        or height < 1
        or isinstance(width, bool)
        or not isinstance(width, int)
        or width < 1
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count != height * width
        or not isinstance(artifact, str)
    ):
        raise QualificationError(f"{label} disparity output is incomplete")
    try:
        payload = Path(artifact).read_bytes()
    except OSError as error:
        raise QualificationError(f"{label} disparity artifact is unavailable") from error
    if len(payload) != count * 4:
        raise QualificationError(f"{label} disparity artifact has the wrong size")
    values = array("f")
    values.frombytes(payload)
    if len(values) != count:
        raise QualificationError(f"{label} disparity artifact has the wrong element count")
    return list(values)


def _stereo_disparity_parity(
    case: QualificationCase,
    context: RuntimeContext,
    dataset: Dataset,
    output: Path,
) -> dict[str, Any]:
    configured = case.values
    sample_limit = _positive_int(configured.get("samples"), "accuracy.samples")
    selected = _stereo_samples(dataset, sample_limit)
    request = configured.get("request", {})
    reference = configured.get("reference", {})
    if not isinstance(request, Mapping) or not isinstance(reference, Mapping):
        raise QualificationError("stereo Accuracy request and reference must be objects")
    command = reference.get("command")
    if not isinstance(command, str) or not command:
        raise QualificationError("stereo Accuracy reference.command must be set")
    runner = (case.source.parent / command).resolve()
    family_root = case.source.parents[2].resolve()
    if not runner.is_relative_to(family_root) or not runner.is_file():
        raise QualificationError(f"stereo reference runner is not family-owned: {runner}")
    reference_request = {
        "model": str(case.candidate["checkpoint"]),
        "revision": case.candidate.get("revision"),
        "precision": str(reference.get("precision", "fp16")),
        "request": dict(request),
        "samples": selected,
    }
    request_path = output / "reference-request.json"
    reference_path = output / "reference.json"
    _json(request_path, reference_request)
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
        timeout=7200,
        verbose=context.verbose,
    )
    if completed.returncode != 0:
        raise QualificationError(f"stereo Accuracy reference failed; see {output}")
    try:
        expected = json.loads(reference_path.read_text(encoding="utf-8")).get("samples")
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationError("stereo Accuracy reference returned unreadable output") from error
    if not isinstance(expected, list) or len(expected) != len(selected):
        raise QualificationError("stereo Accuracy reference returned an invalid sample set")
    candidate_requests = [
        {
            "sample_id": sample["sample_id"],
            "request": {
                **dict(request),
                "left_image_path": sample["left_image_path"],
                "right_image_path": sample["right_image_path"],
            },
        }
        for sample in selected
    ]
    actual, bundle = _candidate_outputs(case, context, output, "disparity", candidate_requests)
    gate = configured.get("gate", {})
    if not isinstance(gate, Mapping):
        raise QualificationError("stereo Accuracy gate must be an object")
    minimum_cosine = float(gate.get("min_cosine", 0.999))
    maximum_mean_error = float(gate.get("max_mean_abs_error", 0.5))
    maximum_bad_fraction = float(gate.get("max_bad_2px_fraction", 0.03))
    minimum_sample_rate = float(gate.get("min_sample_pass_rate", 1.0))
    if not 0.0 <= minimum_cosine <= 1.0 or not 0.0 <= minimum_sample_rate <= 1.0:
        raise QualificationError("stereo cosine and pass-rate gates must be in [0, 1]")
    if (
        not math.isfinite(maximum_mean_error)
        or maximum_mean_error < 0.0
        or not 0.0 <= maximum_bad_fraction <= 1.0
    ):
        raise QualificationError("stereo error gates are invalid")

    rows = []
    for sample, candidate_sample, reference_sample in zip(selected, actual, expected, strict=True):
        candidate_values = _disparity_values(candidate_sample, "candidate")
        reference_values = _disparity_values(reference_sample, "reference")
        if len(candidate_values) != len(reference_values):
            raise QualificationError("candidate and reference disparity shapes differ")
        candidate_valid = all(math.isfinite(value) and value >= 0.0 for value in candidate_values)
        reference_valid = all(math.isfinite(value) and value >= 0.0 for value in reference_values)
        dot = math.fsum(
            left * right for left, right in zip(candidate_values, reference_values, strict=True)
        )
        candidate_norm = math.sqrt(math.fsum(value * value for value in candidate_values))
        reference_norm = math.sqrt(math.fsum(value * value for value in reference_values))
        cosine = (
            dot / (candidate_norm * reference_norm) if candidate_norm and reference_norm else 0.0
        )
        differences = [
            abs(left - right)
            for left, right in zip(candidate_values, reference_values, strict=True)
        ]
        mean_error = math.fsum(differences) / len(differences)
        bad_fraction = sum(value > 2.0 for value in differences) / len(differences)
        passed = (
            candidate_valid
            and reference_valid
            and cosine >= minimum_cosine
            and mean_error <= maximum_mean_error
            and bad_fraction <= maximum_bad_fraction
        )
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "passed": passed,
                "cosine": cosine,
                "mean_abs_error": mean_error,
                "bad_2px_fraction": bad_fraction,
            }
        )
    pass_rate = sum(bool(row["passed"]) for row in rows) / len(rows)
    return {
        "schema_version": "trtmc.qualification-result/v1",
        "case": case.id,
        "kind": "accuracy",
        "status": "passed" if pass_rate >= minimum_sample_rate else "failed",
        "model": case.model,
        "benchmark": case.benchmark,
        "bundle": str(bundle),
        "dataset": dataset.receipt(),
        "metrics": {
            "samples": len(rows),
            "sample_pass_rate": pass_rate,
            "min_cosine": min(row["cosine"] for row in rows),
            "max_mean_abs_error": max(row["mean_abs_error"] for row in rows),
            "max_bad_2px_fraction": max(row["bad_2px_fraction"] for row in rows),
        },
        "gate": {
            "min_cosine": minimum_cosine,
            "max_mean_abs_error": maximum_mean_error,
            "max_bad_2px_fraction": maximum_bad_fraction,
            "min_sample_pass_rate": minimum_sample_rate,
        },
        "samples": rows,
    }


def _text_generation_parity(
    case: QualificationCase,
    context: RuntimeContext,
    definition: Mapping[str, Any],
    dataset: Dataset,
    output: Path,
) -> dict[str, Any]:
    configured = case.values
    sample_limit = _positive_int(configured.get("samples"), "accuracy.samples")
    payload = json.loads(dataset.path.read_text(encoding="utf-8"))
    requests = payload.get("requests") if isinstance(payload, Mapping) else None
    if not isinstance(requests, list) or not requests:
        raise QualificationError("text-generation dataset must contain a non-empty requests list")
    selected_requests, selection = _select_dataset_rows(
        requests, definition, sample_limit, "text-generation"
    )
    selected = []
    for index, request in enumerate(selected_requests):
        if not isinstance(request, Mapping):
            raise QualificationError(f"text-generation request {index} must be an object")
        selected.append(
            {
                "sample_id": str(
                    request.get("id") or request.get("sample_id") or f"sample-{index}"
                ),
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
    reference_model = reference.get("model", case.candidate["checkpoint"])
    if not isinstance(reference_model, str) or not reference_model:
        raise QualificationError("accuracy.reference.model must be a non-empty string")
    reference_revision = reference.get("revision")
    if reference_revision is None and reference_model == case.candidate["checkpoint"]:
        reference_revision = case.candidate.get("revision")
    if reference_revision is not None and (
        not isinstance(reference_revision, str) or not reference_revision
    ):
        raise QualificationError("accuracy.reference.revision must be a non-empty string")
    trust_remote_code = reference.get(
        "trust_remote_code", case.candidate.get("trust_remote_code", False)
    )
    if not isinstance(trust_remote_code, bool):
        raise QualificationError("accuracy.reference.trust_remote_code must be a boolean")
    reference_request = {
        "model": reference_model,
        "revision": reference_revision,
        "trust_remote_code": trust_remote_code,
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
    runner_value = reference.get("command")
    if runner_value is None:
        runner = (
            context.repository / "qualification_tests/benchmark_qualification/references/hf_text_generation.py"
        )
    else:
        if not isinstance(runner_value, str) or not runner_value:
            raise QualificationError("text-generation reference.command must be a string")
        runner = (case.source.parent / runner_value).resolve()
        family_root = case.source.parents[2].resolve()
        if family_root not in runner.parents or not runner.is_file():
            raise QualificationError(
                f"text-generation reference runner is not family-owned: {runner}"
            )
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
        "dataset": {**dataset.receipt(), "selection": selection},
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
    if not all(isinstance(value, str) and value for value in (model_class, tokenizer_class)):
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
    runner = context.repository / "qualification_tests/benchmark_qualification/references/hf_encoder.py"
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
    candidate, bundle = _candidate_outputs(case, context, output, operation, candidate_requests)
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


def _reranking_score_parity(
    case: QualificationCase,
    context: RuntimeContext,
    dataset: Dataset,
    output: Path,
) -> dict[str, Any]:
    configured = case.values
    sample_limit = _positive_int(configured.get("samples"), "accuracy.samples")
    samples = _sts_reranking_samples(dataset.path, sample_limit)
    reference = configured.get("reference", {})
    gate = configured.get("gate", {})
    if not isinstance(reference, Mapping) or not isinstance(gate, Mapping):
        raise QualificationError("reranking reference and gate must be objects")
    runner_value = reference.get("command")
    if not isinstance(runner_value, str) or not runner_value:
        raise QualificationError("reranking reference.command must be set")
    runner = (case.source.parent / runner_value).resolve()
    family_root = case.source.parents[2].resolve()
    if family_root not in runner.parents or not runner.is_file():
        raise QualificationError(f"reranking reference runner is not family-owned: {runner}")
    request_path = output / "reference-request.json"
    reference_path = output / "reference.json"
    _json(
        request_path,
        {
            "model": str(case.candidate["checkpoint"]),
            "revision": case.candidate.get("revision"),
            "precision": str(reference.get("precision", "fp32")),
            "samples": samples,
        },
    )
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
        raise QualificationError(f"reranking Accuracy reference failed; see {output}")
    expected = json.loads(reference_path.read_text(encoding="utf-8")).get("samples")
    if not isinstance(expected, list) or len(expected) != len(samples):
        raise QualificationError("reranking reference returned an invalid sample set")
    candidate_requests = [
        {
            "sample_id": sample["sample_id"],
            "request": {"query": sample["query"], "documents": sample["documents"]},
        }
        for sample in samples
    ]
    actual, bundle = _candidate_outputs(case, context, output, "rerank", candidate_requests)
    maximum_error = float(gate.get("max_score_abs_error", 0.05))
    minimum_rate = float(gate.get("min_sample_pass_rate", 1.0))
    if maximum_error < 0.0 or not math.isfinite(maximum_error):
        raise QualificationError("max_score_abs_error must be finite and nonnegative")
    if not 0.0 <= minimum_rate <= 1.0:
        raise QualificationError("min_sample_pass_rate must be in [0, 1]")

    rows = []
    for sample, candidate_sample, reference_sample in zip(samples, actual, expected, strict=True):
        candidate_scores = _reranking_scores(candidate_sample, "candidate")
        reference_scores = _reranking_scores(reference_sample, "reference")
        if len(candidate_scores) != len(reference_scores) or len(reference_scores) < 2:
            raise QualificationError("reranking reference and candidate score counts differ")
        errors = [
            abs(left - right)
            for left, right in zip(candidate_scores, reference_scores, strict=True)
        ]
        candidate_order = _reranking_order(candidate_scores)
        reference_order = _reranking_order(reference_scores)
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "passed": max(errors) <= maximum_error and candidate_order == reference_order,
                "max_score_abs_error": max(errors),
                "candidate_order": candidate_order,
                "reference_order": reference_order,
            }
        )
    pass_rate = sum(bool(row["passed"]) for row in rows) / len(rows)
    passed = pass_rate >= minimum_rate
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
            "sample_pass_rate": pass_rate,
            "max_score_abs_error": max(row["max_score_abs_error"] for row in rows),
        },
        "gate": {
            "max_score_abs_error": maximum_error,
            "min_sample_pass_rate": minimum_rate,
        },
        "samples": rows,
    }


def _reranking_scores(summary: Mapping[str, Any], label: str) -> list[float]:
    values = summary.get("scores")
    if not isinstance(values, list) or not values:
        raise QualificationError(f"{label} reranking output has no scores")
    scores = [float(value) for value in values]
    if not all(math.isfinite(value) for value in scores):
        raise QualificationError(f"{label} reranking output has non-finite scores")
    return scores


def _reranking_order(scores: Sequence[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


def _sts_reranking_samples(path: Path, count: int) -> list[dict[str, Any]]:
    pairs = []
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
            if not all(
                isinstance(value, str) and value.strip() for value in (sentence1, sentence2)
            ):
                raise QualificationError(
                    f"STSBenchmark row {dataset_index} must contain two sentences"
                )
            pairs.append((str(sentence1).strip(), str(sentence2).strip()))
            if len(pairs) == count * 2:
                break
    if len(pairs) != count * 2:
        raise QualificationError(
            f"STSBenchmark contains {len(pairs)} usable rows; {count * 2} required"
        )
    return [
        {
            "sample_id": f"stsbenchmark-rerank-{index:06d}",
            "query": pairs[index * 2][0],
            "documents": [pairs[index * 2][1], pairs[index * 2 + 1][1]],
        }
        for index in range(count)
    ]


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
            if not all(
                isinstance(value, str) and value.strip() for value in (sentence1, sentence2)
            ):
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
                raise QualificationError(f"STSBenchmark row {dataset_index} has a non-finite score")
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
        reference_similarity = _vector_cosine(expected["sentence1"][1], expected["sentence2"][1])
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
        "passed" if pass_rate >= minimum_rate and max_pair_delta <= maximum_pair_delta else "failed"
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
        raise QualificationError(f"encoder vector dimensions differ: {len(left)} != {len(right)}")
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise QualificationError("encoder vector must have a nonzero norm")
    cosine = math.fsum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
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
    return (
        math.fsum(a * b for a, b in zip(centered_left, centered_right, strict=True)) / denominator
    )


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
    descriptor = write_model_descriptor(case, output, first_request, context=context)
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
    candidate_root = candidate_output.resolve()

    def resolve_artifact(artifact_root: Path, value: str) -> str:
        path = Path(value)
        resolved = path.resolve() if path.is_absolute() else (artifact_root / path).resolve()
        if not resolved.is_relative_to(artifact_root):
            raise QualificationError("trtmc-bench returned an unsafe artifact path")
        return str(resolved)

    for cell in cells:
        if not isinstance(cell, Mapping) or cell.get("status") != "completed":
            raise QualificationError("trtmc-bench reported a failed Accuracy request")
        summary = cell.get("output_summary")
        if not isinstance(summary, Mapping):
            raise QualificationError("trtmc-bench omitted an Accuracy output")
        resolved_summary = dict(summary)
        artifact_dir = cell.get("artifact_dir")
        if isinstance(artifact_dir, str) and artifact_dir:
            artifact_root = (candidate_output / artifact_dir).resolve()
            if not artifact_root.is_relative_to(candidate_root):
                raise QualificationError("trtmc-bench returned an unsafe artifact directory")
            for name, value in tuple(resolved_summary.items()):
                if name.endswith("_artifact") and isinstance(value, str):
                    resolved_summary[name] = resolve_artifact(artifact_root, value)
                elif name.endswith("_artifacts") and isinstance(value, list):
                    resolved_summary[name] = [
                        resolve_artifact(artifact_root, item)
                        for item in value
                        if isinstance(item, str)
                    ]
        outputs.append(resolved_summary)
    return outputs, bundle


def _image_parity_samples(dataset: Dataset, sample_limit: int, task: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(dataset.path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise QualificationError(f"{task} dataset is not valid JSON") from error
    requests = payload.get("requests") if isinstance(payload, Mapping) else None
    if not isinstance(requests, list) or len(requests) < sample_limit:
        raise QualificationError(f"{task} dataset requires at least {sample_limit} requests")
    root = dataset.path.parent.resolve()
    selected = []
    for index, request in enumerate(requests[:sample_limit]):
        relative = request.get("image") if isinstance(request, Mapping) else None
        if not isinstance(relative, str) or not relative:
            raise QualificationError(f"{task} request {index} has no image")
        image = (root / relative).resolve()
        if root not in image.parents or not image.is_file():
            raise QualificationError(f"{task} request {index} image is unavailable: {image}")
        sample = {
            "sample_id": str(request.get("id") or f"sample-{index}"),
            "image_path": str(image),
        }
        label_name = request.get("label_name")
        if isinstance(label_name, str) and label_name.strip():
            sample["label_name"] = label_name.strip()
        selected.append(sample)
    return selected


def _family_image_reference(
    case: QualificationCase,
    context: RuntimeContext,
    output: Path,
    selected: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reference = case.values.get("reference")
    if not isinstance(reference, Mapping):
        raise QualificationError("image Accuracy reference must be an object")
    runner_value = reference.get("command")
    if not isinstance(runner_value, str) or not runner_value:
        raise QualificationError("image Accuracy reference.command must be set")
    runner = (case.source.parent / runner_value).resolve()
    family_root = case.source.parents[2].resolve()
    if family_root not in runner.parents or not runner.is_file():
        raise QualificationError(f"image reference runner is not family-owned: {runner}")
    request = case.values.get("request", {})
    if not isinstance(request, Mapping):
        raise QualificationError("image Accuracy request must be an object")
    reference_request = {
        "model": str(case.candidate["checkpoint"]),
        "revision": case.candidate.get("revision"),
        "precision": str(reference.get("precision", "fp32")),
        "request": dict(request),
        "samples": list(selected),
    }
    request_path = output / "reference-request.json"
    reference_path = output / "reference.json"
    _json(request_path, reference_request)
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
        timeout=7200,
        verbose=context.verbose,
    )
    if completed.returncode != 0:
        raise QualificationError(f"image Accuracy reference failed; see {output}")
    expected = json.loads(reference_path.read_text(encoding="utf-8")).get("samples")
    if not isinstance(expected, list) or len(expected) != len(selected):
        raise QualificationError("image Accuracy reference returned an invalid sample set")
    return expected


def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _detections(value: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    boxes = value.get("boxes")
    scores = value.get("scores")
    classes = value.get("class_ids", value.get("classes"))
    if not isinstance(boxes, list) or not isinstance(scores, list) or not isinstance(classes, list):
        raise QualificationError(f"{label} detection output is incomplete")
    if boxes and isinstance(boxes[0], list):
        rows = boxes
    else:
        if len(boxes) % 4:
            raise QualificationError(f"{label} detection boxes are not xyxy rows")
        rows = [boxes[index : index + 4] for index in range(0, len(boxes), 4)]
    if len(rows) != len(scores) or len(rows) != len(classes):
        raise QualificationError(f"{label} detection arrays have different lengths")
    detections = []
    for index, (box, score, class_id) in enumerate(zip(rows, scores, classes, strict=True)):
        if (
            not isinstance(box, list)
            or len(box) != 4
            or isinstance(class_id, bool)
            or not isinstance(class_id, int)
        ):
            raise QualificationError(f"{label} detection {index} is invalid")
        numeric_box = [float(coordinate) for coordinate in box]
        numeric_score = float(score)
        if not all(math.isfinite(value) for value in (*numeric_box, numeric_score)):
            raise QualificationError(f"{label} detection {index} is not finite")
        detections.append({"box": numeric_box, "score": numeric_score, "class_id": class_id})
    return detections


def _coco_detection_samples(
    dataset: Dataset, sample_limit: int
) -> tuple[list[dict[str, Any]], str]:
    try:
        payload = json.loads(dataset.path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise QualificationError("COCO object-detection dataset is not valid JSON") from error
    requests = payload.get("requests") if isinstance(payload, Mapping) else None
    if not isinstance(requests, list) or len(requests) < sample_limit:
        raise QualificationError(
            f"COCO object-detection dataset requires at least {sample_limit} requests"
        )
    root = dataset.path.parent.resolve()
    selected = []
    for index, request in enumerate(requests[:sample_limit]):
        if not isinstance(request, Mapping):
            raise QualificationError(f"COCO object-detection request {index} is invalid")
        relative = request.get("image")
        annotations = request.get("annotations")
        if not isinstance(relative, str) or not relative:
            raise QualificationError(f"COCO object-detection request {index} has no image")
        if not isinstance(annotations, list) or not annotations:
            raise QualificationError(
                f"COCO object-detection request {index} has no ground-truth annotations"
            )
        image = (root / relative).resolve()
        if root not in image.parents or not image.is_file():
            raise QualificationError(
                f"COCO object-detection request {index} image is unavailable: {image}"
            )
        ground_truth = []
        for annotation_index, annotation in enumerate(annotations):
            if not isinstance(annotation, Mapping):
                raise QualificationError(
                    f"COCO object-detection annotation {index}:{annotation_index} is invalid"
                )
            box = annotation.get("bbox_xyxy")
            category_id = annotation.get("category_id")
            category_index = annotation.get("category_index")
            if (
                not isinstance(box, list)
                or len(box) != 4
                or any(not isinstance(value, (int, float)) for value in box)
                or isinstance(category_id, bool)
                or not isinstance(category_id, int)
                or isinstance(category_index, bool)
                or not isinstance(category_index, int)
            ):
                raise QualificationError(
                    f"COCO object-detection annotation {index}:{annotation_index} is invalid"
                )
            numeric_box = [float(value) for value in box]
            if (
                not all(math.isfinite(value) for value in numeric_box)
                or numeric_box[2] <= numeric_box[0]
                or numeric_box[3] <= numeric_box[1]
            ):
                raise QualificationError(
                    f"COCO object-detection annotation {index}:{annotation_index} has an invalid box"
                )
            ground_truth.append(
                {
                    "box": numeric_box,
                    "category_id": category_id,
                    "category_index": category_index,
                }
            )
        selected.append(
            {
                "sample_id": str(request.get("id") or f"sample-{index}"),
                "image_path": str(image),
                "annotations": ground_truth,
            }
        )
    sampling = payload.get("sampling")
    return selected, str(sampling) if isinstance(sampling, str) else "fixed manifest order"


def _coco_ap_at_iou(
    selected: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    label_field: str,
    iou_threshold: float,
) -> float:
    ground_truth: dict[int, dict[str, list[Sequence[float]]]] = {}
    predictions: dict[int, list[tuple[float, str, Sequence[float]]]] = {}
    for sample, output in zip(selected, outputs, strict=True):
        sample_id = str(sample["sample_id"])
        for annotation in sample["annotations"]:
            class_id = int(annotation[label_field])
            ground_truth.setdefault(class_id, {}).setdefault(sample_id, []).append(
                annotation["box"]
            )
        for detection in sorted(
            _detections(output, "object-detection output"),
            key=lambda value: -float(value["score"]),
        )[:100]:
            predictions.setdefault(int(detection["class_id"]), []).append(
                (float(detection["score"]), sample_id, detection["box"])
            )

    category_aps = []
    for class_id, sample_boxes in ground_truth.items():
        total_ground_truth = sum(len(boxes) for boxes in sample_boxes.values())
        matched: dict[str, set[int]] = defaultdict(set)
        true_positives = false_positives = 0
        precisions = []
        recalls = []
        for _, sample_id, box in sorted(predictions.get(class_id, []), key=lambda value: -value[0]):
            candidates = [
                (_box_iou(box, expected), index)
                for index, expected in enumerate(sample_boxes.get(sample_id, []))
                if index not in matched[sample_id]
            ]
            if candidates and max(candidates)[0] >= iou_threshold:
                _, matched_index = max(candidates)
                matched[sample_id].add(matched_index)
                true_positives += 1
            else:
                false_positives += 1
            precisions.append(true_positives / (true_positives + false_positives))
            recalls.append(true_positives / total_ground_truth)
        category_aps.append(
            sum(
                max(
                    (
                        precision
                        for precision, recall in zip(precisions, recalls, strict=True)
                        if recall >= point
                    ),
                    default=0.0,
                )
                for point in (index / 100.0 for index in range(101))
            )
            / 101.0
        )
    if not category_aps:
        raise QualificationError("COCO object-detection subset has no ground truth")
    return sum(category_aps) / len(category_aps)


def _coco_detection_metrics(
    selected: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    label_field: str,
) -> dict[str, float]:
    if len(selected) != len(outputs):
        raise QualificationError("COCO object-detection output count differs from the dataset")
    thresholds = [0.5 + index * 0.05 for index in range(10)]
    average_precisions = [
        _coco_ap_at_iou(selected, outputs, label_field, threshold) for threshold in thresholds
    ]
    return {
        "map_50_95": sum(average_precisions) / len(average_precisions),
        "map_50": average_precisions[0],
    }


def _coco_object_detection_accuracy(
    case: QualificationCase,
    context: RuntimeContext,
    dataset: Dataset,
    output: Path,
) -> dict[str, Any]:
    configured = case.values
    sample_limit = _positive_int(configured.get("samples"), "accuracy.samples")
    selected, sampling = _coco_detection_samples(dataset, sample_limit)
    expected = _family_image_reference(case, context, output, selected)
    request = configured.get("request", {})
    if not isinstance(request, Mapping):
        raise QualificationError("COCO object-detection request must be an object")
    candidate_requests = [
        {
            "sample_id": sample["sample_id"],
            "request": {**dict(request), "image_path": sample["image_path"]},
        }
        for sample in selected
    ]
    actual, bundle = _candidate_outputs(case, context, output, "detect", candidate_requests)
    gate = configured.get("gate", {})
    if not isinstance(gate, Mapping):
        raise QualificationError("COCO object-detection gate must be an object")
    label_spaces = {
        "coco-category-id": "category_id",
        "coco-contiguous-80": "category_index",
    }
    label_space = configured.get("label_space")
    if label_space not in label_spaces:
        raise QualificationError(
            "COCO object-detection label_space must be coco-category-id or coco-contiguous-80"
        )
    map_50_95_limit = float(gate.get("max_map_50_95_drop", 0.02))
    map_50_limit = float(gate.get("max_map_50_drop", 0.02))
    if not all(math.isfinite(value) and value >= 0.0 for value in (map_50_95_limit, map_50_limit)):
        raise QualificationError("COCO object-detection mAP drops must be finite and nonnegative")
    label_field = label_spaces[str(label_space)]
    candidate_metrics = _coco_detection_metrics(selected, actual, label_field)
    reference_metrics = _coco_detection_metrics(selected, expected, label_field)
    map_50_95_drop = reference_metrics["map_50_95"] - candidate_metrics["map_50_95"]
    map_50_drop = reference_metrics["map_50"] - candidate_metrics["map_50"]
    passed = map_50_95_drop <= map_50_95_limit and map_50_drop <= map_50_limit
    return {
        "schema_version": "trtmc.qualification-result/v1",
        "case": case.id,
        "kind": "accuracy",
        "status": "passed" if passed else "failed",
        "model": case.model,
        "benchmark": case.benchmark,
        "bundle": str(bundle),
        "dataset": {**dataset.receipt(), "sampling": sampling},
        "metrics": {
            "samples": len(selected),
            "label_space": label_space,
            "candidate_map_50_95": candidate_metrics["map_50_95"],
            "reference_map_50_95": reference_metrics["map_50_95"],
            "map_50_95_drop": map_50_95_drop,
            "candidate_map_50": candidate_metrics["map_50"],
            "reference_map_50": reference_metrics["map_50"],
            "map_50_drop": map_50_drop,
        },
        "gate": {
            "max_map_50_95_drop": map_50_95_limit,
            "max_map_50_drop": map_50_limit,
        },
    }


def _semantic_mask(value: Mapping[str, Any], label: str) -> tuple[int, int, list[int]]:
    height = value.get("height")
    width = value.get("width")
    mask = value.get("mask")
    if (
        isinstance(height, bool)
        or not isinstance(height, int)
        or height < 1
        or isinstance(width, bool)
        or not isinstance(width, int)
        or width < 1
        or not isinstance(mask, list)
        or len(mask) != height * width
    ):
        raise QualificationError(f"{label} semantic-segmentation output is invalid")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in mask):
        raise QualificationError(f"{label} semantic-segmentation mask is not integral")
    return height, width, mask


def _binary_masks(value: Mapping[str, Any], label: str) -> tuple[int, int, list[list[bool]]]:
    height = value.get("height")
    width = value.get("width")
    count = value.get("num_masks")
    masks = value.get("masks")
    kind = value.get("mask_kind", "logits")
    if (
        isinstance(height, bool)
        or not isinstance(height, int)
        or height < 1
        or isinstance(width, bool)
        or not isinstance(width, int)
        or width < 1
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or not isinstance(masks, list)
        or len(masks) != count * height * width
        or kind not in {"logits", "probability", "binary"}
    ):
        raise QualificationError(f"{label} prompted-segmentation output is invalid")
    try:
        numeric = [float(value) for value in masks]
    except (TypeError, ValueError) as error:
        raise QualificationError(f"{label} prompted-segmentation masks are not numeric") from error
    if not all(math.isfinite(value) for value in numeric):
        raise QualificationError(f"{label} prompted-segmentation masks are not finite")
    threshold = 0.0 if kind == "logits" else 0.5
    area = height * width
    return (
        height,
        width,
        [
            [value > threshold for value in numeric[index : index + area]]
            for index in range(0, len(numeric), area)
        ],
    )


def _mask_iou(left: Sequence[bool], right: Sequence[bool]) -> float:
    intersection = sum(a and b for a, b in zip(left, right, strict=True))
    union = sum(a or b for a, b in zip(left, right, strict=True))
    return intersection / union if union else 1.0


def _match_masks(
    candidate: Sequence[Sequence[bool]], reference: Sequence[Sequence[bool]]
) -> list[float]:
    matched_reference: set[int] = set()
    matches = []
    for mask in candidate:
        choices = [
            (_mask_iou(mask, expected), index)
            for index, expected in enumerate(reference)
            if index not in matched_reference
        ]
        if not choices:
            continue
        iou, index = max(choices)
        matched_reference.add(index)
        matches.append(iou)
    return matches


def _prompted_segmentation_parity(
    case: QualificationCase,
    context: RuntimeContext,
    dataset: Dataset,
    output: Path,
) -> dict[str, Any]:
    configured = case.values
    sample_limit = _positive_int(configured.get("samples"), "accuracy.samples")
    selected = _image_parity_samples(dataset, sample_limit, "prompted-segmentation")
    expected = _family_image_reference(case, context, output, selected)
    request = configured.get("request", {})
    if not isinstance(request, Mapping):
        raise QualificationError("prompted-segmentation request must be an object")
    candidate_requests = [
        {
            "sample_id": sample["sample_id"],
            "request": {**dict(request), "image_path": sample["image_path"]},
        }
        for sample in selected
    ]
    actual, bundle = _candidate_outputs(
        case, context, output, "segment_prompted", candidate_requests
    )
    gate = configured.get("gate", {})
    if not isinstance(gate, Mapping):
        raise QualificationError("prompted-segmentation gate must be an object")
    minimum_iou = float(gate.get("min_mask_iou", 0.7))
    minimum_match_rate = float(gate.get("min_mask_match_rate", 1.0))
    minimum_sample_rate = float(gate.get("min_sample_pass_rate", 1.0))
    if not all(
        0.0 <= value <= 1.0 for value in (minimum_iou, minimum_match_rate, minimum_sample_rate)
    ):
        raise QualificationError("prompted-segmentation gates must be in [0, 1]")

    rows = []
    total_matches = total_masks = 0
    for sample, candidate_sample, reference_sample in zip(selected, actual, expected, strict=True):
        left_height, left_width, left = _binary_masks(candidate_sample, "candidate")
        right_height, right_width, right = _binary_masks(reference_sample, "reference")
        matches = (
            _match_masks(left, right)
            if (left_height, left_width) == (right_height, right_width)
            else []
        )
        denominator = max(len(left), len(right), 1)
        match_rate = len(matches) / denominator
        minimum_sample_iou = min(matches, default=0.0)
        passed = match_rate >= minimum_match_rate and minimum_sample_iou >= minimum_iou
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "passed": passed,
                "candidate_masks": len(left),
                "reference_masks": len(right),
                "mask_match_rate": match_rate,
                "min_mask_iou": minimum_sample_iou,
            }
        )
        total_matches += len(matches)
        total_masks += denominator
    sample_pass_rate = sum(bool(row["passed"]) for row in rows) / len(rows)
    passed = sample_pass_rate >= minimum_sample_rate
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
            "sample_pass_rate": sample_pass_rate,
            "mask_match_rate": total_matches / total_masks,
            "min_mask_iou": min(row["min_mask_iou"] for row in rows),
        },
        "gate": {
            "min_mask_iou": minimum_iou,
            "min_mask_match_rate": minimum_match_rate,
            "min_sample_pass_rate": minimum_sample_rate,
        },
        "samples": rows,
    }


def _instance_masks(
    value: Mapping[str, Any], label: str
) -> tuple[int, int, list[list[bool]], list[float], list[list[float]]]:
    height, width, masks = _binary_masks(value, label)
    scores = value.get("iou_scores")
    boxes = value.get("boxes")
    if (
        not isinstance(scores, list)
        or len(scores) != len(masks)
        or not isinstance(boxes, list)
        or len(boxes) != len(masks)
        or value.get("box_coordinates") != "original_image_pixels_xyxy"
    ):
        raise QualificationError(f"{label} instance output is incomplete")
    numeric_scores = [float(score) for score in scores]
    numeric_boxes = []
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4:
            raise QualificationError(f"{label} instance box is invalid")
        numeric_boxes.append([float(coordinate) for coordinate in box])
    if not all(
        math.isfinite(number)
        for number in (*numeric_scores, *(value for box in numeric_boxes for value in box))
    ):
        raise QualificationError(f"{label} instance scores or boxes are not finite")
    return height, width, masks, numeric_scores, numeric_boxes


def _match_instances(
    candidate: tuple[int, int, list[list[bool]], list[float], list[list[float]]],
    reference: tuple[int, int, list[list[bool]], list[float], list[list[float]]],
) -> list[dict[str, float]]:
    if candidate[:2] != reference[:2]:
        return []
    matched_reference: set[int] = set()
    matches = []
    for index, mask in enumerate(candidate[2]):
        choices = [
            (_mask_iou(mask, expected), expected_index)
            for expected_index, expected in enumerate(reference[2])
            if expected_index not in matched_reference
        ]
        if not choices:
            continue
        mask_iou, expected_index = max(choices)
        matched_reference.add(expected_index)
        matches.append(
            {
                "mask_iou": mask_iou,
                "box_iou": _box_iou(candidate[4][index], reference[4][expected_index]),
                "score_abs_error": abs(candidate[3][index] - reference[3][expected_index]),
            }
        )
    return matches


def _text_prompted_instance_segmentation_parity(
    case: QualificationCase,
    context: RuntimeContext,
    dataset: Dataset,
    output: Path,
) -> dict[str, Any]:
    configured = case.values
    sample_limit = _positive_int(configured.get("samples"), "accuracy.samples")
    selected = _image_parity_samples(dataset, sample_limit, "text-prompted-segmentation")
    if any(not sample.get("label_name") for sample in selected):
        raise QualificationError("text-prompted segmentation samples require label_name")
    expected = _family_image_reference(case, context, output, selected)
    candidate_requests = [
        {
            "sample_id": sample["sample_id"],
            "request": {
                "image_path": sample["image_path"],
                "prompt": sample["label_name"],
            },
        }
        for sample in selected
    ]
    actual, bundle = _candidate_outputs(
        case, context, output, "segment_prompted", candidate_requests
    )
    gate = configured.get("gate", {})
    if not isinstance(gate, Mapping):
        raise QualificationError("text-prompted segmentation gate must be an object")
    minimum_mask_iou = float(gate.get("min_mask_iou", 0.7))
    minimum_match_rate = float(gate.get("min_mask_match_rate", 1.0))
    minimum_box_iou = float(gate.get("min_box_iou", 0.9))
    maximum_score_error = float(gate.get("max_score_abs_error", 0.05))
    minimum_sample_rate = float(gate.get("min_sample_pass_rate", 1.0))
    if not all(
        0.0 <= value <= 1.0
        for value in (
            minimum_mask_iou,
            minimum_match_rate,
            minimum_box_iou,
            maximum_score_error,
            minimum_sample_rate,
        )
    ):
        raise QualificationError("text-prompted segmentation gates must be in [0, 1]")

    rows = []
    total_matches = total_masks = 0
    for sample, candidate_sample, reference_sample in zip(selected, actual, expected, strict=True):
        candidate = _instance_masks(candidate_sample, "candidate")
        reference = _instance_masks(reference_sample, "reference")
        matches = _match_instances(candidate, reference)
        denominator = max(len(candidate[2]), len(reference[2]), 1)
        match_rate = len(matches) / denominator
        minimum_sample_mask_iou = min((match["mask_iou"] for match in matches), default=0.0)
        minimum_sample_box_iou = min((match["box_iou"] for match in matches), default=0.0)
        maximum_sample_score_error = max(
            (match["score_abs_error"] for match in matches), default=1.0
        )
        passed = (
            match_rate >= minimum_match_rate
            and minimum_sample_mask_iou >= minimum_mask_iou
            and minimum_sample_box_iou >= minimum_box_iou
            and maximum_sample_score_error <= maximum_score_error
        )
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "prompt": sample["label_name"],
                "passed": passed,
                "candidate_masks": len(candidate[2]),
                "reference_masks": len(reference[2]),
                "mask_match_rate": match_rate,
                "min_mask_iou": minimum_sample_mask_iou,
                "min_box_iou": minimum_sample_box_iou,
                "max_score_abs_error": maximum_sample_score_error,
            }
        )
        total_matches += len(matches)
        total_masks += denominator
    sample_pass_rate = sum(bool(row["passed"]) for row in rows) / len(rows)
    passed = sample_pass_rate >= minimum_sample_rate
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
            "sample_pass_rate": sample_pass_rate,
            "mask_match_rate": total_matches / total_masks,
            "min_mask_iou": min(row["min_mask_iou"] for row in rows),
            "min_box_iou": min(row["min_box_iou"] for row in rows),
            "max_score_abs_error": max(row["max_score_abs_error"] for row in rows),
        },
        "gate": {
            "min_mask_iou": minimum_mask_iou,
            "min_mask_match_rate": minimum_match_rate,
            "min_box_iou": minimum_box_iou,
            "max_score_abs_error": maximum_score_error,
            "min_sample_pass_rate": minimum_sample_rate,
        },
        "samples": rows,
    }


def _normalized_answer(value: Any) -> str:
    return " ".join(str(value or "").casefold().split()).strip(".,!?;:'\"")


def _normalized_edit_distance(left: str, right: str) -> float:
    if left == right:
        return 0.0
    if not left or not right:
        return 1.0
    previous = list(range(len(right) + 1))
    for row, left_character in enumerate(left, start=1):
        current = [row]
        for column, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[column - 1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1] / max(len(left), len(right))


def _vision_language_text_parity(
    case: QualificationCase,
    context: RuntimeContext,
    dataset: Dataset,
    output: Path,
) -> dict[str, Any]:
    configured = case.values
    sample_limit = _positive_int(configured.get("samples"), "accuracy.samples")
    selected = _image_parity_samples(dataset, sample_limit, "vision-language")
    expected = _family_image_reference(case, context, output, selected)
    request = configured.get("request", {})
    if not isinstance(request, Mapping):
        raise QualificationError("vision-language request must be an object")
    candidate_requests = [
        {
            "sample_id": sample["sample_id"],
            "request": {**dict(request), "image_path": sample["image_path"]},
        }
        for sample in selected
    ]
    actual, bundle = _candidate_outputs(case, context, output, "generate", candidate_requests)
    gate = configured.get("gate", {})
    if not isinstance(gate, Mapping):
        raise QualificationError("vision-language gate must be an object")
    maximum_distance = float(gate.get("max_normalized_edit_distance", 0.15))
    minimum_sample_rate = float(gate.get("min_sample_pass_rate", 1.0))
    if not 0.0 <= maximum_distance <= 1.0 or not 0.0 <= minimum_sample_rate <= 1.0:
        raise QualificationError("vision-language gates must be in [0, 1]")

    rows = []
    for sample, candidate_sample, reference_sample in zip(selected, actual, expected, strict=True):
        candidate_text = _normalized_answer(candidate_sample.get("text"))
        reference_text = _normalized_answer(reference_sample.get("text"))
        distance = _normalized_edit_distance(candidate_text, reference_text)
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "passed": bool(candidate_text) and distance <= maximum_distance,
                "candidate_text": candidate_text,
                "reference_text": reference_text,
                "normalized_edit_distance": distance,
            }
        )
    sample_pass_rate = sum(bool(row["passed"]) for row in rows) / len(rows)
    passed = sample_pass_rate >= minimum_sample_rate
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
            "sample_pass_rate": sample_pass_rate,
            "max_normalized_edit_distance": max(row["normalized_edit_distance"] for row in rows),
        },
        "gate": {
            "max_normalized_edit_distance": maximum_distance,
            "min_sample_pass_rate": minimum_sample_rate,
        },
        "samples": rows,
    }


def _ocr_samples(
    dataset: Dataset,
    definition: Mapping[str, Any],
    sample_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        payload = json.loads(dataset.path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise QualificationError("OCR dataset is not valid JSON") from error
    samples = payload.get("samples") if isinstance(payload, Mapping) else None
    if not isinstance(samples, list) or len(samples) < sample_limit:
        raise QualificationError(f"OCR dataset requires at least {sample_limit} samples")
    root = dataset.path.parent.resolve()
    selected_rows, selection = _select_dataset_rows(samples, definition, sample_limit, "OCR")
    selected = []
    for index, sample in enumerate(selected_rows):
        if not isinstance(sample, Mapping):
            raise QualificationError(f"OCR sample {index} must be an object")
        media = sample.get("media")
        image_value = None
        if isinstance(media, list):
            for item in media:
                if isinstance(item, Mapping) and item.get("type") == "image":
                    image_value = item.get("path")
                    break
        prompt = sample.get("question")
        answer = sample.get("answer")
        if not isinstance(image_value, str) or not image_value:
            raise QualificationError(f"OCR sample {index} has no image")
        if not isinstance(prompt, str) or not prompt.strip():
            raise QualificationError(f"OCR sample {index} has no question")
        if not isinstance(answer, Mapping):
            raise QualificationError(f"OCR sample {index} has no answer")
        aliases = answer.get("aliases", [])
        primary = answer.get("primary")
        if not isinstance(aliases, list) or not all(
            isinstance(value, str) and value.strip() for value in aliases
        ):
            raise QualificationError(f"OCR sample {index} has invalid answer aliases")
        answers = [str(value).strip() for value in aliases]
        if isinstance(primary, str) and primary.strip() and primary.strip() not in answers:
            answers.insert(0, primary.strip())
        if not answers:
            raise QualificationError(f"OCR sample {index} has no usable answer")
        image = (root / image_value).resolve()
        if root not in image.parents or not image.is_file():
            raise QualificationError(f"OCR sample {index} image is unavailable: {image}")
        selected.append(
            {
                "sample_id": str(sample.get("id") or f"sample-{index}"),
                "image_path": str(image),
                "prompt": prompt.strip(),
                "answers": answers,
            }
        )
    return selected, selection


def _ocr_gold_match(text: str, answers: Sequence[str]) -> bool:
    normalized = _normalized_answer(text).strip(".,!?;:'\"")
    return any(normalized == _normalized_answer(answer).strip(".,!?;:'\"") for answer in answers)


def _ocr_text_parity(
    case: QualificationCase,
    context: RuntimeContext,
    definition: Mapping[str, Any],
    dataset: Dataset,
    output: Path,
) -> dict[str, Any]:
    configured = case.values
    sample_limit = _positive_int(configured.get("samples"), "accuracy.samples")
    selected, selection = _ocr_samples(dataset, definition, sample_limit)
    expected = _family_image_reference(case, context, output, selected)
    request = configured.get("request", {})
    if not isinstance(request, Mapping):
        raise QualificationError("OCR request must be an object")
    candidate_requests = [
        {
            "sample_id": sample["sample_id"],
            "request": {
                **dict(request),
                "image_path": sample["image_path"],
                "prompt": sample["prompt"],
            },
        }
        for sample in selected
    ]
    actual, bundle = _candidate_outputs(case, context, output, "generate", candidate_requests)
    gate = configured.get("gate", {})
    if not isinstance(gate, Mapping):
        raise QualificationError("OCR gate must be an object")
    maximum_distance = float(gate.get("max_normalized_edit_distance", 0.5))
    minimum_sample_rate = float(gate.get("min_sample_pass_rate", 1.0))
    if not 0.0 <= maximum_distance <= 1.0 or not 0.0 <= minimum_sample_rate <= 1.0:
        raise QualificationError("OCR gates must be in [0, 1]")

    rows = []
    for sample, candidate_sample, reference_sample in zip(selected, actual, expected, strict=True):
        candidate_text = _normalized_answer(candidate_sample.get("text"))
        reference_text = _normalized_answer(reference_sample.get("text"))
        distance = _normalized_edit_distance(candidate_text, reference_text)
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "passed": bool(candidate_text) and distance <= maximum_distance,
                "candidate_text": candidate_text,
                "reference_text": reference_text,
                "normalized_edit_distance": distance,
                "candidate_gold_match": _ocr_gold_match(candidate_text, sample["answers"]),
                "reference_gold_match": _ocr_gold_match(reference_text, sample["answers"]),
            }
        )
    sample_pass_rate = sum(bool(row["passed"]) for row in rows) / len(rows)
    candidate_gold_rate = sum(bool(row["candidate_gold_match"]) for row in rows) / len(rows)
    reference_gold_rate = sum(bool(row["reference_gold_match"]) for row in rows) / len(rows)
    return {
        "schema_version": "trtmc.qualification-result/v1",
        "case": case.id,
        "kind": "accuracy",
        "status": "passed" if sample_pass_rate >= minimum_sample_rate else "failed",
        "model": case.model,
        "benchmark": case.benchmark,
        "bundle": str(bundle),
        "dataset": {**dataset.receipt(), "selection": selection},
        "metrics": {
            "samples": len(rows),
            "sample_pass_rate": sample_pass_rate,
            "max_normalized_edit_distance": max(row["normalized_edit_distance"] for row in rows),
            "candidate_normalized_gold_match_rate": candidate_gold_rate,
            "reference_normalized_gold_match_rate": reference_gold_rate,
        },
        "gate": {
            "max_normalized_edit_distance": maximum_distance,
            "min_sample_pass_rate": minimum_sample_rate,
        },
        "samples": rows,
    }


_LOCALIZATION_GROUP = re.compile(r"<(box|point)>(.*?)</\1>", re.DOTALL)


def _localization_values(text: str) -> tuple[str, tuple[tuple[float, ...], ...]]:
    if "<ref>" not in text or "</ref>" not in text:
        raise QualificationError("localization output omitted its reference tag")
    groups = _LOCALIZATION_GROUP.findall(text)
    if not groups:
        raise QualificationError("localization output omitted box or point coordinates")
    kind = groups[0][0]
    values = []
    for current, raw in groups:
        coordinates = tuple(float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", raw))
        expected_size = 4 if current == "box" else 2
        if current != kind or len(coordinates) != expected_size:
            raise QualificationError("localization output mixes coordinate kinds or ranks")
        if not all(math.isfinite(value) and 0.0 <= value <= 1000.0 for value in coordinates):
            raise QualificationError("localization coordinates must be finite values in [0, 1000]")
        if current == "box" and (
            coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]
        ):
            raise QualificationError("localization box coordinates are not ordered xyxy")
        values.append(coordinates)
    return kind, tuple(values)


def _localization_alignment(
    candidate: tuple[tuple[float, ...], ...],
    reference: tuple[tuple[float, ...], ...],
    kind: str,
) -> float:
    if len(candidate) != len(reference):
        return 0.0 if kind == "box" else math.inf
    if kind == "box":
        return max(
            min(_box_iou(left, right) for left, right in zip(candidate, ordering, strict=True))
            for ordering in permutations(reference)
        )
    return min(
        max(math.dist(left, right) for left, right in zip(candidate, ordering, strict=True))
        for ordering in permutations(reference)
    )


def _localization_text_parity(
    case: QualificationCase,
    context: RuntimeContext,
    dataset: Dataset,
    output: Path,
) -> dict[str, Any]:
    configured = case.values
    sample_limit = _positive_int(configured.get("samples"), "accuracy.samples")
    selected = _image_parity_samples(dataset, sample_limit, "localization")
    if any("label_name" not in sample for sample in selected):
        raise QualificationError("localization dataset samples require label_name")
    expected = _family_image_reference(case, context, output, selected)
    request = configured.get("request", {})
    if not isinstance(request, Mapping):
        raise QualificationError("localization request must be an object")
    prompt_template = request.get("prompt_template")
    if not isinstance(prompt_template, str) or "{label}" not in prompt_template:
        raise QualificationError("localization request.prompt_template must contain {label}")
    controls = {name: value for name, value in request.items() if name != "prompt_template"}
    candidate_requests = [
        {
            "sample_id": sample["sample_id"],
            "request": {
                **controls,
                "image_path": sample["image_path"],
                "prompt": prompt_template.format(label=sample["label_name"]),
            },
        }
        for sample in selected
    ]
    actual, bundle = _candidate_outputs(case, context, output, "generate", candidate_requests)
    gate = configured.get("gate", {})
    if not isinstance(gate, Mapping):
        raise QualificationError("localization gate must be an object")
    minimum_box_iou = float(gate.get("min_localization_box_iou", 0.9))
    maximum_point_distance = float(gate.get("max_localization_point_distance", 10.0))
    maximum_text_distance = float(gate.get("max_normalized_edit_distance", 0.5))
    minimum_sample_rate = float(gate.get("min_sample_pass_rate", 1.0))
    if (
        not all(
            0.0 <= value <= 1.0
            for value in (minimum_box_iou, maximum_text_distance, minimum_sample_rate)
        )
        or maximum_point_distance < 0.0
        or not math.isfinite(maximum_point_distance)
    ):
        raise QualificationError("localization gates are outside their valid ranges")

    rows = []
    for sample, candidate_sample, reference_sample in zip(selected, actual, expected, strict=True):
        candidate_text = _normalized_answer(candidate_sample.get("text"))
        reference_text = _normalized_answer(reference_sample.get("text"))
        reference_kind, reference_values = _localization_values(reference_text)
        try:
            candidate_kind, candidate_values = _localization_values(candidate_text)
        except QualificationError:
            candidate_kind = "invalid"
            candidate_values = ()
        same_contract = candidate_kind == reference_kind and bool(candidate_values)
        alignment = (
            _localization_alignment(candidate_values, reference_values, candidate_kind)
            if same_contract
            else 0.0
        )
        text_distance = _normalized_edit_distance(candidate_text, reference_text)
        localization_passed = (
            alignment >= minimum_box_iou
            if candidate_kind == "box"
            else alignment <= maximum_point_distance
            if candidate_kind == "point"
            else False
        )
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "passed": localization_passed and text_distance <= maximum_text_distance,
                "kind": candidate_kind,
                "candidate_count": len(candidate_values),
                "reference_count": len(reference_values),
                "localization_alignment": alignment,
                "normalized_edit_distance": text_distance,
                "candidate_text": candidate_text,
                "reference_text": reference_text,
            }
        )
    sample_pass_rate = sum(bool(row["passed"]) for row in rows) / len(rows)
    return {
        "schema_version": "trtmc.qualification-result/v1",
        "case": case.id,
        "kind": "accuracy",
        "status": "passed" if sample_pass_rate >= minimum_sample_rate else "failed",
        "model": case.model,
        "benchmark": case.benchmark,
        "bundle": str(bundle),
        "dataset": dataset.receipt(),
        "metrics": {
            "samples": len(rows),
            "sample_pass_rate": sample_pass_rate,
            "max_normalized_edit_distance": max(row["normalized_edit_distance"] for row in rows),
        },
        "gate": {
            "min_localization_box_iou": minimum_box_iou,
            "max_localization_point_distance": maximum_point_distance,
            "max_normalized_edit_distance": maximum_text_distance,
            "min_sample_pass_rate": minimum_sample_rate,
        },
        "samples": rows,
    }


def _semantic_segmentation_parity(
    case: QualificationCase,
    context: RuntimeContext,
    dataset: Dataset,
    output: Path,
) -> dict[str, Any]:
    configured = case.values
    sample_limit = _positive_int(configured.get("samples"), "accuracy.samples")
    selected = _image_parity_samples(dataset, sample_limit, "semantic-segmentation")
    expected = _family_image_reference(case, context, output, selected)
    candidate_requests = [
        {
            "sample_id": sample["sample_id"],
            "request": {"image_path": sample["image_path"]},
        }
        for sample in selected
    ]
    actual, bundle = _candidate_outputs(case, context, output, "segment", candidate_requests)
    gate = configured.get("gate", {})
    if not isinstance(gate, Mapping):
        raise QualificationError("semantic-segmentation gate must be an object")
    minimum_pixel_accuracy = float(gate.get("min_pixel_accuracy", 0.99))
    minimum_mean_iou = float(gate.get("min_mean_iou", 0.94))
    minimum_sample_rate = float(gate.get("min_sample_pass_rate", 1.0))
    if not all(
        0.0 <= value <= 1.0
        for value in (minimum_pixel_accuracy, minimum_mean_iou, minimum_sample_rate)
    ):
        raise QualificationError("semantic-segmentation gates must be in [0, 1]")

    rows = []
    for sample, candidate_sample, reference_sample in zip(selected, actual, expected, strict=True):
        left_height, left_width, left = _semantic_mask(candidate_sample, "candidate")
        right_height, right_width, right = _semantic_mask(reference_sample, "reference")
        if (left_height, left_width) != (right_height, right_width):
            pixel_accuracy = mean_iou = 0.0
        else:
            pixel_accuracy = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
            class_ious = []
            for class_id in sorted(set(left) | set(right)):
                intersection = sum(a == class_id and b == class_id for a, b in zip(left, right))
                union = sum(a == class_id or b == class_id for a, b in zip(left, right))
                if union:
                    class_ious.append(intersection / union)
            mean_iou = math.fsum(class_ious) / len(class_ious) if class_ious else 0.0
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "passed": pixel_accuracy >= minimum_pixel_accuracy and mean_iou >= minimum_mean_iou,
                "pixel_accuracy": pixel_accuracy,
                "mean_iou": mean_iou,
                "candidate_shape": [left_height, left_width],
                "reference_shape": [right_height, right_width],
            }
        )
    sample_pass_rate = sum(bool(row["passed"]) for row in rows) / len(rows)
    passed = sample_pass_rate >= minimum_sample_rate
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
            "sample_pass_rate": sample_pass_rate,
            "mean_pixel_accuracy": math.fsum(row["pixel_accuracy"] for row in rows) / len(rows),
            "mean_iou": math.fsum(row["mean_iou"] for row in rows) / len(rows),
            "min_pixel_accuracy": min(row["pixel_accuracy"] for row in rows),
            "min_mean_iou": min(row["mean_iou"] for row in rows),
        },
        "gate": {
            "min_pixel_accuracy": minimum_pixel_accuracy,
            "min_mean_iou": minimum_mean_iou,
            "min_sample_pass_rate": minimum_sample_rate,
        },
        "samples": rows,
    }


def _image_classification_parity(
    case: QualificationCase,
    context: RuntimeContext,
    dataset: Dataset,
    output: Path,
) -> dict[str, Any]:
    configured = case.values
    sample_limit = _positive_int(configured.get("samples"), "accuracy.samples")
    try:
        payload = json.loads(dataset.path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise QualificationError("image-classification dataset is not valid JSON") from error
    requests = payload.get("requests") if isinstance(payload, Mapping) else None
    if not isinstance(requests, list) or len(requests) < sample_limit:
        raise QualificationError(
            f"image-classification dataset requires at least {sample_limit} requests"
        )

    selected: list[dict[str, Any]] = []
    dataset_root = dataset.path.parent.resolve()
    for index, request in enumerate(requests[:sample_limit]):
        if not isinstance(request, Mapping):
            raise QualificationError(f"image-classification request {index} must be an object")
        relative = request.get("image")
        label = request.get("label")
        if not isinstance(relative, str) or not relative:
            raise QualificationError(f"image-classification request {index} has no image")
        if isinstance(label, bool) or not isinstance(label, int) or label < 0:
            raise QualificationError(f"image-classification request {index} has no valid label")
        image = (dataset_root / relative).resolve()
        if dataset_root not in image.parents or not image.is_file():
            raise QualificationError(
                f"image-classification request {index} image is unavailable: {image}"
            )
        selected.append(
            {
                "sample_id": str(request.get("id") or f"sample-{index}"),
                "image_path": str(image),
                "label": label,
            }
        )

    reference = configured.get("reference", {})
    if not isinstance(reference, Mapping):
        raise QualificationError("image-classification reference must be an object")
    reference_request = {
        "model": str(case.candidate["checkpoint"]),
        "revision": case.candidate.get("revision"),
        "precision": str(reference.get("precision", "fp32")),
        "batch_size": int(reference.get("batch_size", 16)),
        "samples": selected,
    }
    request_path = output / "reference-request.json"
    reference_path = output / "reference.json"
    _json(request_path, reference_request)
    runner = (
        context.repository / "qualification_tests/benchmark_qualification/references/timm_image_classification.py"
    )
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
        raise QualificationError(f"TIMM Accuracy reference failed; see {output}")
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    expected = reference_payload.get("samples")
    if not isinstance(expected, list) or len(expected) != len(selected):
        raise QualificationError("TIMM Accuracy reference returned an invalid sample set")

    candidate_requests = [
        {
            "sample_id": sample["sample_id"],
            "request": {"image_path": sample["image_path"]},
        }
        for sample in selected
    ]
    actual, bundle = _candidate_outputs(case, context, output, "classify", candidate_requests)
    rows = []
    for sample, reference_sample, candidate_sample in zip(selected, expected, actual, strict=True):
        reference_class = reference_sample.get("top_class")
        candidate_class = candidate_sample.get("top_class")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (reference_class, candidate_class)
        ):
            raise QualificationError("classification output omitted an integer top_class")
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "label": sample["label"],
                "reference_top_class": reference_class,
                "candidate_top_class": candidate_class,
                "reference_correct": reference_class == sample["label"],
                "candidate_correct": candidate_class == sample["label"],
                "passed": candidate_class == reference_class,
            }
        )

    gate = configured.get("gate", {})
    if not isinstance(gate, Mapping):
        raise QualificationError("image-classification gate must be an object")
    minimum_agreement = float(gate.get("min_top1_agreement", 0.98))
    maximum_drop = float(gate.get("max_top1_accuracy_drop_from_hf", 0.01))
    if not 0.0 <= minimum_agreement <= 1.0:
        raise QualificationError("min_top1_agreement must be in [0, 1]")
    if maximum_drop < 0.0 or not math.isfinite(maximum_drop):
        raise QualificationError("max_top1_accuracy_drop_from_hf must be finite and nonnegative")
    agreement = sum(row["passed"] for row in rows) / len(rows)
    reference_accuracy = sum(row["reference_correct"] for row in rows) / len(rows)
    candidate_accuracy = sum(row["candidate_correct"] for row in rows) / len(rows)
    passed = (
        agreement >= minimum_agreement and candidate_accuracy >= reference_accuracy - maximum_drop
    )
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
            "top1_agreement": agreement,
            "reference_top1_accuracy": reference_accuracy,
            "candidate_top1_accuracy": candidate_accuracy,
            "top1_accuracy_drop_from_hf": reference_accuracy - candidate_accuracy,
        },
        "gate": {
            "min_top1_agreement": minimum_agreement,
            "max_top1_accuracy_drop_from_hf": maximum_drop,
        },
        "samples": rows,
    }


def _speech_transcription_parity(
    case: QualificationCase,
    context: RuntimeContext,
    definition: Mapping[str, Any],
    dataset: Dataset,
    output: Path,
) -> dict[str, Any]:
    configured = case.values
    sample_limit = _positive_int(configured.get("samples"), "accuracy.samples")
    payload = json.loads(dataset.path.read_text(encoding="utf-8"))
    requests = payload.get("requests") if isinstance(payload, Mapping) else None
    if not isinstance(requests, list) or len(requests) < sample_limit:
        raise QualificationError(f"speech dataset requires at least {sample_limit} requests")

    selected_requests, selection = _select_dataset_rows(
        requests, definition, sample_limit, "speech"
    )
    selected = []
    data_root = context.data_root.resolve()
    for index, request in enumerate(selected_requests):
        if not isinstance(request, Mapping):
            raise QualificationError(f"speech request {index} must be an object")
        audio_value = _speech_audio(request)
        audio = (data_root / audio_value).resolve()
        if data_root not in audio.parents or not audio.is_file():
            fallback = (dataset.path.parent / audio_value).resolve()
            if data_root not in fallback.parents or not fallback.is_file():
                raise QualificationError(f"speech request {index} audio is unavailable: {audio}")
            audio = fallback
        gold = request.get("reference")
        if not isinstance(gold, str) or not gold.strip():
            raise QualificationError(f"speech request {index} has no reference transcript")
        selected.append(
            {
                "sample_id": str(request.get("id") or f"sample-{index}"),
                "audio_path": str(audio),
                "gold_text": gold,
            }
        )

    reference = configured.get("reference")
    candidate_request = configured.get("request", {})
    if not isinstance(reference, Mapping) or not isinstance(candidate_request, Mapping):
        raise QualificationError("speech Accuracy reference and request must be objects")
    runner_value = reference.get("command")
    if not isinstance(runner_value, str) or not runner_value:
        raise QualificationError("speech Accuracy reference.command must be set")
    runner = (case.source.parent / runner_value).resolve()
    family_root = case.source.parents[2].resolve()
    if family_root not in runner.parents or not runner.is_file():
        raise QualificationError(f"speech reference runner is not family-owned: {runner}")
    reference_request = {
        "model": str(case.candidate["checkpoint"]),
        "revision": case.candidate.get("revision"),
        "precision": str(reference.get("precision", "fp32")),
        "max_new_tokens": int(candidate_request.get("max_new_tokens", 128)),
        "language": candidate_request.get("language"),
        "samples": selected,
    }
    request_path = output / "reference-request.json"
    reference_path = output / "reference.json"
    _json(request_path, reference_request)
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
        timeout=7200,
        verbose=context.verbose,
    )
    if completed.returncode != 0:
        raise QualificationError(f"speech Accuracy reference failed; see {output}")
    expected = json.loads(reference_path.read_text(encoding="utf-8")).get("samples")
    if not isinstance(expected, list) or len(expected) != len(selected):
        raise QualificationError("speech reference returned an invalid sample set")

    candidate_requests = []
    for sample in expected:
        if not isinstance(sample, Mapping):
            raise QualificationError("speech reference sample must be an object")
        audio_path = sample.get("audio_path")
        if not isinstance(audio_path, str) or not Path(audio_path).is_file():
            raise QualificationError("speech reference did not produce candidate WAV audio")
        candidate_requests.append(
            {
                "sample_id": str(sample.get("sample_id", "")),
                "request": {**dict(candidate_request), "audio_path": audio_path},
            }
        )
    actual, bundle = _candidate_outputs(case, context, output, "transcribe", candidate_requests)

    rows = []
    gold_reference_counts = [0, 0]
    gold_candidate_counts = [0, 0]
    parity_counts = [0, 0]
    for source, reference_sample, candidate_sample in zip(selected, expected, actual, strict=True):
        reference_text = reference_sample.get("text")
        candidate_text = candidate_sample.get("text")
        if not isinstance(reference_text, str) or not reference_text.strip():
            raise QualificationError("speech reference returned an empty transcript")
        if not isinstance(candidate_text, str):
            raise QualificationError("speech candidate omitted its transcript")
        reference_gold = _word_error_counts(source["gold_text"], reference_text)
        candidate_gold = _word_error_counts(source["gold_text"], candidate_text)
        candidate_reference = _word_error_counts(reference_text, candidate_text)
        for totals, counts in (
            (gold_reference_counts, reference_gold),
            (gold_candidate_counts, candidate_gold),
            (parity_counts, candidate_reference),
        ):
            totals[0] += counts[0]
            totals[1] += counts[1]
        rows.append(
            {
                "sample_id": source["sample_id"],
                "reference_text": reference_text,
                "candidate_text": candidate_text,
                "gold_text": source["gold_text"],
                "reference_wer": _rate(reference_gold),
                "candidate_wer": _rate(candidate_gold),
                "wer_to_reference": _rate(candidate_reference),
            }
        )

    reference_wer = _rate(gold_reference_counts)
    candidate_wer = _rate(gold_candidate_counts)
    parity_wer = _rate(parity_counts)
    gate = configured.get("gate", {})
    if not isinstance(gate, Mapping):
        raise QualificationError("speech Accuracy gate must be an object")
    parity_limit = float(gate.get("max_wer_to_reference", 0.1))
    increase_limit = float(gate.get("max_wer_increase_from_reference", 0.02))
    if any(not math.isfinite(value) or value < 0.0 for value in (parity_limit, increase_limit)):
        raise QualificationError("speech WER gates must be finite and nonnegative")
    increase = candidate_wer - reference_wer
    passed = parity_wer <= parity_limit and increase <= increase_limit
    return {
        "schema_version": "trtmc.qualification-result/v1",
        "case": case.id,
        "kind": "accuracy",
        "status": "passed" if passed else "failed",
        "model": case.model,
        "benchmark": case.benchmark,
        "bundle": str(bundle),
        "dataset": {**dataset.receipt(), "selection": selection},
        "metrics": {
            "samples": len(rows),
            "reference_wer": reference_wer,
            "candidate_wer": candidate_wer,
            "wer_increase_from_reference": increase,
            "wer_to_reference": parity_wer,
        },
        "gate": {
            "max_wer_to_reference": parity_limit,
            "max_wer_increase_from_reference": increase_limit,
        },
        "samples": rows,
    }


def _image_feature_knn_parity(
    case: QualificationCase,
    context: RuntimeContext,
    dataset: Dataset,
    output: Path,
) -> dict[str, Any]:
    configured = case.values
    bank_per_class = _positive_int(
        configured.get("bank_samples_per_class"), "accuracy.bank_samples_per_class"
    )
    query_per_class = _positive_int(
        configured.get("query_samples_per_class"), "accuracy.query_samples_per_class"
    )
    bank, queries = _image_feature_samples(dataset.path, bank_per_class, query_per_class)
    samples = [*bank, *queries]

    reference = configured.get("reference")
    if not isinstance(reference, Mapping):
        raise QualificationError("image-feature Accuracy reference must be an object")
    runner_value = reference.get("command")
    if not isinstance(runner_value, str) or not runner_value:
        raise QualificationError("image-feature Accuracy reference.command must be set")
    runner = (case.source.parent / runner_value).resolve()
    family_root = case.source.parents[2].resolve()
    if family_root not in runner.parents or not runner.is_file():
        raise QualificationError(f"image-feature reference runner is not family-owned: {runner}")
    reference_request = {
        "model": str(case.candidate["checkpoint"]),
        "revision": case.candidate.get("revision"),
        "precision": str(reference.get("precision", "fp32")),
        "batch_size": int(reference.get("batch_size", 16)),
        "samples": samples,
    }
    request_path = output / "reference-request.json"
    reference_path = output / "reference.json"
    _json(request_path, reference_request)
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
        timeout=7200,
        verbose=context.verbose,
    )
    if completed.returncode != 0:
        raise QualificationError(f"image-feature Accuracy reference failed; see {output}")
    expected = json.loads(reference_path.read_text(encoding="utf-8")).get("samples")
    if not isinstance(expected, list) or len(expected) != len(samples):
        raise QualificationError("image-feature reference returned an invalid sample set")

    candidate_requests = [
        {
            "sample_id": sample["sample_id"],
            "request": {"image_path": sample["image_path"]},
        }
        for sample in samples
    ]
    actual, bundle = _candidate_outputs(
        case, context, output, "extract_features", candidate_requests
    )
    reference_vectors = [
        _finite_vector(sample.get("pooler_output", []), "HF image feature") for sample in expected
    ]
    candidate_vectors = [
        _finite_vector(sample.get("pooler_output", []), "TRTMC image feature") for sample in actual
    ]
    if any(len(left) != len(right) for left, right in zip(reference_vectors, candidate_vectors)):
        raise QualificationError("image-feature reference and candidate dimensions differ")

    gate = configured.get("gate", {})
    if not isinstance(gate, Mapping):
        raise QualificationError("image-feature Accuracy gate must be an object")
    minimum_cosine = float(gate.get("min_pooler_cosine", 0.999))
    minimum_rate = float(gate.get("min_vector_pass_rate", 1.0))
    minimum_agreement = float(gate.get("min_knn_top1_agreement", 0.98))
    maximum_drop = float(gate.get("max_knn_accuracy_drop_from_hf", 0.02))
    if not -1.0 <= minimum_cosine <= 1.0:
        raise QualificationError("min_pooler_cosine must be in [-1, 1]")
    if not 0.0 <= minimum_rate <= 1.0 or not 0.0 <= minimum_agreement <= 1.0:
        raise QualificationError("image-feature pass rates must be in [0, 1]")
    if maximum_drop < 0.0 or not math.isfinite(maximum_drop):
        raise QualificationError("max_knn_accuracy_drop_from_hf must be finite and nonnegative")

    vector_rows = []
    cosines = []
    for sample, left, right in zip(samples, reference_vectors, candidate_vectors, strict=True):
        cosine = _vector_cosine(left, right)
        cosines.append(cosine)
        vector_rows.append(
            {
                "sample_id": sample["sample_id"],
                "split": sample["split"],
                "label": sample["label"],
                "pooler_cosine": cosine,
                "passed": cosine >= minimum_cosine,
            }
        )

    bank_count = len(bank)
    knn_k = min(_positive_int(configured.get("knn_k", 10), "accuracy.knn_k"), bank_count)
    temperature = float(configured.get("knn_temperature", 0.07))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise QualificationError("accuracy.knn_temperature must be finite and positive")
    labels = [sample["label"] for sample in bank]
    reference_predictions = _knn_predictions(
        reference_vectors[:bank_count], labels, reference_vectors[bank_count:], knn_k, temperature
    )
    candidate_predictions = _knn_predictions(
        candidate_vectors[:bank_count], labels, candidate_vectors[bank_count:], knn_k, temperature
    )
    query_rows = []
    for sample, expected_label, candidate_label in zip(
        queries, reference_predictions, candidate_predictions, strict=True
    ):
        query_rows.append(
            {
                "sample_id": sample["sample_id"],
                "label": sample["label"],
                "reference_top1": expected_label,
                "candidate_top1": candidate_label,
                "reference_correct": expected_label == sample["label"],
                "candidate_correct": candidate_label == sample["label"],
                "passed": candidate_label == expected_label,
            }
        )
    vector_pass_rate = sum(row["passed"] for row in vector_rows) / len(vector_rows)
    agreement = sum(row["passed"] for row in query_rows) / len(query_rows)
    reference_accuracy = sum(row["reference_correct"] for row in query_rows) / len(query_rows)
    candidate_accuracy = sum(row["candidate_correct"] for row in query_rows) / len(query_rows)
    accuracy_drop = reference_accuracy - candidate_accuracy
    passed = (
        vector_pass_rate >= minimum_rate
        and agreement >= minimum_agreement
        and accuracy_drop <= maximum_drop
    )
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
            "bank_samples": bank_count,
            "query_samples": len(queries),
            "vector_pass_rate": vector_pass_rate,
            "mean_pooler_cosine": math.fsum(cosines) / len(cosines),
            "min_pooler_cosine": min(cosines),
            "knn_top1_agreement": agreement,
            "reference_knn_top1_accuracy": reference_accuracy,
            "candidate_knn_top1_accuracy": candidate_accuracy,
            "knn_accuracy_drop_from_hf": accuracy_drop,
        },
        "gate": {
            "min_pooler_cosine": minimum_cosine,
            "min_vector_pass_rate": minimum_rate,
            "min_knn_top1_agreement": minimum_agreement,
            "max_knn_accuracy_drop_from_hf": maximum_drop,
        },
        "samples": vector_rows,
        "queries": query_rows,
    }


def _image_feature_samples(
    dataset_path: Path, bank_per_class: int, query_per_class: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    requests = payload.get("requests") if isinstance(payload, Mapping) else None
    if not isinstance(requests, list) or not requests:
        raise QualificationError("image-feature dataset must contain requests")
    root = dataset_path.parent.resolve()
    bank_path: Path | None = None
    query_paths: list[Path] = []
    for request in requests:
        inputs = request.get("inputs") if isinstance(request, Mapping) else None
        if not isinstance(inputs, Mapping):
            raise QualificationError("image-feature dataset request must contain inputs")
        configured_bank = _dataset_child(root, inputs.get("bank_manifest"), "bank manifest")
        if bank_path is not None and bank_path != configured_bank:
            raise QualificationError("image-feature dataset requests use different banks")
        bank_path = configured_bank
        query_paths.append(_dataset_child(root, inputs.get("query_manifest"), "query manifest"))
    assert bank_path is not None
    bank = _labeled_image_manifest(bank_path, root, "bank")
    queries = [
        sample
        for query_path in query_paths
        for sample in _labeled_image_manifest(query_path, root, "query")
    ]
    return (
        _take_per_class(bank, bank_per_class, "bank"),
        _take_per_class(queries, query_per_class, "query"),
    )


def _dataset_child(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise QualificationError(f"image-feature dataset omitted its {label}")
    path = (root / value).resolve()
    if root not in path.parents or not path.is_file():
        raise QualificationError(f"image-feature {label} is unavailable: {path}")
    return path


def _labeled_image_manifest(path: Path, root: Path, split: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples") if isinstance(payload, Mapping) else None
    if not isinstance(samples, list) or not samples:
        raise QualificationError(f"image-feature {split} manifest has no samples")
    result = []
    for index, sample in enumerate(samples):
        relative = sample.get("image") if isinstance(sample, Mapping) else None
        label = sample.get("label") if isinstance(sample, Mapping) else None
        if not isinstance(relative, str) or isinstance(label, bool) or not isinstance(label, int):
            raise QualificationError(f"image-feature {split} sample {index} is invalid")
        image = (path.parent / relative).resolve()
        if root not in image.parents or not image.is_file():
            raise QualificationError(f"image-feature {split} image is unavailable: {image}")
        result.append(
            {
                "sample_id": f"{split}-{int(sample.get('source_index', index)):06d}",
                "image_path": str(image),
                "label": label,
                "split": split,
            }
        )
    return result


def _take_per_class(
    samples: Sequence[Mapping[str, Any]], count: int, split: str
) -> list[dict[str, Any]]:
    classes = sorted({int(sample["label"]) for sample in samples})
    selected = []
    for label in classes:
        matches = [dict(sample) for sample in samples if sample["label"] == label][:count]
        if len(matches) != count:
            raise QualificationError(
                f"image-feature {split} class {label} has {len(matches)} samples; {count} required"
            )
        selected.extend(matches)
    return selected


def _knn_predictions(
    bank: Sequence[Sequence[float]],
    labels: Sequence[int],
    queries: Sequence[Sequence[float]],
    k: int,
    temperature: float,
) -> list[int]:
    predictions = []
    for query in queries:
        neighbors = sorted(
            (
                (_vector_cosine(query, vector), label)
                for vector, label in zip(bank, labels, strict=True)
            ),
            reverse=True,
        )[:k]
        scores: dict[int, float] = {}
        for similarity, label in neighbors:
            scores[label] = scores.get(label, 0.0) + math.exp(similarity / temperature)
        predictions.append(max(scores, key=lambda label: (scores[label], -label)))
    return predictions


def _speech_audio(request: Mapping[str, Any]) -> str:
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise QualificationError("speech request has no messages")
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "audio":
                audio = item.get("audio")
                if isinstance(audio, str) and audio:
                    return audio
    raise QualificationError("speech request has no audio content")


def _word_error_counts(reference: str, hypothesis: str) -> list[int]:
    def words(value: str) -> list[str]:
        return [
            normalized
            for word in value.split()
            if (normalized := re.sub(r"^[^\w]+|[^\w]+$", "", word).casefold())
        ]

    left = words(reference)
    right = words(hypothesis)
    previous = list(range(len(right) + 1))
    for row, left_word in enumerate(left, start=1):
        current = [row]
        for column, right_word in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_word != right_word),
                )
            )
        previous = current
    return [previous[-1], len(left)]


def _rate(counts: Sequence[int]) -> float:
    return float(counts[0]) / max(int(counts[1]), 1)


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


def _select_dataset_rows(
    rows: Sequence[Any],
    definition: Mapping[str, Any],
    count: int,
    label: str,
) -> tuple[list[Any], dict[str, Any]]:
    selection = definition.get("selection", {})
    if not isinstance(selection, Mapping):
        raise QualificationError(f"{label} dataset selection must be an object")
    method = str(selection.get("method", "first"))
    if method == "first":
        if len(rows) < count:
            raise QualificationError(f"{label} dataset requires at least {count} samples")
        indices = list(range(count))
    elif method == "fixed-indices":
        configured = selection.get("indices")
        if not isinstance(configured, list) or len(configured) < count:
            raise QualificationError(
                f"{label} fixed dataset selection requires at least {count} indices"
            )
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in configured
        ):
            raise QualificationError(f"{label} fixed dataset indices must be nonnegative integers")
        if len(set(configured)) != len(configured):
            raise QualificationError(f"{label} fixed dataset indices must be unique")
        indices = configured[:count]
        if any(index >= len(rows) for index in indices):
            raise QualificationError(f"{label} fixed dataset index is out of range")
    else:
        raise QualificationError(f"unsupported {label} dataset selection method {method!r}")
    return [rows[index] for index in indices], {"method": method, "indices": indices}


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise QualificationError(f"{name} must be a positive integer")
    return value


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
