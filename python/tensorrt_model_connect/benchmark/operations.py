# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry of task-strategy-aware benchmark operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .types import BenchmarkError, MeasurementSpec


RequestFactory = Callable[[Mapping[str, Any], Path], dict[str, Any]]


@dataclass(frozen=True)
class RateMetric:
    """Sum an observation field and divide it by total measured seconds."""

    observation_field: str
    result_name: str
    inverse_result_name: str | None = None


@dataclass(frozen=True)
class PerItemLatencyMetric:
    """Normalize each request latency by the number of produced items."""

    count_field: str
    result_name: str


@dataclass(frozen=True)
class OperationSpec:
    """One public pipeline operation and its benchmark semantics."""

    name: str
    task_strategies: tuple[str, ...]
    default_measurement: MeasurementSpec
    request_factory: RequestFactory
    supports_batch: bool = False
    rate_metrics: tuple[RateMetric, ...] = ()
    stage_timings: tuple[str, ...] = ()
    per_item_latency: PerItemLatencyMetric | None = None

    def request_from_testcase(
        self, testcase: Mapping[str, Any], model_root: Path
    ) -> dict[str, Any]:
        return self.request_factory(testcase, model_root)


def _prompt(testcase: Mapping[str, Any], operation: str) -> str:
    prompt = testcase.get("prompt", testcase.get("test_prompt"))
    if not isinstance(prompt, str) or not prompt:
        raise BenchmarkError(f"{operation} testcase requires prompt/test_prompt")
    return prompt


def _generate_request(testcase: Mapping[str, Any], _model_root: Path) -> dict[str, Any]:
    return {
        "batch_size": 1,
        "prompt": _prompt(testcase, "generate"),
        "max_new_tokens": int(testcase.get("max_new_tokens", 20)),
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": int(testcase.get("seed", 42)),
    }


def _generate_image_request(testcase: Mapping[str, Any], _model_root: Path) -> dict[str, Any]:
    return {
        "batch_size": 1,
        "prompt": _prompt(testcase, "generate_image"),
        "height": int(testcase.get("image_height", 0)),
        "width": int(testcase.get("image_width", 0)),
        "num_inference_steps": int(testcase.get("num_inference_steps", -1)),
        "seed": int(testcase.get("seed", 42)),
        "guidance_scale": float(testcase.get("guidance_scale", -1.0)),
        "cfg_scale": float(testcase.get("cfg_scale", -1.0)),
        "negative_prompt": str(testcase.get("negative_prompt", "")),
    }


def _text_input_request(operation: str, testcase: Mapping[str, Any]) -> dict[str, Any]:
    return {"batch_size": 1, "prompt": _prompt(testcase, operation)}


def _encode_request(testcase: Mapping[str, Any], _model_root: Path) -> dict[str, Any]:
    return _text_input_request("encode", testcase)


def _embed_request(testcase: Mapping[str, Any], _model_root: Path) -> dict[str, Any]:
    return _text_input_request("embed", testcase)


def _solve_request(testcase: Mapping[str, Any], _model_root: Path) -> dict[str, Any]:
    inputs = testcase.get("inputs")
    if not isinstance(inputs, Mapping):
        raise BenchmarkError("solve testcase requires an inputs object")
    request: dict[str, Any] = {"batch_size": 1}
    for key in ("field_input", "branch_input", "trunk_input"):
        if key not in inputs:
            continue
        values = inputs[key]
        if not isinstance(values, list):
            raise BenchmarkError(f"solve input {key} must be a list")
        request[key] = [float(value) for value in values]
    if "field_input" not in request and "branch_input" not in request:
        raise BenchmarkError("solve testcase requires field_input or branch_input")
    return request


def _transcribe_request(testcase: Mapping[str, Any], model_root: Path) -> dict[str, Any]:
    declared_audio = testcase.get("test_input_audio")
    if not isinstance(declared_audio, str) or not declared_audio:
        raise BenchmarkError("transcribe testcase requires test_input_audio")
    audio_path = Path(declared_audio).expanduser()
    resolved_audio = audio_path if audio_path.is_absolute() else model_root / audio_path
    if not resolved_audio.is_file():
        raise BenchmarkError(f"cannot read transcribe audio input {resolved_audio}")
    streaming = testcase.get("streaming", {})
    if not isinstance(streaming, Mapping):
        raise BenchmarkError("transcribe testcase streaming configuration must be an object")
    return {
        "batch_size": 1,
        "audio_path": str(audio_path),
        "max_new_tokens": int(testcase.get("max_new_tokens", 224)),
        "language": str(testcase.get("language", "")),
        "streaming": dict(streaming),
    }


_OPERATIONS = (
    OperationSpec(
        name="generate",
        task_strategies=("text_generation_causal",),
        default_measurement=MeasurementSpec(warmup=5, iterations=50),
        request_factory=_generate_request,
        rate_metrics=(RateMetric("output_tokens", "output_tokens_per_s"),),
        stage_timings=("prefill_ms", "decode_ms"),
    ),
    OperationSpec(
        name="generate_image",
        task_strategies=("diffusion_media_generation",),
        default_measurement=MeasurementSpec(warmup=1, iterations=5),
        request_factory=_generate_image_request,
        supports_batch=True,
        rate_metrics=(RateMetric("generated_images", "images_per_s"),),
        per_item_latency=PerItemLatencyMetric("generated_images", "seconds_per_image_p50"),
    ),
    OperationSpec(
        name="encode",
        task_strategies=("encoder_only_nlp",),
        default_measurement=MeasurementSpec(warmup=50, iterations=500),
        request_factory=_encode_request,
        rate_metrics=(
            RateMetric("embedding_vectors", "embedding_vectors_per_s"),
            RateMetric("embedding_elements", "embedding_elements_per_s"),
        ),
    ),
    OperationSpec(
        name="embed",
        task_strategies=("embedding",),
        default_measurement=MeasurementSpec(warmup=50, iterations=500),
        request_factory=_embed_request,
        rate_metrics=(
            RateMetric("embedding_vectors", "embedding_vectors_per_s"),
            RateMetric("embedding_elements", "embedding_elements_per_s"),
        ),
    ),
    OperationSpec(
        name="solve",
        task_strategies=("neural_operator",),
        default_measurement=MeasurementSpec(warmup=50, iterations=500),
        request_factory=_solve_request,
        rate_metrics=(
            RateMetric("windows", "windows_per_s"),
            RateMetric("forecast_elements", "forecast_elements_per_s"),
        ),
    ),
    OperationSpec(
        name="transcribe",
        task_strategies=("speech_to_text",),
        default_measurement=MeasurementSpec(warmup=1, iterations=10),
        request_factory=_transcribe_request,
        rate_metrics=(
            RateMetric(
                "input_audio_seconds",
                "audio_seconds_per_s",
                inverse_result_name="realtime_factor",
            ),
            RateMetric("output_tokens", "output_tokens_per_s"),
        ),
        stage_timings=("first_partial_ms",),
    ),
)


def _index_operations() -> tuple[dict[str, OperationSpec], dict[str, OperationSpec]]:
    by_name: dict[str, OperationSpec] = {}
    by_task_strategy: dict[str, OperationSpec] = {}
    for operation in _OPERATIONS:
        if operation.name in by_name:
            raise RuntimeError(f"duplicate benchmark operation {operation.name!r}")
        by_name[operation.name] = operation
        for task_strategy in operation.task_strategies:
            if task_strategy in by_task_strategy:
                raise RuntimeError(f"duplicate benchmark task strategy {task_strategy!r}")
            by_task_strategy[task_strategy] = operation
    return by_name, by_task_strategy


_BY_NAME, _BY_TASK_STRATEGY = _index_operations()


def operation_for_name(name: str) -> OperationSpec:
    try:
        return _BY_NAME[name]
    except KeyError as exc:
        raise BenchmarkError(f"unsupported metric operation: {name}") from exc


def operation_for_task_strategy(task_strategy: str) -> OperationSpec:
    try:
        return _BY_TASK_STRATEGY[task_strategy]
    except KeyError as exc:
        raise BenchmarkError(
            f"task strategy {task_strategy!r} has no benchmark operation adapter"
        ) from exc


def registered_operations() -> tuple[OperationSpec, ...]:
    return _OPERATIONS
