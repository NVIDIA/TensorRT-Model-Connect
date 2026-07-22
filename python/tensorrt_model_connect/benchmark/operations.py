# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry of task-strategy-aware benchmark operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .types import BenchmarkError, MeasurementSpec


_MODEL_TESTCASE = "model testcase"
_OPERATION_DEFAULT = "operation default"


@dataclass(frozen=True)
class RequestResolution:
    """A task-aware request plus field-level provenance."""

    request: Mapping[str, Any]
    sources: Mapping[str, str]

    def __post_init__(self) -> None:
        if set(self.request) != set(self.sources):
            raise RuntimeError("benchmark request fields and sources must match")


RequestFactory = Callable[[Mapping[str, Any], Path], RequestResolution]


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


MetricFactory = Callable[
    [Mapping[str, Any]],
    tuple[tuple[RateMetric, ...], PerItemLatencyMetric | None],
]


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
    metric_factory: MetricFactory | None = None

    def request_from_testcase(
        self, testcase: Mapping[str, Any], model_root: Path
    ) -> dict[str, Any]:
        return dict(self.resolve_request_from_testcase(testcase, model_root).request)

    def resolve_request_from_testcase(
        self, testcase: Mapping[str, Any], model_root: Path
    ) -> RequestResolution:
        return self.request_factory(testcase, model_root)

    def metrics_for_request(
        self, request: Mapping[str, Any]
    ) -> tuple[tuple[RateMetric, ...], PerItemLatencyMetric | None]:
        if self.metric_factory is not None:
            return self.metric_factory(request)
        return self.rate_metrics, self.per_item_latency


def _prompt(testcase: Mapping[str, Any], operation: str) -> str:
    prompt = testcase.get("prompt", testcase.get("test_prompt"))
    if not isinstance(prompt, str) or not prompt:
        raise BenchmarkError(f"{operation} testcase requires prompt/test_prompt")
    return prompt


def _model_asset(declared: str, model_root: Path, operation: str) -> tuple[Path, Path]:
    path = Path(declared).expanduser()
    if path.is_absolute():
        resolved = path
        portable = path
    else:
        portable = path
        resolved = model_root / portable
        source_prefix = Path("tests/e2e/models") / model_root.name
        if not resolved.is_file() and path.is_relative_to(source_prefix):
            portable = path.relative_to(source_prefix)
            resolved = model_root / portable
    if not resolved.is_file():
        raise BenchmarkError(f"cannot read {operation} input {resolved}")
    return portable, resolved


def _testcase_value(testcase: Mapping[str, Any], name: str, default: Any) -> tuple[Any, str]:
    if name in testcase:
        return testcase[name], _MODEL_TESTCASE
    return default, _OPERATION_DEFAULT


def _input_or_testcase_value(
    inputs: Mapping[str, Any], testcase: Mapping[str, Any], name: str, default: Any
) -> tuple[Any, str]:
    if name in inputs:
        return inputs[name], _MODEL_TESTCASE
    return _testcase_value(testcase, name, default)


def _resolution(request: Mapping[str, Any], sources: Mapping[str, str]) -> RequestResolution:
    return RequestResolution(request=dict(request), sources=dict(sources))


def _generate_request(testcase: Mapping[str, Any], _model_root: Path) -> RequestResolution:
    inputs = testcase.get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise BenchmarkError("generate testcase inputs must be an object")
    metadata = testcase.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise BenchmarkError("generate testcase metadata must be an object")
    contract = metadata.get("contract_config", {})
    if not isinstance(contract, Mapping):
        raise BenchmarkError("generate testcase contract_config must be an object")

    max_new_tokens, max_new_tokens_source = _testcase_value(testcase, "max_new_tokens", 20)
    temperature, temperature_source = _input_or_testcase_value(inputs, testcase, "temperature", 0.0)
    top_k, top_k_source = _input_or_testcase_value(inputs, testcase, "top_k", 1)
    top_p, top_p_source = _input_or_testcase_value(inputs, testcase, "top_p", 1.0)
    min_p, min_p_source = _input_or_testcase_value(inputs, testcase, "min_p", 0.0)
    seed, seed_source = _input_or_testcase_value(inputs, testcase, "seed", -1)
    use_chat_template, use_chat_template_source = _testcase_value(
        contract, "use_chat_template", False
    )
    enable_thinking, enable_thinking_source = _testcase_value(contract, "enable_thinking", True)

    request: dict[str, Any] = {
        "batch_size": 1,
        "prompt": _prompt(testcase, "generate"),
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "min_p": float(min_p),
        "seed": int(seed),
        "use_chat_template": bool(use_chat_template),
        "enable_thinking": bool(enable_thinking),
    }
    sources = {
        "batch_size": _OPERATION_DEFAULT,
        "prompt": _MODEL_TESTCASE,
        "max_new_tokens": max_new_tokens_source,
        "temperature": temperature_source,
        "top_k": top_k_source,
        "top_p": top_p_source,
        "min_p": min_p_source,
        "seed": seed_source,
        "use_chat_template": use_chat_template_source,
        "enable_thinking": enable_thinking_source,
    }
    generation_mode, generation_mode_source = _input_or_testcase_value(
        inputs, testcase, "generation_mode", None
    )
    if generation_mode is not None:
        if not isinstance(generation_mode, str) or not generation_mode:
            raise BenchmarkError("generate generation_mode must be a non-empty string")
        request["generation_mode"] = generation_mode
        sources["generation_mode"] = generation_mode_source
    block_length, block_length_source = _input_or_testcase_value(
        inputs, testcase, "block_length", None
    )
    if block_length is not None:
        request["block_length"] = int(block_length)
        sources["block_length"] = block_length_source
    threshold, threshold_source = _input_or_testcase_value(inputs, testcase, "threshold", None)
    if threshold is not None:
        request["threshold"] = float(threshold)
        sources["threshold"] = threshold_source
    return _resolution(request, sources)


def _generate_image_request(testcase: Mapping[str, Any], model_root: Path) -> RequestResolution:
    inputs = testcase.get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise BenchmarkError("generate_image testcase inputs must be an object")
    declared_prompts = inputs.get("batch_prompts")
    prompts: list[str] | None = None
    seeds: list[int] | None = None
    if declared_prompts is not None:
        if not isinstance(declared_prompts, list) or not declared_prompts:
            raise BenchmarkError("generate_image batch_prompts must be a non-empty list")
        if not all(isinstance(prompt, str) and prompt for prompt in declared_prompts):
            raise BenchmarkError("generate_image batch_prompts must contain non-empty strings")
        prompts = list(declared_prompts)
        declared_seeds = inputs.get("batch_seeds")
        if not isinstance(declared_seeds, list) or len(declared_seeds) != len(prompts):
            raise BenchmarkError("generate_image batch_seeds must match batch_prompts")
        seeds = [int(seed) for seed in declared_seeds]
        expected_batch_size = int(inputs.get("expected_batch_size", len(prompts)))
        if expected_batch_size != len(prompts):
            raise BenchmarkError("generate_image expected_batch_size must match batch_prompts")
        batch_size = expected_batch_size
        batch_size_source = _MODEL_TESTCASE
    else:
        batch_size_value, batch_size_source = _testcase_value(testcase, "batch_size", 1)
        batch_size = int(batch_size_value)
    video_height = testcase.get("video_height")
    video_width = testcase.get("video_width")
    video_num_frames = int(testcase.get("video_num_frames", 1))
    is_video = video_height is not None or video_width is not None or video_num_frames > 1
    if "image_height" in testcase:
        height, height_source = testcase["image_height"], _MODEL_TESTCASE
    elif video_height is not None:
        height, height_source = video_height, _MODEL_TESTCASE
    else:
        height, height_source = 0, _OPERATION_DEFAULT
    if "image_width" in testcase:
        width, width_source = testcase["image_width"], _MODEL_TESTCASE
    elif video_width is not None:
        width, width_source = video_width, _MODEL_TESTCASE
    else:
        width, width_source = 0, _OPERATION_DEFAULT
    num_inference_steps, num_inference_steps_source = _testcase_value(
        testcase, "num_inference_steps", -1
    )
    seed, seed_source = _testcase_value(testcase, "seed", 42)
    guidance_scale, guidance_scale_source = _testcase_value(testcase, "guidance_scale", -1.0)
    cfg_scale, cfg_scale_source = _testcase_value(testcase, "cfg_scale", -1.0)
    negative_prompt, negative_prompt_source = _testcase_value(testcase, "negative_prompt", "")
    request: dict[str, Any] = {
        "batch_size": batch_size,
        "media_type": "video" if is_video else "image",
        "prompt": _prompt(testcase, "generate_image"),
        "height": int(height),
        "width": int(width),
        "num_inference_steps": int(num_inference_steps),
        "seed": int(seed),
        "guidance_scale": float(guidance_scale),
        "cfg_scale": float(cfg_scale),
        "negative_prompt": str(negative_prompt),
    }
    sources = {
        "batch_size": batch_size_source,
        "media_type": _MODEL_TESTCASE if is_video else _OPERATION_DEFAULT,
        "prompt": _MODEL_TESTCASE,
        "height": height_source,
        "width": width_source,
        "num_inference_steps": num_inference_steps_source,
        "seed": seed_source,
        "guidance_scale": guidance_scale_source,
        "cfg_scale": cfg_scale_source,
        "negative_prompt": negative_prompt_source,
    }
    if is_video:
        if video_height is None or video_width is None or video_num_frames <= 1:
            raise BenchmarkError(
                "generate_image video testcase requires video_height, video_width, "
                "and video_num_frames > 1"
            )
        request.update(
            {
                "video_height": int(video_height),
                "video_width": int(video_width),
                "video_num_frames": video_num_frames,
            }
        )
        sources.update(
            {
                "video_height": _MODEL_TESTCASE,
                "video_width": _MODEL_TESTCASE,
                "video_num_frames": _MODEL_TESTCASE,
            }
        )
    if prompts is not None and seeds is not None:
        request["prompts"] = prompts
        request["seeds"] = seeds
        sources["prompts"] = _MODEL_TESTCASE
        sources["seeds"] = _MODEL_TESTCASE
    declared_image = testcase.get("test_image")
    if declared_image is not None:
        if not isinstance(declared_image, str) or not declared_image:
            raise BenchmarkError("generate_image test_image must be a non-empty path")
        image_path, _ = _model_asset(declared_image, model_root, "generate_image")
        if batch_size != 1:
            raise BenchmarkError("image-conditioned generate_image supports batch_size=1 only")
        request["image_path"] = str(image_path)
        sources["image_path"] = _MODEL_TESTCASE
    return _resolution(request, sources)


def _text_input_request(operation: str, testcase: Mapping[str, Any]) -> RequestResolution:
    return _resolution(
        {"batch_size": 1, "prompt": _prompt(testcase, operation)},
        {"batch_size": _OPERATION_DEFAULT, "prompt": _MODEL_TESTCASE},
    )


def _encode_request(testcase: Mapping[str, Any], _model_root: Path) -> RequestResolution:
    return _text_input_request("encode", testcase)


def _embed_request(testcase: Mapping[str, Any], _model_root: Path) -> RequestResolution:
    return _text_input_request("embed", testcase)


def _solve_request(testcase: Mapping[str, Any], _model_root: Path) -> RequestResolution:
    inputs = testcase.get("inputs")
    if not isinstance(inputs, Mapping):
        raise BenchmarkError("solve testcase requires an inputs object")
    request: dict[str, Any] = {"batch_size": 1}
    sources = {"batch_size": _OPERATION_DEFAULT}
    for key in ("field_input", "branch_input", "trunk_input"):
        if key not in inputs:
            continue
        values = inputs[key]
        if not isinstance(values, list):
            raise BenchmarkError(f"solve input {key} must be a list")
        request[key] = [float(value) for value in values]
        sources[key] = _MODEL_TESTCASE
    if "field_input" not in request and "branch_input" not in request:
        raise BenchmarkError("solve testcase requires field_input or branch_input")
    return _resolution(request, sources)


def _transcribe_request(testcase: Mapping[str, Any], model_root: Path) -> RequestResolution:
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
    max_new_tokens, max_new_tokens_source = _testcase_value(testcase, "max_new_tokens", 224)
    language, language_source = _testcase_value(testcase, "language", "")
    streaming_source = _MODEL_TESTCASE if "streaming" in testcase else _OPERATION_DEFAULT
    return _resolution(
        {
            "batch_size": 1,
            "audio_path": str(audio_path),
            "max_new_tokens": int(max_new_tokens),
            "language": str(language),
            "streaming": dict(streaming),
        },
        {
            "batch_size": _OPERATION_DEFAULT,
            "audio_path": _MODEL_TESTCASE,
            "max_new_tokens": max_new_tokens_source,
            "language": language_source,
            "streaming": streaming_source,
        },
    )


def _generated_media_metrics(
    request: Mapping[str, Any],
) -> tuple[tuple[RateMetric, ...], PerItemLatencyMetric]:
    media_type = str(request.get("media_type", "image") or "image")
    if media_type == "image":
        return (
            (RateMetric("generated_images", "images_per_s"),),
            PerItemLatencyMetric("generated_images", "seconds_per_image_p50"),
        )
    if media_type == "video":
        return (
            (
                RateMetric("generated_images", "videos_per_s"),
                RateMetric("generated_frames", "frames_per_s"),
            ),
            PerItemLatencyMetric("generated_images", "seconds_per_video_p50"),
        )
    raise BenchmarkError(f"unsupported generated media type {media_type!r}")


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
        metric_factory=_generated_media_metrics,
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
