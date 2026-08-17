# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate E2E task strategies into stable benchmark operation requests.

A model family is not registered here.  Any current or future family using an
existing ``task_strategy`` automatically receives the matching adapter.  A new
paradigm adds one adapter and can reuse an existing public operation whenever
its output contract matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

from .operations import operation_for_name
from .types import BenchmarkError, MeasurementSpec


_MODEL_TESTCASE = "model testcase"
_TASK_DEFAULT = "operation default"


@dataclass(frozen=True)
class CaseResolution:
    """Task-aware request/runtime values with field-level provenance."""

    request: Mapping[str, Any]
    request_sources: Mapping[str, str]
    runtime: Mapping[str, Any] = field(default_factory=dict)
    runtime_sources: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.request) != set(self.request_sources):
            raise RuntimeError("benchmark request fields and sources must match")
        if set(self.runtime) != set(self.runtime_sources):
            raise RuntimeError("benchmark runtime fields and sources must match")


CaseFactory = Callable[[Mapping[str, Any], Path], CaseResolution]


@dataclass(frozen=True)
class TaskAdapter:
    """One E2E task strategy mapped to one public pipeline operation."""

    task_strategy: str
    operation: str
    default_measurement: MeasurementSpec
    case_factory: CaseFactory

    def resolve_case(self, testcase: Mapping[str, Any], model_root: Path) -> CaseResolution:
        return self.case_factory(testcase, model_root)


def _prompt(testcase: Mapping[str, Any], operation: str, *, allow_empty: bool = False) -> str:
    repeat = testcase.get("prompt_repeat")
    if repeat is not None:
        if not isinstance(repeat, Mapping):
            raise BenchmarkError(f"{operation} testcase prompt_repeat must be an object")
        unknown = sorted(set(repeat) - {"text", "separator", "count", "suffix"})
        if unknown:
            raise BenchmarkError(
                f"{operation} testcase prompt_repeat has unsupported fields: {unknown}"
            )
        text = repeat.get("text")
        count = repeat.get("count")
        separator = repeat.get("separator", "")
        suffix = repeat.get("suffix", "")
        if not isinstance(text, str) or not text:
            raise BenchmarkError(
                f"{operation} testcase prompt_repeat.text must be a non-empty string"
            )
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise BenchmarkError(
                f"{operation} testcase prompt_repeat.count must be a positive integer"
            )
        if not isinstance(separator, str):
            raise BenchmarkError(
                f"{operation} testcase prompt_repeat.separator must be a string"
            )
        if not isinstance(suffix, str):
            raise BenchmarkError(
                f"{operation} testcase prompt_repeat.suffix must be a string"
            )
        return separator.join([text] * count) + suffix

    prompt = testcase.get("prompt", testcase.get("test_prompt"))
    if not isinstance(prompt, str) or (not prompt and not allow_empty):
        raise BenchmarkError(
            f"{operation} testcase requires prompt/test_prompt/prompt_repeat"
        )
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


def _asset_field(
    testcase: Mapping[str, Any], model_root: Path, field_name: str, operation: str
) -> str:
    declared = testcase.get(field_name)
    if not isinstance(declared, str) or not declared:
        raise BenchmarkError(f"{operation} testcase requires {field_name}")
    portable, _ = _model_asset(declared, model_root, operation)
    return str(portable)


def _testcase_value(testcase: Mapping[str, Any], name: str, default: Any) -> tuple[Any, str]:
    if name in testcase:
        return testcase[name], _MODEL_TESTCASE
    return default, _TASK_DEFAULT


def _input_or_testcase_value(
    inputs: Mapping[str, Any], testcase: Mapping[str, Any], name: str, default: Any
) -> tuple[Any, str]:
    if name in inputs:
        return inputs[name], _MODEL_TESTCASE
    return _testcase_value(testcase, name, default)


def _required_python_runtime(testcase: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    metadata = testcase.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise BenchmarkError("testcase metadata must be an object")
    if not metadata.get("runtime_cli_requires_hf_python"):
        return {}, {}
    return {"hf_python": sys.executable}, {"hf_python": _MODEL_TESTCASE}


def _resolution(
    request: Mapping[str, Any],
    sources: Mapping[str, str],
    *,
    testcase: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    runtime_sources: Mapping[str, str] | None = None,
) -> CaseResolution:
    selected_runtime = dict(runtime or {})
    selected_runtime_sources = dict(runtime_sources or {})
    if testcase is not None:
        required_runtime, required_sources = _required_python_runtime(testcase)
        selected_runtime.update(required_runtime)
        selected_runtime_sources.update(required_sources)
    return CaseResolution(
        request=dict(request),
        request_sources=dict(sources),
        runtime=selected_runtime,
        runtime_sources=selected_runtime_sources,
    )


def _generation_values(
    testcase: Mapping[str, Any], *, operation: str = "generate"
) -> tuple[dict[str, Any], dict[str, str]]:
    inputs = testcase.get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise BenchmarkError(f"{operation} testcase inputs must be an object")
    metadata = testcase.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise BenchmarkError(f"{operation} testcase metadata must be an object")
    contract = metadata.get("contract_config", {})
    if not isinstance(contract, Mapping):
        raise BenchmarkError(f"{operation} testcase contract_config must be an object")

    values: dict[str, Any] = {"batch_size": 1}
    sources: dict[str, str] = {"batch_size": _TASK_DEFAULT}
    fields = (
        ("max_new_tokens", 20, int),
        ("temperature", 0.0, float),
        ("top_k", 1, int),
        ("top_p", 1.0, float),
        ("min_p", 0.0, float),
        ("seed", -1, int),
    )
    for name, default, convert in fields:
        value, source = _input_or_testcase_value(inputs, testcase, name, default)
        values[name] = convert(value)
        sources[name] = source
    for name, default in (("use_chat_template", False), ("enable_thinking", True)):
        value, source = _testcase_value(contract, name, default)
        values[name] = bool(value)
        sources[name] = source
    for name, convert in (
        ("num_samples", int),
        ("generation_mode", str),
        ("block_length", int),
        ("threshold", float),
        ("num_inference_steps", int),
        ("guidance_scale", float),
        ("cfg_scale", float),
        ("sde_gamma", float),
    ):
        value, source = _input_or_testcase_value(inputs, testcase, name, None)
        if value is not None:
            values[name] = convert(value)
            sources[name] = source
    return values, sources


def _generate_request(testcase: Mapping[str, Any], _model_root: Path) -> CaseResolution:
    request, sources = _generation_values(testcase)
    request["prompt"] = _prompt(testcase, "generate")
    sources["prompt"] = _MODEL_TESTCASE
    return _resolution(request, sources, testcase=testcase)


def _vision_language_request(testcase: Mapping[str, Any], model_root: Path) -> CaseResolution:
    request, sources = _generation_values(testcase)
    request.update(
        {
            "prompt": _prompt(testcase, "vision_language_generation"),
            "image_path": _asset_field(
                testcase, model_root, "test_image", "vision_language_generation"
            ),
        }
    )
    sources.update({"prompt": _MODEL_TESTCASE, "image_path": _MODEL_TESTCASE})
    return _resolution(request, sources, testcase=testcase)


def _diffusion_text_request(testcase: Mapping[str, Any], _model_root: Path) -> CaseResolution:
    request, sources = _generation_values(testcase, operation="diffusion_text_generation")
    inputs = testcase.get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise BenchmarkError("diffusion_text_generation testcase inputs must be an object")
    source_text = inputs.get("source_text", testcase.get("prompt", ""))
    if not isinstance(source_text, str):
        raise BenchmarkError("diffusion_text_generation source_text must be a string")
    request["prompt"] = source_text
    sources["prompt"] = _MODEL_TESTCASE if source_text else _TASK_DEFAULT
    request["generation_mode"] = str(inputs.get("generation_mode", "diffusion"))
    sources["generation_mode"] = _MODEL_TESTCASE if "generation_mode" in inputs else _TASK_DEFAULT
    mappings = (
        ("num_sampling_steps", "num_inference_steps", int),
        ("self_cond_cfg_scale", "guidance_scale", float),
        ("cfg_scale", "cfg_scale", float),
        ("sde_gamma", "sde_gamma", float),
    )
    for source_name, target_name, convert in mappings:
        if source_name in inputs:
            request[target_name] = convert(inputs[source_name])
            sources[target_name] = _MODEL_TESTCASE
    return _resolution(request, sources, testcase=testcase)


def _prompt_from_file(testcase: Mapping[str, Any], model_root: Path) -> tuple[str, str | None]:
    prompt = testcase.get("prompt", testcase.get("test_prompt"))
    if isinstance(prompt, str) and prompt:
        return prompt, None
    declared = testcase.get("prompt_file")
    if not isinstance(declared, str) or not declared:
        raise BenchmarkError("generate_image testcase requires prompt/test_prompt/prompt_file")
    portable, resolved = _model_asset(declared, model_root, "generate_image")
    try:
        value = resolved.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise BenchmarkError(f"cannot read generate_image prompt file {resolved}: {exc}") from exc
    if not value:
        raise BenchmarkError(f"generate_image prompt file is empty: {resolved}")
    return value, str(portable)


def _sana_runtime(
    testcase: Mapping[str, Any], image_path: str
) -> tuple[dict[str, Any], dict[str, str]]:
    if "action" not in testcase and "camera_intrinsics" not in testcase:
        return {}, {}
    config: dict[str, Any] = {"sana_wm.image_path": image_path}
    for source, target in (
        ("action", "sana_wm.action"),
        ("translation_speed", "sana_wm.translation_speed"),
        ("rotation_speed_deg", "sana_wm.rotation_speed_deg"),
        ("video_num_frames", "sana_wm.num_frames"),
        ("fps", "sana_wm.fps"),
        ("flow_shift", "sana_wm.flow_shift"),
    ):
        if source in testcase:
            config[target] = testcase[source]
    intrinsics = testcase.get("camera_intrinsics")
    if isinstance(intrinsics, list) and intrinsics:
        config["sana_wm.intrinsics"] = ",".join(str(value) for value in intrinsics)
    return {"config": config}, {"config": _MODEL_TESTCASE}


def _generate_image_request(testcase: Mapping[str, Any], model_root: Path) -> CaseResolution:
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
        batch_size = int(inputs.get("expected_batch_size", len(prompts)))
        if batch_size != len(prompts):
            raise BenchmarkError("generate_image expected_batch_size must match batch_prompts")
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
        height, height_source = 0, _TASK_DEFAULT
    if "image_width" in testcase:
        width, width_source = testcase["image_width"], _MODEL_TESTCASE
    elif video_width is not None:
        width, width_source = video_width, _MODEL_TESTCASE
    else:
        width, width_source = 0, _TASK_DEFAULT
    prompt, prompt_path = _prompt_from_file(testcase, model_root)
    request: dict[str, Any] = {
        "batch_size": batch_size,
        "media_type": "video" if is_video else "image",
        "prompt": prompt,
        "height": int(height),
        "width": int(width),
    }
    sources: dict[str, str] = {
        "batch_size": batch_size_source,
        "media_type": _MODEL_TESTCASE if is_video else _TASK_DEFAULT,
        "prompt": _MODEL_TESTCASE,
        "height": height_source,
        "width": width_source,
    }
    for name, default, convert in (
        ("num_inference_steps", -1, int),
        ("seed", 42, int),
        ("guidance_scale", -1.0, float),
        ("cfg_scale", -1.0, float),
        ("negative_prompt", "", str),
    ):
        value, source = _testcase_value(testcase, name, default)
        request[name] = convert(value)
        sources[name] = source
    for name, convert in (("flow_shift", float), ("fps", int)):
        if name in testcase:
            request[name] = convert(testcase[name])
            sources[name] = _MODEL_TESTCASE
    if prompt_path is not None:
        request["prompt_path"] = prompt_path
        sources["prompt_path"] = _MODEL_TESTCASE
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
        request.update({"prompts": prompts, "seeds": seeds})
        sources.update({"prompts": _MODEL_TESTCASE, "seeds": _MODEL_TESTCASE})

    declared_image = testcase.get("test_image")
    runtime: dict[str, Any] = {}
    runtime_sources: dict[str, str] = {}
    if declared_image is not None:
        image_path = _asset_field(testcase, model_root, "test_image", "generate_image")
        if batch_size != 1:
            raise BenchmarkError("image-conditioned generate_image supports batch_size=1 only")
        request["image_path"] = image_path
        sources["image_path"] = _MODEL_TESTCASE
        runtime, runtime_sources = _sana_runtime(testcase, image_path)
    return _resolution(
        request,
        sources,
        testcase=testcase,
        runtime=runtime,
        runtime_sources=runtime_sources,
    )


def _generate_audio_request(testcase: Mapping[str, Any], _model_root: Path) -> CaseResolution:
    runtime_config = testcase.get("runtime_config", {})
    if not isinstance(runtime_config, Mapping):
        raise BenchmarkError("generate_audio runtime_config must be an object")
    config = _flatten_runtime_config(runtime_config)
    request = {
        "batch_size": 1,
        "prompt": _prompt(testcase, "generate_audio"),
        "max_new_tokens": int(testcase.get("max_new_tokens", 0)),
    }
    sources = {
        "batch_size": _TASK_DEFAULT,
        "prompt": _MODEL_TESTCASE,
        "max_new_tokens": _MODEL_TESTCASE if "max_new_tokens" in testcase else _TASK_DEFAULT,
    }
    seed = _generate_audio_seed(testcase, config)
    if seed is not None:
        request["seed"] = seed
        sources["seed"] = _MODEL_TESTCASE
    runtime = {"config": config} if config else {}
    runtime_sources = {"config": _MODEL_TESTCASE} if config else {}
    return _resolution(
        request,
        sources,
        testcase=testcase,
        runtime=runtime,
        runtime_sources=runtime_sources,
    )


def _generate_audio_seed(
    testcase: Mapping[str, Any], runtime_config: Mapping[str, Any]
) -> int | None:
    inputs = testcase.get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise BenchmarkError("generate_audio testcase inputs must be an object")
    determinism = testcase.get("determinism", {})
    if not isinstance(determinism, Mapping):
        raise BenchmarkError("generate_audio testcase determinism must be an object")

    declared: Any = None
    found = False
    for values in (inputs, testcase, determinism):
        if "seed" in values:
            declared = values["seed"]
            found = True
            break

    if not found:
        runtime_seeds = [
            value for name, value in runtime_config.items() if name.endswith(".seed")
        ]
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in runtime_seeds
        ):
            raise BenchmarkError("generate_audio seed must be an integer")
        distinct_runtime_seeds = set(runtime_seeds)
        if len(distinct_runtime_seeds) > 1:
            raise BenchmarkError("generate_audio runtime config has conflicting seed values")
        if runtime_seeds:
            declared = runtime_seeds[0]
            found = True

    if not found:
        return None
    if isinstance(declared, bool) or not isinstance(declared, int):
        raise BenchmarkError("generate_audio seed must be an integer")
    return declared


def _flatten_runtime_config(value: Mapping[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    def visit(prefix: str, nested: Any) -> None:
        if isinstance(nested, Mapping):
            for name, item in nested.items():
                child = f"{prefix}.{name}" if prefix else str(name)
                visit(child, item)
            return
        if not prefix:
            raise BenchmarkError("runtime config values must belong to a namespace")
        flattened[prefix] = nested

    visit("", value)
    return flattened


def _speak_request(testcase: Mapping[str, Any], model_root: Path) -> CaseResolution:
    max_frames = testcase.get("speech_test_max_frames", testcase.get("max_new_tokens", 50))
    max_frames_source = (
        _MODEL_TESTCASE
        if "speech_test_max_frames" in testcase or "max_new_tokens" in testcase
        else _TASK_DEFAULT
    )
    request = {
        "batch_size": 1,
        "audio_path": _asset_field(testcase, model_root, "test_input_audio", "speak"),
        "max_new_tokens": int(max_frames),
        "tail_frames": int(testcase.get("tail_frames", 0)),
    }
    sources = {
        "batch_size": _TASK_DEFAULT,
        "audio_path": _MODEL_TESTCASE,
        "max_new_tokens": max_frames_source,
        "tail_frames": _MODEL_TESTCASE if "tail_frames" in testcase else _TASK_DEFAULT,
    }
    return _resolution(request, sources, testcase=testcase)


def _image_request(
    testcase: Mapping[str, Any], model_root: Path, operation: str
) -> tuple[dict[str, Any], dict[str, str]]:
    return (
        {
            "batch_size": 1,
            "image_path": _asset_field(testcase, model_root, "test_image", operation),
        },
        {"batch_size": _TASK_DEFAULT, "image_path": _MODEL_TESTCASE},
    )


def _prompted_segmentation_request(testcase: Mapping[str, Any], model_root: Path) -> CaseResolution:
    request, sources = _image_request(testcase, model_root, "segment_prompted")
    if "point_x" in testcase or "point_y" in testcase:
        request.update(
            {
                "point_x": float(testcase.get("point_x", 0.5)),
                "point_y": float(testcase.get("point_y", 0.5)),
                "is_foreground": bool(testcase.get("is_foreground", True)),
            }
        )
        sources.update(
            {
                "point_x": _MODEL_TESTCASE if "point_x" in testcase else _TASK_DEFAULT,
                "point_y": _MODEL_TESTCASE if "point_y" in testcase else _TASK_DEFAULT,
                "is_foreground": (
                    _MODEL_TESTCASE if "is_foreground" in testcase else _TASK_DEFAULT
                ),
            }
        )
    else:
        request["prompt"] = _prompt(testcase, "segment_prompted")
        sources["prompt"] = _MODEL_TESTCASE
    return _resolution(request, sources, testcase=testcase)


def _segmentation_request(testcase: Mapping[str, Any], model_root: Path) -> CaseResolution:
    request, sources = _image_request(testcase, model_root, "segment")
    return _resolution(request, sources, testcase=testcase)


def _classification_request(testcase: Mapping[str, Any], model_root: Path) -> CaseResolution:
    request, sources = _image_request(testcase, model_root, "classify")
    return _resolution(request, sources, testcase=testcase)


def _image_feature_extraction_request(
    testcase: Mapping[str, Any], model_root: Path
) -> CaseResolution:
    request, sources = _image_request(testcase, model_root, "extract_features")
    return _resolution(request, sources, testcase=testcase)


def _detection_request(testcase: Mapping[str, Any], model_root: Path) -> CaseResolution:
    inputs = testcase.get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise BenchmarkError("detect testcase inputs must be an object")
    declared = testcase.get(
        "test_image",
        inputs.get("image", inputs.get("test_image", inputs.get("image_path"))),
    )
    if not isinstance(declared, str) or not declared:
        raise BenchmarkError("detect testcase requires test_image or inputs.image")
    portable, _ = _model_asset(declared, model_root, "detect")
    threshold, threshold_source = _input_or_testcase_value(inputs, testcase, "score_threshold", 0.3)
    return _resolution(
        {
            "batch_size": 1,
            "image_path": str(portable),
            "score_threshold": float(threshold),
        },
        {
            "batch_size": _TASK_DEFAULT,
            "image_path": _MODEL_TESTCASE,
            "score_threshold": threshold_source,
        },
        testcase=testcase,
    )


def _rerank_request(testcase: Mapping[str, Any], _model_root: Path) -> CaseResolution:
    inputs = testcase.get("inputs")
    if not isinstance(inputs, Mapping):
        raise BenchmarkError("rerank testcase requires an inputs object")
    query = inputs.get("prompt", inputs.get("query"))
    documents = inputs.get("documents")
    if not isinstance(query, str) or not query:
        raise BenchmarkError("rerank testcase requires inputs.prompt/query")
    if (
        not isinstance(documents, list)
        or not documents
        or not all(isinstance(document, str) and document for document in documents)
    ):
        raise BenchmarkError("rerank testcase requires non-empty inputs.documents strings")
    return _resolution(
        {"batch_size": 1, "query": query, "documents": list(documents)},
        {"batch_size": _TASK_DEFAULT, "query": _MODEL_TESTCASE, "documents": _MODEL_TESTCASE},
        testcase=testcase,
    )


def _text_input_request(
    operation: str, testcase: Mapping[str, Any], _model_root: Path
) -> CaseResolution:
    return _resolution(
        {"batch_size": 1, "prompt": _prompt(testcase, operation)},
        {"batch_size": _TASK_DEFAULT, "prompt": _MODEL_TESTCASE},
        testcase=testcase,
    )


def _encode_request(testcase: Mapping[str, Any], model_root: Path) -> CaseResolution:
    return _text_input_request("encode", testcase, model_root)


def _embed_request(testcase: Mapping[str, Any], model_root: Path) -> CaseResolution:
    return _text_input_request("embed", testcase, model_root)


def _solve_request(testcase: Mapping[str, Any], _model_root: Path) -> CaseResolution:
    inputs = testcase.get("inputs")
    if not isinstance(inputs, Mapping):
        raise BenchmarkError("solve testcase requires an inputs object")
    request: dict[str, Any] = {"batch_size": 1}
    sources = {"batch_size": _TASK_DEFAULT}
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
    return _resolution(request, sources, testcase=testcase)


def _transcribe_request(testcase: Mapping[str, Any], model_root: Path) -> CaseResolution:
    streaming = testcase.get("streaming", {})
    if not isinstance(streaming, Mapping):
        raise BenchmarkError("transcribe testcase streaming configuration must be an object")
    max_new_tokens, max_new_tokens_source = _testcase_value(testcase, "max_new_tokens", 224)
    language, language_source = _testcase_value(testcase, "language", "")
    return _resolution(
        {
            "batch_size": 1,
            "audio_path": _asset_field(testcase, model_root, "test_input_audio", "transcribe"),
            "max_new_tokens": int(max_new_tokens),
            "language": str(language),
            "streaming": dict(streaming),
        },
        {
            "batch_size": _TASK_DEFAULT,
            "audio_path": _MODEL_TESTCASE,
            "max_new_tokens": max_new_tokens_source,
            "language": language_source,
            "streaming": _MODEL_TESTCASE if "streaming" in testcase else _TASK_DEFAULT,
        },
        testcase=testcase,
    )


_TASK_ADAPTERS = (
    TaskAdapter(
        "text_generation_causal",
        "generate",
        MeasurementSpec(warmup=5, iterations=50),
        _generate_request,
    ),
    TaskAdapter(
        "vision_language_generation",
        "generate",
        MeasurementSpec(warmup=1, iterations=10),
        _vision_language_request,
    ),
    TaskAdapter(
        "diffusion_text_generation",
        "generate",
        MeasurementSpec(warmup=1, iterations=5),
        _diffusion_text_request,
    ),
    TaskAdapter(
        "diffusion_media_generation",
        "generate_image",
        MeasurementSpec(warmup=1, iterations=5),
        _generate_image_request,
    ),
    TaskAdapter(
        "text_to_audio",
        "generate_audio",
        MeasurementSpec(warmup=1, iterations=10),
        _generate_audio_request,
    ),
    TaskAdapter(
        "omni_multimodal",
        "generate_audio",
        MeasurementSpec(warmup=1, iterations=5),
        _generate_audio_request,
    ),
    TaskAdapter(
        "speech_to_speech",
        "speak",
        MeasurementSpec(warmup=1, iterations=10),
        _speak_request,
    ),
    TaskAdapter(
        "prompted_segmentation",
        "segment_prompted",
        MeasurementSpec(warmup=10, iterations=100),
        _prompted_segmentation_request,
    ),
    TaskAdapter(
        "segmentation",
        "segment",
        MeasurementSpec(warmup=50, iterations=500),
        _segmentation_request,
    ),
    TaskAdapter(
        "image_classification",
        "classify",
        MeasurementSpec(warmup=50, iterations=500),
        _classification_request,
    ),
    TaskAdapter(
        "image_feature_extraction",
        "extract_features",
        MeasurementSpec(warmup=50, iterations=500),
        _image_feature_extraction_request,
    ),
    TaskAdapter(
        "object_detection",
        "detect",
        MeasurementSpec(warmup=50, iterations=500),
        _detection_request,
    ),
    TaskAdapter(
        "reranking",
        "rerank",
        MeasurementSpec(warmup=10, iterations=100),
        _rerank_request,
    ),
    TaskAdapter(
        "encoder_only_nlp",
        "encode",
        MeasurementSpec(warmup=50, iterations=500),
        _encode_request,
    ),
    TaskAdapter(
        "embedding",
        "embed",
        MeasurementSpec(warmup=50, iterations=500),
        _embed_request,
    ),
    TaskAdapter(
        "neural_operator",
        "solve",
        MeasurementSpec(warmup=50, iterations=500),
        _solve_request,
    ),
    TaskAdapter(
        "speech_to_text",
        "transcribe",
        MeasurementSpec(warmup=1, iterations=10),
        _transcribe_request,
    ),
)


def _index_task_adapters() -> dict[str, TaskAdapter]:
    indexed: dict[str, TaskAdapter] = {}
    for adapter in _TASK_ADAPTERS:
        if adapter.task_strategy in indexed:
            raise RuntimeError(f"duplicate benchmark task adapter {adapter.task_strategy!r}")
        operation_for_name(adapter.operation)
        indexed[adapter.task_strategy] = adapter
    return indexed


_BY_TASK_STRATEGY = _index_task_adapters()


def adapter_for_task_strategy(task_strategy: str) -> TaskAdapter:
    try:
        return _BY_TASK_STRATEGY[task_strategy]
    except KeyError as exc:
        available = ", ".join(sorted(_BY_TASK_STRATEGY))
        raise BenchmarkError(
            f"task strategy {task_strategy!r} has no benchmark adapter; available: {available}"
        ) from exc


def registered_task_adapters() -> tuple[TaskAdapter, ...]:
    return _TASK_ADAPTERS
