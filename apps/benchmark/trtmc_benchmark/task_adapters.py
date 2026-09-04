# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map family-owned manifest tasks to public Task API benchmark calls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .types import BenchmarkError, MeasurementSpec


_MANIFEST = "family manifest"
_DEFAULT = "task default"


@dataclass(frozen=True)
class CaseResolution:
    operation: str
    request: Mapping[str, Any]
    sources: Mapping[str, str]
    measurement: MeasurementSpec


_DEFAULTS: dict[str, tuple[str, int, int]] = {
    "text_generation": ("generate", 5, 50),
    "vision_language_generation": ("generate", 1, 10),
    "image_generation": ("generate_image", 1, 5),
    "image_edit": ("generate_image", 1, 5),
    "image_generation_batch": ("generate_image", 1, 5),
    "world_model_generation": ("generate_image", 1, 5),
    "audio_generation": ("generate_audio", 1, 10),
    "speech_to_speech": ("speak", 1, 10),
    "transcription": ("transcribe", 1, 10),
    "transcription_streaming": ("transcribe", 1, 10),
    "embedding": ("embed", 50, 500),
    "encoding": ("encode", 50, 500),
    "reranking": ("rerank", 10, 100),
    "segmentation": ("segment", 50, 500),
    "prompted_segmentation": ("segment_prompted", 10, 100),
    "text_prompted_segmentation": ("segment_prompted", 10, 100),
    "classification": ("classify", 50, 500),
    "image_features": ("extract_features", 50, 500),
    "stereo_disparity": ("disparity", 3, 100),
    "time_series_forecast": ("solve", 50, 500),
    "robot_control": ("control", 2, 10),
}

_ALLOWED_OPERATIONS: dict[str, frozenset[str]] = {
    task: frozenset({operation}) for task, (operation, _, _) in _DEFAULTS.items()
}
_ALLOWED_OPERATIONS["embedding"] = frozenset({"embed", "encode"})


def supported_tasks() -> tuple[str, ...]:
    return tuple(sorted(_DEFAULTS))


def default_operation(task: str) -> str:
    try:
        return _DEFAULTS[task][0]
    except KeyError as error:
        raise BenchmarkError(f"task {task!r} has no benchmark implementation") from error


def resolve_task_case(
    task: str,
    testcase: Mapping[str, Any],
    model_root: Path,
    *,
    operation: str | None = None,
) -> CaseResolution:
    try:
        default_operation, warmup, iterations = _DEFAULTS[task]
    except KeyError as error:
        raise BenchmarkError(f"task {task!r} has no benchmark implementation") from error
    selected_operation = operation or default_operation
    if selected_operation not in _ALLOWED_OPERATIONS[task]:
        allowed = ", ".join(sorted(_ALLOWED_OPERATIONS[task]))
        raise BenchmarkError(
            f"task {task!r} cannot run operation {selected_operation!r}; expected {allowed}"
        )
    request = _request(task, testcase, model_root)
    sources = {name: _MANIFEST for name in request}
    return CaseResolution(
        selected_operation,
        request,
        sources,
        MeasurementSpec(warmup=warmup, iterations=iterations),
    )


def _request(task: str, case: Mapping[str, Any], root: Path) -> dict[str, Any]:
    if task == "text_generation":
        return _text_request(case, root)
    if task == "vision_language_generation":
        return {**_text_request(case, root), "image_path": _image_path(case, root)}
    if task in {"image_generation", "image_edit", "image_generation_batch"}:
        return _image_generation_request(task, case, root)
    if task == "world_model_generation":
        return _world_request(case, root)
    if task == "audio_generation":
        return _audio_generation_request(case, root)
    if task == "speech_to_speech":
        return {
            "audio_path": _audio_path(case, root),
            "max_new_tokens": int(
                case.get("speech_test_max_frames", case.get("max_new_tokens", 50))
            ),
            "seed": int(case.get("seed", -1)),
            "tail_frames": int(_inputs(case).get("tail_frames", 0)),
        }
    if task in {"transcription", "transcription_streaming"}:
        return _transcription_request(case, root, streaming=task.endswith("_streaming"))
    if task in {"embedding", "encoding"}:
        return {"prompt": _prompt(case, root), "batch_size": 1}
    if task == "reranking":
        inputs = _inputs(case)
        query = inputs.get("query", inputs.get("prompt", case.get("prompt")))
        documents = inputs.get("documents")
        if not isinstance(query, str) or not query:
            raise BenchmarkError("reranking testcase requires inputs.prompt or inputs.query")
        if not isinstance(documents, list) or not documents:
            raise BenchmarkError("reranking testcase requires non-empty inputs.documents")
        return {"query": query, "documents": [str(value) for value in documents]}
    if task in {
        "segmentation",
        "prompted_segmentation",
        "text_prompted_segmentation",
        "classification",
        "image_features",
    }:
        request: dict[str, Any] = {"image_path": _image_path(case, root), "batch_size": 1}
        if task == "prompted_segmentation":
            request.update(
                point_x=float(case.get("point_x", 0.5)),
                point_y=float(case.get("point_y", 0.5)),
                is_foreground=bool(case.get("is_foreground", True)),
            )
        if task == "text_prompted_segmentation":
            request["prompt"] = _prompt(case, root)
        return request
    if task == "stereo_disparity":
        inputs = _inputs(case)
        return {
            "left_image_path": _required_asset(inputs, ("left_image",), root, "left image"),
            "right_image_path": _required_asset(inputs, ("right_image",), root, "right image"),
        }
    if task == "time_series_forecast":
        inputs = _inputs(case)
        values = inputs.get("past_values")
        if not isinstance(values, list) or not values:
            raise BenchmarkError("forecast testcase requires inputs.past_values")
        frequency = inputs.get("frequency", 0)
        if isinstance(frequency, bool) or not isinstance(frequency, int):
            raise BenchmarkError("forecast frequency must be an integer")
        request = {
            "past_values": [float(value) for value in values],
            "frequency": frequency,
        }
        mask = inputs.get("observed_mask")
        if mask is not None:
            if not isinstance(mask, list):
                raise BenchmarkError("inputs.observed_mask must be a list")
            request["observed_mask"] = [float(value) for value in mask]
        return request
    if task == "robot_control":
        inputs = _inputs(case)
        return {
            "image_path": _required_asset(inputs, ("image",), root, "observation image"),
            "state_path": _required_asset(inputs, ("state",), root, "observation state"),
        }
    raise BenchmarkError(f"task {task!r} has no request resolver")


def _text_request(case: Mapping[str, Any], root: Path) -> dict[str, Any]:
    inputs = _inputs(case)
    request: dict[str, Any] = {
        "prompt": _prompt(case, root),
        "max_new_tokens": int(case.get("max_new_tokens", 128)),
        "temperature": float(case.get("temperature", inputs.get("temperature", 1.0))),
        "top_k": int(case.get("top_k", 1)),
        "top_p": float(case.get("top_p", 1.0)),
        "min_p": float(case.get("min_p", 0.0)),
        "seed": int(case.get("seed", -1)),
        "repetition_penalty": float(case.get("repetition_penalty", 1.0)),
        "use_chat_template": bool(case.get("use_chat_template", False)),
        "enable_thinking": bool(case.get("enable_thinking", True)),
    }
    if "generation_mode" in inputs:
        request["text_generation_mode"] = str(inputs["generation_mode"])
    if "block_length" in inputs:
        request["block_length"] = int(inputs["block_length"])
    if "threshold" in inputs:
        request["confidence_threshold"] = float(inputs["threshold"])
    for source, target in (
        ("guidance_scale", "guidance_scale"),
        ("cfg_scale", "cfg_scale"),
        ("num_inference_steps", "num_steps"),
    ):
        if source in case:
            request[target] = case[source]
        elif source in inputs:
            request[target] = inputs[source]
    return request


def _image_generation_request(task: str, case: Mapping[str, Any], root: Path) -> dict[str, Any]:
    inputs = _inputs(case)
    prompts = inputs.get("batch_prompts")
    if task == "image_generation_batch":
        if not isinstance(prompts, list) or not prompts:
            raise BenchmarkError("batch image testcase requires inputs.batch_prompts")
        prompt_value: str | list[str] = [str(value) for value in prompts]
    else:
        prompt_value = _prompt(case, root)
    request: dict[str, Any] = {
        "prompt": prompt_value,
        "negative_prompt": str(case.get("negative_prompt", "")),
        "seed": int(case.get("seed", -1)),
        "num_steps": int(case.get("num_inference_steps", inputs.get("num_sampling_steps", -1))),
        "guidance_scale": float(case.get("guidance_scale", inputs.get("guidance_scale", -1.0))),
        "cfg_scale": float(case.get("cfg_scale", inputs.get("cfg_scale", -1.0))),
        "height": int(case.get("height", 0)),
        "width": int(case.get("width", 0)),
        "batch_size": len(prompt_value) if isinstance(prompt_value, list) else 1,
    }
    if task == "image_edit":
        request["image_path"] = _image_path(case, root)
    seeds = inputs.get("batch_seeds")
    if seeds is not None:
        request["seeds"] = [int(value) for value in seeds]
    return request


def _world_request(case: Mapping[str, Any], root: Path) -> dict[str, Any]:
    request = _image_generation_request("image_generation", case, root)
    request.update(
        image_path=_image_path(case, root),
        action=str(case.get("action", "")),
        camera_intrinsics=[float(value) for value in case.get("camera_intrinsics", [])],
        media_type="video",
    )
    return request


def _audio_generation_request(case: Mapping[str, Any], root: Path) -> dict[str, Any]:
    request = {
        "prompt": _prompt(case, root),
        "max_new_tokens": int(case.get("max_new_tokens", 128)),
        "seed": int(case.get("seed", -1)),
    }
    if "talker_max_new_tokens" in case:
        request["talker_max_new_tokens"] = int(case["talker_max_new_tokens"])
    if "speaker" in case:
        request["speaker"] = str(case["speaker"])
    return request


def _transcription_request(
    case: Mapping[str, Any], root: Path, *, streaming: bool
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "audio_path": _audio_path(case, root),
        "max_new_tokens": int(case.get("max_new_tokens", 224)),
        "language": str(case.get("language", "")),
        "streaming": streaming,
    }
    if streaming:
        request["chunk_ms"] = int(case.get("chunk_ms", 160))
    return request


def _inputs(case: Mapping[str, Any]) -> Mapping[str, Any]:
    value = case.get("inputs", {})
    if not isinstance(value, Mapping):
        raise BenchmarkError("testcase inputs must be an object")
    return value


def _prompt(case: Mapping[str, Any], root: Path) -> str:
    for key in ("prompt", "test_prompt"):
        value = case.get(key)
        if isinstance(value, str) and value:
            return value
    inputs = _inputs(case)
    for key in ("prompt", "source_text"):
        value = inputs.get(key)
        if isinstance(value, str) and value:
            return value
    repeated = case.get("prompt_repeat")
    if isinstance(repeated, Mapping):
        count = int(repeated.get("count", 0))
        if count < 1:
            raise BenchmarkError("prompt_repeat.count must be positive")
        return str(repeated.get("separator", "")).join(
            [str(repeated.get("text", ""))] * count
        ) + str(repeated.get("suffix", ""))
    prompt_file = case.get("prompt_file")
    if isinstance(prompt_file, str) and prompt_file:
        path = _asset(prompt_file, root)
        value = path.read_text(encoding="utf-8").strip()
        if path.suffix == ".json":
            parsed = json.loads(value)
            value = str(parsed.get("prompt", ""))
        if value:
            return value
    raise BenchmarkError("testcase requires a non-empty prompt")


def _image_path(case: Mapping[str, Any], root: Path) -> str:
    return _required_asset(case, ("test_image", "image"), root, "image")


def _audio_path(case: Mapping[str, Any], root: Path) -> str:
    inputs = _inputs(case)
    for mapping, keys in (
        (case, ("test_input_audio", "audio_path")),
        (inputs, ("audio", "speech_source_relative_path")),
    ):
        try:
            return _required_asset(mapping, keys, root, "audio")
        except BenchmarkError:
            pass
    raise BenchmarkError("testcase requires an audio input")


def _required_asset(
    values: Mapping[str, Any], keys: tuple[str, ...], root: Path, label: str
) -> str:
    for key in keys:
        value = values.get(key)
        if isinstance(value, str) and value:
            return str(_asset(value, root))
    raise BenchmarkError(f"testcase requires {label}")


def _asset(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    resolved = path if path.is_absolute() else (root / path).resolve()
    if not resolved.is_file():
        raise BenchmarkError(f"benchmark asset does not exist: {resolved}")
    return resolved
