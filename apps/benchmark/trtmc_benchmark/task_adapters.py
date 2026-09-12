# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map family-owned manifest tasks to public Task API benchmark calls."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .types import BenchmarkError, MeasurementSpec


_MANIFEST = "family manifest"
_DEFAULT = "task default"
_BATCH_FORECAST = frozenset({
    "batch_series_to_point_forecast", "batch_series_to_quantile_forecast",
    "batch_series_to_point_and_quantile_forecast",
})
_BATCH_SPEECH = frozenset({
    "batch_speech_transcription", "batch_speech_translation", "mixed_batch_speech_to_text",
})
_LATENT_TEXT = frozenset({"latent_conditioned_text_generation", "latent_replay_to_text"})
_LATENT_STEPS = frozenset({"latent_denoising_step", "latent_to_token_logits"})
_SPEECH_DIALOGUE = frozenset({"duplex_speech_dialogue", "offline_speech_dialogue", "tool_speech_dialogue"})
_TRACKING = frozenset({"frames_to_detected_mask_tracks", "frames_text_to_mask_tracks", "prompt_frame_text_to_mask_tracks"})
_POSE = frozenset({"crop_pose_tracking", "pose_hypotheses_crops_to_refined_poses"})


@dataclass(frozen=True)
class CaseResolution:
    operation: str
    request: Mapping[str, Any]
    sources: Mapping[str, str]
    measurement: MeasurementSpec
    task: str


_DEFAULTS: dict[str, tuple[str, int, int]] = {
    "text_to_image": ("generate_image", 1, 5),
    "images_text_to_image_edit": ("generate_image", 1, 5),
    "batch_text_to_image": ("generate_image", 1, 5),
    "text_to_video": ("generate_image", 1, 5),
    "image_text_action_to_video": ("generate_image", 1, 5),
    "image_to_class_scores": ("classify", 50, 500),
    "image_to_boxes": ("detect", 50, 500),
    "molecular_document_to_structure": ("predict_structure", 1, 5),
    "image_to_token_and_pooled_features": ("extract_features", 50, 500),
    "image_to_token_features": ("extract_features", 50, 500),
    "image_to_pooled_features": ("extract_features", 50, 500),
    "image_to_spatial_features": ("extract_features", 50, 500),
    "image_to_semantic_segmentation": ("segment", 50, 500),
    "image_points_to_masks": ("segment", 50, 500),
    "image_text_to_instance_masks": ("segment_prompted", 10, 100),
    "stereo_images_to_disparity": ("disparity", 3, 100),
    "image_to_metric_geometry": ("geometry", 3, 100),
    "text_to_pooled_features": ("encode", 50, 500),
    "text_to_head_scores": ("head_scores", 50, 500),
    "text_to_token_features": ("encode", 50, 500),
    "text_to_embedding": ("embed", 50, 500),
    "text_query_documents_to_relevance": ("rerank", 10, 100),
    "image_state_to_action_chunk": ("control", 2, 10),
    "image_state_action_queue": ("control_queue", 2, 10),
    "duplex_speech_dialogue": ("speech_dialogue", 1, 5),
    "offline_speech_dialogue": ("speech_dialogue", 1, 5),
    "tool_speech_dialogue": ("speech_dialogue", 1, 5),
    "frames_to_detected_mask_tracks": ("track_masks", 1, 5),
    "frames_text_to_mask_tracks": ("track_masks", 1, 5),
    "prompt_frame_text_to_mask_tracks": ("track_masks", 1, 5),
    "crop_pose_tracking": ("track_pose", 1, 5),
    "pose_hypotheses_crops_to_refined_poses": ("refine_pose", 1, 5),
    "text_continuation": ("generate", 5, 50),
    "conditional_text_generation": ("generate", 5, 50),
    "corrupted_text_reconstruction": ("generate", 5, 50),
    "text_summarization": ("generate", 5, 50),
    "unconditional_text_generation": ("generate", 5, 50),
    "latent_conditioned_text_generation": ("generate", 5, 50),
    "latent_replay_to_text": ("generate", 5, 50),
    "latent_denoising_step": ("denoise", 5, 50),
    "latent_to_token_logits": ("decode_logits", 5, 50),
    "text_translation": ("translate", 5, 50),
    "images_text_to_text": ("generate", 1, 10),
    "series_to_point_forecast": ("solve", 50, 500),
    "series_to_quantile_forecast": ("solve", 50, 500),
    "series_to_point_and_quantile_forecast": ("solve", 50, 500),
    "series_to_regression_distribution": ("regress", 50, 500),
    "series_to_regression_values": ("regress", 50, 500),
    "batch_series_to_point_forecast": ("solve", 50, 500),
    "batch_series_to_quantile_forecast": ("solve", 50, 500),
    "batch_series_to_point_and_quantile_forecast": ("solve", 50, 500),
    "text_to_audio": ("generate_audio", 1, 10),
    "text_to_speech": ("generate_audio", 1, 10),
    "streaming_text_to_speech": ("generate_audio", 1, 10),
    "speech_to_speech_response": ("speak", 1, 10),
    "speech_transcription": ("transcribe", 1, 10),
    "speech_translation": ("transcribe", 1, 10),
    "streaming_speech_transcription": ("transcribe", 1, 10),
    "batch_speech_transcription": ("transcribe", 1, 10),
    "batch_speech_translation": ("transcribe", 1, 10),
    "mixed_batch_speech_to_text": ("transcribe", 1, 10),
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
    "object_detection": ("detect", 50, 500),
    "image_features": ("extract_features", 50, 500),
    "stereo_disparity": ("disparity", 3, 100),
    "time_series_forecast": ("solve", 50, 500),
    "robot_control": ("control", 2, 10),
}

_ALLOWED_OPERATIONS: dict[str, frozenset[str]] = {
    task: frozenset({operation}) for task, (operation, _, _) in _DEFAULTS.items()
}
_ALLOWED_OPERATIONS["embedding"] = frozenset({"embed", "encode"})
_ALLOWED_OPERATIONS["text_to_embedding"] = frozenset({"embed", "encode"})
_ALLOWED_OPERATIONS["image_points_to_masks"] = frozenset({"segment", "segment_prompted"})

_SEMANTIC_REMAINING = frozenset({
    "text_to_image", "images_text_to_image_edit", "batch_text_to_image", "text_to_video",
    "image_text_action_to_video", "image_to_class_scores", "image_to_token_and_pooled_features",
    "image_to_token_features", "image_to_pooled_features", "image_to_spatial_features",
    "image_to_semantic_segmentation", "image_points_to_masks", "image_text_to_instance_masks",
    "stereo_images_to_disparity", "text_to_pooled_features", "text_to_token_features",
    "text_to_embedding", "text_query_documents_to_relevance", "image_state_to_action_chunk",
    "text_to_head_scores",
})


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
    request_task = "text_to_pooled_features" if (
        task == "text_to_embedding" and selected_operation == "encode"
    ) else task
    request = _request(request_task, testcase, model_root)
    if _explicit(testcase, "token_ids") is not _MISSING and "token_ids" not in request:
        raise BenchmarkError(f"token_ids is not accepted by {request_task}")
    if task == "image_points_to_masks" and selected_operation == "segment" and any(
        name in request for name in ("point_x", "point_y", "is_foreground")
    ):
        raise BenchmarkError("segment uses the center helper; explicit point controls require segment_prompted")
    if "config" in testcase:
        if task in _BATCH_SPEECH:
            raise BenchmarkError("batch speech Config belongs to each inputs.items entry")
        if task in _BATCH_FORECAST:
            raise BenchmarkError("batch forecast Config belongs to each inputs.items entry")
        if task not in _SEMANTIC_REMAINING | _LATENT_TEXT | _LATENT_STEPS | _SPEECH_DIALOGUE | _TRACKING | _POSE and task not in {
            "text_continuation", "conditional_text_generation", "corrupted_text_reconstruction",
            "unconditional_text_generation",
            "text_summarization", "text_translation", "images_text_to_text", "series_to_point_forecast",
            "series_to_quantile_forecast", "series_to_point_and_quantile_forecast",
            "series_to_regression_distribution", "series_to_regression_values", "image_to_metric_geometry",
            "image_to_boxes", "molecular_document_to_structure",
            "image_state_action_queue",
            "text_to_audio", "text_to_speech", "streaming_text_to_speech",
            "speech_to_speech_response", "speech_transcription", "speech_translation",
            "streaming_speech_transcription",
        }:
            raise BenchmarkError("family Config requires a semantic Task benchmark")
        if not isinstance(testcase["config"], Mapping):
            raise BenchmarkError("testcase config must be an object")
        request["config"] = dict(testcase["config"])
    if task in _SEMANTIC_REMAINING:
        _check_semantic_duplicates(request)
    sources = {name: _MANIFEST for name in request}
    if task == "molecular_document_to_structure":
        if _explicit(testcase, "input_encoding", "encoding") is _MISSING:
            sources["input_encoding"] = _DEFAULT
        if _explicit(testcase, "source_path") is _MISSING:
            sources["source_path"] = _DEFAULT
    if task in _SEMANTIC_REMAINING and "media_type" in request and _explicit(testcase, "media_type") is _MISSING:
        sources["media_type"] = _DEFAULT
    return CaseResolution(
        selected_operation,
        request,
        sources,
        MeasurementSpec(warmup=warmup, iterations=iterations),
        request_task,
    )


def _request(task: str, case: Mapping[str, Any], root: Path) -> dict[str, Any]:
    if task in _TRACKING:
        frames = _explicit(case, "frame_paths", "frames")
        if not isinstance(frames, list) or not frames or not all(isinstance(path, str) and path for path in frames):
            raise BenchmarkError("tracking requires nonempty frame_paths in complete clip order")
        request = {"frame_paths": [str(_asset(path, root)) for path in frames]}
        _copy_explicit(request, case, "timestamps_seconds", "device_masks", "segment_config")
        if task == "frames_to_detected_mask_tracks":
            if any(_explicit(case, name) is not _MISSING for name in ("prompt", "source_text", "prompt_file", "prompt_repeat")):
                raise BenchmarkError("detector tracking has no text prompt input")
        else:
            request["prompt"] = _semantic_prompt(case, root)
            if "device_masks" in request:
                raise BenchmarkError("device_masks requires frames_to_detected_mask_tracks")
        if task == "prompt_frame_text_to_mask_tracks" and "segment_config" in request:
            raise BenchmarkError("prompt-frame Task accepts Config at session creation only")
        return request
    if task == "pose_hypotheses_crops_to_refined_poses":
        return _pose_input(case, root)
    if task == "crop_pose_tracking":
        initial = _explicit(case, "initialization")
        updates = _explicit(case, "updates")
        if not isinstance(initial, Mapping) or not isinstance(updates, list) or not updates:
            raise BenchmarkError("pose tracking requires initialization and a nonempty updates array")
        prepared = []
        for update in updates:
            if not isinstance(update, Mapping):
                raise BenchmarkError("pose tracking updates must be objects")
            value = deepcopy(dict(update))
            value["crop_batches"] = _crop_batches(_explicit(update, "crop_batches"), root)
            prepared.append(value)
        return {"initialization": _pose_input(initial, root), "updates": prepared}
    if task == "image_state_action_queue":
        items = _explicit(case, "observations")
        if not isinstance(items, list) or not items or not all(isinstance(item, Mapping) for item in items):
            raise BenchmarkError("action queue requires a nonempty observations array")
        observations = []
        for item in items:
            value = deepcopy(dict(item))
            value["image_path"] = _semantic_asset(item, root, "image_path", "image")
            value["state_path"] = _semantic_asset(item, root, "state_path", "state")
            value.pop("image", None)
            value.pop("state", None)
            observations.append(value)
        return {"observations": observations}
    if task in _SPEECH_DIALOGUE:
        request = {"audio_path": _semantic_asset(case, root, "audio_path", "audio", "input")}
        _copy_explicit(request, case, "system_prompt", "chunk_frames", "timeout_ms")
        tool_fields = ("tools", "tool_replies", "acknowledgements", "default_acknowledgements")
        if task == "tool_speech_dialogue":
            tools = _explicit(case, "tools")
            replies = _explicit(case, "tool_replies")
            if not isinstance(tools, list) or not tools or not isinstance(replies, list):
                raise BenchmarkError("tool dialogue requires nonempty tools and an explicit tool_replies array")
            for name in tool_fields:
                value = _explicit(case, name)
                if value is not _MISSING:
                    request[name] = deepcopy(value)
        elif any(_explicit(case, name) is not _MISSING for name in tool_fields):
            raise BenchmarkError("tool definitions and preset replies require tool_speech_dialogue")
        return request
    if task == "image_to_boxes":
        if _explicit(case, "prompt", "test_prompt", "source_text") is not _MISSING:
            raise BenchmarkError("image-only detection has no prompt input")
        return {"image_path": _semantic_asset(case, root, "image_path", "image", "test_image")}
    if task == "molecular_document_to_structure":
        document = _semantic_asset(case, root, "document_path", "input_path", "input")
        encoding = _explicit(case, "input_encoding", "encoding")
        if encoding is _MISSING:
            encoding = {".yaml": "yaml", ".yml": "yaml", ".json": "json", ".b2rq": "b2rq"}.get(Path(document).suffix)
            if encoding is None:
                raise BenchmarkError("unknown structure request extension; specify input_encoding")
        source_path = _explicit(case, "source_path")
        request = {
            "document_path": document, "input_encoding": encoding,
            "source_path": document if source_path is _MISSING else source_path,
        }
        _copy_explicit(request, case, "seed", "include_confidence", "output_format")
        steps = _explicit(case, "sampling_steps", "num_steps")
        if steps is not _MISSING:
            request["sampling_steps"] = steps
        return request
    if task == "unconditional_text_generation":
        if any(_explicit(case, name) is not _MISSING for name in (
            "prompt", "test_prompt", "source_text", "prompt_file", "prompt_repeat", "token_ids",
        )):
            raise BenchmarkError("unconditional text generation has no prompt input")
        return _text_controls(case, include_defaults=False)
    if task in _LATENT_TEXT | _LATENT_STEPS:
        return _latent_request(task, case, root)
    if task == "image_to_metric_geometry":
        request = {"image_path": _semantic_asset(case, root, "image_path", "image", "test_image")}
        _copy_explicit(request, case, "fov_x")
        return request
    if task in {"series_to_regression_distribution", "series_to_regression_values"}:
        values = _explicit(case, "past_values")
        if not isinstance(values, list) or not values:
            raise BenchmarkError("regression testcase requires nonempty past_values")
        request = {"past_values": deepcopy(values)}
        _copy_explicit(request, case, "observed_mask", "shape", "frequency")
        if task == "series_to_regression_distribution":
            _copy_explicit(request, case, "distribution")
        elif _explicit(case, "distribution") is not _MISSING:
            raise BenchmarkError("deterministic target regression has no distribution selector")
        return request
    if task in _BATCH_SPEECH:
        for name in ("source_language", "target_language", "language", "max_new_tokens",
                     "max_output_tokens", "streaming"):
            if name in case:
                raise BenchmarkError(f"batch speech {name} belongs to each inputs.items entry")
        inputs = _inputs(case)
        items = inputs.get("items")
        if set(inputs) != {"items"} or not isinstance(items, list) or not items:
            raise BenchmarkError("batch speech requires only a nonempty inputs.items array")
        prepared = []
        for item in items:
            if not isinstance(item, Mapping):
                raise BenchmarkError("batch speech items must be objects")
            entry = deepcopy(dict(item))
            path = _semantic_asset(item, root, "audio_path", "audio")
            entry.pop("audio", None)
            entry["audio_path"] = path
            prepared.append(entry)
        return {"items": prepared}
    if task == "text_translation":
        request = {
            "source_text": _semantic_prompt(case, root),
            **_text_controls(case, include_defaults=False),
        }
        _copy_explicit(request, case, "source_language", "target_language")
        return request
    if task in _BATCH_FORECAST:
        inputs = _inputs(case)
        items = inputs.get("items")
        if set(inputs) != {"items"} or not isinstance(items, list) or not items:
            raise BenchmarkError("batch forecast requires only a nonempty inputs.items array")
        if any(not isinstance(item, Mapping) for item in items):
            raise BenchmarkError("batch forecast items must be objects")
        # Preserve masks, nulls, axes and per-item Config. The native Task owns validation.
        return {"items": deepcopy(items)}
    if task in _SEMANTIC_REMAINING:
        return _semantic_remaining_request(task, case, root)
    if task in {
        "text_to_audio", "text_to_speech", "streaming_text_to_speech",
        "speech_to_speech_response", "speech_transcription", "speech_translation",
        "streaming_speech_transcription",
    }:
        return _semantic_audio_request(task, case, root)
    if task in {"text_continuation", "conditional_text_generation"}:
        return {**_text_source_request(case, root), **_text_controls(case, include_defaults=False)}
    if task in {"corrupted_text_reconstruction", "text_summarization", "images_text_to_text"}:
        if _explicit(case, "token_ids") is not _MISSING:
            raise BenchmarkError(f"token_ids is not accepted by {task}")
        request = _text_request(case, root, include_defaults=False)
        if task == "images_text_to_text":
            request["image_path"] = _image_path(case, root)
        return request
    if task == "text_generation":
        if _explicit(case, "token_ids") is not _MISSING:
            raise BenchmarkError("token_ids requires a semantic TextSource Task")
        return _text_request(case, root)
    if task == "vision_language_generation":
        if _explicit(case, "token_ids") is not _MISSING:
            raise BenchmarkError("token_ids is not a vision-language text input")
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
        "object_detection",
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
    if task in {
        "time_series_forecast", "series_to_point_forecast", "series_to_quantile_forecast",
        "series_to_point_and_quantile_forecast",
    }:
        inputs = _inputs(case)
        values = inputs.get("past_values")
        if not isinstance(values, list) or not values:
            raise BenchmarkError("forecast testcase requires inputs.past_values")
        frequency = inputs.get("frequency", 0)
        if isinstance(frequency, bool) or not isinstance(frequency, int):
            raise BenchmarkError("forecast frequency must be an integer")
        request = {
            "past_values": [float(value) for value in values],
        }
        if "frequency" in inputs or task == "time_series_forecast":
            request["frequency"] = frequency
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


_MISSING = object()


def _explicit(case: Mapping[str, Any], *names: str) -> Any:
    values = [
        (name, source[name]) for source in (case, _inputs(case)) for name in names
        if name in source
    ]
    if len(values) > 1:
        raise BenchmarkError(f"duplicate input/control for {names[0]}: {', '.join(name for name, _ in values)}")
    return values[0][1] if values else _MISSING


def _semantic_asset(case: Mapping[str, Any], root: Path, *names: str) -> str:
    value = _explicit(case, *names)
    if not isinstance(value, str) or not value:
        raise BenchmarkError(f"testcase requires {names[0]} path")
    return str(_asset(value, root))


def _semantic_prompt(case: Mapping[str, Any], root: Path) -> str:
    prompt = _explicit(case, "prompt", "test_prompt", "source_text")
    repeated = _explicit(case, "prompt_repeat")
    file = _explicit(case, "prompt_file")
    if sum(value is not _MISSING for value in (prompt, repeated, file)) > 1:
        raise BenchmarkError("duplicate prompt sources")
    if prompt is not _MISSING:
        if not isinstance(prompt, str):
            raise BenchmarkError("prompt must be a string")
        return prompt
    if repeated is not _MISSING:
        if not isinstance(repeated, Mapping):
            raise BenchmarkError("prompt_repeat must be an object")
        count = repeated.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise BenchmarkError("prompt_repeat.count must be a positive integer")
        parts = [repeated.get(key, "") for key in ("text", "separator", "suffix")]
        if not all(isinstance(value, str) for value in parts):
            raise BenchmarkError("prompt_repeat text/separator/suffix must be strings")
        return parts[1].join([parts[0]] * count) + parts[2]
    if file is not _MISSING:
        if not isinstance(file, str) or not file:
            raise BenchmarkError("prompt_file must be a path string")
        path = _asset(file, root)
        value = path.read_text(encoding="utf-8").strip()
        if path.suffix == ".json":
            parsed = json.loads(value)
            value = parsed.get("prompt") if isinstance(parsed, Mapping) else None
        if not isinstance(value, str):
            raise BenchmarkError("prompt file must contain a string prompt")
        return value
    raise BenchmarkError("testcase requires a prompt")


def _text_source_request(case: Mapping[str, Any], root: Path) -> dict[str, Any]:
    token_ids = _explicit(case, "token_ids")
    if token_ids is _MISSING:
        return {"prompt": _semantic_prompt(case, root)}
    if any(_explicit(case, name) is not _MISSING for name in (
        "prompt", "test_prompt", "source_text", "prompt_repeat", "prompt_file",
    )):
        raise BenchmarkError("text source requires exactly one text or token_ids input")
    if not isinstance(token_ids, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or not -(1 << 31) <= value < (1 << 31)
        for value in token_ids
    ):
        raise BenchmarkError("token_ids must be an int32 array")
    return {"token_ids": list(token_ids)}


def _copy_explicit(request: dict[str, Any], case: Mapping[str, Any], *names: str) -> None:
    for name in names:
        value = _explicit(case, name)
        if value is not _MISSING:
            request[name] = value


def _latent_request(task: str, case: Mapping[str, Any], root: Path) -> dict[str, Any]:
    if task in _LATENT_TEXT:
        request = _text_controls(case, include_defaults=False)
        operands = (
            ("condition_latents_path", "condition_latents_raw"),
            ("condition_mask_path", "condition_mask_raw"),
            ("initial_latents_path", "initial_latents_raw"),
            ("sde_noises_path", "sde_noise_path", "sde_noise_raw"),
            ("sampling_steps_path", "sampling_steps_raw"),
        )
        for names in operands:
            if _explicit(case, *names) is not _MISSING:
                request[names[0]] = _semantic_asset(case, root, *names)
        conditioned = task == "latent_conditioned_text_generation"
        if conditioned:
            if not {"condition_latents_path", "condition_mask_path"} <= request.keys():
                raise BenchmarkError("conditioned generation requires condition latents and mask")
        elif "condition_latents_path" in request or "condition_mask_path" in request:
            raise BenchmarkError("raw condition inputs require the conditioned latent Task")
        elif not {"initial_latents_path", "sde_noises_path"} & request.keys():
            raise BenchmarkError("latent replay requires initial latents or SDE noise")
        if any(_explicit(case, name) is not _MISSING for name in (
            "prompt", "test_prompt", "source_text", "prompt_file", "prompt_repeat",
        )):
            request["prompt"] = _semantic_prompt(case, root)
        _copy_explicit(request, case, "sampling_steps")
        return request
    request: dict[str, Any] = {}
    packed = any(_explicit(case, *names) is not _MISSING for names in (
        ("branch_path", "branch"), ("trunk_path", "trunk"),
    ))
    if packed:
        for names in (("branch_path", "branch"), ("trunk_path", "trunk")):
            request[names[0]] = _semantic_asset(case, root, *names)
        if any(_explicit(case, name) is not _MISSING for name in (
            "latents_path", "shape", "timestep", "self_condition_path",
        )):
            raise BenchmarkError("native-packed branch/trunk cannot be combined with logical latent inputs")
    else:
        request["latents_path"] = _semantic_asset(case, root, "latents_path")
        for name in ("shape", "timestep"):
            value = _explicit(case, name)
            if value is _MISSING:
                raise BenchmarkError(f"logical latent input requires {name}")
            request[name] = deepcopy(value)
        if _explicit(case, "self_condition_path") is not _MISSING:
            request["self_condition_path"] = _semantic_asset(case, root, "self_condition_path")
    _copy_explicit(request, case, "self_cond_cfg_scale")
    return request


def _crop_batches(items: Any, root: Path) -> list[dict[str, Any]]:
    if not isinstance(items, list) or not items:
        raise BenchmarkError("pose provider requires nonempty preprocessed crop_batches")
    prepared = []
    for item in items:
        if not isinstance(item, Mapping):
            raise BenchmarkError("crop_batches entries must be objects")
        result = {}
        for name in ("stage", "iteration", "shape"):
            value = _explicit(item, name)
            if value is _MISSING:
                raise BenchmarkError(f"preprocessed crop batch requires {name}")
            result[name] = deepcopy(value)
        for name in ("query_poses_path", "rendered_path", "observed_path"):
            result[name] = _semantic_asset(item, root, name)
        prepared.append(result)
    return prepared


def _pose_input(case: Mapping[str, Any], root: Path) -> dict[str, Any]:
    result = {"candidate_poses_path": _semantic_asset(case, root, "candidate_poses_path")}
    for name in ("hypothesis_count", "mesh_diameter_meters"):
        value = _explicit(case, name)
        if value is _MISSING:
            raise BenchmarkError(f"pose input requires {name}")
        result[name] = value
    result["crop_batches"] = _crop_batches(_explicit(case, "crop_batches"), root)
    if "config" in case:
        result["config"] = deepcopy(case["config"])
    return result


def _request_count(case: Mapping[str, Any], count: int, request: dict[str, Any]) -> None:
    supplied = _explicit(case, "batch_size")
    if supplied is _MISSING:
        return
    if isinstance(supplied, bool) or not isinstance(supplied, int) or supplied != count:
        raise BenchmarkError(f"batch_size must equal actual request count {count}")
    request["batch_size"] = supplied


def _check_semantic_duplicates(request: Mapping[str, Any]) -> None:
    shared = set(request) - {"config", "item_configs", "seeds"}
    nested = request.get("config", {})
    if shared & nested.keys():
        raise BenchmarkError("duplicate flat/nested family Config")
    shared |= nested.keys()
    if "seeds" in request:
        if "seed" in shared:
            raise BenchmarkError("duplicate global seed and item seeds")
        shared.add("seed")
    for config in request.get("item_configs", []):
        if shared & config.keys():
            raise BenchmarkError("duplicate shared/item family Config")


def _semantic_remaining_request(task: str, case: Mapping[str, Any], root: Path) -> dict[str, Any]:
    inputs = _inputs(case)
    if "config" in inputs:
        raise BenchmarkError("family Config belongs in testcase.config, not inputs.config")
    generation = task in {
        "text_to_image", "images_text_to_image_edit", "batch_text_to_image",
        "text_to_video", "image_text_action_to_video",
    }
    if generation:
        return _semantic_media_request(task, case, root)
    request: dict[str, Any] = {}
    if task == "text_to_head_scores":
        if _explicit(case, "role") is not _MISSING:
            raise BenchmarkError("head scores do not accept an embedding role")
        return _text_source_request(case, root)
    if task in {"text_to_pooled_features", "text_to_token_features", "text_to_embedding"}:
        if task == "text_to_embedding":
            if _explicit(case, "token_ids") is not _MISSING:
                raise BenchmarkError("text_to_embedding requires UTF-8 text, not token_ids")
            request["prompt"] = _semantic_prompt(case, root)
        else:
            request.update(_text_source_request(case, root))
        role = _explicit(case, "role")
        if role is not _MISSING:
            if task != "text_to_embedding" or role not in ("default", "query", "document"):
                raise BenchmarkError("embedding role must be default, query, or document on text_to_embedding")
            request["role"] = role
    elif task == "text_query_documents_to_relevance":
        query = _explicit(case, "query", "prompt")
        documents = _explicit(case, "documents")
        if not isinstance(query, str):
            raise BenchmarkError("rerank query must be a string")
        if not isinstance(documents, list) or not all(isinstance(item, str) for item in documents):
            raise BenchmarkError("rerank documents must be a list of strings")
        request.update(query=query, documents=list(documents))
    elif task == "stereo_images_to_disparity":
        request["left_image_path"] = _semantic_asset(case, root, "left_image_path", "left_image")
        request["right_image_path"] = _semantic_asset(case, root, "right_image_path", "right_image")
    else:
        request["image_path"] = _semantic_asset(case, root, "image_path", "image", "test_image")
        if task == "image_state_to_action_chunk":
            request["state_path"] = _semantic_asset(case, root, "state_path", "state")
        elif task in {"image_points_to_masks", "image_text_to_instance_masks"}:
            prompt = _explicit(case, "prompt", "test_prompt", "source_text", "prompt_file", "prompt_repeat")
            if task == "image_points_to_masks":
                if prompt is not _MISSING:
                    raise BenchmarkError("point and text prompts are mutually exclusive")
                for name in ("point_x", "point_y", "is_foreground"):
                    value = _explicit(case, name)
                    if value is _MISSING:
                        continue
                    if name == "is_foreground":
                        if not isinstance(value, bool):
                            raise BenchmarkError("is_foreground must be a boolean")
                    elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise BenchmarkError(f"{name} must be a finite numeric fraction")
                    request[name] = value
            else:
                if any(_explicit(case, name) is not _MISSING for name in ("point_x", "point_y", "is_foreground")):
                    raise BenchmarkError("point and text prompts are mutually exclusive")
                request["prompt"] = _semantic_prompt(case, root)
    _request_count(case, 1, request)
    for name in ("seeds", "batch_seeds", "item_configs", "batch_prompts"):
        if _explicit(case, name) is not _MISSING:
            raise BenchmarkError(f"{name} is not a scalar Task input")
    return request


def _semantic_media_request(task: str, case: Mapping[str, Any], root: Path) -> dict[str, Any]:
    batch = task == "batch_text_to_image"
    if batch:
        if any(_explicit(case, name) is not _MISSING for name in (
            "test_prompt", "source_text", "prompt_file", "prompt_repeat"
        )):
            raise BenchmarkError("batch prompts cannot also supply scalar prompt sources")
        prompts = _explicit(case, "batch_prompts", "prompt")
        if not isinstance(prompts, list) or not prompts or not all(isinstance(value, str) for value in prompts):
            raise BenchmarkError("batch prompts must be a non-empty list of strings")
        request: dict[str, Any] = {"prompt": list(prompts)}
        count = len(prompts)
    else:
        if _explicit(case, "batch_prompts") is not _MISSING:
            raise BenchmarkError("batch_prompts require a batch Task")
        request = {"prompt": _semantic_prompt(case, root)}
        count = 1
    _request_count(case, count, request)
    media = "video" if task in {"text_to_video", "image_text_action_to_video"} else "image"
    supplied_media = _explicit(case, "media_type")
    if supplied_media is not _MISSING and supplied_media != media:
        raise BenchmarkError(f"{task} requires media_type={media}")
    request["media_type"] = media
    _copy_explicit(request, case, "seed", "negative_prompt", "height", "width", "guidance_scale", "cfg_scale", "num_frames")
    steps = _explicit(case, "num_steps", "num_inference_steps", "num_sampling_steps")
    if steps is not _MISSING:
        request["num_steps"] = steps
    single_image = _explicit(case, "image_path", "image", "test_image")
    images = _explicit(case, "image_paths", "images")
    if single_image is not _MISSING and images is not _MISSING:
        raise BenchmarkError("duplicate image_path and image_paths inputs")
    if task == "images_text_to_image_edit":
        if images is not _MISSING:
            if not isinstance(images, list) or not images or not all(isinstance(value, str) and value for value in images):
                raise BenchmarkError("image_paths must be a non-empty ordered list of paths")
            request["image_paths"] = [str(_asset(value, root)) for value in images]
        else:
            request["image_path"] = _semantic_asset(case, root, "image_path", "image", "test_image")
    elif task == "image_text_action_to_video":
        if images is not _MISSING:
            raise BenchmarkError("SANA input requires one image_path, not image_paths")
        request["image_path"] = _semantic_asset(case, root, "image_path", "image", "test_image")
        action = _explicit(case, "action")
        camera = _explicit(case, "camera_intrinsics")
        if not isinstance(action, str):
            raise BenchmarkError("SANA action must be a string")
        if not isinstance(camera, list) or not all(
            not isinstance(value, bool) and isinstance(value, (int, float)) and
            (not isinstance(value, float) or math.isfinite(value)) for value in camera
        ):
            raise BenchmarkError("camera_intrinsics must be a numeric array")
        request.update(action=action, camera_intrinsics=list(camera))
        _copy_explicit(request, case, "translation_speed", "rotation_speed_deg", "fps", "flow_shift", "no_action_overlay")
    elif single_image is not _MISSING or images is not _MISSING:
        raise BenchmarkError("image conditioning requires an image-input Task")
    if task != "image_text_action_to_video" and any(
        _explicit(case, name) is not _MISSING for name in ("action", "camera_intrinsics")
    ):
        raise BenchmarkError("action/camera inputs require image_text_action_to_video")
    replay = _explicit(case, "initial_latents_path")
    if replay is not _MISSING:
        if batch:
            raise BenchmarkError("batch replay requires an explicit per-item input protocol")
        request["initial_latents_path"] = _semantic_asset(case, root, "initial_latents_path")
    seeds = _explicit(case, "seeds", "batch_seeds")
    configs = _explicit(case, "item_configs")
    if not batch and (seeds is not _MISSING or configs is not _MISSING):
        raise BenchmarkError("seeds/item_configs require a batch Task")
    if seeds is not _MISSING:
        if not isinstance(seeds, list) or len(seeds) != count or any(
            isinstance(value, bool) or not isinstance(value, int) or not -(2**63) <= value < 2**63
            for value in seeds
        ):
            raise BenchmarkError("seeds must contain one signed 64-bit integer per prompt")
        request["seeds"] = list(seeds)
    if configs is not _MISSING:
        if not isinstance(configs, list) or len(configs) != count or not all(isinstance(value, Mapping) for value in configs):
            raise BenchmarkError("item_configs must contain one object per prompt")
        request["item_configs"] = [dict(value) for value in configs]
    return request


def _text_request(
    case: Mapping[str, Any], root: Path, *, include_defaults: bool = True
) -> dict[str, Any]:
    return {"prompt": _prompt(case, root), **_text_controls(case, include_defaults=include_defaults)}


def _text_controls(case: Mapping[str, Any], *, include_defaults: bool) -> dict[str, Any]:
    inputs = _inputs(case)
    request: dict[str, Any] = {}
    controls = (
        ("max_new_tokens", 128, int), ("temperature", 1.0, float), ("top_k", 1, int),
        ("top_p", 1.0, float), ("min_p", 0.0, float), ("seed", -1, int),
        ("repetition_penalty", 1.0, float), ("use_chat_template", False, bool),
        ("enable_thinking", True, bool),
    )
    for name, default, convert in controls:
        source = case if name in case else inputs if name == "temperature" else {}
        if name in source:
            value = source[name]
        elif include_defaults:
            value = default
        else:
            continue
        # Existing paths keep their workload defaults and coercions. Semantic
        # paths preserve explicit types and use family defaults for absent keys.
        request[name] = convert(value) if include_defaults else value
    if "generation_mode" in inputs:
        value = inputs["generation_mode"]
        request["text_generation_mode"] = str(value) if include_defaults else value
    if "block_length" in inputs:
        value = inputs["block_length"]
        request["block_length"] = int(value) if include_defaults else value
    if "threshold" in inputs:
        value = inputs["threshold"]
        request["confidence_threshold"] = float(value) if include_defaults else value
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
    for name, convert in (
        ("translation_speed", float),
        ("rotation_speed_deg", float),
        ("fps", int),
        ("flow_shift", float),
        ("no_action_overlay", bool),
    ):
        if name in case:
            request[name] = convert(case[name])
    return request


def _semantic_audio_request(task: str, case: Mapping[str, Any], root: Path) -> dict[str, Any]:
    inputs = _inputs(case)
    request = (
        {"prompt": _prompt(case, root)}
        if task in {"text_to_audio", "text_to_speech", "streaming_text_to_speech"}
        else {"audio_path": _audio_path(case, root)}
    )
    # Explicit controls retain their type; absent controls remain family-owned.
    # Preserve irrelevant explicit inputs too, so native validation rejects them.
    for name in (
        "max_new_tokens", "talker_max_new_tokens", "seed", "speaker", "tail_frames",
        "language", "target_language", "streaming", "chunk_ms",
    ):
        source = case if name in case else inputs
        if name in source:
            request[name] = source[name]
    if task == "speech_to_speech_response" and "speech_test_max_frames" in case:
        if "max_new_tokens" in request:
            raise BenchmarkError("speech_test_max_frames and max_new_tokens are duplicate limits")
        request["max_new_tokens"] = case["speech_test_max_frames"]
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
