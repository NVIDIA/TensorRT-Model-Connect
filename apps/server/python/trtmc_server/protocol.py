# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate public text requests into the private native-worker protocol."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .errors import WorkerProtocolError, WorkerRemoteError, WorkerRequestTooLargeError
from .schemas import ChatCompletionRequest, GenerationRequest, TextContentPart, ImageContentPart


def generation_config(request: GenerationRequest, max_tokens: int) -> dict[str, Any]:
    result: dict[str, Any] = {"max_new_tokens": max_tokens}
    for name in ("temperature", "top_p", "min_p", "top_k", "seed", "enable_thinking"):
        value = getattr(request, name)
        if value is not None:
            result[name] = value
    return result


def chat_prompt(request: ChatCompletionRequest) -> tuple[str, str]:
    messages = request.messages
    if messages[-1].role != "user" or any(
        message.role not in {"user", "assistant"} and not (index == 0 and message.role == "system")
        for index, message in enumerate(messages)
    ):
        raise ValueError("messages require user/assistant roles, an optional leading system, and a final user")
    return _message_text(messages[-1].content), (
        _message_text(messages[0].content) if messages[0].role == "system" else "")


def chat_history(request: ChatCompletionRequest) -> list[dict[str, str]] | None:
    if len(request.messages) == 1 or (len(request.messages) == 2 and request.messages[0].role == "system"):
        return None
    return [{"role": message.role, "content": _message_text(message.content)} for message in request.messages]


def chat_image(request: ChatCompletionRequest) -> str | None:
    images = [part.image_url.url for message in request.messages if isinstance(message.content, list)
              for part in message.content if isinstance(part, ImageContentPart)]
    if not images:
        return None
    if len(images) != 1 or len(request.messages) != 1 or request.messages[0].role != "user":
        raise ValueError("image chat requires one user message with exactly one image")
    return images[0]


def _message_text(content: str | list[TextContentPart | ImageContentPart]) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.text for part in content if isinstance(part, TextContentPart))


def extract_vision_result(result: Any, height: int, width: int) -> tuple[str, int, dict[str, float]]:
    text, tokens, timings = extract_result(result)
    if (type(result.get("image_height")) is not int or
            type(result.get("image_width")) is not int or result["image_height"] != height or
            result["image_width"] != width):
        raise WorkerProtocolError("worker vision result has invalid text or image identity")
    extract_task_timing(result)
    return text, tokens, timings


def extract_result(result: Any) -> tuple[str, int, dict[str, float]]:
    if not isinstance(result, Mapping) or not isinstance(result.get("text"), str):
        raise WorkerProtocolError("worker generate result does not contain text")
    tokens = result.get("completion_tokens", 0)
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        raise WorkerProtocolError("worker generate result has invalid completion_tokens")
    timings: dict[str, float] = {}
    for name in ("setup_ms", "prefill_ms", "decode_ms", "model_call_ms"):
        if name == "model_call_ms" and name not in result:
            continue
        value = result.get(name, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise WorkerProtocolError(f"worker generate result has invalid {name}")
        timings[name] = float(value)
    return str(result["text"]), tokens, timings


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def extract_embedding(result: Any) -> tuple[list[float], dict[str, float]]:
    if not isinstance(result, Mapping):
        raise WorkerProtocolError("worker embedding result must be an object")
    values = result.get("values")
    if (not isinstance(values, list) or not values or
            type(result.get("dim")) is not int or result["dim"] != len(values) or
            any(not _finite_number(value) for value in values)):
        raise WorkerProtocolError("worker embedding result must contain a complete finite vector")
    return values, extract_task_timing(result)


def extract_ranking(result: Any, expected_count: int) -> tuple[list[float], dict[str, float]]:
    if not isinstance(result, Mapping):
        raise WorkerProtocolError("worker ranking result must be an object")
    scores = result.get("scores")
    if (not isinstance(scores, list) or len(scores) != expected_count or
            any(not _finite_number(value) for value in scores)):
        raise WorkerProtocolError("worker ranking result must contain one finite score per passage")
    return scores, extract_task_timing(result)


def extract_task_timing(result: Mapping[str, Any]) -> dict[str, float]:
    wall = result.get("model_call_ms")
    if (not _finite_number(wall) or wall < 0 or
            result.get("timing_scope") != "public_task_call_wall"):
        raise WorkerProtocolError("worker result has invalid public Task timing")
    return {"model_call_ms": float(wall)}


def extract_transcription(result: Any, sample_count: int, sample_rate: int) -> str:
    if (not isinstance(result, Mapping) or not isinstance(result.get("text"), str) or
            type(result.get("audio_sample_count")) is not int or
            type(result.get("audio_sample_rate")) is not int or
            result["audio_sample_count"] != sample_count or result["audio_sample_rate"] != sample_rate):
        raise WorkerProtocolError("worker transcription result has invalid text or audio identity")
    extract_task_timing(result)
    return result["text"]


def public_worker_error(error: Exception) -> str:
    if isinstance(error, WorkerRequestTooLargeError):
        return "request exceeds the native worker transport limit"
    if isinstance(error, WorkerRemoteError):
        details = error.details
        if isinstance(details, Mapping) and details.get("type") == "invalid_request_error":
            message = details.get("message")
            if isinstance(message, str) and message:
                return message
    return "The model worker is unavailable"


def extract_detection(result: Any, height: int, width: int) -> tuple[dict[str, list], dict[str, float]]:
    if (not isinstance(result, Mapping) or not isinstance(result.get("boxes"), list) or
            type(result.get("image_height")) is not int or type(result.get("image_width")) is not int or
            result["image_height"] != height or result["image_width"] != width):
        raise WorkerProtocolError("worker detection result has invalid image identity")
    grouped: dict[str, list] = {}
    for box in result["boxes"]:
        if (not isinstance(box, Mapping) or type(box.get("class_id")) is not int or box["class_id"] < 0 or
                any(not _finite_number(box.get(key)) for key in
                    ("x_min", "y_min", "x_max", "y_max", "confidence")) or
                box["x_min"] > box["x_max"] or
                box["y_min"] > box["y_max"] or not 0 <= box["confidence"] <= 1):
            raise WorkerProtocolError("worker detection result has invalid boxes")
        grouped.setdefault(str(box["class_id"]), []).append(
            {"x_min": box["x_min"] / width, "y_min": box["y_min"] / height,
             "x_max": box["x_max"] / width, "y_max": box["y_max"] / height,
             "confidence": box["confidence"]})
    return grouped, extract_task_timing(result)
