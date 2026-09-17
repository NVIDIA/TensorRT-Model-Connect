# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate public text requests into the private native-worker protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import WorkerProtocolError, WorkerRemoteError, WorkerRequestTooLargeError
from .schemas import ChatCompletionRequest, GenerationRequest, TextContentPart


def generation_config(request: GenerationRequest, max_tokens: int) -> dict[str, Any]:
    result: dict[str, Any] = {"max_new_tokens": max_tokens}
    for name in ("temperature", "top_p", "min_p", "top_k", "seed", "enable_thinking"):
        value = getattr(request, name)
        if value is not None:
            result[name] = value
    return result


def chat_prompt(request: ChatCompletionRequest) -> tuple[str, str]:
    messages = request.messages
    if len(messages) == 1 and messages[0].role == "user":
        return _message_text(messages[0].content), ""
    if (
        len(messages) == 2
        and messages[0].role == "system"
        and messages[1].role == "user"
    ):
        return _message_text(messages[1].content), _message_text(messages[0].content)
    raise ValueError("messages must be one user message with an optional preceding system message")


def _message_text(content: str | list[TextContentPart]) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.text for part in content)


def extract_result(result: Any) -> tuple[str, int, dict[str, float]]:
    if not isinstance(result, Mapping) or not isinstance(result.get("text"), str):
        raise WorkerProtocolError("worker generate result does not contain text")
    tokens = result.get("completion_tokens", 0)
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        raise WorkerProtocolError("worker generate result has invalid completion_tokens")
    timings: dict[str, float] = {}
    for name in ("setup_ms", "prefill_ms", "decode_ms"):
        value = result.get(name, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise WorkerProtocolError(f"worker generate result has invalid {name}")
        timings[name] = float(value)
    return str(result["text"]), tokens, timings


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
