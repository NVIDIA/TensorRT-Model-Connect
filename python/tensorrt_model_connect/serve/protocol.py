# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translation helpers between API envelopes and the native worker protocol."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from .errors import (
    WorkerCrashedError,
    WorkerError,
    WorkerProtocolError,
    WorkerRemoteError,
    WorkerRequestTooLargeError,
    WorkerSaturatedError,
    WorkerTimeoutError,
)


def invalid_request_message(error: WorkerRemoteError) -> str | None:
    """Return the native message only for client-caused worker failures."""

    details = error.details
    if not isinstance(details, Mapping) or details.get("type") != "invalid_request_error":
        return None
    message = details.get("message")
    if isinstance(message, str) and message:
        return message
    return "The model worker rejected the request"


def public_worker_error_message(error: WorkerError) -> str:
    """Return a stable public message without worker diagnostics."""

    if isinstance(error, WorkerTimeoutError):
        return "The model worker timed out"
    if isinstance(error, WorkerCrashedError):
        return "The model worker is unavailable"
    if isinstance(error, WorkerProtocolError):
        return "The model worker returned an invalid response"
    if isinstance(error, WorkerRequestTooLargeError):
        return "The request exceeds the model worker transport limit"
    if isinstance(error, WorkerSaturatedError):
        return "All model worker replicas are busy"
    return "The model worker operation failed"


def extract_text(result: Any, *, operation: str) -> str:
    """Extract text from the private v2 worker result."""

    if isinstance(result, Mapping) and isinstance(result.get("text"), str):
        return str(result["text"])
    raise WorkerProtocolError(f"worker {operation!r} result did not contain a string text field")


def extract_transcription_segments(result: Any) -> list[dict[str, float | str]]:
    """Copy only the public fields from native transcription segments."""

    if not isinstance(result, Mapping):
        raise WorkerProtocolError("worker transcription result was not a JSON object")
    raw_segments = result.get("segments", [])
    if not isinstance(raw_segments, list):
        raise WorkerProtocolError("worker transcription segments were not a JSON array")

    segments: list[dict[str, float | str]] = []
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, Mapping):
            raise WorkerProtocolError(f"worker transcription segment {index} was not an object")
        start = raw_segment.get("start_seconds")
        end = raw_segment.get("end_seconds")
        text = raw_segment.get("text")
        if (
            isinstance(start, bool)
            or not isinstance(start, (int, float))
            or not math.isfinite(start)
            or isinstance(end, bool)
            or not isinstance(end, (int, float))
            or not math.isfinite(end)
            or not isinstance(text, str)
        ):
            raise WorkerProtocolError(
                f"worker transcription segment {index} has invalid public fields"
            )
        segments.append(
            {
                "start_seconds": float(start),
                "end_seconds": float(end),
                "text": text,
            }
        )
    return segments


def extract_usage(result: Any) -> dict[str, int]:
    if not isinstance(result, Mapping):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    usage = result.get("usage")
    source = usage if isinstance(usage, Mapping) else result
    prompt = _non_negative_int(source.get("prompt_tokens"))
    completion = _non_negative_int(source.get("completion_tokens", source.get("generated_tokens")))
    total = _non_negative_int(source.get("total_tokens")) or prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def prepare_chat_prompt(
    messages: list[Mapping[str, Any]],
) -> tuple[str, bool, str]:
    """Choose the only safe native chat-template path or an explicit fallback.

    The native API's chat-template flag accepts one user string, not a message
    array. Applying it to a flattened conversation would incorrectly wrap all
    roles as one user message. Only a single, text-only user message takes that
    path; every other conversation carries explicit role markers with native
    templating disabled.
    """

    if (
        len(messages) == 1
        and str(messages[0].get("role", "")).lower() == "user"
        and _is_text_only(messages[0].get("content"))
    ):
        return _render_content(messages[0].get("content")), True, "single_user_template"

    rendered: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user").strip().lower()
        rendered.append(f"--- trtmc-role:{role} ---\n{_render_content(message.get('content'))}")
    rendered.append("--- trtmc-role:assistant ---\n")
    return "\n".join(rendered), False, "role_annotated_flattened"


def _is_text_only(content: Any) -> bool:
    if isinstance(content, str):
        return True
    if not isinstance(content, list):
        return False
    return all(
        isinstance(part, str)
        or (
            isinstance(part, Mapping)
            and part.get("type", "text") in {"text", "input_text"}
            and isinstance(part.get("text"), str)
        )
        for part in content
    )


def _render_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(str(part["text"]))
            else:
                parts.append(json.dumps(part, ensure_ascii=False, separators=(",", ":")))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)) and value >= 0:
        return int(value)
    return 0
