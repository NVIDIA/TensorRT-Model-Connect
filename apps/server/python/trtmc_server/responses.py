# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stateless Responses wire conversion; native Task execution stays in the server."""
from __future__ import annotations

import uuid

from .schemas import GenerationRequest, ResponsesRequest


def generation_input(request: ResponsesRequest):
    if isinstance(request.input, str):
        prompt = request.input
    else:
        if len(request.input) != 1:
            raise ValueError("Responses supports exactly one user input message")
        content = request.input[0].content
        prompt = content if isinstance(content, str) else "".join(part.text for part in content)
    if not prompt:
        raise ValueError("Responses requires nonempty text input")
    controls = request.model_dump(include=set(GenerationRequest.model_fields))
    return GenerationRequest(**controls, max_tokens=request.max_output_tokens), prompt


def response_body(*, text: str, model: str, created: int, max_tokens: int,
                  request: GenerationRequest, instructions: str):
    return {
        "id": f"resp_{uuid.uuid4().hex}", "object": "response", "created_at": created,
        "status": "completed", "error": None, "incomplete_details": None, "model": model,
        "output": [{"id": f"msg_{uuid.uuid4().hex}", "type": "message", "role": "assistant",
                    "status": "completed", "content": [{"type": "output_text", "text": text, "annotations": []}]}],
        "instructions": instructions or None, "max_output_tokens": max_tokens,
        "parallel_tool_calls": False, "tools": [], "tool_choice": "none",
        "temperature": request.temperature, "top_p": request.top_p,
        "text": {"format": {"type": "text"}}, "truncation": "disabled", "metadata": {},
        "previous_response_id": None, "store": False,
        # The public Task returns output counts but no native input count.
        # A complete Responses usage object would imply a measured input/total.
        "usage": None,
    }
