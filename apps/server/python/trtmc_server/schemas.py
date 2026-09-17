# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible request shapes supported by the text MVP."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextContentPart(StrictRequest):
    type: Literal["text"]
    text: str


class ChatMessage(StrictRequest):
    role: str
    content: str | list[TextContentPart]


class StreamOptions(StrictRequest):
    include_usage: bool = False


class GenerationRequest(StrictRequest):
    model: str
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)
    min_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    seed: int | None = Field(default=None, ge=0)
    enable_thinking: bool | None = None
    n: int = Field(default=1, ge=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: Any | None = None


class CompletionRequest(GenerationRequest):
    prompt: str


class ChatCompletionRequest(GenerationRequest):
    messages: list[ChatMessage] = Field(min_length=1, max_length=2)
    max_completion_tokens: int | None = Field(default=None, ge=1)
