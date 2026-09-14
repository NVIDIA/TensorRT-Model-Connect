# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small request schemas used by the serving facade."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _OpenAIRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class ChatMessage(_OpenAIRequest):
    role: str
    content: Any


class ChatCompletionRequest(_OpenAIRequest):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)
    min_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    seed: int | None = None
    enable_thinking: bool | None = None
    stop: str | list[str] | None = None
    stream: bool = False


def model_to_dict(model: BaseModel, *, exclude_none: bool = False) -> dict[str, Any]:
    return dict(model.model_dump(exclude_none=exclude_none))
