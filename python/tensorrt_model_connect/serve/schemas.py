# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small request schemas used by the serving facade."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - Pydantic 1 compatibility
    ConfigDict = None  # type: ignore[assignment,misc]


if hasattr(BaseModel, "model_fields"):

    class _OpenAIRequest(BaseModel):
        model_config = ConfigDict(extra="allow")  # type: ignore[misc,operator]

else:  # pragma: no cover - Pydantic 1 compatibility

    class _OpenAIRequest(BaseModel):
        class Config:
            extra = "allow"


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
    """Support both Pydantic 1 and 2 without importing version internals."""

    dump = getattr(model, "model_dump", None)
    if dump is not None:
        return dict(dump(exclude_none=exclude_none))
    return dict(model.dict(exclude_none=exclude_none))
