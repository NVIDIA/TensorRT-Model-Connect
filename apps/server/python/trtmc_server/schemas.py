# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict request shapes for the supported serving endpoints."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextContentPart(StrictRequest):
    type: Literal["text"]
    text: str


class ImageURL(StrictRequest):
    url: str = Field(min_length=1)


class ImageContentPart(StrictRequest):
    type: Literal["image_url"]
    image_url: ImageURL


class ChatMessage(StrictRequest):
    role: str
    content: str | list[TextContentPart | ImageContentPart]


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
    messages: list[ChatMessage] = Field(min_length=1, max_length=64)
    max_completion_tokens: int | None = Field(default=None, ge=1)


class ResponseInputText(StrictRequest):
    type: Literal["input_text"]
    text: str


class ResponseInputMessage(StrictRequest):
    type: Literal["message"] = "message"
    role: Literal["user"]
    content: str | list[ResponseInputText] = Field(min_length=1)


class ResponsesRequest(StrictRequest):
    """The supported stateless, nonstreaming, single-user text subset."""
    model: str
    input: str | list[ResponseInputMessage] = Field(min_length=1)
    instructions: str | None = None
    max_output_tokens: int | None = Field(default=None, ge=16)
    temperature: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    top_p: float | None = Field(default=None, ge=0, le=1)
    min_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    seed: int | None = Field(default=None, ge=0)
    enable_thinking: bool | None = None
    stream: Literal[False] = False
    store: Literal[False] = False


class EmbeddingRequest(StrictRequest):
    model: str
    input: str | list[str]
    encoding_format: Literal["float"] = "float"
    user: str | None = None


class RankingText(StrictRequest):
    text: str = Field(min_length=1)


class RankingRequest(StrictRequest):
    model: str
    query: RankingText
    passages: list[RankingText] = Field(min_length=1)
    truncate: Literal["NONE"] = "NONE"


class DetectionImage(StrictRequest):
    type: Literal["image_url"]
    url: str = Field(min_length=1)


class DetectionRequest(StrictRequest):
    model: str | None = Field(default=None, min_length=1)
    input: list[DetectionImage] = Field(min_length=1, max_length=1)


class TranscriptionRequest(StrictRequest):
    model: str = Field(min_length=1)
    response_format: Literal["json", "text"] = "json"
    max_output_tokens: int | None = Field(default=None, ge=1)


class ImageGenerationRequest(StrictRequest):
    model: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    n: Literal[1] = 1
    stream: Literal[False] = False
    response_format: Literal["b64_json"] = "b64_json"
    size: str | None = None
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1, strict=True)
    num_inference_steps: int | None = Field(default=None, ge=1, strict=True)
    guidance_scale: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    cfg_scale: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    negative_prompt: str | None = None
    # Optional replay noise in the existing family's raw float32 layout; never a client path.
    initial_latents_b64: str | None = None

    @classmethod
    def from_form(cls, fields: dict[str, Any]) -> ImageGenerationRequest:
        """Multipart uses strings; retain the JSON route's strict numeric contract."""
        values = dict(fields)
        for key in ("seed", "num_inference_steps", "n"):
            if key in values:
                if not isinstance(values[key], str) or not re.fullmatch(r"[0-9]+", values[key]):
                    raise ValueError(f"{key} must be an integer")
                values[key] = int(values[key])
        if "stream" in values:
            if values["stream"] not in {"false", "true"}:
                raise ValueError("stream must be false")
            values["stream"] = values["stream"] == "true"
        return cls.model_validate(values)
