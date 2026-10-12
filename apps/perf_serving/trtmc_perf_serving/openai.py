# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map OpenAI/NIM-shaped bodies (aiperf standard endpoints) onto worker operation requests.

Each mapper starts from the serving profile's base request (the catalog testcase)
and overrides only fields the load generator varies. Fields that cannot be honored
identically by every backend are rejected rather than silently ignored.
"""

from __future__ import annotations

import base64
from typing import Any, Callable, Mapping

from .files import FILE_KEY

# Sent by aiperf/OpenAI clients but irrelevant to the operation request.
_TRANSPORT_FIELDS = {"model", "stream", "stream_options", "user", "n", "response_format", "encoding_format"}


class RequestError(ValueError):
    """The body cannot be mapped to the served operation (HTTP 400)."""


def _check_fields(body: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(body) - allowed - _TRANSPORT_FIELDS)
    if unknown:
        raise RequestError(f"unsupported fields: {', '.join(unknown)}")
    if int(body.get("n", 1)) != 1:
        raise RequestError("only n=1 is supported")


def _sampling(body: Mapping[str, Any], request: dict[str, Any]) -> None:
    for source, target in (("max_tokens", "max_new_tokens"), ("max_completion_tokens", "max_new_tokens"),
                           ("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"),
                           ("min_p", "min_p"), ("seed", "seed")):
        if body.get(source) is not None:
            request[target] = body[source]


_SAMPLING_FIELDS = {"max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k", "min_p", "seed",
                    "stop"}
# Renders a text-only conversation with the model's chat template: (messages, enable_thinking) -> prompt.
ChatRenderer = Callable[[list[dict[str, str]], bool], str]


def stop_sequences(body: Mapping[str, Any]) -> tuple[str, ...]:
    """OpenAI ``stop`` as a tuple. Applied by truncating the returned text, identically for all backends."""
    value = body.get("stop")
    if value is None:
        return ()
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not all(isinstance(item, str) and item for item in values):
        raise RequestError("stop must be a non-empty string or a list of them")
    return tuple(values)


def truncate_at_stop(text: str, stops: tuple[str, ...]) -> str:
    cut = min((index for index in (text.find(stop) for stop in stops) if index >= 0), default=len(text))
    return text[:cut]


def completion(body: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    _check_fields(body, {"prompt"} | _SAMPLING_FIELDS)
    if not isinstance(body.get("prompt"), str):
        raise RequestError("prompt must be one string")
    request = {**base, "prompt": body["prompt"], "use_chat_template": False}
    request.pop("image_path", None)
    _sampling(body, request)
    return request


def _parts(message: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    content = message.get("content")
    parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
    texts, images = [], []
    for part in parts or []:
        if part.get("type") == "text":
            texts.append(part.get("text", ""))
        elif part.get("type") == "image_url":
            images.append(_data_url_file(part["image_url"]["url"]))
        else:
            raise RequestError(f"unsupported content part: {part.get('type')!r}")
    return "".join(texts), images


def chat(body: Mapping[str, Any], base: Mapping[str, Any], renderer: ChatRenderer | None = None) -> dict[str, Any]:
    """Map chat messages onto a ``generate`` request.

    One user message (optionally with one image) uses the backend's own chat template.
    Longer text-only conversations (system prompts, few-shot turns, multi-turn history)
    are rendered once by ``renderer`` and sent as a plain prompt, so every backend
    receives the same token sequence.
    """
    _check_fields(body, {"messages", "enable_thinking"} | _SAMPLING_FIELDS)
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages or messages[-1].get("role") != "user":
        raise RequestError("messages must be a non-empty list ending with a user message")
    thinking = body.get("enable_thinking")
    request = {**base}
    request.pop("image_path", None)
    if len(messages) == 1:
        text, images = _parts(messages[0])
        if len(images) > 1:
            raise RequestError("at most one image is supported")
        request.update(prompt=text, use_chat_template=True)
        if images:
            request["image_path"] = images[0]
        if thinking is not None:
            request["enable_thinking"] = bool(thinking)
    else:
        if renderer is None:
            raise RequestError("multi-message chat requires a chat-template renderer for this model")
        conversation = []
        for message in messages:
            text, images = _parts(message)
            if images:
                raise RequestError("images are supported only in single-message chat")
            conversation.append({"role": str(message.get("role")), "content": text})
        request.update(prompt=renderer(conversation, bool(thinking)), use_chat_template=False)
    _sampling(body, request)
    return request


def embeddings(body: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    _check_fields(body, {"input", "dimensions"})
    value = body.get("input")
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str):
        raise RequestError("input must be one string")
    return {**base, "prompt": value}


def ranking(body: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    """NIM ``/v1/ranking``: {"query": {"text"}, "passages": [{"text"}]}."""
    _check_fields(body, {"query", "passages", "truncate"})
    try:
        query = body["query"]["text"]
        documents = [passage["text"] for passage in body["passages"]]
    except (KeyError, TypeError) as error:
        raise RequestError("ranking requires query.text and passages[].text") from error
    return {**base, "query": query, "documents": documents}


def image_generation(body: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    _check_fields(body, {"prompt", "size", "seed", "num_inference_steps", "guidance_scale", "negative_prompt",
                         "quality", "num_frames", "seconds", "fps"})
    request = {**base, "prompt": str(body.get("prompt", ""))}
    if body.get("size"):
        try:
            width, height = (int(value) for value in str(body["size"]).lower().split("x"))
        except ValueError as error:
            raise RequestError("size must be WIDTHxHEIGHT") from error
        request.update(width=width, height=height)
    for source, target in (("num_inference_steps", "num_steps"), ("seed", "seed"),
                           ("guidance_scale", "guidance_scale"), ("negative_prompt", "negative_prompt"),
                           ("num_frames", "num_frames")):
        if body.get(source) is not None:
            request[target] = _number(body[source]) if target != "negative_prompt" else body[source]
    return request


def transcription(audio: bytes, filename: str, fields: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    _check_fields(fields, {"language", "temperature", "prompt"})
    suffix = "." + filename.rsplit(".", 1)[-1] if "." in filename else ".wav"
    request = {**base, "audio_path": {FILE_KEY: {"suffix": suffix, "b64": base64.b64encode(audio).decode()}}}
    if fields.get("language"):
        request["language"] = fields["language"]
    return request


def _number(value: Any) -> Any:
    if isinstance(value, str):  # multipart/form fields arrive as strings
        return float(value) if "." in value else int(value)
    return value


def _data_url_file(url: str) -> dict[str, Any]:
    if not url.startswith("data:") or ";base64," not in url:
        raise RequestError("images must be base64 data URLs")
    header, data = url.split(",", 1)
    subtype = header[5:].split(";", 1)[0].split("/")[-1] or "png"
    return {FILE_KEY: {"suffix": "." + subtype.replace("jpeg", "jpg"), "b64": data}}
