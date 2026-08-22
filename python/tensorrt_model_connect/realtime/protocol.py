# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation for the deliberately small TRTMC Realtime protocol surface."""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from typing import Any


PCM16_SAMPLE_RATE = 24_000
DEFAULT_MAX_MESSAGE_BYTES = 1 << 20
# Match the native worker's 100 ms PCM16 chunk limit at 24 kHz mono.
DEFAULT_MAX_AUDIO_BYTES = 4_800
MAX_TOOL_OUTPUT_BYTES = 256 * 1024
MAX_AUDIO_END_MS = ((1 << 63) - 1) // (PCM16_SAMPLE_RATE // 1000)

SUPPORTED_CLIENT_EVENTS = frozenset(
    {
        "session.update",
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
        "input_audio_buffer.clear",
        "response.create",
        "response.cancel",
        "conversation.item.truncate",
        "conversation.item.create",
        "session.close",
    }
)

JsonObject = dict[str, Any]


class RealtimeProtocolError(ValueError):
    """A client-visible validation error with Realtime error metadata."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_request",
        param: str | None = None,
        event_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.param = param
        self.event_id = event_id


def parse_client_message(
    message: str | bytes,
    *,
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    max_audio_bytes: int = DEFAULT_MAX_AUDIO_BYTES,
) -> JsonObject:
    """Parse and normalize one supported client event.

    Binary WebSocket frames are intentionally rejected. Audio belongs in the
    Base64 ``audio`` member of a JSON text event, matching the Realtime
    WebSocket transport.
    """

    if max_message_bytes <= 0 or max_audio_bytes <= 0:
        raise ValueError("Realtime message and audio bounds must be positive")
    if isinstance(message, bytes):
        raise RealtimeProtocolError(
            "binary WebSocket messages are not supported; send a JSON text event",
            code="invalid_event",
        )
    if not isinstance(message, str):
        raise RealtimeProtocolError("client event must be a JSON text message")
    if len(message.encode("utf-8")) > max_message_bytes:
        raise RealtimeProtocolError(
            "client event exceeds the configured message bound",
            code="message_too_large",
        )
    try:
        value = json.loads(message)
    except json.JSONDecodeError as exc:
        raise RealtimeProtocolError("client event is not valid JSON") from exc
    return validate_client_event(value, max_audio_bytes=max_audio_bytes)


def validate_client_event(
    value: object,
    *,
    max_audio_bytes: int = DEFAULT_MAX_AUDIO_BYTES,
) -> JsonObject:
    """Validate one decoded event against the supported official subset."""

    event = _object(value, "event")
    event_id = _optional_string(event, "event_id", "event.event_id", allow_empty=False)
    if event_id is not None and (
        len(event_id) > 128 or any(not 0x20 <= ord(character) <= 0x7E for character in event_id)
    ):
        raise RealtimeProtocolError(
            "event_id must be a short printable ASCII string",
            code="invalid_event_id",
            param="event_id",
        )
    event_type = _required_string(event, "type", "event.type")

    if event_type not in SUPPORTED_CLIENT_EVENTS:
        raise RealtimeProtocolError(
            f"event type {event_type!r} is not supported by this server",
            code="unsupported_event",
            param="type",
            event_id=event_id,
        )

    try:
        if event_type == "session.update":
            normalized = _validate_session_update(event)
        elif event_type == "input_audio_buffer.append":
            normalized = _validate_audio_append(event, max_audio_bytes=max_audio_bytes)
        elif event_type == "input_audio_buffer.commit":
            normalized = _validate_empty_control(event, event_type)
        elif event_type == "input_audio_buffer.clear":
            normalized = _validate_empty_control(event, event_type)
        elif event_type == "response.create":
            normalized = _validate_response_create(event)
        elif event_type == "response.cancel":
            normalized = _validate_response_cancel(event)
        elif event_type == "conversation.item.truncate":
            normalized = _validate_conversation_truncate(event)
        elif event_type == "conversation.item.create":
            normalized = _validate_function_output(event)
        else:
            _only_keys(event, {"type", "event_id"}, "session.close")
            normalized = {"type": "session.close"}
    except RealtimeProtocolError as exc:
        if exc.event_id is None:
            exc.event_id = event_id
        raise

    if event_id is not None:
        normalized["event_id"] = event_id
    return normalized


def protocol_error_event(error: RealtimeProtocolError, event_id: str) -> JsonObject:
    """Render a protocol exception as an official ``error`` server event."""

    cause: JsonObject = {
        "type": "invalid_request_error",
        "code": error.code,
        "message": str(error),
        "param": error.param,
        "event_id": error.event_id,
    }
    return {"type": "error", "event_id": event_id, "error": cause}


def session_snapshot(
    session_id: str,
    *,
    instructions: str = "",
    tools: list[JsonObject] | None = None,
    tool_choice: str = "auto",
) -> JsonObject:
    """Return the stable public session shape advertised by this host."""

    return {
        "id": session_id,
        "object": "realtime.session",
        "type": "realtime",
        "output_modalities": ["audio"],
        "instructions": instructions,
        "tools": [] if tools is None else tools,
        "tool_choice": tool_choice,
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": PCM16_SAMPLE_RATE},
                "turn_detection": {
                    "type": "server_vad",
                    "create_response": True,
                    "interrupt_response": True,
                },
            },
            "output": {"format": {"type": "audio/pcm", "rate": PCM16_SAMPLE_RATE}},
        },
    }


def _validate_session_update(event: Mapping[str, Any]) -> JsonObject:
    _only_keys(event, {"type", "event_id", "session"}, "session.update")
    session = _object(event.get("session"), "session.update.session")
    _only_keys(
        session,
        {"type", "instructions", "tools", "tool_choice", "output_modalities", "audio"},
        "session.update.session",
    )

    normalized: JsonObject = {"type": "session.update"}
    if "type" in session:
        session_type = _required_string(session, "type", "session.update.session.type")
        if session_type != "realtime":
            raise _invalid("session type must be 'realtime'", "session.type")

    if "instructions" in session:
        normalized["instructions"] = _required_string(
            session, "instructions", "session.update.session.instructions", allow_empty=True
        )
    if "output_modalities" in session:
        if session["output_modalities"] != ["audio"]:
            raise _unsupported("only audio responses are supported", "session.output_modalities")
        normalized["output_modalities"] = ["audio"]
    if "audio" in session:
        _validate_audio_config(session["audio"])
        normalized["input_sample_rate"] = PCM16_SAMPLE_RATE
        normalized["output_sample_rate"] = PCM16_SAMPLE_RATE
    if "tools" in session:
        normalized["tools"] = _validate_tools(session["tools"])
    if "tool_choice" in session:
        tool_choice = _required_string(session, "tool_choice", "session.update.session.tool_choice")
        if tool_choice not in {"auto", "none"}:
            raise _unsupported(
                "only 'auto' and 'none' tool choices are supported", "session.tool_choice"
            )
        normalized["tool_choice"] = tool_choice
    return normalized


def _validate_audio_config(value: object) -> None:
    audio = _object(value, "session.update.session.audio")
    _only_keys(audio, {"input", "output"}, "session.update.session.audio")
    if "input" in audio:
        input_config = _object(audio["input"], "session.update.session.audio.input")
        _only_keys(
            input_config,
            {"format", "turn_detection"},
            "session.update.session.audio.input",
        )
        if "format" in input_config:
            _validate_pcm_format(
                input_config["format"], "session.audio.input.format", require_rate=True
            )
        if "turn_detection" in input_config:
            turn = _object(
                input_config["turn_detection"], "session.update.session.audio.input.turn_detection"
            )
            _only_keys(
                turn,
                {"type", "create_response", "interrupt_response"},
                "session.update.session.audio.input.turn_detection",
            )
            if turn.get("type") != "server_vad":
                raise _unsupported(
                    "only model-owned server VAD is supported", "session.audio.input.turn_detection"
                )
            if turn.get("create_response", True) is not True:
                raise _unsupported(
                    "create_response must remain enabled for model-owned server VAD",
                    "session.audio.input.turn_detection.create_response",
                )
            if turn.get("interrupt_response", True) is not True:
                raise _unsupported(
                    "interrupt_response must remain enabled for this native session",
                    "session.audio.input.turn_detection.interrupt_response",
                )
    if "output" in audio:
        output_config = _object(audio["output"], "session.update.session.audio.output")
        _only_keys(output_config, {"format"}, "session.update.session.audio.output")
        if "format" in output_config:
            _validate_pcm_format(
                output_config["format"], "session.audio.output.format", require_rate=False
            )


def _validate_pcm_format(value: object, where: str, *, require_rate: bool) -> None:
    audio_format = _object(value, where)
    _only_keys(audio_format, {"type", "rate"}, where)
    if audio_format.get("type") != "audio/pcm":
        raise _unsupported("only PCM audio is supported", f"{where}.type")
    rate = audio_format.get("rate")
    if require_rate and rate is None:
        raise _invalid("PCM input rate is required", f"{where}.rate")
    if rate is not None and (type(rate) is not int or rate != PCM16_SAMPLE_RATE):
        raise _unsupported(f"only {PCM16_SAMPLE_RATE} Hz PCM audio is supported", f"{where}.rate")


def _validate_tools(value: object) -> list[JsonObject]:
    if not isinstance(value, list):
        raise _invalid("session.tools must be an array", "session.tools")
    tools: list[JsonObject] = []
    names: set[str] = set()
    for index, raw_tool in enumerate(value):
        where = f"session.tools[{index}]"
        tool = _object(raw_tool, where)
        _only_keys(tool, {"type", "name", "description", "parameters"}, where)
        if tool.get("type") != "function":
            raise _unsupported("only function tools are supported", f"{where}.type")
        name = _required_string(tool, "name", f"{where}.name")
        if len(name) > 64 or any(
            not (character.isascii() and (character.isalnum() or character in "_-"))
            for character in name
        ):
            raise _unsupported(
                "function names must use 1-64 ASCII letters, digits, _ or -",
                f"{where}.name",
            )
        if name in names:
            raise _invalid("function tool names must be unique", f"{where}.name")
        names.add(name)
        normalized: JsonObject = {"type": "function", "name": name}
        if "description" in tool:
            description = _required_string(
                tool, "description", f"{where}.description", allow_empty=True
            )
            _require_ascii_text(description, f"{where}.description")
            normalized["description"] = description
        if "parameters" in tool:
            parameters = _object(tool["parameters"], f"{where}.parameters")
            if parameters.get("type") not in (None, "object"):
                raise _unsupported(
                    "function parameters must be an object schema", f"{where}.parameters.type"
                )
            _require_ascii_json(parameters, f"{where}.parameters")
            # JSON round-tripping produces a detached, data-only object and
            # rejects values a native JSONL worker could not consume.
            try:
                normalized["parameters"] = json.loads(json.dumps(parameters))
            except (TypeError, ValueError) as exc:
                raise _invalid("tool parameters must be JSON data", f"{where}.parameters") from exc
        tools.append(normalized)
    return tools


def _validate_audio_append(event: Mapping[str, Any], *, max_audio_bytes: int) -> JsonObject:
    _only_keys(event, {"type", "event_id", "audio"}, "input_audio_buffer.append")
    audio = _required_string(event, "audio", "input_audio_buffer.append.audio")
    try:
        decoded = base64.b64decode(audio, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise _invalid("audio must be canonical Base64-encoded PCM16LE", "audio") from exc
    if not decoded:
        raise _invalid("audio chunk must not be empty", "audio")
    if len(decoded) % 2 != 0:
        raise _invalid("PCM16LE audio must contain complete 16-bit samples", "audio")
    if len(decoded) > max_audio_bytes:
        raise RealtimeProtocolError(
            "audio chunk exceeds the configured input bound",
            code="audio_too_large",
            param="audio",
        )
    canonical = base64.b64encode(decoded).decode("ascii")
    if canonical != audio:
        raise _invalid("audio must use canonical Base64 encoding", "audio")
    return {"type": "input_audio_buffer.append", "audio": audio}


def _validate_function_output(event: Mapping[str, Any]) -> JsonObject:
    _only_keys(event, {"type", "event_id", "previous_item_id", "item"}, "conversation.item.create")
    if event.get("previous_item_id") not in (None, "root"):
        raise _unsupported("previous_item_id placement is not supported", "previous_item_id")
    item = _object(event.get("item"), "conversation.item.create.item")
    _only_keys(item, {"type", "call_id", "output"}, "conversation.item.create.item")
    if item.get("type") != "function_call_output":
        raise _unsupported(
            "only function_call_output conversation items are supported", "item.type"
        )
    output = _required_string(item, "output", "item.output", allow_empty=True)
    _require_ascii_text(output, "item.output")
    if len(output.encode("ascii")) > MAX_TOOL_OUTPUT_BYTES:
        raise _invalid("function output exceeds the configured bound", "item.output")
    return {
        "type": "conversation.item.create",
        "call_id": _required_string(item, "call_id", "item.call_id"),
        "output": output,
    }


def _validate_empty_control(event: Mapping[str, Any], event_type: str) -> JsonObject:
    _only_keys(event, {"type", "event_id"}, event_type)
    return {"type": event_type}


def _validate_response_cancel(event: Mapping[str, Any]) -> JsonObject:
    _only_keys(event, {"type", "event_id", "response_id"}, "response.cancel")
    normalized: JsonObject = {"type": "response.cancel"}
    if "response_id" in event:
        normalized["response_id"] = _required_string(
            event, "response_id", "response.cancel.response_id"
        )
    return normalized


def _validate_response_create(event: Mapping[str, Any]) -> JsonObject:
    _only_keys(event, {"type", "event_id", "response"}, "response.create")
    if "response" in event:
        response = _object(event["response"], "response.create.response")
        if response:
            raise _unsupported(
                "per-response overrides are not supported",
                "response",
            )
    return {"type": "response.create"}


def _validate_conversation_truncate(event: Mapping[str, Any]) -> JsonObject:
    _only_keys(
        event,
        {"type", "event_id", "item_id", "content_index", "audio_end_ms"},
        "conversation.item.truncate",
    )
    content_index = event.get("content_index")
    if type(content_index) is not int or content_index != 0:
        raise _invalid(
            "conversation.item.truncate.content_index must be 0",
            "content_index",
        )
    audio_end_ms = event.get("audio_end_ms")
    if type(audio_end_ms) is not int or audio_end_ms < 0 or audio_end_ms > MAX_AUDIO_END_MS:
        raise _invalid(
            "conversation.item.truncate.audio_end_ms is outside the supported range",
            "audio_end_ms",
        )
    return {
        "type": "conversation.item.truncate",
        "item_id": _required_string(event, "item_id", "conversation.item.truncate.item_id"),
        "content_index": 0,
        "audio_end_ms": audio_end_ms,
    }


def _object(value: object, where: str) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise _invalid(f"{where} must be an object", where)
    return value


def _required_string(
    value: Mapping[str, Any],
    key: str,
    where: str,
    *,
    allow_empty: bool = False,
) -> str:
    result = value.get(key)
    if not isinstance(result, str) or (not allow_empty and not result):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise _invalid(f"{where} must be {qualifier}", where)
    return result


def _optional_string(
    value: Mapping[str, Any],
    key: str,
    where: str,
    *,
    allow_empty: bool,
) -> str | None:
    if key not in value:
        return None
    return _required_string(value, key, where, allow_empty=allow_empty)


def _only_keys(value: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise _unsupported(
            f"{where} contains unsupported field {unknown[0]!r}", f"{where}.{unknown[0]}"
        )


def _require_ascii_text(value: str, where: str) -> None:
    if any(not (character in "\n\r\t" or 0x20 <= ord(character) <= 0x7E) for character in value):
        raise _unsupported(f"{where} must contain printable ASCII text", where)


def _require_ascii_json(value: object, where: str) -> None:
    if isinstance(value, str):
        _require_ascii_text(value, where)
    elif isinstance(value, list):
        for item in value:
            _require_ascii_json(item, where)
    elif isinstance(value, dict):
        for key, item in value.items():
            _require_ascii_text(key, where)
            _require_ascii_json(item, where)


def _invalid(message: str, param: str | None = None) -> RealtimeProtocolError:
    return RealtimeProtocolError(message, code="invalid_value", param=param)


def _unsupported(message: str, param: str | None = None) -> RealtimeProtocolError:
    return RealtimeProtocolError(message, code="unsupported_value", param=param)
