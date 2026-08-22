# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real WebSocket-to-native-worker probe for the VoiceChat L4 E2E case."""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
import wave
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


_WIRE_SAMPLE_RATE = 24_000
_MAX_AUDIO_BYTES = 4_800
_TOKEN_ENV = "TRTMC_REALTIME_TOKEN"
_TOKEN = "voicechat-local-e2e"
_TOOL_NAME = "generate_random_number"
_TOOL_RESULT = '{"result":20}'

JsonObject = dict[str, Any]
EventPredicate = Callable[[JsonObject], bool]


class ProbeFailure(RuntimeError):
    """A bounded, public failure suitable for the JSON receipt."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_pcm16(path: Path) -> tuple[int, array[int]]:
    try:
        with wave.open(str(path), "rb") as source:
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getcomptype() != "NONE"
            ):
                raise ProbeFailure("invalid_source_audio", "source audio must be mono PCM16 WAV")
            sample_rate = source.getframerate()
            payload = source.readframes(source.getnframes())
    except (OSError, EOFError, wave.Error) as error:
        raise ProbeFailure("invalid_source_audio", "source audio could not be read") from error
    samples = array("h")
    samples.frombytes(payload)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        raise ProbeFailure("invalid_source_audio", "source audio must not be empty")
    return sample_rate, samples


def _resample_linear(samples: array[int], source_rate: int) -> bytes:
    if source_rate <= 0:
        raise ProbeFailure("invalid_source_audio", "source sample rate must be positive")
    output_count = round(len(samples) * _WIRE_SAMPLE_RATE / source_rate)
    output = array("h")
    for index in range(output_count):
        source = index * source_rate / _WIRE_SAMPLE_RATE
        left = min(int(source), len(samples) - 1)
        right = min(left + 1, len(samples) - 1)
        fraction = source - left
        value = round(samples[left] + fraction * (samples[right] - samples[left]))
        output.append(max(-32768, min(32767, value)))
    if sys.byteorder != "little":
        output.byteswap()
    return output.tobytes()


def _audio_stats(payload: bytes) -> JsonObject:
    if len(payload) % 2:
        raise ProbeFailure("invalid_server_audio", "server PCM16 audio is not sample aligned")
    values = array("h")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    count = len(values)
    square_sum = sum((value / 32768.0) ** 2 for value in values)
    peak = max((abs(value) / 32768.0 for value in values), default=0.0)
    return {
        "encoding": "pcm_s16le",
        "sample_rate": _WIRE_SAMPLE_RATE,
        "num_samples": count,
        "rms": math.sqrt(square_sum / count) if count else 0.0,
        "peak": peak,
        "sha256": _sha256(payload),
    }


def _write_wav(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(_WIRE_SAMPLE_RATE)
        output.writeframes(payload)


def _nested_string(value: object, *keys: str) -> str | None:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if isinstance(current, str) else None


def _event_summary(direction: str, event: JsonObject, ordinal: int) -> JsonObject:
    event_type = event.get("type")
    summary: JsonObject = {
        "ordinal": ordinal,
        "direction": direction,
        "type": event_type if isinstance(event_type, str) else "",
    }
    for key in ("event_id", "response_id", "item_id", "call_id"):
        value = event.get(key)
        if isinstance(value, str):
            summary[key] = value
    if "previous_item_id" in event:
        previous_item_id = event.get("previous_item_id")
        if previous_item_id is None or isinstance(previous_item_id, str):
            summary["previous_item_id"] = previous_item_id
    for key in ("content_index", "audio_end_ms"):
        value = event.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            summary[key] = value

    response_id = _nested_string(event, "response", "id")
    response_status = _nested_string(event, "response", "status")
    response_reason = _nested_string(event, "response", "status_details", "reason")
    if response_id is not None:
        summary["response_id"] = response_id
    if response_status is not None:
        summary["response_status"] = response_status
    if response_reason is not None:
        summary["response_reason"] = response_reason
    response = event.get("response")
    if isinstance(response, dict):
        if isinstance(response.get("object"), str):
            summary["response_object"] = response["object"]
        if isinstance(response.get("output"), list):
            summary["response_output_count"] = len(response["output"])
        summary["response_status_details_is_null"] = response.get("status_details") is None

    item_call_id = _nested_string(event, "item", "call_id")
    item_name = _nested_string(event, "item", "name")
    item_type = _nested_string(event, "item", "type")
    item_output = _nested_string(event, "item", "output")
    item_id = _nested_string(event, "item", "id")
    item_object = _nested_string(event, "item", "object")
    item_status = _nested_string(event, "item", "status")
    if item_call_id is not None:
        summary["call_id"] = item_call_id
    if item_name is not None:
        summary["name"] = item_name
    if item_type is not None:
        summary["item_type"] = item_type
    if item_output is not None:
        summary["output_sha256"] = _sha256(item_output.encode("utf-8"))
    if item_id is not None:
        summary["item_id"] = item_id
    if item_object is not None:
        summary["item_object"] = item_object
    if item_status is not None:
        summary["item_status"] = item_status
    name = event.get("name")
    if isinstance(name, str):
        summary["name"] = name
    arguments = event.get("arguments")
    if isinstance(arguments, str):
        summary["arguments"] = arguments[:4096]

    text = event.get("delta")
    if not isinstance(text, str):
        text = event.get("transcript")
    if isinstance(text, str) and summary["type"] != "response.output_audio.delta":
        summary["text"] = text[:4096]

    error_code = _nested_string(event, "error", "code")
    if error_code is not None:
        summary["error_code"] = error_code

    audio = event.get("audio") if direction == "client" else event.get("delta")
    audio_event = summary["type"] in {
        "input_audio_buffer.append",
        "response.output_audio.delta",
    }
    if audio_event and isinstance(audio, str):
        try:
            decoded = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ProbeFailure("invalid_audio_base64", "transport audio was not Base64") from error
        if len(decoded) % 2:
            raise ProbeFailure("invalid_audio_alignment", "transport audio was not PCM16 aligned")
        summary["audio_samples"] = len(decoded) // 2
        summary["audio_sha256"] = _sha256(decoded)
        if summary["type"] == "response.output_audio.delta":
            stats = _audio_stats(decoded)
            summary["audio_rms"] = stats["rms"]
            summary["audio_peak"] = stats["peak"]
    return summary


@dataclass
class ScenarioTrace:
    name: str
    timeline: list[JsonObject] = field(default_factory=list)
    output_audio: bytearray = field(default_factory=bytearray)
    clean_close: bool = False
    close_code: int | None = None

    def record(self, direction: str, event: JsonObject) -> None:
        self.timeline.append(_event_summary(direction, event, len(self.timeline)))
        if direction == "server" and event.get("type") == "response.output_audio.delta":
            delta = event.get("delta")
            if not isinstance(delta, str):
                raise ProbeFailure("invalid_server_audio", "server audio delta was missing")
            try:
                decoded = base64.b64decode(delta, validate=True)
            except (binascii.Error, ValueError) as error:
                raise ProbeFailure(
                    "invalid_server_audio", "server audio delta was not Base64"
                ) from error
            if not decoded or len(decoded) % 2:
                raise ProbeFailure("invalid_server_audio", "server audio delta was invalid PCM16")
            self.output_audio.extend(decoded)

    def receipt(self) -> JsonObject:
        return {
            "name": self.name,
            "clean_close": self.clean_close,
            "close_code": self.close_code,
            "timeline": self.timeline,
            "audio": _audio_stats(bytes(self.output_audio)),
        }


class LiveConnection:
    def __init__(self, websocket: Any, trace: ScenarioTrace, timeout_s: int) -> None:
        self.websocket = websocket
        self.trace = trace
        self.timeout_s = timeout_s
        self.events: asyncio.Queue[JsonObject | None] = asyncio.Queue()
        self.receiver = asyncio.create_task(self._receive())

    async def _receive(self) -> None:
        try:
            async for message in self.websocket:
                if not isinstance(message, str):
                    raise ProbeFailure(
                        "binary_server_message", "server returned a binary WebSocket message"
                    )
                try:
                    event = json.loads(message)
                except json.JSONDecodeError as error:
                    raise ProbeFailure(
                        "invalid_server_json", "server returned invalid JSON"
                    ) from error
                if not isinstance(event, dict):
                    raise ProbeFailure("invalid_server_json", "server event was not an object")
                self.trace.record("server", event)
                await self.events.put(event)
        finally:
            await self.events.put(None)

    async def send(self, event: JsonObject) -> None:
        self.trace.record("client", event)
        await self.websocket.send(json.dumps(event, separators=(",", ":")))

    async def wait_for(
        self,
        predicate: EventPredicate,
        label: str,
        *,
        timeout_s: float | None = None,
    ) -> JsonObject:
        deadline = time.monotonic() + (self.timeout_s if timeout_s is None else timeout_s)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProbeFailure("event_timeout", f"timed out waiting for {label}")
            try:
                event = await asyncio.wait_for(self.events.get(), timeout=remaining)
            except asyncio.TimeoutError as error:
                raise ProbeFailure("event_timeout", f"timed out waiting for {label}") from error
            if event is None:
                if self.receiver.done() and self.receiver.exception() is not None:
                    raise self.receiver.exception()  # type: ignore[misc]
                raise ProbeFailure("early_websocket_close", f"connection closed before {label}")
            if event.get("type") == "error":
                code = _nested_string(event, "error", "code") or "server_error"
                raise ProbeFailure("server_error", f"server returned error event {code}")
            if predicate(event):
                return event

    async def assert_no(self, predicate: EventPredicate, label: str, duration_s: float) -> None:
        deadline = time.monotonic() + duration_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                event = await asyncio.wait_for(self.events.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return
            if event is None:
                return
            if event.get("type") == "error":
                code = _nested_string(event, "error", "code") or "server_error"
                raise ProbeFailure("server_error", f"server returned error event {code}")
            if predicate(event):
                raise ProbeFailure("stale_output", f"observed {label}")

    async def close(self, event_id: str) -> None:
        await self.send({"type": "session.close", "event_id": event_id})
        try:
            await asyncio.wait_for(self.receiver, timeout=self.timeout_s)
        except asyncio.TimeoutError as error:
            raise ProbeFailure("close_timeout", "session.close did not close the socket") from error
        code = getattr(self.websocket, "close_code", None)
        self.trace.close_code = code if isinstance(code, int) else None
        self.trace.clean_close = self.trace.close_code == 1000


def _session_update(event_id: str, *, tools: bool) -> JsonObject:
    session: JsonObject = {
        "type": "realtime",
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": _WIRE_SAMPLE_RATE},
                "turn_detection": {
                    "type": "server_vad",
                    "create_response": True,
                    "interrupt_response": True,
                },
            },
            "output": {"format": {"type": "audio/pcm", "rate": _WIRE_SAMPLE_RATE}},
        },
    }
    if tools:
        session["tools"] = [
            {
                "type": "function",
                "name": _TOOL_NAME,
                "description": "Generate a random integer between min and max (inclusive).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "min": {"type": "integer"},
                        "max": {"type": "integer"},
                    },
                    "required": ["min", "max"],
                },
            }
        ]
        session["tool_choice"] = "auto"
    return {"type": "session.update", "event_id": event_id, "session": session}


async def _stream_audio(connection: LiveConnection, payload: bytes, prefix: str) -> None:
    for index, offset in enumerate(range(0, len(payload), _MAX_AUDIO_BYTES)):
        chunk = payload[offset : offset + _MAX_AUDIO_BYTES]
        await connection.send(
            {
                "type": "input_audio_buffer.append",
                "event_id": f"{prefix}_audio_{index}",
                "audio": base64.b64encode(chunk).decode("ascii"),
            }
        )
        await asyncio.sleep(0.005)


def _type(expected: str) -> EventPredicate:
    return lambda event: event.get("type") == expected


def _response_audio_for(response_id: str) -> EventPredicate:
    return lambda event: (
        event.get("type") == "response.output_audio.delta"
        and event.get("response_id") == response_id
    )


def _non_silent_response_audio_for(response_id: str) -> EventPredicate:
    def matches(event: JsonObject) -> bool:
        if not _response_audio_for(response_id)(event):
            return False
        delta = event.get("delta")
        if not isinstance(delta, str):
            return False
        try:
            stats = _audio_stats(base64.b64decode(delta, validate=True))
        except (binascii.Error, ValueError):
            return False
        return float(stats["rms"]) >= 0.001 and float(stats["peak"]) >= 0.01

    return matches


def _response_media_for(response_id: str) -> EventPredicate:
    return lambda event: (
        event.get("type")
        in {
            "response.output_audio.delta",
            "response.output_audio.done",
            "response.output_audio_transcript.delta",
            "response.output_audio_transcript.done",
        }
        and event.get("response_id") == response_id
    )


async def _configure(connection: LiveConnection, name: str, *, tools: bool) -> None:
    await connection.wait_for(_type("session.created"), f"{name} session.created")
    await connection.send(_session_update(f"{name}_update", tools=tools))
    await connection.wait_for(_type("session.updated"), f"{name} session.updated")


async def _commit_and_create(connection: LiveConnection, name: str) -> JsonObject:
    await connection.send({"type": "input_audio_buffer.commit", "event_id": f"{name}_commit"})
    await connection.wait_for(
        _type("input_audio_buffer.committed"), f"{name} input_audio_buffer.committed"
    )
    await connection.send({"type": "response.create", "event_id": f"{name}_create"})
    return await connection.wait_for(_type("response.created"), f"{name} response.created")


async def _create_recovery_response(
    connection: LiveConnection, name: str, interrupted_response_id: str
) -> None:
    await connection.send({"type": "response.create", "event_id": f"{name}_recovery_create"})
    created = await connection.wait_for(
        _type("response.created"), f"{name} recovery response.created"
    )
    recovery_response_id = _nested_string(created, "response", "id")
    if not recovery_response_id or recovery_response_id == interrupted_response_id:
        raise ProbeFailure(
            "invalid_recovery_response",
            f"{name} recovery did not create a distinct response",
        )
    media = _response_media_for(recovery_response_id)
    audio = _non_silent_response_audio_for(recovery_response_id)
    recovery_item_id: str | None = None
    saw_audio = False
    saw_audio_done = False
    saw_transcript_done = False
    while not (saw_audio and saw_audio_done and saw_transcript_done):
        event = await connection.wait_for(
            lambda candidate: (
                media(candidate)
                or (
                    candidate.get("type") == "response.done"
                    and _nested_string(candidate, "response", "id") == recovery_response_id
                )
            ),
            f"{name} recovery audio and transcript",
        )
        if event.get("type") == "response.done":
            raise ProbeFailure(
                "incomplete_recovery_response",
                f"{name} recovery completed before audio and transcript completion",
            )
        item_id = event.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ProbeFailure(
                "invalid_recovery_item",
                f"{name} recovery media did not contain an item ID",
            )
        if recovery_item_id is None:
            recovery_item_id = item_id
        elif item_id != recovery_item_id:
            raise ProbeFailure(
                "invalid_recovery_item",
                f"{name} recovery media crossed response items",
            )
        saw_audio = saw_audio or audio(event)
        saw_audio_done = saw_audio_done or event.get("type") == "response.output_audio.done"
        if event.get("type") == "response.output_audio_transcript.done":
            transcript = event.get("transcript")
            if not isinstance(transcript, str) or not transcript.strip():
                raise ProbeFailure(
                    "invalid_recovery_transcript",
                    f"{name} recovery transcript.done was empty",
                )
            saw_transcript_done = True
    done = await connection.wait_for(
        lambda event: (
            event.get("type") == "response.done"
            and _nested_string(event, "response", "id") == recovery_response_id
        ),
        f"{name} recovery response.done",
    )
    if _nested_string(done, "response", "status") != "completed":
        raise ProbeFailure(
            "incomplete_recovery_response",
            f"{name} recovery response did not complete",
        )
    await connection.assert_no(
        media,
        f"audio or transcript from the completed {name} recovery response",
        0.5,
    )


async def _function_scenario(connection: LiveConnection, audio: bytes) -> None:
    await _configure(connection, "function", tools=True)
    # Exercise clear after real frames have reached perception/RNNT, not only
    # the transport queue.  This crop begins inside the pinned sample's spoken
    # request and contains two seconds of continuous speech.
    clear_start = round(3.7 * _WIRE_SAMPLE_RATE) * 2
    clear_probe = audio[clear_start : clear_start + 4 * _WIRE_SAMPLE_RATE]
    await _stream_audio(connection, clear_probe, "function_clear_probe")
    await connection.wait_for(
        _type("conversation.item.input_audio_transcription.delta"),
        "processed input before clear",
    )
    await connection.send({"type": "input_audio_buffer.clear", "event_id": "function_clear"})
    await connection.wait_for(
        _type("input_audio_buffer.cleared"), "function input_audio_buffer.cleared"
    )
    await _stream_audio(connection, audio, "function")
    await connection.wait_for(_type("response.created"), "function response.created")
    call = await connection.wait_for(
        _type("response.function_call_arguments.done"), "function call arguments"
    )
    call_id = call.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        raise ProbeFailure("invalid_function_call", "function call did not contain call_id")
    if call.get("name") != _TOOL_NAME:
        raise ProbeFailure("wrong_function_call", "model selected an unexpected tool")
    call_response_id = call.get("response_id")
    if not isinstance(call_response_id, str) or not call_response_id:
        raise ProbeFailure("invalid_function_call", "function call did not contain response_id")
    await connection.wait_for(
        lambda event: (
            event.get("type") == "response.done"
            and _nested_string(event, "response", "id") == call_response_id
            and _nested_string(event, "response", "status") == "completed"
        ),
        "function call response completion",
    )
    await connection.send(
        {
            "type": "conversation.item.create",
            "event_id": "function_output",
            "item": {"type": "function_call_output", "call_id": call_id, "output": _TOOL_RESULT},
        }
    )
    await connection.wait_for(_type("conversation.item.added"), "function output item added")
    await connection.wait_for(_type("conversation.item.done"), "function output item done")
    await connection.send({"type": "response.create", "event_id": "function_continue"})
    continuation = await connection.wait_for(
        _type("response.created"), "function continuation response.created"
    )
    continuation_response_id = _nested_string(continuation, "response", "id")
    if continuation_response_id is None or continuation_response_id == call_response_id:
        raise ProbeFailure(
            "invalid_function_continuation",
            "function continuation did not create a distinct response",
        )
    await connection.wait_for(
        lambda event: (
            event.get("type") == "response.output_audio.delta"
            and event.get("response_id") == continuation_response_id
        ),
        "function continuation audio",
    )
    await connection.wait_for(
        lambda event: (
            event.get("type")
            in {"response.output_audio_transcript.delta", "response.output_audio_transcript.done"}
        ),
        "function continuation transcript",
    )
    await connection.wait_for(
        lambda event: (
            event.get("type") == "response.done"
            and _nested_string(event, "response", "id") == continuation_response_id
            and _nested_string(event, "response", "status") == "completed"
        ),
        "function response completion",
    )
    await connection.close("function_close")


async def _truncate_scenario(connection: LiveConnection, audio: bytes) -> None:
    await _configure(connection, "truncate", tools=False)
    await _stream_audio(connection, audio[: 4 * _WIRE_SAMPLE_RATE], "truncate")
    created = await _commit_and_create(connection, "truncate")
    response_id = _nested_string(created, "response", "id")
    if response_id is None:
        raise ProbeFailure("invalid_response", "response.created did not contain an ID")
    delta = await connection.wait_for(
        _non_silent_response_audio_for(response_id), "non-silent truncate response audio"
    )
    item_id = delta.get("item_id")
    if not isinstance(item_id, str) or not item_id:
        raise ProbeFailure("invalid_response", "response audio did not contain an item ID")
    await connection.send(
        {
            "type": "conversation.item.truncate",
            "event_id": "truncate_control",
            "item_id": item_id,
            "content_index": 0,
            "audio_end_ms": 80,
        }
    )
    await connection.wait_for(_type("conversation.item.truncated"), "conversation item truncated")
    await connection.wait_for(
        lambda event: (
            event.get("type") == "response.done"
            and _nested_string(event, "response", "id") == response_id
            and _nested_string(event, "response", "status") == "cancelled"
        ),
        "truncated response.done",
    )
    await connection.assert_no(
        _response_audio_for(response_id), "audio from the truncated response", 0.5
    )
    await _create_recovery_response(connection, "truncate", response_id)
    await connection.assert_no(
        _response_audio_for(response_id), "audio from the truncated response", 0.5
    )
    await connection.close("truncate_close")


async def _cancel_scenario(connection: LiveConnection, audio: bytes) -> None:
    await _configure(connection, "cancel", tools=False)
    await _stream_audio(connection, audio[: 4 * _WIRE_SAMPLE_RATE], "cancel")
    created = await _commit_and_create(connection, "cancel")
    response_id = _nested_string(created, "response", "id")
    if response_id is None:
        raise ProbeFailure("invalid_response", "response.created did not contain an ID")
    await connection.wait_for(
        _non_silent_response_audio_for(response_id), "non-silent cancel response audio"
    )
    await connection.send(
        {"type": "response.cancel", "event_id": "cancel_control", "response_id": response_id}
    )
    await connection.wait_for(
        lambda event: (
            event.get("type") == "response.done"
            and _nested_string(event, "response", "id") == response_id
            and _nested_string(event, "response", "status") == "cancelled"
            and _nested_string(event, "response", "status_details", "reason") == "client_cancelled"
        ),
        "cancelled response.done",
    )
    await connection.assert_no(
        _response_audio_for(response_id), "audio from the cancelled response", 0.5
    )
    await _create_recovery_response(connection, "cancel", response_id)
    await connection.assert_no(
        _response_audio_for(response_id), "audio from the cancelled response", 0.5
    )
    await connection.close("cancel_close")


def _pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


async def _connect(
    connect: Any, uri: str, server: asyncio.subprocess.Process, timeout_s: int
) -> Any:
    deadline = time.monotonic() + min(timeout_s, 30)
    while True:
        if server.returncode is not None:
            raise ProbeFailure("server_start_failed", "realtime server exited during startup")
        try:
            return await connect(
                uri,
                additional_headers={"Authorization": f"Bearer {_TOKEN}"},
                compression=None,
                max_size=1 << 20,
            )
        except (OSError, asyncio.TimeoutError):
            if time.monotonic() >= deadline:
                raise ProbeFailure("server_start_timeout", "realtime server did not accept sockets")
            await asyncio.sleep(0.05)


async def _run_scenario(
    connect: Any,
    uri: str,
    server: asyncio.subprocess.Process,
    timeout_s: int,
    trace: ScenarioTrace,
    run: Callable[[LiveConnection], Any],
) -> None:
    websocket = await _connect(connect, uri, server, timeout_s)
    connection = LiveConnection(websocket, trace, timeout_s)
    try:
        await run(connection)
    finally:
        if not connection.receiver.done():
            connection.receiver.cancel()
            await asyncio.gather(connection.receiver, return_exceptions=True)
        if not getattr(websocket, "closed", False):
            await websocket.close()


async def _terminate_server(server: asyncio.subprocess.Process) -> None:
    if server.returncode is not None:
        return
    server.terminate()
    try:
        await asyncio.wait_for(server.wait(), timeout=2)
    except asyncio.TimeoutError:
        server.kill()
        await server.wait()


async def _run_probe(arguments: argparse.Namespace, traces: list[ScenarioTrace]) -> None:
    try:
        from websockets.asyncio.client import connect
    except ImportError as error:
        raise ProbeFailure(
            "websockets_unavailable", "the realtime WebSocket dependency is unavailable"
        ) from error

    general_rate, general_samples = _read_pcm16(arguments.general_audio)
    function_rate, function_samples = _read_pcm16(arguments.function_audio)
    general_audio = _resample_linear(general_samples, general_rate)
    function_audio = _resample_linear(function_samples, function_rate)

    port = _pick_port()
    env = dict(os.environ)
    env[_TOKEN_ENV] = _TOKEN
    command = [
        sys.executable,
        "-m",
        "tensorrt_model_connect.realtime",
        "--worker",
        str(arguments.worker),
        "--bundle",
        str(arguments.bundle),
        "--backend-dir",
        str(arguments.backend_dir),
        "--model-plugin-dir",
        str(arguments.model_plugin_dir),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    server = await asyncio.create_subprocess_exec(
        *command,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    uri = f"ws://127.0.0.1:{port}/v1/realtime"
    try:
        function_trace = ScenarioTrace("function")
        traces.append(function_trace)
        await _run_scenario(
            connect,
            uri,
            server,
            arguments.event_timeout_s,
            function_trace,
            lambda connection: _function_scenario(connection, function_audio),
        )
        truncate_trace = ScenarioTrace("truncate")
        traces.append(truncate_trace)
        await _run_scenario(
            connect,
            uri,
            server,
            arguments.event_timeout_s,
            truncate_trace,
            lambda connection: _truncate_scenario(connection, general_audio),
        )
        cancel_trace = ScenarioTrace("cancel")
        traces.append(cancel_trace)
        await _run_scenario(
            connect,
            uri,
            server,
            arguments.event_timeout_s,
            cancel_trace,
            lambda connection: _cancel_scenario(connection, general_audio),
        )
    finally:
        await _terminate_server(server)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--backend-dir", type=Path, required=True)
    parser.add_argument("--model-plugin-dir", type=Path, required=True)
    parser.add_argument("--general-audio", type=Path, required=True)
    parser.add_argument("--function-audio", type=Path, required=True)
    parser.add_argument("--output-wav", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--event-timeout-s", type=int, default=180)
    arguments = parser.parse_args()
    if arguments.event_timeout_s <= 0:
        parser.error("--event-timeout-s must be positive")
    return arguments


def main() -> int:
    arguments = _parse_arguments()
    traces: list[ScenarioTrace] = []
    failure_code = ""
    failure_message = ""
    try:
        asyncio.run(_run_probe(arguments, traces))
    except ProbeFailure as error:
        failure_code = error.code
        failure_message = str(error)
    except Exception:
        failure_code = "unexpected_probe_failure"
        failure_message = "realtime transport probe failed unexpectedly"

    output_audio = b"".join(bytes(trace.output_audio) for trace in traces)
    _write_wav(arguments.output_wav, output_audio)
    receipt = {
        "schema_version": 1,
        "pass": not failure_code and len(traces) == 3,
        "runtime": "Python /v1/realtime host with native JSONL worker",
        "wire_sample_rate": _WIRE_SAMPLE_RATE,
        "failure_code": failure_code,
        "failure_message": failure_message,
        "scenarios": [trace.receipt() for trace in traces],
        "combined_audio": _audio_stats(output_audio),
    }
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
