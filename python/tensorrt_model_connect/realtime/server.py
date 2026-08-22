# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local WebSocket host for native full-duplex speech workers."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hmac
import inspect
import json
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any
from urllib.parse import urlsplit

from .protocol import (
    DEFAULT_MAX_AUDIO_BYTES,
    DEFAULT_MAX_MESSAGE_BYTES,
    PCM16_SAMPLE_RATE,
    JsonObject,
    RealtimeProtocolError,
    parse_client_message,
    protocol_error_event,
    session_snapshot,
)
from .worker import RealtimeWorker, WorkerError


REALTIME_PATH = "/v1/realtime"

WorkerFactory = Callable[[], RealtimeWorker | Awaitable[RealtimeWorker]]
IdFactory = Callable[[str], str]


class QueueCapacityError(RuntimeError):
    """A bounded realtime queue has no remaining capacity."""


@dataclass(frozen=True)
class RealtimeServerConfig:
    """Network and resource limits for a local realtime host."""

    bearer_token: str = field(repr=False)
    host: str = "127.0.0.1"
    port: int = 8765
    input_queue_size: int = 32
    message_queue_size: int = 32
    output_queue_size: int = 128
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES
    max_audio_bytes: int = DEFAULT_MAX_AUDIO_BYTES

    def __post_init__(self) -> None:
        if not self.bearer_token:
            raise ValueError("Realtime bearer token must not be empty")
        if not self.host:
            raise ValueError("Realtime host must not be empty")
        if not 1 <= self.port <= 65_535:
            raise ValueError("Realtime port must be between 1 and 65535")
        for name in ("input_queue_size", "message_queue_size", "output_queue_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"Realtime {name} must be positive")
        if self.max_message_bytes <= 0 or self.max_audio_bytes <= 0:
            raise ValueError("Realtime message and audio bounds must be positive")


@dataclass(frozen=True)
class _Outbound:
    event: JsonObject
    epoch: int | None = None
    stale_sensitive: bool = False


@dataclass
class _Turn:
    epoch: int
    response_id: str
    message_item_id: str
    text: str = ""
    audio_emitted: bool = False
    text_completed: bool = False
    function_items: list[JsonObject] = field(default_factory=list)
    done: bool = False


@dataclass(frozen=True)
class _PendingCall:
    epoch: int
    item_id: str
    output_index: int


@dataclass(frozen=True)
class _BufferedToolOutput:
    item_id: str
    output: str


@dataclass(frozen=True)
class _PendingNative:
    native_event_id: str
    kind: str
    client_event_id: str | None
    data: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class _PreparedClient:
    messages: tuple[JsonObject, ...]
    pending: tuple[_PendingNative, ...] = ()


class RealtimeSession:
    """Protocol state and three bounded queues for one WebSocket session."""

    def __init__(
        self,
        worker: RealtimeWorker,
        config: RealtimeServerConfig,
        *,
        id_factory: IdFactory | None = None,
    ) -> None:
        self.worker = worker
        self.config = config
        self._id_factory = id_factory or _new_id
        self.session_id = self._id_factory("sess")
        self.input_queue: asyncio.Queue[str | bytes] = asyncio.Queue(
            maxsize=config.input_queue_size
        )
        self.message_queue: asyncio.Queue[JsonObject] = asyncio.Queue(
            maxsize=config.message_queue_size
        )
        self.output_queue: asyncio.Queue[_Outbound] = asyncio.Queue(
            maxsize=config.output_queue_size
        )
        self.current_epoch = 0
        self.close_requested = False
        self.worker_ended = False
        self._ready = False
        self._configuration_requested = False
        self._audio_started = False
        self._instructions = ""
        self._tools: list[JsonObject] = []
        self._tool_choice = "auto"
        self._turns: dict[int, _Turn] = {}
        self._turn_history: list[_Turn] = []
        self._active_turn_epoch: int | None = None
        self._user_item_ids: dict[int, str] = {}
        self._user_transcripts: dict[int, str] = {}
        self._pending_calls: dict[str, _PendingCall] = {}
        self._submitted_outputs: dict[str, _BufferedToolOutput] = {}
        self._pending_native: dict[str, _PendingNative] = {}
        self._response_create_native_ids: set[str] = set()
        self._response_create_kind: str | None = None
        self._last_conversation_item_id: str | None = None
        self._last_sequence: dict[int, int] = {}

    def enqueue_input(self, message: str | bytes) -> None:
        """Queue one client frame without exceeding the configured bound."""

        try:
            self.input_queue.put_nowait(message)
        except asyncio.QueueFull as exc:
            raise QueueCapacityError("Realtime input queue capacity exceeded") from exc

    async def process_next_input(self) -> None:
        message = await self.input_queue.get()
        try:
            await self.handle_client_message(message)
        finally:
            self.input_queue.task_done()

    async def handle_client_message(self, message: str | bytes) -> None:
        """Validate one client message and queue its native JSONL request."""

        try:
            event = parse_client_message(
                message,
                max_message_bytes=self.config.max_message_bytes,
                max_audio_bytes=self.config.max_audio_bytes,
            )
            prepared = self._prepare_client_event(event)
            self._put_worker_messages(prepared.messages)
            self._record_client_event(event, prepared)
        except RealtimeProtocolError as exc:
            self._put_output(_Outbound(protocol_error_event(exc, self._id_factory("event"))))

    def handle_worker_message(self, message: object) -> None:
        """Validate and map one native worker event into public server events."""

        worker_event = _worker_object(message)
        event_type = _worker_string(worker_event, "type")
        if event_type == "session.ready":
            if self._ready:
                raise WorkerError("native worker repeated session.ready")
            self._ready = True
            self._put_output(
                _Outbound(
                    {
                        "type": "session.created",
                        "event_id": self._id_factory("event"),
                        "session": self._session_snapshot(),
                    }
                )
            )
            return
        if event_type == "session.updated":
            self._put_output(
                _Outbound(
                    {
                        "type": "session.updated",
                        "event_id": self._id_factory("event"),
                        "session": self._session_snapshot(),
                    }
                )
            )
            return
        if event_type in {
            "input_audio_buffer.committed",
            "input_audio_buffer.cleared",
            "response.cancelled",
            "conversation.item.truncated",
        }:
            self._handle_control_ack(worker_event)
            return
        if event_type == "error":
            self._put_outputs(self._map_worker_error(worker_event))
            return
        if event_type == "session.end":
            self.worker_ended = True
            return
        if event_type != "session.event":
            raise WorkerError("native worker returned an unsupported event type")
        self._handle_native_session_event(worker_event)

    def handle_worker_failure(self) -> None:
        """Queue a sanitized failure without forwarding native diagnostics."""

        self._purge_stale_outputs(self.current_epoch + 1)
        self._put_output(_Outbound(self._worker_error_event()))

    def _handle_control_ack(self, event: JsonObject) -> None:
        event_type = _worker_string(event, "type")
        expected_kind = {
            "input_audio_buffer.committed": "input_audio_buffer.commit",
            "input_audio_buffer.cleared": "input_audio_buffer.clear",
            "response.cancelled": "response.cancel",
            "conversation.item.truncated": "conversation.item.truncate",
        }[event_type]
        native_event_id = _worker_string(event, "event_id")
        pending = self._pending_native.get(native_event_id)
        if pending is None or pending.kind != expected_kind:
            raise WorkerError("native worker acknowledgement does not match a pending control")
        del self._pending_native[native_event_id]

        if event_type == "input_audio_buffer.committed":
            item_id = pending.data["item_id"]
            epoch = pending.data["epoch"]
            self._user_item_ids[epoch] = item_id
            self._last_conversation_item_id = item_id
            public = {
                "type": event_type,
                "event_id": self._id_factory("event"),
                "previous_item_id": pending.data["previous_item_id"],
                "item_id": item_id,
            }
        elif event_type == "input_audio_buffer.cleared":
            epoch = pending.data["epoch"]
            self._user_item_ids.pop(epoch, None)
            self._user_transcripts.pop(epoch, None)
            public = {"type": event_type, "event_id": self._id_factory("event")}
        elif event_type == "response.cancelled":
            turn = self._turns.get(pending.data["epoch"])
            if turn is None or turn.response_id != pending.data["response_id"]:
                raise WorkerError("native response cancellation lost its public response state")
            self._put_outputs(self._cancelled_response_events(turn, "client_cancelled"))
            return
        else:
            epoch = _worker_integer(event, "epoch", minimum=0)
            samples = _worker_integer(event, "played_output_samples", minimum=0)
            if epoch != pending.data["epoch"] or samples != pending.data["played_output_samples"]:
                raise WorkerError("native truncation acknowledgement changed its target")
            turn = next(
                (
                    candidate
                    for candidate in self._turn_history
                    if candidate.message_item_id == pending.data["item_id"]
                ),
                None,
            )
            if turn is None:
                raise WorkerError("native truncation lost its public response item")
            self._purge_turn_outputs(turn)
            turn.text = ""
            turn.text_completed = False
            public = {
                "type": event_type,
                "event_id": self._id_factory("event"),
                "item_id": pending.data["item_id"],
                "content_index": pending.data["content_index"],
                "audio_end_ms": pending.data["audio_end_ms"],
            }
        self._put_output(_Outbound(public))

    def _map_worker_error(self, event: JsonObject) -> list[_Outbound]:
        native_event_id = event.get("event_id")
        public_event = dict(event)
        failed_turn: _Turn | None = None
        if isinstance(native_event_id, str) and native_event_id in self._pending_native:
            pending = self._pending_native.pop(native_event_id)
            public_event["event_id"] = pending.client_event_id
            if native_event_id in self._response_create_native_ids:
                if self._response_create_kind == "tool.resume":
                    failed_turn = self._active_turn()
                self._clear_response_create()
        events = [_Outbound(self._worker_error_event(public_event))]
        if failed_turn is not None:
            events.extend(self._failed_response_events(failed_turn))
        return events

    def _failed_response_events(self, turn: _Turn) -> list[_Outbound]:
        if turn.done:
            return []
        self._purge_turn_outputs(turn)
        turn.done = True
        if self._active_turn_epoch == turn.epoch:
            self._active_turn_epoch = None
        return [
            _Outbound(
                {
                    "type": "response.done",
                    "event_id": self._id_factory("event"),
                    "response": self._response(
                        turn,
                        status="failed",
                        status_details={
                            "type": "failed",
                            "error": {"type": "server_error", "code": "worker_error"},
                        },
                    ),
                },
                epoch=turn.epoch,
            )
        ]

    def _clear_response_create(self) -> None:
        for native_event_id in self._response_create_native_ids:
            self._pending_native.pop(native_event_id, None)
        self._response_create_native_ids.clear()
        self._response_create_kind = None

    def _purge_turn_outputs(self, turn: _Turn) -> None:
        retained: list[_Outbound] = []
        while True:
            try:
                outbound = self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self.output_queue.task_done()
            if not (
                outbound.stale_sensitive and outbound.event.get("response_id") == turn.response_id
            ):
                retained.append(outbound)
        for outbound in retained:
            self.output_queue.put_nowait(outbound)

    def _cancelled_response_events(self, turn: _Turn, reason: str) -> list[_Outbound]:
        if turn.done:
            return []
        self._purge_turn_outputs(turn)
        turn.done = True
        if self._active_turn_epoch == turn.epoch:
            self._active_turn_epoch = None
        events: list[_Outbound] = []
        if turn.text and not turn.text_completed:
            events.append(
                _Outbound(
                    {
                        "type": "response.output_audio_transcript.done",
                        "event_id": self._id_factory("event"),
                        "response_id": turn.response_id,
                        "item_id": turn.message_item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "transcript": turn.text,
                    },
                    epoch=turn.epoch,
                )
            )
        if turn.audio_emitted:
            events.append(
                _Outbound(
                    {
                        "type": "response.output_audio.done",
                        "event_id": self._id_factory("event"),
                        "response_id": turn.response_id,
                        "item_id": turn.message_item_id,
                        "output_index": 0,
                        "content_index": 0,
                    },
                    epoch=turn.epoch,
                )
            )
        events.append(
            _Outbound(
                {
                    "type": "response.done",
                    "event_id": self._id_factory("event"),
                    "response": self._response(
                        turn,
                        status="cancelled",
                        status_details={"type": "cancelled", "reason": reason},
                    ),
                },
                epoch=turn.epoch,
            )
        )
        return events

    async def next_output(self) -> JsonObject:
        """Return the next non-stale public event."""

        while True:
            outbound = await self.output_queue.get()
            self.output_queue.task_done()
            if self._is_stale(outbound):
                continue
            return outbound.event

    def drain_output(self) -> list[JsonObject]:
        """Drain all currently queued, non-stale public events."""

        events: list[JsonObject] = []
        while True:
            try:
                outbound = self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                return events
            self.output_queue.task_done()
            if not self._is_stale(outbound):
                events.append(outbound.event)

    def drain_worker_messages(self) -> list[JsonObject]:
        """Drain native requests. Primarily useful for deterministic tests."""

        messages: list[JsonObject] = []
        while True:
            try:
                message = self.message_queue.get_nowait()
            except asyncio.QueueEmpty:
                return messages
            self.message_queue.task_done()
            messages.append(message)

    def _prepare_client_event(self, event: JsonObject) -> _PreparedClient:
        event_type = event["type"]
        event_id = event.get("event_id")
        if self.close_requested:
            raise RealtimeProtocolError(
                "session is already closing",
                code="invalid_state",
                event_id=event_id,
            )
        if event_type == "session.update":
            if self._audio_started:
                raise RealtimeProtocolError(
                    "session cannot be updated after audio starts",
                    code="invalid_state",
                    event_id=event_id,
                )
            instructions = event.get("instructions", self._instructions)
            configured_tools = event.get("tools", self._tools)
            tool_choice = event.get("tool_choice", self._tool_choice)
            tools = configured_tools if tool_choice != "none" else []
            message: JsonObject = {
                "type": "session.update",
                "input_sample_rate": PCM16_SAMPLE_RATE,
                "output_sample_rate": PCM16_SAMPLE_RATE,
                "instructions": instructions,
                "tools": tools,
                "on_hold_messages": {},
            }
            return _PreparedClient((self._with_client_event_id(message, event_id),))
        elif event_type == "input_audio_buffer.append":
            if not self._configuration_requested:
                raise RealtimeProtocolError(
                    "session.update is required before audio",
                    code="invalid_state",
                    event_id=event_id,
                )
            message = {"type": event_type, "audio": event["audio"]}
            return _PreparedClient((self._with_client_event_id(message, event_id),))
        elif event_type == "input_audio_buffer.commit":
            self._require_configuration(event_id)
            item_id = self._user_item_ids.get(self.current_epoch) or self._id_factory("item")
            return self._pending_preparation(
                event_type,
                event_id,
                {"type": event_type},
                {
                    "item_id": item_id,
                    "previous_item_id": self._last_conversation_item_id,
                    "epoch": self.current_epoch,
                },
            )
        elif event_type == "input_audio_buffer.clear":
            self._require_configuration(event_id)
            return self._pending_preparation(
                event_type,
                event_id,
                {"type": event_type},
                {"epoch": self.current_epoch},
            )
        elif event_type == "response.create":
            return self._prepare_response_create(event_id)
        elif event_type == "response.cancel":
            turn = self._validate_cancel_target(event.get("response_id"), event_id)
            return self._pending_preparation(
                event_type,
                event_id,
                {"type": event_type},
                {"epoch": turn.epoch, "response_id": turn.response_id},
            )
        elif event_type == "conversation.item.truncate":
            turn = self._turn_for_message_item(event["item_id"], event_id)
            played_samples = event["audio_end_ms"] * PCM16_SAMPLE_RATE // 1000
            return self._pending_preparation(
                event_type,
                event_id,
                {
                    "type": event_type,
                    "epoch": turn.epoch,
                    "played_output_samples": played_samples,
                },
                {
                    "epoch": turn.epoch,
                    "played_output_samples": played_samples,
                    "item_id": event["item_id"],
                    "content_index": event["content_index"],
                    "audio_end_ms": event["audio_end_ms"],
                },
            )
        elif event_type == "conversation.item.create":
            call_id = event["call_id"]
            pending = self._pending_calls.get(call_id)
            if pending is None or pending.epoch < self.current_epoch:
                raise RealtimeProtocolError(
                    "function output does not match a pending call",
                    code="invalid_call_id",
                    param="item.call_id",
                    event_id=event_id,
                )
            if call_id in self._submitted_outputs:
                raise RealtimeProtocolError(
                    "function output is already buffered",
                    code="invalid_state",
                    param="item.call_id",
                    event_id=event_id,
                )
            return _PreparedClient(())
        elif event_type == "session.close":
            return _PreparedClient(
                (self._with_client_event_id({"type": "session.close"}, event_id),)
            )
        else:  # validate_client_event() owns the public event allowlist.
            raise AssertionError(f"unrouted validated client event: {event_type}")

    def _record_client_event(self, event: JsonObject, prepared: _PreparedClient) -> None:
        event_type = event["type"]
        for pending in prepared.pending:
            self._pending_native[pending.native_event_id] = pending
        if event_type == "session.update":
            if "instructions" in event:
                self._instructions = event["instructions"]
            if "tools" in event:
                self._tools = event["tools"]
            if "tool_choice" in event:
                self._tool_choice = event["tool_choice"]
            self._configuration_requested = True
        elif event_type == "input_audio_buffer.append":
            self._audio_started = True
        elif event_type in {
            "input_audio_buffer.commit",
            "input_audio_buffer.clear",
            "response.cancel",
            "conversation.item.truncate",
        }:
            pass
        elif event_type == "response.create":
            self._response_create_native_ids = {
                pending.native_event_id for pending in prepared.pending
            }
            self._response_create_kind = prepared.pending[0].kind
            if self._response_create_kind == "tool.resume":
                epochs = {pending.data["epoch"] for pending in prepared.pending}
                if len(epochs) != 1:
                    raise WorkerError("buffered function outputs span native epochs")
                self._start_tool_continuation(next(iter(epochs)))
        elif event_type == "conversation.item.create":
            item_id = self._id_factory("item")
            buffered = _BufferedToolOutput(item_id, event["output"])
            self._submitted_outputs[event["call_id"]] = buffered
            item = {
                "id": item_id,
                "object": "realtime.item",
                "type": "function_call_output",
                "status": "completed",
                "call_id": event["call_id"],
                "output": event["output"],
            }
            self._put_outputs(
                [
                    _Outbound(
                        {
                            "type": "conversation.item.added",
                            "event_id": self._id_factory("event"),
                            "previous_item_id": self._last_conversation_item_id,
                            "item": item,
                        }
                    ),
                    _Outbound(
                        {
                            "type": "conversation.item.done",
                            "event_id": self._id_factory("event"),
                            "previous_item_id": self._last_conversation_item_id,
                            "item": item,
                        }
                    ),
                ]
            )
            self._last_conversation_item_id = item_id
        elif event_type == "session.close":
            self.close_requested = True
        else:
            raise AssertionError(f"unrecorded validated client event: {event_type}")

    def _prepare_response_create(self, event_id: str | None) -> _PreparedClient:
        if self._response_create_native_ids:
            raise RealtimeProtocolError(
                "response creation is already in progress",
                code="invalid_state",
                event_id=event_id,
            )
        live_calls = {
            call_id: pending
            for call_id, pending in self._pending_calls.items()
            if pending.epoch >= self.current_epoch
        }
        if live_calls:
            missing = sorted(set(live_calls) - set(self._submitted_outputs))
            if missing:
                raise RealtimeProtocolError(
                    "response.create requires output for every pending function call",
                    code="invalid_state",
                    param="response",
                    event_id=event_id,
                )
            messages: list[JsonObject] = []
            pending_requests: list[_PendingNative] = []
            for call_id, call in live_calls.items():
                buffered = self._submitted_outputs[call_id]
                prepared = self._pending_preparation(
                    "tool.resume",
                    event_id,
                    {
                        "type": "conversation.item.create",
                        "epoch": call.epoch,
                        "call_id": call_id,
                        "output": buffered.output,
                    },
                    {"call_id": call_id, "epoch": call.epoch},
                )
                messages.extend(prepared.messages)
                pending_requests.extend(prepared.pending)
            return _PreparedClient(tuple(messages), tuple(pending_requests))
        return self._pending_preparation("response.create", event_id, {"type": "response.create"})

    def _start_tool_continuation(self, epoch: int) -> None:
        previous = self._turns.get(epoch)
        if previous is None or not previous.done:
            raise WorkerError("tool continuation does not follow a completed public response")
        turn = self._new_public_turn(epoch)
        self._put_output(
            _Outbound(
                {
                    "type": "response.created",
                    "event_id": self._id_factory("event"),
                    "response": self._response(turn, status="in_progress"),
                },
                epoch=epoch,
            )
        )

    def _pending_preparation(
        self,
        kind: str,
        client_event_id: str | None,
        message: JsonObject,
        data: JsonObject | None = None,
    ) -> _PreparedClient:
        native_event_id = self._id_factory("native")
        native_message = {**message, "event_id": native_event_id}
        pending = _PendingNative(
            native_event_id,
            kind,
            client_event_id,
            {} if data is None else data,
        )
        return _PreparedClient((native_message,), (pending,))

    @staticmethod
    def _with_client_event_id(message: JsonObject, event_id: str | None) -> JsonObject:
        if event_id is None:
            return message
        return {**message, "event_id": event_id}

    def _require_configuration(self, event_id: str | None) -> None:
        if not self._configuration_requested:
            raise RealtimeProtocolError(
                "session.update is required before input buffer control",
                code="invalid_state",
                event_id=event_id,
            )

    def _validate_cancel_target(self, response_id: object, event_id: str | None) -> _Turn:
        active = self._active_turn()
        if active is None or (response_id is not None and response_id != active.response_id):
            raise RealtimeProtocolError(
                "response_id does not match the active response",
                code="invalid_response_id",
                param="response_id",
                event_id=event_id,
            )
        return active

    def _turn_for_message_item(self, item_id: str, event_id: str | None) -> _Turn:
        for turn in self._turn_history:
            if turn.message_item_id == item_id and turn.audio_emitted:
                return turn
        raise RealtimeProtocolError(
            "item_id does not match an assistant audio message",
            code="invalid_item_id",
            param="item_id",
            event_id=event_id,
        )

    def _put_worker_messages(self, messages: tuple[JsonObject, ...]) -> None:
        if self.message_queue.qsize() + len(messages) > self.message_queue.maxsize:
            raise QueueCapacityError("Realtime native message queue capacity exceeded")
        for message in messages:
            self.message_queue.put_nowait(message)

    def _handle_native_session_event(self, event: JsonObject) -> None:
        kind = _worker_string(event, "kind")
        epoch = _worker_integer(event, "epoch", minimum=0)
        sequence = _worker_integer(event, "sequence", minimum=0)
        if epoch < self.current_epoch:
            return
        previous_sequence = self._last_sequence.get(epoch)
        if previous_sequence is not None and sequence <= previous_sequence:
            raise WorkerError("native worker sequence did not increase within its epoch")
        self._last_sequence[epoch] = sequence
        turn = self._turns.get(epoch)
        if (
            turn is not None
            and turn.done
            and kind
            in {
                "agent_audio",
                "agent_text",
                "function_call",
                "function_call_started",
            }
        ):
            return

        interrupted_turn = self._active_turn()
        if epoch > self.current_epoch:
            self._advance_epoch(epoch)

        mapped: list[_Outbound]
        if kind == "user_speech_started":
            mapped = [self._map_speech_boundary(event, epoch, started=True)]
        elif kind == "user_speech_stopped":
            mapped = [self._map_speech_boundary(event, epoch, started=False)]
        elif kind == "user_transcript":
            mapped = self._map_user_transcript(event, epoch)
        elif kind == "turn_started":
            mapped = [self._map_turn_started(epoch)]
        elif kind == "agent_audio":
            mapped = [self._map_agent_audio(event, epoch)]
        elif kind == "agent_text":
            mapped = self._map_agent_text(event, epoch)
        elif kind == "function_call":
            mapped = self._map_function_call(event, epoch)
        elif kind == "function_response_finished":
            self._finish_tool_resume(epoch)
            mapped = []
        elif kind in {"function_call_started", "input_finished"}:
            mapped = []
        elif kind == "turn_finished":
            mapped = self._map_turn_finished(epoch)
        elif kind in {"yielded", "cancelled", "reset"}:
            mapped = self._map_interruption(kind, epoch, interrupted_turn, event)
        elif kind == "error":
            mapped = [_Outbound(self._worker_error_event())]
        else:
            raise WorkerError("native worker returned an unsupported speech event kind")
        self._put_outputs(mapped)

    def _advance_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        self._purge_stale_outputs(epoch)
        self._pending_calls = {
            call_id: call for call_id, call in self._pending_calls.items() if call.epoch >= epoch
        }
        self._submitted_outputs = {
            call_id: output
            for call_id, output in self._submitted_outputs.items()
            if call_id in self._pending_calls
        }

    def _finish_tool_resume(self, epoch: int) -> None:
        if self._response_create_kind == "tool.resume":
            self._clear_response_create()
        completed = {
            call_id for call_id, call in self._pending_calls.items() if call.epoch == epoch
        }
        for call_id in completed:
            self._pending_calls.pop(call_id, None)
            self._submitted_outputs.pop(call_id, None)

    def _purge_stale_outputs(self, epoch: int) -> None:
        retained: list[_Outbound] = []
        while True:
            try:
                outbound = self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self.output_queue.task_done()
            if not (
                outbound.stale_sensitive and outbound.epoch is not None and outbound.epoch < epoch
            ):
                retained.append(outbound)
        for outbound in retained:
            self.output_queue.put_nowait(outbound)

    def _map_speech_boundary(self, event: JsonObject, epoch: int, *, started: bool) -> _Outbound:
        item_id = self._user_item_ids.setdefault(epoch, self._id_factory("item"))
        sample_rate = _optional_worker_integer(event, "sample_rate", minimum=1) or PCM16_SAMPLE_RATE
        coordinate_name = "media_start_sample" if started else "media_end_sample"
        sample = _optional_worker_integer(event, coordinate_name, minimum=-1)
        milliseconds = 0 if sample is None or sample < 0 else sample * 1000 // sample_rate
        suffix = "started" if started else "stopped"
        coordinate = "audio_start_ms" if started else "audio_end_ms"
        return _Outbound(
            {
                "type": f"input_audio_buffer.speech_{suffix}",
                "event_id": self._id_factory("event"),
                coordinate: milliseconds,
                "item_id": item_id,
            },
            epoch=epoch,
        )

    def _map_user_transcript(self, event: JsonObject, epoch: int) -> list[_Outbound]:
        text = _optional_worker_string(event, "text") or ""
        final = _optional_worker_boolean(event, "is_final") or False
        item_id = self._user_item_ids.setdefault(epoch, self._id_factory("item"))
        previous = self._user_transcripts.get(epoch, "")
        self._user_transcripts[epoch] = text
        if final:
            return [
                _Outbound(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "event_id": self._id_factory("event"),
                        "item_id": item_id,
                        "content_index": 0,
                        "transcript": text,
                    },
                    epoch=epoch,
                )
            ]
        delta = text[len(previous) :] if text.startswith(previous) else text
        if not delta:
            return []
        events = [
            _Outbound(
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "event_id": self._id_factory("event"),
                    "item_id": item_id,
                    "content_index": 0,
                    "delta": delta,
                },
                epoch=epoch,
            )
        ]
        return events

    def _map_turn_started(self, epoch: int) -> _Outbound:
        if epoch in self._turns:
            raise WorkerError("native worker repeated turn_started")
        turn = self._new_public_turn(epoch)
        if self._response_create_native_ids:
            self._clear_response_create()
        return _Outbound(
            {
                "type": "response.created",
                "event_id": self._id_factory("event"),
                "response": self._response(turn, status="in_progress"),
            },
            epoch=epoch,
        )

    def _new_public_turn(self, epoch: int) -> _Turn:
        turn = _Turn(epoch, self._id_factory("resp"), self._id_factory("item"))
        self._turns[epoch] = turn
        self._turn_history.append(turn)
        self._active_turn_epoch = epoch
        return turn

    def _map_agent_audio(self, event: JsonObject, epoch: int) -> _Outbound:
        turn = self._require_turn(epoch)
        audio = _worker_string(event, "audio")
        try:
            decoded = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise WorkerError("native worker returned invalid PCM16 audio") from exc
        if not decoded or len(decoded) % 2 != 0 or len(decoded) > self.config.max_audio_bytes:
            raise WorkerError("native worker returned invalid PCM16 audio")
        sample_rate = _optional_worker_integer(event, "sample_rate", minimum=1)
        if sample_rate not in (None, PCM16_SAMPLE_RATE):
            raise WorkerError("native worker returned an unsupported output sample rate")
        turn.audio_emitted = True
        return _Outbound(
            {
                "type": "response.output_audio.delta",
                "event_id": self._id_factory("event"),
                "response_id": turn.response_id,
                "item_id": turn.message_item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": audio,
            },
            epoch=epoch,
            stale_sensitive=True,
        )

    def _map_agent_text(self, event: JsonObject, epoch: int) -> list[_Outbound]:
        turn = self._require_turn(epoch)
        text = _optional_worker_string(event, "text") or ""
        final = _optional_worker_boolean(event, "is_final") or False
        if final:
            turn.text = text
            turn.text_completed = True
            return [
                _Outbound(
                    {
                        "type": "response.output_audio_transcript.done",
                        "event_id": self._id_factory("event"),
                        "response_id": turn.response_id,
                        "item_id": turn.message_item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "transcript": text,
                    },
                    epoch=epoch,
                    stale_sensitive=True,
                )
            ]
        if not text:
            return []
        turn.text += text
        events = [
            _Outbound(
                {
                    "type": "response.output_audio_transcript.delta",
                    "event_id": self._id_factory("event"),
                    "response_id": turn.response_id,
                    "item_id": turn.message_item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text,
                },
                epoch=epoch,
                stale_sensitive=True,
            )
        ]
        return events

    def _map_function_call(self, event: JsonObject, epoch: int) -> list[_Outbound]:
        turn = self._require_turn(epoch)
        text = _worker_string(event, "text")
        try:
            call = json.loads(text)
        except json.JSONDecodeError as exc:
            raise WorkerError("native worker returned an invalid function call") from exc
        if not isinstance(call, dict) or call.get("type") not in (None, "function_call"):
            raise WorkerError("native worker returned an invalid function call")
        call_id = call.get("call_id")
        name = call.get("name")
        arguments_value = call.get("arguments")
        if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
            raise WorkerError("native worker returned an invalid function call")
        if call_id in self._pending_calls:
            raise WorkerError("native worker repeated a function call ID")
        try:
            arguments = (
                arguments_value
                if isinstance(arguments_value, str)
                else json.dumps(arguments_value, separators=(",", ":"), sort_keys=True)
            )
            json.loads(arguments)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise WorkerError("native worker returned invalid function arguments") from exc
        item_id = self._id_factory("item")
        message_items = 1 if turn.audio_emitted or turn.text else 0
        output_index = message_items + len(turn.function_items)
        item: JsonObject = {
            "id": item_id,
            "object": "realtime.item",
            "type": "function_call",
            "status": "completed",
            "name": name,
            "call_id": call_id,
            "arguments": arguments,
        }
        turn.function_items.append(item)
        self._pending_calls[call_id] = _PendingCall(epoch, item_id, output_index)
        self._last_conversation_item_id = item_id
        common = {
            "response_id": turn.response_id,
            "item_id": item_id,
            "output_index": output_index,
        }
        events = [
            _Outbound(
                {
                    "type": "response.output_item.added",
                    "event_id": self._id_factory("event"),
                    **common,
                    "item": {**item, "status": "in_progress", "arguments": ""},
                },
                epoch=epoch,
                stale_sensitive=True,
            ),
            _Outbound(
                {
                    "type": "response.function_call_arguments.done",
                    "event_id": self._id_factory("event"),
                    **common,
                    "name": name,
                    "call_id": call_id,
                    "arguments": arguments,
                },
                epoch=epoch,
                stale_sensitive=True,
            ),
            _Outbound(
                {
                    "type": "response.output_item.done",
                    "event_id": self._id_factory("event"),
                    **common,
                    "item": item,
                },
                epoch=epoch,
                stale_sensitive=True,
            ),
        ]
        events.extend(self._completed_response_events(turn))
        return events

    def _map_turn_finished(self, epoch: int) -> list[_Outbound]:
        turn = self._require_turn(epoch)
        return self._completed_response_events(turn)

    def _completed_response_events(self, turn: _Turn) -> list[_Outbound]:
        if turn.done:
            return []
        events: list[_Outbound] = []
        if turn.text and not turn.text_completed:
            turn.text_completed = True
            events.append(
                _Outbound(
                    {
                        "type": "response.output_audio_transcript.done",
                        "event_id": self._id_factory("event"),
                        "response_id": turn.response_id,
                        "item_id": turn.message_item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "transcript": turn.text,
                    },
                    epoch=turn.epoch,
                    stale_sensitive=True,
                )
            )
        if turn.audio_emitted:
            events.append(
                _Outbound(
                    {
                        "type": "response.output_audio.done",
                        "event_id": self._id_factory("event"),
                        "response_id": turn.response_id,
                        "item_id": turn.message_item_id,
                        "output_index": 0,
                        "content_index": 0,
                    },
                    epoch=turn.epoch,
                    stale_sensitive=True,
                )
            )
        events.append(
            _Outbound(
                {
                    "type": "response.done",
                    "event_id": self._id_factory("event"),
                    "response": self._response(turn, status="completed"),
                },
                epoch=turn.epoch,
            )
        )
        if self._active_turn_epoch == turn.epoch:
            self._active_turn_epoch = None
        turn.done = True
        if turn.function_items:
            self._last_conversation_item_id = turn.function_items[-1]["id"]
        elif turn.audio_emitted or turn.text:
            self._last_conversation_item_id = turn.message_item_id
        return events

    def _map_interruption(
        self,
        kind: str,
        epoch: int,
        interrupted_turn: _Turn | None,
        event: JsonObject,
    ) -> list[_Outbound]:
        if interrupted_turn is None or interrupted_turn.epoch >= epoch:
            return []
        text = _optional_worker_string(event, "text") or ""
        reason = (
            "client_cancelled"
            if kind != "yielded" or text in {"response-cancel", "response-truncate"}
            else "turn_detected"
        )
        return self._cancelled_response_events(interrupted_turn, reason)

    def _response(
        self,
        turn: _Turn,
        *,
        status: str,
        status_details: JsonObject | None = None,
    ) -> JsonObject:
        content: list[JsonObject] = []
        if turn.audio_emitted or turn.text:
            content.append({"type": "audio", "transcript": turn.text})
        output: list[JsonObject] = []
        if content:
            output.append(
                {
                    "id": turn.message_item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "status": status,
                    "role": "assistant",
                    "content": content,
                }
            )
        output.extend(turn.function_items)
        return {
            "object": "realtime.response",
            "id": turn.response_id,
            "status": status,
            "status_details": status_details,
            "output": output,
        }

    def _active_turn(self) -> _Turn | None:
        if self._active_turn_epoch is None:
            return None
        return self._turns.get(self._active_turn_epoch)

    def _require_turn(self, epoch: int) -> _Turn:
        turn = self._turns.get(epoch)
        if turn is None:
            raise WorkerError("native worker emitted response data outside a turn")
        return turn

    def _session_snapshot(self) -> JsonObject:
        tools = self._tools if self._tool_choice != "none" else []
        return session_snapshot(
            self.session_id,
            instructions=self._instructions,
            tools=tools,
            tool_choice=self._tool_choice,
        )

    def _worker_error_event(self, worker_event: JsonObject | None = None) -> JsonObject:
        client_event_id = None
        if worker_event is not None:
            value = worker_event.get("event_id")
            if isinstance(value, str):
                client_event_id = value
        native_code = worker_event.get("code") if worker_event is not None else None
        if not isinstance(native_code, str):
            native_code = None
        public_code = {
            "unsupported": "unsupported_event",
            "stale_epoch": "stale_epoch",
            "invalid_state": "invalid_state",
            "invalid_message": "invalid_request",
        }.get(native_code, "worker_error")
        public_type = "invalid_request_error" if public_code != "worker_error" else "server_error"
        message = {
            "unsupported_event": "requested operation is not supported by this model session",
            "stale_epoch": "requested response is no longer current",
            "invalid_state": "requested operation is not valid in the current session state",
            "invalid_request": "native realtime request was rejected",
        }.get(public_code, "native realtime worker failed")
        return {
            "type": "error",
            "event_id": self._id_factory("event"),
            "error": {
                "type": public_type,
                "code": public_code,
                "message": message,
                "param": None,
                "event_id": client_event_id,
            },
        }

    def _put_output(self, outbound: _Outbound) -> None:
        try:
            self.output_queue.put_nowait(outbound)
        except asyncio.QueueFull as exc:
            raise QueueCapacityError("Realtime output queue capacity exceeded") from exc

    def _put_outputs(self, outbound: list[_Outbound]) -> None:
        if self.output_queue.qsize() + len(outbound) > self.output_queue.maxsize:
            raise QueueCapacityError("Realtime output queue capacity exceeded")
        for event in outbound:
            self.output_queue.put_nowait(event)

    def _is_stale(self, outbound: _Outbound) -> bool:
        return (
            outbound.stale_sensitive
            and outbound.epoch is not None
            and outbound.epoch < self.current_epoch
        )


class RealtimeHost:
    """Authenticate WebSockets and bind each one to an isolated worker."""

    def __init__(self, config: RealtimeServerConfig, worker_factory: WorkerFactory) -> None:
        self.config = config
        self.worker_factory = worker_factory

    def authorize_request(self, path: str, headers: Mapping[str, str]) -> str | None:
        """Return a safe rejection reason, or ``None`` when authorized."""

        split = urlsplit(path)
        if split.path != REALTIME_PATH or split.query or split.fragment:
            return "unsupported endpoint"
        authorization = _header(headers, "authorization")
        if authorization is None or not authorization.startswith("Bearer "):
            return "unauthorized"
        presented = authorization[len("Bearer ") :]
        if not presented or not hmac.compare_digest(presented, self.config.bearer_token):
            return "unauthorized"
        return None

    async def handle_connection(self, websocket: Any) -> None:
        path, headers = _websocket_request(websocket)
        rejection = self.authorize_request(path, headers)
        if rejection is not None:
            await websocket.close(code=1008, reason=rejection)
            return
        worker_value = self.worker_factory()
        worker = await worker_value if inspect.isawaitable(worker_value) else worker_value
        session = RealtimeSession(worker, self.config)
        try:
            await worker.start()
        except Exception:
            await _send_sanitized_worker_error(websocket)
            await worker.close()
            await websocket.close(code=1011, reason="worker unavailable")
            return
        tasks = {
            asyncio.create_task(self._read_client(websocket, session)),
            asyncio.create_task(self._process_client(session)),
            asyncio.create_task(self._write_worker(session)),
            asyncio.create_task(self._read_worker(session)),
            asyncio.create_task(self._write_client(websocket, session)),
        }
        try:
            active = set(tasks)
            while active:
                done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                should_stop = False
                for task in done:
                    exception = task.exception()
                    if exception is not None:
                        raise exception
                    if task.get_coro().__name__ in {"_read_client", "_read_worker"}:
                        should_stop = True
                if should_stop:
                    if session.worker_ended:
                        try:
                            await asyncio.wait_for(session.output_queue.join(), timeout=1.0)
                        except asyncio.TimeoutError:
                            pass
                    break
        except (WorkerError, QueueCapacityError):
            await _send_sanitized_worker_error(websocket)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await worker.close()
            if not getattr(websocket, "closed", False):
                await websocket.close(code=1000, reason="session ended")

    @staticmethod
    async def _read_client(websocket: Any, session: RealtimeSession) -> None:
        async for message in websocket:
            await session.input_queue.put(message)

    @staticmethod
    async def _process_client(session: RealtimeSession) -> None:
        while True:
            await session.process_next_input()

    @staticmethod
    async def _write_worker(session: RealtimeSession) -> None:
        while True:
            message = await session.message_queue.get()
            try:
                await session.worker.send(message)
            finally:
                session.message_queue.task_done()

    @staticmethod
    async def _read_worker(session: RealtimeSession) -> None:
        while True:
            message = await session.worker.receive()
            if message is None:
                session.worker_ended = True
                return
            session.handle_worker_message(message)
            if session.worker_ended:
                return

    @staticmethod
    async def _write_client(websocket: Any, session: RealtimeSession) -> None:
        while True:
            event = await session.next_output()
            await websocket.send(json.dumps(event, separators=(",", ":")))


async def serve(config: RealtimeServerConfig, worker_factory: WorkerFactory) -> None:
    """Serve until cancelled, importing the optional WebSocket dependency lazily."""

    try:
        from websockets.asyncio.server import serve as websocket_serve
    except ImportError as exc:  # pragma: no cover - exercised without the optional extra
        raise RuntimeError("Realtime serving requires the 'realtime' optional dependency") from exc
    host = RealtimeHost(config, worker_factory)

    async def process_request(connection: Any, request: Any) -> Any:
        rejection = host.authorize_request(request.path, request.headers)
        if rejection is None:
            return None
        status = (
            HTTPStatus.NOT_FOUND if rejection == "unsupported endpoint" else HTTPStatus.UNAUTHORIZED
        )
        return connection.respond(status, f"{rejection}\n")

    async with websocket_serve(
        host.handle_connection,
        config.host,
        config.port,
        compression=None,
        max_size=config.max_message_bytes,
        max_queue=config.input_queue_size,
        process_request=process_request,
    ):
        await asyncio.Future()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _header(headers: Mapping[str, str], name: str) -> str | None:
    expected = name.casefold()
    for key, value in headers.items():
        if key.casefold() == expected:
            return value
    return None


def _websocket_request(websocket: Any) -> tuple[str, Mapping[str, str]]:
    request = getattr(websocket, "request", None)
    if request is not None:
        return request.path, request.headers
    return getattr(websocket, "path", ""), getattr(websocket, "request_headers", {})


async def _send_sanitized_worker_error(websocket: Any) -> None:
    event = {
        "type": "error",
        "event_id": _new_id("event"),
        "error": {
            "type": "server_error",
            "code": "worker_error",
            "message": "native realtime worker failed",
            "param": None,
            "event_id": None,
        },
    }
    try:
        await websocket.send(json.dumps(event, separators=(",", ":")))
    except Exception:
        pass


def _worker_object(value: object) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise WorkerError("native worker event must be a JSON object")
    return value


def _worker_string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise WorkerError(f"native worker event requires string {key}")
    return result


def _optional_worker_string(value: Mapping[str, Any], key: str) -> str | None:
    if key not in value:
        return None
    result = value[key]
    if not isinstance(result, str):
        raise WorkerError(f"native worker event {key} must be a string")
    return result


def _worker_integer(value: Mapping[str, Any], key: str, *, minimum: int) -> int:
    result = value.get(key)
    if type(result) is not int or result < minimum:
        raise WorkerError(f"native worker event requires integer {key}")
    return result


def _optional_worker_integer(value: Mapping[str, Any], key: str, *, minimum: int) -> int | None:
    if key not in value:
        return None
    return _worker_integer(value, key, minimum=minimum)


def _optional_worker_boolean(value: Mapping[str, Any], key: str) -> bool | None:
    if key not in value:
        return None
    result = value[key]
    if type(result) is not bool:
        raise WorkerError(f"native worker event {key} must be a boolean")
    return result
