# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import json
from collections import deque
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.realtime.cli import build_parser
from tensorrt_model_connect.realtime.server import (
    QueueCapacityError,
    RealtimeHost,
    RealtimeServerConfig,
    RealtimeSession,
)
from tensorrt_model_connect.realtime.worker import WorkerError
from tensorrt_model_connect.realtime.worker import find_native_worker


class FakeWorker:
    def __init__(self, events=()) -> None:
        self.events = deque(events)
        self.sent = []
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def send(self, message) -> None:
        self.sent.append(message)

    async def receive(self):
        if self.events:
            event = self.events.popleft()
            if isinstance(event, BaseException):
                raise event
            return event
        return None

    async def close(self) -> None:
        self.closed = True


class FakeWebsocket:
    def __init__(self, path: str, authorization: str) -> None:
        self.request = SimpleNamespace(
            path=path,
            headers={"Authorization": authorization},
        )
        self.sent = []
        self.close_calls = []
        self.closed = False

    async def send(self, message) -> None:
        self.sent.append(message)

    async def close(self, **kwargs) -> None:
        self.close_calls.append(kwargs)
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class SequentialIds:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self, prefix: str) -> str:
        self.value += 1
        return f"{prefix}_{self.value}"


def make_session(**config_overrides):
    worker = FakeWorker()
    config = RealtimeServerConfig(bearer_token="test-secret", **config_overrides)
    return RealtimeSession(worker, config, id_factory=SequentialIds()), worker


def send_client(session: RealtimeSession, event: dict) -> None:
    asyncio.run(session.handle_client_message(json.dumps(event)))


def configure_session(session: RealtimeSession) -> None:
    send_client(session, {"type": "session.update", "session": {}})
    session.drain_worker_messages()


def native_event(kind: str, epoch: int, sequence: int, **fields):
    return {
        "type": "session.event",
        "kind": kind,
        "epoch": epoch,
        "sequence": sequence,
        **fields,
    }


def test_golden_full_duplex_trace_and_function_output_forwarding() -> None:
    session, _worker = make_session()
    pcm = base64.b64encode(b"\x01\x00\xff\x7f").decode("ascii")

    session.handle_worker_message({"type": "session.ready"})
    created = session.drain_output()[0]
    assert created["type"] == "session.created"
    assert created["session"]["audio"]["input"]["format"] == {
        "type": "audio/pcm",
        "rate": 24_000,
    }
    assert created["session"]["audio"]["input"]["turn_detection"] == {
        "type": "server_vad",
        "create_response": True,
        "interrupt_response": True,
    }

    send_client(
        session,
        {
            "event_id": "client_update",
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": "Answer briefly.",
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24_000},
                        "turn_detection": {
                            "type": "server_vad",
                            "create_response": True,
                            "interrupt_response": True,
                        },
                    },
                    "output": {"format": {"type": "audio/pcm", "rate": 24_000}},
                },
                "tools": [
                    {
                        "type": "function",
                        "name": "weather",
                        "description": "Return the weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ],
                "tool_choice": "auto",
            },
        },
    )
    update = session.drain_worker_messages()[0]
    assert update == {
        "type": "session.update",
        "event_id": "client_update",
        "input_sample_rate": 24_000,
        "output_sample_rate": 24_000,
        "instructions": "Answer briefly.",
        "tools": [
            {
                "type": "function",
                "name": "weather",
                "description": "Return the weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        "on_hold_messages": {},
    }
    session.handle_worker_message({"type": "session.updated"})
    assert session.drain_output()[0]["type"] == "session.updated"

    send_client(
        session,
        {"event_id": "audio_1", "type": "input_audio_buffer.append", "audio": pcm},
    )
    assert session.drain_worker_messages() == [
        {
            "type": "input_audio_buffer.append",
            "event_id": "audio_1",
            "audio": pcm,
        }
    ]

    session.handle_worker_message(
        native_event(
            "user_speech_started",
            1,
            1,
            sample_rate=24_000,
            media_start_sample=2_400,
        )
    )
    session.handle_worker_message(native_event("user_transcript", 1, 2, text="hello"))
    session.handle_worker_message(native_event("user_transcript", 1, 3, text="hello world"))
    session.handle_worker_message(
        native_event("user_transcript", 1, 4, text="hello world", is_final=True)
    )
    session.handle_worker_message(
        native_event(
            "user_speech_stopped",
            1,
            5,
            sample_rate=24_000,
            media_end_sample=14_400,
        )
    )
    session.handle_worker_message(native_event("turn_started", 2, 1))
    session.handle_worker_message(native_event("agent_audio", 2, 2, sample_rate=24_000, audio=pcm))
    session.handle_worker_message(native_event("agent_text", 2, 3, text="It is "))
    session.handle_worker_message(
        native_event(
            "function_call",
            2,
            4,
            text=json.dumps(
                {
                    "type": "function_call",
                    "call_id": "call_weather",
                    "name": "weather",
                    "arguments": {"city": "Seattle"},
                }
            ),
            is_final=True,
        )
    )

    send_client(
        session,
        {
            "event_id": "tool_result",
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": "call_weather",
                "output": '{"temperature":18}',
            },
        },
    )
    assert session.drain_worker_messages() == []
    send_client(session, {"event_id": "tool_resume", "type": "response.create"})
    tool_resume = session.drain_worker_messages()
    assert len(tool_resume) == 1
    assert tool_resume[0] == {
        "type": "conversation.item.create",
        "event_id": tool_resume[0]["event_id"],
        "epoch": 2,
        "call_id": "call_weather",
        "output": '{"temperature":18}',
    }
    assert tool_resume[0]["event_id"].startswith("native_")
    session.handle_worker_message(native_event("function_response_finished", 2, 5))
    session.handle_worker_message(native_event("agent_audio", 2, 6, sample_rate=24_000, audio=pcm))
    session.handle_worker_message(
        native_event("agent_text", 2, 7, text="It is 18 degrees.", is_final=True)
    )
    session.handle_worker_message(native_event("turn_finished", 2, 8))

    events = session.drain_output()
    types = [event["type"] for event in events]
    assert types == [
        "input_audio_buffer.speech_started",
        "conversation.item.input_audio_transcription.delta",
        "conversation.item.input_audio_transcription.delta",
        "conversation.item.input_audio_transcription.completed",
        "input_audio_buffer.speech_stopped",
        "response.created",
        "response.output_audio.delta",
        "response.output_audio_transcript.delta",
        "response.output_item.added",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.output_audio_transcript.done",
        "response.output_audio.done",
        "response.done",
        "conversation.item.added",
        "conversation.item.done",
        "response.created",
        "response.output_audio.delta",
        "response.output_audio_transcript.done",
        "response.output_audio.done",
        "response.done",
    ]
    assert events[0]["audio_start_ms"] == 100
    assert events[2]["delta"] == " world"
    assert events[4]["audio_end_ms"] == 600
    assert events[6]["delta"] == pcm
    assert events[9]["arguments"] == '{"city":"Seattle"}'
    first_response_id = events[5]["response"]["id"]
    resumed_response_id = events[16]["response"]["id"]
    assert first_response_id != resumed_response_id
    assert events[13]["response"]["id"] == first_response_id
    assert events[13]["response"]["status"] == "completed"
    assert events[13]["response"]["output"][1]["call_id"] == "call_weather"
    assert events[17]["response_id"] == resumed_response_id
    assert events[18]["response_id"] == resumed_response_id
    assert events[19]["response_id"] == resumed_response_id
    assert events[-1]["response"]["id"] == resumed_response_id
    assert events[-1]["response"]["status"] == "completed"


def test_tool_output_is_buffered_until_one_response_create_and_never_double_triggers() -> None:
    session, _worker = make_session()
    session.handle_worker_message(native_event("turn_started", 1, 1))
    session.drain_output()
    session.handle_worker_message(
        native_event(
            "function_call",
            1,
            2,
            text='{"call_id":"call_1","name":"lookup","arguments":{}}',
        )
    )
    function_response = session.drain_output()
    assert [event["type"] for event in function_response] == [
        "response.output_item.added",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.done",
    ]
    assert function_response[1]["output_index"] == 0
    assert function_response[-1]["response"]["output"][0]["call_id"] == "call_1"

    send_client(
        session,
        {
            "event_id": "output_1",
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
        },
    )
    assert session.drain_worker_messages() == []
    buffered = session.drain_output()
    assert [event["type"] for event in buffered] == [
        "conversation.item.added",
        "conversation.item.done",
    ]
    assert buffered[-1]["item"]["call_id"] == "call_1"
    assert buffered[0]["previous_item_id"] == function_response[2]["item"]["id"]

    send_client(session, {"event_id": "resume_1", "type": "response.create"})
    resume = session.drain_worker_messages()
    assert len(resume) == 1
    assert resume[0]["type"] == "conversation.item.create"
    assert resume[0]["call_id"] == "call_1"
    assert not any(message["type"] == "response.create" for message in resume)
    continuation = session.drain_output()
    assert [event["type"] for event in continuation] == ["response.created"]

    send_client(session, {"event_id": "resume_2", "type": "response.create"})
    assert session.drain_worker_messages() == []
    duplicate = session.drain_output()[0]
    assert duplicate["error"]["code"] == "invalid_state"
    assert duplicate["error"]["event_id"] == "resume_2"

    session.handle_worker_message(native_event("function_response_finished", 1, 3))
    assert session.drain_output() == []


def test_buffered_tool_validation_error_correlates_to_the_triggering_response_create() -> None:
    session, _worker = make_session()
    session.handle_worker_message(native_event("turn_started", 1, 1))
    session.drain_output()
    session.handle_worker_message(
        native_event(
            "function_call",
            1,
            2,
            text='{"call_id":"call_1","name":"lookup","arguments":{}}',
        )
    )
    session.drain_output()
    send_client(
        session,
        {
            "event_id": "output_accepted",
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
        },
    )
    assert [event["type"] for event in session.drain_output()] == [
        "conversation.item.added",
        "conversation.item.done",
    ]

    # The public item is already buffered. Native validation happens only when
    # response.create atomically submits and resumes it, so errors correlate to
    # that response.create event rather than the earlier item event.
    send_client(session, {"event_id": "resume_validation", "type": "response.create"})
    resume = session.drain_worker_messages()[0]
    created = session.drain_output()[0]
    assert created["type"] == "response.created"
    failed_response_id = created["response"]["id"]
    session.handle_worker_message(
        {
            "type": "error",
            "code": "session_error",
            "message": "private native detail",
            "event_id": resume["event_id"],
        }
    )
    failure = session.drain_output()
    assert [event["type"] for event in failure] == ["error", "response.done"]
    error = failure[0]
    assert error["error"]["event_id"] == "resume_validation"
    assert "output_accepted" not in json.dumps(error)
    assert "private native detail" not in json.dumps(error)
    assert failure[-1]["response"]["id"] == failed_response_id
    assert failure[-1]["response"]["status"] == "failed"


def test_response_create_error_is_correlated_without_a_synthetic_success() -> None:
    session, _worker = make_session()
    send_client(session, {"event_id": "create_1", "type": "response.create"})
    create = session.drain_worker_messages()[0]
    assert create["type"] == "response.create"
    assert session.drain_output() == []

    session.handle_worker_message(
        {
            "type": "error",
            "code": "invalid_state",
            "message": "native state detail",
            "event_id": create["event_id"],
        }
    )
    error = session.drain_output()[0]
    assert error["type"] == "error"
    assert error["error"]["event_id"] == "create_1"
    assert "native state detail" not in json.dumps(error)
    assert session.drain_output() == []


def test_official_commit_then_response_create_waits_for_native_success() -> None:
    session, _worker = make_session()
    configure_session(session)
    pcm = base64.b64encode(b"\x01\x00\x02\x00").decode("ascii")
    send_client(session, {"type": "input_audio_buffer.append", "audio": pcm})
    session.drain_worker_messages()

    send_client(session, {"event_id": "commit_1", "type": "input_audio_buffer.commit"})
    send_client(session, {"event_id": "create_1", "type": "response.create"})
    commit, create = session.drain_worker_messages()
    assert commit["type"] == "input_audio_buffer.commit"
    assert set(commit) == {"type", "event_id"}
    assert create["type"] == "response.create"
    assert set(create) == {"type", "event_id"}
    assert session.drain_output() == []

    session.handle_worker_message(
        {"type": "input_audio_buffer.committed", "event_id": commit["event_id"]}
    )
    committed = session.drain_output()[0]
    assert committed["type"] == "input_audio_buffer.committed"
    assert committed["previous_item_id"] is None
    assert committed["item_id"].startswith("item_")

    session.handle_worker_message(native_event("turn_started", 1, 1))
    created = session.drain_output()
    assert [event["type"] for event in created] == ["response.created"]


def test_clear_and_control_errors_never_fabricate_success() -> None:
    session, _worker = make_session()
    configure_session(session)
    send_client(session, {"event_id": "clear_1", "type": "input_audio_buffer.clear"})
    clear = session.drain_worker_messages()[0]
    assert clear["type"] == "input_audio_buffer.clear"
    assert session.drain_output() == []

    session.handle_worker_message(
        {
            "type": "error",
            "code": "unsupported",
            "message": "private native detail",
            "event_id": clear["event_id"],
        }
    )
    error = session.drain_output()[0]
    assert error["error"]["event_id"] == "clear_1"
    assert error["error"]["code"] == "unsupported_event"
    assert "private native detail" not in json.dumps(error)
    assert session.drain_output() == []
    with pytest.raises(WorkerError, match="acknowledgement"):
        session.handle_worker_message(
            {"type": "input_audio_buffer.cleared", "event_id": clear["event_id"]}
        )


def test_clear_ack_maps_to_the_official_public_event() -> None:
    session, _worker = make_session()
    configure_session(session)
    send_client(session, {"event_id": "clear_1", "type": "input_audio_buffer.clear"})
    clear = session.drain_worker_messages()[0]
    assert session.drain_output() == []
    session.handle_worker_message(
        {"type": "input_audio_buffer.cleared", "event_id": clear["event_id"]}
    )
    assert session.drain_output()[0]["type"] == "input_audio_buffer.cleared"


def test_cancel_ack_is_the_public_cancellation_authority_and_purges_audio() -> None:
    session, _worker = make_session()
    pcm = base64.b64encode(b"\x00\x00").decode("ascii")
    session.handle_worker_message(native_event("turn_started", 1, 1))
    response_id = session.drain_output()[0]["response"]["id"]
    session.handle_worker_message(native_event("agent_audio", 1, 2, sample_rate=24_000, audio=pcm))

    send_client(
        session,
        {"event_id": "cancel_1", "type": "response.cancel", "response_id": response_id},
    )
    cancel = session.drain_worker_messages()[0]
    assert cancel["type"] == "response.cancel"
    session.handle_worker_message({"type": "response.cancelled", "event_id": cancel["event_id"]})
    events = session.drain_output()
    assert [event["type"] for event in events] == [
        "response.output_audio.done",
        "response.done",
    ]
    assert events[-1]["response"]["status"] == "cancelled"
    assert events[-1]["response"]["status_details"]["reason"] == "client_cancelled"

    session.handle_worker_message(native_event("agent_audio", 1, 3, sample_rate=24_000, audio=pcm))
    assert session.drain_output() == []
    session.handle_worker_message(native_event("yielded", 2, 1, text="response-cancel"))
    assert session.drain_output() == []


def test_response_cancel_requires_an_active_public_response() -> None:
    session, _worker = make_session()
    send_client(session, {"event_id": "cancel_1", "type": "response.cancel"})
    assert session.drain_worker_messages() == []
    assert session.drain_output()[0]["error"]["code"] == "invalid_response_id"


def test_truncate_ack_maps_samples_back_to_the_official_item_event() -> None:
    session, _worker = make_session()
    pcm = base64.b64encode(b"\x00\x00").decode("ascii")
    session.handle_worker_message(native_event("turn_started", 3, 1))
    session.drain_output()
    session.handle_worker_message(native_event("agent_audio", 3, 2, sample_rate=24_000, audio=pcm))
    item_id = session.drain_output()[0]["item_id"]

    send_client(
        session,
        {
            "event_id": "truncate_1",
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": 0,
            "audio_end_ms": 125,
        },
    )
    truncate = session.drain_worker_messages()[0]
    assert truncate["type"] == "conversation.item.truncate"
    assert truncate["epoch"] == 3
    assert truncate["played_output_samples"] == 3_000
    assert session.drain_output() == []

    session.handle_worker_message(
        {
            "type": "conversation.item.truncated",
            "event_id": truncate["event_id"],
            "epoch": 3,
            "played_output_samples": 3_000,
        }
    )
    public = session.drain_output()[0]
    assert public == {
        "type": "conversation.item.truncated",
        "event_id": public["event_id"],
        "item_id": item_id,
        "content_index": 0,
        "audio_end_ms": 125,
    }

    session.handle_worker_message(native_event("yielded", 4, 1, text="response-truncate"))
    truncated_completion = session.drain_output()
    assert [event["type"] for event in truncated_completion] == [
        "response.output_audio.done",
        "response.done",
    ]
    done = truncated_completion[-1]
    assert done["type"] == "response.done"
    assert done["response"]["status_details"]["reason"] == "client_cancelled"


@pytest.mark.parametrize(
    ("event", "code"),
    [
        (
            {
                "type": "session.update",
                "session": {
                    "audio": {
                        "input": {
                            "turn_detection": {
                                "type": "server_vad",
                                "create_response": False,
                                "interrupt_response": True,
                            }
                        }
                    }
                },
            },
            "unsupported_value",
        ),
        ({"type": "input_audio_buffer.commit", "create_response": True}, "unsupported_value"),
        ({"type": "input_audio_buffer.clear", "unexpected": True}, "unsupported_value"),
        ({"type": "response.cancel", "response_id": ""}, "invalid_value"),
        (
            {
                "type": "conversation.item.truncate",
                "item_id": "item",
                "content_index": 1,
                "audio_end_ms": 10,
            },
            "invalid_value",
        ),
        (
            {
                "type": "conversation.item.truncate",
                "item_id": "item",
                "content_index": 0,
                "audio_end_ms": -1,
            },
            "invalid_value",
        ),
        (
            {"type": "response.create", "response": {"instructions": "override"}},
            "unsupported_value",
        ),
    ],
)
def test_invalid_realtime_controls_fail_before_native_dispatch(event: dict, code: str) -> None:
    session, _worker = make_session()
    event["event_id"] = "bad_control"
    send_client(session, event)

    assert session.drain_worker_messages() == []
    error = session.drain_output()[0]
    assert error["error"]["code"] == code
    assert error["error"]["event_id"] == "bad_control"
    assert not session.close_requested


def test_control_targets_must_match_public_response_objects() -> None:
    session, _worker = make_session()
    session.handle_worker_message(native_event("turn_started", 4, 1))
    response_id = session.drain_output()[0]["response"]["id"]

    send_client(
        session,
        {"event_id": "wrong_response", "type": "response.cancel", "response_id": "other"},
    )
    error = session.drain_output()[0]
    assert error["error"]["code"] == "invalid_response_id"

    send_client(
        session,
        {
            "event_id": "not_audio",
            "type": "conversation.item.truncate",
            "item_id": response_id,
            "content_index": 0,
            "audio_end_ms": 0,
        },
    )
    error = session.drain_output()[0]
    assert error["error"]["code"] == "invalid_item_id"
    assert session.drain_worker_messages() == []


def test_only_function_outputs_are_accepted_as_conversation_items() -> None:
    session, _worker = make_session()
    send_client(
        session,
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "run this"}],
            },
        },
    )
    error = session.drain_output()[0]
    assert error["type"] == "error"
    assert error["error"]["code"] == "unsupported_value"
    assert session.drain_worker_messages() == []


def test_function_output_requires_a_live_native_call() -> None:
    session, _worker = make_session()
    send_client(
        session,
        {
            "event_id": "bad_call",
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": "missing", "output": "{}"},
        },
    )
    error = session.drain_output()[0]
    assert error["error"]["code"] == "invalid_call_id"
    assert error["error"]["event_id"] == "bad_call"
    assert session.drain_worker_messages() == []


def test_session_close_is_forwarded_without_claiming_completion() -> None:
    session, _worker = make_session()
    send_client(session, {"event_id": "close_1", "type": "session.close"})
    assert session.close_requested
    assert session.drain_worker_messages() == [{"type": "session.close", "event_id": "close_1"}]
    assert session.drain_output() == []


def test_session_state_errors_are_explicit_before_native_dispatch() -> None:
    session, _worker = make_session()
    pcm = base64.b64encode(b"\x00\x00").decode("ascii")
    send_client(session, {"event_id": "early", "type": "input_audio_buffer.append", "audio": pcm})
    assert session.drain_output()[0]["error"]["code"] == "invalid_state"

    configure_session(session)
    send_client(session, {"type": "input_audio_buffer.append", "audio": pcm})
    session.drain_worker_messages()
    send_client(session, {"event_id": "late", "type": "session.update", "session": {}})
    assert session.drain_output()[0]["error"]["code"] == "invalid_state"

    send_client(session, {"type": "session.close"})
    session.drain_worker_messages()
    send_client(session, {"event_id": "closed", "type": "session.update", "session": {}})
    assert session.drain_output()[0]["error"]["code"] == "invalid_state"


def test_auth_is_exact_and_token_is_not_rendered() -> None:
    worker_calls = []
    config = RealtimeServerConfig(bearer_token="do-not-print-this")
    host = RealtimeHost(config, lambda: worker_calls.append(True) or FakeWorker())

    assert config.host == "127.0.0.1"
    assert "do-not-print-this" not in repr(config)
    assert host.authorize_request("/v1/realtime", {"Authorization": "Bearer wrong"}) == (
        "unauthorized"
    )
    assert (
        host.authorize_request("/v1/realtime", {"authorization": "Bearer do-not-print-this"})
        is None
    )
    assert (
        host.authorize_request(
            "/v1/realtime?model=other", {"Authorization": "Bearer do-not-print-this"}
        )
        == "unsupported endpoint"
    )
    assert (
        host.authorize_request("/other", {"Authorization": "Bearer do-not-print-this"})
        == "unsupported endpoint"
    )
    assert worker_calls == []


def test_unauthorized_connection_never_creates_a_worker() -> None:
    worker_calls = []
    host = RealtimeHost(
        RealtimeServerConfig(bearer_token="right"),
        lambda: worker_calls.append(True) or FakeWorker(),
    )
    websocket = FakeWebsocket("/v1/realtime", "Bearer wrong")

    asyncio.run(host.handle_connection(websocket))

    assert worker_calls == []
    assert websocket.sent == []
    assert websocket.close_calls == [{"code": 1008, "reason": "unauthorized"}]


def test_worker_start_failure_is_sanitized() -> None:
    class StartFailure(FakeWorker):
        async def start(self) -> None:
            raise WorkerError("sensitive process details")

    worker = StartFailure()
    host = RealtimeHost(RealtimeServerConfig(bearer_token="right"), lambda: worker)
    websocket = FakeWebsocket("/v1/realtime", "Bearer right")

    asyncio.run(host.handle_connection(websocket))

    assert worker.closed
    assert len(websocket.sent) == 1
    assert "native realtime worker failed" in websocket.sent[0]
    assert "sensitive process details" not in websocket.sent[0]
    assert websocket.close_calls == [{"code": 1011, "reason": "worker unavailable"}]


def test_cli_defaults_to_loopback_and_reads_a_named_token_environment() -> None:
    arguments = build_parser().parse_args(
        [
            "--bundle",
            "/model.bundle",
            "--backend-dir",
            "/backends/one",
            "--backend-dir",
            "/backends/two",
            "--model-plugin-dir",
            "/plugins",
        ]
    )
    assert arguments.host == "127.0.0.1"
    assert arguments.worker is None
    assert arguments.token_env == "TRTMC_REALTIME_TOKEN"
    assert not hasattr(arguments, "token")
    assert [str(path) for path in arguments.backend_dir] == ["/backends/one", "/backends/two"]
    assert [str(path) for path in arguments.model_plugin_dir] == ["/plugins"]


def test_explicit_native_worker_discovery(tmp_path) -> None:
    worker = tmp_path / "trtmc_realtime_worker"
    worker.write_text("#!/bin/sh\n", encoding="utf-8")
    worker.chmod(0o700)
    assert find_native_worker(worker) == worker.resolve()


def test_input_message_and_output_queues_are_bounded() -> None:
    input_session, _ = make_session(input_queue_size=1)
    input_session.enqueue_input("{}")
    with pytest.raises(QueueCapacityError, match="input queue"):
        input_session.enqueue_input("{}")

    message_session, _ = make_session(message_queue_size=1)
    pcm = base64.b64encode(b"\x00\x00").decode("ascii")
    configure_session(message_session)
    send_client(message_session, {"type": "input_audio_buffer.append", "audio": pcm})
    with pytest.raises(QueueCapacityError, match="message queue"):
        send_client(message_session, {"type": "input_audio_buffer.append", "audio": pcm})

    output_session, _ = make_session(output_queue_size=1)
    output_session.handle_worker_message({"type": "session.ready"})
    with pytest.raises(QueueCapacityError, match="output queue"):
        output_session.handle_worker_message({"type": "session.updated"})

    assert input_session.input_queue.maxsize == 1
    assert message_session.message_queue.maxsize == 1
    assert output_session.output_queue.maxsize == 1


def test_audio_and_message_bounds_fail_without_reaching_worker() -> None:
    session, _worker = make_session(max_audio_bytes=2, max_message_bytes=96)
    configure_session(session)
    oversized_audio = base64.b64encode(b"\x00\x00\x01\x00").decode("ascii")
    send_client(
        session,
        {"event_id": "large_audio", "type": "input_audio_buffer.append", "audio": oversized_audio},
    )
    audio_error = session.drain_output()[0]
    assert audio_error["error"]["code"] == "audio_too_large"
    assert audio_error["error"]["event_id"] == "large_audio"

    asyncio.run(session.handle_client_message(" " * 97))
    message_error = session.drain_output()[0]
    assert message_error["error"]["code"] == "message_too_large"
    assert session.drain_worker_messages() == []


def test_noncanonical_or_misaligned_pcm16_is_rejected() -> None:
    session, _worker = make_session()
    configure_session(session)
    send_client(
        session,
        {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(b"\x00").decode("ascii"),
        },
    )
    assert session.drain_output()[0]["error"]["code"] == "invalid_value"
    asyncio.run(session.handle_client_message(b"{}"))
    assert session.drain_output()[0]["error"]["code"] == "invalid_event"
    assert session.drain_worker_messages() == []


def test_new_epoch_purges_queued_audio_and_emits_cancelled_done() -> None:
    session, _worker = make_session()
    pcm = base64.b64encode(b"\x00\x00").decode("ascii")
    session.handle_worker_message(native_event("turn_started", 1, 1))
    assert session.drain_output()[0]["type"] == "response.created"
    session.handle_worker_message(native_event("agent_audio", 1, 2, sample_rate=24_000, audio=pcm))
    assert session.output_queue.qsize() == 1

    session.handle_worker_message(native_event("yielded", 2, 1, text="barge-in"))
    events = session.drain_output()
    assert [event["type"] for event in events] == [
        "response.output_audio.done",
        "response.done",
    ]
    assert events[-1]["response"]["status"] == "cancelled"
    assert events[-1]["response"]["status_details"]["reason"] == "turn_detected"

    # A late native event from the invalidated epoch is ignored.
    session.handle_worker_message(native_event("agent_audio", 1, 3, sample_rate=24_000, audio=pcm))
    assert session.drain_output() == []


def test_stale_function_call_cannot_receive_a_result() -> None:
    session, _worker = make_session()
    session.handle_worker_message(native_event("turn_started", 1, 1))
    session.handle_worker_message(
        native_event(
            "function_call",
            1,
            2,
            text='{"call_id":"old","name":"lookup","arguments":{}}',
        )
    )
    session.handle_worker_message(native_event("yielded", 2, 1))
    session.drain_output()

    send_client(
        session,
        {
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": "old", "output": "{}"},
        },
    )
    assert session.drain_output()[0]["error"]["code"] == "invalid_call_id"
    assert session.drain_worker_messages() == []


def test_worker_errors_are_sanitized_and_contract_violations_fail_closed() -> None:
    session, _worker = make_session()
    session.handle_worker_message(
        {"type": "error", "message": "secret token and /private/path must not escape"}
    )
    rendered = json.dumps(session.drain_output()[0])
    assert "native realtime worker failed" in rendered
    assert "secret token" not in rendered
    assert "/private/path" not in rendered

    with pytest.raises(WorkerError, match="unsupported event type"):
        session.handle_worker_message({"type": "pretend.success"})

    session.handle_worker_failure()
    assert session.drain_output()[0]["error"]["code"] == "worker_error"


def test_worker_sequence_must_increase_within_an_epoch() -> None:
    session, _worker = make_session()
    session.handle_worker_message(native_event("user_transcript", 1, 2, text="a"))
    with pytest.raises(WorkerError, match="sequence"):
        session.handle_worker_message(native_event("user_transcript", 1, 2, text="b"))
