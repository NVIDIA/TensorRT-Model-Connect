# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Realtime-style transcription WebSocket session handling."""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import uuid
from collections.abc import Mapping
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from .errors import (
    ModelNotFoundError,
    ServeError,
    WorkerError,
    WorkerRemoteError,
    WorkerRequestTooLargeError,
    WorkerSaturatedError,
)
from .protocol import extract_text, invalid_request_message, public_worker_error_message
from .registry import ModelRegistry
from .worker import WorkerSession


_LOGGER = logging.getLogger(__name__)
_TIMEOUT_ERRORS = (TimeoutError, asyncio.TimeoutError)


class RealtimeTranscriptionConnection:
    """Translate browser realtime events into one exclusive native stream."""

    def __init__(
        self,
        websocket: WebSocket,
        registry: ModelRegistry,
        *,
        max_audio_chunk_bytes: int = 1024 * 1024,
        max_session_audio_bytes: int = 512 * 1024 * 1024,
        idle_timeout_seconds: float = 30.0,
        max_session_seconds: float = 4 * 60 * 60,
    ) -> None:
        self.websocket = websocket
        self.registry = registry
        self.max_audio_chunk_bytes = max_audio_chunk_bytes
        self.max_session_audio_bytes = max_session_audio_bytes
        self.idle_timeout_seconds = idle_timeout_seconds
        self.max_session_seconds = max_session_seconds
        self.connection_id = f"sess_{uuid.uuid4().hex}"
        self.item_id = f"item_{uuid.uuid4().hex}"
        self.model = registry.default_transcription_model
        self.language: str | None = None
        self.sample_rate_hz = 24000
        self.channels = 1
        self.transcript = ""
        self.total_audio_bytes = 0
        self._lease: WorkerSession | None = None
        self._stop_requested = False

    async def run(self) -> None:
        await self._send(
            {
                "type": "session.created",
                "event_id": self._event_id(),
                "session": self._session_payload(),
            }
        )
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        try:
            while True:
                elapsed = loop.time() - started_at
                remaining = self.max_session_seconds - elapsed
                if remaining <= 0:
                    await self._send_failure(
                        "session_duration_exceeded",
                        f"realtime session exceeded {self.max_session_seconds:g}s",
                        error_type="invalid_request_error",
                    )
                    break
                try:
                    event = await asyncio.wait_for(
                        self.websocket.receive_json(),
                        timeout=min(self.idle_timeout_seconds, remaining),
                    )
                except _TIMEOUT_ERRORS:
                    elapsed = loop.time() - started_at
                    if elapsed >= self.max_session_seconds:
                        code = "session_duration_exceeded"
                        message = f"realtime session exceeded {self.max_session_seconds:g}s"
                    else:
                        code = "session_idle_timeout"
                        message = f"no realtime event received for {self.idle_timeout_seconds:g}s"
                    await self._send_failure(
                        code,
                        message,
                        error_type="invalid_request_error",
                    )
                    break
                if not isinstance(event, Mapping):
                    await self._send_error("invalid_event", "event must be a JSON object")
                    continue
                await self._dispatch(event)
                if self._stop_requested:
                    break
        except WebSocketDisconnect:
            pass
        except (ValueError, TypeError) as exc:
            await self._send_error("invalid_event", str(exc))
        except WorkerError as exc:
            await self._send_failed(exc)
        except ServeError as exc:
            await self._send_failure(exc.code, str(exc))
        except Exception:
            await self._send_failure("internal_error", "realtime transcription session failed")
        finally:
            try:
                await self._release_stream(reset=True)
            except asyncio.CancelledError:
                # Client shutdown may cancel the endpoint while the native
                # reset is being joined. Cleanup is complete at this point, so
                # let the WebSocket handler return normally.
                pass

    async def _dispatch(self, event: Mapping[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "session.update":
            await self._update_session(event)
        elif event_type == "input_audio_buffer.append":
            await self._append(event)
        elif event_type == "input_audio_buffer.commit":
            await self._commit()
        elif event_type == "input_audio_buffer.clear":
            await self._clear()
        else:
            await self._send_error(
                "unsupported_event",
                f"unsupported realtime event type: {event_type!r}",
            )

    async def _update_session(self, event: Mapping[str, Any]) -> None:
        raw_session = event.get("session")
        if not isinstance(raw_session, Mapping):
            await self._send_error("invalid_session", "session.update requires session object")
            return

        transcription = raw_session.get("input_audio_transcription")
        transcription_options = transcription if isinstance(transcription, Mapping) else {}
        requested_model = raw_session.get("model", transcription_options.get("model"))
        if requested_model is not None and not isinstance(requested_model, str):
            await self._send_error("invalid_model", "session model must be a string")
            return
        next_model = requested_model or self.model
        try:
            spec = self.registry.resolve_model("transcription", next_model)
        except ServeError as exc:
            await self._send_error(exc.code, str(exc))
            return

        audio_format = raw_session.get("input_audio_format", "pcm16")
        if isinstance(audio_format, Mapping):
            format_name = audio_format.get("type", "pcm16")
            format_rate = audio_format.get("rate")
        else:
            format_name = audio_format
            format_rate = None
        if str(format_name).lower() not in {"pcm16", "audio/pcm", "audio/pcm16"}:
            await self._send_error(
                "unsupported_audio_format", "only little-endian PCM16 audio is supported"
            )
            return

        trtmc_options = raw_session.get("trtmc")
        if not isinstance(trtmc_options, Mapping):
            trtmc_options = {}
        sample_rate = raw_session.get(
            "sample_rate_hz",
            trtmc_options.get("sample_rate_hz", format_rate or self.sample_rate_hz),
        )
        if not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 192000:
            await self._send_error(
                "invalid_sample_rate", "sample_rate_hz must be an integer from 8000 to 192000"
            )
            return
        language = raw_session.get("language", transcription_options.get("language"))
        if language is not None and not isinstance(language, str):
            await self._send_error("invalid_language", "language must be a string")
            return
        if isinstance(language, str) and len(language.encode("utf-8")) > 128:
            await self._send_error("invalid_language", "language exceeds 128 UTF-8 bytes")
            return

        if self._lease is not None and (
            spec.name != self.model
            or sample_rate != self.sample_rate_hz
            or language != self.language
        ):
            await self._release_stream(reset=True)
        self.model = spec.name
        self.sample_rate_hz = sample_rate
        self.language = language
        await self._send(
            {
                "type": "session.updated",
                "event_id": self._event_id(),
                "session": self._session_payload(),
            }
        )

    async def _append(self, event: Mapping[str, Any]) -> None:
        encoded = event.get("audio")
        if not isinstance(encoded, str) or not encoded:
            await self._send_error("invalid_audio", "append requires non-empty base64 audio")
            return
        try:
            audio = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            await self._send_error("invalid_audio", "audio is not valid base64")
            return
        if len(audio) > self.max_audio_chunk_bytes:
            await self._send_error(
                "audio_chunk_too_large",
                f"decoded audio chunk exceeds {self.max_audio_chunk_bytes} bytes",
            )
            return
        if len(audio) % 2:
            await self._send_error(
                "invalid_audio", "PCM16 audio must contain an even number of bytes"
            )
            return
        if self.total_audio_bytes + len(audio) > self.max_session_audio_bytes:
            await self._send_failure(
                "audio_session_too_large",
                f"realtime session audio exceeds {self.max_session_audio_bytes} bytes",
                error_type="invalid_request_error",
            )
            await self._release_stream(reset=True)
            await self.websocket.close(code=1009, reason="audio session limit exceeded")
            self._stop_requested = True
            return
        self.total_audio_bytes += len(audio)

        try:
            lease = await self._ensure_stream()
            result = await _lease_request(
                lease,
                "stream_chunk",
                {"audio": base64.b64encode(audio).decode("ascii")},
            )
            cumulative = extract_text(result, operation="stream_chunk")
        except WorkerRequestTooLargeError as exc:
            await self._send_error(exc.code, public_worker_error_message(exc))
            await self._release_stream(reset=True)
            return
        except WorkerSaturatedError as exc:
            await self._send_error(
                exc.code,
                public_worker_error_message(exc),
                error_type="rate_limit_error",
            )
            await self._release_stream(reset=True)
            return
        except WorkerRemoteError as exc:
            message = invalid_request_message(exc)
            if message is not None:
                await self._send_error("invalid_request", message)
            else:
                await self._send_failed(exc)
            await self._release_stream(reset=True)
            return
        except WorkerError as exc:
            await self._send_failed(exc)
            await self._release_stream(reset=True)
            return
        except ServeError as exc:
            await self._send_error(exc.code, str(exc))
            return

        delta = (
            cumulative[len(self.transcript) :]
            if cumulative.startswith(self.transcript)
            else cumulative
        )
        self.transcript = cumulative
        await self._send(
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "event_id": self._event_id(),
                "item_id": self.item_id,
                "content_index": 0,
                "delta": delta,
                "transcript": self.transcript,
            }
        )

    async def _commit(self) -> None:
        try:
            lease = await self._ensure_stream()
            result = await _lease_request(lease, "stream_finish")
            self.transcript = extract_text(result, operation="stream_finish")
        except WorkerRequestTooLargeError as exc:
            await self._send_error(exc.code, public_worker_error_message(exc))
            await self._release_stream(reset=False)
            return
        except WorkerSaturatedError as exc:
            await self._send_error(
                exc.code,
                public_worker_error_message(exc),
                error_type="rate_limit_error",
            )
            await self._release_stream(reset=False)
            return
        except WorkerRemoteError as exc:
            message = invalid_request_message(exc)
            if message is not None:
                await self._send_error("invalid_request", message)
            else:
                await self._send_failed(exc)
            await self._release_stream(reset=False)
            return
        except WorkerError as exc:
            await self._send_failed(exc)
            await self._release_stream(reset=False)
            return
        except ServeError as exc:
            await self._send_error(exc.code, str(exc))
            return

        await self._send(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "event_id": self._event_id(),
                "item_id": self.item_id,
                "content_index": 0,
                "transcript": self.transcript,
            }
        )
        await self._release_stream(reset=False)
        self.transcript = ""
        self.item_id = f"item_{uuid.uuid4().hex}"

    async def _clear(self) -> None:
        await self._release_stream(reset=True)
        self.transcript = ""
        self.item_id = f"item_{uuid.uuid4().hex}"
        await self._send(
            {
                "type": "input_audio_buffer.cleared",
                "event_id": self._event_id(),
            }
        )

    async def _ensure_stream(self) -> WorkerSession:
        if self._lease is not None:
            return self._lease
        if self.model is None:
            raise ModelNotFoundError("no default transcription model is configured")
        _spec, lease = self.registry.acquire_session("transcription", self.model)
        config: dict[str, Any] = {
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
            "audio_format": "pcm16le",
        }
        if self.language:
            config["language"] = self.language
        try:
            await _lease_request(lease, "stream_start", {"config": config})
        except BaseException:
            try:
                await _lease_request(
                    lease,
                    "stream_reset",
                    timeout=min(2.0, self.registry.request_timeout),
                )
            except BaseException:
                pass
            finally:
                lease.close()
            raise
        self._lease = lease
        return lease

    async def _release_stream(self, *, reset: bool) -> None:
        lease = self._lease
        self._lease = None
        if lease is None:
            return
        try:
            if reset:
                try:
                    await _lease_request(
                        lease,
                        "stream_reset",
                        timeout=min(2.0, self.registry.request_timeout),
                    )
                except WorkerError:
                    pass
        finally:
            lease.close()

    async def _send_failed(self, error: WorkerError) -> None:
        _log_worker_failure()
        await self._send_failure(error.code, public_worker_error_message(error))

    async def _send_failure(
        self,
        code: str,
        message: str,
        *,
        error_type: str = "server_error",
    ) -> None:
        await self._send(
            {
                "type": "conversation.item.input_audio_transcription.failed",
                "event_id": self._event_id(),
                "item_id": self.item_id,
                "content_index": 0,
                "transcript": self.transcript,
                "error": {
                    "type": error_type,
                    "code": code,
                    "message": message,
                },
            }
        )

    async def _send_error(
        self,
        code: str,
        message: str,
        *,
        error_type: str = "invalid_request_error",
    ) -> None:
        await self._send(
            {
                "type": "error",
                "event_id": self._event_id(),
                "error": {
                    "type": error_type,
                    "code": code,
                    "message": message,
                },
            }
        )

    async def _send(self, payload: dict[str, Any]) -> None:
        try:
            await self.websocket.send_json(payload)
        except (RuntimeError, WebSocketDisconnect):
            pass

    def _session_payload(self) -> dict[str, Any]:
        return {
            "id": self.connection_id,
            "object": "realtime.transcription_session",
            "model": self.model,
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": self.model,
                "language": self.language,
            },
            "trtmc": {
                "sample_rate_hz": self.sample_rate_hz,
                "channels": self.channels,
                "audio_bytes_received": self.total_audio_bytes,
                "max_session_audio_bytes": self.max_session_audio_bytes,
                "idle_timeout_seconds": self.idle_timeout_seconds,
                "max_session_seconds": self.max_session_seconds,
            },
        }

    @staticmethod
    def _event_id() -> str:
        return f"event_{uuid.uuid4().hex}"


async def _lease_request(
    lease: WorkerSession,
    operation: str,
    payload: Mapping[str, Any] | None = None,
    *,
    timeout: float | None = None,
) -> Any:
    """Join a native operation before propagating client cancellation."""

    task = asyncio.wrap_future(lease.submit(operation, payload, timeout=timeout))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        try:
            task.result()
        except BaseException:
            pass
        raise


def _log_worker_failure() -> None:
    _LOGGER.error("Realtime model worker request failed")
