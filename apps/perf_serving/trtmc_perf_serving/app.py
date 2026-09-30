# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI surface: one backend, one operation, one serialized execution lane."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from . import openai
from .backends.base import Backend, BackendError, BackendUnavailable, Invocation
from .digests import add_media_digests
from .files import materialize_files, server_path_fields

# Request IDs name a per-request directory under the scratch root: one path component that is never
# "." or ".." (leading alphanumeric, no separators).
_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


@dataclass(frozen=True)
class ServingConfig:
    base_request: Mapping[str, Any]
    records: Path
    scratch: Path
    model_name: str = "trtmc"
    # Requests allowed to wait while the lane is busy; beyond this the server answers 429.
    max_queue: int = 64
    keep_artifacts: bool = False
    # Report whole observations (masks, logits, feature maps) instead of compacting long arrays.
    full_observations: bool = False
    # Renders multi-message chat into one prompt with the model's chat template (generate only).
    chat_renderer: openai.ChatRenderer | None = None
    info: Mapping[str, Any] = field(default_factory=dict)


MAX_INLINE_ITEMS = 64


def compact(value: Any) -> Any:
    """Replace arrays longer than MAX_INLINE_ITEMS by their length.

    Worker observations can carry whole masks or feature maps; serializing them into
    every response and record would dominate client latency for small models.
    """
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) > MAX_INLINE_ITEMS:
            return {"$array_length": len(value)}
        return [compact(item) for item in value]
    return value


class Saturated(RuntimeError):
    pass


class Lane:
    """Serializes backend calls; queue time is reported separately from model time."""

    def __init__(self, max_queue: int) -> None:
        self._lock = asyncio.Lock()
        self._max_queue = max_queue
        self._waiting = 0

    async def run(self, call: Callable[[], Invocation]) -> tuple[Invocation, float, float]:
        if self._lock.locked() and self._waiting >= self._max_queue:
            raise Saturated("execution lane and queue are full")
        arrived = time.perf_counter()
        self._waiting += 1
        try:
            await self._lock.acquire()
        finally:
            self._waiting -= 1
        try:
            started = time.perf_counter()
            result = await run_in_threadpool(call)
            return result, (started - arrived) * 1000.0, (time.perf_counter() - started) * 1000.0
        finally:
            self._lock.release()


class Records:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()

    def write(self, record: Mapping[str, Any]) -> None:
        with self._lock, open(self._path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")


def create_app(backend: Backend, config: ServingConfig) -> FastAPI:
    app = FastAPI(title="TRTMC performance serving")
    lane = Lane(config.max_queue)
    records = Records(config.records)
    state = {"available": True, "jobs": {}}
    identity = dict(backend.describe())
    operation = backend.operation

    async def execute(request_id: str, route: str, operation_request: Mapping[str, Any]) -> dict[str, Any]:
        workdir = config.scratch / request_id
        try:
            # Backends write output artifacts below the request directory even without inputs.
            workdir.mkdir(parents=True, exist_ok=True)
            resolved = materialize_files(operation_request, workdir / "inputs")
            invocation, queue_ms, handler_ms = await lane.run(lambda: backend.invoke(resolved, workdir / "output"))
            # Generated media leave the request as digests (the files are removed below).
            invocation = Invocation(add_media_digests(invocation.observation, operation, workdir),
                                    invocation.model_call_ms, invocation.extra)
        except BackendUnavailable:
            state["available"] = False
            raise
        finally:
            if not config.keep_artifacts:
                shutil.rmtree(workdir, ignore_errors=True)
        timing = {"queue_ms": queue_ms, "model_call_ms": invocation.model_call_ms, "handler_ms": handler_ms,
                  **{key: value for key, value in invocation.extra.items() if key != "prompt_tokens"}}
        reported = invocation.observation if config.full_observations else compact(invocation.observation)
        records.write({"request_id": request_id, "route": route, "operation": operation,
                       "backend": identity.get("backend"), "time": time.time(), "timing": timing,
                       "observation": reported})
        return {"timing": timing, "observation": invocation.observation, "reported": reported,
                "extra": invocation.extra}

    def request_id_of(request: Request) -> str:
        value = request.headers.get("x-request-id", "")
        return value if _REQUEST_ID.match(value) else uuid.uuid4().hex

    def respond(body: dict[str, Any], result: Mapping[str, Any], request_id: str) -> JSONResponse:
        body["trtmc_timing"] = result["timing"]
        body["trtmc_observation"] = result["reported"]
        return JSONResponse(body, headers=_headers(request_id, result["timing"]))

    async def guarded(request: Request, route: str, operations: tuple[str, ...],
                      handler: Callable[[str], Awaitable[Any]]) -> Any:
        request_id = request_id_of(request)
        if operation not in operations:
            return _error(404, "route_not_served", f"{route} is not served for operation {operation!r}", request_id)
        if not state["available"]:
            return _error(503, "backend_unavailable", "backend is no longer available", request_id)
        try:
            return await handler(request_id)
        except (openai.RequestError, ValueError, json.JSONDecodeError) as error:
            return _error(400, "invalid_request", str(error), request_id)
        except BackendError as error:
            return _error(422, "backend_rejected_request", str(error), request_id)
        except Saturated as error:
            response = _error(429, "server_busy", str(error), request_id)
            response.headers["Retry-After"] = "1"
            return response
        except BackendUnavailable as error:
            return _error(503, "backend_unavailable", str(error), request_id)
        except Exception as error:  # noqa: BLE001 - a failed call must keep its request id
            return _error(500, "backend_failed", f"{type(error).__name__}: {error}", request_id)

    @app.post("/v1/tasks/{name}")
    async def task(name: str, request: Request):
        async def handle(request_id: str):
            if name != operation:
                raise openai.RequestError(f"this server serves {operation!r}, not {name!r}")
            body = await request.json()
            if set(body) - {"request", "model"} or not isinstance(body.get("request", {}), dict):
                raise openai.RequestError("body must be {\"request\": {...operation request...}}")
            local = server_path_fields(body.get("request", {}))
            if local:  # only the server's own base request may name server files
                raise openai.RequestError(f"send {', '.join(sorted(set(local)))} inline as "
                                          '{"$file": {"suffix": ..., "b64": ...}}, not as a server path')
            result = await execute(request_id, f"/v1/tasks/{name}", {**config.base_request, **body.get("request", {})})
            # Generic clients (aiperf `raw` endpoint) need a text field to count a response as valid.
            text = str(result["observation"].get("text", operation))
            return respond({"id": request_id, "object": "task.result", "operation": operation, "text": text},
                           result, request_id)
        return await guarded(request, "/v1/tasks", (operation,), handle)

    @app.post("/v1/completions")
    async def completions(request: Request):
        async def handle(request_id: str):
            body = await request.json()
            result = await execute(request_id, "/v1/completions", openai.completion(body, config.base_request))
            return _text_response(body, result, request_id, config.model_name, chat=False)
        return await guarded(request, "/v1/completions", ("generate",), handle)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        async def handle(request_id: str):
            body = await request.json()
            mapped = openai.chat(body, config.base_request, config.chat_renderer)
            result = await execute(request_id, "/v1/chat/completions", mapped)
            return _text_response(body, result, request_id, config.model_name, chat=True)
        return await guarded(request, "/v1/chat/completions", ("generate",), handle)

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        async def handle(request_id: str):
            result = await execute(request_id, "/v1/embeddings",
                                   openai.embeddings(await request.json(), config.base_request))
            vector = result["observation"].get("values", [])
            return respond({"object": "list", "model": config.model_name,
                            "data": [{"object": "embedding", "index": 0, "embedding": vector}],
                            "usage": {"prompt_tokens": 0, "total_tokens": 0}}, result, request_id)
        return await guarded(request, "/v1/embeddings", ("embed", "encode"), handle)

    @app.post("/v1/ranking")
    async def ranking(request: Request):
        async def handle(request_id: str):
            result = await execute(request_id, "/v1/ranking", openai.ranking(await request.json(), config.base_request))
            scores = result["observation"].get("scores") or []
            order = sorted(range(len(scores)), key=lambda index: -float(scores[index]))
            return respond({"rankings": [{"index": index, "logit": float(scores[index])} for index in order]},
                           result, request_id)
        return await guarded(request, "/v1/ranking", ("rerank",), handle)

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(request: Request):
        async def handle(request_id: str):
            form = await request.form()
            upload = form.get("file")
            if upload is None or isinstance(upload, str):
                raise openai.RequestError("multipart field 'file' is required")
            fields = {key: value for key, value in form.items() if key != "file"}
            mapped = openai.transcription(await upload.read(), upload.filename or "audio.wav", fields,
                                          config.base_request)
            result = await execute(request_id, "/v1/audio/transcriptions", mapped)
            return respond({"text": result["observation"].get("text", "")}, result, request_id)
        return await guarded(request, "/v1/audio/transcriptions", ("transcribe",), handle)

    @app.post("/v1/images/generations")
    async def images(request: Request):
        async def handle(request_id: str):
            mapped = openai.image_generation(await request.json(), config.base_request)
            result = await execute(request_id, "/v1/images/generations", mapped)
            return respond({"created": int(time.time()), "data": [{"url": _artifact_url(result)}]}, result, request_id)
        return await guarded(request, "/v1/images/generations", ("generate_image",), handle)

    @app.post("/v1/videos")
    async def create_video(request: Request):
        # OpenAI Videos API (aiperf video_generation): multipart form, then polling.
        async def handle(request_id: str):
            if request.headers.get("content-type", "").startswith("multipart/form-data"):
                body = {key: value for key, value in (await request.form()).items()}
            else:
                body = await request.json()
            mapped = openai.image_generation(body, {**config.base_request, "media_type": "video"})
            job = {"id": request_id, "object": "video", "status": "in_progress", "created_at": int(time.time())}
            state["jobs"][request_id] = job

            async def work() -> None:
                try:
                    result = await execute(request_id, "/v1/videos", mapped)
                    job.update(status="completed", url=_artifact_url(result), trtmc_timing=result["timing"],
                               inference_time_s=result["timing"]["model_call_ms"] / 1000.0)
                except Exception as error:  # noqa: BLE001 - surfaced through the polled job
                    job.update(status="failed", error={"message": str(error)})

            asyncio.create_task(work())
            return JSONResponse(dict(job), headers={"X-Request-ID": request_id})
        return await guarded(request, "/v1/videos", ("generate_image",), handle)

    @app.get("/v1/videos/{job_id}")
    async def get_video(job_id: str):
        job = state["jobs"].get(job_id)
        return JSONResponse(dict(job)) if job else _error(404, "not_found", "unknown video job", None)

    @app.get("/health/live")
    async def live():
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready():
        if not state["available"]:
            return JSONResponse({"status": "unavailable"}, status_code=503)
        return {"status": "ready", "operation": operation}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": config.model_name, "object": "model", "owned_by": "trtmc"}]}

    @app.get("/v1/serving/info")
    async def info():
        return {**identity, **dict(config.info), "max_queue": config.max_queue,
                "base_request": dict(config.base_request)}

    return app


def _headers(request_id: str, timing: Mapping[str, Any]) -> dict[str, str]:
    return {"X-Request-ID": request_id, "Server-Timing": f"model;dur={float(timing['model_call_ms']):.3f}"}


def _error(status: int, code: str, message: str, request_id: str | None) -> JSONResponse:
    kind = "server_error" if status >= 500 else "rate_limit_error" if status == 429 else "invalid_request_error"
    headers = {"X-Request-ID": request_id} if request_id else None
    return JSONResponse({"error": {"message": message, "type": kind, "code": code}}, status_code=status,
                        headers=headers)


def _artifact_url(result: Mapping[str, Any]) -> str:
    artifact = result["observation"].get("artifact") or result["observation"].get("path") or ""
    return f"file://{artifact}" if artifact else ""


def _text_response(body: Mapping[str, Any], result: Mapping[str, Any], request_id: str, model: str,
                   *, chat: bool) -> Any:
    observation = result["observation"]
    text = openai.truncate_at_stop(str(observation.get("text", "")), openai.stop_sequences(body))
    completion_tokens = int(observation.get("output_tokens", 0))
    prompt_tokens = int(result["extra"].get("prompt_tokens", 0))
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
             "total_tokens": prompt_tokens + completion_tokens}
    kind = "chat.completion" if chat else "text_completion"
    if not body.get("stream"):
        choice = ({"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": None}
                  if chat else {"index": 0, "text": text, "finish_reason": None})
        payload = {"id": request_id, "object": kind, "created": int(time.time()), "model": model,
                   "choices": [choice], "usage": usage, "trtmc_timing": result["timing"],
                   "trtmc_observation": result["reported"]}
        return JSONResponse(payload, headers=_headers(request_id, result["timing"]))

    # Buffered SSE: the complete generation is emitted as one chunk for every backend,
    # so TTFT equals request latency and candidate/reference streaming stays comparable.
    chunk_kind = "chat.completion.chunk" if chat else "text_completion"
    delta = {"delta": {"role": "assistant", "content": text}} if chat else {"text": text}

    def event(payload: Mapping[str, Any]) -> bytes:
        return f"data: {json.dumps(payload)}\n\n".encode()

    async def events():
        yield event({"id": request_id, "object": chunk_kind, "model": model,
                     "choices": [{"index": 0, **delta, "finish_reason": None}]})
        yield event({"id": request_id, "object": chunk_kind, "model": model, "choices": [], "usage": usage,
                     "trtmc_timing": result["timing"]})
        yield b"data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers=_headers(request_id, result["timing"]))
