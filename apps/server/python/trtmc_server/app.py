# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI control plane for persistent native Task workers."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import math
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any
from pathlib import Path

from anyio import CancelScope
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .errors import (
    ModelNotFoundError,
    WorkerError,
    WorkerCrashedError,
    WorkerProtocolError,
    WorkerRequestTooLargeError,
    WorkerRemoteError,
    WorkerSaturatedError,
    WorkerTimeoutError,
)
from .metrics import Metrics
from .protocol import chat_prompt, chat_image, extract_vision_result, extract_embedding, extract_ranking, extract_result, extract_transcription, extract_detection, generation_config, public_worker_error
from .records import RequestPolicy, RequestRecords
from .registry import ModelRegistry
from .schemas import ChatCompletionRequest, CompletionRequest, EmbeddingRequest, GenerationRequest, RankingRequest, TranscriptionRequest, DetectionRequest
from .uploads import UploadedInput, multipart_audio, multipart_image, image_file, wav_metadata, image_upload, create_uploaded_input, ImagePixelLimitError
from .worker import WorkerSession
from .videos import VideoJobs
from trtmc_perf_serving.images import png_response
from trtmc_perf_serving.openai import image_generation
from .protocol import extract_task_timing
from .schemas import ImageGenerationRequest



class ServerConfig:
    def __init__(
        self,
        *,
        max_body_bytes: int,
        max_prompt_bytes: int,
        max_generation_tokens: int,
        records: Path | None = None,
        max_audio_seconds: float = 30.0,
        max_image_pixels: int = 16_000_000,
        max_video_jobs: int = 8,
        max_video_bytes: int = 4 * 1024**3,
        video_ttl_seconds: float = 600,
    ) -> None:
        self.max_body_bytes = max_body_bytes
        self.max_prompt_bytes = max_prompt_bytes
        self.max_generation_tokens = max_generation_tokens
        self.records = records
        if not math.isfinite(max_audio_seconds) or max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be finite and positive")
        self.max_audio_seconds = max_audio_seconds
        if type(max_image_pixels) is not int or max_image_pixels <= 0:
            raise ValueError("max_image_pixels must be positive")
        self.max_image_pixels = max_image_pixels
        if type(max_video_jobs) is not int or max_video_jobs < 1 or type(max_video_bytes) is not int or max_video_bytes < 1:
            raise ValueError("video job and storage limits must be positive integers")
        if not math.isfinite(video_ttl_seconds) or video_ttl_seconds <= 0:
            raise ValueError("video TTL must be finite and positive")
        self.max_video_jobs, self.max_video_bytes = max_video_jobs, max_video_bytes
        self.video_ttl_seconds = video_ttl_seconds


class _BodyLimitMiddleware:
    """Reject oversized generation bodies before FastAPI parses them."""

    def __init__(self, app: ASGIApp, *, limit: int) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] not in {
            "/v1/completions",
            "/v1/chat/completions",
            "/v1/embeddings",
            "/v1/ranking",
            "/v1/audio/transcriptions",
            "/v1/infer",
            "/v1/images/generations",
            "/v1/images/edits",
            "/v1/videos",
        }:
            await self.app(scope, receive, send)
            return
        for name, raw_value in scope.get("headers", ()):
            if name.lower() == b"content-length":
                try:
                    if int(raw_value) > self.limit:
                        await error_response(
                            413, "content_too_large", "request body exceeds limit"
                        )(scope, receive, send)
                        return
                except ValueError:
                    await error_response(400, "invalid_request", "invalid Content-Length header")(scope, receive, send)
                    return
                break
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.limit:
                    raise _BodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await error_response(
                413, "content_too_large", "request body exceeds limit"
            )(scope, receive, send)


class _BodyTooLarge(Exception):
    pass


def error_response(
    status: int,
    code: str,
    message: str,
    *,
    param: str | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    headers = {"X-Request-ID": request_id} if request_id else None
    if status >= 500:
        error_type = "server_error"
    elif status == 429:
        error_type = "rate_limit_error"
    else:
        error_type = "invalid_request_error"
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
        headers=headers,
    )


def _sse_data(payload: dict[str, Any] | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return f"data: {data}\n\n".encode("utf-8")


def _streaming_completion_response(
    *,
    response_id: str,
    created: int,
    model: str,
    text: str,
    completion_tokens: int,
    chat: bool,
    include_usage: bool,
    request_id: str,
) -> StreamingResponse:
    if chat:
        object_name = "chat.completion.chunk"
        content_choice: dict[str, Any] = {
            "index": 0,
            "delta": {"role": "assistant", "content": text},
            "logprobs": None,
            "finish_reason": None,
        }
        terminal_choice: dict[str, Any] = {
            "index": 0,
            "delta": {},
            "logprobs": None,
            "finish_reason": None,
        }
    else:
        object_name = "text_completion"
        content_choice = {
            "index": 0,
            "text": text,
            "logprobs": None,
            "finish_reason": None,
        }
        terminal_choice = {
            "index": 0,
            "text": "",
            "logprobs": None,
            "finish_reason": None,
        }

    def chunk(choices: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "id": response_id,
            "object": object_name,
            "created": created,
            "model": model,
            "choices": choices,
        }

    async def events() -> AsyncIterator[bytes]:
        yield _sse_data(chunk([content_choice]))
        yield _sse_data(chunk([terminal_choice]))
        if include_usage:
            usage_chunk = chunk([])
            usage_chunk["usage"] = {
                "prompt_tokens": 0,
                "completion_tokens": completion_tokens,
                "total_tokens": completion_tokens,
            }
            yield _sse_data(usage_chunk)
        yield _sse_data("[DONE]")

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
            "X-TRTMC-Streaming": "buffered",
        },
    )


def create_app(registry: ModelRegistry, config: ServerConfig) -> FastAPI:
    metrics = Metrics()
    records = RequestRecords(config.records)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            await asyncio.to_thread(registry.start)
            yield
        finally:
            videos.accepting = False
            await asyncio.to_thread(registry.close)
            await videos.close()
            records.close()

    app = FastAPI(title="TensorRT-Model-Connect Server", version="0.1.0", lifespan=lifespan)
    app.add_middleware(_BodyLimitMiddleware, limit=config.max_body_bytes)

    app.add_middleware(RequestPolicy, records=records)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, error: RequestValidationError) -> JSONResponse:
        first = error.errors()[0] if error.errors() else {}
        location = first.get("loc", ())
        param = str(location[-1]) if location else None
        return error_response(
            400,
            "invalid_request",
            str(first.get("msg", "request validation failed")),
            param=param,
        )

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready() -> JSONResponse:
        status = registry.status()
        return JSONResponse(
            status_code=200 if status["ready"] else 503,
            content={
                "status": "ready" if status["ready"] else "not_ready",
                "degraded": status["degraded"],
            },
        )

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": registry.models()}

    @app.get("/metrics")
    async def prometheus_metrics() -> PlainTextResponse:
        status = registry.status()
        busy = sum(
            model["ready_replicas"] - model["idle_replicas"]
            for model in status["models"].values()
        )
        return PlainTextResponse(
            metrics.render(ready=status["ready"], busy=busy),
            media_type="text/plain; version=0.0.4",
        )

    async def execute(
        route: str,
        request: GenerationRequest,
        prompt: str,
        *,
        request_id: str,
        context: dict[str, Any],
        system_prompt: str = "",
        chat: bool,
        chat_max_tokens: int | None = None,
    ) -> Any:
        context["model"] = request.model
        context.update(prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                       prompt_bytes=len(prompt.encode("utf-8")))
        if request.n != 1:
            metrics.reject(route, 400)
            return error_response(400, "unsupported_parameter", "n must be 1", param="n")
        if request.stream_options is not None and not request.stream:
            metrics.reject(route, 400)
            return error_response(
                400,
                "invalid_request",
                "stream_options requires stream=true",
                param="stream_options",
            )
        if request.stop is not None:
            metrics.reject(route, 400)
            return error_response(
                400, "unsupported_parameter", "stop is not supported", param="stop"
            )
        if len(prompt.encode("utf-8")) + len(system_prompt.encode("utf-8")) > config.max_prompt_bytes:
            metrics.reject(route, 413)
            return error_response(
                413, "content_too_large", "prompt exceeds the server limit", param="prompt"
            )
        requested_tokens = chat_max_tokens if chat_max_tokens is not None else request.max_tokens
        try:
            default_tokens = registry.max_tokens(request.model, config.max_generation_tokens)
        except ModelNotFoundError as error:
            metrics.reject(route, 404)
            return error_response(404, error.code, str(error), param="model")
        max_tokens = requested_tokens or default_tokens
        if max_tokens > config.max_generation_tokens:
            metrics.reject(route, 400)
            return error_response(
                400,
                "max_tokens_exceeded",
                f"requested generation exceeds {config.max_generation_tokens} tokens",
                param="max_tokens",
            )

        queued_at = time.monotonic()
        try:
            session = registry.acquire(request.model)
        except ModelNotFoundError as error:
            metrics.reject(route, 404)
            return error_response(404, error.code, str(error), param="model")
        except WorkerSaturatedError:
            metrics.reject(route, 429)
            response = error_response(
                429, "server_busy", "all model worker replicas are busy", request_id=request_id
            )
            response.headers["Retry-After"] = "1"
            return response
        except RuntimeError:
            metrics.reject(route, 503)
            return error_response(
                503, "server_draining", "server is not accepting requests", request_id=request_id
            )

        queue_seconds = time.monotonic() - queued_at
        context["replica"] = getattr(session, "replica_name", None)
        context["admission_ms"] = queue_seconds * 1000
        worker_config = generation_config(request, max_tokens)
        worker_config["use_chat_template"] = chat
        if system_prompt:
            worker_config["system_prompt"] = system_prompt
        context["generation_config"] = {key: value for key, value in worker_config.items() if key != "system_prompt"}
        context["has_system_prompt"] = bool(system_prompt)
        metrics.begin()
        inference_started = time.monotonic()
        if request.stream and registry.supports_streaming(request.model):
            return await incremental_response(session, {"prompt": prompt, "config": worker_config},
                model=request.model, chat=chat, include_usage=(request.stream_options is not None
                and request.stream_options.include_usage), request_id=request_id, context=context,
                route=route, queue_seconds=queue_seconds, started=inference_started, metrics=metrics)
        try:
            result = await _worker_request(
                session, {"prompt": prompt, "config": worker_config}
            )
            text, completion_tokens, timings = extract_result(result)
        except asyncio.CancelledError:
            metrics.finish(
                route,
                499,
                queue_seconds=queue_seconds,
                inference_seconds=time.monotonic() - inference_started,
            )
            raise
        except WorkerTimeoutError as error:
            metrics.finish(
                route,
                504,
                queue_seconds=queue_seconds,
                inference_seconds=time.monotonic() - inference_started,
            )
            return error_response(504, error.code, public_worker_error(error), request_id=request_id)
        except WorkerCrashedError as error:
            metrics.finish(
                route,
                503,
                queue_seconds=queue_seconds,
                inference_seconds=time.monotonic() - inference_started,
            )
            return error_response(503, error.code, public_worker_error(error), request_id=request_id)
        except WorkerRemoteError as error:
            details = error.details
            invalid = isinstance(details, Mapping) and details.get("type") == "invalid_request_error"
            status = 400 if invalid else 502
            metrics.finish(
                route,
                status,
                queue_seconds=queue_seconds,
                inference_seconds=time.monotonic() - inference_started,
            )
            return error_response(
                status, error.code, public_worker_error(error), request_id=request_id
            )
        except WorkerRequestTooLargeError as error:
            metrics.finish(
                route,
                413,
                queue_seconds=queue_seconds,
                inference_seconds=time.monotonic() - inference_started,
            )
            return error_response(
                413,
                error.code,
                public_worker_error(error),
                request_id=request_id,
            )
        except WorkerProtocolError as error:
            metrics.finish(
                route,
                502,
                queue_seconds=queue_seconds,
                inference_seconds=time.monotonic() - inference_started,
            )
            return error_response(502, error.code, public_worker_error(error), request_id=request_id)

        inference_seconds = time.monotonic() - inference_started
        context.update(timings, worker_roundtrip_ms=inference_seconds * 1000,
                       completion_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                       completion_tokens=completion_tokens, completion_token_source="native_task",
                       timing_scope=result.get("timing_scope", "unavailable"),
                       streaming="buffered" if request.stream else "none")
        metrics.finish(
            route,
            200,
            queue_seconds=queue_seconds,
            inference_seconds=inference_seconds,
            timings=timings,
        )
        choice: dict[str, Any]
        if chat:
            choice = {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "logprobs": None,
                "finish_reason": None,
            }
            object_name = "chat.completion"
            response_id = f"chatcmpl-{uuid.uuid4().hex}"
        else:
            choice = {
                "index": 0,
                "text": text,
                "logprobs": None,
                "finish_reason": None,
            }
            object_name = "text_completion"
            response_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if request.stream:
            return _streaming_completion_response(
                response_id=response_id,
                created=created,
                model=request.model,
                text=text,
                completion_tokens=completion_tokens,
                chat=chat,
                include_usage=(
                    request.stream_options is not None and request.stream_options.include_usage
                ),
                request_id=request_id,
            )
        return JSONResponse(
            content={
                "id": response_id,
                "object": object_name,
                "created": created,
                "model": request.model,
                "choices": [choice],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": completion_tokens,
                    "total_tokens": completion_tokens,
                },
            },
            headers={"X-Request-ID": request_id},
        )

    @app.post("/v1/completions")
    async def completions(http_request: Request, request: CompletionRequest) -> Any:
        return await execute(
            "/v1/completions",
            request,
            request.prompt,
            request_id=http_request.state.request_id,
            context=http_request.state.timing,
            chat=False,
        )

    async def execute_task(
        http_request: Request, model: str, capability: str, operation: str,
        payload: dict[str, Any], validate: Callable[[Any], Any],
        release_input: Callable[[], None] | None = None,
        acquired_session: WorkerSession | None = None,
        task_context: dict[str, Any] | None = None,
    ) -> Any:
        route = http_request.url.path
        context = task_context if task_context is not None else http_request.state.timing
        context["model"] = model
        request_id = http_request.state.request_id
        admitted = time.monotonic()
        session = acquired_session
        admitted_ok = False
        try:
            if not registry.supports(model, capability):
                metrics.reject(route, 400)
                return error_response(400, "unsupported_task", "model does not support this endpoint", param="model")
            if session is None:
                session = registry.acquire(model)
            admitted_ok = True
        except ModelNotFoundError as error:
            metrics.reject(route, 404)
            return error_response(404, error.code, str(error), param="model")
        except WorkerSaturatedError:
            metrics.reject(route, 429)
            response = error_response(429, "server_busy", "all model worker replicas are busy")
            response.headers["Retry-After"] = "1"
            return response
        except RuntimeError:
            metrics.reject(route, 503)
            return error_response(503, "server_draining", "server is not accepting requests")
        finally:
            if not admitted_ok:
                if session is not None:
                    session.close()
                if release_input is not None:
                    release_input()
        queue_seconds = time.monotonic() - admitted
        context.update(replica=getattr(session, "replica_name", None),
                       admission_ms=queue_seconds * 1000, streaming="none")
        metrics.begin()
        started = time.monotonic()
        status = 500
        try:
            result = await _worker_request(session, payload, operation=operation, validate=validate,
                                           release_input=release_input)
            context.update(model_call_ms=result["model_call_ms"], timing_scope=result["timing_scope"])
            status = 200
            return result
        except asyncio.CancelledError:
            status = 499
            raise
        except WorkerError as error:
            invalid = (isinstance(error, WorkerRemoteError) and isinstance(error.details, Mapping) and
                       error.details.get("type") == "invalid_request_error")
            status = (504 if isinstance(error, WorkerTimeoutError) else
                      503 if isinstance(error, WorkerCrashedError) else
                      413 if isinstance(error, WorkerRequestTooLargeError) else
                      400 if invalid else 502)
            return error_response(status, error.code, public_worker_error(error), request_id=request_id)
        finally:
            elapsed = time.monotonic() - started
            context["worker_roundtrip_ms"] = elapsed * 1000
            metrics.finish(route, status, queue_seconds=queue_seconds, inference_seconds=elapsed)

    async def render_image(http_request: Request, request: ImageGenerationRequest, *,
                           reference: tuple[bytes, str, dict[str, Any]] | None = None) -> Any:
        context = http_request.state.timing
        context["model"] = request.model
        try:
            if len(request.prompt.encode("utf-8")) > config.max_prompt_bytes:
                return error_response(413, "content_too_large", "prompt exceeds the server limit")
            mapped = image_generation(request.model_dump(exclude_none=True), {})
            if mapped.get("height", 0) * mapped.get("width", 0) > config.max_image_pixels:
                return error_response(413, "content_too_large", "output exceeds the pixel limit")
            noise = base64.b64decode(request.initial_latents_b64, validate=True) if request.initial_latents_b64 is not None else b""
            if request.initial_latents_b64 is not None and (not noise or len(noise) % 4):
                raise ValueError("initial_latents_b64 must encode nonempty raw float32 values")
        except (ValueError, binascii.Error):
            return error_response(400, "invalid_request", "invalid image generation options")
        # Reuse upload ownership: files survive cancellation until the native call ends.
        media = await create_uploaded_input(noise, ".f32")
        source = None
        try:
            if reference is not None:
                source = await create_uploaded_input(reference[0], reference[1])
        except BaseException:
            media.close()
            raise

        def release() -> None:
            media.close()
            if source is not None:
                source.close()

        output = media.path.with_name("output.png")
        controls = {key: value for key, value in mapped.items()
                    if key not in {"prompt", "initial_latents_path"}}
        payload = {"prompt": request.prompt, "config": controls, "output_path": str(output)}
        if request.initial_latents_b64 is not None:
            payload["initial_latents_path"] = str(media.path)
        if source is not None:
            payload["image_path"] = str(source.path)
            context.update(reference[2], image_sha256=hashlib.sha256(reference[0]).hexdigest())
        negative = controls.get("negative_prompt", "").encode("utf-8")
        context.update(prompt_sha256=hashlib.sha256(request.prompt.encode()).hexdigest(),
                       generation_config={key: value for key, value in controls.items() if key != "negative_prompt"},
                       negative_prompt_sha256=hashlib.sha256(negative).hexdigest(),
                       negative_prompt_bytes=len(negative),
                       initial_latents_sha256=hashlib.sha256(noise).hexdigest() if noise else None)
        encoded: dict[str, Any] = {}

        def validate(value):
            if not isinstance(value, Mapping):
                raise WorkerProtocolError("worker image result must be an object")
            extract_task_timing(value)
            if (type(value.get("width")) is not int or type(value.get("height")) is not int or
                    type(value.get("output_count")) is not int or value["output_count"] != 1):
                raise WorkerProtocolError("worker returned invalid image dimensions or count")
            try:
                image = png_response(output, max_pixels=config.max_image_pixels)
            except (OSError, ValueError) as error:
                raise WorkerProtocolError("worker returned an unreadable PNG") from error
            if (image["width"], image["height"]) != (value["width"], value["height"]):
                raise WorkerProtocolError("worker image metadata disagrees with its PNG")
            if image["channels"] != value.get("channels"):
                raise WorkerProtocolError("worker image channels disagree with its PNG")
            if request.size and (image["width"], image["height"]) != (controls["width"], controls["height"]):
                raise WorkerProtocolError("worker did not honor the requested image size")
            if reference is not None and (value.get("image_width"), value.get("image_height")) != (
                    reference[2]["image_width"], reference[2]["image_height"]):
                raise WorkerProtocolError("worker reference dimensions disagree with the upload")
            encoded.update(image)

        capability, operation = ("image_edit", "edit_image") if reference is not None else ("image_generation", "generate_image")
        result = await execute_task(http_request, request.model, capability, operation,
                                    payload, validate, release_input=release)
        if isinstance(result, JSONResponse):
            return result
        context.update(output_sha256=encoded["sha256"], output_width=encoded["width"],
                       output_height=encoded["height"], output_count=1)
        return {"created": int(time.time()), "data": [{"b64_json": encoded["b64_json"]}]}

    @app.post("/v1/images/generations")
    async def images(http_request: Request, request: ImageGenerationRequest) -> Any:
        return await render_image(http_request, request)

    @app.post("/v1/images/edits")
    async def image_edits(http_request: Request) -> Any:
        try:
            fields, data = await multipart_image(http_request, config.max_body_bytes)
            request = ImageGenerationRequest.from_form(fields)
            reference = await asyncio.to_thread(image_file, data, config.max_image_pixels)
        except ImagePixelLimitError:
            return error_response(413, "content_too_large", "reference image exceeds the pixel limit")
        except ValueError:
            return error_response(400, "invalid_request", "expected one RGB PNG/JPEG image and valid image edit controls")
        return await render_image(http_request, request, reference=reference)

    @app.post("/v1/embeddings")
    async def embeddings(http_request: Request, request: EmbeddingRequest) -> Any:
        route = http_request.url.path
        http_request.state.timing["model"] = request.model
        value = request.input
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if not isinstance(value, str) or not value:
            metrics.reject(route, 400)
            return error_response(400, "unsupported_parameter", "input must be one non-empty string", param="input")
        if len(value.encode("utf-8")) > config.max_prompt_bytes:
            metrics.reject(route, 413)
            return error_response(413, "content_too_large", "input exceeds the server limit", param="input")
        result = await execute_task(http_request, request.model, "embeddings", "embed",
                                    {"input": value}, extract_embedding)
        if isinstance(result, JSONResponse):
            return result
        vector, _ = extract_embedding(result)
        http_request.state.timing.update(input_items=1, embedding_dimensions=len(vector))
        # The native Task does not expose token usage; omit it rather than fabricate zero.
        return {"object": "list", "model": request.model,
                "data": [{"object": "embedding", "index": 0, "embedding": vector}]}

    @app.post("/v1/ranking")
    async def ranking(http_request: Request, request: RankingRequest) -> Any:
        route = http_request.url.path
        context = http_request.state.timing
        context["model"] = request.model
        documents = [passage.text for passage in request.passages]
        context.update(input_items=len(documents), passages=len(documents),
                       query_bytes=len(request.query.text.encode("utf-8")),
                       passage_bytes=[len(text.encode("utf-8")) for text in documents])
        size = context["query_bytes"] + sum(context["passage_bytes"])
        if size > config.max_prompt_bytes:
            metrics.reject(route, 413)
            return error_response(413, "content_too_large", "query and passages exceed the server limit")
        result = await execute_task(http_request, request.model, "reranking", "rerank",
                                    {"query": request.query.text, "documents": documents},
                                    lambda value: extract_ranking(value, len(documents)))
        if isinstance(result, JSONResponse):
            return result
        scores, _ = extract_ranking(result, len(documents))
        order = sorted(range(len(scores)), key=lambda index: -scores[index])
        return {"rankings": [{"index": index, "logit": scores[index]} for index in order]}

    @app.post("/v1/infer")
    async def detection(http_request: Request, request: DetectionRequest) -> Any:
        route = http_request.url.path
        context = http_request.state.timing
        context.update(model=request.model, input_items=1)
        try:
            data, suffix, metadata = await asyncio.to_thread(image_upload, request.input[0].url,
                                                            config.max_image_pixels)
            context.update(metadata)
        except ImagePixelLimitError as error:
            context.update(error.metadata)
            metrics.reject(route, 413)
            return error_response(413, "content_too_large", str(error))
        except ValueError as error:
            metrics.reject(route, 400)
            return error_response(400, "invalid_request", str(error))
        try:
            model = request.model if request.model is not None else registry.default_model("object_detection")
        except ValueError as error:
            metrics.reject(route, 400)
            return error_response(400, "model_required", str(error), param="model")
        context["model"] = model
        media = await create_uploaded_input(data, suffix)

        def validate(value):
            return extract_detection(value, context["image_height"], context["image_width"])

        result = await execute_task(http_request, model, "object_detection", "detect",
                                    {"image_path": str(media.path)}, validate, release_input=media.close)
        if isinstance(result, JSONResponse):
            return result
        boxes, _ = validate(result)
        context["detection_boxes"] = len(result["boxes"])
        return {"data": [{"index": 0, "bounding_boxes": boxes}]}

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(http_request: Request) -> Any:
        route = http_request.url.path
        context = http_request.state.timing
        try:
            fields, data = await multipart_audio(http_request, config.max_body_bytes)
            context.update(model=fields.get("model"), input_items=1, audio_bytes=len(data))
            context.update(wav_metadata(data))
            request = TranscriptionRequest.model_validate(fields)
        except ValueError as error:
            metrics.reject(route, 400)
            return error_response(400, "invalid_request", str(error))
        if context["audio_duration_seconds"] > config.max_audio_seconds:
            metrics.reject(route, 413)
            return error_response(413, "content_too_large", "audio duration exceeds the server limit")
        if request.max_output_tokens is not None and request.max_output_tokens > config.max_generation_tokens:
            metrics.reject(route, 400)
            return error_response(400, "unsupported_parameter", "max_output_tokens exceeds the server limit")
        native_config = ({"max_output_tokens": request.max_output_tokens}
                         if request.max_output_tokens is not None else {})
        media = UploadedInput(data, ".wav")
        def validate(value):
            return extract_transcription(value, context["audio_sample_count"],
                                         context["audio_sample_rate"])
        result = await execute_task(http_request, request.model, "audio_transcription", "transcribe",
                                    {"audio_path": str(media.path), "config": native_config}, validate,
                                    release_input=media.close)
        if isinstance(result, JSONResponse):
            return result
        text = validate(result)
        context["transcript_bytes"] = len(text.encode("utf-8"))
        return PlainTextResponse(text) if request.response_format == "text" else {"text": text}

    async def vision_chat(http_request: Request, request: ChatCompletionRequest,
                          prompt: str, image_url: str) -> Any:
        route = http_request.url.path
        context = http_request.state.timing
        context.update(model=request.model, input_items=1)
        if (request.stream or request.stream_options is not None or request.n != 1 or
                request.stop is not None or request.enable_thinking is not None):
            metrics.reject(route, 400)
            return error_response(400, "unsupported_parameter",
                                  "image chat supports one nonstreaming answer without stop or thinking controls")
        if not prompt.strip():
            metrics.reject(route, 400)
            return error_response(400, "invalid_request", "image chat requires a text prompt", param="text")
        if len(prompt.encode("utf-8")) > config.max_prompt_bytes:
            metrics.reject(route, 413)
            return error_response(413, "content_too_large", "prompt exceeds the server limit")
        try:
            default_tokens = registry.max_tokens(request.model, config.max_generation_tokens)
        except ModelNotFoundError as error:
            metrics.reject(route, 404)
            return error_response(404, error.code, str(error), param="model")
        max_tokens = request.max_completion_tokens or request.max_tokens or default_tokens
        if max_tokens > config.max_generation_tokens:
            metrics.reject(route, 400)
            return error_response(400, "max_tokens_exceeded", "requested generation exceeds the server limit")
        try:
            data, suffix, metadata = await asyncio.to_thread(image_upload, image_url, config.max_image_pixels)
            context.update(metadata)
        except ImagePixelLimitError as error:
            context.update(error.metadata)
            metrics.reject(route, 413)
            return error_response(413, "content_too_large", str(error))
        except ValueError as error:
            metrics.reject(route, 400)
            return error_response(400, "invalid_request", str(error))

        worker_config = generation_config(request, max_tokens)
        context["generation_config"] = worker_config
        media = await create_uploaded_input(data, suffix)

        def validate(value):
            return extract_vision_result(value, context["image_height"], context["image_width"])

        result = await execute_task(http_request, request.model, "vision_language_generation", "generate_vision",
                                    {"prompt": prompt, "image_path": str(media.path), "config": worker_config},
                                    validate, release_input=media.close)
        if isinstance(result, JSONResponse):
            return result
        text, tokens, timings = validate(result)
        context.update(timings, completion_tokens=tokens, completion_token_source="native_task")
        return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
                "created": int(time.time()), "model": request.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "logprobs": None, "finish_reason": None}],
                "usage": {"prompt_tokens": 0, "completion_tokens": tokens, "total_tokens": tokens}}

    @app.post("/v1/chat/completions")
    async def chat_completions(http_request: Request, request: ChatCompletionRequest) -> Any:
        try:
            prompt, system_prompt = chat_prompt(request)
            image_url = chat_image(request)
        except ValueError as error:
            metrics.reject("/v1/chat/completions", 400)
            return error_response(400, "invalid_request", str(error), param="messages")
        if request.max_tokens is not None and request.max_completion_tokens is not None:
            metrics.reject("/v1/chat/completions", 400)
            return error_response(
                400,
                "invalid_request",
                "max_tokens and max_completion_tokens are mutually exclusive",
            )
        if image_url is not None:
            return await vision_chat(http_request, request, prompt, image_url)
        return await execute(
            "/v1/chat/completions",
            request,
            prompt,
            request_id=http_request.state.request_id,
            context=http_request.state.timing,
            system_prompt=system_prompt,
            chat=True,
            chat_max_tokens=request.max_completion_tokens,
        )

    videos = VideoJobs(registry, config, records, execute_task, error_response)
    app.include_router(videos.router)
    return app


async def _worker_request(session: WorkerSession, payload: dict[str, Any], *,
                          operation: str = "generate", validate: Callable[[Any], Any] = extract_result,
                          release_input: Callable[[], None] | None = None) -> Any:
    released = False

    def release() -> None:
        nonlocal released
        if released:
            return
        released = True
        try:
            session.close()
        finally:
            if release_input is not None:
                release_input()

    try:
        task = asyncio.wrap_future(session.submit(operation, payload))
    except BaseException:
        release()
        raise
    ownership = task
    try:
        result = await asyncio.shield(task)
        try:
            validate(result)
        except WorkerProtocolError as error:
            ownership = asyncio.wrap_future(session.retire(error))
            await asyncio.shield(ownership)
            raise
        return result
    except asyncio.CancelledError:
        def completed_release(completed: asyncio.Future[Any]) -> None:
            release()
            try:
                completed.exception()
            except BaseException:
                pass

        ownership.add_done_callback(completed_release)
        raise
    finally:
        if ownership.done():
            release()


async def incremental_response(session: WorkerSession, payload: dict[str, Any], *, model: str,
                         chat: bool, include_usage: bool, request_id: str, context: dict[str, Any],
                         route: str, queue_seconds: float, started: float, metrics: Metrics) -> StreamingResponse:
    response_id = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex}"
    created = int(time.time())

    def chunk(text: str, terminal: bool = False) -> dict[str, Any]:
        choice = {"index": 0, "logprobs": None, "finish_reason": None}
        if chat:
            choice["delta"] = {} if terminal else {"role": "assistant", "content": text}
        else:
            choice["text"] = text
        return {"id": response_id, "object": "chat.completion.chunk" if chat else "text_completion",
                "created": created, "model": model, "choices": [choice]}

    stream = None
    try:
        stream = session.stream(payload)
        first = await asyncio.to_thread(stream.next)
    except asyncio.CancelledError:
        if stream is not None:
            stream.abort()
            stream.future.add_done_callback(lambda _future: session.close())
            with CancelScope(shield=True):
                await asyncio.shield(asyncio.to_thread(stream.cancel))
        else:
            session.close()
        metrics.finish(route, 499, queue_seconds=queue_seconds,
                       inference_seconds=time.monotonic() - started)
        raise
    except (WorkerTimeoutError, WorkerCrashedError, WorkerProtocolError, WorkerRemoteError,
            WorkerRequestTooLargeError) as error:
        session.close()
        invalid = (isinstance(error, WorkerRemoteError) and isinstance(error.details, Mapping)
                   and error.details.get("type") == "invalid_request_error")
        status = (400 if invalid else 504 if isinstance(error, WorkerTimeoutError)
                  else 503 if isinstance(error, WorkerCrashedError) else 502)
        metrics.finish(route, status, queue_seconds=queue_seconds,
                       inference_seconds=time.monotonic() - started)
        return error_response(status, error.code, public_worker_error(error), request_id=request_id)

    state = {"started": False}

    async def events() -> AsyncIterator[bytes]:
        state["started"] = True
        status = 499
        timings = {}
        retire = False
        try:
            fragments = []
            token_count = 0
            delta = first
            while delta is not None:
                fragments.append(delta["text_delta"])
                token_count += delta["token_count"]
                if delta["text_delta"]:
                    yield _sse_data(chunk(delta["text_delta"]))
                delta = await asyncio.to_thread(stream.next)
            result = stream.future.result()
            text, completion_tokens, timings = extract_result(result)
            if text != "".join(fragments) or completion_tokens != token_count:
                raise WorkerProtocolError("stream deltas disagree with the final result")
            context.update(timings, completion_tokens=completion_tokens,
                completion_token_source="native_task", timing_scope=result.get("timing_scope", "unavailable"),
                streaming="incremental", worker_roundtrip_ms=(time.monotonic() - started) * 1000)
            yield _sse_data(chunk("", terminal=True))
            if include_usage:
                usage = chunk("", terminal=True)
                usage["choices"] = []
                usage["usage"] = {"prompt_tokens": 0, "completion_tokens": completion_tokens,
                                  "total_tokens": completion_tokens}
                yield _sse_data(usage)
            yield _sse_data("[DONE]")
            status = 200
        except (WorkerTimeoutError, WorkerCrashedError, WorkerProtocolError, WorkerRemoteError,
                WorkerRequestTooLargeError) as error:
            status = (504 if isinstance(error, WorkerTimeoutError)
                      else 503 if isinstance(error, WorkerCrashedError) else 502)
            retire = isinstance(error, WorkerProtocolError)
            context["terminal_status"] = status
            yield _sse_data({"error": {"message": public_worker_error(error), "type": "server_error",
                                      "code": error.code}})
            yield _sse_data("[DONE]")
        finally:
            if not stream.future.done() or retire:
                stream.abort()
                # The lease stays held until confirmed worker exit / native completion.
                stream.future.add_done_callback(lambda _future: session.close())
                # Starlette's disconnect listener cancels an AnyIO scope.
                # Shield that scope too so cleanup confirms worker exit before returning.
                with CancelScope(shield=True):
                    await asyncio.shield(asyncio.to_thread(stream.cancel))
            else:
                session.close()
            context.setdefault("terminal_status", status)
            metrics.finish(route, status, queue_seconds=queue_seconds,
                           inference_seconds=time.monotonic() - started, timings=timings)

    body = events()

    class LeasedResponse(StreamingResponse):
        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            try:
                await super().__call__(scope, receive, send)
            finally:
                if state["started"]:
                    # A failed ASGI send can leave the iterator suspended at
                    # its yield. Close it explicitly to finish lane retirement.
                    await body.aclose()
                else:
                    if not stream.future.done():
                        stream.abort()
                        stream.future.add_done_callback(lambda _future: session.close())
                        with CancelScope(shield=True):
                            await asyncio.shield(asyncio.to_thread(stream.cancel))
                    else:
                        session.close()
                    context["terminal_status"] = 499
                    metrics.finish(route, 499, queue_seconds=queue_seconds,
                                   inference_seconds=time.monotonic() - started)

    return LeasedResponse(body, media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Request-ID": request_id,
        "X-TRTMC-Streaming": "incremental"})
