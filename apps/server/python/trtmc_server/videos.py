# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded asynchronous video jobs using the existing worker admission and Task path."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Literal
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import Field, ValidationError
from starlette.datastructures import UploadFile

from .errors import ModelNotFoundError, WorkerProtocolError, WorkerSaturatedError
from .protocol import extract_task_timing
from .schemas import ImageGenerationRequest
from .uploads import ImagePixelLimitError, create_uploaded_input, image_file
from trtmc_perf_serving.openai import image_generation
from trtmc_perf_serving.video_media import encode

# Reference-to-video requests (multipart: input_reference files plus these text fields).
MAX_REFERENCE_IMAGES = 16
REFERENCE_VIDEO_FIELDS = {"model", "prompt", "seed", "continuation_prompt"}
# Native frames are unknown until the Task returns; reserve storage for a long clip.
REFERENCE_VIDEO_RESERVE_BYTES = 2 * 1024**3


class VideoRequest(ImageGenerationRequest):
    num_frames: int | None = Field(default=None, ge=2, strict=True)
    fps: int = Field(default=24, ge=1, le=120, strict=True)
    # Duration is a consequence of the built frame count and presentation FPS.
    response_format: Literal["b64_json"] = "b64_json"

    @classmethod
    def from_form(cls, fields: dict[str, Any]) -> VideoRequest:
        values = dict(fields)
        for key in ("num_frames", "fps"):
            if key in values:
                raw = values[key]
                if not isinstance(raw, str) or not raw.isascii() or not raw.isdecimal():
                    raise ValueError(f"{key} must be an integer")
                values[key] = int(raw)
        return super().from_form(values)


class VideoJobs:
    def __init__(self, registry, config, records, execute, error_response):
        self.registry, self.config, self.records = registry, config, records
        self.execute, self.error = execute, error_response
        self.jobs: dict[str, dict] = {}
        self.tasks: set[asyncio.Task] = set()
        self.downloads_idle = asyncio.Event()
        self.downloads_idle.set()
        self.setups = 0
        self.setups_idle = asyncio.Event()
        self.setups_idle.set()
        self.accepting = True
        self.router = APIRouter()
        self.router.add_api_route("/v1/videos", self.submit, methods=["POST"])
        self.router.add_api_route("/v1/videos/{job_id}", self.poll, methods=["GET"])
        self.router.add_api_route("/v1/videos/{job_id}/content", self.content, methods=["GET"])

    async def expire(self):
        for identifier, job in list(self.jobs.items()):
            if job["status"] in {"completed", "failed"} and not job["downloads"] and not job["active"] and (
                    time.monotonic() - job["finished"] >= self.config.video_ttl_seconds):
                del self.jobs[identifier]
                await asyncio.to_thread(shutil.rmtree, job["root"])

    async def close(self):
        self.accepting = False
        await self.setups_idle.wait()
        # Registry shutdown retires native calls. Encoding still owns its files.
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.downloads_idle.wait()
        for job in self.jobs.values():
            await asyncio.to_thread(shutil.rmtree, job["root"])
        self.jobs.clear()

    @staticmethod
    def view(job):
        value = {key: job[key] for key in ("id", "object", "model", "created_at", "status", "size", "seconds", "num_frames", "fps")}
        if job["status"] == "completed":
            value.update(inference_time_s=job["model_call_ms"] / 1000, completed_at=job["completed_at"],
                         expires_at=job["expires_at"])
        if job["status"] == "failed":
            value["error"] = {"code": "video_generation_failed", "message": "video generation or encoding failed"}
        return value

    async def submit(self, request: Request):
        await self.expire()
        if not self.accepting:
            return self.error(503, "server_draining", "server is not accepting requests")
        try:
            if request.headers.get("content-type", "").split(";", 1)[0] == "application/json":
                body = VideoRequest.model_validate(await request.json())
            else:
                async with request.form(max_files=MAX_REFERENCE_IMAGES, max_fields=24,
                                        max_part_size=self.config.max_body_bytes) as form:
                    if any(isinstance(v, UploadFile) for _, v in form.multi_items()):
                        return await self.submit_references(request, form)
                    if len(form.multi_items()) != len(form):
                        raise ValueError("video requests require unique text fields")
                    body = VideoRequest.from_form(dict(form))
            if len(body.prompt.encode()) > self.config.max_prompt_bytes:
                return self.error(413, "content_too_large", "prompt exceeds the server limit")
            if not self.registry.supports(body.model, "video_generation"):
                return self.error(400, "unsupported_task", "model does not support this endpoint")
            geometry = self.registry.video_geometry(body.model)
            width, height, count = geometry["width"], geometry["height"], geometry["num_frames"]
            if body.num_frames not in (None, count) or body.size not in (None, f"{width}x{height}"):
                raise ValueError("size and num_frames must match the built video geometry")
            reserved = width * height * count * 3 * 4 + self.config.max_body_bytes + 1024 * 1024
            if (len(self.jobs) >= self.config.max_video_jobs or
                    sum(job["reserved"] for job in self.jobs.values()) + reserved > self.config.max_video_bytes):
                return self.error(429, "video_storage_full", "video job storage is full; retry after expiry")
            values = body.model_dump(exclude_none=True, exclude={"num_frames", "fps"})
            values["size"] = f"{width}x{height}"
            mapped = image_generation(values, {})
            mapped.pop("initial_latents_path", None)
            noise = base64.b64decode(body.initial_latents_b64, validate=True) if body.initial_latents_b64 else b""
            if body.initial_latents_b64 is not None and (not noise or len(noise) % 4):
                raise ValueError("initial_latents_b64 must encode nonempty raw float32 values")
            session = self.registry.acquire(body.model)
        except ModelNotFoundError as error:
            return self.error(404, error.code, str(error))
        except WorkerSaturatedError:
            return self.error(429, "server_busy", "all model worker replicas are busy")
        except (ValueError, ValidationError, json.JSONDecodeError):
            return self.error(400, "invalid_request", "invalid or unsupported video controls")
        except RuntimeError:
            return self.error(503, "server_draining", "server is not accepting requests")
        self.setups += 1
        self.setups_idle.clear()
        root = None
        identifier = None
        media = None
        try:
            root = Path(tempfile.mkdtemp(prefix="trtmc-video-"))
            identifier = f"video-{uuid.uuid4().hex}"
            job = {"id": identifier, "object": "video", "created_at": int(time.time()),
                   "model": body.model, "status": "in_progress", "root": root,
                   "reserved": reserved, "downloads": 0, "active": True,
                   "size": f"{width}x{height}", "seconds": format(count / body.fps, ".9g"),
                   "num_frames": count, "fps": body.fps}
            self.jobs[identifier] = job
            request.state.timing.update(model=body.model, record_kind="video_submission", job_id=identifier)
            media = await create_uploaded_input(noise, ".f32")
            payload = {"prompt": mapped.pop("prompt"), "config": mapped, "output_path": str(root / "frames")}
            if noise:
                payload["initial_latents_path"] = str(media.path)
            context = {**request.state.timing, "record_kind": "video_completion", "job_id": identifier,
                       "prompt_sha256": hashlib.sha256(body.prompt.encode()).hexdigest(),
                       "negative_prompt_sha256": hashlib.sha256(body.negative_prompt.encode() if body.negative_prompt else b"").hexdigest(),
                       "generation_config": {k: v for k, v in mapped.items() if k != "negative_prompt"},
                       "initial_latents_sha256": hashlib.sha256(noise).hexdigest() if noise else None}
            task = asyncio.create_task(self.run(job, request, payload, geometry, body.fps, session, media, context))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
            return JSONResponse(self.view(job), status_code=202)
        except BaseException:
            if root is not None:
                self.jobs.pop(identifier, None)
            session.close()
            if media is not None:
                media.close()
            if root is not None:
                shutil.rmtree(root)
            raise
        finally:
            self.setups -= 1
            if not self.setups:
                self.setups_idle.set()

    async def submit_references(self, request: Request, form) -> JSONResponse:
        """Queue a reference-to-video job from multipart input_reference images and text fields."""
        items = form.multi_items()
        files = [value for _, value in items if isinstance(value, UploadFile)]
        try:
            if any(key != "input_reference" for key, value in items if isinstance(value, UploadFile)):
                raise ValueError("reference images use the input_reference field")
            fields = [(key, value) for key, value in items if not isinstance(value, UploadFile)]
            names = [key for key, _ in fields]
            if (len(names) != len(set(names)) or not set(names) <= REFERENCE_VIDEO_FIELDS
                    or not {"model", "prompt"} <= set(names)):
                raise ValueError("unsupported reference video fields")
            values = dict(fields)
            model, prompt = values["model"], values["prompt"]
            if not prompt.strip() or not 1 <= len(files) <= MAX_REFERENCE_IMAGES:
                raise ValueError("a prompt and 1 to 16 reference images are required")
            if (len(prompt.encode()) > self.config.max_prompt_bytes or
                    len(values.get("continuation_prompt", "").encode()) > self.config.max_prompt_bytes):
                return self.error(413, "content_too_large", "prompt exceeds the server limit")
            config: dict[str, Any] = {}
            if "seed" in values:
                if not values["seed"].isascii() or not values["seed"].isdecimal():
                    raise ValueError("seed must be a non-negative integer")
                config["seed"] = int(values["seed"])
            if "continuation_prompt" in values:
                config["continuation_prompt"] = values["continuation_prompt"]
            if not self.registry.supports(model, "reference_video_generation"):
                return self.error(400, "unsupported_task", "model does not support reference video generation")
            reserved = REFERENCE_VIDEO_RESERVE_BYTES
            if (len(self.jobs) >= self.config.max_video_jobs or
                    sum(job["reserved"] for job in self.jobs.values()) + reserved > self.config.max_video_bytes):
                return self.error(429, "video_storage_full", "video job storage is full; retry after expiry")
            session = self.registry.acquire(model)
        except ModelNotFoundError as error:
            return self.error(404, error.code, str(error))
        except WorkerSaturatedError:
            return self.error(429, "server_busy", "all model worker replicas are busy")
        except ValueError:
            return self.error(400, "invalid_request", "invalid or unsupported reference video request")
        except RuntimeError:
            return self.error(503, "server_draining", "server is not accepting requests")
        self.setups += 1
        self.setups_idle.clear()
        root = None
        identifier = None
        try:
            root = Path(tempfile.mkdtemp(prefix="trtmc-video-"))
            identifier = f"video-{uuid.uuid4().hex}"
            job = {"id": identifier, "object": "video", "created_at": int(time.time()),
                   "model": model, "status": "in_progress", "root": root,
                   "reserved": reserved, "downloads": 0, "active": True,
                   "size": None, "seconds": None, "num_frames": None, "fps": None}
            # Hold the reservation before the first await so concurrent submissions see it.
            self.jobs[identifier] = job
            (root / "references").mkdir()
            paths = []
            pixels = 0
            for index, upload in enumerate(files):
                # The same RGB PNG/JPEG decode and pixel limit as image edits, for all references together.
                data, suffix, metadata = await asyncio.to_thread(
                    image_file, await upload.read(), self.config.max_image_pixels - pixels)
                pixels += metadata["image_pixels"]
                path = root / "references" / f"reference-{index:02d}{suffix}"
                path.write_bytes(data)
                paths.append(str(path))
            request.state.timing.update(model=model, record_kind="video_submission", job_id=identifier)
            payload = {"prompt": prompt, "config": config, "output_path": str(root / "frames"),
                       "audio_path": str(root / "audio.wav"), "reference_paths": paths}
            context = {**request.state.timing, "record_kind": "video_completion", "job_id": identifier,
                       "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                       "generation_config": {k: v for k, v in config.items() if k != "continuation_prompt"},
                       "reference_count": len(paths)}
            task = asyncio.create_task(self.run_references(job, request, payload, session, context))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
            return JSONResponse(self.view(job), status_code=202)
        except ValueError as error:
            session.close()
            if root is not None:
                self.jobs.pop(identifier, None)
                shutil.rmtree(root)
            if isinstance(error, ImagePixelLimitError):
                return self.error(413, "content_too_large", "reference images exceed the pixel limit")
            return self.error(400, "invalid_request", "invalid or unsupported reference video request")
        except BaseException:
            session.close()
            if root is not None:
                self.jobs.pop(identifier, None)
                shutil.rmtree(root)
            raise
        finally:
            self.setups -= 1
            if not self.setups:
                self.setups_idle.set()

    async def run_references(self, job, request, payload, session, context):
        started = time.monotonic()

        def validate(result):
            extract_task_timing(result)
            if (not all(type(result.get(key)) is int and result[key] > 0 for key in ("width", "height", "num_frames"))
                    or result.get("channels") != 3 or not isinstance(result.get("fps"), (int, float))):
                raise WorkerProtocolError("native reference video result is incomplete")

        try:
            result = await self.execute(request, job["model"], "reference_video_generation", "generate_reference_video",
                                        payload, validate, acquired_session=session, task_context=context)
            if isinstance(result, JSONResponse):
                context["status"] = result.status_code
                job["status"] = "failed"
                return
            fps = round(result["fps"])
            if fps < 1 or abs(result["fps"] - fps) > 1e-3:
                raise ValueError("reference video frame rate is not an integer")
            width, height, count = result["width"], result["height"], result["num_frames"]
            if width * height * count * 3 > job["reserved"]:
                raise ValueError("reference video output exceeds the job's storage reservation")
            context.update(await asyncio.to_thread(encode, job["root"] / "frames", job["root"] / "video.mp4",
                                                   width, height, count, fps, job["reserved"],
                                                   Path(payload["audio_path"])))
            job.update(size=f"{width}x{height}", seconds=format(count / fps, ".9g"), num_frames=count, fps=fps,
                       model_call_ms=result["model_call_ms"], completed_at=int(time.time()),
                       expires_at=int(time.time() + self.config.video_ttl_seconds))
            job["status"] = "completed"
            context["status"] = 200
        except Exception:
            logging.getLogger("uvicorn.error").exception("Reference video completion failed: %s", job["id"])
            job["status"] = "failed"
            context["status"] = 502
        finally:
            job["finished"] = time.monotonic()
            context["completion_ms"] = (time.monotonic() - started) * 1000
            try:
                self.records.write(context)
            finally:
                for name in ("frames", "references"):
                    await asyncio.to_thread(shutil.rmtree, job["root"] / name, True)
                (job["root"] / "audio.wav").unlink(missing_ok=True)
            video = job["root"] / "video.mp4"
            job["reserved"] = video.stat().st_size if video.is_file() else 0
            job["active"] = False

    async def run(self, job, request, payload, geometry, fps, session, media, context):
        started = time.monotonic()

        def validate(result):
            extract_task_timing(result)
            if (result.get("width"), result.get("height"), result.get("num_frames"), result.get("channels")) != (
                    geometry["width"], geometry["height"], geometry["num_frames"], 3):
                raise WorkerProtocolError("native video geometry differs from the built clip")

        try:
            result = await self.execute(request, job["model"], "video_generation", "generate_video", payload,
                                        validate, release_input=media.close, acquired_session=session,
                                        task_context=context)
            if isinstance(result, JSONResponse):
                context["status"] = result.status_code
                job["status"] = "failed"
                return
            context.update(await asyncio.to_thread(encode, job["root"] / "frames", job["root"] / "video.mp4",
                            geometry["width"], geometry["height"], geometry["num_frames"], fps,
                            job["reserved"] // 2))
            job.update(model_call_ms=result["model_call_ms"], completed_at=int(time.time()),
                       expires_at=int(time.time() + self.config.video_ttl_seconds))
            job["status"] = "completed"
            context["status"] = 200
        except Exception:
            logging.getLogger("uvicorn.error").exception("Video completion failed: %s", job["id"])
            job["status"] = "failed"
            context["status"] = 502
        finally:
            job["finished"] = time.monotonic()
            context["completion_ms"] = (time.monotonic() - started) * 1000
            try:
                self.records.write(context)
            finally:
                await asyncio.to_thread(shutil.rmtree, job["root"] / "frames", True)
            job["reserved"] = (job["root"] / "video.mp4").stat().st_size if (job["root"] / "video.mp4").is_file() else 0
            job["active"] = False

    async def poll(self, job_id: str):
        await self.expire()
        job = self.jobs.get(job_id)
        return self.view(job) if job else self.error(404, "video_not_found", "video job not found or expired")

    async def content(self, job_id: str):
        await self.expire()
        job = self.jobs.get(job_id)
        if job is None:
            return self.error(404, "video_not_found", "video job not found or expired")
        if job["status"] != "completed":
            return self.error(409, "video_not_ready", "video job has no completed content")
        job["downloads"] += 1
        self.downloads_idle.clear()
        jobs = self

        async def stream():
            with (job["root"] / "video.mp4").open("rb") as file:
                while data := await asyncio.to_thread(file.read, 1024 * 1024):
                    yield data

        class DownloadResponse(StreamingResponse):
            async def __call__(self, scope, receive, send):
                try:
                    await super().__call__(scope, receive, send)
                finally:
                    job["downloads"] -= 1
                    if not any(value["downloads"] for value in jobs.jobs.values()):
                        jobs.downloads_idle.set()

        return DownloadResponse(stream(), media_type="video/mp4")
