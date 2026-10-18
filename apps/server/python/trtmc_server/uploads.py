# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate uploaded media containers without model preprocessing or resampling."""

from __future__ import annotations

import asyncio
import io
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any

from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.requests import Request


async def multipart_audio(request: Request, limit: int) -> tuple[dict[str, Any], bytes]:
    try:
        async with request.form(max_files=1, max_fields=4) as form:
            if any(len(form.getlist(key)) != 1 for key in form):
                raise ValueError("multipart fields must occur once")
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise ValueError("file must contain one uploaded WAV clip")
            data = await upload.read(limit + 1)
            if len(data) > limit:
                raise ValueError("uploaded audio exceeds the server limit")
            return {key: value for key, value in form.items() if key != "file"}, data
    except HTTPException as error:
        raise ValueError("invalid multipart WAV upload") from error


async def multipart_image(request: Request, limit: int) -> tuple[dict[str, Any], bytes]:
    """Read one reference image and bounded text controls, without client paths."""
    try:
        async with request.form(max_files=1, max_fields=16, max_part_size=limit) as form:
            if any(len(form.getlist(key)) != 1 for key in form):
                raise ValueError("multipart fields must occur once")
            upload = form.get("image")
            if not isinstance(upload, UploadFile):
                raise ValueError("image must contain one uploaded reference image")
            fields = {key: value for key, value in form.items() if key != "image"}
            if any(not isinstance(value, str) for value in fields.values()):
                raise ValueError("only one image upload is supported")
            data = await upload.read(limit + 1)
            if len(data) > limit:
                raise ValueError("uploaded image exceeds the server limit")
            return fields, data
    except HTTPException as error:
        raise ValueError("invalid multipart image upload") from error


def image_file(data: bytes, max_pixels: int) -> tuple[bytes, str, dict[str, Any]]:
    """Use the existing image validator for an uploaded PNG or JPEG container."""
    import base64

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        kind = "png"
    elif data.startswith(b"\xff\xd8\xff"):
        kind = "jpeg"
    else:
        raise ValueError("image must be a PNG or JPEG file")
    return image_upload(f"data:image/{kind};base64," + base64.b64encode(data).decode(), max_pixels)


def wav_metadata(data: bytes) -> dict[str, Any]:
    try:
        with wave.open(io.BytesIO(data), "rb") as audio:
            channels, width, rate, frames = (
                audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()
            )
            if channels != 1 or width != 2 or audio.getcomptype() != "NONE":
                raise ValueError("audio must be a mono 16-bit PCM WAV file")
            if not 0 < rate <= 2**31 - 1 or frames <= 0:
                raise ValueError("audio must contain samples and a positive sample rate")
            if len(audio.readframes(frames)) != frames * width:
                raise ValueError("audio contains truncated PCM samples")
    except (wave.Error, EOFError, OSError) as error:
        raise ValueError("audio must be a readable PCM WAV file") from error
    return {"input_items": 1, "audio_bytes": len(data), "audio_channels": channels,
            "audio_sample_rate": rate, "audio_sample_count": frames,
            "audio_duration_seconds": frames / rate}


class UploadedInput:
    """Keep an upload alive until its worker operation (including cancellation) ends."""

    def __init__(self, data: bytes, suffix: str) -> None:
        self._directory = tempfile.TemporaryDirectory(prefix="trtmc-input-")
        self.path = Path(self._directory.name) / f"input{suffix}"
        try:
            self.path.write_bytes(data)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._directory.cleanup()


async def create_uploaded_input(data: bytes, suffix: str) -> UploadedInput:
    """Write off the event loop and reclaim a cancelled, in-flight creation."""
    created: list[UploadedInput] = []
    finished = threading.Event()

    def create() -> UploadedInput:
        try:
            media = UploadedInput(data, suffix)
            created.append(media)
            return media
        finally:
            finished.set()

    def consume(completed: asyncio.Future[Any]) -> None:
        if not completed.cancelled():
            completed.exception()

    pending = asyncio.create_task(asyncio.to_thread(create))
    pending.add_done_callback(consume)
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        def cleanup() -> None:
            finished.wait()
            for media in created:
                media.close()

        cleanup_task = asyncio.get_running_loop().run_in_executor(None, cleanup)
        cleanup_task.add_done_callback(consume)
        raise


class ImagePixelLimitError(ValueError):
    def __init__(self, metadata: dict[str, Any]) -> None:
        super().__init__("image exceeds the server pixel limit")
        self.metadata = metadata


def image_upload(url: str, max_pixels: int) -> tuple[bytes, str, dict[str, Any]]:
    """Accept one embedded RGB image; never fetch client-provided URLs or paths."""
    import base64
    import binascii
    from PIL import Image

    prefixes = {"data:image/png;base64,": ("PNG", ".png"),
                "data:image/jpeg;base64,": ("JPEG", ".jpg")}
    prefix = next((item for item in prefixes if url.startswith(item)), None)
    if prefix is None:
        raise ValueError("image must be a base64 PNG or JPEG data URL")
    try:
        data = base64.b64decode(url[len(prefix):], validate=True)
        with Image.open(io.BytesIO(data)) as image:
            if image.format != prefixes[prefix][0] or image.mode != "RGB" or getattr(image, "n_frames", 1) != 1:
                raise ValueError("image must contain one RGB PNG or JPEG frame")
            width, height = image.size
            if not 0 < width <= 2**31 - 1 or not 0 < height <= 2**31 - 1:
                raise ValueError("image dimensions are invalid")
            metadata = {"input_items": 1, "image_bytes": len(data),
                        "image_height": height, "image_width": width,
                        "image_pixels": height * width}
            if metadata["image_pixels"] > max_pixels:
                raise ImagePixelLimitError(metadata)
            image.verify()
        # verify() checks the container; a fresh decoder must read the pixels.
        with Image.open(io.BytesIO(data)) as image:
            image.load()
    except (binascii.Error, OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError("image must be a readable RGB PNG or JPEG") from error
    return data, prefixes[prefix][1], metadata
