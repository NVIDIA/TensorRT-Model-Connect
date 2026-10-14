# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate uploaded media containers without model preprocessing or resampling."""

from __future__ import annotations

import io
import tempfile
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


def image_upload(url: str) -> tuple[bytes, str, dict[str, Any]]:
    """Accept one embedded RGB image; never fetch client-provided URLs or paths."""
    import base64
    import binascii
    import warnings

    from PIL import Image

    prefixes = {"data:image/png;base64,": ("PNG", ".png"),
                "data:image/jpeg;base64,": ("JPEG", ".jpg")}
    prefix = next((item for item in prefixes if url.startswith(item)), None)
    if prefix is None:
        raise ValueError("image must be a base64 PNG or JPEG data URL")
    try:
        data = base64.b64decode(url[len(prefix):], validate=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format != prefixes[prefix][0] or image.mode != "RGB" or getattr(image, "n_frames", 1) != 1:
                    raise ValueError("image must contain one RGB PNG or JPEG frame")
                width, height = image.size
                if not 0 < width <= 2**31 - 1 or not 0 < height <= 2**31 - 1:
                    raise ValueError("image dimensions are invalid")
                image.verify()
    except (binascii.Error, OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError("image must be a readable RGB PNG or JPEG") from error
    return data, prefixes[prefix][1], {"input_items": 1, "image_bytes": len(data),
                                      "image_height": height, "image_width": width,
                                      "image_pixels": height * width}
