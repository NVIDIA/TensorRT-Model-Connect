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
