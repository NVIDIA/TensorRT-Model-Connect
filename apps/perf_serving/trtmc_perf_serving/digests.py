# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact, backend-independent digests of generated media.

Generated images, video frames, and audio are written as files (PNG/WAV by the TRTMC worker,
``.npy`` arrays by the Python references). Parity graders and goldens need a small value that
both sides produce the same way, so the server replaces the files with digests: a 64x64 RGB
thumbnail per sampled frame and a banded log power spectrum for audio.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

THUMBNAIL = 64
SPECTRUM_BANDS = 64
IMAGE_OPERATIONS = ("generate_image",)
AUDIO_OPERATIONS = ("generate_audio", "speak")


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.moveaxis(frame, 0, -1)  # CHW -> HWC
    if frame.ndim == 2:
        frame = frame[..., None]
    if np.issubdtype(frame.dtype, np.floating):
        low = -1.0 if float(np.min(frame)) < -0.01 else 0.0  # [-1, 1] or [0, 1] ranges
        frame = (np.clip(frame, low, 1.0) - low) / (1.0 - low) * 255.0 + 0.5
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    return frame[..., :3]


def image_digest(frames: Sequence[np.ndarray]) -> dict[str, Any]:
    """Thumbnails of the first, middle, and last frame plus the frame geometry."""
    from PIL import Image

    frames = [_to_uint8(frame) for frame in frames]
    if not frames:
        raise ValueError("no frames to digest")
    picks = sorted({0, len(frames) // 2, len(frames) - 1})
    thumbnails = []
    for index in picks:
        image = Image.fromarray(frames[index]).resize((THUMBNAIL, THUMBNAIL), Image.BILINEAR)
        thumbnails.append(np.asarray(image, dtype=np.uint8).flatten().tolist())
    height, width = frames[0].shape[:2]
    return {"frames": len(frames), "height": int(height), "width": int(width), "sampled_frames": picks,
            "thumbnail_size": THUMBNAIL, "thumbnails": thumbnails}


def audio_digest(samples: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Duration, RMS, and the mean log power spectrum in equal-width frequency bands."""
    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim > 1:  # [channels, samples] or [samples, channels]
        audio = audio.mean(axis=0 if audio.shape[0] < audio.shape[-1] else -1)
    audio = audio.reshape(-1)
    length = audio.size
    n_fft, hop = 1024, 512
    if audio.size < n_fft:
        audio = np.pad(audio, (0, n_fft - audio.size))
    frames = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::hop] * np.hanning(n_fft)
    power = (np.abs(np.fft.rfft(frames, axis=-1)) ** 2).mean(axis=0)
    bands = np.array_split(power, SPECTRUM_BANDS)
    spectrum = [10.0 * math.log10(float(band.mean()) + 1e-12) for band in bands]
    return {"seconds": length / max(int(sample_rate), 1), "sample_rate": int(sample_rate),
            "rms": float(np.sqrt(np.mean(audio[:length] ** 2))), "log_spectrum": spectrum}


def _read_png(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _media_frames(array: np.ndarray) -> list[np.ndarray]:
    """Frames of a reference media array: [H,W,C], [N,H,W,C], or [N,T,H,W,C] (first item)."""
    array = np.asarray(array)
    while array.ndim > 4:
        array = array[0]
    if array.ndim == 4:
        return list(array)
    return [array]


def media_frames(workdir: Path) -> list[np.ndarray]:
    """Generated frames of one request (uint8 HWC): the worker's PNGs in frame order, else the
    reference's ``output.npy``; empty when the request wrote neither."""
    pngs = sorted((path for path in workdir.glob("output*.png") if ".input." not in path.name),
                  key=lambda path: [(0, int(part), "") if part.isdigit() else (1, 0, part)
                                    for part in path.name.split(".")])
    if pngs:
        return [_read_png(path) for path in pngs]
    if (workdir / "output.npy").is_file():
        return [_to_uint8(frame) for frame in _media_frames(np.load(workdir / "output.npy"))]
    return []


def add_media_digests(observation: Mapping[str, Any], operation: str, workdir: Path) -> dict[str, Any]:
    """Attach ``media_digest``/``audio_digest`` from the request's output files, when present."""
    result = dict(observation)
    if operation in IMAGE_OPERATIONS and "media_digest" not in result:
        frames = media_frames(workdir)
        if frames:
            result["media_digest"] = image_digest(frames)
    if operation in AUDIO_OPERATIONS and "audio_digest" not in result:
        wavs = sorted(workdir.glob("output*.wav"))
        if wavs:
            import soundfile

            samples, rate = soundfile.read(wavs[0], dtype="float32", always_2d=False)
            result["audio_digest"] = audio_digest(samples, rate)
        elif (workdir / "output.npy").is_file() and result.get("sample_rate"):
            result["audio_digest"] = audio_digest(np.load(workdir / "output.npy"), int(result["sample_rate"]))
    return result
