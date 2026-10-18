# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lossless clip packaging and complete RGB frame identity, outside Task timing."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
import subprocess


def frame_identity(directory: Path, width: int, height: int, count: int) -> str:
    from PIL import Image

    paths = sorted(directory.glob("*.png"))
    if [p.name for p in paths] != [f"frame-{i:06d}.png" for i in range(count)]:
        raise ValueError("native video frame sequence is incomplete or out of order")
    digest = hashlib.sha256()
    for path in paths:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG" or image.mode != "RGB" or image.size != (width, height):
                raise ValueError("native video frame geometry differs from the built clip")
            digest.update(image.tobytes())
    return digest.hexdigest()


def decoded_identity(path: Path, width: int, height: int, count: int, *,
                     output: Path | None = None, fps: int | None = None) -> str:
    """Decode every frame, retaining sequence order; reject partial or extra frames."""
    from PIL import Image

    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                       "-show_entries", "stream=width,height,avg_frame_rate", "-of", "json", str(path)], timeout=60))
    streams = probe.get("streams", [])
    if len(streams) != 1 or (streams[0].get("width"), streams[0].get("height")) != (width, height):
        raise ValueError("downloaded video dimensions differ from the native clip")
    if fps is not None and Fraction(streams[0]["avg_frame_rate"]) != fps:
        raise ValueError("downloaded video FPS differs from the requested presentation")
    digest = hashlib.sha256()
    process = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
                                "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        assert process.stdout is not None
        for index in range(count):
            data = process.stdout.read(width * height * 3)
            if len(data) != width * height * 3:
                raise ValueError("downloaded video is truncated or has different dimensions")
            digest.update(data)
            if output is not None:
                Image.frombytes("RGB", (width, height), data).save(output / f"frame-{index:06d}.png")
        if process.stdout.read(1) or process.wait(timeout=120) != 0:
            raise ValueError("downloaded video contains extra frames or cannot be decoded")
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        if process.stdout is not None:
            process.stdout.close()
    return digest.hexdigest()


def encode(directory: Path, path: Path, width: int, height: int, count: int,
           fps: int, max_bytes: int) -> dict:
    pixels = frame_identity(directory, width, height, count)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-framerate", str(fps),
                    "-start_number", "0", "-i", str(directory / "frame-%06d.png"),
                    "-frames:v", str(count), "-an", "-c:v", "libx264rgb", "-preset", "ultrafast",
                    "-threads", "1", "-crf", "0", "-pix_fmt", "rgb24", "-movflags", "+faststart",
                    "-fs", str(max_bytes), "-y", str(path)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=600)
    if path.stat().st_size > max_bytes or decoded_identity(path, width, height, count, fps=fps) != pixels:
        raise ValueError("encoded video does not preserve the complete native RGB frame sequence")
    with path.open("rb") as file:
        encoded_hash = hashlib.file_digest(file, "sha256").hexdigest()
    return {"output_width": width, "output_height": height, "output_frames": count,
            "output_pixels_sha256": pixels, "output_sha256": encoded_hash,
            "output_bytes": path.stat().st_size, "fps": fps, "codec": "libx264rgb-crf0"}
