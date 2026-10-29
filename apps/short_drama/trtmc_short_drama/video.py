# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FFmpeg helpers: vertical 1080x1920 clips, contact sheets, speech audio and episode export."""

from __future__ import annotations

import io
import json
from pathlib import Path
import re
import subprocess
import wave

WIDTH, HEIGHT, FPS = 1080, 1920, 24
TIMEOUT = 1800


def _run(*arguments: str, cwd: Path | None = None) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *arguments], check=True, cwd=cwd,
                   stdout=subprocess.DEVNULL, timeout=TIMEOUT)


def probe(path: Path) -> dict:
    output = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json",
                             str(path)], capture_output=True, text=True, check=True, timeout=60).stdout
    info = json.loads(output)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    return {"width": video["width"], "height": video["height"],
            "duration": round(float(info["format"]["duration"]), 3),
            "audio": any(s["codec_type"] == "audio" for s in info["streams"])}


def finish_clip(raw: Path, clip: Path, *, seconds: float) -> None:
    """Trim to the planned length and scale to 1080x1920 at 24 fps with 48 kHz stereo audio."""

    clip.parent.mkdir(parents=True, exist_ok=True)
    partial = clip.with_name(f".{clip.name}")
    _run("-i", str(raw), "-t", f"{seconds:.3f}",
         "-vf", f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={WIDTH}:{HEIGHT},fps={FPS},format=yuv420p",
         "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-c:a", "aac", "-b:a", "192k",
         "-ar", "48000", "-ac", "2", "-movflags", "+faststart", "-f", "mp4", str(partial))
    partial.replace(clip)


def contact_sheet(clip: Path, sheet: Path, *, tiles: int = 8) -> None:
    sheet.parent.mkdir(parents=True, exist_ok=True)
    rate = tiles / max(probe(clip)["duration"], 0.1)
    _run("-i", str(clip), "-vf", f"fps={rate:.4f},scale=180:320,tile={tiles}x1:padding=4:color=white",
         "-frames:v", "1", str(sheet))


def speech_wav(clip: Path) -> bytes:
    """Mono 16 kHz 16-bit PCM WAV of a clip, the input the speech recognizer expects.

    FFmpeg cannot fill in WAV sizes on a pipe, so read raw samples and write the header here.
    """

    samples = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(clip), "-vn", "-ac", "1",
                              "-ar", "16000", "-f", "s16le", "pipe:1"],
                             capture_output=True, check=True, timeout=TIMEOUT).stdout
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(samples)
    return output.getvalue()


def last_frame(clip: Path, image: Path) -> None:
    _run("-sseof", "-0.2", "-i", str(clip), "-frames:v", "1", "-update", "1", str(image))


def export_episode(clips: list[Path], subtitles: Path, output: Path, *, max_seconds: float,
                   font: str, font_size: int, crf: int) -> None:
    """Concatenate the clips, burn the subtitles and cap the length."""

    if not re.fullmatch(r"[A-Za-z0-9 ]{1,64}", font):  # it goes into an FFmpeg filter string
        raise ValueError("subtitle font names use letters, digits and spaces")
    directory = output.parent
    listing = directory / "clips.txt"
    listing.write_text("".join(f"file '{clip.resolve()}'\n" for clip in clips), encoding="utf-8")
    style = f"FontName={font},FontSize={int(font_size)},MarginV=40,Outline=1,Shadow=0"
    partial = output.with_name(f".{output.name}")
    _run("-f", "concat", "-safe", "0", "-i", listing.name,
         "-vf", f"subtitles={subtitles.name}:force_style='{style}'",
         "-t", f"{max_seconds:.3f}", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-f", "mp4", partial.name,
         cwd=directory)
    partial.replace(output)
