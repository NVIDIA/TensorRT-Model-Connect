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
# TRTMC's MiniMax-H3 opens every clip with a loud burst of about 0.15 s that the official samples
# do not have, so clips start this much later. FADE (seconds) keeps cuts from clicking.
LEAD_IN = 0.25
FADE = 0.04
# The FastH3 OmniRef student often opens on one or more reference pictures held still (a portrait,
# or the previous shot's last frame for up to 1.4 s) and then cuts into the shot. A clip starts
# after every such cut within OPENING seconds: frames that all change by less than STILL (grey
# levels, 0-255; held pictures stay under 2.1, inserts and moving shots reach 4.5 and more), then a
# change of at least CUT.
OPENING = 2.5
CUT = 40.0
STILL = 3.0
# Bump when the way a raw clip is cut changes: assembling an episode cuts older takes again.
CUT_RULE = 2
# One loudness for the whole episode, so the level does not jump at every cut.
LOUDNESS = "loudnorm=I=-16:TP=-1.5:LRA=11"
# Mono 16 kHz as the mean of the channels. FFmpeg's own downmix (-ac 1) adds them at 0.71 each, which
# lifts loud stereo past full scale: speech recognition hears it clipped and the analysis calls it clipping.
MONO_16K = "aresample=16000:rematrix_maxval=1,aformat=channel_layouts=mono"
SHEET_TILES = 8  # frames in a contact sheet, left to right


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


def opening(raw: Path) -> float:
    """Where a clip starts: after the held pictures it opens on, else LEAD_IN."""

    width, height = 36, 64
    grey = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-t", f"{OPENING}", "-i", str(raw),
                           "-vf", f"fps={FPS},scale={width}:{height},format=gray", "-f", "rawvideo", "pipe:1"],
                          capture_output=True, check=True, timeout=TIMEOUT).stdout
    size = width * height
    frames = [grey[i:i + size] for i in range(0, len(grey) - size + 1, size)]
    changes = [sum(abs(a - b) for a, b in zip(before, after)) / size for before, after in zip(frames, frames[1:])]
    start = 0  # first frame of the picture being looked at
    for index, change in enumerate(changes):
        if change < CUT:
            continue
        if max(changes[start:index], default=0.0) >= STILL:  # it moves: the shot itself has begun
            break
        start = index + 1
    return max(LEAD_IN, round(start / FPS, 3))


def finish_clip(raw: Path, clip: Path, *, seconds: float, lead_in: float | None = None) -> float:
    """Skip the opening, trim to the planned length and scale to 1080x1920 at 24 fps.

    The audio becomes 48 kHz stereo with short fades, so a cut between two clips does not click.
    Returns where in the raw clip the finished clip starts.
    """

    clip.parent.mkdir(parents=True, exist_ok=True)
    partial = clip.with_name(f".{clip.name}")
    lead_in = opening(raw) if lead_in is None else lead_in
    length = max(min(seconds, probe(raw)["duration"] - lead_in), 0.5)
    _run("-ss", f"{lead_in:.3f}", "-i", str(raw), "-t", f"{length:.3f}",
         "-vf", f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={WIDTH}:{HEIGHT},fps={FPS},format=yuv420p",
         "-af", f"afade=t=in:d={FADE},afade=t=out:st={length - FADE:.3f}:d={FADE}",
         "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-c:a", "aac", "-b:a", "192k",
         "-ar", "48000", "-ac", "2", "-movflags", "+faststart", "-f", "mp4", str(partial))
    partial.replace(clip)
    return lead_in


def contact_sheet(clip: Path, sheet: Path, *, tiles: int = SHEET_TILES) -> None:
    sheet.parent.mkdir(parents=True, exist_ok=True)
    rate = tiles / max(probe(clip)["duration"], 0.1)
    _run("-i", str(clip), "-vf", f"fps={rate:.4f},scale=180:320,tile={tiles}x1:padding=4:color=white",
         "-frames:v", "1", str(sheet))


def speech_wav(clip: Path) -> bytes:
    """Mono 16 kHz 16-bit PCM WAV of a clip, the input the speech recognizer expects.

    FFmpeg cannot fill in WAV sizes on a pipe, so read raw samples and write the header here.
    """

    samples = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(clip), "-vn", "-af", MONO_16K,
                              "-f", "s16le", "pipe:1"],
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
         "-vf", f"subtitles={subtitles.name}:force_style='{style}'", "-af", LOUDNESS,
         "-t", f"{max_seconds:.3f}", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", "-f", "mp4",
         partial.name, cwd=directory)
    partial.replace(output)
