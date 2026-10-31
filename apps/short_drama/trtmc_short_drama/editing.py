# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The subtitles step (speech recognition checks and times each line) and the assemble step."""

from __future__ import annotations

from difflib import SequenceMatcher
import io
import re
from typing import Callable
import wave

from . import video
from .flow import Context, Step

STEP_SECONDS = 0.1
HEARD = 0.8  # share of the full-clip match a cut must keep to still contain the line
SECONDS_PER_CHARACTER = 0.18  # a subtitle stays up at least this long per character


def normalize(text: str) -> str:
    return "".join(re.findall(r"[\w一-鿿]", text or "")).lower()


def coverage(heard: str, line: str) -> float:
    """Share of the line's characters that appear, in order, in what was heard."""

    line, heard = normalize(line), normalize(heard)
    if not line:
        return 0.0
    blocks = SequenceMatcher(None, line, heard, autojunk=False).get_matching_blocks()
    return sum(block.size for block in blocks) / len(line)


def cut(wav: bytes, start: float, end: float) -> bytes:
    with wave.open(io.BytesIO(wav)) as source:
        rate, width, channels = source.getframerate(), source.getsampwidth(), source.getnchannels()
        frames = source.readframes(source.getnframes())
    step = width * channels
    data = frames[int(start * rate) * step:int(end * rate) * step]
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(width)
        target.setframerate(rate)
        target.writeframes(data)
    return output.getvalue()


def duration(wav: bytes) -> float:
    with wave.open(io.BytesIO(wav)) as source:
        return source.getnframes() / source.getframerate()


def time_line(transcribe: Callable[[bytes], str], wav: bytes, line: str) -> dict:
    """Find when a line is spoken using only a text recognizer.

    The end is the shortest prefix that still contains the line; the start is the latest cut
    whose remainder still contains it. Each is a binary search over 0.1 s steps.
    """

    total = duration(wav)
    heard = transcribe(wav)
    full = coverage(heard, line)
    result = {"heard": heard, "match": round(full, 2)}
    if full < 0.3:
        return {**result, "start": 0.3, "end": min(total, 0.3 + 0.28 * len(normalize(line))),
                "timing": "estimated"}

    def contains(start: float, end: float) -> bool:
        return end - start >= 0.3 and coverage(transcribe(cut(wav, start, end)), line) >= HEARD * full

    low, high = 0, int(total / STEP_SECONDS)
    while low < high:
        middle = (low + high) // 2
        if contains(0.0, middle * STEP_SECONDS):
            high = middle
        else:
            low = middle + 1
    end = min(total, high * STEP_SECONDS + 0.2)
    low, high = 0, int(end / STEP_SECONDS)
    while low < high:
        middle = (low + high + 1) // 2
        if contains(middle * STEP_SECONDS, end):
            low = middle
        else:
            high = middle - 1
    # A weak match (homophones, other scripts) can shrink the window; keep it readable.
    start = min(low * STEP_SECONDS - 0.1, end - SECONDS_PER_CHARACTER * len(normalize(line)))
    return {**result, "start": round(max(0.0, start), 2), "end": round(end, 2), "timing": "asr"}


def run_subtitles(context: Context) -> None:
    project, number = context.project, context.episode
    episode = next(e for e in project.plan["episodes"] if e["number"] == number)
    state = project.shot_state(number)
    spoken = [s for s in episode["shots"] if s.get("line") and state.get(s["id"], {}).get("file")]
    if spoken:
        context.services.load(context.model)
    for index, shot in enumerate(spoken, start=1):
        context.check()
        context.progress(f"episode {number}: timing line {index} of {len(spoken)}")
        clip = state[shot["id"]]["file"]
        timing = time_line(lambda audio: context.services.engine.transcribe(context.model, audio),
                           video.speech_wav(project.root / clip), shot["line"])
        state = project.shot_state(number)  # re-read: the user may pick another take meanwhile
        if state.get(shot["id"], {}).get("file") == clip:
            state[shot["id"]] = {**state[shot["id"]], "speech": timing}
            project.save_shot_state(number, state)


def srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, rest = divmod(milliseconds, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    return f"{hours:02d}:{minutes:02d}:{rest // 1000:02d},{rest % 1000:03d}"


def subtitles(shots: list[dict], state: dict) -> str:
    """SRT of the planned lines placed where the recognizer heard them."""

    entries, offset = [], 0.0
    for shot in shots:
        speech = state.get(shot["id"], {}).get("speech")
        if shot.get("line") and speech:
            start = offset + speech["start"]
            end = offset + max(speech["end"], speech["start"] + 0.8)
            end = min(end, offset + float(shot["seconds"]))
            text = re.sub(r"[\"“”]", "", shot["line"])
            entries.append(f"{len(entries) + 1}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n")
        offset += float(shot["seconds"])
    return "\n".join(entries)


def run_assemble(context: Context) -> None:
    project, params, number = context.project, context.params, context.episode
    episode = next(e for e in project.plan["episodes"] if e["number"] == number)
    state = project.shot_state(number)
    shots = [s for s in episode["shots"] if state.get(s["id"], {}).get("file")]
    if len(shots) != len(episode["shots"]):
        raise ValueError(f"render every shot of episode {number} first")
    directory = project.episode_dir(number)
    (directory / "episode.srt").write_text(subtitles(shots, state), encoding="utf-8")
    output = directory / "episode.mp4"
    context.progress(f"episode {number}: joining {len(shots)} clips")
    video.export_episode([project.root / state[s["id"]]["file"] for s in shots], directory / "episode.srt",
                         output, max_seconds=float(project.meta.get("max_seconds", 60)),
                         font=params.get("font", "Noto Sans CJK SC"),
                         font_size=int(params.get("font_size", 12)), crf=int(params.get("crf", 18)))
    video.last_frame(output, directory / "last_frame.png")
    state["assembled"] = video.probe(output)
    project.save_shot_state(number, state)
    project.log(f"episode {number} assembled: {state['assembled']['duration']} s")


SUBTITLES = Step("subtitles", run_subtitles, role="asr", scope="episode",
                 description="Check every line with speech recognition and time its subtitle")
ASSEMBLE = Step("assemble", run_assemble, scope="episode",
                description="Join the clips, burn in subtitles and export the episode")
