# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video analysis: measure what the models made, ask a model that reads pictures what it sees,
and turn both into findings that go back into the plan.

Measured on this machine for every rendered shot, with FFmpeg and NumPy only:
- sound: a burst in the first 0.3 s, clipping, a nearly silent clip, the level of the shot;
- speech: the planned line against speech recognition (the subtitles step's result when it has one);
- picture: black or frozen frames;
- cuts: how far the picture and the level jump from one shot to the next.
With a vision model set in Settings, each shot's frames, its reference pictures and the previous
shot's last frame go to that model, which reports content, on-screen text, continuity, whether the
people and places match their pictures, and problems in the pictures themselves.

A finding names the plan item it would change, an action (revise the item with a note, render
another take, redraw a picture) and a suggestion the user can edit before applying it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import numpy as np

from . import editing, media, prompts, video
from .flow import Context, Step
from .store import read_json, write_json

RATE = 16000
WINDOW = 320  # 20 ms
OPENING = 0.3  # seconds searched for a burst
BURST = 8.0  # the opening's loudest 20 ms against the median of the rest
CLIPPING = 0.002  # share of samples at full scale
QUIET_DB = -45.0  # a clip below this level is nearly silent
FRAMES = 12  # frames sampled per clip
THUMB = (36, 64)  # width and height of the frames compared
STILL = 1.5  # mean change between sampled frames (0-255) under which the picture barely moves
JUMP = 0.45  # picture change at a cut, 0 (same) to 1 (unrelated), above which a same-place cut jumps
LEVEL_JUMP_DB = 8.0
HEARD = 0.6  # share of the planned line speech recognition must find
VISION_IMAGES = 4  # the frames, at most two reference pictures and the previous shot's last frame
TARGET = re.compile(r"(shot|character|location|prop):([a-z0-9_]{1,40})")
CATEGORIES = ("content", "text", "continuity", "consistency", "reference")
ACTIONS = ("revise", "retake", "redraw")

# (finding, suggestion) of each measured problem, in the project's language.
MESSAGES = {
    "zh": {
        "burst": ("开头 {seconds} 秒内有一声很响的短音（比其余部分响 {ratio} 倍）", "重新渲染一条"),
        "clipping": ("声音过载失真（{share}% 的采样到顶）", "重新渲染一条"),
        "quiet": ("几乎没有声音（{level} dB）", "重新渲染一条，并在声音描述里写清环境声和动作声"),
        "missing": ("没有听到计划的台词「{line}」（识别到：{heard}）",
                    "台词再短一些，放在镜头前半段，由说话的人面向镜头清楚地说出"),
        "partial": ("台词只听清了一部分（匹配 {match}%，识别到：{heard}）", "重新渲染一条"),
        "unplanned": ("这个镜头没有台词，却听到了说话声：{heard}", "写明这个镜头里没有人说话，只有动作和环境声"),
        "black": ("{share}% 的画面几乎全黑", "重新渲染一条"),
        "still": ("画面几乎不动", "给这个镜头一个清楚、连贯的主要动作"),
        "jump": ("从 {before} 切到 {after} 时画面变化很大（{score}），同一场景却不像接着拍的",
                 "从 {before} 结束的地方开始：人物位置、服装、道具和光线保持一致"),
        "level": ("从 {before} 切到 {after} 时音量跳变 {db} dB", "拼接时已统一响度，一般不用处理"),
    },
    "en": {
        "burst": ("a loud burst in the first {seconds} s ({ratio}x the rest)", "Render another take"),
        "clipping": ("the sound clips ({share}% of the samples at full scale)", "Render another take"),
        "quiet": ("nearly silent ({level} dB)", "Render another take and describe the ambience and the action sounds"),
        "missing": ("the planned line “{line}” is not heard (heard: {heard})",
                    "Make the line shorter and say it early, clearly, facing the camera"),
        "partial": ("only part of the line is heard (match {match}%, heard: {heard})", "Render another take"),
        "unplanned": ("someone speaks although the shot has no line: {heard}",
                      "State that nobody speaks in this shot; only the action and the ambience are heard"),
        "black": ("{share}% of the frames are nearly black", "Render another take"),
        "still": ("the picture barely moves", "Give the shot one clear, continuous movement"),
        "jump": ("the picture changes a lot from {before} to {after} ({score}) although both are in one place",
                 "Open where {before} ends: the same positions, clothes, props and light"),
        "level": ("the level jumps {db} dB from {before} to {after}", "Assembly evens out the level; usually nothing to do"),
    },
}

VISION_SYSTEM = ("You check clips of a vertical short drama for its director. Reply with one JSON value and "
                 "nothing else; never use the double-quote character inside strings.")

VISION_PROMPT = """Check shot {shot} of the vertical short drama "{title}" (style: {style}).

The plan of the shot: {plan}
People, places and props in it: {cast}
Speech recognition heard: {heard}
Measured on the clip: {measured}

Image 1 shows {frames} frames of the clip from left to right over {seconds} s.{extra}

Report only problems a viewer would notice, at most 5, each of one category:
- "content": the action, the camera or the place does not match the plan;
- "text": letters, captions or garbled writing appear on screen;
- "continuity": the clip does not open where the previous shot ended (positions, clothes, props, light);
- "consistency": a person or place does not look like its reference picture;
- "reference": a reference picture itself does not fit the plan.
Return a JSON array of {{"category": "<category>", "severity": "high|medium|low",
"finding": "<one sentence in {language}>", "target": "shot:{shot}" or "character:<id>" or "location:<id>"
or "prop:<id>", "action": "revise|retake|redraw", "suggestion": "<the smallest change, as a director's
note in {language}>"}}. Use "redraw" for a character, place or prop whose picture is wrong, "revise" to
change the plan, and "retake" when the plan is fine but the clip went wrong. Return [] when the clip is fine."""


def _ffmpeg(*arguments: str) -> bytes:
    return subprocess.run(["ffmpeg", "-nostdin", "-v", "error", *arguments], capture_output=True, check=True,
                          timeout=video.TIMEOUT).stdout


def samples(clip: Path) -> np.ndarray:
    return np.frombuffer(_ffmpeg("-i", str(clip), "-vn", "-ac", "1", "-ar", str(RATE), "-f", "f32le", "pipe:1"),
                         dtype=np.float32)


def sound(audio: np.ndarray) -> dict[str, Any]:
    """Level (dB), the opening burst ratio and the clipped share of a mono 16 kHz signal."""

    if len(audio) < WINDOW * 8:
        return {"level_db": None, "burst": 0.0, "clipping": 0.0}
    rms = np.sqrt((audio[: len(audio) // WINDOW * WINDOW].reshape(-1, WINDOW) ** 2).mean(axis=1)) + 1e-6
    opening = int(OPENING * RATE / WINDOW)
    rest = float(np.median(rms[opening:]))
    return {"level_db": round(20 * math.log10(float(np.sqrt((audio ** 2).mean())) + 1e-9), 1),
            "burst": round(float(rms[:opening].max()) / rest, 1),
            "clipping": round(float((np.abs(audio) >= 0.99).mean()), 4)}


def frames(clip: Path, seconds: float, count: int = FRAMES) -> np.ndarray:
    width, height = THUMB
    raw = _ffmpeg("-i", str(clip), "-vf", f"fps={count / max(seconds, 0.1):.4f},scale={width}:{height}",
                  "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1")
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)


def edge(clip: Path, *, last: bool) -> np.ndarray:
    width, height = THUMB
    where = ["-sseof", "-0.1"] if last else ["-ss", "0"]
    raw = _ffmpeg(*where, "-i", str(clip), "-frames:v", "1", "-vf", f"scale={width}:{height}",
                  "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1")
    return np.frombuffer(raw, dtype=np.uint8)[: width * height * 3].reshape(height, width, 3)


def picture(sampled: np.ndarray) -> dict[str, Any]:
    """Share of nearly black frames and the mean change between neighbouring sampled frames."""

    if not len(sampled):
        return {"black": 0.0, "motion": None}
    luma = sampled.astype(np.float32).mean(axis=3)
    change = np.abs(np.diff(luma, axis=0)).mean(axis=(1, 2)) if len(luma) > 1 else np.zeros(1)
    return {"black": round(float((luma.mean(axis=(1, 2)) < 16).mean()), 2), "motion": round(float(change.mean()), 2)}


def jump(before: np.ndarray, after: np.ndarray) -> float:
    """How far the picture changes at a cut: 0 for the same frame, toward 1 for unrelated pictures.

    Half is the Hellinger distance of the colour histograms, half the mean grey-level difference.
    """

    def histogram(frame: np.ndarray) -> np.ndarray:
        counts = np.concatenate([np.histogram(frame[..., c], bins=16, range=(0, 256))[0] for c in range(3)])
        return counts / counts.sum()

    colour = math.sqrt(max(0.0, 1.0 - float(np.sqrt(histogram(before) * histogram(after)).sum())))
    grey = float(np.abs(before.astype(np.float32).mean(axis=2) - after.astype(np.float32).mean(axis=2)).mean()) / 255
    return round(0.5 * colour + 0.5 * min(1.0, 2.5 * grey), 2)


def _finding(language: str, kind: str, shot: str, category: str, severity: str, action: str, target: str,
             **values: Any) -> dict[str, Any]:
    finding, suggestion = MESSAGES.get(language, MESSAGES["zh"])[kind]
    return {"id": f"{shot}-{kind}", "shot": shot, "category": category, "severity": severity,
            "finding": finding.format(**values), "suggestion": suggestion.format(**values),
            "action": action, "target": target, "source": "measured"}


def measured_findings(language: str, shot: dict[str, Any], measures: dict[str, Any]) -> list[dict[str, Any]]:
    """Findings from the numbers of one shot."""

    sid, target = shot["id"], f"shot:{shot['id']}"
    found = []
    level = measures["sound"]
    if level["burst"] >= BURST:
        found.append(_finding(language, "burst", sid, "sound", "medium", "retake", target,
                              seconds=OPENING, ratio=round(level["burst"])))
    if level["clipping"] >= CLIPPING:
        found.append(_finding(language, "clipping", sid, "sound", "medium", "retake", target,
                              share=round(100 * level["clipping"], 1)))
    if level["level_db"] is not None and level["level_db"] < QUIET_DB:
        found.append(_finding(language, "quiet", sid, "sound", "high", "retake", target, level=level["level_db"]))
    speech = measures.get("speech")
    if speech:
        heard = speech.get("heard") or "—"
        if shot.get("line") and speech["match"] < 0.3:
            found.append(_finding(language, "missing", sid, "speech", "high", "revise", target,
                                  line=shot["line"], heard=heard))
        elif shot.get("line") and speech["match"] < HEARD:
            found.append(_finding(language, "partial", sid, "speech", "medium", "retake", target,
                                  match=round(100 * speech["match"]), heard=heard))
        elif not shot.get("line") and len(editing.normalize(heard)) >= 4:
            found.append(_finding(language, "unplanned", sid, "speech", "medium", "revise", target, heard=heard))
    pic = measures["picture"]
    if pic["black"] >= 0.15:
        found.append(_finding(language, "black", sid, "picture", "high", "retake", target,
                              share=round(100 * pic["black"])))
    if pic["motion"] is not None and pic["motion"] < STILL:
        found.append(_finding(language, "still", sid, "picture", "low", "revise", target))
    return found


def cut_findings(language: str, cuts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found = []
    for cut in cuts:
        before, after = cut["from"], cut["to"]
        if cut["same_place"] and cut["jump"] >= JUMP:
            found.append({**_finding(language, "jump", after, "continuity", "medium", "revise", f"shot:{after}",
                                     before=before, after=after, score=cut["jump"]), "id": f"{before}-{after}-jump"})
        if cut["level_jump_db"] is not None and cut["level_jump_db"] >= LEVEL_JUMP_DB:
            found.append({**_finding(language, "level", after, "sound", "low", "none", f"shot:{after}",
                                     before=before, after=after, db=cut["level_jump_db"]),
                          "id": f"{before}-{after}-level"})
    return found


def _jpeg(path: Path, height: int = 384) -> bytes:
    return _ffmpeg("-i", str(path), "-vf", f"scale=-2:{height}", "-frames:v", "1", "-q:v", "4",
                   "-f", "image2", "-c:v", "mjpeg", "pipe:1")


def vision_findings(chat, project, plan: dict[str, Any], shot: dict[str, Any], entry: dict[str, Any],
                    measures: dict[str, Any], previous: Path | None) -> list[dict[str, Any]]:
    """Ask a model that reads pictures what it sees in one shot; [] when it sees nothing wrong."""

    language = prompts.words(plan.get("language", "zh"))["language"]
    groups = {"character": plan.get("characters", []), "location": plan.get("locations", []),
              "prop": plan.get("props", [])}
    in_shot = [("character", c) for c in shot.get("characters", [])] + \
        [("location", shot.get("location"))] + [("prop", p) for p in shot.get("props", [])]
    cast, pictures = [], []
    for kind, identifier in in_shot:
        item = next((i for i in groups[kind] if i["id"] == identifier), None)
        if not item:
            continue
        cast.append({"target": f"{kind}:{identifier}", "name": item.get("name", ""), "look": item.get("look", "")})
        still = project.stills.get(f"{kind}:{identifier}", {})
        if still.get("file") and (project.root / still["file"]).is_file():
            pictures.append((f"{kind}:{identifier}", project.root / still["file"]))
    images, labels = [(project.root / entry["sheet"]).read_bytes()], []
    for label, path in pictures[:VISION_IMAGES - 2]:
        images.append(_jpeg(path))
        labels.append(label)
    extra = ""
    if labels:
        extra += f" Images 2 to {len(images)} are the reference pictures of {', '.join(labels)}, in that order."
    if previous is not None and previous.is_file():
        images.append(_jpeg(previous))
        extra += f" Image {len(images)} is the last frame of the previous shot."
    plan_fields = {key: shot.get(key) for key in ("summary", "camera", "action", "speaker", "line", "after",
                                                   "then", "sound")}
    reply = chat.complete_json(VISION_SYSTEM, VISION_PROMPT.format(
        shot=shot["id"], title=plan.get("title", ""), style=plan.get("style", ""),
        plan=json.dumps(plan_fields, ensure_ascii=False), cast=json.dumps(cast, ensure_ascii=False),
        heard=(measures.get("speech") or {}).get("heard") or "nothing", measured=json.dumps(
            {k: measures[k] for k in ("sound", "picture")}), frames=video.SHEET_TILES,
        seconds=entry.get("duration") or shot.get("seconds"), extra=extra, language=language),
        max_tokens=2000, temperature=0.2, images=tuple(images))
    known = {f"shot:{shot['id']}", *(c["target"] for c in cast)}
    found = []
    for index, item in enumerate(reply if isinstance(reply, list) else []):
        if not isinstance(item, dict) or item.get("category") not in CATEGORIES:
            continue
        target = str(item.get("target") or f"shot:{shot['id']}")
        if not TARGET.fullmatch(target) or target not in known:
            target = f"shot:{shot['id']}"
        action = item.get("action") if item.get("action") in ACTIONS else "revise"
        if action == "redraw" and target.startswith("shot:"):
            action = "revise"
        found.append({"id": f"{shot['id']}-vision-{index + 1}", "shot": shot["id"], "category": item["category"],
                      "severity": item.get("severity") if item.get("severity") in ("high", "medium", "low") else "medium",
                      "finding": str(item.get("finding") or "").strip()[:300],
                      "suggestion": str(item.get("suggestion") or "").strip()[:400],
                      "action": action, "target": target, "source": "vision"})
    return [f for f in found if f["finding"]][:5]


def path(project, number: int) -> Path:
    return project.episode_dir(number) / "analysis.json"


def load(project, number: int) -> dict[str, Any] | None:
    return read_json(path(project, number), None)


def run_analyze(context: Context) -> None:
    project, number = context.project, context.episode
    plan = project.plan
    episode = next((e for e in plan.get("episodes", []) if e["number"] == number), None)
    if episode is None or not episode.get("shots"):
        raise ValueError(f"episode {number} has no shot list")
    state = project.shot_state(number)
    shots = [s for s in episode["shots"] if state.get(s["id"], {}).get("file")]
    if not shots:
        raise ValueError(f"render the shots of episode {number} first")
    only = set(context.params.get("only") or [s["id"] for s in shots])
    language = plan.get("language", "zh")
    vision = context.services.vision()
    result: dict[str, Any] = {"time": int(time.time()), "episode": number, "language": language,
                              "vision": vision.model if vision else None, "shots": {}, "cuts": [], "findings": []}
    previous = (load(project, number) or {}).get("shots", {})
    loaded = False
    edges: dict[str, tuple[np.ndarray, np.ndarray, float | None]] = {}
    for index, shot in enumerate(shots, start=1):
        context.check()
        entry = state[shot["id"]]
        clip = project.root / entry["file"]
        if shot["id"] not in only:  # keep its earlier numbers for the cuts and the findings
            if shot["id"] in previous:
                result["shots"][shot["id"]] = previous[shot["id"]]
            edges[shot["id"]] = (edge(clip, last=False), edge(clip, last=True), None)
            continue
        context.progress(f"measuring {shot['id']} ({index} of {len(shots)})")
        seconds = video.probe(clip)["duration"]
        measures = {"take": entry.get("take"), "file": entry["file"], "duration": seconds,
                    "sound": sound(samples(clip)), "picture": picture(frames(clip, seconds))}
        speech = entry.get("speech")
        if speech is None:
            if not loaded:
                context.services.load(context.model)  # speech recognition
                loaded = True
            heard = context.services.engine.transcribe(context.model, video.speech_wav(clip))
            speech = {"heard": heard, "match": round(editing.coverage(heard, shot.get("line") or ""), 2)}
        measures["speech"] = {"heard": speech.get("heard", ""), "match": speech.get("match", 0.0)}
        result["shots"][shot["id"]] = measures
        edges[shot["id"]] = (edge(clip, last=False), edge(clip, last=True), measures["sound"]["level_db"])
        result["findings"] += measured_findings(language, shot, measures)
        if vision is not None:
            context.progress(f"asking {vision.model} about {shot['id']}")
            ids = [s["id"] for s in episode["shots"]]
            position = ids.index(shot["id"])
            before = media.ensure_last_frame(project, number, state.get(ids[position - 1], {})) if position else None
            try:
                result["findings"] += vision_findings(vision, project, plan, shot, {**entry, "duration": seconds},
                                                      measures, before)
            except Exception as error:  # one unreadable reply must not lose the measurements
                result.setdefault("vision_errors", []).append(f"{shot['id']}: {error}"[:300])
    for before, after in zip(shots, shots[1:]):
        levels = (result["shots"].get(before["id"], {}).get("sound", {}).get("level_db"),
                  result["shots"].get(after["id"], {}).get("sound", {}).get("level_db"))
        result["cuts"].append({
            "from": before["id"], "to": after["id"], "same_place": before.get("location") == after.get("location"),
            "jump": jump(edges[before["id"]][1], edges[after["id"]][0]),
            "level_jump_db": round(abs(levels[0] - levels[1]), 1) if None not in levels else None})
    result["findings"] += cut_findings(language, [c for c in result["cuts"] if c["to"] in only or c["from"] in only])
    write_json(path(project, number), result)
    project.log(f"episode {number} analyzed: {len(result['findings'])} findings")


ANALYZE = Step("analyze", run_analyze, role="asr", scope="episode", model_from="subtitles",
               description="Measure the sound, the lines, the picture and the cuts of the rendered shots, "
                           "and ask a vision model what it sees")
