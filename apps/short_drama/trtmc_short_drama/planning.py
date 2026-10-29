# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The plan step: novel -> chapter notes -> characters, places, episodes -> shot lists."""

from __future__ import annotations

import json
import re
from typing import Any

from . import prompts
from .flow import Context, Step

MIN_SHOT_SECONDS, MAX_SHOT_SECONDS = 4, 9
ASCII_ID = re.compile(r"[^a-z0-9_]+")
# Some models put a space between Chinese and Latin text (阿 Q); names and lines should not.
CJK_GAP = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[A-Za-z0-9])|(?<=[A-Za-z0-9])\s+(?=[\u4e00-\u9fff])")


def chinese(text: Any) -> str:
    return CJK_GAP.sub("", str(text or "")).strip()


def run(context: Context) -> None:
    project, params = context.project, context.params
    meta = project.meta
    episodes, max_seconds = int(meta.get("episodes", 4)), int(meta.get("max_seconds", 60))
    chat = context.services.chat(context.model)
    temperature = float(params.get("temperature", 0.7))
    max_tokens = int(params.get("max_tokens", 12000))
    shot_seconds = int(params.get("shot_seconds", 8))

    parts = prompts.chapters(project.novel)
    notes = []
    for number, text in enumerate(parts, start=1):
        context.check()
        context.progress(f"reading chapter {number} of {len(parts)}")
        note = chat.complete_json(prompts.SYSTEM, prompts.CHAPTER_PROMPT.format(text=text),
                                  max_tokens=2048, temperature=0.3)
        notes.append({"number": number, "heading": text.splitlines()[0][:40],
                      **(note if isinstance(note, dict) else {"summary": str(note)})})

    context.progress("planning characters, places and episodes")
    ranges = prompts.episode_chapters([len(part) for part in parts], episodes)
    assignment = " ".join(f"Episode {n} adapts chapters {', '.join(map(str, r))}."
                          for n, r in enumerate(ranges, start=1))
    grouped = [{"episode": n, "chapters": [note for note in notes if note["number"] in r]}
               for n, r in enumerate(ranges, start=1)]
    bible = chat.complete_json(prompts.SYSTEM, prompts.BIBLE_PROMPT.format(
        episodes=episodes, max_seconds=max_seconds, assignment=assignment,
        chapters=json.dumps(grouped, ensure_ascii=False)), max_tokens=max_tokens, temperature=temperature)
    plan = normalize_bible(bible, ranges)
    plan["chapters"] = notes

    for episode in plan["episodes"]:
        context.check()
        context.progress(f"writing the shots of episode {episode['number']}")
        source = "\n".join(parts[number - 1] for number in episode["chapters"])[:9000]
        most = max(2, max_seconds // shot_seconds)
        # The chapter notes, not the outline's summary: models sometimes shift summaries by a chapter.
        own = [{key: note.get(key) for key in ("number", "summary", "quotes")}
               for note in notes if note["number"] in episode["chapters"]]
        shots = chat.complete_json(prompts.SYSTEM, prompts.SHOTS_PROMPT.format(
            number=episode["number"], title=plan["title"], setting=plan["setting"],
            notes=json.dumps(own, ensure_ascii=False), max_seconds=max_seconds,
            min_shots=max(2, most - 2), max_shots=most,
            characters=prompts.roster(plan["characters"], "name", "role"),
            locations=prompts.roster(plan["locations"], "name"),
            props=prompts.roster(plan["props"], "name"), source=source),
            max_tokens=max_tokens, temperature=temperature)
        episode["shots"] = normalize_shots(shots, plan, episode["number"], max_seconds)
        project.save_plan(plan)
    project.save_plan(plan)
    project.log(f"plan: {len(plan['characters'])} characters, {len(plan['episodes'])} episodes")


def _items(value: Any, prefix: str) -> list[dict[str, Any]]:
    """Objects with unique ascii ids; the model sometimes repeats or omits ids."""

    items, seen = [], set()
    for index, item in enumerate(value if isinstance(value, list) else [], start=1):
        if not isinstance(item, dict):
            continue
        identifier = ASCII_ID.sub("_", str(item.get("id", "")).lower()).strip("_") or f"{prefix}{index}"
        while identifier in seen:
            identifier += "_"
        seen.add(identifier)
        fields = {key: str(val) for key, val in item.items() if isinstance(val, (str, int, float))}
        for key in ("name", "role"):
            if key in fields:
                fields[key] = chinese(fields[key])
        items.append({**fields, "id": identifier})
    return items


def normalize_bible(bible: Any, ranges: list[list[int]]) -> dict[str, Any]:
    if not isinstance(bible, dict):
        raise ValueError("the plan must be a JSON object")
    plan = {
        "title": chinese(bible.get("title")) or "短剧",
        "setting": str(bible.get("setting") or ""),
        "style": str(bible.get("style") or "realistic live-action drama with natural light"),
        "characters": _items(bible.get("characters"), "c"),
        "locations": _items(bible.get("locations"), "place"),
        "props": _items(bible.get("props"), "prop"),
    }
    known = {item["id"] for item in plan["characters"]}
    plan["relationships"] = [
        {"from": str(r.get("from")), "to": str(r.get("to")), "relation": str(r.get("relation", ""))}
        for r in bible.get("relationships") or [] if isinstance(r, dict)
        and str(r.get("from")) in known and str(r.get("to")) in known]
    outline = [e for e in bible.get("episodes") or [] if isinstance(e, dict)]
    plan["episodes"] = []
    for number, chapters in enumerate(ranges, start=1):
        entry = outline[number - 1] if number <= len(outline) else {}
        plan["episodes"].append({"number": number, "title": chinese(entry.get("title")) or f"第{number}集",
                                 "summary": chinese(entry.get("summary")), "chapters": chapters,
                                 "shots": []})
    if not plan["characters"] or not plan["locations"]:
        raise ValueError("the plan needs at least one character and one location")
    return plan


def normalize_shots(value: Any, plan: dict[str, Any], episode: int, max_seconds: int) -> list[dict[str, Any]]:
    """Keep known ids, clamp durations and drop shots past the episode length."""

    if isinstance(value, dict):
        value = value.get("shots", [])
    characters = {c["id"] for c in plan["characters"]}
    locations = [p["id"] for p in plan["locations"]]
    props = {p["id"] for p in plan["props"]}
    shots, total = [], 0
    for raw in value if isinstance(value, list) else []:
        if not isinstance(raw, dict):
            continue
        seconds = min(MAX_SHOT_SECONDS, max(MIN_SHOT_SECONDS, int(float(raw.get("seconds") or 6))))
        if total + seconds > max_seconds:
            seconds = max_seconds - total
            if seconds < MIN_SHOT_SECONDS:
                break
        cast = [c for c in raw.get("characters") or [] if c in characters][:3]
        speaker = str(raw.get("speaker") or "")
        line = chinese(raw.get("line"))
        if not line:
            speaker = ""
        elif speaker not in cast:
            speaker = cast[0] if cast else ""
        shot = {key: str(raw.get(key) or "") for key in ("summary", "camera", "action", "after", "then", "sound")}
        shot.update(id=f"e{episode}s{len(shots) + 1}", seconds=seconds, characters=cast,
                    location=raw.get("location") if raw.get("location") in locations else locations[0],
                    props=[p for p in raw.get("props") or [] if p in props][:1],
                    speaker=speaker, line=line)
        shots.append(shot)
        total += seconds
    if not shots:
        raise ValueError(f"episode {episode} has no usable shots")
    return shots


STEP = Step("plan", run, role="llm", scope="project",
            description="Read the novel and write characters, places, episodes and shot lists")
