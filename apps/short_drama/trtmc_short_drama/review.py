# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan checks that run on every edit, and the project's progress through its stages.

Before any GPU time is spent, a reviewer reads the shot lists against the novel: everyone who
acts is cast, every line has a speaker in the shot and fits the shot, every episode fits its
limit. These are plain rules; the model review in `assist.py` adds judgement on top.
"""

from __future__ import annotations

import re
from typing import Any

from . import prompts
from .editing import normalize
from .media import MAX_REFERENCES, shot_keys, usage

CLIP_SECONDS = 9.4  # one FastH3 clip; a shot cannot be longer
SPEECH_CHARACTERS_PER_SECOND = 4.5
SPEECH_MARGIN_SECONDS = 2.0  # the action before a line and the reaction after it
NAMED_TIMES = 3  # a name this frequent in an episode's chapters is likely on screen
TEXT_FIELDS = ("summary", "camera", "action", "after", "then")


def plural(count: int, word: str) -> str:
    return f"{count} {word}{'s' * (count != 1)}"


def problem(level: str, where: str, message: str, fix: str = "") -> dict[str, str]:
    return {"level": level, "where": where, "message": message, "fix": fix}


def check(plan: dict[str, Any], novel: str, max_seconds: float) -> list[dict[str, str]]:
    """Errors stop a shot from rendering as planned; warnings are worth a look."""

    if not plan.get("episodes"):
        return []
    found = uncast(plan, prompts.chapters(novel) if novel.strip() else [])
    for episode in plan["episodes"]:
        shots = episode.get("shots") or []
        total = sum(seconds(shot) for shot in shots)
        if not shots:
            found.append(problem("error", f"episode {episode['number']}", "has no shots"))
        elif total > max_seconds:
            found.append(problem("error", f"episode {episode['number']}",
                                 f"runs {total:g} s, over the {max_seconds:g} s limit", "shorten or remove shots"))
        for shot in shots:
            found += check_shot(plan, shot)
    return found


def seconds(shot: dict[str, Any]) -> float:
    try:
        return float(shot.get("seconds") or 0)
    except (TypeError, ValueError):
        return 0.0


def check_shot(plan: dict[str, Any], shot: dict[str, Any]) -> list[dict[str, str]]:
    where, found = shot.get("id", "?"), []
    cast = {c["id"]: c for c in plan.get("characters", [])}
    known = {"location": {p["id"] for p in plan.get("locations", [])}, "character": set(cast),
             "prop": {p["id"] for p in plan.get("props", [])}}
    unknown = [key for key in shot_keys(shot) if key.partition(":")[2] not in known[key.partition(":")[0]]]
    if unknown:
        found.append(problem("error", where, f"uses {', '.join(unknown)}, which the plan does not have",
                             "pick ids from Characters, Places and Props"))
    if not 0 < seconds(shot) <= CLIP_SECONDS:
        found.append(problem("error", where, f"lasts {shot.get('seconds')} s; a shot is one clip of at most "
                                             f"{CLIP_SECONDS:g} s", "set 4 to 9 seconds"))
    if len(shot_keys(shot)) > MAX_REFERENCES:
        found.append(problem("warning", where, f"shows {len(shot_keys(shot))} things; the video model takes "
                                               f"{MAX_REFERENCES} reference pictures, the rest are dropped"))
    line = normalize(shot.get("line"))
    speaker = shot.get("speaker") or ""
    if line and speaker not in shot.get("characters", []):
        name = cast.get(speaker, {}).get("name") or speaker or "nobody"
        found.append(problem("warning", where, f"the line is spoken by {name}, who is not in the shot, so it "
                                               "becomes an off-screen voice", "add the speaker or change it"))
    need = len(line) / SPEECH_CHARACTERS_PER_SECOND + SPEECH_MARGIN_SECONDS if line else 0
    if need > seconds(shot) > 0:
        found.append(problem("warning", where, f"the line needs about {need:.0f} s with the action around it; "
                                               f"the shot has {seconds(shot):g} s", "shorten the line"))
    text = " ".join(str(shot.get(field) or "") for field in TEXT_FIELDS)
    for identifier, character in cast.items():
        if identifier in shot.get("characters", []):
            continue
        name = normalize(character.get("name"))
        if re.search(rf"\b{re.escape(identifier)}\b", text) or (len(name) > 1 and name in normalize(text)):
            found.append(problem("warning", where, f"mentions {character.get('name') or identifier}, who is not "
                                                   "in the shot's characters, so the video model will invent them",
                                 f"add {identifier} to the shot or reword it"))
    return found


def uncast(plan: dict[str, Any], parts: list[str]) -> list[dict[str, str]]:
    """People the chapter notes name often in an episode's chapters but nobody plays."""

    cast = [normalize(c.get("name")) for c in plan.get("characters", [])]
    notes = {note.get("number"): note for note in plan.get("chapters", []) if isinstance(note, dict)}
    found: dict[str, tuple[int, str, int]] = {}  # name -> (episode, as written, times named)
    for episode in plan.get("episodes", []):
        chapters = [n for n in episode.get("chapters", []) if isinstance(n, int)]
        source = normalize("".join(parts[n - 1] for n in chapters if 0 < n <= len(parts)))
        for number in chapters:
            for raw in notes.get(number, {}).get("characters") or []:
                name = normalize(raw)
                if len(name) < 2 or name in found or any(name in c or c in name for c in cast if c):
                    continue
                if (times := source.count(name)) >= NAMED_TIMES:
                    found[name] = (episode["number"], str(raw), times)
    # 太太 is part of 赵太太: report the fuller name only.
    return [problem("warning", f"episode {number}", f"{raw} is named {times} times in its chapters but is "
                    "not in Characters", "add them if they appear on screen, then add them to their shots")
            for name, (number, raw, times) in found.items()
            if not any(name != other and name in other for other in found)]


def origins(plan: dict[str, Any], novel: str) -> dict[str, str]:
    """Shot id -> "novel" when its line is quoted from the novel, else "adapted".

    A one-character line such as 滚 tells nothing either way and gets no mark.
    """

    text = normalize(novel)
    return {shot["id"]: "novel" if line in text else "adapted"
            for episode in plan.get("episodes", []) for shot in episode.get("shots", [])
            if len(line := normalize(shot.get("line"))) > 1}


def progress(plan: dict[str, Any], stills: dict[str, Any], episodes: dict[str, dict[str, Any]], *,
             novel: bool, problems: list[dict[str, str]]) -> list[dict[str, str]]:
    """The four stages with a short status and the next thing to do in each.

    `episodes` maps an episode number (as text) to its shot state with an "outdated" list per shot.
    """

    errors = sum(p["level"] == "error" for p in problems)
    shots = [(e["number"], s["id"]) for e in plan.get("episodes", []) for s in e.get("shots", [])]
    if not plan.get("episodes"):
        stage = [stage_entry("plan", "todo", "no plan yet",
                             "Plan the drama from the novel" if novel else "Write the story from the idea")]
    else:
        cast = len(plan.get("characters", []))
        stage = [stage_entry("plan", "attention" if errors else "done",
                             f"{len(plan['episodes'])} episodes · {len(shots)} shots · {cast} cast",
                             f"Fix {plural(errors, 'problem')} in the plan" if errors else "")]
    needed = sorted(usage(plan))
    drawn = [k for k in needed if stills.get(k, {}).get("file")]
    approved = [k for k in drawn if stills.get(k, {}).get("status") == "approved"]
    if not needed:
        stage.append(stage_entry("stills", "todo", "waiting for the plan", ""))
    elif len(drawn) < len(needed):
        stage.append(stage_entry("stills", "todo" if not drawn else "attention", f"{len(drawn)} of {len(needed)} drawn",
                                 f"Draw {plural(len(needed) - len(drawn), 'still')}"))
    elif len(approved) < len(needed):
        stage.append(stage_entry("stills", "attention", f"{len(approved)} of {len(needed)} approved",
                                 f"Approve {plural(len(needed) - len(approved), 'still')}"))
    else:
        stage.append(stage_entry("stills", "done", f"{len(needed)} approved", ""))
    state = {(number, shot): episodes.get(str(number), {}).get(shot, {}) for number, shot in shots}
    rendered = [key for key, entry in state.items() if entry.get("file")]
    stale = [key for key in rendered if state[key].get("outdated")]
    ready = needed and len(approved) == len(needed)
    if not shots:
        stage.append(stage_entry("shots", "todo", "waiting for the plan", ""))
    elif not rendered:
        stage.append(stage_entry("shots", "todo", f"0 of {len(shots)} rendered",
                                 "Render a sample shot first" if ready else "Approve the stills first"))
    elif stale or len(rendered) < len(shots):
        todo = f"Re-render {len(stale)} outdated" if stale else f"Render {len(shots) - len(rendered)} more"
        stage.append(stage_entry("shots", "attention", f"{len(rendered)} of {len(shots)} rendered", todo))
    else:
        stage.append(stage_entry("shots", "done", f"{len(shots)} rendered", ""))
    numbers = [e["number"] for e in plan.get("episodes", [])]
    assembled = [n for n in numbers if episodes.get(str(n), {}).get("assembled")]
    complete = [n for n in numbers if n not in assembled and all(state[(m, s)].get("file") for m, s in shots if m == n)
                and any(m == n for m, _ in shots)]
    stage.append(stage_entry("episodes", "done" if numbers and len(assembled) == len(numbers) else
                             "attention" if complete else "todo", f"{len(assembled)} of {len(numbers)} assembled",
                             f"Assemble episode {complete[0]}" if complete else ""))
    return stage


def stage_entry(key: str, status: str, detail: str, next_step: str) -> dict[str, str]:
    return {"key": key, "status": status, "detail": detail, "next": next_step}
