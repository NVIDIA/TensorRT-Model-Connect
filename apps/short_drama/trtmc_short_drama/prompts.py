# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planning prompts for the chat model and the prompts sent to the image and video models.

The chat model returns structured fields; the app composes the video prompts itself, so a
small local model only has to fill in short descriptions, not a strict long format.
"""

from __future__ import annotations

import itertools
import json
import re
from typing import Any

SYSTEM = ("You adapt Chinese novels into vertical live-action short dramas. "
          "Reply with one JSON value and nothing else. Inside JSON strings, quote with 「」 and "
          "never use the double-quote character.")

CHAPTER_PROMPT = """Summarize this chapter of a novel for a screenwriter, in Chinese.
Return a JSON object:
{{"summary": "<the events in order, at most 300 characters>",
  "characters": ["<names that appear>"],
  "places": ["<places that appear>"],
  "quotes": ["<up to 5 short lines of dialogue copied exactly from the text>"]}}

Chapter:
{text}"""

BIBLE_PROMPT = """Plan a vertical short drama in {episodes} episodes of at most {max_seconds} seconds
each, adapted from the novel below (given as chapter notes).

Return a JSON object with these keys:
- "title": the drama title in Chinese.
- "setting": era and place in English, for example "a small river town in Zhejiang, China, around 1911".
- "style": the visual style in English, for example "realistic live-action period drama with soft natural light".
- "characters": the 3 to 8 people who appear in the episodes, each {{"id": "<short lowercase ascii id>",
  "name": "<Chinese name from the novel>", "role": "<who they are, in Chinese>",
  "look": "<English: age, build, face, hair and period clothing, one sentence>",
  "voice": "<English: for example a hoarse middle-aged male voice>"}}.
- "relationships": [{{"from": "<id>", "to": "<id>", "relation": "<in Chinese>"}}].
- "locations": the 2 to 6 places where the episodes happen, each {{"id": "<ascii id>",
  "name": "<Chinese name>", "look": "<English, one sentence>"}}.
- "props": 0 to 4 objects that matter in the story, each {{"id": "<ascii id>", "name": "<Chinese name>",
  "look": "<English>"}}.
- "episodes": exactly {episodes} items in this order, each {{"number": <n>, "title": "<Chinese>",
  "summary": "<what happens in its chapters, in Chinese, at most 150 characters>"}}.
  {assignment}

Include every person who speaks or acts in an episode's events, not only the lead.
Use only people, places and events from the novel. Write every "look", "voice", "setting" and
"style" in English, like this example:
{{"id": "wangu", "name": "王胡", "role": "未庄的闲汉，满脸络腮胡子",
  "look": "a burly man in his thirties with a thick full beard, a shaved forehead and a long queue, wearing a patched grey jacket",
  "voice": "a gruff, loud male voice"}}

Chapter notes, grouped by episode:
{chapters}"""

SHOTS_PROMPT = """Write the shot list of episode {number} of the short drama "{title}".
Setting: {setting}.
This episode adapts these chapters, in order: {notes}

Characters (use these ids): {characters}
Locations (use these ids): {locations}
Props (use these ids): {props}

Rules:
- {min_shots} to {max_shots} shots that tell the events of these chapters from the source text below,
  in order. Do not bring in events, people or objects from other chapters.
- "seconds" add up to at most {max_seconds}; each shot is 5 to 9 seconds.
- At most one line of dialogue per shot, in Chinese, under 16 characters, spoken by a character in
  the shot. Prefer lines from the source text. Use "" when nobody speaks.
- Every shot keeps moving: an action before the line, the line early, a reaction after it.
- At most 3 characters per shot; list a prop only when it is visible in that shot and belongs to
  this episode's events.
- "sound" is never empty: the ambience and the physical sounds of the action.
- Write "camera", "action", "after", "then", "sound" and "summary" in English, and refer to people
  by their ids (for example: aq slaps his own face). Never use double quotes inside them.

Example of one shot from another drama (follow its form, not its content):
{{"location": "office", "characters": ["linlan"], "props": ["phone"], "seconds": 6,
  "summary": "linlan takes a call by the window and hears shocking news",
  "camera": "A handheld medium close-up slowly pushes in",
  "action": "linlan paces a step with the phone at her ear, stops short and turns toward the camera",
  "speaker": "linlan", "line": "你说什么？他回来了？",
  "after": "linlan lowers the phone and presses her free hand to her mouth",
  "then": "she glances back at the office door, breathing fast, and starts toward it",
  "sound": "quiet office room tone, distant keyboard clicks"}}

Return a JSON array of shots with exactly these keys.

Source text of this episode:
{source}"""


def chapters(novel: str, limit: int = 6000) -> list[str]:
    """Split on chapter headings (第…章/回/节); fall back to paragraphs packed up to `limit`."""

    parts = re.split(r"\n(?=\s*第[0-9一二三四五六七八九十百零〇]+[章回节])", "\n" + novel.strip())
    parts = [part.strip() for part in parts if part.strip()]
    packed: list[str] = []
    for part in parts:
        while len(part) > limit:
            cut = part.rfind("\n", 0, limit)
            cut = cut if cut > limit // 2 else limit
            packed.append(part[:cut].strip())
            part = part[cut:].strip()
        packed.append(part)
    return packed


def episode_chapters(lengths: list[int], episodes: int) -> list[list[int]]:
    """Contiguous 1-based chapter ranges of about equal length, one per episode."""

    count = len(lengths)
    if episodes >= count:
        return [[min(count, 1 + index * count // episodes)] for index in range(episodes)]
    ends = list(itertools.accumulate(lengths))
    ranges, start = [], 0
    for index in range(1, episodes):
        goal = ends[-1] * index / episodes
        end = min(range(start + 1, count - (episodes - index) + 1), key=lambda e: abs(ends[e - 1] - goal))
        ranges.append(list(range(start + 1, end + 1)))
        start = end
    ranges.append(list(range(start + 1, count + 1)))
    return ranges


def roster(items: list[dict[str, Any]], *fields: str) -> str:
    return json.dumps([{key: item.get(key, "") for key in ("id", *fields)} for item in items],
                      ensure_ascii=False)


def _english(text: str) -> str:
    """Double quotes make the video model draw text, so drop them from descriptions."""

    return re.sub(r"[\"“”]", "", str(text or "")).strip().rstrip(".")


def _clause(text: str) -> str:
    """A description placed mid-sentence: no trailing period, a leading article in lower case."""

    text = _english(text)
    first = text.split(maxsplit=1)[0] if text else ""
    return text[:1].lower() + text[1:] if first in {"A", "An", "The"} else text


def _sentence(text: str) -> str:
    """A description that starts a sentence: capitalized, with one closing period."""

    text = _english(text)
    return text[:1].upper() + text[1:] + "." if text else ""


def _line(text: str) -> str:
    return re.sub(r"[\"“”'‘’「」『』]", "", str(text or "")).strip()


def still_prompt(plan: dict[str, Any], kind: str, item: dict[str, Any]) -> str:
    setting, style = _english(plan.get("setting")), _english(plan.get("style"))
    look = _english(item.get("look"))
    if kind == "character":
        return (f"{style}, {setting}. Full-body portrait of {look}, standing and facing the camera "
                "with a calm expression, plain light grey studio backdrop, soft even light, sharp "
                "focus on the face, realistic live-action film still.")
    if kind == "location":
        return f"{style}, {setting}. {look}. An empty establishing view, realistic live-action film still."
    return f"{look}, centered on a plain light grey backdrop, soft even light, realistic photograph."


def video_prompts(plan: dict[str, Any], shot: dict[str, Any],
                  references: list[tuple[str, dict[str, Any]]]) -> tuple[str, str]:
    """Compose the first-window and continuation prompts of one shot.

    `references` lists (kind, item) in upload order; kind is character, location, prop or
    previous (the last frame of the previous episode). Picture N is the N-th upload.
    """

    labels: dict[str, str] = {}
    definitions, retention = [], []
    for number, (kind, item) in enumerate(references, start=1):
        subject = f"<Subject {number}>"
        picture = f"<Picture {number}>"
        if kind == "character":
            labels[item["id"]] = subject
            definitions.append(f"{subject} is {item.get('name', '')} from {picture}, "
                               f"{_english(item.get('look'))}.")
            retention.append(f"{subject} (appears in [Shot 1]): fully_preserved - the face, "
                             "hairstyle and clothing are retained.")
        elif kind == "location":
            definitions.append(f"{subject} is the place in {picture}: {_english(item.get('look'))}.")
            retention.append(f"{subject} (appears in [Shot 1]): fully_preserved - the layout, "
                             "materials and light of the place are retained.")
        elif kind == "prop":
            definitions.append(f"{subject} is the {_english(item.get('look'))} in {picture}.")
            retention.append(f"{subject} (appears in [Shot 1]): fully_preserved - its shape and "
                             "color are retained.")
        else:
            definitions.append(f"{subject} is the final frame of the previous episode in {picture}, "
                               "which this shot continues from.")
            retention.append(f"{subject} (appears in [Shot 1]): partially_preserved - the scene "
                             "and poses continue from it.")

    def name(identifier: str) -> str:
        return labels.get(identifier) or identifier

    names = {c["id"]: c.get("name", "") for c in plan.get("characters", [])}

    def mention(text: str) -> str:
        for identifier, label in labels.items():
            text = re.sub(rf"\b{re.escape(identifier)}\b", label, text)
            if names.get(identifier):  # models sometimes write the Chinese name, even as 阿 Q
                pattern = r"\s*".join(map(re.escape, names[identifier].replace(" ", "")))
                text = re.sub(pattern, label, text)
        return text

    voice = {c["id"]: _english(c.get("voice")) for c in plan.get("characters", [])}
    speaker, line = shot.get("speaker") or "", _line(shot.get("line"))
    body = [_sentence(shot.get("camera")), _sentence(mention(_english(shot.get("action"))))]
    if line and speaker in labels:
        manner = f"In {voice[speaker]}, " if voice.get(speaker) else ""
        body.append(f"{manner}{name(speaker)} (S1) says <d>[Chinese] {line}</d> "
                    "and then falls silent.")
    elif line:
        body.append(f"An off-screen narrator says in an off-screen voiceover: <d>[Chinese] {line}</d> "
                    "The people on screen keep their lips closed.")
    body.append(_sentence(mention(_english(shot.get("after")))))
    head = ("subject_definitions:\n" + "\n".join(definitions) +
            "\n\nsummary:\n[reference generation] " + mention(_english(shot.get("summary"))) + "." +
            "\n\nretention_analysis:\n" + "\n".join(retention))
    sound = ("overall_soundscape:\n" + (_english(shot.get("sound")) or "Quiet ambience") + "." +
             "\n\nnon_diegetic_music:\nN/A")
    setting = f" set in {_clause(plan['setting'])}" if plan.get("setting") else ""
    look = f", {_clause(plan['style'])}," if plan.get("style") else ""
    style = (f"The target video uses a realistic live-action vertical short-drama style{setting}{look} "
             "and a clean image without subtitles, captions or any on-screen text.")
    first = f"{head}\n\ndetailed_description:\n{style}\n[Shot 1] {' '.join(body)}\n\n{sound}"
    then = _sentence(mention(_english(shot.get("then")))) or "The people on screen keep moving naturally."
    continuation = (f"{head}\n\ndetailed_description:\n[Shot 1 continued] The same shot keeps "
                    f"drifting slowly. {then} Nobody speaks.\n\n{sound}")
    return first, continuation
