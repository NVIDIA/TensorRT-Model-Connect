# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actions that run with the plan step's language model, outside the flow's stages.

story:  write a short story from a one-line idea, so a project can start without a novel.
revise: rewrite one shot, character, location or prop from a note; the rest of the plan stays.
review: read each episode's shots against its chapter notes and list what to fix.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

from . import prompts
from .flow import Context, Step
from .planning import MIN_SHOT_SECONDS, chinese, normalize_shots

TARGET = re.compile(r"(shot|character|location|prop):([a-z0-9_]{1,40})")
MAX_NOTE = 600
GROUPS = {"character": "characters", "location": "locations", "prop": "props"}
FIELDS = {"character": ("name", "role", "look", "voice"), "location": ("name", "look"), "prop": ("name", "look")}
SHOT_KEYS = ("location", "characters", "props", "seconds", "summary", "camera", "action", "speaker", "line",
             "after", "then", "sound")
CHAPTER = re.compile(r"(^|\n)\s*第[0-9一二三四五六七八九十百零〇]+章")


def revision(params: dict[str, Any]) -> tuple[str, str, str]:
    """(kind, id, note) of a revision request, or ValueError."""

    match = TARGET.fullmatch(str(params.get("target", "")))
    note = str(params.get("instruction", "")).strip()
    if not match or not note or len(note) > MAX_NOTE:
        raise ValueError(f"a revision needs a target such as shot:e1s2 and a note of 1 to {MAX_NOTE} characters")
    return match.group(1), match.group(2), note


def fingerprint(plan: dict[str, Any]) -> str:
    """Identifies the shot lists a review was written for."""

    return hashlib.sha256(json.dumps(plan.get("episodes", []), sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()[:16]


def run_story(context: Context) -> None:
    project = context.project
    meta = project.meta
    idea = str(meta.get("idea") or "").strip()
    if not idea:
        raise ValueError("this project has no idea to write a story from")
    if project.novel.strip():
        raise ValueError("this project already has its story; edit it in the Plan tab")
    chat = context.services.chat(context.model)
    context.progress("writing the story")
    reply = chat.complete(prompts.STORY_SYSTEM, prompts.STORY_PROMPT.format(
        idea=idea, episodes=int(meta.get("episodes", 4)), max_seconds=int(meta.get("max_seconds", 60))),
        max_tokens=int(context.params.get("max_tokens", 6000)),
        temperature=float(context.params.get("temperature", 0.8)))
    project.save_novel(story_text(reply))
    project.log(f"story written from the idea: {len(project.novel)} characters")


def story_text(reply: str) -> str:
    """The story without thinking blocks or fences; it must have chapters to plan from."""

    text = re.sub(r"<think>.*?</think>", "", reply, flags=re.S).strip()
    text = re.sub(r"^```\w*\n|\n```$", "", text).strip()
    if len(text) < 200 or not CHAPTER.search(text):
        raise ValueError("the model did not write a story with chapter headings; try again or paste one")
    return text[CHAPTER.search(text).start():].strip()


def run_revise(context: Context) -> None:
    project = context.project
    kind, identifier, note = revision(context.params)
    plan = project.plan
    if not plan.get("episodes"):
        raise ValueError("plan the drama first")
    chat = context.services.chat(context.model)
    context.progress(f"revising {kind} {identifier}")
    if kind == "shot":
        revised = revise_shot(chat, plan, identifier, note, float(project.meta.get("max_seconds", 60)))
    else:
        revised = revise_item(chat, plan, kind, identifier, note)
    plan = project.plan  # re-read: keep edits saved while the model was writing
    if kind == "shot":
        for episode in plan.get("episodes", []):
            episode["shots"] = [revised if s["id"] == identifier else s for s in episode.get("shots", [])]
    else:
        group = GROUPS[kind]
        plan[group] = [revised if item["id"] == identifier else item for item in plan.get(group, [])]
    project.save_plan(plan, note=f"before revising {kind} {identifier}: {note[:60]}")
    project.log(f"revised {kind} {identifier}")


def shot_fields(shot: dict[str, Any] | None) -> dict[str, Any] | None:
    return {key: shot.get(key) for key in SHOT_KEYS} if shot else None


def find_shot(plan: dict[str, Any], identifier: str) -> tuple[dict[str, Any], int]:
    for episode in plan.get("episodes", []):
        for index, shot in enumerate(episode.get("shots", [])):
            if shot["id"] == identifier:
                return episode, index
    raise ValueError(f"the plan has no shot {identifier}")


def revise_shot(chat, plan: dict[str, Any], identifier: str, note: str, max_seconds: float) -> dict[str, Any]:
    episode, index = find_shot(plan, identifier)
    shots = episode["shots"]
    shot = shots[index]
    left = int(max_seconds - sum(float(s.get("seconds") or 0) for s in shots if s is not shot))
    if left < MIN_SHOT_SECONDS:
        raise ValueError(f"episode {episode['number']} has only {left} s left for this shot; shorten another")
    notes = [{key: n.get(key) for key in ("number", "summary", "quotes")}
             for n in plan.get("chapters", []) if n.get("number") in episode.get("chapters", [])]
    reply = chat.complete_json(prompts.SYSTEM, prompts.REVISE_SHOT_PROMPT.format(
        title=plan.get("title", ""), setting=plan.get("setting", ""), instruction=note,
        shot=json.dumps(shot_fields(shot), ensure_ascii=False),
        before=json.dumps(shot_fields(shots[index - 1]) if index else None, ensure_ascii=False),
        after=json.dumps(shot_fields(shots[index + 1]) if index + 1 < len(shots) else None, ensure_ascii=False),
        notes=json.dumps(notes, ensure_ascii=False),
        characters=prompts.roster(plan.get("characters", []), "name", "role"),
        locations=prompts.roster(plan.get("locations", []), "name"),
        props=prompts.roster(plan.get("props", []), "name")), max_tokens=3000, temperature=0.5)
    if isinstance(reply, list):
        reply = reply[0] if reply else {}
    return {**normalize_shots([reply], plan, episode["number"], left)[0], "id": identifier}


def revise_item(chat, plan: dict[str, Any], kind: str, identifier: str, note: str) -> dict[str, Any]:
    item = next((i for i in plan.get(GROUPS[kind], []) if i["id"] == identifier), None)
    if item is None:
        raise ValueError(f"the plan has no {kind} {identifier}")
    fields = FIELDS[kind]
    english = " and ".join(f'"{f}"' for f in fields if f in ("look", "voice"))
    reply = chat.complete_json(prompts.SYSTEM, prompts.REVISE_ITEM_PROMPT.format(
        kind=kind, title=plan.get("title", ""), setting=plan.get("setting", ""), style=plan.get("style", ""),
        instruction=note, item=json.dumps({f: item.get(f, "") for f in fields}, ensure_ascii=False),
        role=' and "role"' if "role" in fields else "", english=english,
        keys=", ".join(fields)), max_tokens=1500, temperature=0.5)
    if not isinstance(reply, dict):
        raise ValueError("the model did not return the revised item as a JSON object")
    changed = {f: str(reply[f]).strip() for f in fields
               if isinstance(reply.get(f), (str, int, float)) and str(reply[f]).strip()}
    for key in ("name", "role"):
        if key in changed:
            changed[key] = chinese(changed[key])
    return {**item, **changed, "id": identifier}


def run_review(context: Context) -> None:
    project = context.project
    plan = project.plan
    if not plan.get("episodes"):
        raise ValueError("plan the drama first")
    chat = context.services.chat(context.model)
    found = []
    for episode in plan["episodes"]:
        if not episode.get("shots"):
            continue
        context.check()
        context.progress(f"reviewing episode {episode['number']}")
        found += review_episode(chat, plan, episode)
    project.save_review({"items": found, "plan": fingerprint(plan), "time": int(time.time())})
    project.log(f"model review: {len(found)} problems")


def review_episode(chat, plan: dict[str, Any], episode: dict[str, Any]) -> list[dict[str, Any]]:
    notes = [{key: n.get(key) for key in ("number", "summary", "quotes")}
             for n in plan.get("chapters", []) if n.get("number") in episode.get("chapters", [])]
    shots = [{"id": s["id"], **{key: s.get(key) for key in SHOT_KEYS}} for s in episode["shots"]]
    reply = chat.complete_json(prompts.SYSTEM, prompts.REVIEW_PROMPT.format(
        number=episode["number"], title=plan.get("title", ""), notes=json.dumps(notes, ensure_ascii=False),
        characters=prompts.roster(plan.get("characters", []), "name"),
        locations=prompts.roster(plan.get("locations", []), "name"),
        props=prompts.roster(plan.get("props", []), "name"), shots=json.dumps(shots, ensure_ascii=False)),
        max_tokens=2500, temperature=0.3)
    if isinstance(reply, dict):
        reply = reply.get("problems", [])
    ids = {s["id"] for s in episode["shots"]}
    return [{"shot": str(item["shot"]), "episode": episode["number"],
             "problem": chinese(item.get("problem"))[:300], "fix": chinese(item.get("fix"))[:300]}
            for item in (reply if isinstance(reply, list) else [])[:6]
            if isinstance(item, dict) and str(item.get("shot")) in ids and str(item.get("problem") or "").strip()]


STORY = Step("story", run_story, role="llm", model_from="plan",
             description="Write a short story from the project's idea, to plan from")
REVISE = Step("revise", run_revise, role="llm", model_from="plan",
              description="Rewrite one shot, character, location or prop from a note")
REVIEW = Step("review", run_review, role="llm", model_from="plan",
              description="Ask the language model to review every episode's shots before rendering")
