# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The stills step (reference images) and the shots step (reference-to-video clips)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import prompts, video
from .flow import Context, Step

MAX_REFERENCES = 6
STILL_SIZE = "928x1664"


def inside(project, path: Path) -> Path:
    """Refuse to write anywhere but the project directory, whatever the plan ids say."""

    if not path.resolve().is_relative_to(project.root.resolve()):
        raise ValueError(f"{path.name} would be written outside the project")
    return path


def still_targets(plan: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """(key, kind, item) for every character, location and prop of the plan."""

    return [(f"{kind}:{item['id']}", kind, item)
            for kind, group in (("character", "characters"), ("location", "locations"), ("prop", "props"))
            for item in plan.get(group, [])]


def run_stills(context: Context) -> None:
    project, params = context.project, context.params
    plan, stills = project.plan, project.stills
    if not plan:
        raise ValueError("generate the plan first")
    only = params.get("only")
    targets = [t for t in still_targets(plan)
               if (t[0] in only if only else not stills.get(t[0], {}).get("file"))]
    if targets:
        context.services.load(context.model)
    for number, (key, kind, item) in enumerate(targets, start=1):
        context.check()
        context.progress(f"still {number} of {len(targets)}: {item.get('name', item['id'])}")
        previous = project.stills.get(key, {})  # re-read: the user may edit while stills render
        seed = int(previous["seed"]) + 1 if "seed" in previous else int(params.get("seed", 11))
        prompt = previous.get("prompt_override") or prompts.still_prompt(plan, kind, item)
        png = context.services.engine.image(context.model, prompt, size=params.get("size", STILL_SIZE),
                                            seed=seed)
        relative = f"stills/{kind}-{item['id']}-{seed}.png"
        (project.root / "stills").mkdir(exist_ok=True)
        inside(project, project.root / relative).write_bytes(png)
        stills = project.stills
        stills[key] = {**stills.get(key, {}), "file": relative, "seed": seed, "prompt": prompt,
                       "status": "draft"}
        project.save_stills(stills)


def shot_references(project, plan: dict[str, Any], shot: dict[str, Any], *, previous: Path | None,
                    require_approval: bool = True) -> tuple[list[tuple[str, dict[str, Any]]], list[Path]]:
    """Approved stills for what is in the shot, in Picture order, plus the continuity frame.

    Without `require_approval` (prompt previews) every planned reference is listed.
    """

    stills = project.stills
    lookup = {f"{kind}:{item['id']}": (kind, item) for _, kind, item in still_targets(plan)}
    keys = ([f"character:{c}" for c in shot.get("characters", [])] + [f"location:{shot.get('location')}"] +
            [f"prop:{p}" for p in shot.get("props", [])])
    references, paths, missing = [], [], []
    for key in keys:
        if key not in lookup:
            continue
        still = stills.get(key, {})
        if require_approval and (still.get("status") != "approved" or not still.get("file")):
            missing.append(key)
            continue
        references.append(lookup[key])
        paths.append(project.root / still.get("file", ""))
    if missing:
        raise ValueError(f"approve these reference stills first: {', '.join(missing)}")
    if previous is not None and previous.is_file() and len(paths) < MAX_REFERENCES:
        references.append(("previous", {}))
        paths.append(previous)
    return references[:MAX_REFERENCES], paths[:MAX_REFERENCES]


def continues(plan: dict[str, Any], number: int) -> bool:
    """Whether episode `number` opens where the previous episode ended."""

    episodes = {e["number"]: e for e in plan.get("episodes", [])}
    if number - 1 not in episodes or not episodes[number - 1].get("shots") or not episodes.get(number, {}).get("shots"):
        return False
    return episodes[number - 1]["shots"][-1].get("location") == episodes[number]["shots"][0].get("location")


def run_shots(context: Context) -> None:
    project, params, number = context.project, context.params, context.episode
    plan = project.plan
    episode = next((e for e in plan.get("episodes", []) if e["number"] == number), None)
    if episode is None or not episode.get("shots"):
        raise ValueError(f"episode {number} has no shot list")
    state = project.shot_state(number)
    directory = project.episode_dir(number)
    use_previous = state.get("use_previous_frame",
                             bool(params.get("previous_last_frame", True)) and continues(plan, number))
    previous = project.episode_dir(number - 1) / "last_frame.png" if use_previous and number > 1 else None
    only = params.get("only")
    todo = [s for s in episode["shots"] if (s["id"] in only if only else not state.get(s["id"], {}).get("file"))]
    if todo:
        context.services.load(context.model)
    for index, shot in enumerate(todo, start=1):
        context.check()
        context.progress(f"episode {number} shot {index} of {len(todo)} ({shot['id']})")
        first = shot is episode["shots"][0]
        references, paths = shot_references(project, plan, shot, previous=previous if first else None)
        prompt, continuation = prompts.video_prompts(plan, shot, references)
        entry = project.shot_state(number).get(shot["id"], {})  # the user may edit during a run
        prompt = entry.get("prompt_override") or prompt
        take = max([t["take"] for t in entry.get("takes", [])] + [int(entry.get("take", 0))]) + 1
        seed = int(params.get("seed", 7)) + take - 1
        raw = inside(project, directory / "raw" / f"{shot['id']}-take{take}.mp4")
        raw.parent.mkdir(parents=True, exist_ok=True)
        context.services.engine.video(context.model, prompt, paths, raw, seed=seed,
                                      continuation=continuation)
        clip = inside(project, directory / "clips" / f"{shot['id']}-take{take}.mp4")
        video.finish_clip(raw, clip, seconds=float(shot["seconds"]))
        sheet = inside(project, directory / "sheets" / f"{shot['id']}-take{take}.jpg")
        video.contact_sheet(clip, sheet)
        rendered = {"take": take, "seed": seed, "file": str(clip.relative_to(project.root)),
                    "sheet": str(sheet.relative_to(project.root)), "raw": str(raw.relative_to(project.root))}
        state = project.shot_state(number)
        entry = state.get(shot["id"], {})
        entry.pop("speech", None)  # the new take has new audio
        state[shot["id"]] = {**entry, **rendered, "takes": [*entry.get("takes", []), rendered],
                             "prompt": prompt, "continuation": continuation,
                             "references": [str(p.relative_to(project.root)) if p.is_relative_to(project.root)
                                            else p.name for p in paths]}
        state.pop("assembled", None)
        project.save_shot_state(number, state)
        project.log(f"episode {number} {shot['id']} take {take} rendered")


STILLS = Step("stills", run_stills, role="image", scope="project",
              description="Draw a reference still for every character, place and prop")
SHOTS = Step("shots", run_shots, role="video", scope="episode",
             description="Render each shot from its approved reference stills, with sound")
