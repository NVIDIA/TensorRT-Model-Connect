# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The stills step (reference images) and the shots step (reference-to-video clips)."""

from __future__ import annotations

from pathlib import Path
import time
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


def shot_keys(shot: dict[str, Any]) -> list[str]:
    """The still keys of what a shot shows, in Picture order."""

    return ([f"character:{c}" for c in shot.get("characters", [])] + [f"location:{shot.get('location')}"] +
            [f"prop:{p}" for p in shot.get("props", [])])


def usage(plan: dict[str, Any]) -> dict[str, list[str]]:
    """Still key -> the shots that show it. Only what appears on screen needs a still."""

    used: dict[str, list[str]] = {}
    for episode in plan.get("episodes", []):
        for shot in episode.get("shots", []):
            for key in shot_keys(shot):
                used.setdefault(key, []).append(shot["id"])
    return used


def still_outdated(plan: dict[str, Any], kind: str, item: dict[str, Any], still: dict[str, Any]) -> bool:
    """Whether the description changed after the still was drawn from it."""

    return (bool(still.get("file")) and not still.get("prompt_override") and
            still.get("prompt") != prompts.still_prompt(plan, kind, item))


def run_stills(context: Context) -> None:
    project, params = context.project, context.params
    plan, stills = project.plan, project.stills
    if not plan:
        raise ValueError("generate the plan first")
    only, used = params.get("only"), usage(plan)
    targets = [t for t in still_targets(plan)
               if (t[0] in only if only else t[0] in used and not stills.get(t[0], {}).get("file"))]
    if targets:
        context.services.load(context.model)
    for number, (key, kind, item) in enumerate(targets, start=1):
        context.check()
        context.progress(f"still {number} of {len(targets)}: {item.get('name', item['id'])}")
        previous = project.stills.get(key, {})  # re-read: the user may edit while stills render
        seed = int(previous["seed"]) + 1 if "seed" in previous else int(params.get("seed", 11))
        prompt = previous.get("prompt_override") or prompts.still_prompt(plan, kind, item)
        started = time.monotonic()
        png = context.services.engine.image(context.model, prompt, size=params.get("size", STILL_SIZE),
                                            seed=seed)
        relative = f"stills/{kind}-{item['id']}-{seed}.png"
        (project.root / "stills").mkdir(exist_ok=True)
        inside(project, project.root / relative).write_bytes(png)
        stills = project.stills
        stills[key] = {**stills.get(key, {}), "file": relative, "seed": seed, "prompt": prompt,
                       "status": "draft", "model": context.model,
                       "seconds": round(time.monotonic() - started, 1)}
        project.save_stills(stills)


def shot_references(project, plan: dict[str, Any], shot: dict[str, Any], *, previous: Path | None,
                    require_approval: bool = True) -> tuple[list[tuple[str, dict[str, Any]]], list[Path]]:
    """Approved stills for what is in the shot, in Picture order, plus the continuity frame.

    Without `require_approval` (prompt previews) every planned reference is listed.
    """

    stills = project.stills
    lookup = {f"{kind}:{item['id']}": (kind, item) for _, kind, item in still_targets(plan)}
    references, paths, missing = [], [], []
    for key in shot_keys(shot):
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


def uses_previous_frame(project, plan: dict[str, Any], number: int, default: bool = True) -> bool:
    """The episode's switch, or by default whether it opens where the previous one ended."""

    return number > 1 and bool(project.shot_state(number).get("use_previous_frame",
                                                                default and continues(plan, number)))


def shot_inputs(project, plan: dict[str, Any], episode: dict[str, Any], shot: dict[str, Any], *,
                use_previous: bool, require_approval: bool = True):
    """(references, upload paths, prompt, continuation) the shot would render with now."""

    first = bool(episode.get("shots")) and episode["shots"][0]["id"] == shot["id"]
    previous = (project.episode_dir(episode["number"] - 1) / "last_frame.png"
                if use_previous and first else None)
    references, paths = shot_references(project, plan, shot, previous=previous,
                                        require_approval=require_approval)
    return (references, paths, *prompts.video_prompts(plan, shot, references))


def relative(project, path: Path) -> str:
    return str(path.relative_to(project.root)) if path.is_relative_to(project.root) else path.name


def outdated(project, plan: dict[str, Any], episode: dict[str, Any], shot: dict[str, Any],
             entry: dict[str, Any], *, use_previous: bool) -> list[str]:
    """Why the chosen take no longer matches what the shot would render with now.

    "references": a still it used was redrawn, or the shot shows other things now;
    "prompt": the plan or the kept prompt changed since the take.
    """

    if not entry.get("file"):
        return []
    take = next((t for t in entry.get("takes", []) if t.get("take") == entry.get("take")), {})
    latest = entry.get("take") == max([t.get("take", 0) for t in entry.get("takes", [])], default=0)
    used = {key: take.get(key, entry.get(key) if latest else None) for key in ("references", "prompt")}
    _, paths, prompt, _ = shot_inputs(project, plan, episode, shot, use_previous=use_previous,
                                      require_approval=False)
    reasons = []
    if used["references"] is not None and sorted(relative(project, p) for p in paths) != sorted(used["references"]):
        reasons.append("references")
    if used["prompt"] is not None and (entry.get("prompt_override") or prompt) != used["prompt"]:
        reasons.append("prompt")
    return reasons


def average_seconds(entries: list[dict[str, Any]], model: str | None) -> float | None:
    """Mean time per still or take of `model` measured on this machine, if any were timed."""

    timed = [float(e["seconds"]) for e in entries if e.get("model") == model and e.get("seconds")]
    return round(sum(timed) / len(timed), 1) if timed else None


def run_shots(context: Context) -> None:
    project, params, number = context.project, context.params, context.episode
    plan = project.plan
    episode = next((e for e in plan.get("episodes", []) if e["number"] == number), None)
    if episode is None or not episode.get("shots"):
        raise ValueError(f"episode {number} has no shot list")
    state = project.shot_state(number)
    directory = project.episode_dir(number)
    use_previous = uses_previous_frame(project, plan, number, bool(params.get("previous_last_frame", True)))
    only = params.get("only")
    todo = [s for s in episode["shots"] if (s["id"] in only if only else not state.get(s["id"], {}).get("file"))]
    if todo:
        context.services.load(context.model)
    for index, shot in enumerate(todo, start=1):
        context.check()
        context.progress(f"episode {number} shot {index} of {len(todo)} ({shot['id']})")
        _, paths, prompt, continuation = shot_inputs(project, plan, episode, shot, use_previous=use_previous)
        entry = project.shot_state(number).get(shot["id"], {})  # the user may edit during a run
        prompt = entry.get("prompt_override") or prompt
        take = max([t["take"] for t in entry.get("takes", [])] + [int(entry.get("take", 0))]) + 1
        seed = int(params.get("seed", 7)) + take - 1
        raw = inside(project, directory / "raw" / f"{shot['id']}-take{take}.mp4")
        raw.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        context.services.engine.video(context.model, prompt, paths, raw, seed=seed,
                                      continuation=continuation)
        clip = inside(project, directory / "clips" / f"{shot['id']}-take{take}.mp4")
        video.finish_clip(raw, clip, seconds=float(shot["seconds"]))
        sheet = inside(project, directory / "sheets" / f"{shot['id']}-take{take}.jpg")
        video.contact_sheet(clip, sheet)
        # Each take keeps what it was rendered from, so a later edit shows which takes are outdated.
        rendered = {"take": take, "seed": seed, "file": str(clip.relative_to(project.root)),
                    "sheet": str(sheet.relative_to(project.root)), "raw": str(raw.relative_to(project.root)),
                    "prompt": prompt, "continuation": continuation,
                    "references": [relative(project, p) for p in paths],
                    "model": context.model, "seconds": round(time.monotonic() - started, 1)}
        state = project.shot_state(number)
        entry = state.get(shot["id"], {})
        entry.pop("speech", None)  # the new take has new audio
        state[shot["id"]] = {**entry, **rendered, "takes": [*entry.get("takes", []), rendered]}
        state.pop("assembled", None)
        project.save_shot_state(number, state)
        project.log(f"episode {number} {shot['id']} take {take} rendered")


STILLS = Step("stills", run_stills, role="image", scope="project",
              description="Draw a reference still for every character, place and prop")
SHOTS = Step("shots", run_shots, role="video", scope="episode",
             description="Render each shot from its approved reference stills, with sound")
