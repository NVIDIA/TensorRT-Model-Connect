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
                    require_approval: bool = True, previous_scope: str = "episode",
                    ) -> tuple[list[tuple[str, dict[str, Any]]], list[Path]]:
    """Approved stills for what is in the shot, in Picture order, plus the continuity frame.

    The continuity frame ends the previous episode (`previous_scope` "episode") or the shot before
    this one ("shot"). Without `require_approval` (prompt previews) every planned reference is listed.
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
    if previous is not None and previous.is_file():
        if len(paths) >= MAX_REFERENCES:  # continuity matters more than the last prop
            references, paths = references[:MAX_REFERENCES - 1], paths[:MAX_REFERENCES - 1]
        if previous_scope == "shot":  # Picture 1: the student tends to open on its first picture
            references, paths = [("previous", {"scope": previous_scope}), *references], [previous, *paths]
        else:
            references.append(("previous", {"scope": previous_scope}))
            paths.append(previous)
    return references[:MAX_REFERENCES], paths[:MAX_REFERENCES]


def last_frame_path(project, number: int, entry: dict[str, Any]) -> Path | None:
    """Where the last frame of the take an entry shows is kept, if it has a take."""

    if not entry.get("file"):
        return None
    return project.episode_dir(number) / "frames" / f"{Path(entry['file']).stem}-last.png"


def ensure_last_frame(project, number: int, entry: dict[str, Any]) -> Path | None:
    """The last frame of the entry's take, extracted from its clip the first time it is needed."""

    frame = last_frame_path(project, number, entry)
    clip = project.root / entry["file"] if frame else None
    if frame and not frame.is_file() and clip.is_file():
        frame.parent.mkdir(parents=True, exist_ok=True)
        video.last_frame(clip, inside(project, frame))
    return frame


def chains_shots(project, number: int, default: bool = True) -> bool:
    """The episode's switch: each shot continues from the end of the shot before it in the same place."""

    return bool(project.shot_state(number).get("chain_shots", default))


def previous_frame(project, episode: dict[str, Any], shot: dict[str, Any], *, use_previous: bool,
                   chain: bool) -> tuple[Path | None, str]:
    """(continuity frame, scope) for a shot: the end of the previous episode for the first shot,
    otherwise the end of the shot before it when both are in the same place."""

    shots = episode.get("shots") or []
    index = next((i for i, s in enumerate(shots) if s["id"] == shot["id"]), 0)
    if index == 0:
        return (project.episode_dir(episode["number"] - 1) / "last_frame.png" if use_previous else None), "episode"
    before = shots[index - 1]
    if not chain or before.get("location") != shot.get("location"):
        return None, "shot"  # a new place is a cut
    return last_frame_path(project, episode["number"], project.shot_state(episode["number"]).get(before["id"], {})), "shot"


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
                use_previous: bool, chain: bool = True, require_approval: bool = True):
    """(references, upload paths, prompt, continuation) the shot would render with now."""

    previous, scope = previous_frame(project, episode, shot, use_previous=use_previous, chain=chain)
    references, paths = shot_references(project, plan, shot, previous=previous, previous_scope=scope,
                                        require_approval=require_approval)
    return (references, paths, *prompts.video_prompts(plan, shot, references))


def relative(project, path: Path) -> str:
    return str(path.relative_to(project.root)) if path.is_relative_to(project.root) else path.name


def outdated(project, plan: dict[str, Any], episode: dict[str, Any], shot: dict[str, Any],
             entry: dict[str, Any], *, use_previous: bool, chain: bool = True) -> list[str]:
    """Why the chosen take no longer matches what the shot would render with now.

    "references": a still it used was redrawn, or the shot shows other things now;
    "previous": the shot before it has another take now, so it no longer starts where that one ends;
    "prompt": the plan or the kept prompt changed since the take.
    """

    if not entry.get("file"):
        return []
    take = next((t for t in entry.get("takes", []) if t.get("take") == entry.get("take")), {})
    latest = entry.get("take") == max([t.get("take", 0) for t in entry.get("takes", [])], default=0)
    used = {key: take.get(key, entry.get(key) if latest else None) for key in ("references", "prompt")}
    _, paths, prompt, _ = shot_inputs(project, plan, episode, shot, use_previous=use_previous, chain=chain,
                                      require_approval=False)
    reasons = []

    def split(names) -> tuple[list[str], list[str]]:
        frames = sorted(n for n in names if "/frames/" in f"/{n}")
        return sorted(n for n in names if n not in frames), frames

    if used["references"] is not None:
        stills_now, frames_now = split([relative(project, p) for p in paths])
        stills_used, frames_used = split(used["references"])
        if stills_now != stills_used:
            reasons.append("references")
        if frames_now != frames_used:
            reasons.append("previous")
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
    chain = chains_shots(project, number, bool(params.get("chain_shots", True)))
    only = params.get("only")
    todo = [s for s in episode["shots"] if (s["id"] in only if only else not state.get(s["id"], {}).get("file"))]
    if todo:
        context.services.load(context.model)
    for index, shot in enumerate(todo, start=1):
        context.check()
        context.progress(f"episode {number} shot {index} of {len(todo)} ({shot['id']})")
        position = next(i for i, s in enumerate(episode["shots"]) if s["id"] == shot["id"])
        if position:  # takes rendered before shots were chained have no saved last frame yet
            ensure_last_frame(project, number, project.shot_state(number).get(episode["shots"][position - 1]["id"], {}))
        _, paths, prompt, continuation = shot_inputs(project, plan, episode, shot, use_previous=use_previous,
                                                     chain=chain)
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
        lead_in = video.finish_clip(raw, clip, seconds=float(shot["seconds"]))
        sheet = inside(project, directory / "sheets" / f"{shot['id']}-take{take}.jpg")
        video.contact_sheet(clip, sheet)
        ensure_last_frame(project, number, {"file": str(clip.relative_to(project.root))})  # the next shot starts here
        # Each take keeps what it was rendered from, so a later edit shows which takes are outdated.
        rendered = {"take": take, "seed": seed, "file": str(clip.relative_to(project.root)),
                    "sheet": str(sheet.relative_to(project.root)), "raw": str(raw.relative_to(project.root)),
                    "prompt": prompt, "continuation": continuation,
                    "references": [relative(project, p) for p in paths], "lead_in": lead_in,
                    "cut": video.CUT_RULE, "model": context.model,
                    "seconds": round(time.monotonic() - started, 1)}
        state = project.shot_state(number)
        entry = state.get(shot["id"], {})
        entry.pop("speech", None)  # the new take has new audio
        state[shot["id"]] = {**entry, **rendered, "takes": [*entry.get("takes", []), rendered]}
        state.pop("assembled", None)
        project.save_shot_state(number, state)
        project.log(f"episode {number} {shot['id']} take {take} rendered")


def cut_by_older_rule(entry: dict[str, Any]) -> bool:
    """Whether the entry's take was cut from its raw clip by an older video.CUT_RULE."""

    return bool(entry.get("file") and entry.get("raw")) and int(entry.get("cut", 1)) < video.CUT_RULE


def recut(project, number: int, shot: dict[str, Any]) -> bool:
    """Cut the shot's chosen take again from its raw clip with the current rule, if an older one cut it.

    The line's timing moves with the new start, so its subtitle stays in place without hearing it again;
    what was heard on the old cut is dropped, so the analysis listens to the new one.
    """

    entry = project.shot_state(number).get(shot["id"], {})
    raw = project.root / entry.get("raw", "")
    if not cut_by_older_rule(entry) or not raw.is_file():
        return False
    clip = inside(project, project.root / entry["file"])
    lead_in = video.finish_clip(inside(project, raw), clip, seconds=float(shot["seconds"]))
    if entry.get("sheet"):
        video.contact_sheet(clip, inside(project, project.root / entry["sheet"]))
    frame = last_frame_path(project, number, entry)
    if frame and frame.is_file():
        frame.unlink()  # the clip ends elsewhere now
    ensure_last_frame(project, number, entry)
    cut = {"lead_in": lead_in, "cut": video.CUT_RULE}
    update = dict(cut)
    if entry.get("speech"):  # takes cut before lead-ins existed started at 0
        moved = lead_in - float(entry.get("lead_in", 0.0))
        speech = {key: value for key, value in entry["speech"].items() if key not in ("heard", "match")}
        update["speech"] = {**speech, "start": round(max(speech["start"] - moved, 0.0), 3),
                            "end": round(max(speech["end"] - moved, 0.0), 3)}
    state = project.shot_state(number)  # re-read: the user may pick another take meanwhile
    current = state.get(shot["id"], {})
    takes = [{**t, **cut} if t.get("take") == entry.get("take") else t for t in current.get("takes", [])]
    state[shot["id"]] = {**current, **(update if current.get("file") == entry["file"] else {}), "takes": takes}
    state.pop("assembled", None)
    project.save_shot_state(number, state)
    project.log(f"episode {number} {shot['id']} take {entry.get('take')} cut again: starts at {lead_in} s")
    return True


STILLS = Step("stills", run_stills, role="image", scope="project",
              description="Draw a reference still for every character, place and prop")
SHOTS = Step("shots", run_shots, role="video", scope="episode",
             description="Render each shot from its approved reference stills, with sound")
