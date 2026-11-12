# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from trtmc_short_drama import editing, media, prompts, video
from trtmc_short_drama.flow import Context
from trtmc_short_drama.store import Store

PLAN = {"title": "t", "setting": "a town", "style": "drama",
        "characters": [{"id": "zhou", "name": "Zhou", "look": "a thin man", "voice": "a hoarse voice"}],
        "locations": [{"id": "dock", "name": "Dock", "look": "a wooden dock"}], "props": [],
        "episodes": [{"number": 1, "title": "One", "summary": "", "chapters": [1],
                      "shots": [{"id": "e1s1", "seconds": 6, "location": "dock", "characters": ["zhou"],
                                 "props": [], "speaker": "zhou", "line": "Hey!", "camera": "c", "action": "a",
                                 "after": "b", "then": "t", "sound": "s", "summary": "x"}]}]}


class FakeEngine:
    def __init__(self):
        self.calls = []

    def video(self, model, prompt, references, output, *, seed, continuation):
        self.calls.append({"prompt": prompt, "references": [p.name for p in references], "seed": seed})
        output.write_bytes(b"clip")


class FakeServices:
    def __init__(self):
        self.engine = FakeEngine()

    def load(self, model):
        pass


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "finish_clip", lambda raw, clip, seconds: (clip.parent.mkdir(parents=True, exist_ok=True)
                        or clip.write_bytes(raw.read_bytes())) and 0.25)
    monkeypatch.setattr(video, "contact_sheet", lambda clip, sheet: sheet.parent.mkdir(parents=True, exist_ok=True)
                        or sheet.write_bytes(b"jpg"))
    monkeypatch.setattr(video, "last_frame", lambda clip, image: image.write_bytes(b"png"))
    project = Store(tmp_path).create("t", "Chapter 1\nText", flow="default@1", episodes=1, max_seconds=60)
    project.save_plan(PLAN)
    (project.root / "stills").mkdir()
    for key in ("character-zhou", "location-dock"):
        (project.root / "stills" / f"{key}.png").write_bytes(b"png")
    project.save_stills({"character:zhou": {"file": "stills/character-zhou.png", "status": "approved", "seed": 1},
                         "location:dock": {"file": "stills/location-dock.png", "status": "draft", "seed": 1}})
    return project


def render(project, services, only=None):
    media.run_shots(Context(project, {"seed": 7, **({"only": only} if only else {})}, "v", 1, services,
                            progress=lambda _: None))


def test_shots_wait_for_every_still_to_be_approved(project):
    with pytest.raises(ValueError, match="location:dock"):
        render(project, FakeServices())


def test_takes_are_numbered_after_every_existing_take(project):
    stills = project.stills
    stills["location:dock"]["status"] = "approved"
    project.save_stills(stills)
    services = FakeServices()
    render(project, services)
    render(project, services, only=["e1s1"])
    state = project.shot_state(1)
    assert [t["take"] for t in state["e1s1"]["takes"]] == [1, 2] and state["e1s1"]["take"] == 2
    state["e1s1"].update(state["e1s1"]["takes"][0])  # the user picks take 1 again
    state["e1s1"]["prompt_override"] = "my prompt"
    project.save_shot_state(1, state)
    render(project, services, only=["e1s1"])
    state = project.shot_state(1)
    assert [t["take"] for t in state["e1s1"]["takes"]] == [1, 2, 3]
    assert state["e1s1"]["file"].endswith("e1s1-take3.mp4") and Path(project.root / state["e1s1"]["file"]).is_file()
    assert [call["seed"] for call in services.engine.calls] == [7, 8, 9]
    assert services.engine.calls[-1]["prompt"] == "my prompt"
    assert services.engine.calls[0]["references"] == ["character-zhou.png", "location-dock.png"]


class TranslatingServices(FakeServices):
    """A chat model that 'translates' by prefixing EN and records what it was asked."""

    def __init__(self):
        super().__init__()
        self.asked = []

    def chat_ready(self):
        return True

    def chat(self, name):
        asked = self.asked

        class Chat:
            def complete_json(self, system, user, **_):
                texts = json.loads(user[user.index("{"):])
                asked.append(sorted(texts.values()))
                return {key: f"EN {value}" for key, value in texts.items()}

        return Chat()


def test_shots_reach_the_video_model_in_english_and_are_translated_once(project):
    stills = project.stills
    stills["location:dock"]["status"] = "approved"
    project.save_stills(stills)
    plan = project.plan
    plan["episodes"][0]["shots"][0] = {**plan["episodes"][0]["shots"][0], "action": "zhou 跪下", "sound": "风声"}
    project.save_plan(plan)
    services = TranslatingServices()
    render(project, services)
    prompt = services.engine.calls[0]["prompt"]
    assert services.asked == [["zhou 跪下", "风声"]]
    assert "EN <Subject 1> 跪下." in prompt and "EN 风声." in prompt and "<d>[English] Hey!</d>" in prompt
    render(project, services, only=["e1s1"])
    assert len(services.asked) == 1 and services.engine.calls[1]["prompt"] == prompt  # kept, not asked again
    episode, shot = plan["episodes"][0], plan["episodes"][0]["shots"][0]
    assert media.outdated(project, plan, episode, shot, project.shot_state(1)["e1s1"], use_previous=False) == []


def test_media_never_writes_outside_the_project(project):
    with pytest.raises(ValueError):
        media.inside(project, project.root / "stills" / ".." / ".." / "evil.png")
    assert media.inside(project, project.root / "stills" / "a.png") == project.root / "stills" / "a.png"


class FakeImages:
    def __init__(self):
        self.sizes = []

    def image(self, model, prompt, *, size, seed):
        self.sizes.append(size)
        return b"png"


def test_only_what_appears_in_a_shot_gets_a_still(project):
    plan = project.plan
    plan["characters"].append({"id": "chen", "name": "Old Chen", "look": "a bearded man"})
    project.save_plan(plan)
    project.save_stills({})
    services = FakeServices()
    services.engine = FakeImages()
    media.run_stills(Context(project, {"seed": 3}, "img", None, services, progress=lambda _: None))
    stills = project.stills
    assert sorted(stills) == ["character:zhou", "location:dock"]  # Old Chen is in no shot
    assert services.engine.sizes == [media.STILL_SIZE] * 2
    assert stills["character:zhou"]["model"] == "img" and stills["character:zhou"]["seconds"] >= 0
    assert media.usage(plan) == {"character:zhou": ["e1s1"], "location:dock": ["e1s1"]}
    assert media.average_seconds([{"model": "img", "seconds": 4}, {"model": "img", "seconds": 6},
                                  {"model": "other", "seconds": 60}], "img") == 5.0


def test_stills_take_the_size_the_served_image_bundle_renders(project):
    project.save_stills({})
    services = FakeServices()
    services.engine = FakeImages()
    services.still_size = {"img": "720x1280"}.get
    media.run_stills(Context(project, {"seed": 3}, "img", None, services, progress=lambda _: None))
    assert services.engine.sizes == ["720x1280"] * 2


def test_a_take_is_outdated_when_a_still_or_the_plan_behind_it_changes(project):
    stills = project.stills
    stills["location:dock"]["status"] = "approved"
    project.save_stills(stills)
    render(project, FakeServices())

    def reasons():
        plan = project.plan
        episode = plan["episodes"][0]
        return media.outdated(project, plan, episode, episode["shots"][0], project.shot_state(1)["e1s1"],
                              use_previous=False)

    assert reasons() == []
    (project.root / "stills" / "character-zhou-2.png").write_bytes(b"png")
    stills["character:zhou"]["file"] = "stills/character-zhou-2.png"  # redrawn
    project.save_stills(stills)
    assert reasons() == ["references"]
    plan = project.plan
    plan["episodes"][0]["shots"][0]["line"] = "All aboard!"
    project.save_plan(plan)
    assert reasons() == ["references", "prompt"]
    render(project, FakeServices(), only=["e1s1"])
    assert reasons() == []
    state = project.shot_state(1)
    state["e1s1"].update(state["e1s1"]["takes"][0])  # back to take 1, rendered from the old still and line
    project.save_shot_state(1, state)
    assert reasons() == ["references", "prompt"]


def test_new_prompt_wording_leaves_takes_and_stills_current_but_plan_edits_do_not(project, monkeypatch):
    services = FakeServices()
    services.engine = FakeImages()
    media.run_stills(Context(project, {"seed": 3, "only": ["character:zhou", "location:dock"]}, "img", None,
                             services, progress=lambda _: None))
    stills = project.stills
    for still in stills.values():
        still["status"] = "approved"
    project.save_stills(stills)
    render(project, FakeServices())

    def outdated():
        plan = project.plan
        episode = plan["episodes"][0]
        return (media.outdated(project, plan, episode, episode["shots"][0], project.shot_state(1)["e1s1"],
                               use_previous=False),
                [key for key, kind, item in media.still_targets(plan)
                 if media.still_outdated(plan, kind, item, project.stills[key])])

    written, drawn = prompts.video_prompts, prompts.still_prompt  # an app update writes prompts another way
    monkeypatch.setattr(prompts, "video_prompts", lambda *args: tuple(f"{p} Calm." for p in written(*args)))
    monkeypatch.setattr(prompts, "still_prompt", lambda *args: f"{drawn(*args)} Sharp.")
    assert outdated() == ([], [])
    plan = project.plan
    plan["episodes"][0]["shots"][0]["action"] = "a, slowly"
    plan["characters"][0]["look"] = "a thin man in rags"
    project.save_plan(plan)
    assert outdated() == (["prompt"], ["character:zhou"])


def test_takes_from_before_takes_kept_their_source_follow_the_plan(project):
    stills = project.stills
    stills["location:dock"]["status"] = "approved"
    project.save_stills(stills)
    plan = project.plan
    plan["episodes"][0]["shots"][0] = {**plan["episodes"][0]["shots"][0], "speaker": "", "line": "",
                                       "action": "zhou 跪下"}
    project.save_plan(plan)
    render(project, FakeServices())
    state = project.shot_state(1)
    for record in (state["e1s1"], state["e1s1"]["takes"][0]):  # as an older app rendered it
        del record["source"]
        record["prompt"] = record["prompt"].replace(f" {prompts.NO_ONE_SPEAKS}", "")
    project.save_shot_state(1, state)
    (project.root / media.TRANSLATIONS).write_text(json.dumps({"zhou 跪下": "zhou kneels"}))  # put into English since

    def reasons():
        plan = project.plan
        episode = plan["episodes"][0]
        return media.outdated(project, plan, episode, episode["shots"][0], project.shot_state(1)["e1s1"],
                              use_previous=False)

    assert reasons() == []
    plan["episodes"][0]["shots"][0]["action"] = "zhou 站起"
    project.save_plan(plan)
    assert reasons() == ["prompt"]
    state = project.shot_state(1)
    state["e1s1"]["prompt_override"] = state["e1s1"]["prompt"]  # the user keeps the old prompt
    project.save_shot_state(1, state)
    assert reasons() == []


def test_each_shot_opens_where_the_shot_before_it_ends_in_the_same_place(project):
    plan = project.plan
    shot = plan["episodes"][0]["shots"][0]
    plan["locations"].append({"id": "street", "name": "Street", "look": "a street"})
    plan["episodes"][0]["shots"] = [shot, {**shot, "id": "e1s2", "line": ""},
                                    {**shot, "id": "e1s3", "location": "street", "line": ""}]
    project.save_plan(plan)
    (project.root / "stills" / "location-street.png").write_bytes(b"png")
    stills = project.stills
    stills["location:dock"]["status"] = "approved"
    stills["location:street"] = {"file": "stills/location-street.png", "status": "approved", "seed": 1}
    project.save_stills(stills)
    services = FakeServices()
    render(project, services)
    first, second, third = services.engine.calls
    assert first["references"] == ["character-zhou.png", "location-dock.png"]
    assert second["references"] == ["e1s1-take1-last.png", "character-zhou.png", "location-dock.png"]
    assert "final frame of the previous shot in <Picture 1>" in second["prompt"]
    assert "[Shot 1] The shot opens exactly on the frame in <Picture 1>" in second["prompt"]
    assert third["references"] == ["character-zhou.png", "location-street.png"]  # a new place is a cut

    def reasons(shot_id):
        plan = project.plan
        episode = plan["episodes"][0]
        shot = next(s for s in episode["shots"] if s["id"] == shot_id)
        return media.outdated(project, plan, episode, shot, project.shot_state(1)[shot_id], use_previous=False)

    assert reasons("e1s2") == []
    render(project, services, only=["e1s1"])
    assert reasons("e1s2") == ["previous"] and reasons("e1s3") == []
    state = project.shot_state(1)
    state["chain_shots"] = False
    project.save_shot_state(1, state)
    render(project, services, only=["e1s2"])
    assert services.engine.calls[-1]["references"] == ["character-zhou.png", "location-dock.png"]


def test_a_clip_skips_the_lead_in_and_keeps_its_planned_length(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(video, "probe", lambda path: {"duration": duration, "audio": True})
    monkeypatch.setattr(video, "opening", lambda raw: video.LEAD_IN)
    monkeypatch.setattr(video, "_run", lambda *arguments, cwd=None: calls.append(arguments)
                        or Path(arguments[-1]).write_bytes(b"mp4"))
    duration = 9.42
    assert video.finish_clip(tmp_path / "raw.mp4", tmp_path / "clip.mp4", seconds=9) == video.LEAD_IN
    duration = 9.1  # too short to keep all 9 s after the lead-in
    video.finish_clip(tmp_path / "raw.mp4", tmp_path / "clip.mp4", seconds=9)
    starts = [arguments[arguments.index("-ss") + 1] for arguments in calls]
    lengths = [arguments[arguments.index("-t") + 1] for arguments in calls]
    assert starts == ["0.250", "0.250"] and lengths == ["9.000", "8.850"]
    assert "afade=t=out:st=8.810:d=0.04" in calls[1][calls[1].index("-af") + 1]


def joined(path: Path, *sources: str) -> Path:
    inputs = [argument for source in sources for argument in ("-f", "lavfi", "-i", source)]
    joins = "".join(f"[{index}]" for index in range(len(sources))) + f"concat=n={len(sources)}"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *inputs, "-filter_complex", joins,
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def still(colour: str, seconds: float) -> str:
    return f"color=c={colour}:s=90x160:d={seconds}:r=24"


MOVING = "color=c=black:s=90x160:d=1.5:r=24,geq=lum='mod(Y*2+N*6\\,256)':cb=128:cr=128"  # changes 10-15 a frame


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs FFmpeg")
def test_a_clip_starts_after_the_reference_pictures_it_holds(tmp_path):
    assert video.opening(joined(tmp_path / "one.mp4", still("white", 0.5), still("0x202020", 1.5))) == 0.5
    two = joined(tmp_path / "two.mp4", still("white", 0.5), still("0x202020", 0.5), MOVING)
    assert video.opening(two) == 1.0  # a portrait, then a prop, then the shot
    assert video.opening(joined(tmp_path / "long.mp4", still("white", 1.5), MOVING)) == 1.5  # the previous shot's end
    moving_first = joined(tmp_path / "moves.mp4", MOVING, still("white", 0.5), MOVING)
    assert video.opening(moving_first) == video.LEAD_IN  # a cut inside the shot is kept
    steady = tmp_path / "steady.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=90x160:d=2:r=24",
                    "-pix_fmt", "yuv420p", str(steady)], check=True)
    assert video.opening(steady) == video.LEAD_IN


def test_assembling_cuts_takes_of_an_older_rule_again(project, monkeypatch):
    stills = project.stills
    stills["location:dock"]["status"] = "approved"
    project.save_stills(stills)
    render(project, FakeServices())
    state = project.shot_state(1)
    assert state["e1s1"]["cut"] == video.CUT_RULE and not media.cut_by_older_rule(state["e1s1"])
    old = {key: value for key, value in state["e1s1"].items() if key not in ("cut", "takes")}
    state["e1s1"] = {**old, "speech": {"start": 1.0, "end": 2.5, "match": 1.0},
                     "takes": [{k: v for k, v in t.items() if k != "cut"} for t in state["e1s1"]["takes"]]}
    project.save_shot_state(1, state)
    frame = media.last_frame_path(project, 1, state["e1s1"])
    frame.write_bytes(b"old end")
    monkeypatch.setattr(video, "finish_clip", lambda raw, clip, seconds, lead_in=None: 1.375)
    phases = []
    editing.recut_older(Context(project, {}, None, 1, FakeServices(), progress=phases.append), PLAN["episodes"][0])
    assert phases == ["episode 1: cutting e1s1 again"] and not media.recut(project, 1, PLAN["episodes"][0]["shots"][0])
    entry = project.shot_state(1)["e1s1"]
    assert (entry["lead_in"], entry["cut"], entry["takes"][0]["lead_in"]) == (1.375, video.CUT_RULE, 1.375)
    assert entry["speech"] == {"start": 0.0, "end": 1.375}  # moved by 1.125 s; heard again by the analysis
    assert frame.read_bytes() == b"png"  # taken again from the new clip


class FirstFrameEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.edits = []

    def edit(self, model, prompt, picture, *, seed, steps, cfg_scale):
        self.edits.append({"model": model, "prompt": prompt, "picture": picture, "seed": seed})
        return b"keyframe"

    def video(self, model, prompt, references, output, *, seed, continuation=None):
        super().video(model, prompt, references, output, seed=seed, continuation=continuation)


class FirstFrameServices(FakeServices):
    """A tier whose video model "v" animates one first frame that the edit model "e" draws."""

    def __init__(self):
        super().__init__()
        self.engine = FirstFrameEngine()
        self.loaded = []

    def load(self, model):
        self.loaded.append(model)

    def first_frame(self, name):
        return name == "v"

    def serving(self, role):
        return {"edit": "e"}.get(role)


def test_an_animated_shot_opens_on_a_keyframe_or_where_the_shot_before_it_ends(project, monkeypatch):
    finished, fitted = [], []
    monkeypatch.setattr(video, "finish_clip", lambda raw, clip, seconds, lead_in=None: finished.append(lead_in)
                        or clip.parent.mkdir(parents=True, exist_ok=True) or clip.write_bytes(b"clip") or 0.0)
    monkeypatch.setattr(video, "fit_picture", lambda picture, output, width, height: fitted.append(
        (picture.name, width, height)) or output.write_bytes(b"picture"))
    plan = project.plan
    shot = plan["episodes"][0]["shots"][0]
    plan["locations"].append({"id": "street", "name": "Street", "look": "a street"})
    plan["episodes"][0]["shots"] = [shot, {**shot, "id": "e1s2", "line": ""},
                                    {**shot, "id": "e1s3", "location": "street", "characters": [], "line": ""}]
    project.save_plan(plan)
    (project.root / "stills" / "location-street.png").write_bytes(b"png")
    stills = project.stills
    stills["location:dock"]["status"] = "approved"
    stills["location:street"] = {"file": "stills/location-street.png", "status": "approved", "seed": 1}
    project.save_stills(stills)
    services = FirstFrameServices()
    render(project, services)
    assert services.loaded == ["e", "v"]  # every keyframe first, so each model loads once
    assert fitted == [("character-zhou.png", 704, 1280), ("location-street.png", 704, 1280),  # the edit inputs
                      # then every first frame, at the size the video bundle takes
                      ("e1s1-take1.png", 704, 1280), ("e1s1-take1-last.png", 704, 1280), ("e1s3-take1.png", 704, 1280)]
    person, place = services.engine.edits
    assert "Use the person in the picture (Zhou, a thin man)" in person["prompt"] and "The place: a wooden dock." in person["prompt"]
    assert "Use the place in the picture" in place["prompt"] and person["picture"] == b"picture"
    first, second, third = services.engine.calls
    assert [c["references"] for c in (first, second, third)] == [
        ["e1s1-take1-first.png"], ["e1s2-take1-first.png"], ["e1s3-take1-first.png"]]
    assert "Zhou talks." in first["prompt"] and "talks" not in second["prompt"] and "<Subject" not in first["prompt"]
    assert finished == [0.0, 0.0, 0.0]  # a first frame is the shot itself: nothing is skipped
    state = project.shot_state(1)
    assert state["e1s1"]["silent"] and state["e1s1"]["keyframe"]["file"] == "episodes/01/keyframes/e1s1-take1.png"
    # the same place: e1s2 opens where e1s1 ends
    assert state["e1s2"]["first_frame"] == "episodes/01/frames/e1s1-take1-last.png" and not state["e1s2"]["keyframe"]

    def reasons(shot_id):
        plan = project.plan
        episode = plan["episodes"][0]
        shot = next(s for s in episode["shots"] if s["id"] == shot_id)
        return media.outdated(project, plan, episode, shot, project.shot_state(1)[shot_id], use_previous=False)

    assert reasons("e1s2") == [] and reasons("e1s3") == []
    render(project, services, only=["e1s2"])  # its shot before has a take: no keyframe, no edit model
    assert services.loaded[-1] == "v" and len(services.engine.edits) == 2
    render(project, FakeServices(), only=["e1s1"])  # a filmed take keeps nothing of an animated one
    entry = project.shot_state(1)["e1s1"]
    assert not any(key in entry for key in media.FIRST_FRAME_KEYS) and reasons("e1s2") == ["previous"]


def test_without_an_edit_model_an_animated_shot_opens_on_its_reference_picture(project, monkeypatch):
    monkeypatch.setattr(video, "finish_clip", lambda raw, clip, seconds, lead_in=None: (
        clip.parent.mkdir(parents=True, exist_ok=True) or clip.write_bytes(b"clip")) and 0.0)
    monkeypatch.setattr(video, "fit_picture", lambda picture, output, width, height: output.write_bytes(b"picture"))
    stills = project.stills
    stills["location:dock"]["status"] = "approved"
    project.save_stills(stills)
    services = FirstFrameServices()
    services.serving = lambda role: None  # a 24 GB tier: no edit model
    render(project, services)
    assert services.loaded == ["v"] and services.engine.edits == []
    assert services.engine.calls[0]["references"] == ["e1s1-take1-first.png"]
    picture = "episodes/01/keyframes/e1s1-take1-picture.png"
    entry = project.shot_state(1)["e1s1"]
    assert entry["keyframe"] == {"file": picture, "still": "stills/character-zhou.png", "picture": picture, "model": None}
    assert entry["first_frame"] == picture and entry["silent"]


def test_keyframe_and_motion_prompts_name_the_people_and_never_quote():
    plan = {**PLAN, "characters": [*PLAN["characters"], {"id": "mei", "name": "Mei", "look": "a young woman"}]}
    shot = {**PLAN["episodes"][0]["shots"][0], "characters": ["zhou", "mei"], "action": 'zhou hands mei a "lantern"',
            "camera": "close-up"}
    keyframe = prompts.keyframe_prompt(plan, shot, ("character", plan["characters"][0]))
    assert "Use the person in the picture (Zhou, a thin man)" in keyframe
    assert "Also in the shot: Mei, a young woman. Exactly 2 people, each of them once." in keyframe
    assert "It shows the moment this begins: Zhou hands Mei a lantern." in keyframe and '"' not in keyframe
    assert "lose-up" not in keyframe  # how the camera moves is the video model's to show
    motion = prompts.motion_prompt(plan, shot)
    assert motion.startswith("A vertical short-drama shot set in a town, drama. Close-up.")
    assert "Zhou is a thin man. Mei is a young woman. Zhou hands Mei a lantern." in motion and "Zhou talks." in motion


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs FFmpeg")
def test_a_still_fits_the_picture_size_the_edit_bundle_takes(tmp_path):
    portrait, output = tmp_path / "portrait.png", tmp_path / "picture.png"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=928x1664",
                    "-frames:v", "1", str(portrait)], check=True)
    video.fit_picture(portrait, output, width=704, height=1280)
    size = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0",
                           str(output)], capture_output=True, text=True, check=True).stdout.strip()
    assert size == "704,1280"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs FFmpeg")
def test_a_clip_without_sound_gets_a_silent_track(tmp_path):
    raw = tmp_path / "raw.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=352x640:rate=24:duration=3",
                    "-pix_fmt", "yuv444p", "-c:v", "libx264", str(raw)], check=True)
    clip = tmp_path / "clip.mp4"
    assert video.finish_clip(raw, clip, seconds=8.0, lead_in=0.0) == 0.0
    info = video.probe(clip)
    assert (info["width"], info["height"], info["audio"]) == (1080, 1920, True) and abs(info["duration"] - 3.0) < 0.1
