# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from trtmc_short_drama import media, video
from trtmc_short_drama.flow import Context
from trtmc_short_drama.store import Store

PLAN = {"title": "t", "setting": "a town", "style": "drama",
        "characters": [{"id": "aq", "name": "阿Q", "look": "a thin man", "voice": "a hoarse voice"}],
        "locations": [{"id": "temple", "name": "土谷祠", "look": "a shrine"}], "props": [],
        "episodes": [{"number": 1, "title": "一", "summary": "", "chapters": [1],
                      "shots": [{"id": "e1s1", "seconds": 6, "location": "temple", "characters": ["aq"],
                                 "props": [], "speaker": "aq", "line": "滚！", "camera": "c", "action": "a",
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
    monkeypatch.setattr(video, "finish_clip", lambda raw, clip, seconds: clip.parent.mkdir(parents=True, exist_ok=True)
                        or clip.write_bytes(raw.read_bytes()))
    monkeypatch.setattr(video, "contact_sheet", lambda clip, sheet: sheet.parent.mkdir(parents=True, exist_ok=True)
                        or sheet.write_bytes(b"jpg"))
    monkeypatch.setattr(video, "last_frame", lambda clip, image: image.write_bytes(b"png"))
    project = Store(tmp_path).create("t", "第一章\n文", flow="default@1", episodes=1, max_seconds=60)
    project.save_plan(PLAN)
    (project.root / "stills").mkdir()
    for key in ("character-aq", "location-temple"):
        (project.root / "stills" / f"{key}.png").write_bytes(b"png")
    project.save_stills({"character:aq": {"file": "stills/character-aq.png", "status": "approved", "seed": 1},
                         "location:temple": {"file": "stills/location-temple.png", "status": "draft", "seed": 1}})
    return project


def render(project, services, only=None):
    media.run_shots(Context(project, {"seed": 7, **({"only": only} if only else {})}, "v", 1, services,
                            progress=lambda _: None))


def test_shots_wait_for_every_still_to_be_approved(project):
    with pytest.raises(ValueError, match="location:temple"):
        render(project, FakeServices())


def test_takes_are_numbered_after_every_existing_take(project):
    stills = project.stills
    stills["location:temple"]["status"] = "approved"
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
    assert services.engine.calls[0]["references"] == ["character-aq.png", "location-temple.png"]


def test_media_never_writes_outside_the_project(project):
    with pytest.raises(ValueError):
        media.inside(project, project.root / "stills" / ".." / ".." / "evil.png")
    assert media.inside(project, project.root / "stills" / "a.png") == project.root / "stills" / "a.png"


class FakeImages:
    def image(self, model, prompt, *, size, seed):
        return b"png"


def test_only_what_appears_in_a_shot_gets_a_still(project):
    plan = project.plan
    plan["characters"].append({"id": "wang", "name": "王胡", "look": "a bearded man"})
    project.save_plan(plan)
    project.save_stills({})
    services = FakeServices()
    services.engine = FakeImages()
    media.run_stills(Context(project, {"seed": 3}, "img", None, services, progress=lambda _: None))
    stills = project.stills
    assert sorted(stills) == ["character:aq", "location:temple"]  # 王胡 is in no shot
    assert stills["character:aq"]["model"] == "img" and stills["character:aq"]["seconds"] >= 0
    assert media.usage(plan) == {"character:aq": ["e1s1"], "location:temple": ["e1s1"]}
    assert media.average_seconds([{"model": "img", "seconds": 4}, {"model": "img", "seconds": 6},
                                  {"model": "other", "seconds": 60}], "img") == 5.0


def test_a_take_is_outdated_when_a_still_or_the_plan_behind_it_changes(project):
    stills = project.stills
    stills["location:temple"]["status"] = "approved"
    project.save_stills(stills)
    render(project, FakeServices())

    def reasons():
        plan = project.plan
        episode = plan["episodes"][0]
        return media.outdated(project, plan, episode, episode["shots"][0], project.shot_state(1)["e1s1"],
                              use_previous=False)

    assert reasons() == []
    (project.root / "stills" / "character-aq-2.png").write_bytes(b"png")
    stills["character:aq"]["file"] = "stills/character-aq-2.png"  # redrawn
    project.save_stills(stills)
    assert reasons() == ["references"]
    plan = project.plan
    plan["episodes"][0]["shots"][0]["line"] = "滚出去！"
    project.save_plan(plan)
    assert reasons() == ["references", "prompt"]
    render(project, FakeServices(), only=["e1s1"])
    assert reasons() == []
    state = project.shot_state(1)
    state["e1s1"].update(state["e1s1"]["takes"][0])  # back to take 1, rendered from the old still and line
    project.save_shot_state(1, state)
    assert reasons() == ["references", "prompt"]


def test_each_shot_opens_where_the_shot_before_it_ends_in_the_same_place(project):
    plan = project.plan
    shot = plan["episodes"][0]["shots"][0]
    plan["locations"].append({"id": "street", "name": "街上", "look": "a street"})
    plan["episodes"][0]["shots"] = [shot, {**shot, "id": "e1s2", "line": ""},
                                    {**shot, "id": "e1s3", "location": "street", "line": ""}]
    project.save_plan(plan)
    (project.root / "stills" / "location-street.png").write_bytes(b"png")
    stills = project.stills
    stills["location:temple"]["status"] = "approved"
    stills["location:street"] = {"file": "stills/location-street.png", "status": "approved", "seed": 1}
    project.save_stills(stills)
    services = FakeServices()
    render(project, services)
    first, second, third = services.engine.calls
    assert first["references"] == ["character-aq.png", "location-temple.png"]
    assert second["references"] == ["character-aq.png", "location-temple.png", "e1s1-take1-last.png"]
    assert "final frame of the previous shot in <Picture 3>" in second["prompt"]
    assert third["references"] == ["character-aq.png", "location-street.png"]  # a new place is a cut

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
    assert services.engine.calls[-1]["references"] == ["character-aq.png", "location-temple.png"]


def test_a_clip_skips_the_lead_in_and_keeps_its_planned_length(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(video, "probe", lambda path: {"duration": duration})
    monkeypatch.setattr(video, "_run", lambda *arguments, cwd=None: calls.append(arguments)
                        or Path(arguments[-1]).write_bytes(b"mp4"))
    duration = 9.42
    video.finish_clip(tmp_path / "raw.mp4", tmp_path / "clip.mp4", seconds=9)
    duration = 9.1  # too short to keep all 9 s after the lead-in
    video.finish_clip(tmp_path / "raw.mp4", tmp_path / "clip.mp4", seconds=9)
    starts = [arguments[arguments.index("-ss") + 1] for arguments in calls]
    lengths = [arguments[arguments.index("-t") + 1] for arguments in calls]
    assert starts == ["0.250", "0.250"] and lengths == ["9.000", "8.850"]
    assert "afade=t=out:st=8.810:d=0.04" in calls[1][calls[1].index("-af") + 1]
