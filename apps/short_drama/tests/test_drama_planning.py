# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from trtmc_short_drama import planning
from trtmc_short_drama.flow import Context
from trtmc_short_drama.store import Store

BIBLE = {
    "title": "The Ferry", "setting": "a river town", "style": "period drama",
    "characters": [{"id": "Lao Zhou", "name": "Lao Zhou", "look": "thin"},
                   {"id": "li", "name": "Shopkeeper Li", "look": "stout"},
                   {"id": "li", "name": "Young Master Li", "look": "young"}],
    "relationships": [{"from": "lao_zhou", "to": "li", "relation": "regular customer"}, {"from": "lao_zhou", "to": "nobody"}],
    "locations": [{"id": "dock", "name": "Dock", "look": "a wooden dock"}],
    "props": [{"id": "lantern", "name": "Lantern", "look": "a paper lantern"}],
    "episodes": [{"number": 1, "title": "The Crossing", "summary": "s1", "chapters": [1, 2]}],
}
SHOTS = [{"location": "dock", "characters": ["lao_zhou", "ghost"], "speaker": "ghost",
          "line": "Lao Zhou says: I have run this ferry for 30 years!",
          "seconds": 12, "camera": "c", "action": "a", "after": "b", "then": "t", "sound": "s", "summary": "x"},
         {"location": "nowhere", "characters": ["li"], "speaker": "li", "line": "", "seconds": 3},
         {"location": "dock", "characters": [], "seconds": 9},
         {"location": "dock", "characters": [], "seconds": 9}]


def test_bible_gets_unique_ascii_ids_and_known_relationships():
    plan = planning.normalize_bible(BIBLE, [[1, 2], [3]], "en")
    assert [c["id"] for c in plan["characters"]] == ["lao_zhou", "li", "li_"]
    assert plan["relationships"] == [{"from": "lao_zhou", "to": "li", "relation": "regular customer"}]
    assert [e["chapters"] for e in plan["episodes"]] == [[1, 2], [3]]
    assert plan["episodes"][0]["title"] == "The Crossing" and plan["episodes"][1]["title"] == "Episode 2"


def test_shots_keep_known_ids_clamp_seconds_and_fit_the_episode():
    plan = planning.normalize_bible(BIBLE, [[1, 2]])
    shots = planning.normalize_shots(SHOTS, plan, episode=1, max_seconds=24)
    assert [s["id"] for s in shots] == ["e1s1", "e1s2", "e1s3"]
    assert [s["seconds"] for s in shots] == [9, 4, 9]
    assert shots[0]["characters"] == ["lao_zhou"] and shots[0]["speaker"] == "lao_zhou"
    assert shots[1]["location"] == "dock" and shots[1]["speaker"] == ""
    phone = {"location": "dock", "characters": ["lao_zhou"], "speaker": "li", "line": "Come back!", "seconds": 6}
    assert planning.normalize_shots([phone], plan, episode=1, max_seconds=24)[0]["speaker"] == "li"
    assert sum(s["seconds"] for s in shots) <= 24


def test_a_plan_needs_cast_and_places():
    with pytest.raises(ValueError):
        planning.normalize_bible({"characters": [], "locations": []}, [[1]])


def test_chinese_projects_drop_the_space_models_put_before_latin_text():
    assert planning.chinese(" 我开了 30 年船！") == "我开了30年船！"
    assert planning.chinese("Run for 30 years") == "Run for 30 years"
    assert planning.normalize_bible(BIBLE, [[1], [2]])["episodes"][1]["title"] == "第2集"


class FakeChat:
    def __init__(self):
        self.prompts = []

    def complete_json(self, system, user, **_):
        self.prompts.append(user)
        if user.startswith("Summarize this chapter"):
            return {"summary": "events", "characters": ["Lao Zhou"], "places": ["Dock"], "quotes": []}
        if user.startswith("Plan a vertical short drama"):
            return BIBLE
        return SHOTS


class FakeServices:
    def __init__(self):
        self.client = FakeChat()

    def chat(self, model):
        assert model == "llm"
        return self.client


def test_plan_step_reads_every_chapter_then_writes_shots(tmp_path):
    project = Store(tmp_path).create("The Ferry", "Chapter 1: Prologue\nA\nChapter 2: The Crossing\nB",
                                     flow="default@1", episodes=1, max_seconds=24, language="en")
    services = FakeServices()
    planning.run(Context(project, {"shot_seconds": 8}, "llm", None, services, progress=lambda _: None))
    plan = project.plan
    assert [note["heading"] for note in plan["chapters"]] == ["Chapter 1: Prologue", "Chapter 2: The Crossing"]
    assert plan["episodes"][0]["shots"][0]["line"] == "Lao Zhou says: I have run this ferry for 30 years!"
    assert "Chapter 1: Prologue" in services.client.prompts[-1]  # the shot list sees the source chapters


def test_a_style_chosen_with_the_project_is_kept_and_shapes_the_looks(tmp_path):
    project = Store(tmp_path).create("The Ferry", "Chapter 1: Prologue\nA", flow="default@1", episodes=1, max_seconds=24,
                                     style="bright 3D cartoon animation")
    services = FakeServices()
    planning.run(Context(project, {}, "llm", None, services, progress=lambda _: None))
    assert project.plan["style"] == "bright 3D cartoon animation"
    request = next(p for p in services.client.prompts if p.startswith("Plan a vertical short drama"))
    assert request.endswith('use "bright 3D cartoon animation" as "style" and write every "look" to suit it.')
