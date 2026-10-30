# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from trtmc_short_drama import planning
from trtmc_short_drama.flow import Context
from trtmc_short_drama.store import Store

BIBLE = {
    "title": "阿Q正传", "setting": "a river town", "style": "period drama",
    "characters": [{"id": "A Q", "name": "阿Q", "look": "thin"}, {"id": "zhao", "name": "赵太爷", "look": "stout"},
                   {"id": "zhao", "name": "赵秀才", "look": "young"}],
    "relationships": [{"from": "a_q", "to": "zhao", "relation": "雇工"}, {"from": "a_q", "to": "nobody"}],
    "locations": [{"id": "temple", "name": "土谷祠", "look": "a shrine"}],
    "props": [{"id": "pipe", "name": "烟管", "look": "a bamboo pipe"}],
    "episodes": [{"number": 1, "title": "优胜", "summary": "s1", "chapters": [1, 2]}],
}
SHOTS = [{"location": "temple", "characters": ["a_q", "ghost"], "speaker": "ghost", "line": "阿 Q 说：我们先前比你阔多啦！",
          "seconds": 12, "camera": "c", "action": "a", "after": "b", "then": "t", "sound": "s", "summary": "x"},
         {"location": "nowhere", "characters": ["zhao"], "speaker": "zhao", "line": "", "seconds": 3},
         {"location": "temple", "characters": [], "seconds": 9},
         {"location": "temple", "characters": [], "seconds": 9}]


def test_bible_gets_unique_ascii_ids_and_known_relationships():
    plan = planning.normalize_bible(BIBLE, [[1, 2], [3]])
    assert [c["id"] for c in plan["characters"]] == ["a_q", "zhao", "zhao_"]
    assert plan["relationships"] == [{"from": "a_q", "to": "zhao", "relation": "雇工"}]
    assert [e["chapters"] for e in plan["episodes"]] == [[1, 2], [3]]
    assert plan["episodes"][0]["title"] == "优胜" and plan["episodes"][1]["title"] == "第2集"


def test_shots_keep_known_ids_clamp_seconds_and_fit_the_episode():
    plan = planning.normalize_bible(BIBLE, [[1, 2]])
    shots = planning.normalize_shots(SHOTS, plan, episode=1, max_seconds=24)
    assert [s["id"] for s in shots] == ["e1s1", "e1s2", "e1s3"]
    assert [s["seconds"] for s in shots] == [9, 4, 9]
    assert shots[0]["characters"] == ["a_q"] and shots[0]["speaker"] == "a_q"
    assert shots[1]["location"] == "temple" and shots[1]["speaker"] == ""
    phone = {"location": "temple", "characters": ["a_q"], "speaker": "zhao", "line": "回来！", "seconds": 6}
    assert planning.normalize_shots([phone], plan, episode=1, max_seconds=24)[0]["speaker"] == "zhao"
    assert sum(s["seconds"] for s in shots) <= 24


def test_a_plan_needs_cast_and_places():
    with pytest.raises(ValueError):
        planning.normalize_bible({"characters": [], "locations": []}, [[1]])


class FakeChat:
    def __init__(self):
        self.prompts = []

    def complete_json(self, system, user, **_):
        self.prompts.append(user)
        if user.startswith("Summarize this chapter"):
            return {"summary": "事件", "characters": ["阿Q"], "places": ["未庄"], "quotes": []}
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
    project = Store(tmp_path).create("阿Q", "第一章 序\n甲\n第二章 优胜\n乙", flow="default@1",
                                     episodes=1, max_seconds=24)
    services = FakeServices()
    planning.run(Context(project, {"shot_seconds": 8}, "llm", None, services, progress=lambda _: None))
    plan = project.plan
    assert [note["heading"] for note in plan["chapters"]] == ["第一章 序", "第二章 优胜"]
    assert plan["episodes"][0]["shots"][0]["line"] == "阿Q说：我们先前比你阔多啦！"
    assert "第一章 序" in services.client.prompts[-1]  # the shot list sees the source chapters
