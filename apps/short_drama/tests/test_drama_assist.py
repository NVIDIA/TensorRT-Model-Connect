# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from trtmc_short_drama import assist
from trtmc_short_drama.flow import Context
from trtmc_short_drama.store import Store

SHOT = {"camera": "c", "action": "a", "after": "b", "then": "t", "sound": "s", "summary": "x", "props": []}
PLAN = {"title": "t", "setting": "a town", "style": "drama",
        "characters": [{"id": "aq", "name": "阿Q", "role": "雇工", "look": "a thin man", "voice": "hoarse"}],
        "locations": [{"id": "temple", "name": "土谷祠", "look": "a shrine"}], "props": [],
        "chapters": [{"number": 1, "summary": "s", "quotes": []}],
        "episodes": [{"number": 1, "chapters": [1], "shots": [
            {**SHOT, "id": "e1s1", "seconds": 6, "location": "temple", "characters": ["aq"], "speaker": "aq",
             "line": "滚！"},
            {**SHOT, "id": "e1s2", "seconds": 8, "location": "temple", "characters": ["aq"], "speaker": "",
             "line": ""}]}]}


class FakeChat:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def complete(self, system, user, **_):
        self.prompts.append(user)
        return self.reply

    def complete_json(self, system, user, **_):
        self.prompts.append(user)
        return copy.deepcopy(self.reply)


class FakeServices:
    def __init__(self, chat):
        self.chat_client = chat

    def chat(self, model):
        return self.chat_client


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path)


def act(project, run, chat, **params):
    run(Context(project, params, "q", None, FakeServices(chat), progress=lambda _: None))


def test_an_idea_becomes_a_story_with_chapters(store):
    project = store.create("idea", "", flow="default@1", episodes=2, max_seconds=60, idea="外卖骑手捡到一部手机")
    reply = "<think>plan</think>\n```\n好的。\n第一章 手机\n" + "他捡起手机。" * 40 + "\n第二章 追踪\n" + "他跑。" * 40 + "\n```"
    chat = FakeChat(reply)
    act(project, assist.run_story, chat)
    assert project.novel.startswith("第一章 手机") and "第二章 追踪" in project.novel and "```" not in project.novel
    assert "外卖骑手捡到一部手机" in chat.prompts[0] and "Exactly 2 chapters" in chat.prompts[0]
    with pytest.raises(ValueError, match="already has its story"):
        act(project, assist.run_story, chat)


def test_a_story_without_chapters_is_refused(store):
    project = store.create("idea", "", flow="default@1", episodes=2, max_seconds=60, idea="一个想法")
    with pytest.raises(ValueError, match="chapter headings"):
        act(project, assist.run_story, FakeChat("我不能写这个故事。"))
    assert project.novel == ""


def test_revising_a_shot_changes_only_that_shot_and_keeps_the_old_plan(store):
    project = store.create("t", "第一章\n文", flow="default@1", episodes=1, max_seconds=20)
    project.save_plan(copy.deepcopy(PLAN))
    revised = {**SHOT, "location": "temple", "characters": ["aq"], "speaker": "aq", "line": "我们先前比你阔多啦！",
               "seconds": 9, "action": "aq rubs his head, then speaks"}
    chat = FakeChat(revised)
    act(project, assist.run_revise, chat, target="shot:e1s1", instruction="让他先摸头再开口")
    shots = project.plan["episodes"][0]["shots"]
    assert shots[0]["id"] == "e1s1" and shots[0]["line"] == "我们先前比你阔多啦！"
    assert shots[0]["seconds"] == 9 and shots[1] == PLAN["episodes"][0]["shots"][1]
    assert "让他先摸头再开口" in chat.prompts[0] and '"e1s2"' not in chat.prompts[0]  # neighbours without ids
    history = project.snapshots()
    assert len(history) == 1 and history[0]["note"].startswith("before revising shot e1s1")
    project.restore(history[0]["name"])
    assert project.plan["episodes"][0]["shots"][0]["line"] == "滚！" and len(project.snapshots()) == 2


def test_a_revised_shot_must_fit_the_time_left_in_its_episode(store):
    project = store.create("t", "第一章\n文", flow="default@1", episodes=1, max_seconds=10)
    project.save_plan(copy.deepcopy(PLAN))
    with pytest.raises(ValueError, match="only 2 s left"):
        act(project, assist.run_revise, FakeChat({}), target="shot:e1s1", instruction="更长")


def test_revising_a_character_keeps_its_id(store):
    project = store.create("t", "第一章\n文", flow="default@1", episodes=1, max_seconds=60)
    project.save_plan(copy.deepcopy(PLAN))
    act(project, assist.run_revise, FakeChat({"id": "evil", "name": "阿 Q", "look": "a thin man with a scar",
                                               "voice": ""}), target="character:aq", instruction="脸上加一道疤")
    character = project.plan["characters"][0]
    assert character == {**PLAN["characters"][0], "name": "阿Q", "look": "a thin man with a scar"}


def test_revision_requests_name_a_target_and_a_short_note():
    assert assist.revision({"target": "prop:pipe", "instruction": " 换成铜的 "}) == ("prop", "pipe", "换成铜的")
    for bad in ({"target": "shot:../x", "instruction": "a"}, {"target": "shot:e1s1", "instruction": " "},
                {"target": "plan:all", "instruction": "a"}, {"target": "shot:e1s1", "instruction": "a" * 601}):
        with pytest.raises(ValueError):
            assist.revision(bad)


def test_the_model_review_keeps_problems_about_real_shots(store):
    project = store.create("t", "第一章\n文", flow="default@1", episodes=1, max_seconds=60)
    project.save_plan(copy.deepcopy(PLAN))
    chat = FakeChat([{"shot": "e1s2", "problem": "阿 Q 的烟管突然消失", "fix": "让他把烟管别在腰间"},
                     {"shot": "e9s9", "problem": "no such shot"}, {"shot": "e1s1", "problem": ""}])
    act(project, assist.run_review, chat)
    saved = project.review
    assert saved["items"] == [{"shot": "e1s2", "episode": 1, "problem": "阿Q的烟管突然消失", "fix": "让他把烟管别在腰间"}]
    assert saved["plan"] == assist.fingerprint(project.plan)
    plan = project.plan
    plan["episodes"][0]["shots"][1]["line"] = "嗯"
    assert assist.fingerprint(plan) != saved["plan"]  # an edited shot list makes the review stale
