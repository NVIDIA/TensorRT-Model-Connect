# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from trtmc_short_drama import review

SHOT = {"camera": "c", "action": "a", "after": "b", "then": "t", "sound": "s", "props": []}
PLAN = {
    "title": "t", "setting": "a town", "style": "drama",
    "characters": [{"id": "zhou", "name": "老周"}, {"id": "mei", "name": "小梅"}],
    "locations": [{"id": "dock", "name": "渡口"}], "props": [],
    "chapters": [{"number": 1, "characters": ["老周", "小学徒", "学徒", "陈伯"]}],
    "episodes": [{"number": 1, "chapters": [1], "shots": [
        {**SHOT, "id": "e1s1", "seconds": 6, "location": "dock", "characters": ["zhou"], "speaker": "zhou",
         "line": "我送你过河！", "summary": "zhou waves to mei"},
        {**SHOT, "id": "e1s2", "seconds": 4, "location": "yard", "characters": ["zhou"], "speaker": "mei",
         "line": "天都黑了你们两个怎么还不回家吃饭呀", "summary": "zhou is called home"}]}],
}
NOVEL = "第一章\n小学徒跑来。老周摸摸小学徒的头。小学徒笑了。陈伯在晒太阳。小梅，我送你过河！"


def test_checks_find_uncast_people_unknown_ids_and_lines_that_do_not_fit():
    problems = review.check(PLAN, NOVEL, max_seconds=60)
    found = {(p["level"], p["where"]) for p in problems}
    text = " | ".join(p["message"] for p in problems)
    assert ("warning", "episode 1") in found and "小学徒 is named 3 times" in text
    assert "学徒 is named" not in text.replace("小学徒 is named", "")  # part of a fuller name
    assert "陈伯" not in text  # named once: probably not on screen
    assert ("warning", "e1s1") in found and "mentions 小梅" in text  # mei acts in e1s1 without a still
    assert ("error", "e1s2") in found and "location:yard" in text
    assert "spoken by 小梅, who is not in the shot" in text
    assert "the line needs about 6 s" in text
    assert not [p for p in review.check(PLAN, NOVEL, max_seconds=10) if "over the" in p["message"]]
    assert [p for p in review.check(PLAN, NOVEL, max_seconds=8) if "over the 8 s limit" in p["message"]]


def test_a_clean_plan_has_no_problems_and_shots_cannot_outlast_a_clip():
    plan = copy.deepcopy(PLAN)
    plan["chapters"] = []
    plan["episodes"][0]["shots"] = plan["episodes"][0]["shots"][:1]
    plan["episodes"][0]["shots"][0]["summary"] = "zhou kneels by the boat"
    assert review.check(plan, NOVEL, max_seconds=60) == []
    plan["episodes"][0]["shots"][0]["seconds"] = 12
    assert [p["level"] for p in review.check(plan, NOVEL, max_seconds=60)] == ["error"]


def test_lines_quoted_from_the_novel_are_marked():
    assert review.origins(PLAN, NOVEL) == {"e1s1": "novel", "e1s2": "adapted"}
    plan = copy.deepcopy(PLAN)
    plan["episodes"][0]["shots"][0]["line"] = "嘿！"
    assert review.origins(plan, NOVEL) == {"e1s2": "adapted"}


def test_progress_names_the_next_step_of_each_stage():
    assert review.progress({}, {}, {}, novel=False, problems=[])[0]["next"] == "Write the story from the idea"
    plan = copy.deepcopy(PLAN)
    plan["episodes"][0]["shots"][1]["location"] = "dock"
    stills = {"character:zhou": {"file": "a.png", "status": "approved"}}
    stages = {s["key"]: s for s in review.progress(plan, stills, {}, novel=True, problems=[])}
    assert stages["plan"]["status"] == "done" and stages["stills"]["next"] == "Draw 1 still"
    assert stages["shots"]["next"] == "Approve the stills first"

    stills["location:dock"] = {"file": "t.png", "status": "approved"}
    stages = {s["key"]: s for s in review.progress(plan, stills, {}, novel=True, problems=[])}
    assert stages["stills"]["status"] == "done" and stages["shots"]["next"] == "Render a sample shot first"

    episodes = {"1": {"e1s1": {"file": "c.mp4", "outdated": ["prompt"]}}}
    stages = {s["key"]: s for s in review.progress(plan, stills, episodes, novel=True,
                                                   problems=[{"level": "error"}])}
    assert stages["plan"]["next"] == "Fix 1 problem in the plan"
    assert stages["shots"]["next"] == "Re-render 1 outdated"

    episodes = {"1": {"e1s1": {"file": "c.mp4"}, "e1s2": {"file": "d.mp4"}}}
    stages = {s["key"]: s for s in review.progress(plan, stills, episodes, novel=True, problems=[])}
    assert stages["shots"]["status"] == "done" and stages["episodes"]["next"] == "Assemble episode 1"
    episodes["1"]["assembled"] = {"duration": 10}
    assert review.progress(plan, stills, episodes, novel=True, problems=[])[3]["status"] == "done"
