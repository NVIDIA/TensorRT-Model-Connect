# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from trtmc_short_drama import review

SHOT = {"camera": "c", "action": "a", "after": "b", "then": "t", "sound": "s", "props": []}
PLAN = {
    "title": "t", "setting": "a town", "style": "drama",
    "characters": [{"id": "zhou", "name": "Zhou"}, {"id": "mei", "name": "Mei"}],
    "locations": [{"id": "dock", "name": "Dock"}], "props": [],
    "chapters": [{"number": 1, "characters": ["Zhou", "Little Apprentice", "Apprentice", "Old Chen"]}],
    "episodes": [{"number": 1, "chapters": [1], "shots": [
        {**SHOT, "id": "e1s1", "seconds": 6, "location": "dock", "characters": ["zhou"], "speaker": "zhou",
         "line": "I'll take you across!", "summary": "zhou waves to mei"},
        {**SHOT, "id": "e1s2", "seconds": 4, "location": "yard", "characters": ["zhou"], "speaker": "mei",
         "line": "It is getting dark, why are you two not home for dinner?", "summary": "zhou is called home"}]}],
}
NOVEL = ("Chapter 1\nThe little apprentice runs in. Zhou pats the little apprentice on the head. "
         "The little apprentice laughs. Old Chen sits in the sun. Mei, I'll take you across!")


def test_checks_find_uncast_people_unknown_ids_and_lines_that_do_not_fit():
    problems = review.check(PLAN, NOVEL, max_seconds=60)
    found = {(p["level"], p["where"]) for p in problems}
    text = " | ".join(p["message"] for p in problems)
    assert ("warning", "episode 1") in found and "Little Apprentice is named 3 times" in text
    assert "Apprentice is named" not in text.replace("Little Apprentice is named", "")  # part of a fuller name
    assert "Old Chen" not in text  # named once: probably not on screen
    assert ("warning", "e1s1") in found and "mentions Mei" in text  # mei acts in e1s1 without a still
    assert ("error", "e1s2") in found and "location:yard" in text
    assert "spoken by Mei, who is not in the shot" in text
    assert "the line needs about 7 s" in text  # 12 words of 1.8 characters at 4.5 a second, plus 2 s
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
    plan["episodes"][0]["shots"][0]["line"] = "嗯"  # one Chinese character tells nothing
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
