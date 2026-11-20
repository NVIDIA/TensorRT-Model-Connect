# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The page and the prompts in both languages."""

from pathlib import Path
import re

from trtmc_short_drama import catalog, prompts

STATIC = Path(__file__).resolve().parents[1] / "trtmc_short_drama" / "static"


def page_texts() -> set[str]:
    """Every literal the page passes to t(), with adjacent literals joined as JavaScript joins them."""

    texts = set()
    for name in ("app.js", "project.js"):
        source = (STATIC / name).read_text(encoding="utf-8")
        for match in re.finditer(r'\bt\(\s*((?:"(?:[^"\\]|\\.)*"\s*\+\s*)*"(?:[^"\\]|\\.)*")', source):
            texts.add("".join(re.findall(r'"((?:[^"\\]|\\.)*)"', match.group(1))).replace("\\'", "'"))
    return texts


def chinese_keys() -> set[str]:
    source = (STATIC / "i18n.js").read_text(encoding="utf-8")
    block = source[source.index("const ZH = {"):source.index("// Text the server writes")]
    return {key.replace("\\'", "'") for key in re.findall(r'"((?:[^"\\]|\\.)*)":\s', block)}


def test_every_page_text_has_a_chinese_translation():
    missing = sorted(page_texts() - chinese_keys())
    assert not missing, missing


def test_the_tier_texts_the_page_shows_are_translated():
    keys = chinese_keys()
    for tier in catalog.describe_tiers():
        for text in (tier["summary"], tier["video"], tier["speech"], *tier["missing"]):
            assert text in keys, text


def test_prompts_follow_the_project_language():
    english, chinese = prompts.words("en"), prompts.words("zh")
    assert english["language"] == "English" and "12 words" in english["line_rule"]
    assert chinese["language"] == "Chinese" and prompts.words("xx") == chinese
    assert prompts.language_of({}) == "zh" and prompts.language_of({"language": "en"}) == "en"
    shot_prompt = prompts.SHOTS_PROMPT.format(number=1, title="t", setting="s", notes="[]", max_seconds=30,
                                              min_shots=2, max_shots=3, characters="[]", locations="[]",
                                              props="[]", source="x", **english)
    assert "in English" in shot_prompt and "What did you say?" in shot_prompt and "{" not in shot_prompt[:200]
    story = prompts.STORY_PROMPT.format(idea="i", episodes=2, max_seconds=30, **english)
    assert "Chapter 1: <title>" in story and "300 to 600 words" in story
    assert prompts.chapters("Chapter 1: A\ntext one\nChapter 2: B\ntext two") == ["Chapter 1: A\ntext one",
                                                                                 "Chapter 2: B\ntext two"]


def test_a_line_is_tagged_with_its_own_language():
    plan = {"characters": [{"id": "lin", "name": "Lin", "look": "a woman"}], "style": "drama"}
    shot = {"speaker": "lin", "line": "Queue first.", "camera": "c", "action": "a", "after": "b", "then": "t",
            "sound": "s", "summary": "x"}
    first, _ = prompts.video_prompts(plan, shot, [("character", plan["characters"][0])])
    assert "<d>[English] Queue first.</d>" in first
    first, _ = prompts.video_prompts(plan, {**shot, "line": "先排队。"}, [("character", plan["characters"][0])])
    assert "<d>[Chinese] 先排队。</d>" in first
