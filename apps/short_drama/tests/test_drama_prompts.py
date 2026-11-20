# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from trtmc_short_drama import prompts
from trtmc_short_drama.llm import ChatClient, parse_json

PLAN = {
    "style": "realistic live-action period drama",
    "setting": "a river town around 1911",
    "characters": [{"id": "zhou", "name": "Zhou", "look": "a thin ferryman with a \"straw\" hat", "voice": "a hoarse male voice"},
                   {"id": "mei", "name": "Mei", "look": "a young woman in a blue tunic", "voice": "a soft female voice"}],
    "locations": [{"id": "kitchen", "name": "Kitchen", "look": "a dim kitchen with a clay stove"}],
    "props": [],
}
SHOT = {"camera": "A handheld medium shot slowly pushes in", "action": "zhou kneels beside mei",
        "speaker": "zhou", "line": "“Dinner is ready!”", "after": "mei drops the ladle and laughs", "then": "zhou stays frozen",
        "sound": "crackling stove", "summary": "zhou surprises mei in the kitchen", "characters": ["zhou", "mei"]}


def test_chapters_split_on_headings_and_pack_long_text():
    novel = "Chapter 1: Prologue\n" + "a" * 50 + "\nChapter 2: The Crossing\n" + ("b" * 30 + "\n") * 10
    parts = prompts.chapters(novel, limit=200)
    assert parts[0].startswith("Chapter 1") and parts[1].startswith("Chapter 2") and parts[2].startswith("b")
    assert all(len(part) <= 200 for part in parts) and len(parts) == 3
    chinese = prompts.chapters("第一章\n甲\n第二章\n乙")  # Chinese headings
    assert [part[:3] for part in chinese] == ["第一章", "第二章"]


def test_episodes_get_contiguous_chapters_of_similar_length():
    lengths = [1728, 2204, 2227, 2665, 2257, 2709, 2461, 2562, 2908]
    assert prompts.episode_chapters(lengths, 4) == [[1, 2, 3], [4, 5], [6, 7], [8, 9]]
    assert prompts.episode_chapters(lengths, 1) == [list(range(1, 10))]
    assert prompts.episode_chapters([10, 10], 3) == [[1], [1], [2]]


def test_chinese_names_in_descriptions_become_subject_labels():
    references = [("character", PLAN["characters"][0]), ("location", PLAN["locations"][0])]
    zhou = {**PLAN["characters"][0], "name": "老周"}  # models may split a Chinese name with spaces
    chinese = {**PLAN, "characters": [zhou, PLAN["characters"][1]]}
    first, _ = prompts.video_prompts(chinese, {**SHOT, "action": "老 周 stares at the stove"},
                                     [("character", zhou), references[1]])
    assert "<Subject 1> stares at the stove" in first
    capital = {**PLAN, "setting": "A river town", "style": "Period drama"}
    assert "style set in a river town, Period drama, and" in prompts.video_prompts(capital, SHOT, references)[0]
    named = {**PLAN, "setting": "Maple Creek, a village"}
    assert "set in Maple Creek, a village" in prompts.video_prompts(named, SHOT, references)[0]
    lower = {**SHOT, "then": "he lowers his hand"}
    assert "drifting slowly. He lowers his hand. Nobody speaks." in prompts.video_prompts(PLAN, lower, references)[1]


def test_video_prompt_numbers_pictures_in_upload_order_and_quotes_nothing():
    references = [("character", PLAN["characters"][0]), ("character", PLAN["characters"][1]),
                  ("location", PLAN["locations"][0]), ("previous", {})]
    first, continuation = prompts.video_prompts(PLAN, SHOT, references)
    assert "<Subject 1> is Zhou from <Picture 1>" in first
    assert "<Subject 3> is the place in <Picture 3>" in first
    assert "<Subject 4> is the final frame of the previous episode in <Picture 4>" in first
    assert "<Subject 1> kneels beside <Subject 2>" in first
    assert "In a hoarse male voice, <Subject 1> (S1) says <d>[English] Dinner is ready!</d>" in first
    assert '"' not in first and "“" not in first
    assert ("uses a vertical short-drama style set in a river town around 1911, "
            "realistic live-action period drama, and a clean image without subtitles") in first
    assert first.index("subject_definitions:") < first.index("detailed_description:") < first.index("non_diegetic_music:")
    assert "[Shot 1 continued]" in continuation and "<d>" not in continuation
    assert "<Subject 1> stays frozen" in continuation


def test_a_line_from_someone_off_screen_becomes_a_voiceover():
    references = [("character", PLAN["characters"][1]), ("location", PLAN["locations"][0])]
    first, _ = prompts.video_prompts(PLAN, {**SHOT, "characters": ["mei"]}, references)
    # zhou is heard, not seen (as on a phone): his voice, a speaker id, and closed lips on screen
    assert "A hoarse male voice (S1) says in an off-screen voiceover: <d>[English] Dinner is ready!</d>" in first
    assert "keep their lips closed" in first
    first, _ = prompts.video_prompts(PLAN, {**SHOT, "characters": ["mei"], "speaker": ""}, references)
    assert "An off-screen narrator says in an off-screen voiceover: <d>[English] Dinner is ready!</d>" in first


def test_descriptions_in_another_language_reach_the_video_model_in_english():
    references = [("character", PLAN["characters"][0]), ("location", PLAN["locations"][0])]
    # a Chinese project: the descriptions are translated, the line is not
    shot = {**SHOT, "action": "zhou 跪在地上", "sound": "风声", "summary": "zhou 道歉", "line": "饭做好了！"}
    assert prompts.needs_english(shot["action"]) and not prompts.needs_english("zhou kneels")
    english = {"zhou 跪在地上": "zhou kneels on the floor", "风声": "Wind", "zhou 道歉": "zhou apologizes"}.get
    first, _ = prompts.video_prompts(PLAN, shot, references, lambda text: english(text, text))
    assert "<Subject 1> kneels on the floor." in first and "overall_soundscape:\nWind." in first
    assert "[reference generation] <Subject 1> apologizes." in first
    assert "<d>[Chinese] 饭做好了！</d>" in first and "跪" not in first  # the line stays as written
    untranslated, _ = prompts.video_prompts(PLAN, shot, references)
    assert "<Subject 1> 跪在地上." in untranslated


def test_a_shot_without_a_line_says_that_no_one_speaks():
    references = [("character", PLAN["characters"][0]), ("location", PLAN["locations"][0])]
    first, _ = prompts.video_prompts(PLAN, {**SHOT, "speaker": "", "line": ""}, references)
    assert "No one speaks; the people on screen keep their lips closed." in first and "<d>" not in first


def test_still_prompts_describe_each_kind():
    assert "Full-body portrait of a thin ferryman with a straw hat" in prompts.still_prompt(PLAN, "character", PLAN["characters"][0])
    assert "An empty establishing view" in prompts.still_prompt(PLAN, "location", PLAN["locations"][0])


def test_the_plan_style_alone_sets_the_look():
    cartoon = {**PLAN, "style": "bright 3D cartoon animation"}
    references = [("character", PLAN["characters"][0]), ("location", PLAN["locations"][0])]
    texts = [prompts.video_prompts(cartoon, SHOT, references)[0],
             prompts.still_prompt(cartoon, "character", PLAN["characters"][0]),
             prompts.still_prompt(cartoon, "location", PLAN["locations"][0]),
             prompts.still_prompt(cartoon, "prop", {"look": "a red kite"})]
    assert all("bright 3D cartoon animation" in text and "live-action" not in text and
               "realistic" not in text for text in texts)
    assert all("live-action" not in text for text in (prompts.SYSTEM, prompts.STORY_SYSTEM, prompts.STORY_PROMPT))
    unstyled = {**PLAN, "style": ""}
    assert prompts.DEFAULT_STYLE in prompts.video_prompts(unstyled, SHOT, references)[0]
    assert prompts.DEFAULT_STYLE in prompts.still_prompt(unstyled, "prop", {"look": "a red kite"})


def test_parse_json_ignores_fences_and_thinking():
    assert parse_json('<think>hm</think>\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert parse_json('Sure: [{"x": 1}] done') == [{"x": 1}]


def test_parse_json_escapes_quotes_inside_chinese_prose():
    reply = '{\n  "summary": "取名“小白"吧",\n  "places": ["dock"]\n}'
    assert parse_json(reply) == {"summary": "取名“小白\"吧", "places": ["dock"]}


def test_chat_client_never_shows_its_key():
    client = ChatClient("https://example.invalid/v1", "m", "secret-key", local=False)
    assert "secret-key" not in repr(client) and "secret-key" not in str(client)
