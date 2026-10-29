# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from trtmc_short_drama import prompts
from trtmc_short_drama.llm import ChatClient, parse_json

PLAN = {
    "style": "realistic live-action period drama",
    "setting": "a river town around 1911",
    "characters": [{"id": "aq", "name": "阿Q", "look": "a thin man with a \"queue\"", "voice": "a hoarse male voice"},
                   {"id": "wu", "name": "吴妈", "look": "a maid in a blue tunic", "voice": "a soft female voice"}],
    "locations": [{"id": "kitchen", "name": "厨房", "look": "a dim kitchen with a clay stove"}],
    "props": [],
}
SHOT = {"camera": "A handheld medium shot slowly pushes in", "action": "aq kneels beside wu",
        "speaker": "aq", "line": "“我和你困觉！”", "after": "wu drops the pipe and runs", "then": "aq stays frozen",
        "sound": "crackling stove", "summary": "aq startles wu in the kitchen", "characters": ["aq", "wu"]}


def test_chapters_split_on_headings_and_pack_long_text():
    novel = "第一章 序\n" + "甲" * 50 + "\n第二章 优胜记略\n" + ("乙" * 30 + "\n") * 10
    parts = prompts.chapters(novel, limit=200)
    assert parts[0].startswith("第一章") and parts[1].startswith("第二章") and parts[2].startswith("乙")
    assert all(len(part) <= 200 for part in parts) and len(parts) == 3


def test_episodes_get_contiguous_chapters_of_similar_length():
    lengths = [1728, 2204, 2227, 2665, 2257, 2709, 2461, 2562, 2908]
    assert prompts.episode_chapters(lengths, 4) == [[1, 2, 3], [4, 5], [6, 7], [8, 9]]
    assert prompts.episode_chapters(lengths, 1) == [list(range(1, 10))]
    assert prompts.episode_chapters([10, 10], 3) == [[1], [1], [2]]


def test_chinese_names_in_descriptions_become_subject_labels():
    references = [("character", PLAN["characters"][0]), ("location", PLAN["locations"][0])]
    first, _ = prompts.video_prompts(PLAN, {**SHOT, "action": "阿 Q stares at the stove"}, references)
    assert "<Subject 1> stares at the stove" in first
    capital = {**PLAN, "setting": "A river town", "style": "Period drama"}
    assert "style set in a river town, Period drama, and" in prompts.video_prompts(capital, SHOT, references)[0]
    named = {**PLAN, "setting": "Weizhuang, a village"}
    assert "set in Weizhuang, a village" in prompts.video_prompts(named, SHOT, references)[0]
    lower = {**SHOT, "then": "he lowers his hand"}
    assert "drifting slowly. He lowers his hand. Nobody speaks." in prompts.video_prompts(PLAN, lower, references)[1]


def test_video_prompt_numbers_pictures_in_upload_order_and_quotes_nothing():
    references = [("character", PLAN["characters"][0]), ("character", PLAN["characters"][1]),
                  ("location", PLAN["locations"][0]), ("previous", {})]
    first, continuation = prompts.video_prompts(PLAN, SHOT, references)
    assert "<Subject 1> is 阿Q from <Picture 1>" in first
    assert "<Subject 3> is the place in <Picture 3>" in first
    assert "<Subject 4> is the final frame of the previous episode in <Picture 4>" in first
    assert "<Subject 1> kneels beside <Subject 2>" in first
    assert "In a hoarse male voice, <Subject 1> (S1) says <d>[Chinese] 我和你困觉！</d>" in first
    assert '"' not in first and "“" not in first
    assert ("uses a realistic live-action vertical short-drama style set in a river town around 1911, "
            "realistic live-action period drama, and a clean image without subtitles") in first
    assert first.index("subject_definitions:") < first.index("detailed_description:") < first.index("non_diegetic_music:")
    assert "[Shot 1 continued]" in continuation and "<d>" not in continuation
    assert "<Subject 1> stays frozen" in continuation


def test_a_line_from_someone_off_screen_becomes_a_voiceover():
    references = [("character", PLAN["characters"][1]), ("location", PLAN["locations"][0])]
    first, _ = prompts.video_prompts(PLAN, {**SHOT, "characters": ["wu"]}, references)
    assert "off-screen voiceover: <d>[Chinese] 我和你困觉！</d>" in first


def test_still_prompts_describe_each_kind():
    assert "Full-body portrait of a thin man with a queue" in prompts.still_prompt(PLAN, "character", PLAN["characters"][0])
    assert "An empty establishing view" in prompts.still_prompt(PLAN, "location", PLAN["locations"][0])


def test_parse_json_ignores_fences_and_thinking():
    assert parse_json('<think>hm</think>\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert parse_json('Sure: [{"x": 1}] done') == [{"x": 1}]


def test_parse_json_escapes_quotes_inside_chinese_prose():
    reply = '{\n  "summary": "最终暂用“阿 Q"。全篇自嘲",\n  "places": ["未庄", "土谷祠"]\n}'
    assert parse_json(reply) == {"summary": "最终暂用“阿 Q\"。全篇自嘲", "places": ["未庄", "土谷祠"]}


def test_chat_client_never_shows_its_key():
    client = ChatClient("https://example.invalid/v1", "m", "secret-key", local=False)
    assert "secret-key" not in repr(client) and "secret-key" not in str(client)
