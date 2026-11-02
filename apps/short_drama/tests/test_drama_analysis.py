# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess

import numpy as np
import pytest

from trtmc_short_drama import analysis, video
from trtmc_short_drama.flow import Context
from trtmc_short_drama.store import Store

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs FFmpeg")


def test_sound_finds_an_opening_burst_and_clipping():
    rate = analysis.RATE
    quiet = 0.01 * np.sin(np.linspace(0, 2000, 3 * rate)).astype(np.float32)
    burst = quiet.copy()
    burst[320:2400] = 0.8 * np.sin(np.linspace(0, 400, 2080))
    assert analysis.sound(quiet)["burst"] < analysis.BURST
    measured = analysis.sound(burst)
    assert measured["burst"] >= analysis.BURST and measured["clipping"] == 0.0
    loud = np.clip(4 * np.sin(np.linspace(0, 2000, 3 * rate)), -1, 1).astype(np.float32)
    assert analysis.sound(loud)["clipping"] >= analysis.CLIPPING
    assert analysis.sound(np.zeros(100, dtype=np.float32))["level_db"] is None


def test_picture_change_at_a_cut():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (64, 36, 3), dtype=np.uint8)
    assert analysis.jump(frame, frame) == 0.0
    warm_dark = np.full((64, 36, 3), (90, 40, 20), dtype=np.uint8)
    cool_bright = np.full((64, 36, 3), (170, 210, 240), dtype=np.uint8)
    assert analysis.jump(warm_dark, cool_bright) >= analysis.JUMP
    still = np.repeat(frame[None], 4, axis=0)
    assert analysis.picture(still) == {"black": 0.0, "motion": 0.0}
    assert analysis.picture(np.zeros((4, 64, 36, 3), dtype=np.uint8))["black"] == 1.0


def test_measured_findings_name_the_shot_and_what_to_do():
    measures = {"sound": {"level_db": -20.0, "burst": 1.0, "clipping": 0.0}, "picture": {"black": 0.0, "motion": 9.0},
                "speech": {"heard": "嗯", "match": 0.0}}
    shot = {"id": "e1s2", "line": "先排队。"}
    [missing] = analysis.measured_findings("zh", shot, measures)
    assert missing["category"] == "speech" and missing["action"] == "revise" and missing["target"] == "shot:e1s2"
    assert "先排队" in missing["finding"]
    unplanned = analysis.measured_findings("en", {"id": "e1s3", "line": ""},
                                           {**measures, "speech": {"heard": "who is there", "match": 0.0}})
    assert [f["id"] for f in unplanned] == ["e1s3-unplanned"] and unplanned[0]["finding"].startswith("someone speaks")
    credits = {**measures, "speech": {"heard": "字幕志愿者 杨茜茜", "match": 0.0}}
    assert analysis.measured_findings("zh", {"id": "e1s2", "line": ""}, credits) == []  # Whisper on quiet audio
    cuts = [{"from": "e1s1", "to": "e1s2", "same_place": True, "jump": 0.6, "level_jump_db": 12.0},
            {"from": "e1s2", "to": "e1s3", "same_place": False, "jump": 0.9, "level_jump_db": 1.0}]
    found = analysis.cut_findings("en", cuts)
    assert [f["id"] for f in found] == ["e1s1-e1s2-jump", "e1s1-e1s2-level"]  # a new place is a cut, not a jump
    assert found[0]["target"] == "shot:e1s2" and found[1]["action"] == "none"


class FakeEngine:
    def __init__(self):
        self.heard = []

    def transcribe(self, model, wav):
        self.heard.append(model)
        return "先排队蛋糕三十八"


class FakeVision:
    model = "seeing-model"

    def __init__(self):
        self.calls = []

    def complete_json(self, system, user, *, max_tokens, temperature, images=()):
        self.calls.append({"user": user, "images": len(images)})
        return [{"category": "consistency", "severity": "high", "finding": "她的发型和参考图不同",
                 "target": "character:linwan", "action": "redraw", "suggestion": "参考图里把头发扎起来"},
                {"category": "text", "finding": "门上有乱码文字", "target": "character:nobody", "action": "redraw",
                 "suggestion": "画面里不要出现文字"},
                {"category": "made-up", "finding": "x"}]


class FakeServices:
    def __init__(self, vision=None):
        self.engine = FakeEngine()
        self.loaded = []
        self._vision = vision

    def load(self, model):
        self.loaded.append(model)

    def vision(self):
        return self._vision


def make_clip(path, colour, tone):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={colour}:s=90x160:d=2:r=24",
                    "-f", "lavfi", "-i", f"sine=frequency={tone}:duration=2:sample_rate=48000",
                    "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)], check=True)


@needs_ffmpeg
def test_analysis_measures_every_shot_and_keeps_vision_findings_in_bounds(tmp_path):
    project = Store(tmp_path).create("t", "第一章\n文", flow="default@1", episodes=1, max_seconds=60)
    shots = [{"id": "e1s1", "seconds": 2, "location": "shop", "characters": ["linwan"], "props": [], "line": "",
              "summary": "s"},
             {"id": "e1s2", "seconds": 2, "location": "shop", "characters": ["linwan"], "props": [],
              "speaker": "linwan", "line": "先排队。蛋糕三十八。", "summary": "s"}]
    project.save_plan({"title": "t", "language": "zh", "characters": [{"id": "linwan", "name": "林晚", "look": "l"}],
                       "locations": [{"id": "shop", "name": "店", "look": "l"}], "props": [],
                       "episodes": [{"number": 1, "title": "一", "summary": "", "shots": shots}]})
    (project.root / "stills").mkdir()
    for key, colour in (("character-linwan", "red"), ("location-shop", "blue")):
        make_clip(tmp_path / "src.mp4", colour, 300)
        video.last_frame(tmp_path / "src.mp4", project.root / "stills" / f"{key}.png")
    project.save_stills({"character:linwan": {"file": "stills/character-linwan.png", "status": "approved"},
                         "location:shop": {"file": "stills/location-shop.png", "status": "approved"}})
    episode = project.episode_dir(1)
    make_clip(episode / "clips" / "e1s1-take1.mp4", "black", 300)
    make_clip(episode / "clips" / "e1s2-take1.mp4", "white", 600)
    for shot in ("e1s1", "e1s2"):
        video.contact_sheet(episode / "clips" / f"{shot}-take1.mp4", episode / "sheets" / f"{shot}-take1.jpg")
    project.save_shot_state(1, {shot: {"take": 1, "file": f"episodes/01/clips/{shot}-take1.mp4",
                                       "sheet": f"episodes/01/sheets/{shot}-take1.jpg"} for shot in ("e1s1", "e1s2")})
    vision = FakeVision()
    services = FakeServices(vision)
    analysis.run_analyze(Context(project, {}, "whisper", 1, services, progress=lambda _: None))
    result = analysis.load(project, 1)
    assert set(result["shots"]) == {"e1s1", "e1s2"} and services.loaded == ["whisper"]
    assert result["shots"]["e1s2"]["speech"]["match"] >= 0.6 and result["shots"]["e1s1"]["picture"]["black"] == 1.0
    [cut] = result["cuts"]
    assert cut["same_place"] and cut["jump"] >= analysis.JUMP  # black to white in one place
    ids = [f["id"] for f in result["findings"]]
    assert "e1s1-black" in ids and "e1s1-unplanned" in ids and "e1s1-e1s2-jump" in ids
    seen = [f for f in result["findings"] if f["source"] == "vision"]
    assert {f["target"] for f in seen} == {"character:linwan", "shot:e1s1", "shot:e1s2"}
    garbled = next(f for f in seen if f["category"] == "text")
    assert garbled["action"] == "revise"  # an unknown target falls back to the shot, which cannot be redrawn
    assert vision.calls[1]["images"] == 4 and "last frame of the previous shot" in vision.calls[1]["user"]
    # a speech model built for Chinese does not judge the lines of an English project
    plan = project.plan
    project.save_plan({**plan, "language": "en"})
    english = FakeServices()
    english.languages = {"whisper": "zh"}
    analysis.run_analyze(Context(project, {"only": ["e1s1"]}, "whisper", 1, english, progress=lambda _: None))
    checked = analysis.load(project, 1)
    assert checked["shots"]["e1s1"]["speech"]["unchecked"] and "e1s1-unplanned" not in {f["id"] for f in checked["findings"]}
    project.save_plan(plan)
    analysis.run_analyze(Context(project, {"only": ["e1s1"]}, "whisper", 1, FakeServices(FakeVision()),
                                 progress=lambda _: None))
    # analysing one shot again keeps the other shot's numbers
    analysis.run_analyze(Context(project, {"only": ["e1s2"]}, "whisper", 1, FakeServices(), progress=lambda _: None))
    again = analysis.load(project, 1)
    assert again["shots"]["e1s1"] == result["shots"]["e1s1"] and again["vision"] is None
    kept = {f["id"] for f in again["findings"]}
    assert {"e1s1-black", "e1s1-unplanned", "e1s1-vision-1"} <= kept  # e1s1 was not analysed again
    assert "e1s1-e1s2-jump" in kept and not any(f["id"].startswith("e1s2-vision") for f in again["findings"])
