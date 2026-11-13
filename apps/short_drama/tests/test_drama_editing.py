# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
from pathlib import Path
import shutil
import subprocess
import wave

import pytest

from trtmc_short_drama import editing, video

RATE = 16000
# The speech model is built for Chinese.
LINE = "王总这是新的报告"


def silent_wav(seconds: float) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(RATE)
        audio.writeframes(b"\0\0" * int(seconds * RATE))
    return output.getvalue()


def fake_recognizer(spoken_from: float, spoken_to: float, total: float):
    """Hear the line only when a cut contains its whole interval; cuts are told apart by length."""

    calls = []

    def transcribe(wav: bytes) -> str:
        length = editing.duration(wav)
        calls.append(length)
        # A prefix ends at `length`; a suffix starts at `total - length` when it is a suffix call.
        return LINE if length >= spoken_to - 1e-6 else "王总"

    return transcribe, calls


def test_coverage_counts_characters_heard_in_order():
    assert editing.coverage(" boss here's  the plan", "Boss, here's the plan.") == 1.0
    assert editing.coverage("Boss", "Boss, here's the plan") == 0.25
    assert editing.coverage("", "") == 0.0


def test_cut_keeps_the_sample_format():
    part = editing.cut(silent_wav(2.0), 0.5, 1.5)
    assert abs(editing.duration(part) - 1.0) < 1e-3


def test_line_end_is_found_by_binary_search():
    transcribe, calls = fake_recognizer(1.0, 2.5, 6.0)
    timing = editing.time_line(transcribe, silent_wav(6.0), LINE)
    assert timing["timing"] == "asr" and timing["match"] == 1.0
    assert 2.5 <= timing["end"] <= 2.8
    assert timing["end"] - timing["start"] >= 0.18 * len(LINE)
    assert len(calls) < 25


def test_unheard_line_gets_an_estimated_slot():
    timing = editing.time_line(lambda _: "something else", silent_wav(4.0), LINE)
    assert timing["timing"] == "estimated" and timing["start"] < timing["end"] <= 4.0


def test_subtitles_follow_the_shots_on_the_timeline():
    shots = [{"id": "a", "seconds": 5, "line": "First line."}, {"id": "b", "seconds": 6, "line": ""},
             {"id": "c", "seconds": 4, "line": "“Third line.”"}]
    state = {"a": {"speech": {"start": 0.4, "end": 1.9}}, "c": {"speech": {"start": 0.2, "end": 9.0}}}
    srt = editing.subtitles(shots, state)
    assert srt.splitlines()[:3] == ["1", "00:00:00,400 --> 00:00:01,900", "First line."]
    assert "00:00:11,200 --> 00:00:15,000\nThird line." in srt
    shorter = editing.subtitles(shots, state, {"a": 5.0, "b": 4.9, "c": 3.5})  # b skipped a held picture
    assert "00:00:10,100 --> 00:00:13,400\nThird line." in shorter


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_clips_become_vertical_1080p_and_join_with_subtitles(tmp_path: Path):
    raw = tmp_path / "raw.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=384x672:rate=24:duration=3",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=32000:duration=3",
                    "-shortest", "-c:v", "libx264", "-c:a", "aac", str(raw)], check=True)
    clip = tmp_path / "clips" / "a.mp4"
    video.finish_clip(raw, clip, seconds=2.0)
    info = video.probe(clip)
    assert (info["width"], info["height"], info["audio"]) == (1080, 1920, True)
    assert abs(info["duration"] - 2.0) < 0.1
    (tmp_path / "episode.srt").write_text("1\n00:00:00,200 --> 00:00:01,500\nTest\n", encoding="utf-8")
    output = tmp_path / "episode.mp4"
    video.export_episode([clip, clip], tmp_path / "episode.srt", output, max_seconds=3.0,
                         font="Noto Sans CJK SC", font_size=12, crf=23)
    assert abs(video.probe(output)["duration"] - 3.0) < 0.15
    video.last_frame(output, tmp_path / "last.png")
    assert (tmp_path / "last.png").stat().st_size > 0
    with wave.open(io.BytesIO(video.speech_wav(clip))) as audio:  # sizes must match the samples
        assert audio.getframerate() == 16000 and audio.getnchannels() == 1
        assert len(audio.readframes(audio.getnframes())) == audio.getnframes() * 2 > 16000


def test_subtitle_fonts_cannot_inject_filters(tmp_path: Path):
    with pytest.raises(ValueError):
        video.export_episode([], tmp_path / "e.srt", tmp_path / "e.mp4", max_seconds=1,
                             font="x',drawtext=text='y", font_size=12, crf=18)


def test_timing_counts_chinese_characters_and_english_words():
    word = editing.WORD_IN_CHARACTERS
    assert editing.spoken_length("先排队。蛋糕三十八。") == 8
    assert editing.spoken_length("Don't be scared, I'm here.") == pytest.approx(5 * word)
    assert editing.spoken_length("蛋糕38") == pytest.approx(2 + word)
    assert editing.read_line(5.04, "Don't be scared, I'm here.")["end"] == round(0.3 + 0.28 * 5 * word, 2)


def test_a_line_nobody_speaks_is_read_and_names_its_speaker():
    assert editing.read_line(5.04, "先排队。蛋糕三十八。") == {"start": 0.3, "end": 2.54, "timing": "read"}
    assert editing.read_line(5.04, "走！")["end"] == 1.8  # long enough to read
    assert editing.read_line(1.0, "先排队。蛋糕三十八。")["end"] == 1.0  # within the clip
    shots = [{"id": "e1s1", "seconds": 5, "speaker": "linwan", "line": "Get in line."}]
    timing = {"start": 0.3, "end": 1.8}
    silent = editing.subtitles(shots, {"e1s1": {"silent": True, "speech": timing}}, names={"linwan": "Lin Wan"})
    assert "Lin Wan: Get in line." in silent
    assert "Lin Wan" not in editing.subtitles(shots, {"e1s1": {"speech": timing}}, names={"linwan": "Lin Wan"})


def test_lines_of_takes_without_sound_are_timed_without_speech_recognition(tmp_path, monkeypatch):
    from trtmc_short_drama.flow import Context
    from trtmc_short_drama.store import Store

    project = Store(tmp_path).create("t", "Chapter 1\nText", flow="default@1", episodes=1, max_seconds=60)
    shot = {"id": "e1s1", "seconds": 5, "location": "shop", "characters": ["linwan"], "props": [],
            "speaker": "linwan", "line": "Wait."}
    project.save_plan({"title": "t", "characters": [{"id": "linwan", "name": "Lin Wan"}], "locations": [], "props": [],
                       "episodes": [{"number": 1, "title": "One", "summary": "", "shots": [shot]}]})
    project.save_shot_state(1, {"e1s1": {"take": 1, "file": "episodes/01/clips/e1s1-take1.mp4", "silent": True}})
    monkeypatch.setattr(video, "probe", lambda path: {"duration": 5.04, "audio": True})

    class NoSpeechModel:
        def load(self, model):
            raise AssertionError("nothing to hear")

    editing.run_subtitles(Context(project, {}, "whisper", 1, NoSpeechModel(), progress=lambda _: None))
    assert project.shot_state(1)["e1s1"]["speech"] == {"start": 0.3, "end": 1.8, "timing": "read"}
