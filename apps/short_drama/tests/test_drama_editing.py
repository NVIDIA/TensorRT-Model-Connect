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
    assert editing.coverage(" 王总 这是新的报告", "王总，这是新的报告。") == 1.0
    assert editing.coverage("王总", "王总这是新的报告") == 0.25
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
    timing = editing.time_line(lambda _: "完全不同", silent_wav(4.0), LINE)
    assert timing["timing"] == "estimated" and timing["start"] < timing["end"] <= 4.0


def test_subtitles_follow_the_shots_on_the_timeline():
    shots = [{"id": "a", "seconds": 5, "line": "第一句"}, {"id": "b", "seconds": 6, "line": ""},
             {"id": "c", "seconds": 4, "line": "“第三句”"}]
    state = {"a": {"speech": {"start": 0.4, "end": 1.9}}, "c": {"speech": {"start": 0.2, "end": 9.0}}}
    srt = editing.subtitles(shots, state)
    assert srt.splitlines()[:3] == ["1", "00:00:00,400 --> 00:00:01,900", "第一句"]
    assert "00:00:11,200 --> 00:00:15,000\n第三句" in srt


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
    (tmp_path / "episode.srt").write_text("1\n00:00:00,200 --> 00:00:01,500\n测试\n", encoding="utf-8")
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
