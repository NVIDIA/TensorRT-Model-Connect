# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from families.personaplex.system_prompt import ForcedPromptCache, SILENCE, SINE, load, wrap_system_text


def test_system_tags_follow_the_official_wrapper():
    assert wrap_system_text(" Teacher. ") == "<system> Teacher. <system>"
    assert wrap_system_text("<system> Teacher. <system>") == "<system> Teacher. <system>"


def test_delayed_streams_read_the_preceding_frame():
    state = ForcedPromptCache(32000)
    assert state.step(SINE, SILENCE, 3) is None
    first = state.step(SINE, SILENCE, 42)
    np.testing.assert_array_equal(first, [32000] + [2048] * 16)
    second = state.step(SINE, SILENCE, 43)
    assert second[0] == 42
    assert second[1] == SILENCE[0]
    assert second[2] == 2048  # one-step-delayed agent stream
    third = state.step(SINE, SILENCE, 44)
    np.testing.assert_array_equal(third[1:9], SILENCE)
    np.testing.assert_array_equal(third[9:], SINE)
    assert third[0] == 43


@pytest.mark.parametrize("precision,capacity", [("fp32", 512), ("fp16", 3)])
def test_prefix_rejects_another_dtype_or_insufficient_context(tmp_path, precision, capacity):
    p = tmp_path / "prompt.npz"
    np.savez(p, embeddings=np.zeros((2, 4), dtype=np.float32), cache=np.zeros((17, 4), dtype=np.int32),
             provided=np.zeros((17, 4), dtype=np.uint8),
             metadata=np.array(json.dumps({"version": 1, "precision": "fp16", "frames": 2, "offset": 3})))
    with pytest.raises(ValueError, match="does not match"):
        load(p, precision, 4, capacity)


@pytest.mark.parametrize("listen", [False, True])
def test_prompt_and_full_reply_window_are_inside_the_native_task(monkeypatch, tmp_path, listen):
    torch = pytest.importorskip("torch")
    soundfile = pytest.importorskip("soundfile")
    from families.personaplex.reference.adapter import Adapter

    events = []
    steps = []

    class Codec:
        sample_rate = 24_000

        def reset_streaming(self):
            events.append("codec-reset")

        def decode(self, codes):
            return torch.zeros((1, 1, 1920))

    class Generator:
        zero_text_code = 3

        def _encode_zero_frame(self):
            return torch.tensor(SILENCE).reshape(1, 8, 1)

        def reset_streaming(self):
            events.append("generator-reset")

        def step_system_prompts(self, codec):
            events.append("prompt")

        def step(self, codes, **forced):
            steps.append(forced)
            return torch.zeros((1, 17, 1), dtype=torch.int64)

    def iterate(audio, **kwargs):
        assert audio.shape == (1, 3 * 1920)
        assert np.count_nonzero(audio[:, 1920:]) == 0
        return range(3)

    lm = ModuleType("moshi.models.lm")
    lm._iterate_audio = iterate
    lm.encode_from_sphn = lambda codec, frames, **kwargs: (torch.zeros((1, 8, 1)) for _ in frames)
    monkeypatch.setitem(sys.modules, "moshi.models.lm", lm)

    def timed(call):
        events.append("timing-start")
        result = call()
        events.append("timing-end")
        return result, 1.0

    adapter = Adapter.__new__(Adapter)
    adapter.frame_size = 1920
    adapter.mimi, adapter.other_mimi, adapter.generator = Codec(), Codec(), Generator()
    adapter.system_prompt = {"voice": "NATM1.pt", "frames": 108}
    adapter.listen_before_answer = listen
    adapter.host = SimpleNamespace(required=lambda r, k: r[k], load_audio=lambda *a: np.ones(1920),
                                   timed=timed, tensor_observation=lambda audio, base: {"samples": audio.size},
                                   invocation=lambda observation, ms: observation)
    question = tmp_path / "question.wav"  # the reference only checks its rate; load_audio is stubbed
    soundfile.write(question, np.zeros(1920, dtype=np.float32), 24_000)
    result = adapter.invoke({"audio_path": str(question), "tail_frames": 2, "max_new_tokens": 10}, tmp_path / "out")
    assert result["output_tokens"] == 3 and result["samples"] == 3 * 1920
    assert result["system_prompt_voice"] == "NATM1.pt"
    assert bool(steps[0]) is listen
    assert steps[1:] == ([{"text_token": 0}, {}] if listen else [{}, {}])
    if listen:
        assert steps[0]["text_token"] == 3
        np.testing.assert_array_equal(steps[0]["moshi_tokens"].numpy().reshape(-1), SILENCE)
    assert events == ["timing-start", "codec-reset", "codec-reset", "generator-reset", "prompt",
                      "codec-reset", "timing-end"]
