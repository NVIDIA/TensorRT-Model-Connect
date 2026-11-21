# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The timed Native task decodes the audio it returns, once per frame, from unresampled input."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
import wave

import numpy as np
import pytest

from families.personaplex.reference.adapter import Adapter


def _write_wav(path, rate, samples):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(np.zeros(samples, dtype="<i2").tobytes())
    return str(path)


def _adapter(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("soundfile")

    class Codec:
        sample_rate = 24_000

        def __init__(self):
            self.decode_calls = 0

        def reset_streaming(self):
            pass

        def decode(self, tokens):
            self.decode_calls += 1
            value = float(tokens[0, 0, 0])
            return torch.tensor([[[value, value + 0.5]]])

    class Generator:
        def reset_streaming(self):
            pass

        def step(self, encoded):
            return encoded[:, :1].expand(1, 17, 1)

    # Exercise the real invoke loop without loading a checkpoint or using CUDA.
    lm = ModuleType("moshi.models.lm")
    lm._iterate_audio = lambda *args, **kwargs: iter(())
    lm.encode_from_sphn = lambda *args, **kwargs: iter(
        [torch.tensor([[[1, 2, 3]]], dtype=torch.int64)]
    )
    monkeypatch.setitem(sys.modules, "moshi.models.lm", lm)

    adapter = Adapter.__new__(Adapter)
    adapter.mimi = Codec()
    adapter.other_mimi = Codec()  # retained for the pinned official warmup
    adapter.generator = Generator()
    adapter.frame_size = 1920
    adapter.host = SimpleNamespace(
        Error=RuntimeError,
        required=lambda request, name: request[name],
        load_audio=lambda *args: np.zeros(5760, dtype=np.float32),
        timed=lambda run: (run(), 1.0),
        tensor_observation=lambda audio, base: {"audio": audio},
        invocation=lambda observation, ms: (observation, ms),
    )
    return adapter


def test_native_speech_does_not_decode_discarded_audio(monkeypatch, tmp_path):
    adapter = _adapter(monkeypatch)
    audio_path = _write_wav(tmp_path / "input.wav", 24_000, 5760)

    observation, _ = adapter.invoke(
        {"audio_path": audio_path, "max_new_tokens": 2}, tmp_path / "result"
    )

    np.testing.assert_array_equal(observation["audio"], [1.0, 1.5, 2.0, 2.5])
    assert observation["output_tokens"] == 2
    assert observation["sample_rate"] == 24_000
    assert adapter.mimi.decode_calls == 2
    assert adapter.other_mimi.decode_calls == 0


def test_native_speech_rejects_audio_it_would_have_to_resample(monkeypatch, tmp_path):
    # The suite resamples once for both sides; a 16 kHz file here would be resampled differently.
    adapter = _adapter(monkeypatch)
    audio_path = _write_wav(tmp_path / "input.wav", 16_000, 3840)

    with pytest.raises(RuntimeError, match="24000 Hz audio, not 16000 Hz"):
        adapter.invoke({"audio_path": audio_path, "max_new_tokens": 2}, tmp_path / "result")
    assert adapter.mimi.decode_calls == 0
