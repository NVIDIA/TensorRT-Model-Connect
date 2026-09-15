# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native reference frontend contracts; ONNX Runtime is a test oracle only."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from families.cosyvoice3.frontend import CHECKPOINTS, FrontendProfile


@pytest.mark.parametrize("values", [(3, 500, 3000), (4, 3, 100), (4, 500, 3001), (True, 500, 3000), (4, 5.0, 6)])
def test_frontend_profile_rejects_invalid(values):
    with pytest.raises(ValueError):
        FrontendProfile(*values)


def test_frontend_profile_bounds():
    assert FrontendProfile(4, 4, 4).min_frames == 4
    assert FrontendProfile().max_frames == 3000


@pytest.mark.parametrize("seconds,sample_rate", [(0.05, 16000), (30.1, 16000), (1, 8000)])
def test_reference_audio_rejects_unsupported_input(tmp_path, seconds, sample_rate):
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("torch")
    pytest.importorskip("whisper")
    from families.cosyvoice3.tts import reference_features

    path = tmp_path / "audio.wav"
    sf.write(path, np.zeros(int(seconds * sample_rate), np.float32), sample_rate)
    with pytest.raises(ValueError, match="Reference audio requires"):
        reference_features(path)


def test_prepare_voice_refuses_existing_output(tmp_path):
    pytest.importorskip("torch")
    from types import SimpleNamespace
    from families.cosyvoice3.tts import prepare_voice

    with pytest.raises(FileExistsError):
        prepare_voice(SimpleNamespace(output=tmp_path))


@pytest.mark.parametrize("sample_rate", [16000, 24000, 44100])
def test_reference_mel_matches_pinned_matcha(tmp_path, sample_rate):
    source = os.environ.get("COSYVOICE3_OFFICIAL_SOURCE")
    if not source:
        pytest.skip("Pinned official Matcha source required for independent signal-processing comparison")
    import runpy
    import soundfile
    import torch
    import torchaudio
    from families.cosyvoice3.tts import reference_features

    path = Path(source) / "third_party/Matcha-TTS/matcha/utils/audio.py"
    reference = runpy.run_path(str(path))["mel_spectrogram"]
    samples = np.random.default_rng(2512).uniform(-0.1, 0.1, (sample_rate, 2)).astype(np.float32)
    audio = tmp_path / "stereo.wav"
    soundfile.write(audio, samples, sample_rate, subtype="FLOAT")
    actual = reference_features(audio)["mel"]
    waveform = torch.from_numpy(samples.T.copy()).mean(0, keepdim=True)
    if sample_rate != 24000:
        waveform = torchaudio.transforms.Resample(sample_rate, 24000)(waveform)
    expected = reference(waveform, n_fft=1920, num_mels=80, sampling_rate=24000,
                         hop_size=480, win_size=1920, fmin=0, fmax=None, center=False).transpose(1, 2)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.fixture(scope="module", params=["campplus", "speech_tokenizer"])
def frontend_pair(request):
    if os.environ.get("COSYVOICE3_RUN_GPU_TESTS") != "1":
        pytest.skip("Opt-in TensorRT frontend comparison")
    component = request.param
    directory = os.environ.get("COSYVOICE3_" + component.upper() + "_ENGINE")
    model = os.environ.get("COSYVOICE3_MODEL_DIR")
    if not directory or not model:
        pytest.skip("Full native frontend engine and pinned model required")
    import onnxruntime as ort
    import torch
    from families.cosyvoice3.frontend import FrontendEngine

    filename = CHECKPOINTS[component]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    oracle = ort.InferenceSession(str(Path(model) / filename), options, providers=["CPUExecutionProvider"])
    engine = FrontendEngine(directory, component)
    yield component, engine, oracle, torch


def test_frontend_rejects_invalid_layout_and_length(frontend_pair):
    component, engine, _, torch = frontend_pair
    shapes = [(2, 20, 80), (1, 3, 80), (1, 3001, 80), (1, 20, 79)] if component == "campplus" else [
        (2, 128, 20), (1, 128, 3), (1, 128, 3001), (1, 127, 20)]
    for shape in shapes:
        with pytest.raises(ValueError):
            engine.extract(torch.zeros(shape, device="cuda"))


def compare_frontend(pair, features, case):
    component, engine, oracle, torch = pair
    feed = {oracle.get_inputs()[0].name: features}
    if component == "speech_tokenizer":
        feed[oracle.get_inputs()[1].name] = np.array([features.shape[2]], np.int32)
    expected = oracle.run(None, feed)[0]
    actual = next(iter(engine.extract(torch.from_numpy(features).cuda()).values())).cpu().numpy()
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    atol, rtol = (0, 0) if component == "speech_tokenizer" else (1e-3, 1e-4)
    row = {"component": component, "case": case, "input_shape": list(features.shape),
           "engine_component": engine.manifest["component"], "shape_equal": actual.shape == expected.shape,
           "checkpoint": CHECKPOINTS[component], "atol": atol, "rtol": rtol,
           "passed": bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
           "max_abs": float(np.max(np.abs(actual.astype(np.float64) - expected))),
           "unequal_elements": int(np.count_nonzero(actual != expected))}
    if directory := os.environ.get("COSYVOICE3_FRONTEND_EVIDENCE"):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with (path / f"{component}-{case}.json").open("x", encoding="utf-8") as handle:
            json.dump(row, handle, indent=2)
        with (path / f"{component}-{case}.npz").open("xb") as handle:
            np.savez(handle, features=features, actual=actual, expected=expected)
    if component == "speech_tokenizer":
        assert actual.dtype == np.int32
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("frames", [4, 17, 199, 200, 201, 501, 3000])
def test_frontend_published_parity(frontend_pair, frames):
    component = frontend_pair[0]
    shape = (1, frames, 80) if component == "campplus" else (1, 128, frames)
    features = np.random.default_rng(2512 + frames).standard_normal(shape).astype(np.float32)
    if component == "campplus":
        features -= features.mean(axis=1, keepdims=True)
    compare_frontend(frontend_pair, features, f"random{frames}")


@pytest.mark.parametrize("seconds", [1, 3, None])
def test_frontend_real_audio_parity(frontend_pair, seconds):
    import soundfile
    import torch
    import torchaudio
    import whisper
    from torchaudio.compliance.kaldi import fbank

    model = Path(os.environ["COSYVOICE3_MODEL_DIR"])
    samples, sr = soundfile.read(model / "zero_shot_prompt.wav", dtype="float32", always_2d=True)
    if seconds is not None:
        samples = samples[:seconds * sr]
    audio = torch.from_numpy(samples.T.copy()).mean(0, keepdim=True)
    audio = torchaudio.functional.resample(audio, sr, 16000)
    if frontend_pair[0] == "campplus":
        features = fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
        features = (features - features.mean(0, keepdim=True))[None].numpy()
    else:
        features = whisper.log_mel_spectrogram(audio, n_mels=128).numpy()
    compare_frontend(frontend_pair, features, f"audio{seconds or 'full'}")


def test_frontend_second_recording_parity(frontend_pair):
    source = os.environ.get("COSYVOICE3_OFFICIAL_SOURCE")
    if not source:
        pytest.skip("Official second recording required")
    from families.cosyvoice3.tts import reference_features

    features = reference_features(Path(source) / "asset/cross_lingual_prompt.wav")
    key = "speaker_features" if frontend_pair[0] == "campplus" else "token_features"
    compare_frontend(frontend_pair, features[key].numpy(), "cross_lingual")


def test_native_voice_preparation(tmp_path, monkeypatch):
    names = ("COSYVOICE3_MODEL_DIR", "COSYVOICE3_CAMPPLUS_ENGINE", "COSYVOICE3_SPEECH_TOKENIZER_ENGINE")
    if os.environ.get("COSYVOICE3_RUN_GPU_TESTS") != "1" or not all(os.environ.get(n) for n in names):
        pytest.skip("Both native frontend engines required")
    import sys
    from types import SimpleNamespace
    import onnxruntime as ort
    import soundfile
    from families.cosyvoice3.tts import prepare_voice, read_voice, reference_features

    model, camp, tokenizer = (Path(os.environ[n]) for n in names)
    audio = tmp_path / "reference.wav"
    samples, sr = soundfile.read(model / "zero_shot_prompt.wav", dtype="float32")
    soundfile.write(audio, samples[:sr], sr, subtype="FLOAT")
    features = reference_features(audio)
    expected = {}
    for component, key in (("campplus", "speaker_features"), ("speech_tokenizer", "token_features")):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        oracle = ort.InferenceSession(str(model / CHECKPOINTS[component]), options, providers=["CPUExecutionProvider"])
        feed = {oracle.get_inputs()[0].name: features[key].numpy()}
        if component == "speech_tokenizer":
            feed[oracle.get_inputs()[1].name] = np.array([features[key].shape[2]], np.int32)
        expected[component] = oracle.run(None, feed)[0]
        del oracle
    # The production frontend must work without either ONNX package available.
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    monkeypatch.setitem(sys.modules, "onnx", None)
    output = tmp_path / "voice"
    prepare_voice(SimpleNamespace(audio=audio, campplus=camp, speech_tokenizer=tokenizer, output=output))
    actual = read_voice(output / "voice.npz")
    count = min(expected["speech_tokenizer"].shape[1], features["mel"].shape[1] // 2)
    np.testing.assert_array_equal(actual["prompt_tokens"], expected["speech_tokenizer"][:, :count])
    np.testing.assert_array_equal(actual["prompt_features"], features["mel"][:, :2 * count].numpy())
    np.testing.assert_allclose(actual["speaker"], expected["campplus"], atol=1e-3, rtol=1e-4)
    report = json.loads((output / "report.json").read_text())
    assert report["voice"] == "voice.npz"
    assert report["source_audio"] == str(audio)
    assert report["learned_execution"] == "native_tensorrt"
