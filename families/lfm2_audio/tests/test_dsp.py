# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU frontend and overlap-add comparisons with the original audio processor."""

import json
import cProfile
from contextlib import nullcontext
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
TESTS = Path(__file__).parent


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("CPU DSP test requires a C++17 compiler")
    output = tmp_path_factory.mktemp("lfm2-audio-dsp") / "probe"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O3",
            "-I",
            str(ROOT),
            str(TESTS / "cpp/dsp_probe.cpp"),
            str(TESTS.parent / "runtime/audio_dsp.cpp"),
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


@pytest.fixture(scope="module")
def reference():
    torch = pytest.importorskip("torch")
    pytest.importorskip("liquid_audio")
    from liquid_audio.detokenizer import ISTFT
    from liquid_audio.model.conformer.processor import AudioToMelSpectrogramPreprocessor

    raw = json.loads((TESTS / "fixtures/config.json").read_text())
    proc = AudioToMelSpectrogramPreprocessor(**raw["preprocessor"]).eval()
    return torch, proc, ISTFT(1280, 320, 1280)


def invoke(probe, tmp_path, mode, source, rate=16000, filters=None):
    input_path, output_path = tmp_path / "input.f32", tmp_path / "output.f32"
    filter_path = tmp_path / "filters.f32"
    np.asarray(source, dtype=np.float32).tofile(input_path)
    if filters is not None:
        np.asarray(filters, dtype=np.float32).tofile(filter_path)
    result = subprocess.run(
        [str(probe), mode, str(input_path), str(output_path), str(rate), str(filter_path)],
        capture_output=True,
        text=True,
    )
    return result, np.fromfile(output_path, np.float32) if output_path.exists() else None


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 24000, 44100, 48000])
def test_resample_reference(probe, reference, tmp_path, rate):
    torch, _, _ = reference
    import torchaudio

    source = (np.random.default_rng(42).standard_normal(rate // 2 + 33) * 0.1).astype(np.float32)
    result, actual = invoke(probe, tmp_path, "resample", source, rate)
    assert result.returncode == 0, result.stderr
    expected = torchaudio.functional.resample(torch.tensor(source), rate, 16000)
    # Standard torch.float32 comparison budget, not a waveform acceptance gate.
    torch.testing.assert_close(torch.tensor(actual), expected)


@pytest.mark.parametrize(
    "name",
    [
        "noise",
        "silence_min",
        "silence_1s",
        "silence_2s",
        "silence_256_frames",
        "silence_1024_frames",
        "silence_4096_frames",
        "impulse",
        "tone",
        "dc",
    ],
)
def test_mel_reference(probe, reference, tmp_path, name):
    torch, proc, _ = reference
    count = {
        "silence_min": 1280,
        "silence_2s": 32000,
        "silence_256_frames": 160 * 256,
        "silence_1024_frames": 160 * 1024,
        "silence_4096_frames": 160 * 4096,
    }.get(name, 16000)
    source = np.zeros(count, dtype=np.float32)
    if name == "noise":
        source = (0.1 * np.random.default_rng(42).standard_normal(count)).astype(np.float32)
    elif name == "impulse":
        source[0] = 1
    elif name == "tone":
        source = (0.1 * np.sin(2 * np.pi * 440 * np.arange(count) / 16000)).astype(np.float32)
    elif name == "dc":
        source.fill(0.1)
    filters = proc.featurizer.filter_banks.numpy()
    result, actual = invoke(probe, tmp_path, "mel", source, filters=filters)
    assert result.returncode == 0, result.stderr
    observation = None
    if name.startswith("silence"):
        from liquid_audio.model.conformer.processor import normalize_batch

        from .dsp_observation import NormalizationObservation

        observation = NormalizationObservation(torch, normalize_batch)
    with observation if observation is not None else nullcontext():
        expected, length = proc(torch.tensor(source)[None], torch.tensor([count]))
    expected = expected[0].T.contiguous()
    actual = torch.tensor(actual.reshape(expected.shape))
    try:
        assert int(result.stdout) == count // 160 + 1
        assert int(length[0]) == count // 160
        assert torch.count_nonzero(actual[-1]) == 0
        assert torch.isfinite(actual).all()
        if name.startswith("silence"):
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        else:
            # Normalized frontend error stays below one eighth of the BF16 ULP at
            # unit scale. Separate strict waveform/token parity remains mandatory.
            torch.testing.assert_close(actual, expected, rtol=0, atol=2**-10)
    except AssertionError:
        if observation is not None:
            try:
                config = json.loads((TESTS / "fixtures/config.json").read_text())["preprocessor"]
                observation.report_failure(
                    actual, expected, sample_count=count, processor_config=config
                )
            except Exception:
                pass  # Never replace the unchanged assertion with an observation failure.
        raise


def test_normalization_observation_preserves_processor_call_and_profiler(reference, capsys):
    from liquid_audio.model.conformer.processor import normalize_batch

    from .dsp_observation import NormalizationObservation

    previous = sys.getprofile()
    if previous is not None and not callable(previous):
        pytest.skip("The capture control leaves an active non-callable profiler untouched")
    torch, proc, _ = reference
    source, count = torch.zeros((1, 1280)), torch.tensor([1280])
    expected = proc(source, count)
    rng = torch.get_rng_state().clone()
    prior_calls = []

    def prior(frame, event, value):
        if previous is not None:
            previous(frame, event, value)
        if frame.f_code is normalize_batch.__code__ and event == "call":
            prior_calls.append(event)

    sys.setprofile(prior)
    try:
        with NormalizationObservation(torch, normalize_batch) as observation:
            actual = proc(source, count)
        assert sys.getprofile() is prior
    finally:
        sys.setprofile(previous)
    assert prior_calls == ["call"]
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert torch.equal(torch.get_rng_state(), rng)
    assert observation.data["normalization_calls"] == observation.data["normalization_returns"] == 1
    assert observation.data["errors"] == []
    assert (
        "def normalize_batch("
        in observation.data["reference_source_context"]["inspected_source_text"]
    )
    assert observation.data["normalization_input"]["all_channels_bit_identical"]
    where = [
        item for item in observation.data["operations"] if item["operator"] == "aten.where.self"
    ]
    assert len(where) == 2
    assert where[0]["output"]["first_channel_rle"][-1] == {"start": 8, "length": 1, "bits": 0}
    assert observation.data["variance_before_sqrt"]["shape"] == [1, 128]
    assert capsys.readouterr().out == ""


def test_normalization_observation_preserves_failures_and_bounds(reference, monkeypatch, capsys):
    from liquid_audio.model.conformer.processor import normalize_batch

    from .dsp_observation import NormalizationObservation, tensor_observation

    torch, _, _ = reference
    previous = sys.getprofile()
    if previous is not None and not callable(previous):
        pytest.skip("The capture control leaves an active non-callable profiler untouched")
    observation = NormalizationObservation(torch, normalize_batch)
    with pytest.raises(AssertionError, match="original assertion"):
        with observation:
            normalize_batch(torch.zeros((1, 128, 9)), torch.tensor([8]), "per_feature")
            raise AssertionError("original assertion")
    assert sys.getprofile() is previous
    assert observation.data["normalization_calls"] == 1
    varied = torch.arange(512, dtype=torch.float32).reshape(1, 2, 256)
    assert tensor_observation(varied)["payload"].startswith("bounded_unavailable")
    monkeypatch.setattr(
        observation, "_snapshot", lambda _: (_ for _ in ()).throw(ValueError("observe"))
    )
    with pytest.raises(AssertionError, match="original assertion"):
        try:
            raise AssertionError("original assertion")
        except AssertionError:
            observation.report_failure(varied, varied, sample_count=1280, processor_config={})
            raise
    output = capsys.readouterr().out
    assert output.count("LFM2_AUDIO_NORMALIZATION_DIAGNOSTIC=") == 1
    assert "ValueError" in output


def test_normalization_observation_preserves_active_cprofile(reference):
    from liquid_audio.model.conformer.processor import normalize_batch

    from .dsp_observation import NormalizationObservation

    if sys.getprofile() is not None:
        pytest.skip("The cProfile control requires ownership of its profiling scope")
    torch, proc, _ = reference
    source, count = torch.zeros((1, 1280)), torch.tensor([1280])
    expected = proc(source, count)
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        active = sys.getprofile()
        # Python may expose cProfile here or use its separate monitoring API.
        assert active is None or (active is profiler and not callable(active))
        with NormalizationObservation(torch, normalize_batch) as observation:
            actual = proc(source, count)
            if active is not None:
                assert sys.getprofile() is active
        assert sys.getprofile() is active
    finally:
        profiler.disable()
    if active is not None:
        assert observation.data["observation_unavailable"] == "active profiler is not callable"
        assert observation.data["normalization_calls"] == 0
    else:
        assert observation.data["normalization_calls"] == 1
    calls = [item for item in profiler.getstats() if item.code is normalize_batch.__code__]
    assert len(calls) == 1 and calls[0].callcount == 1
    assert not observation.profile_installed and not observation.mode_installed
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


@pytest.mark.parametrize("frames", [1, 2, 6, 12, 36])
def test_istft_reference(probe, reference, tmp_path, frames):
    torch, _, istft = reference
    rng = np.random.default_rng(42)
    logabs = torch.tensor(rng.normal(-2, 0.7, (1, 641, frames)), dtype=torch.float32)
    phase = torch.tensor(rng.uniform(-np.pi, np.pi, (1, 641, frames)), dtype=torch.float32)
    expected = istft(torch.polar(logabs.exp(), phase))[0]
    source = torch.cat([logabs, phase], dim=1)[0].T.numpy()
    result, actual = invoke(probe, tmp_path, "istft", source)
    assert result.returncode == 0, result.stderr
    assert actual.size == frames * 320
    # Use FP32 default relative tolerance with an absolute FFT roundoff budget
    # well below a 24-bit PCM quantization step.
    torch.testing.assert_close(torch.tensor(actual), expected, rtol=1.3e-6, atol=1e-8)


@pytest.mark.parametrize(
    "mode,source,rate,filters",
    [
        ("resample", [], 16000, None),
        ("resample", [0], 0, None),
        ("resample", [float("nan")], 16000, None),
        ("resample", [float("inf")], 16000, None),
        ("resample", [0], 384001, None),
        ("mel", [0] * 1280, 16000, [0]),
        ("mel", [0] * 1279, 16000, [0] * (128 * 257)),
        ("istft", [0], 16000, None),
    ],
)
def test_invalid_dsp_input(probe, tmp_path, mode, source, rate, filters):
    result, _ = invoke(probe, tmp_path, mode, source, rate, filters)
    assert result.returncode != 0
    assert result.stderr.strip()


@pytest.mark.parametrize("frames", [1, 2, 6, 37])
@pytest.mark.parametrize("chunk", [1, 6, 12])
def test_streaming_istft_is_exact_offline(probe, tmp_path, frames, chunk):
    rng = np.random.default_rng(71)
    spectra = np.concatenate(
        [rng.normal(-2, 0.7, (frames, 641)), rng.uniform(-np.pi, np.pi, (frames, 641))], axis=1
    ).astype(np.float32)
    result, offline = invoke(probe, tmp_path, "istft", spectra)
    assert result.returncode == 0, result.stderr
    result, streamed = invoke(probe, tmp_path, "istft_stream", spectra, rate=chunk)
    assert result.returncode == 0, result.stderr
    assert streamed.size == frames * 320
    np.testing.assert_array_equal(streamed, offline)
