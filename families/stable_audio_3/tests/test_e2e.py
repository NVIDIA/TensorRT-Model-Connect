# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime and official-reference E2E for stable_audio_3.

Oracle: the official ``stable-audio-3`` pipeline (``official_reference.py``) run
with the same prompt, duration, steps and noise as the native runtime. The
native audio must be well formed and must agree with the official audio within
the case thresholds: waveform SNR and the correlation of multi-resolution
log-magnitude spectrograms. The SDK C and C++ consumers run on the same bundle.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tools.e2e_evidence import evidence_stage, record_evidence
from tensorrt_model_connect import BuildRequest, build

FAMILY = "stable_audio_3"
TASKS = frozenset({"text_to_audio"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"
SAMPLE_RATE = 44100
CHANNELS = 2
# Calibrated on 24 FP16 runs (4 prompts x 3 seeds x 10 / 30 s, 8 steps; medium-base weights sampled like the
# medium): native vs official 18.6-34.0 dB SNR and 0.984-0.998 correlation; the same runs with another seed or
# prompt reach at most 1.3 dB and 0.89; the quietest output has an RMS of 1.1e-3.
DEFAULT_THRESHOLDS = {
    "contract_min_rms": 2.5e-4,
    "contract_min_snr_db": 12.0,
    "contract_min_spectral_correlation": 0.97,
}


def _case_index() -> dict[str, tuple[Path, dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] in TASKS
        for case in manifest["testcases"]:
            name = str(case["name"])
            assert name not in result
            result[name] = (path, manifest, case)
    return result


CASES = _case_index()


def _selected_cases(config) -> tuple[list[str], bool]:
    model_filters = set()
    for raw in config.getoption("--e2e-model") or []:
        model_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        model_filters.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    testcase_filters = set()
    for raw in config.getoption("--e2e-testcase") or []:
        testcase_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    if not model_filters and not testcase_filters:
        return sorted(CASES), False
    selected = []
    for name, (_, manifest, _) in CASES.items():
        model_match = (not model_filters or FAMILY in model_filters or name in model_filters
                       or manifest["name"] in model_filters)
        if model_match and (not testcase_filters or name in testcase_filters):
            selected.append(name)
    return sorted(selected), True


def pytest_generate_tests(metafunc) -> None:
    if "case_name" in metafunc.fixturenames:
        names, enabled = _selected_cases(metafunc.config)
        parameters = names
        if not enabled:
            parameters = [
                pytest.param(name, marks=pytest.mark.skip(
                    reason="direct E2E requires one of the three explicit E2E selectors"))
                for name in names
            ]
        metafunc.parametrize("case_name", parameters, ids=names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected {FAMILY} E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get("TRTMC_STABLE_AUDIO_3_MODEL_DIR")
    if explicit:
        return _required_path(explicit, "TRTMC_STABLE_AUDIO_3_MODEL_DIR")
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(repo_id=manifest["hf_id"], revision=manifest.get("hf_revision"),
                                     local_files_only=True)
    except Exception as error:
        raise AssertionError(f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}") \
            from error
    return Path(snapshot)


def _runtime() -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    return binary, runtime_root


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(BuildRequest(model_dir=model_dir, output_path=bundle, family=FAMILY, task=manifest["task"],
                       precision=manifest["precision"],
                       tensor_parallel_size=int(manifest.get("tensor_parallel_size", 1))))


def _environment(runtime_root: Path) -> dict:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    return env


def _run_json(binary: Path, runtime_root: Path, bundle: Path, case: dict, *arguments: str) -> dict:
    completed = subprocess.run(
        [str(binary), "generate-audio", str(bundle), "--runtime-root", str(runtime_root), *arguments],
        check=True, capture_output=True, text=True, env=_environment(runtime_root),
        timeout=int(case.get("runtime_timeout_s", 1800)))
    record_evidence("commands", {"argv": getattr(completed, "args", None)})
    record_evidence("native", {"stdout": completed.stdout, "stderr": completed.stderr})
    payloads = []
    for line in completed.stdout.splitlines():
        start = line.find("{")
        if start >= 0:
            try:
                payloads.append(json.loads(line[start:]))
            except json.JSONDecodeError:
                pass
    assert payloads, f"native generate-audio returned no JSON: {completed.stdout[-1000:]}"
    return payloads[0]


def read_float_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read the CLI's IEEE-float WAV into [frames, channels]."""
    data = Path(path).read_bytes()
    assert data[:4] == b"RIFF" and data[8:16] == b"WAVEfmt ", "not a RIFF/WAVE file"
    audio_format, channels, sample_rate = np.frombuffer(data[20:28], dtype="<u2,<u2,<u4")[0]
    bits = int(np.frombuffer(data[34:36], dtype="<u2")[0])
    assert int(audio_format) == 3 and bits == 32, "expected 32-bit float PCM"
    assert data[36:40] == b"data"
    size = int(np.frombuffer(data[40:44], dtype="<u4")[0])
    samples = np.frombuffer(data[44:44 + size], dtype="<f4")
    return samples.reshape(-1, int(channels)).copy(), int(sample_rate)


def _thresholds(case_name: str) -> dict:
    thresholds = dict(DEFAULT_THRESHOLDS)
    path = THRESHOLD_ROOT / f"{case_name}.json"
    if path.is_file():
        thresholds.update(json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"])
    return thresholds


def snr_db(actual: np.ndarray, expected: np.ndarray) -> float:
    error = np.sum((actual.astype(np.float64) - expected.astype(np.float64)) ** 2)
    signal = np.sum(expected.astype(np.float64) ** 2)
    return float("inf") if error == 0 else float(10.0 * np.log10(signal / error))


def _log_spectrogram(signal: np.ndarray, size: int) -> np.ndarray:
    hop = size // 4
    window = np.hanning(size)
    frames = 1 + max(0, (signal.shape[0] - size) // hop)
    stacked = np.stack([signal[index * hop:index * hop + size] * window for index in range(frames)])
    return np.log10(np.abs(np.fft.rfft(stacked, axis=-1)) + 1e-5)


def spectral_correlation(actual: np.ndarray, expected: np.ndarray) -> float:
    """Smallest Pearson correlation of log-magnitude STFTs over resolutions and channels."""
    scores = []
    for channel in range(expected.shape[1]):
        for size in (512, 2048, 8192):
            a = _log_spectrogram(actual[:, channel].astype(np.float64), size).ravel()
            e = _log_spectrogram(expected[:, channel].astype(np.float64), size).ravel()
            scores.append(float(np.corrcoef(a, e)[0, 1]))
    return min(scores)


def _assert_audio_health(samples: np.ndarray, sample_rate: int, case: dict, thresholds: dict) -> None:
    assert sample_rate == SAMPLE_RATE
    assert samples.ndim == 2 and samples.shape[1] == CHANNELS
    assert samples.shape[0] == int(float(case["duration"]) * SAMPLE_RATE)
    assert np.isfinite(samples).all()
    assert float(np.abs(samples).max()) <= 1.0
    assert float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) >= float(thresholds["contract_min_rms"])


def _assert_contract(actual: dict, expected: dict, case: dict, thresholds: dict) -> None:
    samples, sample_rate = read_float_wav(Path(actual["audio"]))
    _assert_audio_health(samples, sample_rate, case, thresholds)
    reference = expected["samples"]
    assert reference.shape == samples.shape
    metrics = {"snr_db": snr_db(samples, reference), "spectral_correlation": spectral_correlation(samples, reference)}
    record_evidence("metrics", metrics)
    assert metrics["snr_db"] >= float(thresholds["contract_min_snr_db"]), metrics
    assert metrics["spectral_correlation"] >= float(thresholds["contract_min_spectral_correlation"]), metrics


def _native(binary: Path, runtime_root: Path, bundle: Path, case: dict, tmp_path: Path) -> dict:
    output = tmp_path / "native.wav"
    payload = _run_json(
        binary, runtime_root, bundle, case,
        "--prompt", str(case["prompt"]), "--set", f"duration={float(case['duration'])!r}",
        "--set", f"steps={int(case['steps'])}", "--seed", str(int(case["seed"])), "--output", str(output))
    assert output.is_file()
    assert payload["sample_rate"] == SAMPLE_RATE and payload["channels"] == CHANNELS
    payload["audio"] = str(output)
    record_evidence("native_artifacts", {"files": [output]})
    return payload


def _official_reference(model_dir: Path, manifest: dict, case: dict) -> dict:
    import torch

    from families.stable_audio_3.tests import official_reference

    official = official_reference.load_official(model_dir, half=manifest["precision"] == "fp16")
    try:
        samples = official_reference.generate(official, str(case["prompt"]), duration=float(case["duration"]),
                                              steps=int(case["steps"]), seed=int(case["seed"]))
    finally:
        del official
        torch.cuda.empty_cache()
    return {"samples": samples, "sample_rate": SAMPLE_RATE}


def _assert_sdk_consumers(runtime_root: Path, bundle: Path) -> None:
    native_build = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    for language in ("c", "cpp"):
        consumer = native_build / f"test_{FAMILY}_sdk_{language}"
        assert consumer.is_file(), f"build the family-owned SDK consumer: {consumer.name}"
        completed = subprocess.run([str(consumer), str(bundle), str(runtime_root)], check=True,
                                   capture_output=True, text=True, env=_environment(runtime_root),
                                   timeout=900)
        record_evidence(f"sdk_{language}", {"stdout": completed.stdout})
        assert "all tasks passed" in completed.stdout


def test_contract_metrics_are_strict() -> None:
    time = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    tone = 0.2 * np.sin(2.0 * np.pi * 440.0 * time)
    stereo = np.stack([tone, 0.5 * tone], axis=1).astype(np.float32)
    assert snr_db(stereo, stereo) == float("inf")
    assert spectral_correlation(stereo, stereo) > 0.999999
    noise = np.random.default_rng(0).standard_normal(stereo.shape).astype(np.float32) * 0.2
    assert snr_db(noise, stereo) < 1.0
    assert spectral_correlation(noise, stereo) < DEFAULT_THRESHOLDS["contract_min_spectral_correlation"]
    with pytest.raises(AssertionError):
        _assert_audio_health(stereo[:, :1], SAMPLE_RATE, {"duration": 1.0}, DEFAULT_THRESHOLDS)
    with pytest.raises(AssertionError):
        _assert_audio_health(stereo, SAMPLE_RATE, {"duration": 2.0}, DEFAULT_THRESHOLDS)
    _assert_audio_health(stereo, SAMPLE_RATE, {"duration": 1.0}, DEFAULT_THRESHOLDS)


def test_float_wav_reader_round_trips(tmp_path: Path) -> None:
    samples = np.linspace(-1.0, 1.0, 20, dtype=np.float32).reshape(10, 2)
    data = samples.astype("<f4").tobytes()
    header = (b"RIFF" + (36 + len(data)).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little")
              + (3).to_bytes(2, "little") + (2).to_bytes(2, "little") + SAMPLE_RATE.to_bytes(4, "little")
              + (SAMPLE_RATE * 8).to_bytes(4, "little") + (8).to_bytes(2, "little") + (32).to_bytes(2, "little")
              + b"data" + len(data).to_bytes(4, "little"))
    path = tmp_path / "clip.wav"
    path.write_bytes(header + data)
    restored, rate = read_float_wav(path)
    assert rate == SAMPLE_RATE
    np.testing.assert_array_equal(restored, samples)


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    model_dir = _model_dir(manifest)
    record_evidence("checkpoint", {"model_dir": str(model_dir), "hf_id": manifest.get("hf_id"),
                                   "hf_revision": manifest.get("hf_revision")})
    binary, runtime_root = _runtime()
    bundle = tmp_path.parent / manifest["bundle"]
    if not bundle.is_file():
        with evidence_stage("build"):
            _build(model_dir, bundle, manifest)
    with evidence_stage("native"):
        actual = _native(binary, runtime_root, bundle, case, tmp_path)
    record_evidence("native", actual)
    with evidence_stage("sdk"):
        _assert_sdk_consumers(runtime_root, bundle)
    with evidence_stage("reference"):
        expected = _official_reference(model_dir, manifest, case)
    with evidence_stage("compare"):
        _assert_contract(actual, expected, case, record_evidence("thresholds", _thresholds(case_name)))
