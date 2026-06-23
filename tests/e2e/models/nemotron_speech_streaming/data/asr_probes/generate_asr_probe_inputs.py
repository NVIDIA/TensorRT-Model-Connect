#!/usr/bin/env python3
"""Generate ASR probe WAV inputs for TensorRT Model Connect E2E coverage."""

from __future__ import annotations

import json
import shutil
import wave
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "Recording.wav"


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise ValueError(f"Only int16 WAV is supported, got sample_width={sample_width}")
    audio_i16 = np.frombuffer(frames, dtype="<i2")
    audio = audio_i16.reshape(-1, channels).astype(np.float32) / 32768.0
    return sample_rate, audio


def write_wav(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    audio_i16 = np.clip(audio, -1.0, 1.0)
    audio_i16 = np.round(audio_i16 * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(audio_i16.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_i16.tobytes())


def mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


def stereo_from_mono(audio: np.ndarray, *, right_gain: float = 1.0) -> np.ndarray:
    audio = mono(audio)
    return np.stack([audio, audio * right_gain], axis=1)


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio.copy()
    if audio.ndim == 1:
        audio_2d = audio[:, None]
    else:
        audio_2d = audio
    src_n = audio_2d.shape[0]
    dst_n = int(round(src_n * dst_rate / src_rate))
    src_x = np.arange(src_n, dtype=np.float64)
    dst_x = np.linspace(0, src_n - 1, dst_n, dtype=np.float64)
    channels = [
        np.interp(dst_x, src_x, audio_2d[:, c]).astype(np.float32)
        for c in range(audio_2d.shape[1])
    ]
    out = np.stack(channels, axis=1)
    return out[:, 0] if audio.ndim == 1 else out


def add_white_noise(audio: np.ndarray, snr_db: float, seed: int = 20260611) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal_power = float(np.mean(np.square(audio))) or 1e-12
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0)


def main() -> None:
    sample_rate, original = read_wav(SOURCE)
    mono_48k = mono(original)
    mono_16k = resample_linear(mono_48k, sample_rate, 16000)
    silence_48k = np.zeros((sample_rate,), dtype=np.float32)
    cases: list[dict[str, object]] = []

    def add_case(
        name: str,
        sample_rate_hz: int,
        audio: np.ndarray,
        *,
        purpose: str,
        model_connect_path: str,
        readiness: str,
        notes: str,
    ) -> None:
        file_name = f"{name}.wav"
        write_wav(ROOT / file_name, sample_rate_hz, audio)
        channels = 1 if audio.ndim == 1 else audio.shape[1]
        duration_s = round(float(audio.shape[0]) / sample_rate_hz, 3)
        cases.append(
            {
                "id": name,
                "file": file_name,
                "sample_rate_hz": sample_rate_hz,
                "channels": channels,
                "duration_s": duration_s,
                "purpose": purpose,
                "model_connect_path": model_connect_path,
                "readiness": readiness,
                "contract_oracle": (
                    "Compare TRT transcript with reference backend transcript; "
                    "do not judge model accuracy."
                ),
                "notes": notes,
            }
        )

    shutil.copyfile(SOURCE, ROOT / "probe_01_clean_48k_stereo_baseline.wav")
    cases.append(
        {
            "id": "probe_01_clean_48k_stereo_baseline",
            "file": "probe_01_clean_48k_stereo_baseline.wav",
            "sample_rate_hz": sample_rate,
            "channels": original.shape[1],
            "duration_s": round(float(original.shape[0]) / sample_rate, 3),
            "purpose": "Current smoke input copied as baseline.",
            "model_connect_path": "WAV decode + stereo input + model-specific resample.",
            "readiness": "ready_now",
            "contract_oracle": (
                "Compare TRT transcript with reference backend transcript; "
                "do not judge model accuracy."
            ),
            "notes": "Expected transcript is the existing short weather sentence.",
        }
    )

    add_case(
        "probe_02_clean_16k_mono_no_resample",
        16000,
        mono_16k,
        purpose="Cover native 16k mono path.",
        model_connect_path=(
            "WAV decode without model-side downmix; should avoid 48k->16k "
            "resample in ASR preprocessing."
        ),
        readiness="ready_now",
        notes="Useful to isolate resampling differences from core decode/runtime differences.",
    )
    add_case(
        "probe_03_clean_48k_mono_resample",
        48000,
        mono_48k,
        purpose="Cover mono 48k resample path.",
        model_connect_path="WAV decode + 48k->16k resample, without stereo downmix.",
        readiness="ready_now",
        notes="Separates resample handling from stereo downmix handling.",
    )
    add_case(
        "probe_04_clean_48k_stereo_gain_skew",
        48000,
        stereo_from_mono(mono_48k, right_gain=0.55),
        purpose="Cover stereo-to-mono with asymmetric channel levels.",
        model_connect_path="Stereo downmix should be robust when L/R channels are not identical.",
        readiness="ready_now",
        notes="Still the same speech content; intended to catch channel handling mistakes.",
    )
    add_case(
        "probe_05_low_volume_48k_stereo",
        48000,
        np.clip(stereo_from_mono(mono_48k) * 0.12, -1.0, 1.0),
        purpose="Cover low-amplitude PCM handling.",
        model_connect_path="PCM int16 decode + float normalization + ASR preprocessing.",
        readiness="ready_now",
        notes="Reference and TRT should see the same low-volume input.",
    )
    add_case(
        "probe_06_leading_trailing_silence_48k_stereo",
        48000,
        np.concatenate(
            [
                stereo_from_mono(silence_48k),
                stereo_from_mono(mono_48k),
                stereo_from_mono(silence_48k),
            ]
        ),
        purpose="Cover non-speech padding around valid speech.",
        model_connect_path="Longer waveform decode + silence handling + transcript stability.",
        readiness="ready_now",
        notes="Detects unexpected trimming, transcript drift, or EOS issues.",
    )
    add_case(
        "probe_08_noisy_48k_stereo_snr20",
        48000,
        add_white_noise(stereo_from_mono(mono_48k), snr_db=20.0),
        purpose="Cover deterministic noisy input handling.",
        model_connect_path="PCM decode + preprocessing under non-clean waveform.",
        readiness="nightly_optional",
        notes="Use only as TRT-vs-reference parity; do not evaluate model noise robustness.",
    )

    manifest = {
        "source_audio": f"tests/e2e/models/{ROOT.parent.parent.name}/data/Recording.wav",
        "source_properties": {
            "sample_rate_hz": sample_rate,
            "channels": int(original.shape[1]),
            "duration_s": round(float(original.shape[0]) / sample_rate, 3),
        },
        "principle": (
            "These probes test Model Connect input handling and TRT-vs-reference "
            "contract parity, not model accuracy."
        ),
        "recommended_metrics": [
            "CER for ASR-specific character-level transcript parity",
            "WER substitution/insertion/deletion breakdown for ASR failure triage",
            "no-speech/blank-audio state reporting for future ASR contract work",
        ],
        "cases": cases,
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
