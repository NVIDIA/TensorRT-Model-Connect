# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stereo WAV evidence and waveform-parity gates for MiniMax-H3."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import struct
from typing import Mapping

import numpy as np


EXPECTED_AUDIO_BATCHES = 1
EXPECTED_AUDIO_CHANNELS = 2
EXPECTED_AUDIO_NUM_SAMPLES = 165_600
EXPECTED_AUDIO_SAMPLE_RATE = 32_000
EXPECTED_AUDIO_DURATION_S = EXPECTED_AUDIO_NUM_SAMPLES / EXPECTED_AUDIO_SAMPLE_RATE

REQUIRED_AUDIO_THRESHOLDS = frozenset(
    {
        "exact_audio_channels",
        "exact_audio_num_samples",
        "exact_audio_sample_rate_hz",
        "exact_audio_duration_s",
        "minimum_audio_waveform_correlation",
        "maximum_audio_normalized_rmse",
        "minimum_audio_si_sdr_db",
        "minimum_reference_audio_rms",
        "minimum_candidate_audio_rms_ratio",
        "maximum_candidate_audio_rms_ratio",
        "maximum_audio_peak_absolute",
    }
)


@dataclass(frozen=True)
class Float32Wav:
    """A decoded IEEE-float32 WAV with channel-major samples."""

    samples: np.ndarray
    sample_rate: int
    encoding: str = "ieee_float32le"


@dataclass(frozen=True)
class DecodedAudioMetrics:
    """Direct waveform comparison plus independent artifact invariants."""

    reference_shape: tuple[int, ...]
    candidate_shape: tuple[int, ...]
    reference_sample_rate: int
    candidate_sample_rate: int
    reference_duration_s: float
    candidate_duration_s: float
    reference_all_finite: bool
    candidate_all_finite: bool
    reference_peak_absolute: float
    candidate_peak_absolute: float
    reference_rms: float
    candidate_rms: float
    candidate_rms_ratio: float
    waveform_correlation_minimum: float
    waveform_correlation_mean: float
    normalized_rmse_maximum: float
    si_sdr_db_minimum: float
    maximum_absolute_error: float


@dataclass(frozen=True)
class AudioGateResult:
    """One self-describing audio gate result."""

    value: float
    threshold: float | None
    operator: str
    passed: bool
    note: str = ""

    def to_dict(self) -> dict[str, float | str | bool | None]:
        return {
            "value": self.value,
            "threshold": self.threshold,
            "operator": self.operator,
            "passed": self.passed,
            "note": self.note,
        }


def canonical_hf_audio(audio: object, sample_rate: object) -> np.ndarray:
    """Validate the official batched output and return `[channels, samples]` float32."""

    if isinstance(sample_rate, np.ndarray):
        if sample_rate.size != 1:
            raise ValueError("MiniMax-H3 HF sampling_rate must be a scalar")
        sample_rate = sample_rate.item()
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise ValueError("MiniMax-H3 HF sampling_rate must be an integer")
    if int(sample_rate) != EXPECTED_AUDIO_SAMPLE_RATE:
        raise ValueError(
            "MiniMax-H3 HF returned audio at "
            f"{sample_rate} Hz instead of {EXPECTED_AUDIO_SAMPLE_RATE} Hz"
        )

    array = np.asarray(audio)
    expected_shape = (
        EXPECTED_AUDIO_BATCHES,
        EXPECTED_AUDIO_CHANNELS,
        EXPECTED_AUDIO_NUM_SAMPLES,
    )
    if array.shape != expected_shape:
        raise ValueError(f"MiniMax-H3 HF audio must have shape {expected_shape}, got {array.shape}")
    if array.dtype.kind not in "fiu":
        raise ValueError(f"MiniMax-H3 HF audio dtype is not float32-compatible: {array.dtype}")
    canonical = np.ascontiguousarray(array[0], dtype=np.float32)
    if not np.isfinite(canonical).all():
        raise ValueError("MiniMax-H3 HF audio contains non-finite samples")
    if float(np.max(np.abs(canonical))) > 1.0:
        raise ValueError("MiniMax-H3 HF audio contains samples outside [-1, 1]")
    return canonical


def write_float32_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write deterministic stereo IEEE-float32 WAV without quantization or downmixing."""

    audio = np.asarray(samples)
    if audio.ndim != 2 or audio.shape[0] != EXPECTED_AUDIO_CHANNELS:
        raise ValueError(
            f"MiniMax-H3 WAV samples must be channel-major stereo [2, samples], got {audio.shape}"
        )
    if audio.dtype.kind not in "fiu":
        raise ValueError(f"MiniMax-H3 WAV dtype is not float32-compatible: {audio.dtype}")
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if not np.isfinite(audio).all():
        raise ValueError("MiniMax-H3 WAV contains non-finite samples")
    if float(np.max(np.abs(audio))) > 1.0:
        raise ValueError("MiniMax-H3 WAV contains samples outside [-1, 1]")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise ValueError("MiniMax-H3 WAV sample rate must be an integer")
    if int(sample_rate) <= 0:
        raise ValueError("MiniMax-H3 WAV sample rate must be positive")

    interleaved = np.ascontiguousarray(audio.T, dtype="<f4")
    data = interleaved.tobytes(order="C")
    channels = int(audio.shape[0])
    sample_width = 4
    block_align = channels * sample_width
    byte_rate = int(sample_rate) * block_align
    fmt = struct.pack(
        "<HHIIHH",
        3,  # WAVE_FORMAT_IEEE_FLOAT
        channels,
        int(sample_rate),
        byte_rate,
        block_align,
        32,
    )
    riff_size = 4 + (8 + len(fmt)) + (8 + len(data))
    with path.open("wb") as output:
        output.write(b"RIFF")
        output.write(struct.pack("<I", riff_size))
        output.write(b"WAVEfmt ")
        output.write(struct.pack("<I", len(fmt)))
        output.write(fmt)
        output.write(b"data")
        output.write(struct.pack("<I", len(data)))
        output.write(data)


def read_float32_wav(path: Path) -> Float32Wav:
    """Read a stereo IEEE-float32 WAV while preserving left and right channels."""

    payload = path.read_bytes()
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise ValueError(f"MiniMax-H3 audio is not a RIFF/WAVE file: {path}")

    fmt: bytes | None = None
    audio: bytes | None = None
    offset = 12
    while offset + 8 <= len(payload):
        chunk_name = payload[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", payload, offset + 4)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(payload):
            raise ValueError(f"MiniMax-H3 WAV contains a truncated {chunk_name!r} chunk")
        if chunk_name == b"fmt ":
            fmt = payload[chunk_start:chunk_end]
        elif chunk_name == b"data":
            audio = payload[chunk_start:chunk_end]
        offset = chunk_end + (chunk_size & 1)

    if fmt is None or len(fmt) < 16 or audio is None:
        raise ValueError("MiniMax-H3 WAV is missing its fmt or data chunk")
    audio_format, channels, sample_rate, byte_rate, block_align, bits_per_sample = (
        struct.unpack_from("<HHIIHH", fmt)
    )
    if audio_format != 3 or bits_per_sample != 32:
        raise ValueError("MiniMax-H3 WAV must use IEEE-float32 encoding")
    if channels != EXPECTED_AUDIO_CHANNELS:
        raise ValueError(
            f"MiniMax-H3 WAV must be stereo, got {channels} channel(s); downmixing is forbidden"
        )
    if sample_rate <= 0:
        raise ValueError("MiniMax-H3 WAV sample rate must be positive")
    expected_block_align = channels * 4
    if block_align != expected_block_align or byte_rate != sample_rate * expected_block_align:
        raise ValueError("MiniMax-H3 WAV has inconsistent byte-rate or block alignment")
    if not audio or len(audio) % block_align:
        raise ValueError("MiniMax-H3 WAV data is empty or not sample-frame aligned")

    interleaved = np.frombuffer(audio, dtype="<f4").reshape(-1, channels)
    samples = np.ascontiguousarray(interleaved.T, dtype=np.float32)
    return Float32Wav(samples=samples, sample_rate=int(sample_rate))


def validate_fixed_audio(
    samples: np.ndarray,
    sample_rate: int,
    *,
    label: str,
) -> np.ndarray:
    """Require the fixed decoded T2VA audio surface and return stable float32."""

    audio = np.asarray(samples)
    expected_shape = (EXPECTED_AUDIO_CHANNELS, EXPECTED_AUDIO_NUM_SAMPLES)
    if audio.shape != expected_shape:
        raise ValueError(
            f"MiniMax-H3 {label} audio must have shape {expected_shape}, got {audio.shape}"
        )
    if audio.dtype.kind not in "fiu":
        raise ValueError(f"MiniMax-H3 {label} audio dtype is not float32-compatible: {audio.dtype}")
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if not np.isfinite(audio).all():
        raise ValueError(f"MiniMax-H3 {label} audio contains non-finite samples")
    if float(np.max(np.abs(audio))) > 1.0:
        raise ValueError(f"MiniMax-H3 {label} audio contains samples outside [-1, 1]")
    if int(sample_rate) != EXPECTED_AUDIO_SAMPLE_RATE:
        raise ValueError(
            f"MiniMax-H3 {label} audio rate is {sample_rate} Hz instead of "
            f"{EXPECTED_AUDIO_SAMPLE_RATE} Hz"
        )
    return audio


def audio_summary(
    samples: np.ndarray, sample_rate: int
) -> dict[str, float | int | bool | list[int]]:
    """Return receipt-friendly invariant evidence for a channel-major waveform."""

    audio = np.asarray(samples, dtype=np.float32)
    return {
        "shape": [int(value) for value in audio.shape],
        "channels": int(audio.shape[0]),
        "num_samples_per_channel": int(audio.shape[1]),
        "sample_rate_hz": int(sample_rate),
        "duration_s": float(audio.shape[1] / sample_rate),
        "all_finite": bool(np.isfinite(audio).all()),
        "rms": float(np.sqrt(np.mean(np.square(audio), dtype=np.float64))),
        "peak_absolute": float(np.max(np.abs(audio))),
        "layout": "channel_major",
        "encoding": "float32",
    }


def audio_thresholds(thresholds: Mapping[str, float]) -> dict[str, float]:
    """Validate and normalize the fixed audio threshold schema."""

    missing = sorted(REQUIRED_AUDIO_THRESHOLDS - thresholds.keys())
    if missing:
        raise ValueError(f"MiniMax-H3 threshold sidecar is missing {missing}")
    values = {name: float(thresholds[name]) for name in REQUIRED_AUDIO_THRESHOLDS}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("MiniMax-H3 audio thresholds must be finite")
    for name in ("exact_audio_channels", "exact_audio_num_samples", "exact_audio_sample_rate_hz"):
        if values[name] <= 0 or not values[name].is_integer():
            raise ValueError(f"{name} must be a positive integer")
    if values["exact_audio_duration_s"] <= 0:
        raise ValueError("exact_audio_duration_s must be positive")
    if not -1.0 <= values["minimum_audio_waveform_correlation"] <= 1.0:
        raise ValueError("minimum_audio_waveform_correlation must be in [-1, 1]")
    if values["maximum_audio_normalized_rmse"] < 0:
        raise ValueError("maximum_audio_normalized_rmse must be non-negative")
    if values["minimum_reference_audio_rms"] <= 0:
        raise ValueError("minimum_reference_audio_rms must be positive")
    minimum_ratio = values["minimum_candidate_audio_rms_ratio"]
    maximum_ratio = values["maximum_candidate_audio_rms_ratio"]
    if not 0.0 < minimum_ratio <= maximum_ratio:
        raise ValueError("invalid MiniMax-H3 candidate audio RMS ratio interval")
    if not 0.0 < values["maximum_audio_peak_absolute"] <= 1.0:
        raise ValueError("maximum_audio_peak_absolute must be in (0, 1]")
    expected_duration = values["exact_audio_num_samples"] / values["exact_audio_sample_rate_hz"]
    if not math.isclose(
        values["exact_audio_duration_s"], expected_duration, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("MiniMax-H3 exact audio duration disagrees with sample count and rate")
    return values


def _waveform_correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference64 = np.asarray(reference, dtype=np.float64)
    candidate64 = np.asarray(candidate, dtype=np.float64)
    denominator = float(np.linalg.norm(reference64) * np.linalg.norm(candidate64))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.clip(np.dot(reference64, candidate64) / denominator, -1.0, 1.0))


def _si_sdr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference64 = np.asarray(reference, dtype=np.float64)
    candidate64 = np.asarray(candidate, dtype=np.float64)
    target_energy = float(np.dot(reference64, reference64))
    if target_energy <= np.finfo(np.float64).eps:
        return -300.0
    projection = reference64 * (float(np.dot(candidate64, reference64)) / target_energy)
    projection_energy = float(np.dot(projection, projection))
    residual = candidate64 - projection
    residual_energy = float(np.dot(residual, residual))
    epsilon = np.finfo(np.float64).eps
    return float(10.0 * math.log10((projection_energy + epsilon) / (residual_energy + epsilon)))


def compute_decoded_audio_metrics(
    reference_path: Path,
    candidate_path: Path,
    *,
    reference_sample_rate: int,
    candidate_sample_rate: int,
) -> DecodedAudioMetrics:
    """Compare persisted channel-major waveforms against the actual HF samples."""

    reference = np.load(reference_path, mmap_mode="r", allow_pickle=False)
    candidate = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    if reference_sample_rate <= 0 or candidate_sample_rate <= 0:
        raise ValueError("MiniMax-H3 decoded audio sample rates must be positive")
    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError(
            "MiniMax-H3 decoded audio arrays must have shape [channels, samples], got "
            f"{reference.shape} and {candidate.shape}"
        )
    if reference.dtype.kind not in "fiu" or candidate.dtype.kind not in "fiu":
        raise ValueError("MiniMax-H3 decoded audio arrays must be numeric")

    reference_finite = bool(np.isfinite(reference).all())
    candidate_finite = bool(np.isfinite(candidate).all())
    comparable = (
        reference.shape == candidate.shape
        and reference.shape[0] > 0
        and reference.shape[1] > 0
        and reference_finite
        and candidate_finite
    )
    reference64 = np.asarray(reference, dtype=np.float64)
    candidate64 = np.asarray(candidate, dtype=np.float64)
    reference_rms = float(np.sqrt(np.mean(np.square(reference64)))) if reference_finite else 0.0
    candidate_rms = float(np.sqrt(np.mean(np.square(candidate64)))) if candidate_finite else 0.0
    epsilon = np.finfo(np.float64).eps
    rms_ratio = candidate_rms / max(reference_rms, epsilon)

    if comparable:
        correlations = [
            _waveform_correlation(reference64[channel], candidate64[channel])
            for channel in range(reference.shape[0])
        ]
        normalized_rmse = [
            float(
                np.sqrt(np.mean(np.square(candidate64[channel] - reference64[channel])))
                / max(float(np.sqrt(np.mean(np.square(reference64[channel])))), epsilon)
            )
            for channel in range(reference.shape[0])
        ]
        si_sdr = [
            _si_sdr_db(reference64[channel], candidate64[channel])
            for channel in range(reference.shape[0])
        ]
        maximum_error = float(np.max(np.abs(candidate64 - reference64)))
    else:
        correlations = [-1.0]
        normalized_rmse = [1.0e30]
        si_sdr = [-300.0]
        maximum_error = 1.0e30

    reference_peak = float(np.max(np.abs(reference64))) if reference_finite else 1.0e30
    candidate_peak = float(np.max(np.abs(candidate64))) if candidate_finite else 1.0e30
    return DecodedAudioMetrics(
        reference_shape=tuple(int(value) for value in reference.shape),
        candidate_shape=tuple(int(value) for value in candidate.shape),
        reference_sample_rate=int(reference_sample_rate),
        candidate_sample_rate=int(candidate_sample_rate),
        reference_duration_s=float(reference.shape[1] / reference_sample_rate),
        candidate_duration_s=float(candidate.shape[1] / candidate_sample_rate),
        reference_all_finite=reference_finite,
        candidate_all_finite=candidate_finite,
        reference_peak_absolute=reference_peak,
        candidate_peak_absolute=candidate_peak,
        reference_rms=reference_rms,
        candidate_rms=candidate_rms,
        candidate_rms_ratio=rms_ratio,
        waveform_correlation_minimum=float(min(correlations)),
        waveform_correlation_mean=float(np.mean(correlations)),
        normalized_rmse_maximum=float(max(normalized_rmse)),
        si_sdr_db_minimum=float(min(si_sdr)),
        maximum_absolute_error=maximum_error,
    )


def evaluate_audio_quality(
    metrics: DecodedAudioMetrics,
    thresholds: Mapping[str, float],
) -> dict[str, AudioGateResult]:
    """Apply exact media invariants and direct HF/native waveform-parity gates."""

    values = audio_thresholds(thresholds)
    expected_channels = int(values["exact_audio_channels"])
    expected_samples = int(values["exact_audio_num_samples"])
    expected_rate = int(values["exact_audio_sample_rate_hz"])
    expected_duration = values["exact_audio_duration_s"]
    expected_shape = (expected_channels, expected_samples)
    duration_tolerance = 0.5 / expected_rate
    minimum_ratio = values["minimum_candidate_audio_rms_ratio"]
    maximum_ratio = values["maximum_candidate_audio_rms_ratio"]
    peak_maximum = values["maximum_audio_peak_absolute"]

    gates = {
        "reference_audio_shape": AudioGateResult(
            1.0 if metrics.reference_shape == expected_shape else 0.0,
            1.0,
            "==",
            metrics.reference_shape == expected_shape,
            f"expected {expected_shape}, got {metrics.reference_shape}",
        ),
        "candidate_audio_shape": AudioGateResult(
            1.0 if metrics.candidate_shape == expected_shape else 0.0,
            1.0,
            "==",
            metrics.candidate_shape == expected_shape,
            f"expected {expected_shape}, got {metrics.candidate_shape}",
        ),
        "reference_audio_sample_rate_hz": AudioGateResult(
            float(metrics.reference_sample_rate),
            float(expected_rate),
            "==",
            metrics.reference_sample_rate == expected_rate,
        ),
        "candidate_audio_sample_rate_hz": AudioGateResult(
            float(metrics.candidate_sample_rate),
            float(expected_rate),
            "==",
            metrics.candidate_sample_rate == expected_rate,
        ),
        "reference_audio_duration_s": AudioGateResult(
            metrics.reference_duration_s,
            expected_duration,
            "==",
            math.isclose(
                metrics.reference_duration_s,
                expected_duration,
                rel_tol=0.0,
                abs_tol=duration_tolerance,
            ),
        ),
        "candidate_audio_duration_s": AudioGateResult(
            metrics.candidate_duration_s,
            expected_duration,
            "==",
            math.isclose(
                metrics.candidate_duration_s,
                expected_duration,
                rel_tol=0.0,
                abs_tol=duration_tolerance,
            ),
        ),
        "reference_audio_finite": AudioGateResult(
            1.0 if metrics.reference_all_finite else 0.0,
            1.0,
            "==",
            metrics.reference_all_finite,
        ),
        "candidate_audio_finite": AudioGateResult(
            1.0 if metrics.candidate_all_finite else 0.0,
            1.0,
            "==",
            metrics.candidate_all_finite,
        ),
        "reference_audio_peak_absolute": AudioGateResult(
            metrics.reference_peak_absolute,
            peak_maximum,
            "<=",
            metrics.reference_peak_absolute <= peak_maximum,
        ),
        "candidate_audio_peak_absolute": AudioGateResult(
            metrics.candidate_peak_absolute,
            peak_maximum,
            "<=",
            metrics.candidate_peak_absolute <= peak_maximum,
        ),
        "reference_audio_rms": AudioGateResult(
            metrics.reference_rms,
            values["minimum_reference_audio_rms"],
            ">=",
            metrics.reference_rms >= values["minimum_reference_audio_rms"],
            "The fixed HF soundtrack must contain measurable audio.",
        ),
        "candidate_audio_rms_ratio_minimum": AudioGateResult(
            metrics.candidate_rms_ratio,
            minimum_ratio,
            ">=",
            metrics.candidate_rms_ratio >= minimum_ratio,
        ),
        "candidate_audio_rms_ratio_maximum": AudioGateResult(
            metrics.candidate_rms_ratio,
            maximum_ratio,
            "<=",
            metrics.candidate_rms_ratio <= maximum_ratio,
        ),
        "audio_waveform_correlation_minimum": AudioGateResult(
            metrics.waveform_correlation_minimum,
            values["minimum_audio_waveform_correlation"],
            ">=",
            metrics.waveform_correlation_minimum >= values["minimum_audio_waveform_correlation"],
            "Minimum is taken across the left and right channel comparisons.",
        ),
        "audio_normalized_rmse_maximum": AudioGateResult(
            metrics.normalized_rmse_maximum,
            values["maximum_audio_normalized_rmse"],
            "<=",
            metrics.normalized_rmse_maximum <= values["maximum_audio_normalized_rmse"],
            "RMSE is normalized independently by each HF channel RMS.",
        ),
        "audio_si_sdr_db_minimum": AudioGateResult(
            metrics.si_sdr_db_minimum,
            values["minimum_audio_si_sdr_db"],
            ">=",
            metrics.si_sdr_db_minimum >= values["minimum_audio_si_sdr_db"],
            "Minimum scale-invariant SDR across the left and right channels.",
        ),
        "audio_waveform_correlation_mean": AudioGateResult(
            metrics.waveform_correlation_mean,
            None,
            "diagnostic",
            True,
        ),
        "audio_maximum_absolute_error": AudioGateResult(
            metrics.maximum_absolute_error,
            None,
            "diagnostic",
            True,
        ),
    }
    return gates


def audio_quality_passed(gates: Mapping[str, AudioGateResult]) -> bool:
    return all(result.passed for result in gates.values())
