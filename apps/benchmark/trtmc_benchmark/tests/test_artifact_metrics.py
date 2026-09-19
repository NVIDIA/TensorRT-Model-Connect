# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import wave

import numpy as np
from PIL import Image

from trtmc_benchmark.artifact_metrics import (
    generated_audio_metrics,
    generated_audio_passes,
    generated_media_metrics,
    generated_media_passes,
)


def test_generated_media_compares_retained_video_frames(tmp_path: Path) -> None:
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    pixels = np.full((8, 8, 3), 96, dtype=np.uint8)
    Image.fromarray(pixels).save(left)
    Image.fromarray(pixels).save(right)
    candidate = {
        "media_count": 5,
        "artifact_indices": [0, 2, 4],
        "frame_artifacts": [str(left), str(left), str(left)],
    }
    reference = {
        "media_count": 5,
        "artifact_indices": [0, 2, 4],
        "frame_artifacts": [str(right), str(right), str(right)],
    }

    metrics = generated_media_metrics(candidate, reference)

    assert metrics["media_count"] == 5
    assert metrics["min_psnr"] == 100.0
    assert generated_media_passes(metrics, {"min_psnr": 5.0, "min_ssim": 0.1})


def test_generated_audio_compares_complete_waveforms(tmp_path: Path) -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    samples = np.sin(2.0 * np.pi * 440.0 * time).astype(np.float32)
    candidate = tmp_path / "candidate.wav"
    reference = tmp_path / "reference.wav"
    pcm = np.rint(samples * 32767.0).astype("<i2").tobytes()
    for path in (candidate, reference):
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(pcm)

    metrics = generated_audio_metrics(
        {"audio_artifact": str(candidate)}, {"audio_artifact": str(reference)}
    )

    assert metrics["duration_ratio"] == 1.0
    assert metrics["rms_ratio"] == 1.0
    assert metrics["log_spectral_distance"] == 0.0
    assert generated_audio_passes(
        metrics,
        {
            "min_duration_ratio": 0.8,
            "max_duration_ratio": 1.2,
            "min_rms_ratio": 0.5,
            "max_rms_ratio": 2.0,
            "max_log_spectral_distance": 3.0,
        },
    )
