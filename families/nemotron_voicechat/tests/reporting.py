# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Listening views of the waveforms already used by this family's E2E test."""

from pathlib import Path

from tools.e2e_evidence import evidence_enabled, record_evidence


def record_audio_views(actual: dict, expected: dict, directory: Path) -> None:
    if not evidence_enabled():
        return
    views = []
    for role, output in (("Native", actual), ("Reference", expected)):
        try:
            import numpy as np
            import soundfile as sf

            raw_path = output.get("audio")
            if raw_path:
                path = Path(raw_path)
                if not path.is_file():
                    raise ValueError("the recorded waveform file is unavailable")
            elif "samples" in output:
                samples = np.asarray(output["samples"])
                if samples.ndim not in (1, 2) or samples.size == 0:
                    raise ValueError("waveform shape is unavailable")
                if samples.size * 4 + 1024 > 32 * 1024 * 1024:
                    raise ValueError("waveform exceeds the 32 MiB evidence bound")
                sample_rate = int(output["sample_rate"])
                if sample_rate <= 0:
                    raise ValueError("waveform sample rate is unavailable")
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"{role.lower()}-listening.wav"
                sf.write(path, samples, sample_rate, subtype="FLOAT")
            else:
                views.append(
                    {
                        "title": f"{role} audio",
                        "caption": "Listening unavailable: this result contains no waveform. "
                        "See the recorded token, text, or contract evidence.",
                    }
                )
                continue
            info = sf.info(path)
            views.append(
                {
                    "title": f"{role} audio",
                    "audio": path,
                    "caption": f"{info.samplerate} Hz; {info.channels} channel(s); "
                    f"{info.frames} samples/channel; {info.duration:.3f} seconds. "
                    "Listening is diagnostic; the family's original checks determine the result.",
                }
            )
        except Exception as error:
            views.append(
                {
                    "title": f"{role} audio",
                    "caption": f"Listening unavailable: {type(error).__name__}: {error}",
                }
            )
    record_evidence("views", views)
