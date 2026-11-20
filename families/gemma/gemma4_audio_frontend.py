# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-owned Gemma4 audio frontend constants for the native C++ runtime."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def frontend_constants(source: Path, raw: dict, capacity: int):
    path = source / "processor_config.json"
    metadata = json.loads(path.read_text()).get("feature_extractor", {}) if path.is_file() else {}
    rate = int(metadata.get("sampling_rate", 16000))
    if raw.get("model_type") == "gemma4_unified":
        width = int(raw["audio_config"]["audio_embed_dim"])
        if (rate != 16000 or capacity <= 0 or width <= 0
                or int(metadata.get("feature_size", width)) != width
                or int(metadata.get("audio_samples_per_token", width)) != width
                or float(metadata.get("padding_value", 0)) != 0):
            raise ValueError("Gemma4 unified waveform metadata does not match its projection")
        return ({
            "waveform": True, "sample_rate": rate, "mel_bins": width, "max_frames": capacity,
            "hidden_size": raw["text_config"]["hidden_size"],
            "audio_token": raw["audio_token_id"], "begin_token": raw["boa_token_id"],
            "end_token": raw["eoa_token_index"],
        }, np.empty(0, np.float32), np.empty(0, np.float64))
    frame = int(metadata.get("frame_length", 320))
    hop = int(metadata.get("hop_length", 160))
    bins = int(metadata.get("feature_size", 128))
    fft = int(metadata.get("fft_length", 1 << (frame - 1).bit_length()))
    if rate != 16000 or frame < 2 or hop < 1 or fft < frame or fft & (fft - 1):
        raise ValueError("Gemma4 native audio requires valid 16 kHz power-of-two FFT geometry")
    if bins != int(raw["audio_config"]["subsampling_conv_channels"][0]) or capacity <= 0:
        raise ValueError("Gemma4 frontend geometry does not match its audio projection")
    if float(metadata.get("dither", 0)) != 0:
        raise NotImplementedError("Gemma4 native audio does not inject stochastic frontend dithering")
    mean = np.asarray(metadata.get("per_bin_mean") or [0] * bins, np.float64)
    std = np.asarray(metadata.get("per_bin_stddev") or [1] * bins, np.float64)
    minimum, maximum = float(metadata.get("min_frequency", 0)), float(metadata.get("max_frequency", 8000))
    if mean.shape != (bins,) or std.shape != (bins,) or np.any(std <= 0):
        raise ValueError("Gemma4 audio bin normalization has invalid dimensions or scale")
    if not 0 <= minimum < maximum <= rate / 2:
        raise ValueError("Gemma4 audio mel frequency range is invalid")
    # HTK mel triangles, with the same double precision frequency grid as the provider.
    mel = np.linspace(2595 * np.log10(1 + minimum / 700),
                      2595 * np.log10(1 + maximum / 700), bins + 2)
    frequencies = 700 * (10**(mel / 2595) - 1)
    differences = np.diff(frequencies)
    grid = np.linspace(0, rate // 2, fft // 2 + 1)
    slopes = frequencies[None, :] - grid[:, None]
    filters = np.maximum(0, np.minimum(-slopes[:, :-2] / differences[:-1],
                                      slopes[:, 2:] / differences[1:]))
    window = np.hanning(frame + 1)[:-1].astype(np.float32)
    spec = {
        "sample_rate": rate, "frame_length": frame, "hop_length": hop, "fft_length": fft,
        "mel_bins": bins, "max_frames": 4 * capacity,
        "hidden_size": raw["text_config"]["hidden_size"], "audio_token": raw["audio_token_id"],
        "begin_token": raw["boa_token_id"], "end_token": raw["eoa_token_id"],
        "input_scale": float(metadata.get("input_scale_factor", 1)),
        "preemphasis": float(metadata.get("preemphasis", 0)),
        "preemphasis_htk": bool(metadata.get("preemphasis_htk_flavor", True)),
        "mel_floor": float(metadata.get("mel_floor", 0.001)),
        "mean": mean.tolist(), "std": std.tolist(),
    }
    if spec["mel_floor"] <= 0 or not np.isfinite([*mean, *std, spec["input_scale"],
                                                 spec["preemphasis"], spec["mel_floor"]]).all():
        raise ValueError("Gemma4 audio frontend contains invalid numeric parameters")
    return spec, window, filters
