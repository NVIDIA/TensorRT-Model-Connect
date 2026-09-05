# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The sphn subset needed by the official pipeline for the fixed 24 kHz WAV."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def read(path: str) -> tuple[np.ndarray, int]:
    payload = Path(path).read_bytes()
    if payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise RuntimeError("PersonaPlex input must be a RIFF/WAVE file")
    fmt = None
    audio_bytes = None
    offset = 12
    while offset + 8 <= len(payload):
        chunk_id = payload[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", payload, offset + 4)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(payload):
            raise RuntimeError("PersonaPlex WAV contains a truncated chunk")
        if chunk_id == b"fmt ":
            fmt = payload[chunk_start:chunk_end]
        elif chunk_id == b"data":
            audio_bytes = payload[chunk_start:chunk_end]
        offset = chunk_end + (chunk_size & 1)
    if fmt is None or len(fmt) < 16 or audio_bytes is None:
        raise RuntimeError("PersonaPlex WAV is missing fmt or data")
    audio_format, channels, sample_rate, _, block_align, bits = struct.unpack_from("<HHIIHH", fmt)
    if channels != 1 or sample_rate != 24_000:
        raise RuntimeError("PersonaPlex input must be mono 24 kHz audio")
    if audio_format == 1 and bits == 16 and block_align == 2:
        audio = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32)
        audio *= 1.0 / 32768.0
    elif audio_format == 3 and bits == 32 and block_align == 4:
        audio = np.frombuffer(audio_bytes, dtype="<f4").astype(np.float32, copy=True)
    else:
        raise RuntimeError("PersonaPlex input must use PCM16 or IEEE float32 samples")
    if not np.isfinite(audio).all():
        raise RuntimeError("PersonaPlex input contains non-finite audio")
    return np.ascontiguousarray(audio.reshape(1, -1)), int(sample_rate)


def resample(
    audio: np.ndarray,
    *,
    src_sample_rate: int,
    dst_sample_rate: int,
) -> np.ndarray:
    if src_sample_rate == dst_sample_rate == 24_000:
        return np.ascontiguousarray(audio, dtype=np.float32)
    raise RuntimeError("PersonaPlex reference only supports native 24 kHz audio")
