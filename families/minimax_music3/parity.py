# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a native MiniMax-Music3 run is compared against, and how.

The pipeline is three stages that fail differently, so a single end-to-end
audio check cannot say which one is wrong. Each stage therefore has its own
comparison point, ordered so the first failure names the stage:

    frame_hiddens    the autoregressive stage's per-frame conditioning
    latent_chunks    each denoising window's Flow-VAE latents
    waveform         the stitched stereo output

The autoregressive stage samples, so `frame_hiddens` only reproduces under a
fixed seed and matching top-k; the flow-matching stage is deterministic given
its noise, so `latent_chunks` is the sharpest signal and is where a wrong
attention mask, a wrong sigma schedule, or a missing `preprocess_conv`
residual shows up first.

The recorded values below come from a run of the pinned reference on an
NVIDIA A40 with bfloat16 components. They are coarse -- a standard deviation
to one part in a thousand -- because their job is to catch an implementation
that is wrong by a lot, not to certify one that is right. Sample-level
agreement is what `RELATIVE_TOLERANCE` is for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

#: Inputs of the recorded reference run.
BASELINE_SEED = 7
BASELINE_AUDIO_SECONDS = 20.0
BASELINE_DTYPE = "bfloat16"
BASELINE_GPU = "NVIDIA A40"

#: Shapes the recorded run produced.
BASELINE_CHUNK_STARTS = (0, 100, 200, 300)
BASELINE_LATENT_SHAPE = (1, 128, 689)
BASELINE_FRAME_HIDDENS_SHAPE = (1, 500, 32768)
BASELINE_WAVEFORM_SAMPLES = 882688
BASELINE_WAVEFORM_CHANNELS = 2

#: Per-window standard deviation of the recorded latents, in window order.
BASELINE_LATENT_STD = (2.4088, 2.3549, 2.3566, 2.3701)

#: Standard deviation of the recorded conditioning.
BASELINE_FRAME_HIDDENS_STD = 0.9412

#: How far a recomputed statistic may drift before the run is not the same run.
STATISTIC_TOLERANCE = 0.05

#: Element-wise tolerance for a native stage against the reference tensor.
RELATIVE_TOLERANCE = 2e-2

#: The recorded waveform and the recorded latents came from two invocations of
#: the reference, not one. Decoding the recorded latents -- with the reference
#: vocoder or with the engine, identically -- and stitching them reproduces the
#: recorded sample count exactly but differs from the recorded waveform by an
#: RMS of about 1.06e-03, spread evenly across all four windows. Uniformity
#: rules out a stitching error; the cause is bfloat16 run-to-run drift through
#: an autoregressive stage and thirty diffusion steps.
#:
#: So :func:`check_waveform` compares length and channel count, which the
#: arithmetic does pin, and sample-level agreement is asserted against a
#: reference decode of the *same* latents rather than against this file. A
#: future capture should emit the waveform and the latents from one call.
CROSS_RUN_WAVEFORM_RMS = 1.06e-03


class _Array(Protocol):
    shape: tuple[int, ...]

    def std(self) -> float: ...


@dataclass(frozen=True)
class StageResult:
    """Outcome of comparing one stage."""

    stage: str
    passed: bool
    detail: str


def _fail(stage: str, detail: str) -> StageResult:
    return StageResult(stage=stage, passed=False, detail=detail)


def _pass(stage: str, detail: str = "") -> StageResult:
    return StageResult(stage=stage, passed=True, detail=detail)


def check_chunk_starts(starts: tuple[int, ...]) -> StageResult:
    """The window plan must match; everything downstream is indexed by it."""

    actual = tuple(int(value) for value in starts)
    if actual != BASELINE_CHUNK_STARTS:
        return _fail(
            "chunk_starts",
            f"got {actual}, the recorded run used {BASELINE_CHUNK_STARTS}",
        )
    return _pass("chunk_starts")


def check_frame_hiddens(hidden: _Array) -> StageResult:
    """The autoregressive stage's conditioning, by shape and spread."""

    shape = tuple(int(dim) for dim in hidden.shape)
    if shape != BASELINE_FRAME_HIDDENS_SHAPE:
        return _fail(
            "frame_hiddens",
            f"shape {shape}, expected {BASELINE_FRAME_HIDDENS_SHAPE}",
        )
    drift = abs(float(hidden.std()) - BASELINE_FRAME_HIDDENS_STD)
    if drift > STATISTIC_TOLERANCE:
        return _fail(
            "frame_hiddens",
            f"std drifted by {drift:.4f} from {BASELINE_FRAME_HIDDENS_STD}",
        )
    return _pass("frame_hiddens")


def check_latent_chunks(chunks: list[_Array]) -> StageResult:
    """Each denoising window, by shape and spread."""

    if len(chunks) != len(BASELINE_LATENT_STD):
        return _fail(
            "latent_chunks",
            f"{len(chunks)} windows, the recorded run produced "
            f"{len(BASELINE_LATENT_STD)}",
        )
    for index, (chunk, expected) in enumerate(zip(chunks, BASELINE_LATENT_STD)):
        shape = tuple(int(dim) for dim in chunk.shape)
        if shape != BASELINE_LATENT_SHAPE:
            return _fail(
                "latent_chunks",
                f"window {index} shape {shape}, expected {BASELINE_LATENT_SHAPE}",
            )
        drift = abs(float(chunk.std()) - expected)
        if drift > STATISTIC_TOLERANCE:
            return _fail(
                "latent_chunks",
                f"window {index} std drifted by {drift:.4f} from {expected}",
            )
    return _pass("latent_chunks")


def check_waveform(samples: int, channels: int) -> StageResult:
    """The stitched output's length and channel count."""

    if channels != BASELINE_WAVEFORM_CHANNELS:
        return _fail(
            "waveform",
            f"{channels} channels, expected {BASELINE_WAVEFORM_CHANNELS}",
        )
    if samples != BASELINE_WAVEFORM_SAMPLES:
        return _fail(
            "waveform",
            f"{samples} samples, the recorded run produced "
            f"{BASELINE_WAVEFORM_SAMPLES}",
        )
    return _pass("waveform")


def first_failure(results: list[StageResult]) -> StageResult | None:
    """Return the earliest failing stage, which is the one worth debugging."""

    for result in results:
        if not result.passed:
            return result
    return None
