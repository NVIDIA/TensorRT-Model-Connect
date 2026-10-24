# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestration the native runtime has to reproduce.

Read from the reference implementation in diffusers v0.40.0, under
``modular_pipelines/minimax_music3``. None of this is in the checkpoint
configs: the window size, the hop, the crop widths and the sigma schedule are
constants in the reference, and a runtime that guesses them produces audio that
is plausible and wrong.

The pipeline is three sequential stages::

    semantic_generator   tokenize -> autoregressive
                         Qwen2Tokenizer, Qwen3ForCausalLM, RVQ depth decoder
                         out: text_ids [2, seq], frame_hiddens [1, frames, 8 * 4096]

    denoise              prepare_chunks -> chunk denoise
                         condition_encoder, transformer, scheduler, CFG guider
                         out: chunk_starts, latent_chunks

    decode               vocoder, crop, concatenate
                         out: audios (batch, channels, samples) in [-1, 1]

``frame_hiddens`` concatenates the global language model's hidden state with
the depth decoder's seven, which is what the condition encoder's eight
``layer_weight_logits`` weight -- they index codebook streams, not layers.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Stage 1: autoregressive -------------------------------------------------

#: Frames the language model emits per second of audio.
FRAME_RATE_HZ = 25.0

#: Upstream cap on generated frames.
MAX_AUDIO_FRAMES = 9000

#: Token budget for the tokenized description and lyrics.
MAX_PROMPT_TOKENS = 5000

#: Classifier-free guidance scale. From the reference's guider config in
#: ``diffusers.modular_pipelines.minimax_music3.denoise``, where the
#: unconditional branch conditions on zeros rather than on a re-encoded
#: empty prompt.
GUIDANCE_SCALE = 1.7

#: ``text_ids`` carries the conditional prompt and its classifier-free
#: counterpart in one batch.
TEXT_IDS_BATCH = 2

# --- Stage 2: chunked flow matching -----------------------------------------

#: Frames per denoising window, and the stride between windows: a 50% overlap.
CHUNK_FRAMES = 200
CHUNK_HOP = 100

#: Euler steps per window, the reference default.
DEFAULT_INFERENCE_STEPS = 30

#: Latent frames two neighbouring windows share.
OVERLAP_LATENT_FRAMES = 344

# --- Stage 3: decode ---------------------------------------------------------

#: Latent frames dropped from a window's left edge, for every window after the
#: first, and from its right edge, for every window before the last. Together
#: they remove exactly one overlap.
CROP_LEFT_LATENT = 86
CROP_RIGHT_LATENT = OVERLAP_LATENT_FRAMES - CROP_LEFT_LATENT

#: Waveform samples per latent frame, from ``condition_encoder.output_hop_length``.
LATENT_HOP_LENGTH = 512

#: Output rate, from ``vocoder.config.sampling_rate``.
SAMPLING_RATE = 44100


@dataclass(frozen=True)
class ChunkPlan:
    """Where each denoising window starts, in autoregressive frames."""

    starts: tuple[int, ...]
    frames: int

    @property
    def count(self) -> int:
        return len(self.starts)


def chunk_starts(num_frames: int) -> ChunkPlan:
    """Return the window starts for ``num_frames`` autoregressive frames.

    Mirrors ``MiniMaxMusic3PrepareChunksStep``: a single window when the whole
    generation fits in one, otherwise a window every ``CHUNK_HOP`` frames up to
    but not including the last partial hop.
    """

    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if num_frames <= CHUNK_FRAMES:
        starts: tuple[int, ...] = (0,)
    else:
        starts = tuple(range(0, num_frames - CHUNK_HOP, CHUNK_HOP))
    return ChunkPlan(starts=starts, frames=num_frames)


def sigma_schedule(num_inference_steps: int = DEFAULT_INFERENCE_STEPS) -> tuple[float, ...]:
    """Return the flow-matching sigmas, as the reference builds them.

    ``np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)`` -- a
    linear ramp, not the scheduler's own default schedule.
    """

    if num_inference_steps < 1:
        raise ValueError(
            f"num_inference_steps must be at least 1, got {num_inference_steps}"
        )
    if num_inference_steps == 1:
        return (1.0,)
    last = 1.0 / num_inference_steps
    step = (last - 1.0) / (num_inference_steps - 1)
    return tuple(1.0 + step * index for index in range(num_inference_steps))


def latent_frames_per_second() -> float:
    """Latent frames per second the vocoder consumes."""

    return SAMPLING_RATE / LATENT_HOP_LENGTH


def max_audio_seconds() -> float:
    """Longest generation the upstream frame cap allows."""

    return MAX_AUDIO_FRAMES / FRAME_RATE_HZ


def transformer_calls(num_frames: int, num_inference_steps: int = DEFAULT_INFERENCE_STEPS) -> int:
    """Diffusion-transformer evaluations for one generation.

    One per Euler step per window, doubled because classifier-free guidance
    evaluates the conditional and unconditional branches.
    """

    return chunk_starts(num_frames).count * num_inference_steps * TEXT_IDS_BATCH
