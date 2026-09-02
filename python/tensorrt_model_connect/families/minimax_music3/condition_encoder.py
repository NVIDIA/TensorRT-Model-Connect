# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The condition encoder, as an oracle for the engine that will replace it.

This is the smallest of the pipeline's five networks -- four tensors -- and it
sits between two things already measured: it consumes ``frame_hiddens`` from
the autoregressive stage and produces the conditioning each denoising window
is given. That makes it the right first engine: its input and its output shape
are both pinned by the recorded baseline.

The forward, from the reference implementation
(diffusers v0.40.0, ``models/condition_embedders/condition_embedder_minimax_music3.py``,
Apache-2.0, (c) The MiniMax Team and The HuggingFace Team)::

    (b, frames, 8 * 4096) -> transpose -> (b, 8, 4096, frames)
    softmax(layer_weight_logits) over the eight streams, weighted sum
    scale by layer_scale
    Conv1d(4096 -> 2048, kernel 3, padding 1)
    nearest-neighbour resample to the latent timeline
    -> (b, latent_length, 2048)

The resample ratio is where the two clocks meet: the language model emits
frames at ``input_sampling_rate / input_hop_length`` = 25 Hz and the vocoder
consumes latents at ``output_sampling_rate / output_hop_length`` = 86.13 Hz.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Published component config.
CONDITION_HIDDEN_DIM = 4096
NUM_CONDITION_LAYERS = 8
OUT_DIM = 2048
INPUT_SAMPLING_RATE = 24000
INPUT_HOP_LENGTH = 960
OUTPUT_SAMPLING_RATE = 44100
OUTPUT_HOP_LENGTH = 512

#: Conv1d shape, fixed by the checkpoint.
PROJ_KERNEL_SIZE = 3
PROJ_PADDING = 1


def resample_ratio() -> float:
    """Latent frames produced per autoregressive frame."""

    return (
        OUTPUT_SAMPLING_RATE
        / INPUT_SAMPLING_RATE
        * INPUT_HOP_LENGTH
        / OUTPUT_HOP_LENGTH
    )


def latent_length(num_frames: int) -> int:
    """Latent frames the encoder emits for ``num_frames`` input frames.

    Truncating rather than rounding is what the reference does, and it is why a
    200-frame window becomes 689 latent frames rather than 690.
    """

    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    return max(1, int(num_frames * resample_ratio()))


def nearest_indices(num_frames: int) -> tuple[int, ...]:
    """Source frame chosen for each output position by nearest resampling.

    ``F.interpolate(..., mode="nearest")`` maps output position ``i`` to
    ``min(floorf(i * scale), in - 1)`` where ``scale`` is ``in / out`` in
    **single precision**. The precision matters: at an output position where
    ``i * in / out`` is exactly an integer, float32 lands a hair below it and
    floors to one frame earlier. Computing the same quotient in exact integer
    arithmetic is mathematically cleaner and reproduces the reference wrongly
    -- for 500 input frames it disagrees at output position 861, where
    ``861 * 500 / 1722`` is exactly 250 and the reference picks 249.
    """

    import numpy as np

    out = latent_length(num_frames)
    scale = np.float32(num_frames) / np.float32(out)
    positions = np.arange(out, dtype=np.float32) * scale
    source = np.floor(positions).astype(np.int64)
    np.minimum(source, num_frames - 1, out=source)
    return tuple(int(index) for index in source)


def forward(hidden_states, layer_weight_logits, layer_scale, proj_weight, proj_bias):
    """Run the reference forward with numpy arrays.

    ``hidden_states`` is ``(batch, frames, 8 * 4096)``; the return is
    ``(batch, latent_length, 2048)``.
    """

    import numpy as np

    hidden = np.asarray(hidden_states, dtype=np.float32)
    batch, frames, width = hidden.shape
    expected_width = NUM_CONDITION_LAYERS * CONDITION_HIDDEN_DIM
    if width != expected_width:
        raise ValueError(
            f"hidden_states last dimension is {width}, expected {expected_width}"
        )

    # (b, frames, layers * dim) -> (b, layers, dim, frames)
    stacked = hidden.transpose(0, 2, 1).reshape(
        batch, NUM_CONDITION_LAYERS, CONDITION_HIDDEN_DIM, frames
    )

    logits = np.asarray(layer_weight_logits, dtype=np.float32)
    weights = np.exp(logits - logits.max())
    weights = weights / weights.sum()
    mixed = np.einsum("blht,l->bht", stacked, weights)
    mixed = np.asarray(layer_scale, dtype=np.float32).reshape(()) * mixed

    weight = np.asarray(proj_weight, dtype=np.float32)
    bias = np.asarray(proj_bias, dtype=np.float32)
    padded = np.pad(mixed, ((0, 0), (0, 0), (PROJ_PADDING, PROJ_PADDING)))
    windows = np.stack(
        [padded[:, :, i : i + frames] for i in range(PROJ_KERNEL_SIZE)], axis=-1
    )
    projected = np.einsum("bhtk,ohk->bot", windows, weight) + bias[None, :, None]

    picked = np.asarray(nearest_indices(frames), dtype=np.int64)
    return projected[:, :, picked].transpose(0, 2, 1)


def engine_io_shapes(num_frames: int) -> dict[str, Sequence[int]]:
    """Input and output shapes of the engine for one window."""

    return {
        "hidden_states": (1, num_frames, NUM_CONDITION_LAYERS * CONDITION_HIDDEN_DIM),
        "condition": (1, latent_length(num_frames), OUT_DIM),
    }
