# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph for the condition encoder.

The network is the reference forward with one thing folded away: the softmax
over ``layer_weight_logits`` and the scalar ``layer_scale`` are both constant
at build time, so they collapse into a single eight-element vector applied as
one elementwise product before the reduction.

    hidden_states (1, F, 8*4096)
      -> transpose, reshape          (1, 8, 4096, F)
      -> * folded_mix (1, 8, 1, 1)
      -> reduce sum over the stream axis   (1, 4096, F)
      -> reshape                     (1, 4096, 1, F)
      -> convolution 4096 -> 2048, kernel (1, 3), padding (0, 1)
      -> reshape                     (1, 2048, F)
      -> resize NEAREST to latent_length, ASYMMETRIC + FLOOR
      -> transpose                   (1, latent_length, 2048)

The resize settings are not defaults and are not interchangeable: the
reference computes ``floorf(dst * in / out)`` with the ratio in single
precision, which is ASYMMETRIC coordinates with FLOOR rounding. See
:func:`.condition_encoder.nearest_indices` for the boundary case that
distinguishes it.

**Precision.** Built on an A40 with TensorRT 11.1.0.106, the engine agrees with
the oracle to 3.5e-06 relative with TF32 off and to 1.2e-04 with TF32 on. The
gap is the 4096-deep dot product in ``proj`` losing mantissa; the graph is the
same either way. :data:`DISABLE_TF32` makes that a stated choice rather than a
default inherited from the builder.
"""

from __future__ import annotations

from typing import Any

from .condition_encoder import (
    CONDITION_HIDDEN_DIM,
    NUM_CONDITION_LAYERS,
    OUT_DIM,
    PROJ_KERNEL_SIZE,
    PROJ_PADDING,
    latent_length,
)

INPUT_NAME = "hidden_states"
OUTPUT_NAME = "condition"

#: Keep the projection in full fp32. Measured: TF32 costs about 1e-4 relative
#: on this layer. Whether a 30-step flow-matching loop cares is not yet
#: measured, so the accurate setting is the one recorded here.
DISABLE_TF32 = True

#: Build-time workspace. The engine is ~100 MB, almost all of it the
#: 2048 x 4096 x 3 projection weights.
WORKSPACE_BYTES = 4 << 30


def folded_mix(layer_weight_logits, layer_scale):
    """Return ``softmax(logits) * layer_scale`` as one eight-element vector."""

    import numpy as np

    logits = np.asarray(layer_weight_logits, dtype=np.float32).reshape(-1)
    if logits.size != NUM_CONDITION_LAYERS:
        raise ValueError(
            f"expected {NUM_CONDITION_LAYERS} layer weights, got {logits.size}"
        )
    shifted = np.exp(logits - logits.max())
    weights = shifted / shifted.sum()
    scale = np.asarray(layer_scale, dtype=np.float32).reshape(-1)
    if scale.size != 1:
        raise ValueError(f"layer_scale must be one element, got {scale.size}")
    return np.ascontiguousarray(weights * scale[0], dtype=np.float32)


def add_condition_encoder(
    network: Any,
    trt: Any,
    hidden_states: Any,
    *,
    frames: int,
    mix,
    proj_weight,
    proj_bias,
):
    """Add the condition-encoder subgraph and return its output tensor."""

    import numpy as np

    out_len = latent_length(frames)

    stack = network.add_shuffle(hidden_states)
    stack.first_transpose = (0, 2, 1)
    stack.reshape_dims = (1, NUM_CONDITION_LAYERS, CONDITION_HIDDEN_DIM, frames)

    weights = network.add_constant(
        (1, NUM_CONDITION_LAYERS, 1, 1),
        trt.Weights(np.ascontiguousarray(mix).reshape(1, NUM_CONDITION_LAYERS, 1, 1)),
    )
    weighted = network.add_elementwise(
        stack.get_output(0), weights.get_output(0), trt.ElementWiseOperation.PROD
    )
    mixed = network.add_reduce(
        weighted.get_output(0),
        trt.ReduceOperation.SUM,
        axes=1 << 1,
        keep_dims=False,
    )

    to_conv = network.add_shuffle(mixed.get_output(0))
    to_conv.reshape_dims = (1, CONDITION_HIDDEN_DIM, 1, frames)
    conv = network.add_convolution_nd(
        to_conv.get_output(0),
        OUT_DIM,
        (1, PROJ_KERNEL_SIZE),
        trt.Weights(
            np.ascontiguousarray(
                np.asarray(proj_weight, dtype=np.float32).reshape(
                    OUT_DIM, CONDITION_HIDDEN_DIM, 1, PROJ_KERNEL_SIZE
                )
            )
        ),
        trt.Weights(np.ascontiguousarray(np.asarray(proj_bias, dtype=np.float32))),
    )
    conv.padding_nd = (0, PROJ_PADDING)

    to_resize = network.add_shuffle(conv.get_output(0))
    to_resize.reshape_dims = (1, OUT_DIM, frames)
    resize = network.add_resize(to_resize.get_output(0))
    resize.shape = (1, OUT_DIM, out_len)
    resize.resize_mode = trt.InterpolationMode.NEAREST
    resize.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC
    resize.nearest_rounding = trt.ResizeRoundMode.FLOOR

    result = network.add_shuffle(resize.get_output(0))
    result.first_transpose = (0, 2, 1)
    return result.get_output(0)


def configure(config: Any, trt: Any) -> None:
    """Apply the builder settings this engine is validated under."""

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
    if DISABLE_TF32:
        config.clear_flag(trt.BuilderFlag.TF32)
