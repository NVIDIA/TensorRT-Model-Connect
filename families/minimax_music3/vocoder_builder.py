# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph for the vocoder.

Twenty-seven convolutions, four transposed convolutions and twenty-nine Snake
activations, arranged as::

    latents (1, 128, L) -> reshape (2, 64, 1, L)     the stereo fold
      dec_in_proj, conv_in
      four blocks: Snake -> ConvTranspose -> three residual units
      snake_out, conv_out, tanh
    -> reshape (1, 2, L * 512)

Two constants are folded on the host rather than expressed as layers. Weight
normalisation collapses to a plain kernel, since ``g * v / ||v||`` is fixed
once the checkpoint is loaded. And Snake's divisor becomes a reciprocal:
``x + sin(a*x)**2 / (a + 1e-9)`` is built as ``x + sin(a*x)**2 * r`` with
``r = 1 / (a + 1e-9)`` precomputed, which saves a division layer in each of
the twenty-nine activations.

Convolutions run as 2-D layers over a height of one, because TensorRT's 1-D
convolution support is narrower than its 2-D support and the reshape is free.

**Validated.** Built on an A40 with TensorRT 11.1.0.106 -- 95 s, a 218 MB
engine -- and run over the four recorded latent windows. Against the reference
vocoder on the same input the engine agrees to 5.1e-06 relative. Stitching the
four decoded windows reproduces the recorded waveform's 882688 samples
exactly. It does not reproduce that waveform's samples, but neither does the
reference vocoder, by the same margin: see
:data:`.parity.CROSS_RUN_WAVEFORM_RMS`.
"""

from __future__ import annotations

from typing import Any

from .vocoder import (
    CONV_IN_KERNEL,
    CONV_OUT_KERNEL,
    DECODER_HIDDEN_DIM,
    DECODER_INPUT_DIM,
    LATENT_CHANNELS,
    OUTPUT_CHANNELS,
    RESIDUAL_DILATIONS,
    RESIDUAL_KERNEL,
    STREAM_CHANNELS,
    STREAMS,
    blocks,
    fuse_weight_norm,
    residual_padding,
    waveform_samples,
)

INPUT_NAME = "latents"
OUTPUT_NAME = "waveform"

#: Snake adds this before reciprocating, matching the reference.
SNAKE_EPSILON = 1e-9

#: The projection depths here are shallower than the condition encoder's, so
#: TF32 costs less; it is still cleared so precision is a stated choice.
DISABLE_TF32 = True
WORKSPACE_BYTES = 4 << 30


def snake_reciprocal(alpha):
    """Return ``1 / (alpha + 1e-9)``, shaped for a (N, C, 1, L) tensor."""

    import numpy as np

    a = np.asarray(alpha, dtype=np.float32).reshape(-1)
    return np.ascontiguousarray(
        (1.0 / (a + SNAKE_EPSILON)).reshape(1, a.size, 1, 1), dtype=np.float32
    )


def _alpha_constant(alpha):
    import numpy as np

    a = np.asarray(alpha, dtype=np.float32).reshape(-1)
    return np.ascontiguousarray(a.reshape(1, a.size, 1, 1), dtype=np.float32)


def add_snake(network: Any, trt: Any, hidden: Any, alpha) -> Any:
    """Add ``x + sin(alpha * x) ** 2 * reciprocal``."""

    scale = network.add_constant(_alpha_constant(alpha).shape,
                                 trt.Weights(_alpha_constant(alpha)))
    recip = network.add_constant(snake_reciprocal(alpha).shape,
                                 trt.Weights(snake_reciprocal(alpha)))
    scaled = network.add_elementwise(hidden, scale.get_output(0),
                                     trt.ElementWiseOperation.PROD)
    sine = network.add_unary(scaled.get_output(0), trt.UnaryOperation.SIN)
    squared = network.add_elementwise(sine.get_output(0), sine.get_output(0),
                                      trt.ElementWiseOperation.PROD)
    weighted = network.add_elementwise(squared.get_output(0), recip.get_output(0),
                                       trt.ElementWiseOperation.PROD)
    return network.add_elementwise(hidden, weighted.get_output(0),
                                   trt.ElementWiseOperation.SUM).get_output(0)


def _kernel(trt: Any, array, out_channels: int, in_channels: int, width: int):
    import numpy as np

    return trt.Weights(
        np.ascontiguousarray(
            np.asarray(array, dtype=np.float32).reshape(
                out_channels, in_channels, 1, width
            )
        )
    )


def _bias(trt: Any, array):
    import numpy as np

    return trt.Weights(np.ascontiguousarray(np.asarray(array, dtype=np.float32)))


def add_conv(network, trt, hidden, weight, bias, out_channels, in_channels,
             kernel, *, padding=0, dilation=1):
    """Add a length-preserving 1-D convolution as a 2-D layer."""

    layer = network.add_convolution_nd(
        hidden, out_channels, (1, kernel),
        _kernel(trt, weight, out_channels, in_channels, kernel),
        _bias(trt, bias),
    )
    layer.padding_nd = (0, padding)
    layer.dilation_nd = (1, dilation)
    return layer.get_output(0)


def add_deconv(network, trt, hidden, weight, bias, in_channels, out_channels, stride):
    """Add the block's transposed convolution."""

    import numpy as np

    kernel = 2 * stride
    layer = network.add_deconvolution_nd(
        hidden, out_channels, (1, kernel),
        trt.Weights(
            np.ascontiguousarray(
                np.asarray(weight, dtype=np.float32).reshape(
                    in_channels, out_channels, 1, kernel
                )
            )
        ),
        _bias(trt, bias),
    )
    layer.stride_nd = (1, stride)
    from .vocoder import Block

    layer.padding_nd = (0, Block(0, in_channels, out_channels, stride).padding)
    return layer.get_output(0)


def _fused(weights: dict, prefix: str):
    return fuse_weight_norm(weights[f"{prefix}.weight_g"], weights[f"{prefix}.weight_v"])


def add_residual_unit(network, trt, hidden, weights, prefix, dim, dilation):
    """Snake, dilated convolution, Snake, pointwise convolution, plus input."""

    branch = add_snake(network, trt, hidden, weights[f"{prefix}.snake1.alpha"])
    branch = add_conv(
        network, trt, branch, _fused(weights, f"{prefix}.conv1"),
        weights[f"{prefix}.conv1.bias"], dim, dim, RESIDUAL_KERNEL,
        padding=residual_padding(dilation), dilation=dilation,
    )
    branch = add_snake(network, trt, branch, weights[f"{prefix}.snake2.alpha"])
    branch = add_conv(
        network, trt, branch, _fused(weights, f"{prefix}.conv2"),
        weights[f"{prefix}.conv2.bias"], dim, dim, 1,
    )
    return network.add_elementwise(
        hidden, branch, trt.ElementWiseOperation.SUM
    ).get_output(0)


def add_vocoder(network: Any, trt: Any, latents: Any, *, latent_length: int,
                weights: dict) -> Any:
    """Add the whole decoder and return the stereo waveform tensor."""

    fold = network.add_shuffle(latents)
    fold.reshape_dims = (STREAMS, STREAM_CHANNELS, 1, latent_length)
    hidden = fold.get_output(0)

    hidden = add_conv(
        network, trt, hidden, weights["dec_in_proj.weight"],
        weights["dec_in_proj.bias"], DECODER_INPUT_DIM, STREAM_CHANNELS, 1,
    )
    hidden = add_conv(
        network, trt, hidden, _fused(weights, "conv_in"), weights["conv_in.bias"],
        DECODER_HIDDEN_DIM, DECODER_INPUT_DIM, CONV_IN_KERNEL,
        padding=CONV_IN_KERNEL // 2,
    )

    for block in blocks():
        prefix = f"blocks.{block.index}"
        hidden = add_snake(network, trt, hidden, weights[f"{prefix}.snake1.alpha"])
        hidden = add_deconv(
            network, trt, hidden, _fused(weights, f"{prefix}.conv_t1"),
            weights[f"{prefix}.conv_t1.bias"],
            block.input_dim, block.output_dim, block.stride,
        )
        for unit, dilation in enumerate(RESIDUAL_DILATIONS, start=1):
            hidden = add_residual_unit(
                network, trt, hidden, weights, f"{prefix}.res_unit{unit}",
                block.output_dim, dilation,
            )

    last = blocks()[-1].output_dim
    hidden = add_snake(network, trt, hidden, weights["snake_out.alpha"])
    hidden = add_conv(
        network, trt, hidden, _fused(weights, "conv_out"), weights["conv_out.bias"],
        OUTPUT_CHANNELS, last, CONV_OUT_KERNEL, padding=CONV_OUT_KERNEL // 2,
    )
    hidden = network.add_activation(hidden, trt.ActivationType.TANH).get_output(0)

    unfold = network.add_shuffle(hidden)
    unfold.reshape_dims = (1, STREAMS, waveform_samples(latent_length))
    return unfold.get_output(0)


def configure(config: Any, trt: Any) -> None:
    """Apply the builder settings this engine is validated under."""

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
    if DISABLE_TF32:
        config.clear_flag(trt.BuilderFlag.TF32)


def expected_input_shape(latent_length: int) -> tuple[int, ...]:
    return (1, LATENT_CHANNELS, latent_length)


def expected_output_shape(latent_length: int) -> tuple[int, ...]:
    return (1, STREAMS, waveform_samples(latent_length))
