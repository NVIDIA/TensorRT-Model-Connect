# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor inventory for the MiniMax-Music3 components.

Read from the published safetensors headers at revision
``fbdf52fbaaca799592917417eb05f1899f1255ec``. Each entry names the tensors a
component must carry, so a checkpoint missing a module is rejected before any
engine is built rather than producing an engine with silently absent weights.

Two shapes here contradict a plain reading of the component configs, and both
change what a builder has to do:

**Stereo comes from a mono decoder.** ``vocoder`` declares
``latent_channels = 128`` but its first layer, ``dec_in_proj``, takes 64
channels, and its last, ``conv_out``, emits 1. The 128 latent channels are two
64-channel streams decoded through the same weights, one per output channel.

**The depth decoder predicts seven codebooks, not eight.** ``rvq_depth_decoder``
declares ``num_codebooks = 8`` and carries ``audio_heads.0`` through
``audio_heads.6``, with ``audio_embeddings`` sized ``7 * 1024 = 7168``. The
first codebook comes from the global language model; the depth decoder predicts
the remaining seven.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

#: Number of residual codebooks the depth decoder itself predicts. The
#: component config's ``num_codebooks`` counts the language model's first
#: codebook as well.
DEPTH_DECODER_HEADS = 7

#: The vocoder decodes one audio channel at a time; the latent carries two.
VOCODER_OUTPUT_CHANNELS = 1
VOCODER_CHANNELS_PER_STREAM = 64


class CheckpointError(ValueError):
    """Raised when a component's tensors do not match the published layout."""


@dataclass(frozen=True)
class ComponentTensors:
    """Tensors one component must carry."""

    name: str
    #: Exact tensor names, present exactly once.
    exact: tuple[str, ...] = ()
    #: ``(regex, expected_count)`` for the repeated per-layer tensors.
    repeated: tuple[tuple[str, int], ...] = ()
    #: Tensor name to expected shape, for the shapes a builder depends on.
    shapes: Mapping[str, tuple[int, ...]] = field(default_factory=dict)


CONDITION_ENCODER = ComponentTensors(
    name="condition_encoder",
    exact=("layer_scale", "layer_weight_logits", "proj.bias", "proj.weight"),
    shapes={
        # A learned weighting over the eight per-frame hidden streams -- one
        # from the global language model and seven from the depth decoder,
        # concatenated in `encoders.py` as `cat((last_hidden, depth_hidden))`
        # -- then a Conv1d from the language model's width to the
        # transformer's condition width. The component config calls these
        # `num_condition_layers`, which names codebook streams, not layers.
        "layer_weight_logits": (8,),
        "proj.weight": (2048, 4096, 3),
        "layer_scale": (1,),
    },
)

RVQ_DEPTH_DECODER = ComponentTensors(
    name="rvq_depth_decoder",
    exact=(
        "audio_embeddings.weight",
        "norm.weight",
        "pos_embedding.weight",
        "projection.weight",
    ),
    repeated=(
        (r"^audio_heads\.\d+\.weight$", DEPTH_DECODER_HEADS),
        (r"^layers\.\d+\.attn\.to_[qkv]\.weight$", 12),
        (r"^layers\.\d+\.attn\.to_out\.weight$", 4),
        (r"^layers\.\d+\.(gate|up)_proj\.weight$", 8),
        (r"^layers\.\d+\.down_proj\.weight$", 4),
        (r"^layers\.\d+\.(input|post_attention)_layernorm\.weight$", 8),
    ),
    shapes={
        "audio_embeddings.weight": (DEPTH_DECODER_HEADS * 1024, 4096),
        "audio_heads.0.weight": (1024, 4096),
        "pos_embedding.weight": (16, 4096),
        "projection.weight": (4096, 4096),
    },
)

VOCODER = ComponentTensors(
    name="vocoder",
    exact=(
        "conv_in.bias",
        "conv_in.weight_g",
        "conv_in.weight_v",
        "conv_out.bias",
        "conv_out.weight_g",
        "conv_out.weight_v",
        "dec_in_proj.bias",
        "dec_in_proj.weight",
        "snake_out.alpha",
    ),
    repeated=(
        (r"^blocks\.\d+\.conv_t1\.(bias|weight_g|weight_v)$", 12),
        (r"^blocks\.\d+\.snake1\.alpha$", 4),
        (r"^blocks\.\d+\.res_unit[123]\.conv[12]\.(bias|weight_g|weight_v)$", 72),
        (r"^blocks\.\d+\.res_unit[123]\.snake[12]\.alpha$", 24),
    ),
    shapes={
        # One stream in, one audio channel out.
        "dec_in_proj.weight": (1024, VOCODER_CHANNELS_PER_STREAM, 1),
        "conv_in.weight_v": (1536, 1024, 7),
        "conv_out.weight_v": (VOCODER_OUTPUT_CHANNELS, 96, 7),
    },
)

TRANSFORMER = ComponentTensors(
    name="transformer",
    exact=(
        "preprocess_conv.weight",
        "proj_in.weight",
        "time_proj.weight",
        "time_embed.linear_1.weight",
        "time_embed.linear_1.bias",
        "time_embed.linear_2.weight",
        "time_embed.linear_2.bias",
    ),
    shapes={
        "proj_in.weight": (2048, 2304),
        "preprocess_conv.weight": (2304, 2304, 1),
        # fourier_embedding_dim 256 in, transformer width out.
        "time_embed.linear_1.weight": (2048, 256),
        "time_embed.linear_2.weight": (2048, 2048),
    },
)

COMPONENTS: tuple[ComponentTensors, ...] = (
    CONDITION_ENCODER,
    RVQ_DEPTH_DECODER,
    VOCODER,
    TRANSFORMER,
)

_BY_NAME = {component.name: component for component in COMPONENTS}


def component(name: str) -> ComponentTensors:
    """Return the expected tensor layout of one component."""

    try:
        return _BY_NAME[name]
    except KeyError:
        raise CheckpointError(f"unknown MiniMax-Music3 component {name!r}") from None


def validate_component(name: str, tensors: Mapping[str, Iterable[int]]) -> None:
    """Check one component's tensor names and depended-on shapes.

    ``tensors`` maps tensor name to shape, as read from a safetensors header.
    """

    spec = component(name)
    present = set(tensors)

    missing = [tensor for tensor in spec.exact if tensor not in present]
    if missing:
        raise CheckpointError(
            f"{name} is missing {len(missing)} tensor(s): {', '.join(sorted(missing))}"
        )

    for pattern, expected in spec.repeated:
        matched = sum(1 for tensor in present if re.match(pattern, tensor))
        if matched != expected:
            raise CheckpointError(
                f"{name} has {matched} tensors matching {pattern!r}, expected {expected}"
            )

    for tensor, expected_shape in spec.shapes.items():
        actual = tuple(int(dim) for dim in tensors[tensor])
        if actual != expected_shape:
            raise CheckpointError(
                f"{name}.{tensor} has shape {actual}, expected {expected_shape}"
            )
