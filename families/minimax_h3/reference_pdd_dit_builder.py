# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental complete OmniRef DiT graph; trained validation is still required.

Retain the original owner's refiner, normalization, RoPE, modulation, residual
and feed-forward definitions. Select the learned VSA attention only for the
student's transformer blocks, followed by its eight precomposed output heads.
"""

from collections import Counter
from functools import partial
from pathlib import Path
import sys

import numpy as np
import tensorrt as trt

from . import dit_builder as base
from . import graph_ops as op
from . import reference_vsa as vsa
from .config import MiniMaxH3Config, Ref2VAPackedProfile
from .dit_builder import (
    _final_hidden,
    _native_builder,
    _packed_hidden,
    _rope_tables,
    _serialize,
    _transformer_block,
)
from .reference_pdd import fuse_output_projection
from .reference_pdd_builder import add_output_heads


def _head_banks(weights):
    banks = {}
    for name, prefix in [("video", "proj_out"), ("audio", "audio_proj_out")]:
        weight = np.asarray(weights[prefix + ".weight"], dtype=np.float32)
        bias = np.asarray(weights[prefix + ".bias"], dtype=np.float32)
        fused = [fuse_output_projection(weight, bias, name, step) for step in range(8)]
        banks[name] = {
            "weight": np.stack([w for w, _ in fused]),
            "bias": np.stack([b for _, b in fused]),
        }
    return banks


def _vsa_attention(
    network, hidden, weights, prefix, profile, rows, *, cos=None, sin=None, geometry
):
    if cos is None or sin is None:
        raise ValueError("OmniRef VSA requires the complete packed RoPE coordinates")
    q, k, value = op.fused_qkv(network, hidden, weights, prefix + ".attn")
    q = base._per_head_norm(network, q, weights[prefix + ".attn.norm_q.weight"], profile, rows)
    k = base._per_head_norm(network, k, weights[prefix + ".attn.norm_k.weight"], profile, rows)
    kwargs = dict(
        rows=rows, heads=profile.num_heads, head_dim=128, rotary_dim=6 * profile.rope_freq_dim
    )
    q = op.partial_rope(network, q, cos, sin, **kwargs)
    k = op.partial_rope(network, k, cos, sin, **kwargs)
    gate = op.linear(network, hidden, weights[prefix + ".attn.to_gate_compress.weight"])
    attended = vsa.add_packed_attention(
        network,
        {"q": q, "k": k, "v": value, "gate": gate},
        geometry["gather"],
        geometry["untile"],
        geometry["sizes"],
        geometry["regions"],
        plugin_library=geometry["plugin_library"],
        heads=profile.num_heads,
        max_tiles=geometry["max_tiles"],
    )
    return op.linear(network, attended, weights[prefix + ".attn.to_out.0.weight"])


def _geometry_inputs(network, plugin_library, max_tiles):
    geometry = {
        "gather": network.add_input("vsa_gather_indices", trt.int32, (-1,)),
        "untile": network.add_input("vsa_untile_indices", trt.int32, (-1,)),
        "sizes": network.add_input("vsa_block_sizes", trt.int32, (-1,)),
        "regions": network.add_input("vsa_regions", trt.int32, (-1, 3)),
        "plugin_library": plugin_library,
        "max_tiles": max_tiles,
    }
    selector = network.add_input("pdd_step", trt.int32, (1,))
    return geometry, selector


def _add_reference_profile(builder, config, profile, reference, max_tiles):
    optimization = builder.create_optimization_profile()
    packed = reference.sequence_rows(profile)
    shapes = {
        "encoder_hidden_states": tuple((n, profile.text_dim) for n in reference.text_rows),
        "video_hidden_states": tuple(
            (profile.video_rows + n, profile.video_patch_dim)
            for n in reference.condition_video_rows
        ),
        "audio_hidden_states": tuple(
            (profile.audio_rows + n, profile.audio_in_channels)
            for n in reference.condition_audio_rows
        ),
        "position_ids": tuple((n, 3) for n in packed),
        "vsa_gather_indices": tuple((n * 128,) for n in (2, min(1024, max_tiles), max_tiles)),
        "vsa_block_sizes": tuple((n,) for n in (2, min(1024, max_tiles), max_tiles)),
        "vsa_regions": ((1, 3), (1, 3), (2, 3)),
        "pdd_step": ((1,), (1,), (1,)),
    }
    for name in ("packed_indices", "adaln_indices", "timestep_indices", "vsa_untile_indices"):
        shapes[name] = tuple((n,) for n in packed)
    for name, bounds in shapes.items():
        optimization.set_shape(name, *bounds)
    config.add_optimization_profile(optimization)


def _validate_student_network(network, profile):
    counts = Counter(network.get_layer(i).type for i in range(network.num_layers))
    if (
        counts[trt.LayerType.PLUGIN_V2] != profile.num_layers
        or counts[trt.LayerType.ATTENTION_INPUT] != profile.num_refiner_layers
        or counts[trt.LayerType.ATTENTION_OUTPUT] != profile.num_refiner_layers
        or any(
            counts[k]
            for k in (trt.LayerType.PLUGIN, trt.LayerType.PLUGIN_V3, trt.LayerType.DIST_COLLECTIVE)
        )
    ):
        raise RuntimeError("OmniRef graph requires only its VSA bindings and dense text refiner")


def build_engine(
    weights: dict,
    profile: MiniMaxH3Config,
    *,
    plugin_library,
    max_vsa_tiles=2048,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    reference_profile: Ref2VAPackedProfile | None = None,
    output=None,
) -> bytes | None:
    """Build the explicit experimental eight-step student's complete DiT plan."""

    profile.validate()
    if reference_profile is None or profile.head_dim != 128:
        raise ValueError("OmniRef requires a reference profile and 128-channel attention heads")
    if (
        type(max_vsa_tiles) is not int
        or not 2 <= max_vsa_tiles <= vsa.MAX_TILES
        or max_vsa_tiles % 2
    ):
        raise ValueError("OmniRef VSA tile bound must be a positive even supported capacity")
    if not Path(plugin_library).is_file():
        raise FileNotFoundError("OmniRef requires its compiled TVM-FFI plugin bridge")
    banks = _head_banks(weights)
    if profile.first_block_cache:
        raise ValueError("MiniMax-H3 first_block_cache profile requires the split DiT builders")
    if reference_profile is not None:
        reference_profile.validate(profile)
    rows = -1
    logger, builder, network, config = _native_builder(verbose, workspace_bytes)
    try:
        video = network.add_input(
            "video_hidden_states",
            trt.float32,
            (-1 if reference_profile is not None else profile.video_rows, profile.video_patch_dim),
        )
        audio = network.add_input(
            "audio_hidden_states",
            trt.float32,
            (
                -1 if reference_profile is not None else profile.audio_rows,
                profile.audio_in_channels,
            ),
        )
        text = network.add_input("encoder_hidden_states", trt.float32, (-1, profile.text_dim))
        positions = network.add_input("position_ids", trt.float32, (-1, 3))
        adaln_indices = network.add_input("adaln_indices", trt.int32, (-1,))
        timestep_indices = network.add_input("timestep_indices", trt.int32, (-1,))
        packed_indices, target_video_indices, target_audio_indices = None, None, None
        packed_indices = network.add_input("packed_indices", trt.int32, (-1,))
        target_video_indices = network.add_input(
            "target_video_indices", trt.int32, (profile.video_rows,)
        )
        target_audio_indices = network.add_input(
            "target_audio_indices", trt.int32, (profile.audio_rows,)
        )
        geometry, selector = _geometry_inputs(network, plugin_library, max_vsa_tiles)
        _add_reference_profile(builder, config, profile, reference_profile, max_vsa_tiles)
        block_modulations = [
            network.add_input(
                f"block_modulation_{index}",
                trt.bfloat16,
                (profile.adaln_table_rows, 6, profile.hidden_size),
            )
            for index in range(profile.num_layers)
        ]
        final_modulation = network.add_input(
            "final_modulation", trt.bfloat16, (profile.max_timestep_count, 2, profile.hidden_size)
        )
        # Ref2VA interleaves reference operands; the original text-only path uses
        # its identity text | audio | video packing. Both stay native TRT graphs.
        hidden = _packed_hidden(
            network, video, audio, text, weights, profile, packed_indices=packed_indices
        )

        cos, sin = _rope_tables(network, positions, profile, rows)
        # The dynamic packed sequence contains live rows only, like Diffusers, so
        # its attention mask remains None for every supported prompt length.
        for index in range(profile.num_layers):
            hidden = _transformer_block(
                network,
                hidden,
                block_modulations[index],
                adaln_indices,
                cos,
                sin,
                weights,
                profile,
                index,
                attention=partial(_vsa_attention, geometry=geometry),
            )

        hidden = _final_hidden(
            network, hidden, timestep_indices, final_modulation, weights, profile
        )
        targets = {
            "video": op.gather_rows(network, hidden, target_video_indices),
            "audio": op.gather_rows(network, hidden, target_audio_indices),
        }
        for output in add_output_heads(network, targets, selector, banks).values():
            network.mark_output(output)
        _validate_student_network(network, profile)

        packed_rows = (
            (profile.min_sequence_length, profile.opt_sequence_length, profile.sequence_length)
            if reference_profile is None
            else reference_profile.sequence_rows(profile)
        )
        print(
            f"[minimax-h3] building OmniRef VSA DiT: layers={profile.num_layers}, "
            f"packed={packed_rows[0]}..{packed_rows[2]} "
            f"(opt={packed_rows[1]}), devices=1",
            file=sys.stderr,
        )
        return _serialize(
            logger=logger,
            builder=builder,
            network=network,
            config=config,
            weights=weights,
            consume_weights=consume_weights,
            label="OmniRef PDD VSA DiT",
            output=output,
        )
    finally:
        op.release_weight_buffers(network)
        if consume_weights:
            weights.clear()
