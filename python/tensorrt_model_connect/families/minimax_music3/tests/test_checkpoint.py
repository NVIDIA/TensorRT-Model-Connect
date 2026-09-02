# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MiniMax-Music3 tensor inventory."""

from __future__ import annotations

import importlib

import pytest

checkpoint = importlib.import_module(
    "tensorrt_model_connect.families.minimax_music3.checkpoint"
)

CheckpointError = checkpoint.CheckpointError


def _condition_encoder() -> dict[str, tuple[int, ...]]:
    return {
        "layer_scale": (1,),
        "layer_weight_logits": (8,),
        "proj.bias": (2048,),
        "proj.weight": (2048, 4096, 3),
    }


def _rvq_depth_decoder() -> dict[str, tuple[int, ...]]:
    tensors: dict[str, tuple[int, ...]] = {
        "audio_embeddings.weight": (7168, 4096),
        "norm.weight": (4096,),
        "pos_embedding.weight": (16, 4096),
        "projection.weight": (4096, 4096),
    }
    for head in range(checkpoint.DEPTH_DECODER_HEADS):
        tensors[f"audio_heads.{head}.weight"] = (1024, 4096)
    for layer in range(4):
        for projection in ("to_q", "to_k", "to_v", "to_out"):
            tensors[f"layers.{layer}.attn.{projection}.weight"] = (4096, 4096)
        tensors[f"layers.{layer}.gate_proj.weight"] = (6144, 4096)
        tensors[f"layers.{layer}.up_proj.weight"] = (6144, 4096)
        tensors[f"layers.{layer}.down_proj.weight"] = (4096, 6144)
        tensors[f"layers.{layer}.input_layernorm.weight"] = (4096,)
        tensors[f"layers.{layer}.post_attention_layernorm.weight"] = (4096,)
    return tensors


def _vocoder() -> dict[str, tuple[int, ...]]:
    tensors: dict[str, tuple[int, ...]] = {
        "conv_in.bias": (1536,),
        "conv_in.weight_g": (1536, 1, 1),
        "conv_in.weight_v": (1536, 1024, 7),
        "conv_out.bias": (1,),
        "conv_out.weight_g": (1, 1, 1),
        "conv_out.weight_v": (1, 96, 7),
        "dec_in_proj.bias": (1024,),
        "dec_in_proj.weight": (1024, 64, 1),
        "snake_out.alpha": (1, 96, 1),
    }
    for block in range(4):
        tensors[f"blocks.{block}.conv_t1.bias"] = (768,)
        tensors[f"blocks.{block}.conv_t1.weight_g"] = (1536, 1, 1)
        tensors[f"blocks.{block}.conv_t1.weight_v"] = (1536, 768, 16)
        tensors[f"blocks.{block}.snake1.alpha"] = (1, 1536, 1)
        for unit in (1, 2, 3):
            base = f"blocks.{block}.res_unit{unit}"
            tensors[f"{base}.conv1.bias"] = (768,)
            tensors[f"{base}.conv1.weight_g"] = (768, 1, 1)
            tensors[f"{base}.conv1.weight_v"] = (768, 768, 7)
            tensors[f"{base}.conv2.bias"] = (768,)
            tensors[f"{base}.conv2.weight_g"] = (768, 1, 1)
            tensors[f"{base}.conv2.weight_v"] = (768, 768, 1)
            tensors[f"{base}.snake1.alpha"] = (1, 768, 1)
            tensors[f"{base}.snake2.alpha"] = (1, 768, 1)
    return tensors


def _transformer() -> dict[str, tuple[int, ...]]:
    return {
        "preprocess_conv.weight": (2304, 2304, 1),
        "proj_in.weight": (2048, 2304),
        "time_proj.weight": (128, 1),
        "time_embed.linear_1.weight": (2048, 256),
        "time_embed.linear_1.bias": (2048,),
        "time_embed.linear_2.weight": (2048, 2048),
        "time_embed.linear_2.bias": (2048,),
    }


BUILDERS = {
    "condition_encoder": _condition_encoder,
    "rvq_depth_decoder": _rvq_depth_decoder,
    "vocoder": _vocoder,
    "transformer": _transformer,
}


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_published_layout_validates(name: str) -> None:
    checkpoint.validate_component(name, BUILDERS[name]())


def test_published_vocoder_tensor_count() -> None:
    """The published vocoder header carries 121 tensors."""

    assert len(_vocoder()) == 121


def test_published_depth_decoder_tensor_count() -> None:
    """The published depth-decoder header carries 47 tensors."""

    assert len(_rvq_depth_decoder()) == 47


def test_depth_decoder_predicts_seven_of_the_eight_codebooks() -> None:
    tensors = _rvq_depth_decoder()
    heads = [t for t in tensors if t.startswith("audio_heads.")]

    assert len(heads) == checkpoint.DEPTH_DECODER_HEADS == 7
    # The embedding table covers exactly those seven codebooks.
    assert tensors["audio_embeddings.weight"][0] == 7 * 1024


def test_vocoder_decodes_one_channel_from_a_half_width_stream() -> None:
    tensors = _vocoder()

    assert tensors["dec_in_proj.weight"][1] == checkpoint.VOCODER_CHANNELS_PER_STREAM
    assert tensors["conv_out.weight_v"][0] == checkpoint.VOCODER_OUTPUT_CHANNELS
    # The component config's latent_channels is two of these streams.
    assert checkpoint.VOCODER_CHANNELS_PER_STREAM * 2 == 128


def test_missing_tensor_is_named() -> None:
    tensors = _condition_encoder()
    del tensors["proj.weight"]

    with pytest.raises(CheckpointError, match="proj.weight"):
        checkpoint.validate_component("condition_encoder", tensors)


def test_wrong_repeated_count_is_reported() -> None:
    tensors = _rvq_depth_decoder()
    del tensors["audio_heads.6.weight"]

    with pytest.raises(CheckpointError, match="expected 7"):
        checkpoint.validate_component("rvq_depth_decoder", tensors)


def test_wrong_shape_is_reported() -> None:
    tensors = _vocoder()
    tensors["conv_out.weight_v"] = (2, 96, 7)

    with pytest.raises(CheckpointError, match="expected"):
        checkpoint.validate_component("vocoder", tensors)


def test_unknown_component_is_rejected() -> None:
    with pytest.raises(CheckpointError, match="unknown"):
        checkpoint.validate_component("language_model", {})
