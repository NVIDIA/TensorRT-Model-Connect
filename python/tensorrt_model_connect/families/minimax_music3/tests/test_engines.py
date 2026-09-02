# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the engine inventory and the runtime facts the bundle carries."""

from __future__ import annotations

import importlib

import pytest

np = pytest.importorskip("numpy")

eng = importlib.import_module(
    "tensorrt_model_connect.families.minimax_music3.engines"
)
parity = importlib.import_module(
    "tensorrt_model_connect.families.minimax_music3.parity"
)
spec = importlib.import_module(
    "tensorrt_model_connect.families.minimax_music3.pipeline_spec"
)


def test_all_five_networks_are_built_here() -> None:
    """Including the language model: the architecture asks for duplication."""

    assert len(eng.ENGINE_NAMES) == 5
    assert eng.LANGUAGE_MODEL_ENGINE in eng.ENGINE_NAMES
    assert set(eng.ENGINE_NAMES) == set(eng.ENGINE_COMPONENTS)


def test_engines_are_listed_in_generation_order() -> None:
    """The language model runs first; the vocoder last."""

    assert eng.ENGINE_NAMES == (
        eng.LANGUAGE_MODEL_ENGINE,
        eng.DEPTH_DECODER_ENGINE,
        eng.CONDITION_ENCODER_ENGINE,
        eng.DIT_ENGINE,
        eng.VOCODER_ENGINE,
    )


def test_language_model_io_is_a_cached_decode_step() -> None:
    """The engine decodes one position at a time against its cache.

    An earlier revision asked it for a fixed-length stack of hidden states,
    which no generation can drive: there is nothing to feed token ids to and
    nowhere for the prefix to live.
    """

    from tensorrt_model_connect.families.minimax_music3 import language_model_engine as lme

    io = eng.engine_io(eng.LANGUAGE_MODEL_ENGINE, latent_length=689, steps=64)

    assert io[lme.TOKEN_INPUT] == (1,)
    assert io[lme.POSITION_INPUT] == (1,)
    assert io[lme.LOGITS_OUTPUT] == (1, 200000)
    assert io[lme.HIDDEN_STATES_OUTPUT] == (1, 4096)
    # One cache pair per layer, each spanning the compiled cache length.
    assert io["cache_k_0"] == (1, lme.DEFAULT_MAX_CACHE_LENGTH, 1024)
    assert io["present_v_35"] == (1, lme.DEFAULT_MAX_CACHE_LENGTH, 1024)


def test_every_engine_names_a_checkpoint_component() -> None:
    plugin_mod = importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.plugin"
    )

    for name in eng.ENGINE_NAMES:
        assert eng.engine_component(name) in plugin_mod.REQUIRED_COMPONENTS


def test_unknown_engine_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown MiniMax-Music3 engine"):
        eng.engine_component("vae")


def test_engine_io_shapes_agree_with_the_recorded_run() -> None:
    latent = parity.BASELINE_LATENT_SHAPE[2]

    condition = eng.engine_io(eng.CONDITION_ENCODER_ENGINE, latent_length=latent)
    assert condition["condition"] == (1, latent, 2048)

    transformer = eng.engine_io(eng.DIT_ENGINE, latent_length=latent)
    assert transformer["latents"] == parity.BASELINE_LATENT_SHAPE
    assert transformer["velocity"] == parity.BASELINE_LATENT_SHAPE

    vocoder_io = eng.engine_io(eng.VOCODER_ENGINE, latent_length=latent)
    assert vocoder_io["latents"] == parity.BASELINE_LATENT_SHAPE
    assert vocoder_io["waveform"] == (1, 2, latent * 512)


def test_depth_decoder_io_carries_the_whole_frame() -> None:
    """Eight codes in, seven codebooks' logits and states plus the embedding.

    The eighth code never re-enters the sequence -- there is no position after
    the last -- but the frame embedding reads it, which is why the input is
    eight wide rather than seven.
    """

    from tensorrt_model_connect.families.minimax_music3 import depth_decoder_engine as dde

    io = eng.engine_io(eng.DEPTH_DECODER_ENGINE, latent_length=689, steps=8)

    assert io[dde.HIDDEN_INPUT] == (1, 1, 4096)
    assert io[dde.CODES_INPUT] == (1, 8)
    assert io[dde.LOGITS_OUTPUT] == (1, 7, 1024)
    assert io[dde.HIDDEN_OUTPUT] == (1, 7, 4096)
    assert io[dde.FRAME_EMBED_OUTPUT] == (1, 1, 4096)


def test_latent_length_for_a_window() -> None:
    assert eng.latent_length_for(spec.CHUNK_FRAMES) == 689


def test_stitched_samples_reproduce_the_recorded_run() -> None:
    """Four windows of 689 latent frames come to 882688 samples."""

    assert eng.samples_for(4, 689) == parity.BASELINE_WAVEFORM_SAMPLES


def test_a_single_window_is_not_cropped() -> None:
    assert eng.samples_for(1, 689) == 689 * 512


def test_samples_rejects_an_empty_generation() -> None:
    with pytest.raises(ValueError, match="chunk_count must be positive"):
        eng.samples_for(0, 689)


def test_bundle_config_carries_what_the_checkpoint_does_not() -> None:
    overrides = eng.bundle_config_overrides()

    # The rate the pipeline's property returns, not the model card's prose.
    assert overrides["sampling_rate"] == 44100
    assert overrides["output_channels"] == 2
    # The window plan and the crops, which live only in the reference.
    assert overrides["chunk_frames"] == 200
    assert overrides["chunk_hop"] == 100
    assert (overrides["crop_left_latent"] + overrides["crop_right_latent"]
            == spec.OVERLAP_LATENT_FRAMES)
    # Guidance doubles every transformer evaluation.
    assert overrides["guidance_branches"] == 2


def test_bundle_config_codebook_split() -> None:
    overrides = eng.bundle_config_overrides()

    assert overrides["num_codebooks"] == 8
    assert overrides["num_residual_codebooks"] == 7
    assert overrides["num_codebooks"] - overrides["num_residual_codebooks"] == 1


def test_bundle_config_values_are_all_plain_scalars() -> None:
    """The runtime reads these through a JSON bundle section."""

    for key, value in eng.bundle_config_overrides().items():
        assert isinstance(value, (int, float)), key
