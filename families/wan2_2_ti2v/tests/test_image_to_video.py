# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image-to-video and portrait-profile contracts of the Wan2.2 TI2V build."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from families.wan2_2_ti2v import dit_builder, model, vae_encoder_builder
from families.wan2_2_ti2v.model_config import (
    SUPPORTED_GENERATION_PROFILES,
    WAN22_TI2V_5B,
    WAN22_TI2V_5B_L0,
    WAN22_TI2V_5B_PORTRAIT,
    select_generation_profile,
)
from families.wan2_2_ti2v.vae_step_builder import Wan22VaeStepProfile


def test_portrait_profile_is_the_official_profile_turned_upright() -> None:
    raw = {"video_width": 704, "video_height": 1280}
    assert select_generation_profile(raw) is WAN22_TI2V_5B_PORTRAIT
    assert WAN22_TI2V_5B_PORTRAIT in SUPPORTED_GENERATION_PROFILES
    assert (WAN22_TI2V_5B_PORTRAIT.latent_height, WAN22_TI2V_5B_PORTRAIT.latent_width) == (80, 44)
    assert WAN22_TI2V_5B_PORTRAIT.num_patches == WAN22_TI2V_5B.num_patches == 31 * 880
    assert WAN22_TI2V_5B_PORTRAIT.first_frame_patches == WAN22_TI2V_5B.first_frame_patches == 880
    assert WAN22_TI2V_5B_L0.first_frame_patches == 12 * 21


def test_build_names_its_tasks_before_loading_weights() -> None:
    request = SimpleNamespace(
        dynamic_kv_cache=False,
        max_sequence_length=None,
        context_parallel_size=1,
        task="text_generation",
    )
    with pytest.raises(
        ValueError, match="image_generation, text_to_video, initial_image_text_to_video"
    ):
        model.build(request, Mock())


def test_dit_first_frame_tokens_take_their_own_timestep() -> None:
    constants = []
    network = Mock()

    def add_constant(shape, values):
        constants.append(np.asarray(values))
        return Mock()

    network.add_constant.side_effect = add_constant
    dit_builder._TokenSelect(network, WAN22_TI2V_5B_L0)
    first, later = constants
    tokens = WAN22_TI2V_5B_L0.num_patches
    assert first.shape == later.shape == (tokens, 1)
    assert first[:, 0].tolist() == [1.0] * 252 + [0.0] * (tokens - 252)
    assert (first + later == 1.0).all()  # an exact 0/1 select, never a blend


def test_vae_encoder_takes_only_the_encoder_and_its_quant_convolution(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    state = {
        "encoder.conv1.weight": torch.ones(2, dtype=torch.bfloat16),
        "conv1.bias": torch.zeros(1),
        "decoder.conv1.weight": torch.ones(1),
        "conv2.weight": torch.ones(1),
    }
    monkeypatch.setattr(vae_encoder_builder, "load_native_vae_state_dict", lambda path: state)
    weights = vae_encoder_builder.load_vae_encoder_weights("unused")
    assert sorted(weights) == ["conv1.bias", "encoder.conv1.weight"]
    assert all(value.dtype == np.float32 and value.flags.c_contiguous for value in weights.values())


def test_vae_encoder_is_qualified_only_for_the_generation_profiles() -> None:
    with pytest.raises(ValueError, match="qualified only"):
        vae_encoder_builder.build_vae_encoder_engine({}, profile=Wan22VaeStepProfile(44, 44))


def test_vae_encoder_patchify_matches_upstream_rearrange(monkeypatch) -> None:
    einops = pytest.importorskip("einops")
    monkeypatch.setattr(vae_encoder_builder.trt, "Permutation", tuple)
    layers = []
    network = Mock()

    def add_shuffle(tensor):
        layer = SimpleNamespace(reshape_dims=None, second_transpose=None)
        layer.get_output = lambda index: layer
        layers.append(layer)
        return layer

    network.add_shuffle.side_effect = add_shuffle
    vae_encoder_builder._patchify(network, Mock(), height=4, width=6)
    image = np.random.default_rng(0).normal(size=(1, 3, 1, 4, 6))
    factored, patched = layers
    actual = image.reshape(factored.reshape_dims).transpose(factored.second_transpose)
    actual = actual.reshape(patched.reshape_dims)
    expected = einops.rearrange(image, "b c f (h q) (w r) -> b (c r q) f h w", q=2, r=2)
    np.testing.assert_array_equal(actual, expected)
