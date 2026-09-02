# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest


def _configure_trt():
    from tensorrt_model_connect import trt_compat

    if trt_compat.is_available("tensorrt"):
        return trt_compat
    if trt_compat.is_available("tensorrt_rtx"):
        trt_compat.configure_backend(rtx=True)
        return trt_compat
    pytest.skip("TensorRT or TensorRT-RTX bindings are unavailable")


def test_audio_encoder_public_profile_and_partition() -> None:
    _configure_trt()
    from tensorrt_model_connect.families.minimax_h3.ref2va_audio_encoder_builder import (
        DEFAULT_REF2VA_AUDIO_ENCODER_PROFILE,
        build_ref2va_audio_encoder_engine,
        checkpoint_keys,
        ref2va_audio_encoder_abi,
    )

    profile = DEFAULT_REF2VA_AUDIO_ENCODER_PROFILE
    profile.validate()
    assert profile.sample_profile == (64_000, 165_600, 480_000)
    assert profile.latent_profile == (80, 207, 600)
    assert len(checkpoint_keys()) == len(set(checkpoint_keys())) == 171
    abi = ref2va_audio_encoder_abi(profile)
    assert abi.filename == "ref2va_audio_vae_encoder.plan"
    assert abi.inputs[0].name == "audio_samples"
    assert abi.outputs[0].name == "posterior_mean"
    assert abi.outputs[0].max_shape == (2, 32, 600)
    with pytest.raises(ValueError, match="audio encoder checkpoint partition mismatch"):
        build_ref2va_audio_encoder_engine({})


def test_temporal_video_encoder_abi_reuses_visual_partition() -> None:
    _configure_trt()
    from tensorrt_model_connect.families.minimax_h3.fl2va_vae_encoder_builder import (
        checkpoint_keys as visual_checkpoint_keys,
    )
    from tensorrt_model_connect.families.minimax_h3.ref2va_video_encoder_builder import (
        build_ref2va_video_encoder_engine,
        checkpoint_keys,
        ref2va_video_encoder_abi,
    )

    assert checkpoint_keys() == visual_checkpoint_keys()
    assert len(checkpoint_keys()) == len(set(checkpoint_keys())) == 118
    abi = ref2va_video_encoder_abi()
    assert abi.filename == "ref2va_video_vae_encoder.plan"
    assert abi.inputs[0].opt_shape == (1, 3, 17, 256, 256)
    assert abi.outputs[0].opt_shape == (1, 48, 5, 16, 16)
    with pytest.raises(ValueError, match="video encoder checkpoint partition mismatch"):
        build_ref2va_video_encoder_engine({})


def test_ref2va_qwen_builders_delegate_to_shared_graphs_with_superset_profiles(
    monkeypatch,
) -> None:
    _configure_trt()
    from tensorrt_model_connect.families.minimax_h3 import (
        multimodal_text_encoder_builder,
        multimodal_vision_builder,
    )
    from tensorrt_model_connect.families.minimax_h3.ref2va_qwen_builder import (
        build_ref2va_shared_text_encoder_engine,
        build_ref2va_shared_vision_encoder_engine,
    )

    calls = []

    def fake_builder(weights, profile, **kwargs):
        calls.append((weights, profile, kwargs))
        return b"shared-plan"

    monkeypatch.setattr(
        multimodal_vision_builder,
        "build_multimodal_vision_encoder_engine",
        fake_builder,
    )
    monkeypatch.setattr(
        multimodal_text_encoder_builder,
        "build_multimodal_text_encoder_engine",
        fake_builder,
    )
    marker = {"one": np.zeros((1,), dtype=np.float32)}
    assert build_ref2va_shared_vision_encoder_engine(marker) == b"shared-plan"
    assert build_ref2va_shared_text_encoder_engine(marker) == b"shared-plan"
    assert calls[0][0] is calls[1][0] is marker
    assert calls[0][1].max_patches == 65_536
    assert calls[1][1].max_sequence_length == 262_144
    assert calls[1][1].max_vision_rows == 262_144


def test_temporal_causal_conv_and_isolated_group_norm_micrograph_serializes() -> None:
    trt_compat = _configure_trt()
    from tensorrt_model_connect.families.minimax_h3 import graph_ops as op
    from tensorrt_model_connect.families.minimax_h3.ref2va_video_encoder_builder import (
        _conv3d,
        _group_norm_isolated,
    )

    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 20)
    hidden = network.add_input("pixels", trt.float32, (1, 32, 3, 8, 8))
    weights = {
        "norm.weight": np.ones((32,), dtype=np.float32),
        "norm.bias": np.zeros((32,), dtype=np.float32),
        "conv.weight": np.full((32, 32, 3, 3, 3), 1.0e-4, dtype=np.float32),
        "conv.bias": np.zeros((32,), dtype=np.float32),
    }
    hidden = _group_norm_isolated(
        network,
        hidden,
        weights,
        "norm",
        channels=32,
        frames=3,
        height=8,
        width=8,
    )
    hidden, frames, height, width = _conv3d(
        network,
        hidden,
        weights,
        "conv",
        in_channels=32,
        out_channels=32,
        frames=3,
        height=8,
        width=8,
        kernel_size=3,
        symmetric_reflect=1,
    )
    assert (frames, height, width) == (3, 8, 8)
    hidden.name = "encoded"
    network.mark_output(hidden)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None and len(bytes(plan)) > 0
    op.release_weight_buffers(network)
