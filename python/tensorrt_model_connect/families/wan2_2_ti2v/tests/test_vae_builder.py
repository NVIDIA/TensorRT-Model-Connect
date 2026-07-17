# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the Wan2.2 TI2V-5B VAE TensorRT builder."""

from __future__ import annotations

import pytest

from tensorrt_model_connect.families.wan2_2_ti2v.vae_builder import (
    OFFICIAL_VAE_DECODER_PROFILE,
    OFFICIAL_VAE_WORKSPACE_GIB,
    VAE_BARRIER_MANIFEST,
    Wan22VaeDecoderProfile,
    _configure_builder_config,
    _format_barrier_target,
    _format_final_rms_norm_names,
    _format_rms_norm_names,
    _format_up_block3_conv_names,
    _validate_network_contract,
)


def test_official_vae_decoder_profile() -> None:
    assert OFFICIAL_VAE_DECODER_PROFILE.input_shape == (1, 48, 31, 44, 80)
    assert OFFICIAL_VAE_DECODER_PROFILE.output_shape == (1, 3, 121, 704, 1280)
    assert OFFICIAL_VAE_WORKSPACE_GIB == 64


def test_official_vae_barrier_manifest() -> None:
    frame = OFFICIAL_VAE_DECODER_PROFILE.latent_frames - 1
    assert [_format_barrier_target(spec, frame) for spec in VAE_BARRIER_MANIFEST[:20]] == [
        "/decoder/mid_block/resnets.0_30/Add_output_0",
        "/decoder/mid_block/attentions.0_30/Add_1_output_0",
        "/decoder/mid_block/resnets.1_30/Add_output_0",
        "/decoder/up_blocks.0/resnets.0_30/Add_output_0",
        "/decoder/up_blocks.0/resnets.1_30/Add_output_0",
        "/decoder/up_blocks.0/resnets.2_30/Add_output_0",
        "/decoder/up_blocks.0_30/Add_output_0",
        "/decoder/up_blocks.1/resnets.0_30/Add_output_0",
        "/decoder/up_blocks.1/resnets.1_30/Add_output_0",
        "/decoder/up_blocks.1/resnets.2_30/Add_output_0",
        "/decoder/up_blocks.1_30/Add_output_0",
        "/decoder/up_blocks.2/resnets.0_30/Add_output_0",
        "/decoder/up_blocks.2/resnets.1_30/Add_output_0",
        "/decoder/up_blocks.2/resnets.2_30/Add_output_0",
        "/decoder/up_blocks.2_30/Add_output_0",
        "/decoder/up_blocks.3/resnets.0_30/Add_output_0",
        "/decoder/up_blocks.3/resnets.1_30/Add_output_0",
        "/decoder/up_blocks.3/resnets.2_30/Add_output_0",
        "/decoder/nonlinearity_30/Mul_output_0",
        "/decoder/conv_out_30/Conv_output_0",
    ]
    assert len(VAE_BARRIER_MANIFEST) == 104
    assert len({spec.label_template for spec in VAE_BARRIER_MANIFEST}) == 104
    assert all(spec.reason for spec in VAE_BARRIER_MANIFEST)
    assert [_format_barrier_target(spec, frame) for spec in VAE_BARRIER_MANIFEST[20:26]] == [
        "/decoder/mid_block/resnets.0/conv1_30/Concat_output_0",
        "/decoder/mid_block/resnets.0/conv1_30/Pad_output_0",
        "/decoder/mid_block/resnets.0/conv1_30/Conv_output_0",
        "/decoder/mid_block/resnets.0/conv2_30/Concat_output_0",
        "/decoder/mid_block/resnets.0/conv2_30/Pad_output_0",
        "/decoder/mid_block/resnets.0/conv2_30/Conv_output_0",
    ]
    assert _format_barrier_target(VAE_BARRIER_MANIFEST[-1], frame) == (
        "/decoder/up_blocks.3/resnets.2/conv2_30/Conv_output_0"
    )


def test_official_vae_barrier_manifest_covers_every_unrolled_frame() -> None:
    profile = OFFICIAL_VAE_DECODER_PROFILE
    targets = [
        _format_barrier_target(spec, frame)
        for frame in range(profile.latent_frames)
        for spec in VAE_BARRIER_MANIFEST
    ]
    labels = [
        spec.label_template.format(frame=frame)
        for frame in range(profile.latent_frames)
        for spec in VAE_BARRIER_MANIFEST
    ]

    assert len(targets) == profile.latent_frames * len(VAE_BARRIER_MANIFEST) == 3224
    assert len(set(targets)) == len(targets)
    assert len(set(labels)) == len(labels)
    assert targets[0] == "/decoder/mid_block/resnets.0/Add_output_0"
    assert targets[19] == "/decoder/conv_out/Conv_output_0"
    assert targets[20] == "/decoder/mid_block/resnets.0/nonlinearity/Mul_output_0"
    assert targets[103] == "/decoder/up_blocks.3/resnets.2/conv2/Conv_output_0"
    assert targets[104] == "/decoder/mid_block/resnets.0_1/Add_output_0"
    assert targets[-1] == "/decoder/up_blocks.3/resnets.2/conv2_30/Conv_output_0"


def test_official_vae_rms_norm_replacement_names() -> None:
    assert _format_rms_norm_names("mid_block/resnets.0", 1, 0) == (
        "/decoder/mid_block/resnets.0/norm1/Add_output_0",
        "/decoder/mid_block/resnets.0/norm1/ReduceL2",
        "vae.decoder.mid_block.resnets.0.norm1.gamma",
    )


def test_native_conv3d_replacement_names_are_exactly_final_up_block() -> None:
    assert _format_up_block3_conv_names(0, 1, 0) == (
        "/decoder/up_blocks.3/resnets.0/conv1/Conv",
        "/decoder/up_blocks.3/resnets.0/conv1/Conv_output_0",
        "vae.decoder.up_blocks.3.resnets.0.conv1.weight",
        "vae.decoder.up_blocks.3.resnets.0.conv1.bias",
    )
    assert _format_up_block3_conv_names(2, 2, 30) == (
        "/decoder/up_blocks.3/resnets.2/conv2_30/Conv",
        "/decoder/up_blocks.3/resnets.2/conv2_30/Conv_output_0",
        "vae.decoder.up_blocks.3.resnets.2.conv2.weight",
        "vae.decoder.up_blocks.3.resnets.2.conv2.bias",
    )


@pytest.mark.parametrize(
    ("resnet", "conv", "frame", "message"),
    [
        (3, 1, 0, "resnet"),
        (0, 3, 0, "convolution"),
        (0, 1, -1, "frame"),
    ],
)
def test_native_conv3d_replacement_names_reject_broader_scope(
    resnet: int, conv: int, frame: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _format_up_block3_conv_names(resnet, conv, frame)
    assert _format_rms_norm_names("up_blocks.3/resnets.2", 2, 30) == (
        "/decoder/up_blocks.3/resnets.2/norm2_30/Add_output_0",
        "/decoder/up_blocks.3/resnets.2/norm2_30/ReduceL2",
        "vae.decoder.up_blocks.3.resnets.2.norm2.gamma",
    )
    assert _format_final_rms_norm_names(30) == (
        "/decoder/norm_out_30/Add_output_0",
        "/decoder/norm_out_30/ReduceL2",
        "vae.decoder.norm_out.gamma",
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"latent_frames": 0, "latent_height": 44, "latent_width": 80}, "latent_frames"),
        ({"latent_frames": 31, "latent_height": 0, "latent_width": 80}, "latent_height"),
        ({"latent_frames": 31, "latent_height": 44, "latent_width": 0}, "latent_width"),
        (
            {"latent_frames": 31, "latent_height": 44, "latent_width": 80, "batch_size": 2},
            "batch size 1",
        ),
        (
            {
                "latent_frames": 31,
                "latent_height": 44,
                "latent_width": 80,
                "latent_channels": 16,
            },
            "48 latent channels",
        ),
    ],
)
def test_vae_decoder_profile_rejects_invalid_contract(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Wan22VaeDecoderProfile(**kwargs)


class _FakeBuilderFlag:
    TF32 = "tf32"


class _FakeMemoryPoolType:
    WORKSPACE = "workspace"


class _FakeTrt:
    BuilderFlag = _FakeBuilderFlag
    MemoryPoolType = _FakeMemoryPoolType
    float32 = "fp32"


class _FakeBuilderConfig:
    def __init__(self) -> None:
        self.flags = {_FakeBuilderFlag.TF32}
        self.pool_limits: list[tuple[str, int]] = []

    def set_memory_pool_limit(self, pool: str, size: int) -> None:
        self.pool_limits.append((pool, size))

    def clear_flag(self, flag: str) -> None:
        self.flags.discard(flag)

    def get_flag(self, flag: str) -> bool:
        return flag in self.flags


def test_vae_builder_preserves_default_tf32() -> None:
    config = _FakeBuilderConfig()
    _configure_builder_config(_FakeTrt, config, workspace_gib=64)
    assert config.pool_limits == [(_FakeMemoryPoolType.WORKSPACE, 64 << 30)]
    assert _FakeBuilderFlag.TF32 in config.flags


class _FakeTensor:
    def __init__(self, name: str, shape: tuple[int, ...], dtype: str = "fp32") -> None:
        self.name = name
        self.shape = shape
        self.dtype = dtype


class _FakeNetwork:
    num_inputs = 1
    num_outputs = 1

    def __init__(self, input_tensor: _FakeTensor, output_tensor: _FakeTensor) -> None:
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def get_input(self, _index: int) -> _FakeTensor:
        return self.input_tensor

    def get_output(self, _index: int) -> _FakeTensor:
        return self.output_tensor


def test_vae_builder_validates_full_resolution_network_contract() -> None:
    profile = OFFICIAL_VAE_DECODER_PROFILE
    network = _FakeNetwork(
        _FakeTensor("latents", profile.input_shape),
        _FakeTensor("video", profile.output_shape),
    )
    _validate_network_contract(_FakeTrt, network, profile)

    wrong = _FakeNetwork(
        _FakeTensor("latents", profile.input_shape),
        _FakeTensor("video", (1, 3, 120, 704, 1280)),
    )
    with pytest.raises(RuntimeError, match="output shape"):
        _validate_network_contract(_FakeTrt, wrong, profile)
