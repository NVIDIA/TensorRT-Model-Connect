# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from tensorrt_model_connect import trt_compat


if trt_compat.is_available("tensorrt"):
    pass
elif trt_compat.is_available("tensorrt_rtx"):
    trt_compat.configure_backend(rtx=True)
else:
    pytest.skip("TensorRT Python bindings are unavailable", allow_module_level=True)

from tensorrt_model_connect.families.minimax_h3.audio_vae_builder import (  # noqa: E402
    DEFAULT_AUDIO_VAE_DECODER_CONFIG,
    AudioVAEDecoderConfig,
    build_audio_vae_decoder_engine,
    checkpoint_keys,
    decoder_config_from_checkpoint,
    fold_weight_norm,
)


def _tiny_profile() -> AudioVAEDecoderConfig:
    return AudioVAEDecoderConfig(
        latent_channels=2,
        latent_frames=207,
        min_latent_frames=207,
        max_latent_frames=575,
        latent_dim=4,
        decoder_dim=4,
        decoder_rates=(2,),
        decoder_kernel_sizes=(4,),
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1,),),
        sampling_rate=8,
    )


def _tiny_weights(profile: AudioVAEDecoderConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(19)

    def weight(shape: tuple[int, ...]) -> np.ndarray:
        return rng.normal(0.0, 0.05, shape).astype(np.float32)

    def bias(channels: int) -> np.ndarray:
        return rng.normal(0.0, 0.01, (channels,)).astype(np.float32)

    def normalized_weight(shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        return np.ones((shape[0],) + (1,) * (len(shape) - 1), np.float32), weight(shape)

    state = {
        "dec_in_proj.weight": weight((profile.latent_dim, profile.latent_channels, 1)),
        "dec_in_proj.bias": bias(profile.latent_dim),
    }
    gain, direction = normalized_weight((profile.decoder_dim, profile.latent_dim, 7))
    state.update(
        {
            "decoder.conv_pre.weight_g": gain,
            "decoder.conv_pre.weight_v": direction,
            "decoder.conv_pre.bias": bias(profile.decoder_dim),
        }
    )

    for stage, kernel in enumerate(profile.decoder_kernel_sizes):
        input_channels = profile.decoder_dim // (1 << stage)
        output_channels = profile.decoder_dim // (1 << (stage + 1))
        prefix = f"decoder.ups.{stage}.0"
        gain, direction = normalized_weight((input_channels, output_channels, kernel))
        state.update(
            {
                f"{prefix}.weight_g": gain,
                f"{prefix}.weight_v": direction,
                f"{prefix}.bias": bias(output_channels),
            }
        )

    for block in range(profile.num_upsamples * profile.num_kernels):
        kernel = profile.resblock_kernel_sizes[block % profile.num_kernels]
        dilations = profile.resblock_dilation_sizes[block % profile.num_kernels]
        stage = block // profile.num_kernels
        channels = profile.decoder_dim // (1 << (stage + 1))
        prefix = f"decoder.resblocks.{block}"
        for activation in range(2 * len(dilations)):
            activation_prefix = f"{prefix}.activations.{activation}"
            state.update(
                {
                    f"{activation_prefix}.act.alpha": weight((channels,)),
                    f"{activation_prefix}.act.beta": weight((channels,)),
                    f"{activation_prefix}.upsample.filter": weight((1, 1, 12)),
                    f"{activation_prefix}.downsample.lowpass.filter": weight((1, 1, 12)),
                }
            )
        for group in ("convs1", "convs2"):
            for index in range(len(dilations)):
                convolution_prefix = f"{prefix}.{group}.{index}"
                gain, direction = normalized_weight((channels, channels, kernel))
                state.update(
                    {
                        f"{convolution_prefix}.weight_g": gain,
                        f"{convolution_prefix}.weight_v": direction,
                        f"{convolution_prefix}.bias": bias(channels),
                    }
                )

    final_channels = profile.decoder_dim // (1 << profile.num_upsamples)
    state.update(
        {
            "decoder.activation_post.act.alpha": weight((final_channels,)),
            "decoder.activation_post.act.beta": weight((final_channels,)),
            "decoder.activation_post.upsample.filter": weight((1, 1, 12)),
            "decoder.activation_post.downsample.lowpass.filter": weight((1, 1, 12)),
        }
    )
    gain, direction = normalized_weight((1, final_channels, 7))
    state.update(
        {
            "decoder.conv_post.weight_g": gain,
            "decoder.conv_post.weight_v": direction,
        }
    )
    return state


def test_public_audio_vae_decoder_contract_is_complete() -> None:
    profile = DEFAULT_AUDIO_VAE_DECODER_CONFIG
    keys = checkpoint_keys(profile)
    assert len(keys) == len(set(keys)) == 914
    assert profile.hop_length == 800
    assert profile.output_samples == 165_600
    assert (profile.min_latent_frames, profile.latent_frames, profile.max_latent_frames) == (
        207,
        207,
        575,
    )
    assert (profile.min_output_samples, profile.max_output_samples) == (165_600, 460_000)
    assert "dec_in_proj.weight" in keys
    assert "decoder.resblocks.20.convs2.2.weight_v" in keys
    assert "decoder.activation_post.downsample.lowpass.filter" in keys
    assert "encoder.block.0.weight_g" not in keys


def test_checkpoint_config_drives_decoder_profile() -> None:
    raw = {
        "encoder_rates": [2, 2],
        "latent_channels": 2,
        "latent_dim": 4,
        "decoder_dim": 8,
        "decoder_rates": [2, 2],
        "decoder_kernel_sizes": [4, 4],
        "resblock_kernel_sizes": [3],
        "resblock_dilation_sizes": [[1, 3]],
        "sampling_rate": 16,
    }
    profile = decoder_config_from_checkpoint(
        raw,
        latent_frames=7,
        min_latent_frames=5,
        max_latent_frames=9,
    )
    assert profile.latent_frames == 7
    assert (profile.min_latent_frames, profile.max_latent_frames) == (5, 9)
    assert profile.hop_length == 4
    assert profile.output_samples == 28

    raw["encoder_rates"] = [2]
    with pytest.raises(ValueError, match="hop mismatch"):
        decoder_config_from_checkpoint(raw, latent_frames=7)


def test_weight_norm_fold_matches_dimension_zero_formula() -> None:
    direction = np.asarray(
        [
            [[3.0, 4.0], [0.0, 0.0]],
            [[1.0, 2.0], [2.0, 1.0]],
        ],
        dtype=np.float32,
    )
    gain = np.asarray([[[10.0]], [[5.0]]], dtype=np.float32)
    expected = direction * (
        gain
        / np.sqrt(
            np.sum(direction * direction, axis=(1, 2), keepdims=True, dtype=np.float32)
        )
    )
    actual = fold_weight_norm(gain, direction)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)


def test_tiny_checkpoint_keys_partition_decoder_exactly() -> None:
    profile = _tiny_profile()
    weights = _tiny_weights(profile)
    assert set(checkpoint_keys(profile)) == set(weights)
    assert len(weights) == 28


def test_audio_vae_builder_fails_closed_on_non_fp32_tensor() -> None:
    profile = _tiny_profile()
    weights = _tiny_weights(profile)
    weights["dec_in_proj.weight"] = weights["dec_in_proj.weight"].astype(np.float16)
    with pytest.raises(ValueError, match="must remain FP32"):
        build_audio_vae_decoder_engine(weights, profile, workspace_bytes=1 << 30)


@pytest.mark.gpu
def test_tiny_audio_vae_decoder_serializes_with_native_layers() -> None:
    profile = _tiny_profile()
    weights = _tiny_weights(profile)
    plan = build_audio_vae_decoder_engine(weights, profile, workspace_bytes=1 << 30)
    assert plan

    trt = trt_compat.get_trt()
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    assert engine.get_tensor_shape("audio_latents") == (
        profile.batch_size,
        profile.latent_channels,
        -1,
    )
    assert engine.get_tensor_shape("decoded_audio") == (
        profile.batch_size,
        1,
        -1,
    )
    assert tuple(
        tuple(shape) for shape in engine.get_tensor_profile_shape("audio_latents", 0)
    ) == (
        (profile.batch_size, profile.latent_channels, profile.min_latent_frames),
        (profile.batch_size, profile.latent_channels, profile.latent_frames),
        (profile.batch_size, profile.latent_channels, profile.max_latent_frames),
    )

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    context = engine.create_execution_context()
    assert context is not None
    for latent_frames in (profile.min_latent_frames, profile.max_latent_frames):
        input_tensor = torch.zeros(
            (profile.batch_size, profile.latent_channels, latent_frames),
            dtype=torch.float32,
            device="cuda",
        )
        assert context.set_input_shape("audio_latents", tuple(input_tensor.shape))
        output_shape = tuple(context.get_tensor_shape("decoded_audio"))
        assert output_shape == (profile.batch_size, 1, latent_frames * profile.hop_length)
        output_tensor = torch.empty(output_shape, dtype=torch.float32, device="cuda")
        assert context.set_tensor_address("audio_latents", input_tensor.data_ptr())
        assert context.set_tensor_address("decoded_audio", output_tensor.data_ptr())
        assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.isfinite(output_tensor).all()
