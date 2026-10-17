# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VisualVAE tile contracts and an independent upstream FP32 execution oracle.

The GPU test uses all six encoder levels and both temporal strides, with
reduced channels only. It does not qualify trained weights or Ref2VA video.
"""

from dataclasses import replace
import json

import numpy as np
import pytest

from families.minimax_h3.vae_encoder_builder import (
    VideoEncoderConfig,
    _validate_weights,
    build_vae_tile_encoder_engine,
    checkpoint_keys,
    checkpoint_shapes,
)
from families.minimax_h3.tests.vae_encoder_localize import upstream_hook_path


TINY = replace(
    VideoEncoderConfig(),
    block_out_channels=(4, 8, 8, 12, 12, 16),
    latent_channels=2,
    norm_num_groups=2,
)


def test_diagnostic_downsample_hooks_observe_the_same_padding_boundary():
    assert (
        upstream_hook_path("encoder.down_blocks.0.downsamplers.0.conv")
        == "encoder.down_blocks.0.downsamplers.0"
    )
    for name in ("encoder.conv_in", "encoder.down_blocks.0.resnets.0.conv1", "encoder.norm_out"):
        assert upstream_hook_path(name) == name


def test_original_encoder_schema_has_exactly_118_fp32_tensors():
    shapes = checkpoint_shapes()
    assert len(shapes) == len(checkpoint_keys()) == 118
    assert shapes["encoder.conv_in.weight"] == (128, 3, 3, 3, 3)
    assert shapes["encoder.down_blocks.5.resnets.0.conv_shortcut.weight"] == (1024, 512, 1, 1, 1)
    assert shapes["encoder.conv_out.weight"] == (48, 1024, 3, 3, 3)
    assert shapes["quant_conv.weight"] == (48, 48, 1, 1, 1)
    assert not any(name.startswith(("decoder.", "post_quant_conv.")) for name in shapes)
    assert VideoEncoderConfig().spatial_ratio == 16
    assert VideoEncoderConfig().temporal_ratio == 4


@pytest.mark.parametrize(
    "change",
    [
        {"block_out_channels": ()},
        {"spatial_downsample_factors": (2,)},
        {"spatial_downsample_factors": (2, 2, 2, 2, 1, 1.0)},
        {"temporal_downsample_factors": (1, 1, 1, 1, 1, 3)},
        {"temporal_downsample_factors": (1, 2, 2, 1, 1, 2)},
        {"block_out_channels": (128, 256, 256, 512, 512, 1025)},
        {"norm_num_groups": True},
        {"latent_channels": 0},
        {"layers_per_block": 1.5},
        {"norm_eps": float("nan")},
        {"norm_eps": 0.0},
    ],
)
def test_encoder_config_rejects_unsupported_geometry(change):
    with pytest.raises(ValueError):
        checkpoint_shapes(replace(VideoEncoderConfig(), **change))


@pytest.mark.parametrize(
    "height,width,frames",
    [
        (31, 48, 1),
        (32, 47, 1),
        (16, 48, 1),
        (True, 48, 1),
        (32, 48, 0),
        (32, 48, 18),
        (32, 48, True),
    ],
)
def test_encoder_rejects_invalid_profiles_before_cuda(height, width, frames):
    with pytest.raises(ValueError, match="tile sizes|at most 17"):
        build_vae_tile_encoder_engine(
            {}, config=TINY, tile_height=height, tile_width=width, max_frames=frames
        )


def test_encoder_weights_reject_extra_missing_wrong_shape_and_downcast():
    weights = {name: np.zeros(shape, np.float32) for name, shape in checkpoint_shapes(TINY).items()}
    _validate_weights(weights, TINY)
    with pytest.raises(ValueError, match="exactly"):
        _validate_weights({**weights, "decoder.unexpected": np.zeros(1)}, TINY)
    key = "encoder.conv_in.weight"
    value = weights.pop(key)
    with pytest.raises(ValueError, match="exactly"):
        _validate_weights(weights, TINY)
    for bad in (value.astype(np.float16), value.reshape(-1)):
        with pytest.raises(ValueError, match="FP32 shape"):
            _validate_weights({**weights, key: bad}, TINY)


@pytest.mark.gpu
@pytest.mark.trt
def test_native_encoder_moments_match_upstream_fp32_and_preserve_causality(record_property):
    import tensorrt as trt
    import torch
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoEncoder3d

    # Freeze the oracle policy before execution, not in response to a mismatch.
    # The tolerance is PyTorch's default FP32 assert_close tolerance.
    torch.manual_seed(203)
    with torch.backends.cudnn.flags(allow_tf32=False), torch.no_grad():
        encoder = MiniMaxH3VideoEncoder3d(
            in_channels=TINY.in_channels,
            out_channels=2 * TINY.latent_channels,
            block_out_channels=TINY.block_out_channels,
            layers_per_block=TINY.layers_per_block,
            spatial_downsample_factors=TINY.spatial_downsample_factors,
            temporal_downsample_factors=TINY.temporal_downsample_factors,
            norm_num_groups=TINY.norm_num_groups,
            norm_eps=TINY.norm_eps,
        ).eval()
        quant = torch.nn.Conv3d(2 * TINY.latent_channels, 2 * TINY.latent_channels, 1).eval()
        weights = {f"encoder.{name}": value.numpy() for name, value in encoder.state_dict().items()}
        weights.update(
            {f"quant_conv.{name}": value.numpy() for name, value in quant.state_dict().items()}
        )
        assert {name: tuple(value.shape) for name, value in weights.items()} == checkpoint_shapes(
            TINY
        )
        plan = build_vae_tile_encoder_engine(
            weights, config=TINY, tile_height=32, tile_width=48, workspace_bytes=1 << 30
        )
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(plan)
        assert engine is not None
        assert engine.num_io_tensors == 2
        assert all(engine.get_tensor_dtype(name) == trt.float32 for name in ("pixels", "moments"))
        context = engine.create_execution_context()
        assert context is not None
        encoder, quant = encoder.cuda(), quant.cuda()
        stream = torch.cuda.current_stream()

        def infer(pixels):
            assert pixels.is_contiguous() and pixels.dtype == torch.float32
            assert context.set_input_shape("pixels", tuple(pixels.shape))
            shape = tuple(context.get_tensor_shape("moments"))
            output = torch.empty(shape, dtype=torch.float32, device="cuda")
            assert context.set_tensor_address("pixels", pixels.data_ptr())
            assert context.set_tensor_address("moments", output.data_ptr())
            assert context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            return output

        pixels = torch.randn(1, 3, 17, 32, 48, device="cuda")
        results = []
        for frames in (1, 2, 5, 17):
            value = pixels[:, :, :frames].contiguous()
            expected = quant(encoder(value))
            actual = infer(value)
            assert actual.shape == (1, 4, (frames + 3) // 4, 2, 3)
            error = (actual - expected).abs()
            results.append(
                dict(
                    frames=frames,
                    max_abs_error=error.max().item(),
                    mean_abs_error=error.mean().item(),
                )
            )
            torch.testing.assert_close(actual, expected, rtol=1.3e-6, atol=1e-5)
        original = infer(pixels)
        changed = pixels.clone()
        changed[:, :, 1:] = changed[:, :, 1:] * 9 + 23
        altered = infer(changed)
        torch.testing.assert_close(original[:, :, :1], altered[:, :, :1], rtol=1.3e-6, atol=1e-5)
        record_property("fp32_moments_comparison", json.dumps(results, allow_nan=False))
