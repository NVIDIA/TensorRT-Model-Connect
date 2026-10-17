# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from families.minimax_h3.reference_posterior_builder import (
    _apply_profile,
    build_reference_posterior_engine,
    posterior_input_shapes,
)


def test_original_reference_posterior_image_profile():
    assert posterior_input_shapes() == {
        "moments": [(48, 16384), (48, 22016), (48, 65536)],
        "noise": [(24, 16384), (24, 22016), (24, 65536)],
    }


def test_posterior_profile_verifies_bounds_with_a_void_python_setter():
    class Profile:
        def __init__(self):
            self.values = {}

        def set_shape(self, name, lower, optimum, upper):
            self.values[name] = [lower, optimum, upper]
            return None

        def get_shape(self, name):
            return self.values[name]

    profile = Profile()
    shapes = posterior_input_shapes((1, 4, 64))
    _apply_profile(profile, shapes)
    assert profile.values == shapes
    profile.get_shape = lambda name: [(48, 1), (48, 4), (48, 32)]
    with pytest.raises(RuntimeError, match="did not retain"):
        _apply_profile(profile, shapes)


def test_posterior_profile_propagates_the_api_validation_error():
    class Profile:
        def set_shape(self, *_args):
            raise ValueError("invalid profile dimensions")

    with pytest.raises(ValueError, match="invalid profile dimensions"):
        _apply_profile(Profile(), posterior_input_shapes())


@pytest.mark.parametrize(
    "bad", [(0, 1, 2), (1, 3, 2), (True, 2, 3), (1, 2), [1, 2, 3], (1, 2, 3.0), (1, 2, 65537)]
)
def test_posterior_profile_rejected_before_cuda(bad):
    with pytest.raises(ValueError, match="latent-pixel bounds"):
        build_reference_posterior_engine(latent_pixels=bad)


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize(
    "pixels_profile", [(1, 4, 64), (16384, 22016, 65536)], ids=["tiny", "released"]
)
def test_reference_posterior_preserves_sampling_clamp_and_half_rounding(
    record_property, pixels_profile
):
    import json
    from unittest.mock import patch
    import tensorrt as trt
    import torch
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    plan = build_reference_posterior_engine(latent_pixels=pixels_profile)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None and engine.num_io_tensors == 3
    context = engine.create_execution_context()
    assert context is not None
    comparisons = []
    for pixels in pixels_profile:
        indices = torch.arange(24 * pixels, device="cuda").reshape(24, pixels)
        mean = ((indices % 31).float() - 15) / 16
        logvar = torch.tensor([-40.0, -2.0, 0.0, 2.0, 21.0], device="cuda")[indices % 5]
        noise = ((indices % 17).float() - 8) / 1024
        moments = torch.cat((mean, logvar), dim=0)
        posterior = DiagonalGaussianDistribution(moments[None])
        # Observe the actual upstream posterior's FP32 operation chain, with
        # identical supplied noise. The separate RNG path is not qualified here.
        with patch("diffusers.models.autoencoders.vae.randn_tensor", return_value=noise[None]):
            expected = posterior.sample(generator=torch.Generator().manual_seed(42)).half()[0]
        assert context.set_input_shape("moments", tuple(moments.shape))
        assert context.set_input_shape("noise", tuple(noise.shape))
        actual = torch.empty_like(mean)
        for name, value in dict(moments=moments, noise=noise, sampled_latents=actual).items():
            assert context.set_tensor_address(name, value.data_ptr())
        assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        # The semantic output boundary is FP16, despite the native host ABI
        # carrying a lossless FP32 widening. Freeze its FP16 dtype-default
        # tolerance, and independently require that no half cast was elided.
        assert torch.equal(actual, actual.half().float())
        torch.testing.assert_close(actual.half(), expected, rtol=1e-3, atol=1e-5)
        comparisons.append(
            dict(pixels=pixels, max_abs_error=(actual - expected.float()).abs().max().item())
        )
    # Exact midpoint rounding with zero perturbation: catches a cast optimized
    # away even if the earlier nonzero-noise comparisons happen to be close.
    midpoint = torch.tensor([1.0 + 2**-11, 1.0 + 3 * 2**-11], device="cuda")
    midpoint_pixels = pixels_profile[1]
    moments = torch.zeros((48, midpoint_pixels), device="cuda")
    moments[:24] = midpoint.repeat(midpoint_pixels // 2)
    noise = torch.zeros((24, midpoint_pixels), device="cuda")
    actual = torch.empty_like(noise)
    assert context.set_input_shape("moments", tuple(moments.shape))
    assert context.set_input_shape("noise", tuple(noise.shape))
    for name, value in dict(moments=moments, noise=noise, sampled_latents=actual).items():
        assert context.set_tensor_address(name, value.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    np.testing.assert_array_equal(actual.cpu().numpy(), moments[:24].half().float().cpu().numpy())
    record_property("posterior_comparisons", json.dumps(comparisons, allow_nan=False))
