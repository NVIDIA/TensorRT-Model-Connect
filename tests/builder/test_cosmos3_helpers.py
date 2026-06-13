"""Unit tests for cosmos3 diffusion-side numerical helpers."""

from __future__ import annotations

import numpy as np
import pytest

from tensorrt_model_connect.families.cosmos3 import cosmos3_diffusion_helpers as h


class TestSinusoidalTimestepEmbedding:
    def test_shape_super(self):
        emb = h.cosmos3_sinusoidal_timestep_embedding(
            np.array([500], dtype=np.float32), hidden_size=5120)
        assert emb.shape == (1, 5120)
        assert emb.dtype == np.float32

    def test_shape_nano(self):
        emb = h.cosmos3_sinusoidal_timestep_embedding(
            np.array([0, 500, 999], dtype=np.float32), hidden_size=4096)
        assert emb.shape == (3, 4096)

    def test_range_bounded(self):
        # sin/cos values must lie in [-1, 1].
        emb = h.cosmos3_sinusoidal_timestep_embedding(
            np.array([0, 100, 999], dtype=np.float32), hidden_size=128)
        assert emb.min() >= -1.0 - 1e-6
        assert emb.max() <= 1.0 + 1e-6

    def test_t0_first_half_cosines_are_one(self):
        # At t=0 the args are zero, so cos(0)=1 across the cosine half.
        emb = h.cosmos3_sinusoidal_timestep_embedding(
            np.array([0], dtype=np.float32), hidden_size=128)
        half = 128 // 2
        np.testing.assert_allclose(emb[0, :half], 1.0, atol=1e-6)
        # sin(0)=0 across the sine half.
        np.testing.assert_allclose(emb[0, half:], 0.0, atol=1e-6)

    def test_timestep_scale_applied(self):
        # With scale=0.001 the effective t for embedding math is t/1000.
        # Two timesteps that differ by 1000 should map close to two
        # timesteps that differ by 1 under scale=1.0.
        small = h.cosmos3_sinusoidal_timestep_embedding(
            np.array([1000], dtype=np.float32), 128, timestep_scale=0.001)
        ref = h.cosmos3_sinusoidal_timestep_embedding(
            np.array([1], dtype=np.float32), 128, timestep_scale=1.0)
        np.testing.assert_allclose(small, ref, atol=1e-5)

    def test_odd_hidden_size_padding(self):
        # Odd hidden_size should still produce a tensor of that exact size.
        emb = h.cosmos3_sinusoidal_timestep_embedding(
            np.array([100], dtype=np.float32), hidden_size=129)
        assert emb.shape == (1, 129)


class TestRectifiedFlowSigmas:
    def test_l0_lane_30steps_480p(self):
        sigmas = h.cosmos3_rectified_flow_sigmas(30, 480)
        assert sigmas.shape == (30,)
        assert sigmas[0] == pytest.approx(1.0, abs=1e-6)
        assert sigmas[-1] > 0
        # Strictly decreasing.
        assert (np.diff(sigmas) < 0).all()

    def test_resolution_shift_table(self):
        # Higher resolution → higher shift → larger sigma values at the
        # mid-schedule (more aggressive denoising).
        s256 = h.cosmos3_rectified_flow_sigmas(30, 256)
        s480 = h.cosmos3_rectified_flow_sigmas(30, 480)
        s720 = h.cosmos3_rectified_flow_sigmas(30, 720)
        # Mid-step comparison.
        assert s720[15] > s480[15] > s256[15]

    def test_closest_resolution_picked(self):
        # 1080 falls back to the closest documented shift (720 → shift=10).
        s1080 = h.cosmos3_rectified_flow_sigmas(30, 1080)
        s720 = h.cosmos3_rectified_flow_sigmas(30, 720)
        np.testing.assert_allclose(s1080, s720)

    def test_dynamic_shifting_raises(self):
        with pytest.raises(NotImplementedError):
            h.cosmos3_rectified_flow_sigmas(30, 480, use_dynamic_shifting=True)


class TestPatchShape:
    def test_l0_shape(self):
        # L0 lane: 5 frames, 480x832.
        t, hh, ww, nd = h.cosmos3_patch_shape(5, 480, 832)
        assert (t, hh, ww) == (1, 30, 52)
        assert nd == 1 * (30 // 2) * (52 // 2)  # 15 * 26 = 390

    def test_highres_shape(self):
        # High-res Super sweep: 49 frames, 720x1280.
        t, hh, ww, nd = h.cosmos3_patch_shape(49, 720, 1280)
        assert (t, hh, ww) == (12, 45, 80)
        assert nd == 12 * (45 // 2) * (80 // 2)  # 12 * 22 * 40 = 10560

    def test_unpatchify_round_trips_pixel_shape(self):
        for frames, height, width in [(5, 480, 832), (49, 720, 1280)]:
            t_lat, h_lat, w_lat, _ = h.cosmos3_patch_shape(frames, height, width)
            num_frames_out, channels, h_out, w_out = h.cosmos3_unpatchify_shape(
                t_lat, h_lat, w_lat)
            # Frame count and dims should round-trip up to VAE granularity.
            assert num_frames_out == t_lat * 4
            assert channels == 3
            assert h_out == height
            assert w_out == width

    def test_single_frame_image_mode(self):
        # 1-frame image case: t_lat is clamped to 1.
        t, hh, ww, nd = h.cosmos3_patch_shape(1, 480, 480)
        assert t == 1
        assert nd == 1 * 15 * 15
