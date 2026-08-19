# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for clip_metrics.py — no GPU, no open_clip, no PIL required.

Strategy: mock _embed_images and _embed_text (the two functions that touch
external libs) so tests exercise all compute_clip_metrics paths purely in
terms of torch tensor arithmetic, which is always available in the harness
environment.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tensorrt_model_connect.models.flux.tests.e2e_plugins.comparators.clip_metrics import (
    ClipMetrics,
    compute_clip_metrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_dummy_png(path: Path) -> None:
    """Write a minimal valid PNG (4×4 grey) without requiring PIL."""
    import struct
    import zlib

    def _chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    w = h = 4
    raw = b"".join(b"\x00" + bytes([128, 128, 128] * w) for _ in range(h))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _frames_dir(tmp: Path, name: str, n: int = 1) -> Path:
    d = tmp / name
    d.mkdir()
    for i in range(n):
        _write_dummy_png(d / f"frame_{i:04d}.png")
    return d


def _unit_vec(dim: int, idx: int = 0):
    """Return a 1-D torch tensor that is a unit vector along `idx`."""
    import torch
    v = torch.zeros(dim)
    v[idx] = 1.0
    return v


# Common mock signatures expected by compute_clip_metrics:
#   _embed_images(paths) -> Tensor[N, D] | None
#   _embed_text(prompt)  -> (Tensor[D], bool)          (embedding, truncated)

_MODULE = "tensorrt_model_connect.models.flux.tests.e2e_plugins.comparators.clip_metrics"


# ---------------------------------------------------------------------------
# Tests: early-exit / None paths
# ---------------------------------------------------------------------------

class TestEarlyReturns:
    def test_none_when_prompt_is_none(self, tmp_path):
        trt = _frames_dir(tmp_path, "trt")
        ref = _frames_dir(tmp_path, "ref")
        assert compute_clip_metrics(str(trt), str(ref), prompt=None) is None

    def test_none_when_prompt_empty(self, tmp_path):
        trt = _frames_dir(tmp_path, "trt")
        ref = _frames_dir(tmp_path, "ref")
        assert compute_clip_metrics(str(trt), str(ref), prompt="") is None

    def test_none_when_no_trt_frames(self, tmp_path):
        trt = tmp_path / "empty"
        trt.mkdir()
        ref = _frames_dir(tmp_path, "ref")
        assert compute_clip_metrics(str(trt), str(ref), "a cat") is None

    def test_none_when_no_ref_frames(self, tmp_path):
        trt = _frames_dir(tmp_path, "trt")
        ref = tmp_path / "empty"
        ref.mkdir()
        assert compute_clip_metrics(str(trt), str(ref), "a cat") is None

    def test_none_when_open_clip_missing(self, tmp_path):
        trt = _frames_dir(tmp_path, "trt")
        ref = _frames_dir(tmp_path, "ref")
        with patch(f"{_MODULE}._load_clip", return_value=None):
            assert compute_clip_metrics(str(trt), str(ref), "a cat") is None

    def test_none_when_embed_images_returns_none(self, tmp_path):
        trt = _frames_dir(tmp_path, "trt")
        ref = _frames_dir(tmp_path, "ref")
        txt = _unit_vec(512, 0)
        with (
            patch(f"{_MODULE}._embed_images", return_value=None),
            patch(f"{_MODULE}._embed_text", return_value=(txt, False)),
        ):
            assert compute_clip_metrics(str(trt), str(ref), "a cat") is None


# ---------------------------------------------------------------------------
# Tests: metric arithmetic
# ---------------------------------------------------------------------------

class TestMetricArithmetic:
    """All paths that reach the actual cosine math."""

    def _run(self, tmp_path, trt_emb, ref_emb, txt_emb, truncated=False, prompt="a cat"):
        """Helper: mock embed functions, return ClipMetrics."""
        trt = _frames_dir(tmp_path, "trt")
        ref = _frames_dir(tmp_path, "ref")
        call = {"n": 0}

        def _fake_embed_images(paths):
            call["n"] += 1
            return trt_emb if call["n"] == 1 else ref_emb

        with (
            patch(f"{_MODULE}._embed_images", side_effect=_fake_embed_images),
            patch(f"{_MODULE}._embed_text", return_value=(txt_emb, truncated)),
        ):
            return compute_clip_metrics(str(trt), str(ref), prompt)

    def test_returns_clip_metrics_namedtuple(self, tmp_path):
        e = _unit_vec(512, 0).unsqueeze(0)
        t = _unit_vec(512, 0)
        result = self._run(tmp_path, e, e, t)
        assert isinstance(result, ClipMetrics)

    def test_identical_embeddings_delta_zero(self, tmp_path):
        e = _unit_vec(512, 0).unsqueeze(0)
        t = _unit_vec(512, 0)
        result = self._run(tmp_path, e, e, t)
        assert result is not None
        assert result.prompt_clipscore_delta == pytest.approx(0.0, abs=1e-4)

    def test_identical_embeddings_image_cosine_one(self, tmp_path):
        e = _unit_vec(512, 0).unsqueeze(0)
        t = _unit_vec(512, 0)
        result = self._run(tmp_path, e, e, t)
        assert result is not None
        assert result.trt_hf_image_clip_cosine == pytest.approx(1.0, abs=1e-4)

    def test_positive_delta_when_trt_more_aligned(self, tmp_path):
        """TRT embedding closer to text than HF → delta > 0."""
        import torch
        # trt cos(text)=0.9, hf cos(text)=0.7
        trt_e = torch.zeros(512)
        trt_e[0] = 0.9
        trt_e[1] = (1 - 0.81) ** 0.5
        ref_e = torch.zeros(512)
        ref_e[0] = 0.7
        ref_e[1] = (1 - 0.49) ** 0.5
        txt   = _unit_vec(512, 0)
        result = self._run(tmp_path, trt_e.unsqueeze(0), ref_e.unsqueeze(0), txt)
        assert result is not None
        assert result.prompt_clipscore_delta > 0

    def test_large_negative_delta_when_trt_orthogonal_to_text(self, tmp_path):
        """TRT cosine=0 (orthogonal), HF cosine=0.28 → clipscore delta ≈ -28."""
        import torch
        trt_e = _unit_vec(512, 1).unsqueeze(0)          # orthogonal to text
        ref_e = torch.zeros(512)
        ref_e[0] = 0.28
        ref_e[1] = (1 - 0.0784) ** 0.5
        txt = _unit_vec(512, 0)
        result = self._run(tmp_path, trt_e, ref_e.unsqueeze(0), txt)
        assert result is not None
        assert result.trt_prompt_clipscore == pytest.approx(0.0, abs=1e-4)
        assert result.prompt_clipscore_delta < -20.0

    def test_clipscore_formula_100x_max_cos_0(self, tmp_path):
        """CLIPScore = 100 * max(cos, 0): negative cosine yields 0, not negative."""
        import torch
        # cos(trt, text) = -0.5  →  clipscore should be 0.0
        trt_e = torch.zeros(512)
        trt_e[0] = -0.5
        trt_e[1] = (1 - 0.25) ** 0.5
        ref_e = _unit_vec(512, 0).unsqueeze(0)
        txt   = _unit_vec(512, 0)
        result = self._run(tmp_path, trt_e.unsqueeze(0), ref_e, txt)
        assert result is not None
        assert result.trt_prompt_clipscore == pytest.approx(0.0, abs=1e-4)

    def test_hf_floor_when_reference_broken(self, tmp_path):
        """HF embedding orthogonal to text → hf_prompt_clipscore == 0."""
        trt_e = _unit_vec(512, 0).unsqueeze(0)
        ref_e = _unit_vec(512, 1).unsqueeze(0)  # orthogonal to text
        txt   = _unit_vec(512, 0)
        result = self._run(tmp_path, trt_e, ref_e, txt)
        assert result is not None
        assert result.hf_prompt_clipscore == pytest.approx(0.0, abs=1e-4)

    def test_prompt_truncated_flag_propagated(self, tmp_path):
        e = _unit_vec(512, 0).unsqueeze(0)
        t = _unit_vec(512, 0)
        result = self._run(tmp_path, e, e, t, truncated=True, prompt="a cat " * 30)
        assert result is not None
        assert result.prompt_truncated is True

    def test_prompt_not_truncated_flag(self, tmp_path):
        e = _unit_vec(512, 0).unsqueeze(0)
        t = _unit_vec(512, 0)
        result = self._run(tmp_path, e, e, t, truncated=False)
        assert result is not None
        assert result.prompt_truncated is False

    def test_max_frames_limits_glob(self, tmp_path):
        """max_frames=2 should pass only 2 paths to _embed_images."""
        trt = _frames_dir(tmp_path, "trt", n=5)
        ref = _frames_dir(tmp_path, "ref", n=5)
        e = _unit_vec(512, 0).unsqueeze(0)
        t = _unit_vec(512, 0)
        captured = []

        def _fake_embed(paths):
            captured.append(len(paths))
            return e

        with (
            patch(f"{_MODULE}._embed_images", side_effect=_fake_embed),
            patch(f"{_MODULE}._embed_text", return_value=(t, False)),
        ):
            compute_clip_metrics(str(trt), str(ref), "a cat", max_frames=2)

        assert all(n == 2 for n in captured)
