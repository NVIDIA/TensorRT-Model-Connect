# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit checks for the rt_detr_v2 builders that need no checkpoint."""

from __future__ import annotations

import numpy as np
import pytest

from families.rt_detr_v2 import decoder_builder, graph


def test_anchors_use_the_measured_constants():
    # Anchors are stored in logit space. Positions outside the 1e-2 validity
    # band carry infinity so they can never win the top-k - but that only bites
    # on a grid fine enough to put a cell centre below 0.01, which a coarse
    # grid never does. The real 80x80 level does: its first centre is 0.00625.
    coarse = decoder_builder.build_anchors([(4, 4), (2, 2)])
    assert coarse.shape == (1, 20, 4)
    assert np.isfinite(coarse).all(), "a coarse grid should mask nothing"

    fine = decoder_builder.build_anchors([(80, 80)])
    assert np.isinf(fine).any(), "the 80x80 level must mask its border cells"
    finite = fine[np.isfinite(fine).all(-1)]
    assert finite.size > 0
    centres = 1.0 / (1.0 + np.exp(-finite[:, :2]))
    assert centres.min() > 0.01 and centres.max() < 0.99


def test_anchor_box_size_doubles_per_level():
    anchors = decoder_builder.build_anchors([(4, 4), (2, 2)])
    sizes = 1.0 / (1.0 + np.exp(-anchors[0, :, 2]))
    first = sizes[:16]
    second = sizes[16:]
    assert np.allclose(first[np.isfinite(first)], 0.05, atol=1e-6)
    assert np.allclose(second[np.isfinite(second)], 0.10, atol=1e-6)


def test_batch_norm_folding_matches_the_explicit_form():
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
    gamma = rng.random(4).astype(np.float32) + 0.5
    beta = rng.standard_normal(4).astype(np.float32)
    mean = rng.standard_normal(4).astype(np.float32)
    variance = rng.random(4).astype(np.float32) + 0.5
    folded, bias = graph.fold_batch_norm(weight, gamma, beta, mean, variance, 1e-5)
    scale = gamma / np.sqrt(variance + 1e-5)
    assert np.allclose(folded, weight * scale.reshape(-1, 1, 1, 1), atol=1e-6)
    assert np.allclose(bias, beta - mean * scale, atol=1e-6)


def test_position_embedding_order_is_sin_cos_width_then_height():
    from families.rt_detr_v2 import encoder_builder

    embedding = encoder_builder.sine_position_embedding(2, 3, 8, 10000.0)
    assert embedding.shape == (1, 6, 8)
    # The first quarter is sin(width); with indexing="ij" the width index is
    # the slow axis, so the first two rows share a width of 0.
    assert np.allclose(embedding[0, 0, :2], embedding[0, 1, :2], atol=1e-6)


def test_resolve_rejects_an_unsupported_decoder(tmp_path):
    from families.rt_detr_v2 import config

    (tmp_path / "config.json").write_text('{"decoder_method": "discrete"}', encoding="utf-8")
    with pytest.raises(NotImplementedError):
        config.resolve(tmp_path, image_size=640)
