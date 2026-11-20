# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The augmented Q.K^T tables reproduce HF SAM's decomposed rel-pos logits."""

from __future__ import annotations

import numpy as np
import pytest

from families.sam import rel_pos


def _hf_logits(q, k, rp_h, rp_w, size, scale):
    """HF SamVisionAttention: scaled Q.K^T plus get_decomposed_rel_pos."""
    batch, seq, dim = q.shape
    logits = (q * scale) @ k.transpose(0, 2, 1)
    grid_q = q.reshape(batch, size, size, dim)
    rel_h = np.einsum("bhwc,hkc->bhwk", grid_q, rp_h)
    rel_w = np.einsum("bhwc,wkc->bhwk", grid_q, rp_w)
    bias = rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    return logits + bias.reshape(batch, seq, seq)


def _augmented_logits(q, k, rp_h, rp_w, size, scale):
    """The graph's form: one Q.K^T over appended rows, then the scale."""
    batch, seq, dim = q.shape
    grid_q = q.reshape(batch, size, size, dim)
    row = np.einsum("bijc,icw->bijw", grid_q, rel_pos.rel_pos_query_table(rp_h, scale))
    col = np.einsum("bijc,jcw->bijw", grid_q, rel_pos.rel_pos_query_table(rp_w, scale))
    q_aug = np.concatenate([q, row.reshape(batch, seq, -1), col.reshape(batch, seq, -1)], -1)
    key_rows = np.broadcast_to(rel_pos.rel_pos_key_table(size), (batch,) + (seq, q_aug.shape[-1] - dim))
    k_aug = np.concatenate([k, key_rows], -1)
    return scale * (q_aug @ k_aug.transpose(0, 2, 1))


def _case(size, dim, seed, batch=3):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((batch, size * size, dim)).astype(np.float32).astype(np.float64)
    k = rng.standard_normal((batch, size * size, dim)).astype(np.float32).astype(np.float64)
    rp_h = rng.standard_normal((size, size, dim)).astype(np.float32)
    rp_w = rng.standard_normal((size, size, dim)).astype(np.float32)
    return q, k, rp_h, rp_w


@pytest.mark.parametrize("size", [14, 8, 5, 1])
def test_augmented_logits_match_hf_decomposed_rel_pos(size: int) -> None:
    # head_dim 64 (every SAM ViT): the scale 1/8 is a power of two, so the
    # tables are exact and only float64 summation order differs.
    q, k, rp_h, rp_w = _case(size, 64, seed=size)
    scale = 1.0 / np.sqrt(64)
    expected = _hf_logits(q, k, rp_h.astype(np.float64), rp_w.astype(np.float64), size, scale)
    actual = _augmented_logits(q, k, rp_h, rp_w, size, scale)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def test_augmented_attention_with_zero_padded_value_matches_hf() -> None:
    # The graph pads V with zero columns to the augmented width (one head
    # dimension for the fused attention) and drops them from the context.
    q, k, rp_h, rp_w = _case(14, 64, seed=7)
    v = np.random.default_rng(8).standard_normal(q.shape)
    scale = 1.0 / np.sqrt(64)
    expected = _softmax(_hf_logits(q, k, rp_h.astype(np.float64), rp_w.astype(np.float64), 14, scale)) @ v
    logits = _augmented_logits(q, k, rp_h, rp_w, 14, scale)
    width = 64 + 2 * rel_pos.rel_pos_block_width(14)
    v_aug = np.zeros(v.shape[:-1] + (width,))
    v_aug[..., :64] = v
    actual = (_softmax(logits) @ v_aug)[..., :64]
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    assert not (_softmax(logits) @ v_aug)[..., 64:].any()


def test_augmented_logits_match_with_an_inexact_scale() -> None:
    q, k, rp_h, rp_w = _case(6, 48, seed=48)
    scale = 1.0 / np.sqrt(48)
    expected = _hf_logits(q, k, rp_h.astype(np.float64), rp_w.astype(np.float64), 6, scale)
    actual = _augmented_logits(q, k, rp_h, rp_w, 6, scale)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize(("size", "width"), [(1, 8), (5, 8), (8, 8), (14, 16), (64, 64)])
def test_block_width_pads_each_block_to_eight_columns(size: int, width: int) -> None:
    assert rel_pos.rel_pos_block_width(size) == width


def test_query_table_pads_with_zero_columns() -> None:
    _, _, rp_h, _ = _case(14, 64, seed=1)
    table = rel_pos.rel_pos_query_table(rp_h, 0.125)
    assert table.shape == (14, 64, 16)
    np.testing.assert_array_equal(table[:, :, :14], rp_h.transpose(0, 2, 1) * 8.0)
    assert not table[:, :, 14:].any()


def test_key_table_is_one_hot_row_then_column() -> None:
    table = rel_pos.rel_pos_key_table(5)
    assert table.shape == (25, 16)
    for key in range(25):
        row, col = divmod(key, 5)
        expected = np.zeros(16, dtype=np.float32)
        expected[row] = 1.0
        expected[8 + col] = 1.0
        np.testing.assert_array_equal(table[key], expected)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((np.zeros((4, 5, 8), dtype=np.float32), 0.125), "size, size, head_dim"),
        ((np.zeros((4, 4, 8), dtype=np.float32), 0.0), "logit scale"),
    ],
)
def test_query_table_refuses_bad_inputs(args, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rel_pos.rel_pos_query_table(*args)


def test_block_width_refuses_an_empty_grid() -> None:
    with pytest.raises(ValueError, match="positive"):
        rel_pos.rel_pos_block_width(0)
