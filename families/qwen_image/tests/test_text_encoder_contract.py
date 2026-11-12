# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen2.5-VL text-encoder attention precision contract."""

from __future__ import annotations

import inspect

import numpy as np

from families.qwen_image import qwen25_vl_text_encoder_builder as text_builder


def test_attention_scores_and_softmax_run_in_fp32() -> None:
    # Qwen2.5-VL's q/k biases give attention logits in the thousands. The
    # reference SDPA keeps them in FP32; BF16 logits that large lose the
    # softmax (layer-0 errors of 14-25% on real prompts).
    source = inspect.getsource(text_builder.build_qwen25vl_text_encoder_engine)
    call = source[source.index("graph_ops.add_attention_from_rows(") :]
    call = call[: call.index('tag=f"{prefix}.attn"')]
    assert "fp32_scores=True" in call
    assert "mask=attn_mask_fp32" in call


def _hf_mrope_cos(position_ids, head_dim, theta, mrope_section):
    # transformers' Qwen2.5-VL: emb = cat(freqs, freqs) per axis, then
    # cat([chunk[i % 3] for i, chunk in enumerate(split(mrope_section * 2))]).
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    freqs = position_ids[:, :, None].astype(np.float64) * inv_freq[None, None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    bounds = np.cumsum([0] + list(mrope_section) * 2)
    chunks = [emb[i % 3, :, bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]
    return np.cos(np.concatenate(chunks, axis=-1))


def test_mrope_tables_follow_the_reference_section_interleave() -> None:
    kwargs = dict(max_seq_len=24, head_dim=16, rope_theta=10000.0, mrope_section=[2, 3, 3],
                  image_token_start=3, image_grid_thw=(1, 4, 6), spatial_merge_size=2,
                  tokens_per_second=2)
    cos, _ = text_builder._make_qwen25vl_mrope_full_tables(**kwargs)
    # Reference position ids: text 0..2, image (t, h, w) + 3, text after max+1.
    ids = np.zeros((3, 24), dtype=np.int64)
    ids[:, :3] = np.arange(3)
    h, w = np.meshgrid(np.arange(2), np.arange(3), indexing="ij")
    ids[0, 3:9], ids[1, 3:9], ids[2, 3:9] = 3, h.reshape(-1) + 3, w.reshape(-1) + 3
    ids[:, 9:] = np.arange(6, 6 + 15)
    expected = _hf_mrope_cos(ids, 16, 10000.0, [2, 3, 3])
    np.testing.assert_allclose(cos, expected, rtol=0, atol=1e-6)
