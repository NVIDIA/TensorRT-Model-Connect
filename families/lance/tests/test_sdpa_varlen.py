# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SDPA stand-in for flash-attn's varlen attention used by the native Lance reference."""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from families.lance.tests.benchmark import upstream_reference as reference  # noqa: E402


def _expected(q, k, v, bounds_q, bounds_k, causal):
    """Per-sequence softmax(QK^T / sqrt(d)) V with grouped heads and a bottom-right causal mask."""
    output = torch.zeros_like(q)
    groups = q.shape[1] // k.shape[1]
    for index in range(len(bounds_q) - 1):
        q_start, q_end = bounds_q[index], bounds_q[index + 1]
        k_start, k_end = bounds_k[index], bounds_k[index + 1]
        offset = (k_end - k_start) - (q_end - q_start)
        for head in range(q.shape[1]):
            query = q[q_start:q_end, head]
            key, value = k[k_start:k_end, head // groups], v[k_start:k_end, head // groups]
            scores = query @ key.T / math.sqrt(q.shape[2])
            if causal:
                for row in range(scores.shape[0]):
                    scores[row, row + offset + 1 :] = float("-inf")
            output[q_start:q_end, head] = torch.softmax(scores, dim=-1) @ value
    return output


@pytest.mark.parametrize("causal", [False, True])
def test_sdpa_varlen_matches_per_sequence_attention(causal: bool) -> None:
    generator = torch.Generator().manual_seed(0)
    # Two sequences: 3 new queries over 7 keys (a cached prefix of 4), then 5 queries over 5 keys.
    bounds_q, bounds_k = [0, 3, 8], [0, 7, 12]
    q = torch.randn(8, 4, 16, generator=generator)
    k = torch.randn(12, 2, 16, generator=generator)
    v = torch.randn(12, 2, 16, generator=generator)

    got = reference._sdpa_varlen_attention(
        q,
        k,
        v,
        torch.tensor(bounds_q, dtype=torch.int32),
        torch.tensor(bounds_k, dtype=torch.int32),
        max_seqlen_q=5,
        max_seqlen_k=7,
        causal=causal,
    )

    torch.testing.assert_close(
        got, _expected(q, k, v, bounds_q, bounds_k, causal), rtol=1e-5, atol=1e-5
    )


def test_fallback_replaces_the_language_model_attention_only_when_flash_attn_cannot_run(
    monkeypatch,
) -> None:
    navit = SimpleNamespace(flash_attn_varlen_func="flash")
    monkeypatch.setitem(sys.modules, "modeling.lance.qwen2_navit", navit)

    monkeypatch.setattr(reference, "_flash_attention_runs", lambda: True)
    assert not reference._use_sdpa_without_flash_kernels()
    assert navit.flash_attn_varlen_func == "flash"

    monkeypatch.setattr(reference, "_flash_attention_runs", lambda: False)
    assert reference._use_sdpa_without_flash_kernels()
    assert navit.flash_attn_varlen_func is reference._sdpa_varlen_attention
