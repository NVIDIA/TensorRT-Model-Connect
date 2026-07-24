# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed tests for the qualified Qwen tied-embedding LM head."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt")

from tensorrt_model_connect.families.qwen.dual_profile_decoder_builder import (
    _qualified_lm_head_reuses_embedding,
)

pytestmark = pytest.mark.dynamic_memory


def _config(*, tied: bool) -> SimpleNamespace:
    return SimpleNamespace(tie_word_embeddings=tied)


def test_qualified_tied_lm_head_reuses_exact_embedding_transpose() -> None:
    embedding = np.arange(24, dtype=np.float16).reshape(6, 4)
    weights = {
        "embedding": embedding,
        "w_out": np.ascontiguousarray(embedding.T),
    }

    assert _qualified_lm_head_reuses_embedding(
        _config(tied=True),
        weights,
    )


def test_qualified_untied_lm_head_keeps_independent_projection() -> None:
    embedding = np.arange(24, dtype=np.float16).reshape(6, 4)
    weights = {
        "embedding": embedding,
        "w_out": np.zeros((4, 6), dtype=np.float16),
    }

    assert not _qualified_lm_head_reuses_embedding(
        _config(tied=False),
        weights,
    )


@pytest.mark.parametrize(
    ("w_out", "message"),
    (
        (np.zeros((5, 6), dtype=np.float16), "incompatible shapes"),
        (np.zeros((4, 6), dtype=np.float32), "incompatible dtypes"),
        (np.zeros((4, 6), dtype=np.float16), "differ from embedding.T"),
    ),
)
def test_qualified_tied_lm_head_rejects_nonidentical_views(
    w_out: np.ndarray,
    message: str,
) -> None:
    embedding = np.arange(24, dtype=np.float16).reshape(6, 4)

    with pytest.raises(ValueError, match=message):
        _qualified_lm_head_reuses_embedding(
            _config(tied=True),
            {
                "embedding": embedding,
                "w_out": w_out,
            },
        )
