# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from tests.e2e.models.dinov3.e2e_plugins.knn import (
    normalize_rows,
    weighted_knn_predictions,
)


def test_weighted_knn_uses_normalized_features_and_multiple_k_values() -> None:
    bank = np.vstack(
        [
            np.tile([1.0, 0.0, 0.0], (4, 1)),
            np.tile([0.0, 1.0, 0.0], (4, 1)),
            np.tile([0.0, 0.0, 1.0], (4, 1)),
        ]
    )
    labels = np.repeat([0, 1, 2], 4)
    queries = np.asarray([[9.0, 0.1, 0.0], [0.0, 2.0, 0.1]], dtype=np.float32)

    predictions = weighted_knn_predictions(
        bank,
        labels,
        queries,
        num_classes=3,
        ks=(1, 4, 10),
    )

    assert predictions == {"1": [0, 1], "4": [0, 1], "10": [0, 1]}


def test_normalize_rows_rejects_zero_norm_features() -> None:
    with pytest.raises(ValueError, match="zero-norm"):
        normalize_rows([[0.0, 0.0]])
