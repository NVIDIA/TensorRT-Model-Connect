# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The E2E token oracles accept exactly what they claim to accept."""

import numpy as np
import pytest

from families.smolvlm.tests.oracle import first_divergence, reference_disagreements


def _logits(*rows):
    return [np.asarray(row, dtype=np.float32) for row in rows]


def test_first_divergence():
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([1, 2, 4], [1, 2, 3]) == 2
    assert first_divergence([1, 2], [1, 2, 3]) == 2
    assert first_divergence([], []) is None


def test_reference_disagreements_report_every_step_with_its_gap():
    logits = _logits([0.0, 2.0, 1.0], [3.0, 2.95, 0.0], [0.0, 0.0, 5.0], [1.0, 1.0, 0.0])
    # Step 0 agrees, step 1 takes the runner-up 0.05 behind, step 2 takes a token 5 behind,
    # step 3 takes one of two tied maxima (agreement).
    rows = reference_disagreements([1, 1, 0, 1], logits)
    assert [row["step"] for row in rows] == [1, 2]
    assert rows[0]["reference_token"] == 0 and rows[0]["native_token"] == 1
    assert rows[0]["logit_gap"] == pytest.approx(0.05, abs=1e-6)
    assert rows[1]["logit_gap"] == pytest.approx(5.0)
    # Agreement is judged step by step on the native prefix, not against one fixed sequence.
    assert reference_disagreements([1, 0, 2, 0], logits) == []


def test_reference_disagreements_need_one_row_per_token():
    with pytest.raises(ValueError):
        reference_disagreements([1, 2], _logits([0.0, 1.0, 0.0]))


def test_reference_disagreements_reject_non_finite_reference_logits():
    # A NaN would otherwise win argmax and hide a native token far behind the finite maximum.
    with pytest.raises(ValueError):
        reference_disagreements([0], _logits([0.0, float("nan"), 5.0]))
    with pytest.raises(ValueError):
        reference_disagreements([2], _logits([0.0, float("inf"), 5.0]))
