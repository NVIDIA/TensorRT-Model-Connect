# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from families.minimax_h3.tests.reference_language_case import ATOL, ROW_COUNTS, RTOL, SEED
from families.minimax_h3.tests.reference_language_localize import compare


def test_language_diagnostic_keeps_frozen_case_and_tolerance():
    assert (SEED, ROW_COUNTS, RTOL, ATOL) == (491, (4, 7, 11), 0.016, 1e-5)
    expected = np.array([0.0, 1.0, -1.0], np.float32)
    actual = expected.copy()
    actual[0] = 2 * ATOL
    result = compare(actual, expected)
    assert result["mismatched_elements"] == 1
    assert result["passes_original_tolerance"] is False
    assert compare(expected, expected)["passes_original_tolerance"] is True


@pytest.mark.parametrize("bad", [np.array([np.nan]), np.array([np.inf]), np.ones((1, 1))])
def test_language_diagnostic_rejects_nonfinite_or_misaligned_values(bad):
    with pytest.raises(AssertionError):
        compare(bad, np.ones((1,), np.float32))
