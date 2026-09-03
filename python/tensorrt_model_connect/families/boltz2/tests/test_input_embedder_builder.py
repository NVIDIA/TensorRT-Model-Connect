# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from tensorrt_model_connect.families.boltz2.input_embedder_builder import (
    ATOM_COUNT,
    ATOM_WINDOW_KEYS,
    ATOM_WINDOW_QUERIES,
    ATOM_WINDOWS,
    _indexing_matrix,
    atom_attention_detail_shapes,
    atom_windows,
)


def test_atom_key_indexing_matches_boltz_window_equation():
    matrix = _indexing_matrix()
    assert matrix.shape == (2 * ATOM_WINDOWS, 8 * ATOM_WINDOWS)
    single = np.arange(ATOM_COUNT, dtype=np.float32).reshape(
        2 * ATOM_WINDOWS, ATOM_WINDOW_QUERIES // 2
    )
    actual = np.einsum("ji,jk->ki", single, matrix).reshape(ATOM_WINDOWS, ATOM_WINDOW_KEYS)

    # Interior query windows contain their own 32 atoms plus the nearest 48
    # atoms on either side. Boundary key slots are zero padded by Boltz.
    middle = ATOM_WINDOWS // 2
    assert np.array_equal(
        actual[middle],
        np.arange(
            middle * ATOM_WINDOW_QUERIES - 48,
            middle * ATOM_WINDOW_QUERIES + 80,
            dtype=np.float32,
        ),
    )
    assert np.count_nonzero(actual[0, :48]) == 0
    assert np.array_equal(actual[0, 48:], np.arange(80, dtype=np.float32))


def test_atom_profile_is_integral_windows():
    assert ATOM_COUNT % ATOM_WINDOW_QUERIES == 0
    assert ATOM_WINDOW_KEYS % (ATOM_WINDOW_QUERIES // 2) == 0


@pytest.mark.parametrize("atom_count", [0, 927, 929])
def test_input_embedder_rejects_nonintegral_atom_windows(atom_count):
    # The public builder validates this before touching TensorRT network state.
    from tensorrt_model_connect.families.boltz2 import input_embedder_builder

    with pytest.raises(ValueError, match="positive multiple of 32"):
        input_embedder_builder.define_input_embedder_network(
            None,
            None,
            {},
            token_count=117,
            atom_count=atom_count,
        )


@pytest.mark.parametrize("atom_count", [320, 928, 1024])
def test_atom_window_contract_scales_with_processed_shape(atom_count):
    windows = atom_windows(atom_count)
    assert windows == atom_count // ATOM_WINDOW_QUERIES
    assert _indexing_matrix(windows).shape == (2 * windows, 8 * windows)
    assert atom_attention_detail_shapes(atom_count)["query"][0] == windows
