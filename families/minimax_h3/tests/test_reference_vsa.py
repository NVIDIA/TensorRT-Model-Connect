# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from families.minimax_h3 import reference_vsa as vsa


@pytest.mark.parametrize(
    "segments",
    [
        (),
        [],
        (True, (1, 2, 2)),
        (-1, (1, 2, 2)),
        (1,),
        ((0, 2, 2),),
        ((1, 3, 2),),
        ((1, 2, 3),),
        ((1.0, 2, 2),),
        ((1, 2),),
        (2**31, (1, 2, 2)),
        ((100000, 128, 128),),
        ((1, 2, 2),) * 65,
        (8192 * 128, (1, 2, 2)),
    ],
)
def test_invalid_transport_fails_before_device_use(segments):
    with pytest.raises(ValueError):
        vsa.make_layout(segments)


def test_interleaved_reference_videos_keep_distinct_key_budgets():
    layout = vsa.make_layout(((5, 10, 18), 129, (9, 18, 34), 17, (13, 26, 50)))
    assert layout.prefix_tiles == 3
    assert layout.regions.shape == (3, 3)
    assert np.array_equal(layout.regions[1:, 0], layout.regions[:-1, 1])
    assert layout.regions[:, 2].sum() > 1
    assert not layout.gather_indices.flags.writeable
    assert not layout.regions.flags.writeable


def test_zero_valid_pair_does_not_change_logical_budget():
    layout = vsa.make_layout(((1, 2, 2),))
    assert layout.logical_tiles == 1 and layout.block_sizes.tolist() == [1, 0]
    assert layout.regions.tolist() == [[0, 1, 1]]
    assert layout.untile_indices.tolist() == [0]


@pytest.mark.parametrize("heads,tiles", [(0, 2), (True, 2), (257, 2), (2, 0), (2, True), (2, 8193)])
def test_invalid_workspace_profile_is_rejected(heads, tiles):
    with pytest.raises(ValueError):
        vsa.workspace_bytes(heads, tiles)
