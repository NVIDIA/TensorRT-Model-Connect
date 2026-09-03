# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tensorrt_model_connect.families.boltz2.diffusion_token_builder import (
    TOKEN_LAYERS,
    define_diffusion_token_network,
)


@pytest.mark.parametrize(
    ("first_layer", "layer_count"),
    [(-1, 6), (0, 0), (18, 7), (24, 1)],
)
def test_diffusion_token_segments_fail_closed(first_layer, layer_count):
    with pytest.raises(ValueError, match="within"):
        define_diffusion_token_network(
            None,
            None,
            {},
            token_count=117,
            first_layer=first_layer,
            layer_count=layer_count,
        )


def test_diffusion_token_partition_is_exact():
    segments = tuple(range(0, TOKEN_LAYERS, 6))
    covered = [layer for start in segments for layer in range(start, start + 6)]
    assert covered == list(range(TOKEN_LAYERS))
