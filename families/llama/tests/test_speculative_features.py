# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from ..speculative.features import target_feature_indices


def test_original_llama_eagle3_feature_inputs():
    # HF includes the embedding plus one output per decoder layer. The trained
    # high-level feature is hidden_states[-4], i.e. the input to layer 29.
    hidden_states = tuple(range(33))
    expected = (hidden_states[2], hidden_states[16], hidden_states[-4])
    assert target_feature_indices(32, {}) == expected == (2, 16, 29)


def test_checkpoint_feature_order_is_preserved():
    assert target_feature_indices(32, {"eagle_aux_hidden_state_layer_ids": [29, 2, 16]}) == (29, 2, 16)


@pytest.mark.parametrize("indices", [[], [2, 16], [2, 16, 29, 30], [2, 2, 29],
                                      [-1, 16, 29], [2, 16, 32], [True, 16, 29],
                                      [2, 16, 29.0], "2,16,29"])
def test_invalid_feature_inputs_are_rejected(indices):
    with pytest.raises(ValueError, match="three distinct"):
        target_feature_indices(32, {"eagle_aux_hidden_state_layer_ids": indices})
