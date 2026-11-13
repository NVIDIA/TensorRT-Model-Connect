# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from families.longformer.model import local_attention_mask
from tensorrt_model_connect.model_support import ModelMetadata,resolve_family


def test_longformer_has_one_owner():
    family,support = resolve_family(ModelMetadata({'model_type':'longformer'},{}))
    assert family == 'longformer'
    assert support.tasks == ('text_to_token_features',)


def test_window_boundary_includes_half_window_on_both_sides():
    mask = local_attention_mask(1024,512)[0,0]
    assert np.all(mask[512,256:769] == 0)
    assert mask[512,255] < -1e8 and mask[512,769] < -1e8
    assert np.all(mask[0,:257] == 0) and mask[0,257] < -1e8
    for window in (0,-2,3):
        with pytest.raises(ValueError,match='positive and even'):
            local_attention_mask(1024,window)
