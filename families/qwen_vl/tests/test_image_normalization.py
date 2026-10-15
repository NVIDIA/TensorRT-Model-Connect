# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt")
from families.qwen_vl.model import _QwenVLModel


@pytest.mark.parametrize("qwen3", [False, True])
def test_bundle_normalization_matches_the_checkpoint_processor(qwen3):
    vision = {"patch_size": 14, "spatial_merge_size": 2}
    if qwen3:
        vision["deepstack_visual_indexes"] = [5, 11, 17]
    config = SimpleNamespace(raw={"vision_config": vision}, hidden_size=16)
    runtime = _QwenVLModel().get_vl_config(config)
    black_white = np.asarray([[0.0] * 3, [1.0] * 3])
    actual = (black_white - runtime["image_mean"]) / runtime["image_std"]
    if qwen3:
        np.testing.assert_allclose(actual, [[-1.0] * 3, [1.0] * 3])
    else:
        mean = np.asarray([0.48145466, 0.4578275, 0.40821073])
        std = np.asarray([0.26862954, 0.26130258, 0.27577711])
        np.testing.assert_allclose(actual, (black_white - mean) / std)
