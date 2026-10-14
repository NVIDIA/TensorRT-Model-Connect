# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one-to-one reference keeps overlapping detections and filters scores."""

import torch

from families.yolov10.tests.reference.model import detections


def test_end_to_end_head_does_not_suppress_overlapping_boxes():
    raw = torch.tensor([[[1, 2, 3, 4, 0.9, 0], [1, 2, 3, 4, 0.8, 0], [1, 2, 3, 4, 0.1, 1]]])
    torch.testing.assert_close(detections(raw), raw[0, :2])
    assert detections(raw, threshold=1.0).shape == (0, 6)
