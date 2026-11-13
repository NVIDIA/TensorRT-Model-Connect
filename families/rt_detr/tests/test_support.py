# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_original_and_v2_detectors_have_separate_owners():
    for model_type,architecture,family in (
        ('rt_detr','RTDetrForObjectDetection','rt_detr'),
        ('rt_detr_v2','RTDetrV2ForObjectDetection','rt_detr_v2')):
        owner,support=resolve_family(ModelMetadata({'model_type':model_type,'architectures':[architecture]},{}))
        assert owner == family
        assert support.tasks == ('object_detection',)
