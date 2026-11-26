# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from families.pp_doclayout_v3.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "pp_doclayout_v3"},
        {"architectures": ["PPDocLayoutV3ForObjectDetection"]},
        {"model_type": "pp_doclayout_v3", "architectures": ["PPDocLayoutV3ForObjectDetection"]},
    ],
)
def test_exact_v3_metadata_resolves_layout_owner(config):
    family, support = resolve_family(ModelMetadata(config, {}))
    assert family == "pp_doclayout_v3"
    assert support.tasks == ("document_layout",)
    assert support.default_task == "document_layout"


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "pp_doclayout_v2", "architectures": ["PPDocLayoutV2ForObjectDetection"]},
        {"model_type": "rt_detr_v2", "architectures": ["RtDetrV2ForObjectDetection"]},
    ],
)
def test_other_detectors_do_not_match(config):
    assert describe(ModelMetadata(config, {})) is None
