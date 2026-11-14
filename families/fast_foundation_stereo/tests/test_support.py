# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_model_connect.model_support import ModelMetadata

from families.fast_foundation_stereo.support import describe


def test_prepared_official_model_without_hugging_face_config_is_supported() -> None:
    support = describe(
        ModelMetadata({}, {}, ("weights/23-36-37/model_best_bp2_serialize.pth",))
    )

    assert support is not None
    assert support.default_task == "stereo_disparity"


def test_unrelated_rootless_checkpoint_is_not_supported() -> None:
    assert describe(ModelMetadata({}, {}, ("model.pt",))) is None
