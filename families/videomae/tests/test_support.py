# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for VideoMAE."""

from families.videomae.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def _metadata(model_type: str, architecture: str) -> ModelMetadata:
    return ModelMetadata(config={"model_type": model_type, "architectures": [architecture]},
                         model_index={})


def test_pretraining_checkpoint_selects_masked_video_reconstruction():
    support = describe(_metadata("videomae", "VideoMAEForPreTraining"))
    assert support is not None
    assert support.tasks == ("masked_video_text_to_video",)
    assert support.default_task == "masked_video_text_to_video"
    assert support.default_precision == "fp32"


def test_heads_without_the_pixel_decoder_are_not_claimed():
    assert describe(_metadata("videomae", "VideoMAEForVideoClassification")) is None
    assert describe(_metadata("videomae", "VideoMAEModel")) is None
    assert describe(_metadata("vit_mae", "ViTMAEForPreTraining")) is None
