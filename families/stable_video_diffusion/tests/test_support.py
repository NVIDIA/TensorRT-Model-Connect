# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover Stable Video Diffusion checkpoints from their Diffusers pipeline index."""

import json

from families.stable_video_diffusion.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, load_model_metadata, resolve_family


def test_svd_xt_resolves_to_this_family(tmp_path):
    # stabilityai/stable-video-diffusion-img2vid-xt@9e43909513c6714f1bc78bcb44d96e733cd242aa.
    (tmp_path / "model_index.json").write_text(json.dumps(
        {"_class_name": "StableVideoDiffusionPipeline", "_diffusers_version": "0.24.0.dev0"}))
    family, support = resolve_family(load_model_metadata(tmp_path))
    assert family == "stable_video_diffusion"
    assert support.default_task == "initial_image_text_to_video"
    assert support.tasks == ("initial_image_text_to_video",)
    assert support.default_precision == "fp16"


def test_the_pipeline_class_alone_identifies_the_family():
    support = describe(ModelMetadata({}, {"_class_name": "StableVideoDiffusionPipeline"}))
    assert support is not None
    assert support.default_task == "initial_image_text_to_video"


def test_other_diffusers_pipelines_are_not_claimed():
    for pipeline in ("StableDiffusionPipeline", "I2VGenXLPipeline", "WanImageToVideoPipeline",
                     "StableVideoDiffusionPipelineXT"):
        assert describe(ModelMetadata({}, {"_class_name": pipeline})) is None
