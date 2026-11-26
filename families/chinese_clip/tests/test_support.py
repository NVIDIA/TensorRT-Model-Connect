# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from families.chinese_clip.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_chinese_clip_has_one_owner_and_three_tasks():
    family, support = resolve_family(ModelMetadata({'model_type':'chinese_clip','architectures':['ChineseCLIPModel']},{}))
    assert family == 'chinese_clip'
    assert support.default_task == 'text_image_to_relevance'
    assert support.tasks == ('image_to_embedding','text_to_embedding','text_image_to_relevance')


def test_clip_is_not_claimed_as_chinese_clip():
    assert describe(ModelMetadata({'model_type':'clip','architectures':['CLIPModel']},{})) is None
