# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare one-view native framing with actual upstream media insertion."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt")
pytest.importorskip("transformers.models.internvl.processing_internvl")

from transformers import InternVLProcessor

from families.internvl.model import _InternVLModel


def test_native_prompt_matches_upstream_media_insertion() -> None:
    processor = SimpleNamespace(
        start_image_token="<img>", end_image_token="</img>",
        image_token="<IMG_CONTEXT>", video_token="<video>", image_seq_length=256,
    )
    # A single user message, before the processor expands the image placeholder.
    chat = "<|im_start|>user\n<IMG_CONTEXT>\nQuestion?<|im_end|>\n<|im_start|>assistant\n"
    # Exercise the real upstream insertion step with one already-prepared view.
    # Pixel transforms and video processing are separate from prompt framing.
    official, _, image_count, video_count = InternVLProcessor._insert_media_placeholders(
        processor, text=[chat], image_pixel_values=np.zeros((1, 3, 1, 1)),
        video_pixel_values=None, image_num_patches=[1], video_num_patches=[],
        image_num_patches_indices=np.asarray([1]), video_num_patches_indices=np.asarray([0]),
        video_patch_indices=np.asarray([0]),
    )
    assert image_count == 1 and video_count == 0
    config = _InternVLModel().get_vl_config(
        SimpleNamespace(raw={"vision_config": {"patch_size": 14}}, hidden_size=16)
    )
    prompt = config["vl_prompt_template"].replace("{prompt}", "Question?").replace(
        "{image_pads}", config["image_token_str"] * config["num_image_pad_tokens"]
    )
    assert prompt == official[0]
