# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

pytest.importorskip("tensorrt")
from families.internvl.model import _InternVLModel


def test_prompt_template_matches_processor_chat_template():
    # InternVLProcessor.apply_chat_template for one image and one text part:
    # no default system turn, and the image tokens are framed by <img></img>.
    config = SimpleNamespace(raw={"vision_config": {"patch_size": 14}}, hidden_size=16)
    runtime = _InternVLModel().get_vl_config(config)
    rendered = (
        runtime["vl_prompt_template"]
        .replace("{image_pads}", runtime["image_token_str"] * 2)
        .replace("{prompt}", "Who took this photo?")
    )
    assert rendered == (
        "<|im_start|>user\n"
        "<img><IMG_CONTEXT><IMG_CONTEXT></img>\n"
        "Who took this photo?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
