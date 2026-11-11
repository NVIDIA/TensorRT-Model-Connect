# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SmolVLM checkpoints have exactly one Model Connect owner and one default task."""

import pytest

from tensorrt_model_connect.model_support import FamilyResolutionError, ModelMetadata, resolve_family


def test_smolvlm2_checkpoint_resolves_to_this_family():
    name, support = resolve_family(ModelMetadata(
        config={"model_type": "smolvlm", "architectures": ["SmolVLMForConditionalGeneration"]},
        model_index={},
    ))
    assert name == "smolvlm"
    assert support.default_task == "images_text_to_text"
    assert support.tasks == ("images_text_to_text",)
    assert support.default_precision == "fp32"


def test_idefics3_checkpoints_are_not_claimed():
    # SmolVLM v1 checkpoints declare the Idefics3 architecture; this family does not own it.
    with pytest.raises(FamilyResolutionError):
        resolve_family(ModelMetadata(
            config={"model_type": "idefics3", "architectures": ["Idefics3ForConditionalGeneration"]},
            model_index={},
        ))
