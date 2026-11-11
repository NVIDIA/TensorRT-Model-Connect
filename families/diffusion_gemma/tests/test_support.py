# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from families.diffusion_gemma.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_exact_checkpoint_resolves_to_diffusion_owner_and_bf16():
    config = json.loads((Path(__file__).parent / "data/diffusiongemma-26b-a4b-config.json").read_text())
    family, support = resolve_family(ModelMetadata(config=config, model_index={}))
    assert family == "diffusion_gemma"
    assert support.tasks == ("text_continuation",)
    assert support.default_task == "text_continuation"
    assert support.default_precision == "bf16"


@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_text", "gemma3", "diffusion_gemma_text", "diffusion_gemma_other"])
def test_sibling_and_prefix_identities_are_not_claimed(model_type):
    assert describe(ModelMetadata(config={"model_type": model_type}, model_index={})) is None


def test_exact_architecture_is_recognized_without_a_repository_name():
    metadata = ModelMetadata(config={"architectures": ["DiffusionGemmaForBlockDiffusion"]}, model_index={})
    assert describe(metadata).default_task == "text_continuation"
