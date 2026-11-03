# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover Stable Diffusion checkpoints from their Diffusers pipeline index."""

import json

from families.stable_diffusion.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, load_model_metadata, resolve_family


def _resolve(tmp_path, model_index):
    (tmp_path / "model_index.json").write_text(json.dumps(model_index))
    return resolve_family(load_model_metadata(tmp_path))


def test_sdxl_turbo_resolves_to_this_family(tmp_path):
    # stabilityai/sdxl-turbo@71153311d3dbb46851df1931d3ca6e939de83304 (the sdxl-turbo profile).
    family, support = _resolve(tmp_path, {"_class_name": "StableDiffusionXLPipeline",
                                          "_diffusers_version": "0.24.0.dev0"})
    assert family == "stable_diffusion"
    assert support.default_task == "image_generation"


def test_sd15_still_resolves_to_this_family(tmp_path):
    # stable-diffusion-v1-5/stable-diffusion-v1-5@451f4fe16113bff5a5d2269ed5ad43b0592e9a14.
    family, _ = _resolve(tmp_path, {"_class_name": "StableDiffusionPipeline", "_diffusers_version": "0.6.0"})
    assert family == "stable_diffusion"


def test_other_diffusers_pipelines_are_not_claimed():
    for pipeline in ("FluxPipeline", "StableDiffusion3Pipeline", "PixArtSigmaPipeline"):
        assert describe(ModelMetadata({}, {"_class_name": pipeline})) is None
