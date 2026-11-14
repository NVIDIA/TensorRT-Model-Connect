# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover Stable Audio 3 checkpoints from their native file layout."""

import json

from families.stable_audio_3.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, load_model_metadata, resolve_family

# stabilityai/stable-audio-3-medium@d4d3579395a8ac0df4ab13b5b1d7138ce7290f81, metadata files only.
_LAYOUT = (
    "model_config.json",
    "model.safetensors",
    "t5gemma-b-b-ul2/config.json",
    "t5gemma-b-b-ul2/model.safetensors",
    "t5gemma-b-b-ul2/tokenizer.json",
    "t5gemma-b-b-ul2/tokenizer_config.json",
)


def _snapshot(tmp_path, files):
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model_type": "diffusion_cond_inpaint"}) if name == "model_config.json" else "")
    return load_model_metadata(tmp_path)


def test_stable_audio_3_medium_resolves_to_this_family(tmp_path):
    family, support = resolve_family(_snapshot(tmp_path, _LAYOUT))
    assert family == "stable_audio_3"
    assert support.default_task == "text_to_audio"
    assert support.default_precision == "fp16"


def test_layout_without_the_t5gemma_encoder_is_not_claimed(tmp_path):
    # Stable Audio Open 1.0 ships model_config.json + model.safetensors with a T5 from the Hub.
    assert describe(_snapshot(tmp_path, ("model_config.json", "model.safetensors"))) is None


def test_transformers_and_diffusers_repositories_are_not_claimed():
    files = tuple(_LAYOUT)
    assert describe(ModelMetadata({"model_type": "t5gemma"}, {}, files)) is None
    assert describe(ModelMetadata({}, {"_class_name": "StableAudioPipeline"}, files)) is None
