# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for Stable Audio Open."""

from families.stable_audio.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def _index(pipeline_class: str) -> ModelMetadata:
    return ModelMetadata(config={}, model_index={"_class_name": pipeline_class})


def test_stable_audio_pipeline_offers_text_to_audio():
    # model_index.json of stabilityai/stable-audio-open-1.0 names this pipeline class.
    support = describe(_index("StableAudioPipeline"))
    assert support is not None
    assert support.tasks == ("text_to_audio",)
    assert support.default_task == "text_to_audio"


def test_other_audio_pipelines_are_not_claimed():
    # Stable Audio 3 has a rectified-flow denoiser, a T5Gemma encoder and its own VAE;
    # AudioLDM and MusicLDM are UNet pipelines. None of them is this graph.
    for pipeline_class in ("StableAudio3Pipeline", "AudioLDMPipeline", "AudioLDM2Pipeline",
                           "MusicLDMPipeline", "StableDiffusionPipeline"):
        assert describe(_index(pipeline_class)) is None


def test_transformers_metadata_alone_is_not_claimed():
    # The T5 text encoder inside the checkpoint is not the model.
    assert describe(ModelMetadata(config={"model_type": "t5"}, model_index={})) is None
