# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_wav2vec2_bert_checkpoint_has_one_owner():
    family, support = resolve_family(ModelMetadata({"model_type": "wav2vec2-bert"}, {}))
    assert family == "wav2vec2_bert"
    assert support.tasks == ("spectrogram_to_frame_features",)
    assert support.default_task == "spectrogram_to_frame_features"
