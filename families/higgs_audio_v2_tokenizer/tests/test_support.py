# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_higgs_audio_v2_tokenizer_checkpoint_has_one_owner():
    family, support = resolve_family(ModelMetadata({"model_type": "higgs_audio_v2_tokenizer"}, {}))
    assert family == "higgs_audio_v2_tokenizer"
    assert support.tasks == ("audio_codebook_tokens_to_audio",)
