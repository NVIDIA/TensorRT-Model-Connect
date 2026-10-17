# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from families.nomic_bert.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_metadata_identity_and_default_task():
    config = {
        "model_type": "nomic_bert",
        "activation_function": "swiglu",
        "rotary_emb_fraction": 1.0,
        "prenorm": False,
        "n_embd": 768,
        "n_layer": 12,
    }
    family, support = resolve_family(ModelMetadata(config, {}))
    assert family == "nomic_bert"
    assert support.default_task == "text_to_embedding"
    assert support.tasks == ("text_to_embedding",)
    assert describe(ModelMetadata({**config, "model_type": "bert"}, {})) is None
    assert describe(ModelMetadata({**config, "prenorm": True}, {})) is None
