# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_model_connect.model_support import FamilySupport


def describe(metadata):
    config = metadata.config
    if (
        config.get("model_type") == "nomic_bert"
        and config.get("activation_function") == "swiglu"
        and config.get("rotary_emb_fraction") == 1.0
        and config.get("prenorm") is False
        and config.get("n_embd") == 768
        and config.get("n_layer") == 12
    ):
        return FamilySupport(("text_to_embedding",), "text_to_embedding")
    return None
