# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from families.molmo2.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def metadata():
    return {
        "model_type": "molmo2",
        "architectures": ["Molmo2ForConditionalGeneration"],
        "text_config": {"qk_norm_type": "qwen3", "use_qk_norm": True},
        "vit_config": {"image_default_input_size": [378, 378]},
    }


def test_exact_ownership():
    config = metadata()
    family, support = resolve_family(ModelMetadata(config, {}))
    assert family == "molmo2" and support.default_task == "vision_language_generation"
    for kind, architecture in [
        ("molmo", "MolmoForCausalLM"),
        ("qwen3", "Qwen3ForCausalLM"),
        ("molmo2", "OtherModel"),
    ]:
        assert (
            describe(ModelMetadata({"model_type": kind, "architectures": [architecture]}, {}))
            is None
        )
