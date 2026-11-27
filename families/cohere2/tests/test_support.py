# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_support_resolves_to_parallel_cohere_decoder():
    family, support = resolve_family(
        ModelMetadata(
            config={"model_type": "cohere2", "architectures": ["Cohere2ForCausalLM"]},
            model_index={},
        )
    )
    assert family == "cohere2"
    assert support.tasks == ("text_continuation",)
    assert support.default_task == "text_continuation"
    assert support.default_precision == "fp32"
