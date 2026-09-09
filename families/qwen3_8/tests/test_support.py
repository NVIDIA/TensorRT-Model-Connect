# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_qwen38_marker_default_task_is_checkpoint_owned() -> None:
    _, support = resolve_family(
        ModelMetadata(
            {
                "model_type": "qwen3_5",
                "text_config": {"output_gate_type": "sigmoid"},
            },
            {},
        )
    )
    assert support.default_task == "text_generation"
