# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_minimax_h3_default_task_is_checkpoint_owned() -> None:
    family, support = resolve_family(ModelMetadata({"model_type": "minimax-h3"}, {}))
    assert family == "minimax_h3"
    assert support.default_task == "text_to_audio_video"
