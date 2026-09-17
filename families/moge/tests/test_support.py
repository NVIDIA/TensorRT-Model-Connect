# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_rootless_moge_default_task_is_checkpoint_owned() -> None:
    family, support = resolve_family(ModelMetadata({}, {}, ("model.pt",)))
    assert family == "moge"
    assert support.default_task == "monocular_geometry"
