# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free checkpoint identity and default-task assertions.

No TensorRT or torch import here; this must run on the CPU-only public gate.
"""

from tensorrt_model_connect.model_support import ModelMetadata

from families.depth_anything_v2.support import describe


_REAL_CONFIG = {
    "architectures": ["DepthAnythingForDepthEstimation"],
    "model_type": "depth_anything",
    "backbone_config": {
        "architectures": ["Dinov2Model"],
        "hidden_size": 384,
        "image_size": 518,
        "model_type": "dinov2",
        "num_attention_heads": 6,
        "out_features": ["stage3", "stage6", "stage9", "stage12"],
        "out_indices": [3, 6, 9, 12],
        "patch_size": 14,
        "reshape_hidden_states": False,
    },
    "fusion_hidden_size": 64,
    "head_hidden_size": 32,
    "neck_hidden_sizes": [48, 96, 192, 384],
    "reassemble_factors": [4, 2, 1, 0.5],
    "reassemble_hidden_size": 384,
}


def test_depth_anything_v2_config_resolves_to_this_family():
    metadata = ModelMetadata(
        config=_REAL_CONFIG,
        model_index={},
        files=("config.json", "model.safetensors", "preprocessor_config.json"),
    )
    support = describe(metadata)
    assert support is not None
    assert support.default_task == "monocular_geometry"
    assert support.tasks == ("monocular_geometry",)


def test_unrelated_config_does_not_resolve_to_this_family():
    metadata = ModelMetadata(
        config={"model_type": "bert", "architectures": ["BertModel"]}, model_index={}, files=()
    )
    assert describe(metadata) is None
