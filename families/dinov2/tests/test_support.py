# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


@pytest.mark.parametrize(
    ("model_type", "architecture"),
    [
        ("dinov2", "Dinov2Model"),
        ("dinov2_with_registers", "Dinov2WithRegistersModel"),
    ],
)
def test_dinov2_owns_exact_encoder_identities(model_type: str, architecture: str) -> None:
    family, support = resolve_family(
        ModelMetadata({"model_type": model_type, "architectures": [architecture]}, {})
    )

    assert family == "dinov2"
    assert support.tasks == ("image_to_token_and_pooled_features",)
    assert support.default_task == "image_to_token_and_pooled_features"


def test_dinov2_does_not_claim_dinov3() -> None:
    family, _ = resolve_family(
        ModelMetadata({"model_type": "dinov3_vit", "architectures": ["DINOv3ViTModel"]}, {})
    )

    assert family == "dinov3"
