# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact checkpoint identity and semantic Task for Table Transformer."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata

STRUCTURE_LABELS = {
    "0": "table",
    "1": "table column",
    "2": "table row",
    "3": "table column header",
    "4": "table projected row header",
    "5": "table spanning cell",
}


def validate_structure_config(raw: dict) -> None:
    """Keep discovery and construction on the structure-recognition contract."""
    if (
        raw.get("model_type") != "table-transformer"
        or raw.get("architectures") != ["TableTransformerForObjectDetection"]
        or raw.get("num_queries") != 125
        or raw.get("num_labels", 6) != 6
        or raw.get("id2label") != STRUCTURE_LABELS
    ):
        raise ValueError("table_transformer requires the 125-query, six-label structure checkpoint")
    if (
        raw.get("backbone") != "resnet18"
        or raw.get("dilation", False)
        or raw.get("position_embedding_type", "sine") != "sine"
        or raw.get("activation_function", "relu") != "relu"
    ):
        raise ValueError(
            "table_transformer requires a non-dilated ResNet-18, sine positions, and ReLU"
        )


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    try:
        validate_structure_config(metadata.config)
    except ValueError:
        return None
    return FamilySupport(tasks=("image_to_boxes",), default_task="image_to_boxes")
