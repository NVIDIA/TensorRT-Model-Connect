# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free identity/default checks; no trained-model qualification."""

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_h3_config_aliases_have_one_owner_and_original_default():
    for model_type in ("minimax-h3", "minimax_h3", "minimaxh3"):
        family, support = resolve_family(
            ModelMetadata(config={"model_type": model_type}, model_index={})
        )
        assert family == "minimax_h3"
        assert support.default_task == "image_generation"
        assert support.tasks == ("image_generation",)


def test_h3_pipeline_metadata_has_original_owner_and_default():
    for pipeline in ("MiniMaxH3ModularPipeline", "MiniMaxH3Pipeline"):
        family, support = resolve_family(
            ModelMetadata(config={}, model_index={"_class_name": pipeline})
        )
        assert family == "minimax_h3"
        assert support.default_task == "image_generation"
        assert support.tasks == ("image_generation",)


def test_reference_files_do_not_expand_unqualified_automatic_tasks():
    metadata = ModelMetadata(
        config={"model_type": "minimax-h3"},
        model_index={},
        files=("transformer_ref/config.json", "text_encoder/config.json", "vae/config.json"),
    )
    family, support = resolve_family(metadata)
    assert family == "minimax_h3"
    assert support.default_task == "image_generation"
    assert support.tasks == ("image_generation",)
