# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free identity and default-task contract for the trained encoder."""

from families.hstu_research.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_trained_checkpoint_identity_and_default_task():
    support = describe(ModelMetadata(config={"model_type": "hstu_meta_research"}, model_index={}))
    assert support is not None
    assert support.tasks == ("structured_ranking",)
    assert support.default_task == "structured_ranking"
    assert tuple(variant.precision for variant in support.variants) == ("fp32", "bf16")


def test_other_checkpoint_schemas_are_not_claimed():
    for identity in ("hstu_public_ranking", "hstu", "unrelated-model"):
        assert describe(ModelMetadata(config={"model_type": identity}, model_index={})) is None


def test_adapter_weight_filename_is_not_claimed():
    assert describe(ModelMetadata(config={}, model_index={}, files=("hstu_weights.npz",))) is None


def test_converted_checkpoint_has_one_discovery_owner():
    metadata = ModelMetadata(
        config={"model_type": "hstu_meta_research"},
        model_index={},
        files=("hstu_research_weights.npz",),
    )
    owner, support = resolve_family(metadata)
    assert owner == "hstu_research" and support.default_task == "structured_ranking"
