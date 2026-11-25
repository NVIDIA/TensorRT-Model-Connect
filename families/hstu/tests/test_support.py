# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free discovery and serving policy coverage in Community CPU CI."""

from families.hstu.config import Config
from families.hstu.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_official_dense_identity_and_task():
    support = describe(
        ModelMetadata(
            config={"model_type": "dlrmv3_hstu"}, model_index={}, files=("dlrmv3_weights.npz",)
        )
    )
    assert support is not None
    assert support.default_task == "structured_ranking"
    assert support.tasks == ("structured_ranking",)
    assert support.default_precision == "fp32"
    assert {(v.precision, v.quantization) for v in support.variants} == {
        ("fp32", None),
        ("bf16", None),
    }


def test_unrelated_identity_is_not_claimed():
    assert describe(ModelMetadata(config={"model_type": "unrelated"}, model_index={})) is None


def test_default_precision_policy_is_explicit():
    metadata = Config().runtime(4, "bf16")
    assert metadata["bf16_policy"] == "expanded"
    assert metadata["fp32_block_indices"] == [0, 1, 2]
    assert metadata["bf16_extra_ops"] == [
        "candidate_av",
        "item_mlp",
        "time_projection",
        "prediction_hidden",
    ]
    assert Config().runtime(4, "fp32")["bf16_extra_ops"] == []
