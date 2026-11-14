# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for ESM-2."""

from families.esm.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_masked_token_scores_are_offered():
    support = describe(ModelMetadata(config={"model_type": "esm"}, model_index={}))
    assert support is not None
    assert support.tasks == ("masked_text_to_token_scores",)
    assert support.default_task == "masked_text_to_token_scores"


def test_the_folding_checkpoints_share_the_model_type():
    # ESMFold publishes model_type "esm" too, so discovery alone cannot tell
    # it apart. Nothing is refused here; config.resolve is what refuses the
    # folding config, because a structure predictor is a different model.
    assert describe(ModelMetadata(config={"model_type": "esm", "is_folding_model": True},
                                  model_index={})) is not None


def test_the_other_encoders_are_not_claimed():
    for model_type in ("bert", "roberta", "deberta-v2", "mpnet", "nomic_bert", "esmc"):
        assert describe(ModelMetadata(config={"model_type": model_type},
                                      model_index={})) is None
