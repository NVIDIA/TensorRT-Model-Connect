# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for ESMC."""

from families.esmc.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_masked_token_scores_are_offered():
    support = describe(ModelMetadata(config={"model_type": "esmc"}, model_index={}))
    assert support is not None
    assert support.tasks == ("masked_text_to_token_scores",)
    assert support.default_task == "masked_text_to_token_scores"


def test_the_architecture_alone_identifies_the_checkpoint():
    # Unlike ESM-2, ESMC has no second model type sharing the name: the
    # architecture in config.json is enough, and a snapshot that carries only
    # that field still resolves to this family.
    support = describe(ModelMetadata(config={"architectures": ["EsmcForMaskedLM"]}, model_index={}))
    assert support is not None
    assert support.default_task == "masked_text_to_token_scores"


def test_the_other_encoders_are_not_claimed():
    # "esm" is the ESM-2 family next door; the rest are the encoders whose
    # configs a loose prefix match would also catch.
    for model_type in ("esm", "bert", "roberta", "deberta-v2", "mpnet", "nomic_bert"):
        assert describe(ModelMetadata(config={"model_type": model_type}, model_index={})) is None
