# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from families.jina_embeddings_v3.model import _validate_identity
from families.jina_embeddings_v3.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


@pytest.mark.parametrize("model_type", ["jina_embeddings_v3", "Jina-Embeddings-V3"])
def test_builder_accepts_the_identity_accepted_by_discovery(model_type):
    cfg = {"model_type": model_type, "hidden_act": "gelu"}
    assert describe(ModelMetadata(cfg, {})) is not None
    _validate_identity(cfg)


def test_builder_still_rejects_other_activations_and_models():
    with pytest.raises(ValueError, match="GELU"):
        _validate_identity({"model_type": "jina_embeddings_v3", "hidden_act": "relu"})
    with pytest.raises(ValueError, match="checkpoint"):
        _validate_identity({"model_type": "bert", "hidden_act": "gelu"})
