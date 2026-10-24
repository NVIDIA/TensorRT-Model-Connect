# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from transformers import (
    CamembertConfig,
    CamembertModel,
    RobertaConfig,
    RobertaModel,
    XLMRobertaConfig,
    XLMRobertaModel,
)

from families.roberta.config import ModelConfig
from families.roberta.model import _RobertaModel


_MODEL_CLASSES = (
    (RobertaConfig, RobertaModel),
    (XLMRobertaConfig, XLMRobertaModel),
    (CamembertConfig, CamembertModel),
)


def _checkpoint(tmp_path, config_class, model_class, namespace, *, pooler=True):
    config = config_class(
        vocab_size=19,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=18,
        type_vocab_size=1,
        pad_token_id=1,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = model_class(config, add_pooling_layer=pooler)
    model.save_pretrained(tmp_path)
    state = model.state_dict()
    if namespace:
        save_file(
            {namespace + key: value for key, value in state.items()},
            str(tmp_path / "model.safetensors"),
        )
    return state, ModelConfig.from_dir(tmp_path)


def _expected_weights(state):
    expected = {}
    for source, target in (
        ("embeddings.word_embeddings.weight", "embedding"),
        ("embeddings.token_type_embeddings.weight", "token_type_embedding"),
        ("embeddings.LayerNorm.weight", "embed_norm"),
        ("embeddings.LayerNorm.bias", "embed_norm_beta"),
    ):
        expected[target] = state[source].numpy()
    expected["position_embedding"] = state["embeddings.position_embeddings.weight"].numpy()[2:]
    for layer in range(2):
        root = f"encoder.layer.{layer}."
        for source, target, transpose in (
            ("attention.self.query.weight", "w_q", True),
            ("attention.self.key.weight", "w_k", True),
            ("attention.self.value.weight", "w_v", True),
            ("attention.self.query.bias", "q_bias", False),
            ("attention.self.key.bias", "k_bias", False),
            ("attention.self.value.bias", "v_bias", False),
            ("attention.output.dense.weight", "w_o", True),
            ("attention.output.dense.bias", "o_bias", False),
            ("attention.output.LayerNorm.weight", "post_attn_norm", False),
            ("attention.output.LayerNorm.bias", "post_attn_norm_beta", False),
            ("intermediate.dense.weight", "w_fc1", True),
            ("intermediate.dense.bias", "fc1_bias", False),
            ("output.dense.weight", "w_fc2", True),
            ("output.dense.bias", "fc2_bias", False),
            ("output.LayerNorm.weight", "output_norm", False),
            ("output.LayerNorm.bias", "output_norm_beta", False),
        ):
            value = state[root + source].numpy()
            expected[f"layer.{layer}.{target}"] = value.T if transpose else value
    if "pooler.dense.weight" in state:
        expected["pooler_w"] = state["pooler.dense.weight"].numpy().T
        expected["pooler_bias"] = state["pooler.dense.bias"].numpy()
    return expected


@pytest.mark.parametrize("config_class,model_class", _MODEL_CLASSES)
@pytest.mark.parametrize("namespace", ["", "roberta.", "model.roberta."])
@pytest.mark.parametrize("pooler", [True, False])
def test_load_weights_preserves_saved_encoder_tensors(
    tmp_path, config_class, model_class, namespace, pooler
):
    state, config = _checkpoint(tmp_path, config_class, model_class, namespace, pooler=pooler)

    weights = _RobertaModel().load_weights(str(tmp_path), config)

    expected = _expected_weights(state)
    assert weights.keys() == expected.keys()
    for key, value in expected.items():
        np.testing.assert_array_equal(weights[key], value, err_msg=key)
        assert weights[key].dtype == np.float32
        assert weights[key].flags.c_contiguous
