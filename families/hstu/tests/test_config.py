# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Malformed checkpoint contracts must fail before engine construction."""

from __future__ import annotations

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from families.hstu.config import parse_config
from families.hstu.model import load_weights
from families.hstu.tests.fixtures import make_checkpoint, tiny_config


@pytest.mark.parametrize("changes", [
    {"schema_version": 2}, {"model_type": "bert"}, {"num_heads": 0},
    {"head_dim": True}, {"hidden_size": 1.5}, {"is_causal": "false"},
    {"scaling_seqlen": 0}, {"scaling_seqlen": -2},
    {"layer_norm_epsilon": float("nan")}, {"output_norm_epsilon": 0},
    {"time_buckets": 8}, {"time_buckets": 2048, "position_buckets": 0},
    {"prediction_head": []}, {"prediction_activation": "sigmoid"},
    {"target_group_size": 0}, {"unknown_attention_option": True},
    {"output_postprocessor": "layer_norm"},
])
def test_rejects_unrepresented_or_invalid_semantics(changes):
    with pytest.raises(ValueError):
        parse_config({**tiny_config(), **changes})


def test_table_names_and_roles_are_unambiguous():
    config = tiny_config()
    config["embedding_tables"][1]["name"] = "item"
    with pytest.raises(ValueError, match="unique identifiers"):
        parse_config(config)
    config = tiny_config()
    config["embedding_tables"][1]["role"] = "item"
    with pytest.raises(ValueError, match="exactly one item"):
        parse_config(config)


@pytest.mark.parametrize("corruption", ["missing", "unknown", "shape", "nan", "unsorted_keys", "duplicate_keys", "key_dtype"])
def test_rejects_corrupt_checkpoint(tmp_path, corruption):
    config = make_checkpoint(tmp_path)
    path = tmp_path / "model.safetensors"
    tensors = load_file(path)
    if corruption == "missing":
        del tensors["blocks.0.uvqk.weight"]
    elif corruption == "unknown":
        tensors["blocks.0.relative_attention_bias"] = np.ones((1,), np.float32)
    elif corruption == "shape":
        tensors["blocks.0.uvqk.weight"] = tensors["blocks.0.uvqk.weight"].T.copy()
    elif corruption == "nan":
        tensors["head.0.weight"][0, 0] = np.nan
    elif corruption == "unsorted_keys":
        tensors["embeddings.item.keys"] = tensors["embeddings.item.keys"][::-1].copy()
    elif corruption == "duplicate_keys":
        tensors["embeddings.item.keys"][1] = tensors["embeddings.item.keys"][0]
    else:
        tensors["embeddings.item.keys"] = tensors["embeddings.item.keys"].astype(np.float32)
    save_file(tensors, str(path))
    with pytest.raises(ValueError):
        load_weights(tmp_path, config)


def test_large_raw_ids_survive_checkpoint_loading(tmp_path):
    config = make_checkpoint(tmp_path)
    weights = load_weights(tmp_path, config)
    keys = weights["embeddings.item.keys"]
    assert keys.dtype == np.int64
    assert keys[0] == 2**40
    assert keys[1] - keys[0] == 17
