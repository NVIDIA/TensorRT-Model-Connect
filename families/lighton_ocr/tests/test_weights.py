# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise genuine small safetensors inventories without allocating the full model."""

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from families.lighton_ocr.config import Config
from families.lighton_ocr.weights import Checkpoint, expected_shapes


def tiny_config():
    raw = json.loads((Path(__file__).parent / "data/checkpoint-config.json").read_text())
    raw["text_config"].update(
        hidden_size=8, intermediate_size=12, num_attention_heads=2, num_key_value_heads=1,
        head_dim=4, num_hidden_layers=2, layer_types=["full_attention"] * 2,
        vocab_size=16, max_position_embeddings=32,
    )
    raw["vision_config"].update(
        hidden_size=8, intermediate_size=12, num_attention_heads=2, head_dim=4,
        num_hidden_layers=2, patch_size=2, image_size=8,
    )
    raw.update(image_token_id=15, eos_token_id=2, pad_token_id=0)
    return Config.from_dict(raw)


def tiny_weights(config, dtype=np.float32):
    return {name: (np.arange(np.prod(shape)).reshape(shape) / max(1, np.prod(shape))).astype(dtype)
            for name, shape in expected_shapes(config).items()}


def test_original_inventory_has_all_532_checkpoint_tensors_and_compact_keys():
    config = Config.from_dict(json.loads((Path(__file__).parent / "data/checkpoint-config.json").read_text()))
    shapes = expected_shapes(config)
    assert len(shapes) == 532
    assert shapes["model.language_model.layers.27.self_attn.q_proj.weight"] == (2048, 1024)
    assert shapes["model.language_model.layers.27.self_attn.k_proj.weight"] == (1024, 1024)
    assert shapes["model.language_model.layers.27.self_attn.k_norm.weight"] == (128,)
    assert shapes["model.vision_encoder.transformer.layers.23.attention.q_proj.weight"] == (1024, 1024)
    assert shapes["model.vision_projection.patch_merger.merging_layer.weight"] == (1024, 4096)
    assert "lm_head.weight" not in shapes


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_real_small_checkpoint_loads_original_values_and_partitioned_views(tmp_path, dtype):
    config = tiny_config()
    weights = tiny_weights(config, dtype)
    save_file(weights, tmp_path / "model.safetensors")
    checkpoint = Checkpoint(tmp_path, config)
    checkpoint.validate()
    embedding = checkpoint.embedding()
    np.testing.assert_array_equal(embedding, weights["model.language_model.embed_tokens.weight"].astype(np.float32))
    decoder = checkpoint.decoder(embedding)
    assert decoder["lm_head.weight"] is embedding
    np.testing.assert_array_equal(decoder["layers.1.self_attn.k_proj.weight"], weights["model.language_model.layers.1.self_attn.k_proj.weight"].astype(np.float32))
    vision = checkpoint.vision()
    np.testing.assert_array_equal(vision["vision_projection.linear_2.weight"], weights["model.vision_projection.linear_2.weight"].astype(np.float32))
    assert "embed_tokens.weight" not in decoder
    assert not any(name.startswith("model.") for name in vision)


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape", "integer_storage"])
def test_invalid_inventory_fails_before_tensor_materialization(tmp_path, monkeypatch, mutation):
    config = tiny_config()
    weights = tiny_weights(config)
    name = "model.language_model.layers.1.self_attn.k_proj.weight"
    if mutation == "missing":
        del weights[name]
    elif mutation == "extra":
        weights["unowned.weight"] = np.zeros(1, np.float32)
    elif mutation == "shape":
        weights[name] = np.zeros((1, 1), np.float32)
    else:
        weights[name] = weights[name].astype(np.int32)
    save_file(weights, tmp_path / "model.safetensors")
    checkpoint = Checkpoint(tmp_path, config)

    def no_materialization(_name):
        raise AssertionError("Header validation must reject this inventory without reading tensor values")

    monkeypatch.setattr(checkpoint, "tensor", no_materialization)
    with pytest.raises(ValueError):
        checkpoint.validate()


def test_optional_output_weight_must_be_exactly_tied(tmp_path):
    config = tiny_config()
    weights = tiny_weights(config)
    weights["lm_head.weight"] = weights["model.language_model.embed_tokens.weight"].copy()
    save_file(weights, tmp_path / "model.safetensors")
    Checkpoint(tmp_path, config).validate()
    weights["lm_head.weight"][0, 0] = 1
    save_file(weights, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="exactly tied"):
        Checkpoint(tmp_path, config).validate()


@pytest.mark.parametrize("filename", ["../weights.safetensors", "/tmp/weights.safetensors", "nested/model.safetensors", "weights.bin"])
def test_index_cannot_escape_the_checkpoint_or_select_other_formats(tmp_path, filename):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": filename}}))
    with pytest.raises(ValueError, match="unsafe shard"):
        Checkpoint(tmp_path, tiny_config())


@pytest.mark.parametrize("mutation", ["duplicate", "unowned", "missing"])
def test_sharded_index_must_own_every_tensor_exactly_once(tmp_path, mutation):
    config = tiny_config()
    weights = tiny_weights(config)
    names = sorted(weights)
    midpoint = len(names) // 2
    first = {name: weights[name] for name in names[:midpoint]}
    second = {name: weights[name] for name in names[midpoint:]}
    mapping = {name: "a.safetensors" if name in first else "b.safetensors" for name in names}
    if mutation == "duplicate":
        second[names[0]] = weights[names[0]]
    elif mutation == "unowned":
        first["unowned.weight"] = np.zeros(1, np.float32)
    else:
        mapping["absent.weight"] = "a.safetensors"
    save_file(first, tmp_path / "a.safetensors")
    save_file(second, tmp_path / "b.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises(ValueError):
        Checkpoint(tmp_path, config)
