# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checkpoint reader takes exactly the tensors the configuration names, from one file or from shards."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from families.granite_speech5.checkpoint import Checkpoint, inventory
from families.granite_speech5.tests.test_config import _resolve, _with_encoder

TINY = {"hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2,
        "num_key_value_heads": 2, "head_dim": 4, "context_size": 4, "max_position_embeddings": 4,
        "conv_kernel_size": 3, "subsample_layers": [0], "vocab_size": 8}


@pytest.fixture()
def tiny(tmp_path):
    config = _with_encoder(**TINY)
    config["vocab_size"] = TINY["vocab_size"]
    cfg = _resolve(tmp_path / "config", config=config)
    tensors = {name: (np.zeros(shape, dtype=np.int64) if name.endswith("num_batches_tracked")
                      else np.full(shape, 0.5, dtype=np.float32)) for name, shape in inventory(cfg).items()}
    return cfg, tensors


def _sharded(directory: Path, shards: dict[str, dict[str, np.ndarray]], weight_map: dict[str, str]) -> Path:
    from safetensors.numpy import save_file

    directory.mkdir(parents=True, exist_ok=True)
    for shard, tensors in shards.items():
        save_file(tensors, str(directory / shard))
    index = {"metadata": {}, "weight_map": weight_map}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return directory


def _split(tensors: dict[str, np.ndarray]) -> tuple[dict, dict, dict[str, str]]:
    names = sorted(tensors)
    first, second = names[: len(names) // 2], names[len(names) // 2:]
    weight_map = {name: "model-00001-of-00002.safetensors" for name in first}
    weight_map.update({name: "model-00002-of-00002.safetensors" for name in second})
    return {name: tensors[name] for name in first}, {name: tensors[name] for name in second}, weight_map


def test_shards_holding_exactly_their_indexed_tensors_load(tiny, tmp_path):
    cfg, tensors = tiny
    first, second, weight_map = _split(tensors)
    directory = _sharded(tmp_path / "sharded", {"model-00001-of-00002.safetensors": first,
                                                "model-00002-of-00002.safetensors": second}, weight_map)
    with Checkpoint(directory, cfg) as checkpoint:
        assert np.all(checkpoint.tensor("encoder.out.weight") == 0.5)


def test_a_shard_tensor_the_index_omits_is_refused(tiny, tmp_path):
    """The single file refuses an extra tensor by name; a shard must too, although its index looks complete."""
    cfg, tensors = tiny
    first, second, weight_map = _split(tensors)
    first = {**first, "encoder.layers.0.extra.weight": np.zeros((2,), dtype=np.float32)}
    directory = _sharded(tmp_path / "extra", {"model-00001-of-00002.safetensors": first,
                                              "model-00002-of-00002.safetensors": second}, weight_map)
    with pytest.raises(NotImplementedError, match=r"unlisted \['encoder\.layers\.0\.extra\.weight'\]"):
        Checkpoint(directory, cfg)


def test_an_index_that_maps_a_tensor_to_the_wrong_shard_is_refused(tiny, tmp_path):
    cfg, tensors = tiny
    first, second, weight_map = _split(tensors)
    moved = next(iter(first))
    weight_map[moved] = "model-00002-of-00002.safetensors"
    directory = _sharded(tmp_path / "moved", {"model-00001-of-00002.safetensors": first,
                                              "model-00002-of-00002.safetensors": second}, weight_map)
    with pytest.raises(NotImplementedError, match="does not hold exactly the tensors the index maps to it"):
        Checkpoint(directory, cfg)
