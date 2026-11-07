# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the published larger-checkpoint contracts without loading weights."""

import json
from pathlib import Path
import shutil

import pytest

from families.qwen.embedding import validate_request
from tensorrt_model_connect import BuildRequest
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


@pytest.fixture(params=[("4b", 2560), ("8b", 4096)])
def checkpoint(request, tmp_path):
    size, dimension = request.param
    source = Path(__file__).parent / "fixtures" / f"qwen3-embedding-{size}"
    root = tmp_path / "checkpoint"
    shutil.copytree(source, root)
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    return root, dimension


def _request(root):
    return BuildRequest(
        model_dir=root,
        output_path=root / "fixture.bundle",
        family="qwen",
        task="embedding",
        precision="bf16",
        max_sequence_length=256,
    )


def test_larger_checkpoint_contract_and_unique_owner(checkpoint):
    root, dimension = checkpoint
    config, contract, length = validate_request(_request(root))
    assert contract.embedding_dimension == config.hidden_size == dimension
    assert contract.eos_token_id == 151645
    assert contract.pooling == "last_token" and contract.normalize
    assert length == 256
    metadata = ModelMetadata(
        config=json.loads((root / "config.json").read_text(encoding="utf-8")),
        model_index={},
        files=("modules.json", "1_Pooling/config.json"),
    )
    family, support = resolve_family(metadata)
    assert family == "qwen" and support.tasks == ("embedding",)


@pytest.mark.parametrize("field", ["num_hidden_layers", "num_attention_heads", "eos_token_id"])
def test_larger_checkpoint_rejects_unknown_topology(checkpoint, field):
    root, _ = checkpoint
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[field] += 1
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="sentence-transformers"):
        validate_request(_request(root))


def test_larger_checkpoint_rejects_wrong_pooling_dimension(checkpoint):
    root, dimension = checkpoint
    path = root / "1_Pooling/config.json"
    pooling = json.loads(path.read_text(encoding="utf-8"))
    pooling["word_embedding_dimension"] = dimension - 1
    path.write_text(json.dumps(pooling), encoding="utf-8")
    with pytest.raises(ValueError, match="sentence-transformers"):
        validate_request(_request(root))
