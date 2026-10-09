# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from safetensors.numpy import save_file


REPO_ROOT = Path(__file__).resolve().parents[3]


def _checkpoint(directory: Path, prefix: str, sharded: bool, malformed: str | None = None):
    config = {
        "model_type": "bert",
        "vocab_size": 8,
        "hidden_size": 4,
        "num_hidden_layers": 0,
        "num_attention_heads": 2,
        "intermediate_size": 8,
        "max_position_embeddings": 6,
        "type_vocab_size": 2,
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "word_embeddings": np.arange(32, dtype=np.float32).reshape(8, 4),
        "position_embeddings": np.arange(24, dtype=np.float32).reshape(6, 4),
        "token_type_embeddings": np.arange(8, dtype=np.float32).reshape(2, 4),
    }
    expected = {
        "embedding": tensors["word_embeddings"].tolist(),
        "position_embedding": tensors["position_embeddings"].tolist(),
        "token_type_embedding": tensors["token_type_embeddings"].tolist(),
        "embed_norm": [1.0] * 4,
        "embed_norm_beta": [0.0] * 4,
    }
    if malformed is not None:
        tensors[malformed] = tensors[malformed][:-1].copy()
    checkpoint = {f"{prefix}embeddings.{name}.weight": tensor for name, tensor in tensors.items()}
    checkpoint[f"{prefix}embeddings.LayerNorm.weight"] = np.ones(4, dtype=np.float32)
    checkpoint[f"{prefix}embeddings.LayerNorm.bias"] = np.zeros(4, dtype=np.float32)
    if sharded:
        weight_map = {}
        entries = list(checkpoint.items())
        for index in range(2):
            filename = f"model-{index + 1:05d}-of-00002.safetensors"
            shard = dict(entries[index::2])
            save_file(shard, directory / filename)
            weight_map.update({name: filename for name in shard})
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}), encoding="utf-8"
        )
    else:
        save_file(checkpoint, directory / "model.safetensors")
    return expected


def _load(directory: Path, optimized: bool):
    script = """
import json
import sys
from families.bert.config import ModelConfig
from families.bert.weights import load_bert_weights
weights = load_bert_weights(sys.argv[1], ModelConfig.from_dir(sys.argv[1]))
print(json.dumps({name: value.tolist() for name, value in weights.items()}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if part
    )
    return subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), "-c", script, str(directory)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize("optimized", [False, True], ids=["normal", "optimized"])
@pytest.mark.parametrize("prefix", ["", "bert."], ids=["bare", "prefixed"])
@pytest.mark.parametrize("sharded", [False, True], ids=["single", "sharded"])
@pytest.mark.parametrize(
    ("malformed", "diagnostic"),
    [
        ("word_embeddings", "Embedding shape"),
        ("position_embeddings", "Position embedding shape"),
        ("token_type_embeddings", "Token type embedding shape"),
    ],
)
def test_bert_rejects_malformed_checkpoint_shapes(
    tmp_path: Path, optimized: bool, prefix: str, sharded: bool, malformed: str, diagnostic: str
) -> None:
    _checkpoint(tmp_path, prefix, sharded, malformed)
    result = _load(tmp_path, optimized)
    assert result.returncode != 0, "accepted a malformed BERT checkpoint"
    assert "ValueError" in result.stderr
    assert diagnostic in result.stderr


@pytest.mark.parametrize("optimized", [False, True], ids=["normal", "optimized"])
@pytest.mark.parametrize("prefix", ["", "bert."], ids=["bare", "prefixed"])
@pytest.mark.parametrize("sharded", [False, True], ids=["single", "sharded"])
def test_bert_preserves_valid_checkpoint_mapping(
    tmp_path: Path, optimized: bool, prefix: str, sharded: bool
) -> None:
    expected = _checkpoint(tmp_path, prefix, sharded)
    result = _load(tmp_path, optimized)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected
