# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed shape checks for the family-owned Qwen checkpoint mapper."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from safetensors.numpy import save_file

from families.qwen.checkpoint_mapper import load_standard_weights
from families.qwen.config import ModelConfig


@pytest.fixture
def wrong_embedding_model_dir(tmp_path: Path) -> Path:
    config = {
        "model_type": "standard_decoder",
        "vocab_size": 32,
        "hidden_size": 16,
        "num_hidden_layers": 0,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(
        {
            "model.embed_tokens.weight": np.zeros((64, 16), dtype=np.float32),
            "model.norm.weight": np.ones(16, dtype=np.float32),
            "lm_head.weight": np.zeros((32, 16), dtype=np.float32),
        },
        str(tmp_path / "model.safetensors"),
    )
    return tmp_path


def test_wrong_embedding_shape_raises(wrong_embedding_model_dir: Path) -> None:
    config = ModelConfig.from_dir(wrong_embedding_model_dir)
    with pytest.raises(ValueError, match="Embedding shape"):
        load_standard_weights(wrong_embedding_model_dir, config)


def test_wrong_embedding_shape_raises_with_optimized_python(
    wrong_embedding_model_dir: Path,
) -> None:
    script = """
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[2])

from families.qwen.checkpoint_mapper import load_standard_weights
from families.qwen.config import ModelConfig

model_dir = Path(sys.argv[1])
try:
    load_standard_weights(model_dir, ModelConfig.from_dir(model_dir))
except ValueError as exc:
    if "Embedding shape" not in str(exc):
        raise
else:
    raise RuntimeError("wrong embedding shape was accepted")
"""
    repository = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            script,
            str(wrong_embedding_model_dir),
            str(repository),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
