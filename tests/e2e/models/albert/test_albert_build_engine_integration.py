# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned build_engine integration tests for albert."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.trt]

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file
    from tensorrt_model_connect.config import ModelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _trt_available() -> bool:
    try:
        import tensorrt as trt  # noqa: F401
        try:
            from cuda.bindings import runtime as cudart  # noqa: F401
        except ImportError:
            from cuda import cudart  # type: ignore[no-redef]  # noqa: F401
        return True
    except ImportError:
        return False

requires_trt = pytest.mark.skipif(
    not _trt_available(), reason="TensorRT + CUDA not available"
)

RNG = np.random.RandomState(789)


def _rand(*shape: int) -> np.ndarray:
    return RNG.randn(*shape).astype(np.float32)


def _write_config(model_dir: Path, config: dict) -> None:
    (model_dir / "config.json").write_text(json.dumps(config))


def _write_safetensors(model_dir: Path, tensors: dict[str, np.ndarray],
                       filename: str = "model.safetensors") -> None:
    save_file(tensors, str(model_dir / filename))

# =========================================================================

@requires_trt
class TestAlbertBuildEngine:
    """Test Albert build_engine produces valid TRT engine plan."""

    VOCAB, HIDDEN, LAYERS, HEADS, INTERMEDIATE, MAX_POS = 32, 16, 2, 4, 32, 64
    EMBEDDING_SIZE = 8

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, intermediate, max_pos, embed_size):
        t = {}
        t["albert.embeddings.word_embeddings.weight"] = _rand(vocab, embed_size)
        t["albert.embeddings.position_embeddings.weight"] = _rand(max_pos, embed_size)
        t["albert.embeddings.token_type_embeddings.weight"] = _rand(2, embed_size)
        t["albert.embeddings.LayerNorm.weight"] = _rand(embed_size)
        t["albert.embeddings.LayerNorm.bias"] = _rand(embed_size)
        t["albert.encoder.embedding_hidden_mapping_in.weight"] = _rand(hidden, embed_size)
        t["albert.encoder.embedding_hidden_mapping_in.bias"] = _rand(hidden)

        group_prefix = "albert.encoder.albert_layer_groups.0.albert_layers.0"
        t[f"{group_prefix}.attention.query.weight"] = _rand(hidden, hidden)
        t[f"{group_prefix}.attention.query.bias"] = _rand(hidden)
        t[f"{group_prefix}.attention.key.weight"] = _rand(hidden, hidden)
        t[f"{group_prefix}.attention.key.bias"] = _rand(hidden)
        t[f"{group_prefix}.attention.value.weight"] = _rand(hidden, hidden)
        t[f"{group_prefix}.attention.value.bias"] = _rand(hidden)
        t[f"{group_prefix}.attention.dense.weight"] = _rand(hidden, hidden)
        t[f"{group_prefix}.attention.dense.bias"] = _rand(hidden)
        t[f"{group_prefix}.attention.LayerNorm.weight"] = _rand(hidden)
        t[f"{group_prefix}.attention.LayerNorm.bias"] = _rand(hidden)
        t[f"{group_prefix}.ffn.weight"] = _rand(intermediate, hidden)
        t[f"{group_prefix}.ffn.bias"] = _rand(intermediate)
        t[f"{group_prefix}.ffn_output.weight"] = _rand(hidden, intermediate)
        t[f"{group_prefix}.ffn_output.bias"] = _rand(hidden)
        t[f"{group_prefix}.full_layer_layer_norm.weight"] = _rand(hidden)
        t[f"{group_prefix}.full_layer_layer_norm.bias"] = _rand(hidden)
        t["albert.pooler.weight"] = _rand(hidden, hidden)
        t["albert.pooler.bias"] = _rand(hidden)

        return t

    def test_build_engine_returns_bytes(self, tmp_path):
        from tensorrt_model_connect.families.albert import plugin

        config = {
            "model_type": "albert",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "embedding_size": self.EMBEDDING_SIZE,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "intermediate_size": self.INTERMEDIATE,
            "max_position_embeddings": self.MAX_POS,
            "num_hidden_groups": 1,
            "inner_group_num": 1,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS,
            self.INTERMEDIATE, self.MAX_POS, self.EMBEDDING_SIZE)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)

        assert isinstance(engine, bytes)
        assert len(engine) > 0
