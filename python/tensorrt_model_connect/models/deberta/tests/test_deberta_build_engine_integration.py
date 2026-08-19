# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned build_engine integration tests for deberta."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

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
class TestDebertaBuildEngine:
    """Test DeBERTa build_engine produces valid TRT engine plan."""

    VOCAB, HIDDEN, LAYERS, HEADS, INTERMEDIATE, MAX_POS = 32, 16, 1, 4, 32, 64

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, intermediate, max_pos):
        t = {}
        t["deberta.embeddings.word_embeddings.weight"] = _rand(vocab, hidden)
        t["deberta.embeddings.LayerNorm.weight"] = _rand(hidden)
        t["deberta.embeddings.LayerNorm.bias"] = _rand(hidden)
        t["deberta.encoder.rel_embeddings.weight"] = _rand(2 * max_pos, hidden)

        for i in range(layers):
            p = f"deberta.encoder.layer.{i}"
            t[f"{p}.attention.self.in_proj.weight"] = _rand(3 * hidden, hidden)
            t[f"{p}.attention.self.q_bias"] = _rand(hidden)
            t[f"{p}.attention.self.v_bias"] = _rand(hidden)
            t[f"{p}.attention.self.pos_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.self.pos_q_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.self.pos_q_proj.bias"] = _rand(hidden)
            t[f"{p}.attention.output.dense.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.output.dense.bias"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.bias"] = _rand(hidden)
            t[f"{p}.intermediate.dense.weight"] = _rand(intermediate, hidden)
            t[f"{p}.intermediate.dense.bias"] = _rand(intermediate)
            t[f"{p}.output.dense.weight"] = _rand(hidden, intermediate)
            t[f"{p}.output.dense.bias"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.bias"] = _rand(hidden)

        return t

    def test_build_engine_returns_bytes(self, tmp_path):
        import tensorrt_model_connect.models.deberta.model as plugin

        config = {
            "model_type": "deberta",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "intermediate_size": self.INTERMEDIATE,
            "max_position_embeddings": self.MAX_POS,
            "pos_att_type": "c2p|p2c",
            "max_relative_positions": self.MAX_POS,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.INTERMEDIATE, self.MAX_POS)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)

        assert isinstance(engine, bytes)
        assert len(engine) > 0
